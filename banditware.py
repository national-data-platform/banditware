from math import ceil
import json
import random
import time
import pathlib
import argparse
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_squared_error  # for RMSE calculation
import plotly.io as pio
from models import Model, ModelInterface
# plotly has a bug where the first time a graph is saved as a pdf, there is a loading message
# that gets integrated into the pdf directly. setting mathjax to None bypasses that bug.
pio.kaleido.scope.mathjax = None

from hardware_manager import HardwareManager


"""
TODO:
* add a way to save the fit models in ModelInterface so they don't have to be re-fit every time before prediction.
* call preprocessor if data has not been preprocessed
* change self._hardwares to just be realistic hardwares, and add a self._all_hardwares that includes hardware configurations that have very limited data. 
    * Add an option to specify whether self._hardwares should be a subset of self._all_hardwares
* add support for model params
* Update documentation
"""

# set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class BanditWare():
    # defaults
    DEFAULT_TOLERANCE_RATIO: float = 0.0
    DEFAULT_TOLERANCE_SECONDS: int = 0
    DEFAULT_EPSILON_START: float = 1.0
    DEFAULT_EPSILON_DECAY: float = 0.99
    DEFAULT_EPSILON_MIN: float = 0.0
    # what hardware to suggest when there is no historical data
    NO_DATA_HARDWARE_SUGGESTION: Tuple[int,int] = (2,16)  # (CPU count, RAM in GB)
    DATA_FILE_NAME: str = "bw_data.csv"

    def __init__(self,
                 data:Union[pd.DataFrame,None] = None,
                 feature_cols:Union[List[str],None] = None,
                 save_dir:Union[pathlib.Path,str,None] = None,
                 plot_motivation:bool = False,
                 epsilon_start:float = DEFAULT_EPSILON_START,
                 epsilon_decay:float = DEFAULT_EPSILON_DECAY,
                 epsilon_min:float = DEFAULT_EPSILON_MIN,
                 model_choice:Union[Model,str] = Model.LINEAR_REGRESSION,
                 **model_params):

        # ---- private member variables ----

        self._save_dir: pathlib.Path = self._init_save_dir(save_dir)
        self._historical_data = self._init_data(data)
        self._hardwares = list(range(HardwareManager.num_hardwares))
        self._model_instances: Dict[int, ModelInterface] = {}
        self.reset_models(model_choice, **model_params)
        # whether BanditWare has fully trained on the historical data
        self._fully_trained: bool = False
        # Exploration/exploitation for `suggest_hardware()`:
        assert 0.0 <= epsilon_start <= 1.0, "`epsilon_start` must be a float between 0 and 1"
        self._epsilon = epsilon_start
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

        # ---- public member variables ----

        self.historical_data_csv: str = str(save_dir) + "/bw_data.csv"
        self.plot_motivation: bool = plot_motivation
        self.feature_cols: List[str] = feature_cols or []
        self.reset_feature_cols(feature_cols)

    # ==========================
    #       PUBLIC METHODS
    # ==========================

    # TODO: make sure this works with new_data as a single run pd.Series
    def add_historical_data(self, new_data:pd.DataFrame,
                            update_saved_data:bool=True, retrain:bool=True) -> None:
        """
        Add application run data to BanditWare historical data and optionally retrain models
        Parameters:
            new_data: a dataframe containing all feature columns and runtimes
            update_saved_data: whether to save the new historical data
            retrain: whether to train the models on the newly added data
        """
        # TODO: handle if new_data has hardware index instead of "cpu_ram" str in hardware col
        needed_cols = ["runtime"] + self.feature_cols
        bad_cols_msg = f"new_data must contain all necessary columns: {needed_cols}"
        assert set(needed_cols).issubset(set(new_data.columns)), bad_cols_msg
        old_data = self._historical_data
        old_data["hardware"] = old_data["hardware"].apply(HardwareManager.get_hardware)
        full_data = pd.concat([old_data, new_data], ignore_index=True)
        self._historical_data = self._init_data(full_data)
        if update_saved_data:
            self.save_data()
        self._fully_trained = False
        if retrain:
            self.train()

    def predict_best_hardware(
            self, features:List, tolerance_ratio: Union[float,None] = DEFAULT_TOLERANCE_RATIO,
            tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS):
        """Given a set of feature values and tolerance, predict the best hardware"""
        # Make sure all models have fit before running predictions
        if not self._fully_trained:
            self.train()

        runtimes_by_hardware = {
            h:self._predict_runtime(model, features) for h, model in self._model_instances.items()}
        best_hardware = self._get_best_hardware(
            runtimes_by_hardware, tolerance_ratio=tolerance_ratio,
            tolerance_seconds=tolerance_seconds)
        return best_hardware
    
    def suggest_hardware(self, features:Union[np.ndarray,None] = None,
                         tolerance_ratio: Union[float, None] = DEFAULT_TOLERANCE_RATIO, tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
                         prohibit_exploration:bool = False,
                         update_epsilon:bool = True,
                         prefer_recent_data:bool = False) -> Tuple[int, int]:
        # TODO: implement optional parameter to more heavily weight recent data
        # TODO: add performance data functionality - suggest new hardware based on cpu_usage_% and mem_usage_% (use PREMOA)
        """
        Give a suggestion for the hardware settings to run on next.
        This is the function to use for NDP default hardware suggestions.
        Parameters:
            features: the input features (optional) that the application will be running on
            tolerance_ratio: Fraction of acceptable runtime increase for choosing hardware with fewer resources during exploitation. 
                * Ex: `tolerance_ratio=0.1` allows up to a 10% slower runtime if the hardware uses fewer resources
                * If tolerance_ratio is 0, only updates best_hardware in case of a tie for runtimes.
                * If tolerace_ratio is None, returns the fastest hardware without updating
            tolerance_seconds: acceptable runtime increase in seconds for choosing hardware with fewer resources during exploitation.
                * Ex: `tolerance_seconds=60` allows up to a 60s slowdown if the hardware uses fewer resources.
                * It will always choose the max tolerance between tolerance_ratio and tolerance_seconds.
            prohibit_exploration: whether to only exploit (choose the best predicted hardware) or possibly continue exploring as well
            update_epsilon: whether to slowly decrease exploration chances
            prefer_recent_data: whether to give higher weight to more recent historical data
        Returns:
            hardware_suggestion: tuple of (CPU count, RAM amount in GB)
        """
        if len(self._historical_data) == 0:
            return self.NO_DATA_HARDWARE_SUGGESTION
        
        cpu_suggestion = 0
        mem_suggestion = 0

        # Exploration
        explore = False if prohibit_exploration else np.random.rand() < self._epsilon
        print(f"\t{explore=}") # TODO: remove this
        if update_epsilon and not prohibit_exploration:
            self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)
        if explore:
            hardware_idx = random.choice(self._hardwares)
            cpu_suggestion, mem_suggestion = HardwareManager.spec_from_hardware_idx(hardware_idx)
            hardware_suggestion = cpu_suggestion, mem_suggestion
            err_msg = f"Something went wrong: sugested cpu or memory < 1: ({hardware_suggestion})"
            assert cpu_suggestion >= 1 and mem_suggestion >= 1, err_msg
            return hardware_suggestion

        # Exploitation
        avg_runtimes_by_hardware = {}
        if features is None:
            # find the hardware with the best average runtime across features
            runtimes_by_hardware = {h:[] for h in self._hardwares}
            unique_feature_groups = self._historical_data.groupby(self.feature_cols)
            for feature_vals, features_df in unique_feature_groups:
                for hardware, hardware_df in features_df.groupby("hardware"):
                    runtimes_by_hardware[hardware].append(hardware_df['runtime'].mean())
            avg_runtimes_by_hardware = {h:sum(runtimes)/len(runtimes) for h, runtimes in runtimes_by_hardware.items()}
        else:
            avg_runtimes_by_hardware = {}
            # find all rows where features match the given features
            df_features = self._historical_data[self.feature_cols]
            mask = df_features.eq(features).all(axis=1)
            df = self._historical_data[mask]
            for hardware, hardware_df in df.groupby("hardware"):
                avg_runtimes_by_hardware[hardware] = hardware_df['runtime'].mean()
        best_hardware = self._get_best_hardware(avg_runtimes_by_hardware, tolerance_ratio=tolerance_ratio, tolerance_seconds=tolerance_seconds)
        cpu_suggestion, mem_suggestion = HardwareManager.spec_from_hardware_idx(best_hardware)

        hardware_suggestion = cpu_suggestion, mem_suggestion
        err_msg = f"Something went wrong: sugested cpu or memory < 1: ({hardware_suggestion})"
        assert cpu_suggestion >= 1 and mem_suggestion >= 1, err_msg
        return hardware_suggestion

    def train(self):
        """Train BanditWare on the historical data"""
        if self._fully_trained:
            return
        for h, model in self._model_instances.items():
            df = self._historical_data[self._historical_data['hardware'] == h]
            X = df[self.feature_cols].to_numpy()
            runtimes = df['runtime'].to_numpy()
            if len(runtimes) > 0:
                model.fit(X, runtimes)
        self._fully_trained = True

    def save_data(self):
        """
        Save the historical data to a file
        """
        save_data = self._historical_data.copy()
        save_data["hardware"] = save_data["hardware"].apply(HardwareManager.get_hardware)
        if not self._save_dir.exists():
            self._save_dir.mkdir()
        save_data.to_csv(self._save_dir.joinpath(self.DATA_FILE_NAME), index=False)

    def test_accuracy(
            self,
            data: Union[pd.DataFrame,None] = None,
            tolerance_ratio: Union[float, None] = None,
            tolerance_seconds: int = 0,
            print_results: bool = True,
            ignore_incomplete_feature_rows: bool = False,
            plot_runtime_predictions: bool = False,
            model_choice: Union[str, Model, None] = None,
            **model_params) -> Dict[str,float]:
        """
        Splits the data into train and test data, then trains and tests Banditware on that split.
        Assumes that feature_cols and hardware in `data` are the same as BanditWare's current state.
        Does not modify internal state (self._model_instances, self._historical_data, etc.).
        Statistics:
            RMSE is calculated by how off the runtime predictions are per row in the test data.
            Accuracy for each unique feature set in the training data is calculated by if the predicted best hardware matches the actual best hardware from the entire dataset.

        Parameters:
            data: what data to run BanditWare on. If data is None, uses copy of historical_data
            tolerance_ratio: Fraction of acceptable runtime increase for choosing hardware with fewer resources. 
                * Ex: `tolerance_ratio=0.1` allows up to a 10% slower runtime if the hardware uses fewer resources
                * If tolerance_ratio is 0, only updates best_hardware in case of a tie for runtimes.
                * If tolerace_ratio is None, returns the fastest hardware without updating
            tolerance_seconds: acceptable runtime increase in seconds for choosing hardware with fewer resources.
                * Ex: `tolerance_seconds=60` allows up to a 60s slowdown if the hardware uses fewer resources.
                * It will always choose the max tolerance between tolerance_ratio and tolerance_seconds.
            print_results: whether to print accuracy, rmse, training time, and testing time
            ignore_incomplete_feature_rows: whether to ignore or include feature rows that haven't been run on all hardware options in the accuracy calculation.
                * This impacts accuracy. If the features have not been run on a hardware, that hardware will not be listed as the truth "correct answer".
            model_choice (str | Models | None): the choice of model to use, by name or enum
                * defualt is linear regression
            **model_params: any hyperparameters the underlying models should take in
        Returns:
            stats_dict: a dictionary with "accuracy", "rmse" from testing, as well as "train_time" and "test_time" for how long it took to train and test on the given dataset.
        Raises:
            TypeError: if model_choice is of the wrong type
            ValueError: if model_choice is a string that is not one of the model names
        """
        full_data = self._init_data(data) if data is not None else self._historical_data
        train_data, test_data = self._split_data(full_data)
        if len(test_data) == 0:
            print("Warning: Not enough data to test accuracy on.")
            return {"accuracy":np.nan, "rmse":np.nan, "train_time":np.nan, "test_time":np.nan}
        rmse = 0.0
        accuracy = 0.0
        start_time = time.time()
        
        # train BanditWare on the train data using new models (not updating self._model_instances)
        model = self._resolve_model_choice(model_choice)
        models_by_hardware:Dict[int, ModelInterface] = {h: model.create(**model_params) for h in self._hardwares}
        train_data_by_hardware = train_data.groupby("hardware")
        for h, df in train_data_by_hardware:
            features = df[self.feature_cols].to_numpy()
            runtimes = df["runtime"].to_numpy()
            models_by_hardware[h].fit(features, runtimes)
        train_stop_time = time.time()

        # test BanditWare on the test data, and update accuracy and rmse.
        X_test = test_data[self.feature_cols].to_numpy()
        y_test_true = test_data["runtime"].to_numpy()
        y_test_pred = np.array([self._predict_runtime(models_by_hardware[h], x)
                                for h, x in zip(test_data["hardware"], X_test) ])
        rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

        unique_feature_sets = test_data[self.feature_cols].drop_duplicates().to_numpy()
        accuracy = self._get_models_accuracy(
            models=models_by_hardware,
            unique_feature_sets=unique_feature_sets,
            full_data=full_data,
            tolerance_ratio=tolerance_ratio,
            tolerance_seconds=tolerance_seconds,
            ignore_incomplete_feature_rows=ignore_incomplete_feature_rows)

       
        testing_time = time.time() - train_stop_time
        training_time = train_stop_time - start_time
        
        if print_results:
            acc_msg = f"Accuracy: {(100*accuracy):.2f}%"
            rmse_msg = f"RMSE: {rmse:.2f}"
            train_msg = f"Time to train: {training_time:.2f} sec"
            test_msg = f"Time to test: {testing_time:.2f} sec"
            print("Test Statistics:", acc_msg, rmse_msg, train_msg, test_msg, sep="\n\t")

        # For debugging: see if the runtime predictions are reasonable on each hardware
        if plot_runtime_predictions:
            # TODO: make it so it can work with multiple feature cols
            if len(self.feature_cols) == 1:
                self.plot_runtime_predictions(test_data, models_by_hardware)
            else:
                print("Cannot print runtime predictions if there is more than one feature column.")

        stats_dict = {"accuracy":accuracy, "rmse":rmse, "train_time":training_time, "test_time":testing_time}
        return stats_dict

    def reset_feature_cols(self, new_feature_cols: Union[List[str],None] = None) -> None:
        """
        Resets the feature columns that BanditWare trains on to be the given feature columns
        If no columns are passed in, BanditWare uses all feature columns from the train/test data.
            All feature columns means everything besides "runtime" or "hardware"
        """
        if new_feature_cols is not None:
            cleaned_feature_cols = [col for col in new_feature_cols if col not in ['runtime', 'hardware']]
            self.feature_cols = cleaned_feature_cols
            return

        if self._historical_data is None:
            raise ValueError(
                "new_feature_cols cannot be None if no data has not been given.")
        
        self.feature_cols = [
            col for col in self._historical_data.columns if col not in ["runtime", "hardware"]]
   
    def reset_models(self, model_choice: Union[str, Model, None] = None, **model_params) -> None:
        """
        Set the models to be used for BanditWare's runtime prediction of each hardware.
        Sets self._model_instances
        Parameters:
            model_choice (str | Model | None): the choice of model to use, by name or enum
                * defualt is linear regression
            **model_params: any hyperparameters the underlying models should take in
        Raises:
            TypeError: if model_choice is of the wrong type
            ValueError: if model_choice is a string that is not one of the model names
        """
        model = self._resolve_model_choice(model_choice)
        self._model_instances = {h: model.create(**model_params) for h in self._hardwares}

    def change_application(
            self, new_data:pd.DataFrame, new_save_dir:Union[str,pathlib.Path],
            feature_cols:Union[List[str],None]=None, model_choice:Union[Model,str,None] = None,
            **model_params):
        """
        Changes the application that BanditWare is predicting and saves the old application's data.
        Parameters:
            new_data: application's historical data
            feature_cols: feature columns of the application used to predict runtime
                * default is every column except 'runtime' and 'hardware'
            model_choice: which type of ML model to use (from the `Model` enum)
                * default is Model.LINEAR_REGRESSION
            **model_params: any hyperparameters the underlying models should take in
        """
        self.save_data()
        self._init_save_dir(new_save_dir)
        self._historical_data = self._init_data(new_data)
        self.reset_feature_cols(feature_cols)
        self.reset_models(model_choice, **model_params)

    def plot_runtime_predictions(self, data:Union[pd.DataFrame, None] = None,
                                 model_instances:Union[None, Dict[int,ModelInterface]] = None):
        """
        Plot the predicted runtime over the feature column for each hardware
        Parameters:
            data: the entire dataset, containing the feature column, hardware, and runtime
            model_instances: the trained models by hardware index. 
                * If None, uses self._model_instances
        """
        if data is None:
            data = self._historical_data
        if model_instances is None:
            model_instances = self._model_instances
        # Assert proper conditions have been met
        assert len(self.feature_cols) == 1, "`_plot_predictions` can only be called when there is only one feature column."
        data_hardwares = set(data['hardware'].unique())
        model_hardwares = set(model_instances.keys())
        hardware_misfit_msg = f"`model_instances` have not been trained on this `data`. The unique hardwares differ: {data_hardwares} vs {model_hardwares}"
        assert data_hardwares == model_hardwares, hardware_misfit_msg
        
        feature_col = self.feature_cols[0]
        rows = []

        # find the uncertainty of runtime prediction for each hardware
        runtime_pred_uncertainty = {h: 0 for h in self._hardwares}
        for hardware_idx in self._hardwares:
            data_portion = data[data["hardware"]==hardware_idx]
            X = pd.DataFrame(data=data_portion[feature_col], columns=["area"]).to_numpy()
            pred_runtimes = model_instances[hardware_idx].predict(X)
            actual_runtimes = data_portion["runtime"].to_numpy()
            pred_std_error = np.std(pred_runtimes - actual_runtimes)
            runtime_pred_uncertainty[hardware_idx] = pred_std_error
        
        # format data to be plotted for each hardware
        for hardware_idx in self._hardwares:
            for x in data[feature_col].unique():
                y_pred = self._predict_runtime(model_instances[hardware_idx], x)
                rows.append({
                    "x": x,
                    "y": y_pred,
                    "mode": "Predicted",
                    "Hardware": hardware_idx,
                    "error": runtime_pred_uncertainty[hardware_idx]
                })
                y_spread = data[data["hardware"]==hardware_idx]["runtime"]
                for y in y_spread:
                    rows.append({
                        "x": x + 30000,
                        "y": y,
                        "mode": "Actual",
                        "Hardware": hardware_idx,
                        "error": 0
                    })

        df = pd.DataFrame(rows)
        fig = px.scatter(
            df, x="x", y="y", color="mode",
            facet_col="Hardware",
            template="simple_white",
            error_y="error",
            opacity=0.5,
            symbol="mode"
        )

        fig.update_xaxes(title_text=feature_col.capitalize())
        fig.update_yaxes(title_text="Runtime", matches="y")
        fig.show()
        fig.write_image(f"{self._save_dir}/runtime_predictions_{feature_col}.pdf")

    # ==========================
    #       PRIVATE METHODS
    # ==========================

    # Initialization helpers

    def _init_save_dir(self, save_dir:Union[pathlib.Path,str,None]) -> pathlib.Path:
        """Returns the directory to save results into"""
        this_dir = pathlib.Path(__file__).resolve().parent
        final_save_dir = this_dir if save_dir is None else pathlib.Path(save_dir)
        return final_save_dir

    def _init_data(self, data=None) -> pd.DataFrame:
        """Loads in the preprocessed data and updates the hardware settings"""
        save_file = self._save_dir.joinpath(self.DATA_FILE_NAME)
        # handle case of no data
        if not save_file.exists() and data is None:
            necessary_cols = self.feature_cols + ['hardware', 'runtime']
            df = pd.DataFrame(columns=necessary_cols)
            return df
        # get data
        full_data = data if data is not None else pd.read_csv(save_file)
        HardwareManager.init_manager(full_data)
        # Replace the hardware name with an integer identifier in hardware manager
        full_data["hardware"] = full_data["hardware"].apply(lambda x: int(HardwareManager.get_hardware_idx(x)))
        return full_data
    
    def _resolve_model_choice(self, model_choice: Union[str, Model, None] = None) -> Model:
        """
        Given the choice of model in some format, return it in the correct Model enum format
        Parameters:
            model_choice (str | Models | None): the choice of model to use, by name or enum
                * defualt is linear regression
        Returns:
            the correct model Enum
        """
        if model_choice is None:
            model_choice = Model.LINEAR_REGRESSION
        # If model_choice is a normal enum model, set model and pass in the hyperparameters
        if isinstance(model_choice, Model):
            return model_choice
        # If model_choice is a string, get the corresponding model and pass in hyperparameters
        elif isinstance(model_choice, str):
            try:
                model = Model[model_choice]
                return model
            except KeyError as e:
                all_models = [m.name for m in Model]
                raise ValueError(f"model_choice must be one of the following:\n{all_models}") from e
        # Make sure model_choice is the right type
        else:
            raise TypeError("model_choice must be of type `str` or `Models` enum")
            
    # Helpers for running a simulation

    def _predict_runtime(self, model:ModelInterface, feature_row: np.ndarray) -> float:
        """Given a single model (for one hardware) and feature row, predict the runtime"""
        # model has no data from this hardware to fit -> predict 0 runtime so it gets explored soon
        if not model.has_fit():
            return 0.0
        
        # predict the runtime given the features
        batch = np.atleast_2d(feature_row)
        predicted_runtime = model.predict(batch)[0]
        return predicted_runtime

    def _get_runtimes(self, features: np.ndarray, hardware: int, df: Union[pd.DataFrame,None] = None) -> np.ndarray:
        """Get the runtime of a workflow on a specific hardware"""
        if df is None:
            df = self._historical_data
        # get the runtimes from either the test or the train data
        filtered_data = df[
            (df["hardware"] == hardware) &
            np.logical_and.reduce(
                [df[col] == feature for col, feature in zip(self.feature_cols, features)])
        ]

        return filtered_data["runtime"].to_numpy()

    def _sample_runtime(self, features: np.ndarray, hardware: int, df: pd.DataFrame) -> float:
        """Sample the runtime of a workflow on a specific hardware."""
        runtimes = self._get_runtimes(features, hardware, df)
        return random.choice(runtimes)

    def _get_hardware_avg_runtimes(self, features: np.ndarray, hardware: List[int],
                                  df: pd.DataFrame) -> Dict[int, float]:
        avg_hardware_runtimes = {}
        for h in hardware:
            runtimes = self._get_runtimes(features, h, df)
            # If a feature set is not represented, it returns NaN for that hardware
            if len(runtimes) > 0:
                avg_hardware_runtimes[h] = np.mean(runtimes)
            else:
                avg_hardware_runtimes[h] = np.nan
        return avg_hardware_runtimes

    def _get_best_hardware(self, runtimes_by_hardware: Dict[int,float], tolerance_ratio: Union[float, None] = DEFAULT_TOLERANCE_RATIO, tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS) -> int:
        """
        Determines the best hardware within an acceptable runtime tolerance, preferring options with fewer resources. 

        Parameters:
            runtimes_by_hardware: Dictionary of hardware configurations mapped to their average runtimes for a given set of feature values.
            tolerance_ratio: Fraction of acceptable runtime increase for choosing hardware with fewer resources. 
                * Ex: `tolerance_ratio=0.1` allows up to a 10% slower runtime if the hardware uses fewer resources
                * If tolerance_ratio is 0, only updates best_hardware in case of a tie for runtimes.
                * If tolerace_ratio is None, returns the fastest hardware without updating
            tolerance_seconds: acceptable runtime increase in seconds for choosing hardware with fewer resources.
                * Ex: `tolerance_seconds=60` allows up to a 60s slowdown if the hardware uses fewer resources.
                * It will always choose the max tolerance between tolerance_ratio and tolerance_seconds.
        Returns:
            The optimal hardware configuration, accounting for resource efficiency within the tolerance.
        """
        assert (tolerance_ratio is None) or (tolerance_ratio >= 0), "tolerance_ratio must be a float greater than 0 or None."
        assert tolerance_seconds >= 0, "tolerance seconds must be non-negative"

        # get information on the hardware that has the fastest runtime
        fastest_hardware = min(runtimes_by_hardware, key=runtimes_by_hardware.get)
        # if requesting no tolerance effects, just use the fastest hardware
        if tolerance_ratio is None:
            if tolerance_seconds == 0:
                return fastest_hardware
            # if tolerance seconds is specified without ratio, set ratio to 0.
            tolerance_ratio = 0
        
        fastest_runtime = runtimes_by_hardware[fastest_hardware]
        base_cpu_count, base_memory_gb = HardwareManager.spec_from_hardware_idx(fastest_hardware)

        best_hardware = fastest_hardware
        tolerance_limit = (1 + tolerance_ratio) * fastest_runtime
        tolerance_limit = max(tolerance_limit, fastest_runtime + tolerance_seconds)
        max_resource_decrease = 0
        
        # potentially update the best hardware if a new one is fast enough with fewer resources
        for hardware_idx, runtime in runtimes_by_hardware.items():
            if runtime > tolerance_limit or np.isnan(runtime):
                continue
            
            cpu_count, memory_gb = HardwareManager.spec_from_hardware(HardwareManager.get_hardware(hardware_idx))
            
            # calculate proportional decreases in CPU and memory
            cpu_decrease_ratio = (base_cpu_count - cpu_count) / base_cpu_count
            memory_decrease_ratio = (base_memory_gb - memory_gb) / base_memory_gb
            overall_resource_decrease = (cpu_decrease_ratio + memory_decrease_ratio) / 2
            
            # update best hardware if a larger average proportional decrease is found
            if overall_resource_decrease > max_resource_decrease:
                best_hardware = hardware_idx
                max_resource_decrease = overall_resource_decrease
        
        return best_hardware
    
    # helpers for testing accuracy

    def _split_data(self, data:pd.DataFrame, training_frac:float=0.8):
        assert 0 < training_frac < 1, "training_frac must be between 0 and 1"
        # don't modify actual data while creating train and test splits
        data = data.copy()
        # reserve 20% of the data for testing
        total_rows = len(data)
        train_data = None
        test_data = None
        train_data_indices = []
        unique_hardware = data["hardware"].unique()
        # if possible, ensure that training data contains at least one run per hardware
        if len(unique_hardware) <= training_frac * total_rows:
            for h in unique_hardware:
                subset:pd.DataFrame = data[data["hardware"] == h]
                indices = subset.index
                row_idx = random.choice(indices)
                train_data_indices.append(row_idx)
        train_data = data.iloc[train_data_indices]
        data = data.drop(index=train_data_indices)
        n_training_rows_left = ceil(training_frac * total_rows) - len(train_data_indices)
        additional_training_data = data.sample(n=n_training_rows_left, replace=False)
        # finalize training data
        train_data = pd.concat([train_data, additional_training_data])
        # finalize testing data
        data = data.drop(index=additional_training_data.index)
        test_data = data
        # shuffle the dataframes to make sure all future accesses are random
        train_data = train_data.sample(frac=1, replace=False)
        test_data = test_data.sample(frac=1, replace=False)
        return train_data, test_data

    def _best_hardwares_truth(
            self, unique_feature_sets:np.ndarray, full_data:pd.DataFrame, tolerance_ratio: Union[float, None] = None, tolerance_seconds: int = 0, void_missing_hardwares:bool = False) -> List[Union[int, None]]:
        """
        Finds the best hardware for each row in the test data.
        Parameters:
            ...
            void_missing_hardwares: if True, ignores all rows where data for a hardware is missing
        """
        warn_missing_data = False
        best_hardwares = []
        # get the best hardwrae for the 
        for features in unique_feature_sets:
            hardware_avg_runtimes = self._get_hardware_avg_runtimes(features, self._hardwares, df=full_data)
            missing_hardware_data = any(np.isnan(val) for val in hardware_avg_runtimes.values())
            if missing_hardware_data:
                if void_missing_hardwares:
                    best_hardwares.append(np.nan)
                    continue
                warn_missing_data = True
            best_hardware = self._get_best_hardware(hardware_avg_runtimes, tolerance_ratio, tolerance_seconds)
            best_hardwares.append(best_hardware)
        
        if warn_missing_data:
            print("Warning: Some features are not represented across all hardwares; Best hardware prediction may be inaccurate.")
        return best_hardwares

    def _get_models_accuracy(
            self, models: Dict[int, ModelInterface], unique_feature_sets:np.ndarray,
            full_data:pd.DataFrame, tolerance_ratio:Union[float, None], tolerance_seconds:int, ignore_incomplete_feature_rows:bool=False) -> float:
        """
        Get the accuracy of the model on the test data - how often does it predict the best hardware.
        """
        # use get_best_hardware to get the ground truth for the test values
        truth_hardwares = self._best_hardwares_truth(unique_feature_sets, full_data, tolerance_ratio=tolerance_ratio, tolerance_seconds=tolerance_seconds, void_missing_hardwares=ignore_incomplete_feature_rows)
        
        # use the model to predict the best hardware
        predictions = []
        for features in unique_feature_sets:
            predicted_runtimes = {
                h: float(self._predict_runtime(models[h], features))
                for h in self._hardwares
            }
            predicted_hardware = self._get_best_hardware(predicted_runtimes, tolerance_ratio, tolerance_seconds)
            predictions.append(predicted_hardware)

        # calculate accuracy
        prediction_results = []
        for truth_h, pred_h in zip(truth_hardwares, predictions):
            if np.isnan(truth_h) and ignore_incomplete_feature_rows:
                continue
            prediction_results.append(truth_h == pred_h)

        hardware_prediction_accuracy = np.mean(prediction_results)
        return hardware_prediction_accuracy

    def _plot_motivation(self, df: pd.DataFrame):
        assert len(self.feature_cols) == 1, "`_plot_predictions` can only be called when there is only one feature column."
        feature_col = self.feature_cols[0]
        # Convert the Names of the hardwares from ints to H0, H1, etc.
        df["hardware"] = df["hardware"].astype("str")
        df["hardware"] = df["hardware"].apply(lambda h_idx: f'H{h_idx}')
        # Convert Hardware to a categorical type
        df["hardware"] = df["hardware"].astype("category")
        # Define custom color mapping for Hardware
        # color_map = {
        #     "H0": "#FF6347",  # Tomato
        #     "H1": "#4682B4",  # SteelBlue
        #     "H2": "#32CD32",  # LimeGreen
        #     "H3": "#FFD700",  # Gold
        # }
        fig = px.scatter(
            df,
            x=feature_col,
            y="runtime",
            color="hardware",  # Color by Hardware
            template="simple_white",
            opacity=0.8,
            category_orders={"hardware": df["hardware"].unique().tolist()},
            # color_discrete_map=color_map
        )
        # Update marker size and add black borders
        fig.update_traces(marker=dict(
            size=8,                     # Increase marker size (adjust as needed)
            line=dict(
                width=2,                 # Width of the border
                color="black"            # Border color (black)
            )
        ))
        # Rename the axes
        fig.update_xaxes(title_text=feature_col.capitalize())
        fig.update_yaxes(title_text="Runtime (s)")
        fig.show()


def get_parser_args():
    """
    Get the parsed command line arguments.
    All arguments are: n_sims, savedir, motivation, model
    """
    # TODO: add a model_params argument (parsed using JSON to create a dict)
    # Parse arguments from the commandline
    parser = argparse.ArgumentParser(
        description="Run the contextual bandit algorithm")
    parser.add_argument("--n_sims", type=int,
                        help="Number of simulations to run")
    parser.add_argument("--savedir", type=str, help="Name of the savedir.")
    parser.add_argument("--motivation", type=bool,
                        help="Flag to set the plot for the motivation use-case")
    parser.add_argument("--model", type=str,
                        help="Type of model to use for runtime prediction")
    return parser.parse_args()


def main():
    args = get_parser_args()
    
    preprocessed_data = pd.read_csv("results/data/data.csv")
    final_rows = preprocessed_data.iloc[-5:]
    preprocessed_data = preprocessed_data.iloc[:-5]
    # feat_cols = ['canopy_moisture', 'surface_moisture', 'threads', 'wind_direction', 'wind_speed', 'area']
    bw = BanditWare(
        data = preprocessed_data,
        feature_cols = ['area', 'canopy_moisture'],
        save_dir = args.savedir or "./bp3d",
        plot_motivation = args.motivation or False,
        # model_choice = args.model or Model.DECISION_TREE
        model_choice = Model.LINEAR_REGRESSION
    )
    bw.train()
    print("Suggesting hardware")
    for _ in range(10):
        print("\tSuggested Hardware:", bw.suggest_hardware())
    bw.add_historical_data(final_rows)
    print("Predicted Best Hardware:", bw.predict_best_hardware([50000, 0.3]))
    bw.test_accuracy(print_results=True)

if __name__ == "__main__":
    main()
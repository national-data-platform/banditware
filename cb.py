"""
A contextual Bandit algorithm to predict most efficient hardware configurations to optimize runtime and (optionally) hardware resource allocation.

General code flow:
* `run()` runs n simulations of the contextual bandit algorithm, calling `run_sim` n times.
* `run_sim` runs the contextual bandit algorithm once
    * initializes a new backend model for each hardware configuration to predict runtime from features
    * each round for n rounds:
        * selects a new run on a chosen (exploration v exploitation) hardware
        * adds that run to the training set for that hardware's backend model
        * fits the model to update predictions on the best hardwares for a given feature set.
"""

from models import Model, ModelInterface
import json
from functools import lru_cache, wraps
from itertools import product
import random
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import pathlib
import argparse
import plotly.express as px

from sklearn.metrics import mean_squared_error  # for RMSE calculation
import plotly.io as pio

# plotly has a bug where the first time a graph is saved as a pdf, there is a loading message
# that gets integrated into the pdf directly. setting mathjax to None bypasses that bug.
pio.kaleido.scope.mathjax = None

from hardware_manager import HardwareManager


this_dir = pathlib.Path(__file__).parent.absolute()

ALL_FEATURE_COLS = [
    # "num_tasks",
    "canopy_moisture",
    # "run_max_mem_rss_bytes",
    # "sim_time",
    "surface_moisture",
    # "threads",
    "wind_direction",
    "wind_speed",
    # "run_uuid",
    "area",
    # "runtime",
    # "cpu_usage_total",
    # "mem_usage_total",
    # "transmitted_packets_total",
    # "received_packets_total",
    # "transmitted_bandwidth_total",
    # "received_bandwidth_total",
    # "queue_seconds",
    # "hardware"
]


@lru_cache(maxsize=None)
def load_data(*feature_cols: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the training data and testing data.
    If the data has not yet been read from the CSV, do so and create the split.
    If the data has already been read, keep it in memory and return that immediately.

    Returns:
        The train data and test data split
    """
    feature_cols = list(feature_cols)
    _data = pd.read_csv(f"{this_dir}/results/data/data.csv")
    # Replace the hardware name with an integer identifier in hardware manager
    _data["hardware"] = _data["hardware"].apply(
        lambda x: int(HardwareManager.get_hardware_idx(x))
    )

    unique_features = _data[feature_cols].drop_duplicates().values
    unique_hardware = sorted(_data["hardware"].unique())

    # set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # reserve 20% of the data for testing
    # must make sure that training data contains all feature/hardware combinations
    train_datas = []
    test_datas = []
    for features, hardware in product(unique_features, unique_hardware):
        full_data: pd.DataFrame = _data[
            (
                np.logical_and.reduce(
                    [
                        _data[col] == feature
                        for col, feature in zip(feature_cols, features)
                    ]
                )
            )
            & (_data["hardware"] == hardware)
        ]
        # Replace the hardware name with an integer identifier in hardware manager
        test_data = full_data.sample(frac=0.2)
        train_data = full_data.drop(test_data.index)
        train_datas.append(train_data)
        test_datas.append(test_data)

    # shuffle the dataframes to make sure all future accesses are random
    train_data = pd.concat(train_datas).sample(frac=1, replace=False)
    test_data = pd.concat(test_datas).sample(frac=1, replace=False)

    return train_data, test_data


def run_sim(
    n_rounds: int = 100,
    tolerance_ratio: Union[float, None] = None,
    tolerance_seconds: int = 0,
    e_start: float = 1,
    e_decay: float = 0.99,
    e_min: float = 0.0,
    feature_cols: List[str] = ["area"],
    savedir: pathlib.Path = None,
    motivation: bool = False,
    model_enum: Model = Model.LINEAR_REGRESSION,
    model_kwargs: Dict = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run a single simulation contextual bandit algorithm on the preprocessed data (from `preprocess.py`).
    Put the resulting figures and data into the specified `savedir` directory.

    Parameters:
        n_rounds: how many rounds to run the simulation on
        tolerance_ratio: ratio of acceptable slowdown for choosing a less resource-intensive hardware.
            * Ex: `tolerance_ratio=0.1` will choose a less intensive hardware as long as it is at most 10% slower than the fastest hardware.
        tolerance_seconds: acceptable seconds of slowdown for choosing a less resource-intensive hardware.
            * Ex: `tolerance_seconds=60` will choose a less intensive hardware as long as it is at most 60s slower than the fastest hardware.
        e_start: initial epsilon value for exploration
        e_decay: value of decay for the exponential exploration function
        e_min: minimum epsilon value for exploration
        feature_cols: which feature columns to use to predict runtime from
        savedir: where to save any figures and data from the simulation
        motivation: whether to plot a graph showing the accuracy of the contextual bandit algorithm at predicting runtime per hardware.
            * if `motivation` is True, len(feature_cols) should be 1 to accurately portray this graph.
        model_enum: which backend model to use for the prediction of runtimes based on features.
            The contextual bandit algorithm chooses which runs to train on for each hardware, and the backend models predict the relationship between features and runtimes per hardware.
        model_kwargs: any hyperparameters for the model to take in upon initialization.

    Returns:
        a Dataframe of algorithm accuracy/rmse by round, a dict of baseline info of the model
    """
    train_data, test_data = load_data(*feature_cols)
    unique_features = train_data[feature_cols].drop_duplicates().values
    unique_hardware = sorted(train_data["hardware"].unique())

    # instantiate one model per hardware
    model_kwargs = model_kwargs or {}
    model_instances: Dict = {
        h: model_enum.create(**model_kwargs) for h in unique_hardware
    }

    def predict_one(model, feature_row: np.ndarray) -> float:
        """Given a single feature row, predict the runtime"""
        # model has no data to fit off of -> predict 0 runtime so it gets explored soon
        if not model.has_fit():
            return 0.0

        # predict the runtime given the features
        batch = np.atleast_2d(feature_row)
        y_hat = model.predict(batch)[0]
        return y_hat

    @lru_cache(maxsize=None)
    def get_truth() -> Tuple[Dict[int, ModelInterface], Dict[int, float]]:
        """
        Fit one “ground-truth” model per hardware using the same model_enum,
        then record its training-set residual standard deviation.
        """
        truth_models: Dict[int, ModelInterface] = {}
        truth_std: Dict[int, float] = {}
        for hardware in unique_hardware:
            _data = train_data[train_data["hardware"] == hardware]
            X = _data[feature_cols].values
            y = _data["runtime"].values
            model = model_enum.create(**model_kwargs).fit(X, y)
            truth_models[hardware] = model
            preds = model.predict(X)
            truth_std[hardware] = float(np.std(preds - y))
        return truth_models, truth_std

    def get_runtimes(
        features: np.ndarray, hardware: int, df: pd.DataFrame = train_data
    ) -> np.ndarray:
        """Get the runtime of a workflow on a specific hardware"""
        # get the runtimes from either the test or the train data
        filtered_data = df[
            (df["hardware"] == hardware)
            & np.logical_and.reduce(
                [df[col] == feature for col, feature in zip(feature_cols, features)]
            )
        ]

        return filtered_data["runtime"].values

    def sample_runtime(features: np.ndarray, hardware: int, df: pd.DataFrame) -> float:
        """Sample the runtime of a workflow on a specific hardware."""
        runtimes = get_runtimes(features, hardware, df)
        return random.choice(runtimes)

    def get_hardware_avg_runtimes(
        features: np.ndarray, hardware: List[int], df: pd.DataFrame = train_data
    ) -> Dict[int, float]:
        avg_hardware_runtimes = {
            h: np.mean(get_runtimes(features, h, df)) for h in hardware
        }
        return avg_hardware_runtimes

    def get_best_hardware(
        avg_hardware_runtimes: Dict[int, float],
        tolerance_ratio: Union[float, None],
        tolerance_seconds: int = 0,
    ) -> int:
        """
        Determines the best hardware within an acceptable runtime tolerance, preferring options with fewer resources.
        If tolerance_ratio is 0, only updates best_hardware in case of a tie for runtimes.
        If tolerace_ratio is None, returns the fastest hardware without updating

        Parameters:
            avg_hardware_runtimes: Dictionary of hardware configurations mapped to their average runtimes.
            tolerance_ratio: Fraction of acceptable runtime increase for choosing hardware with fewer resources.
                * Ex: `tolerance_ratio=0.1` allows up to a 10% slower runtime if the hardware uses fewer resources
            tolerance_seconds: acceptable runtime increase in seconds for choosing hardware with fewer resources.
                * Ex: `tolerance_seconds=60` allows up to a 60s slowdown if the hardware uses fewer resources.
        Returns:
            The optimal hardware configuration, accounting for resource efficiency within the tolerance.
        """
        assert (tolerance_ratio is None) or (
            tolerance_ratio >= 0
        ), "tolerance_ratio must be a float greater than 0 or None."
        assert tolerance_seconds >= 0, "tolerance seconds must be non-negative"

        # get information on the hardware that has the fastest runtime
        fastest_hardware = min(avg_hardware_runtimes, key=avg_hardware_runtimes.get)
        # if requesting no tolerance effects, just use the fastest hardware
        if tolerance_ratio is None:
            if tolerance_seconds == 0:
                return fastest_hardware
            # if tolerance seconds is specified without ratio, set ratio to 0.
            tolerance_ratio = 0

        fastest_runtime = avg_hardware_runtimes[fastest_hardware]
        base_cpu_count, base_memory_gb = HardwareManager.spec_from_hardware(
            HardwareManager.get_hardware(fastest_hardware)
        )

        best_hardware = fastest_hardware
        tolerance_limit = (1 + tolerance_ratio) * fastest_runtime
        tolerance_limit = max(tolerance_limit, fastest_runtime + tolerance_seconds)
        max_resource_decrease = 0

        # potentially update the best hardware if a new one is fast enough with fewer resources
        for hardware, runtime in avg_hardware_runtimes.items():
            if runtime > tolerance_limit:
                continue

            cpu_count, memory_gb = HardwareManager.spec_from_hardware(
                HardwareManager.get_hardware(hardware)
            )

            # calculate proportional decreases in CPU and memory
            cpu_decrease_ratio = (base_cpu_count - cpu_count) / base_cpu_count
            memory_decrease_ratio = (base_memory_gb - memory_gb) / base_memory_gb
            overall_resource_decrease = (cpu_decrease_ratio + memory_decrease_ratio) / 2

            # update best hardware if a larger average proportional decrease is found
            if overall_resource_decrease > max_resource_decrease:
                best_hardware = hardware
                max_resource_decrease = overall_resource_decrease

        return best_hardware

    @lru_cache(maxsize=None)
    def get_best_hardwares(
        tolerance_ratio: Union[float, None] = None, tolerance_seconds: int = 0
    ) -> List[int]:
        """
        Finds the best hardware for each row in the test data.
        """
        best_hardwares = []
        for features in test_data[feature_cols].values:
            hardware_avg_runtimes = get_hardware_avg_runtimes(
                features, unique_hardware, df=test_data
            )
            best_hardware = get_best_hardware(
                hardware_avg_runtimes, tolerance_ratio, tolerance_seconds
            )
            best_hardwares.append(best_hardware)
        return best_hardwares

    def get_model_accuracy(models: Dict[int, ModelInterface]) -> float:
        """Get the accuracy of the model on the test data - how often does it predict the best hardware"""
        # use get_best_hardware to get the ground truth for the test values
        truth = get_best_hardwares(
            tolerance_ratio=tolerance_ratio, tolerance_seconds=tolerance_seconds
        )
        # use the model to predict the best hardware
        predictions = []

        for features in test_data[feature_cols].values:
            predicted_runtimes = {
                h: float(predict_one(models[h], features)) for h in unique_hardware
            }
            predicted_hardware = get_best_hardware(
                predicted_runtimes, tolerance_ratio, tolerance_seconds
            )
            predictions.append(predicted_hardware)

        return np.mean([t == p for t, p in zip(truth, predictions)])

    def select_features_and_hardware(
        unexplored_feature_hardware_pairs: List[Tuple[np.ndarray, int]],
    ) -> Tuple[np.ndarray, int]:
        assert (
            len(unexplored_feature_hardware_pairs) > 0
        ), "Nothing to choose from - list is empty"

        rand_index = random.randint(0, len(unexplored_feature_hardware_pairs) - 1)
        # get a random choice from the unexplored (feature,hardware) pairs
        features, hardware = unexplored_feature_hardware_pairs.pop(rand_index)
        return features, hardware

    # Check that there are runtimes for all hardware/features in the training data
    for hardware, features in product(unique_hardware, unique_features):
        runtimes = get_runtimes(features, hardware, train_data)
        assert len(runtimes) > 0

    samples: Dict[int, List[Tuple[List[float], float]]] = {
        i: [] for i in unique_hardware
    }

    std = {i: 0 for i in unique_hardware}
    rows_runtime = []
    e = e_start

    # random_feature_choices = rand_feature_choices()
    for i in range(n_rounds):
        # Exploration vs Exploitation
        features = random.choice(unique_features)
        if np.random.rand() >= e:
            hardware = min(
                unique_hardware,
                key=lambda h: float(predict_one(model_instances[h], features)),
            )
        else:
            hardware = random.choice(unique_hardware)

        # Sample the runtime of the workflow on the selected features and hardware
        runtime = sample_runtime(features, hardware, df=train_data)
        # print(f"features:\n{features}\nhardware: {hardware}\nruntime: {runtime}")
        samples[hardware].append((features, runtime))

        # fit the chosen model on this hardware’s samples
        X, y = zip(*samples[hardware])
        X = np.array(X)
        y = np.array(y)
        curr_model = model_instances[hardware].fit(X, y)
        # if you still need an “error” estimate, compute residuals:
        preds = curr_model.predict(X)
        std[hardware] = np.std(preds - y)

        # Calculate quality of the model on test data
        X_test = test_data[feature_cols].values
        y_test = test_data["runtime"].values
        # predicted runtimes
        y_pred = np.array(
            [
                predict_one(model_instances[h], x)
                for h, x in zip(test_data["hardware"], X_test)
            ]
        )
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        acc = get_model_accuracy(model_instances)

        rows_runtime.append(
            {"round": i, "runtime": runtime, "rmse": rmse, "accuracy": acc}
        )

        # Decay epsilon
        e = max(e * e_decay, e_min)

    truth_models, std_truth = get_truth()

    if savedir is not None and len(feature_cols) == 1:
        savedir.mkdir(parents=True, exist_ok=True)

        feature_col = feature_cols[0]
        rows = []
        for hardware_idx in unique_hardware:
            # for x in np.linspace(train_data[feature_col].min(), train_data[feature_col].max(), 10):
            for x in train_data[feature_col].unique():
                y_pred = predict_one(model_instances[hardware_idx], x)
                rows.append(
                    {
                        "x": x,
                        "y": y_pred,
                        "mode": "Predicted",
                        "Hardware": hardware_idx,
                        "error": std[hardware_idx],
                    }
                )
                y_actual = predict_one(truth_models[hardware_idx], x)
                # rows.append({
                #     "x": x + 0.01,
                #     "y": y_actual,
                #     "mode": "Actual",
                #     "Hardware": hardware_idx,
                #     "error": std_truth[hardware_idx]
                # })
                hardware_data = train_data[train_data["hardware"] == hardware_idx]
                y_spread = hardware_data[hardware_data[feature_col] == x]["runtime"]
                for y in y_spread:
                    rows.append(
                        {
                            "x": x + 30000,
                            "y": y,
                            "mode": "Actual",
                            "Hardware": hardware_idx,
                            "error": 0,
                        }
                    )

        df = pd.DataFrame(rows)
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="mode",
            facet_col="Hardware",
            template="simple_white",
            error_y="error",
            opacity=0.5,
            symbol="mode",
        )

        # Rename the axis
        fig.update_xaxes(title_text=feature_cols[0].capitalize())
        fig.update_yaxes(title_text="Runtime", matches="y")

        # Hide duplicate y-axis titles on other facets
        for axis in fig.layout:
            if axis.startswith("yaxis") and axis != "yaxis":
                fig.layout[axis].title.text = ""
        fig.show()
        fig.write_html(f"{savedir}/cb_{feature_cols[0]}.html")
        fig.write_image(f"{savedir}/cb_{feature_cols[0]}.png")
        fig.write_image(f"{savedir}/cb_{feature_cols[0]}.pdf")

        if motivation:
            plot_motivation(df, feature_cols[0], savedir)

    # compute rmse on testing data
    X_test = test_data[feature_cols].values
    y_test = test_data["runtime"].values
    truth_models, _ = get_truth()
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            np.array(
                [
                    predict_one(truth_models[h], x)
                    for h, x in zip(test_data["hardware"], X_test)
                ]
            ),
        )
    )
    acc = get_model_accuracy(truth_models)

    baseline_info = {"round": n_rounds, "rmse": rmse, "accuracy": acc}

    return pd.DataFrame(rows_runtime), baseline_info


def plot_motivation(df: pd.DataFrame, feature_cols: str, savedir: pathlib.Path):
    # Convert the Names of the hardwares
    # Mapping from integers to custom labels (H0, H1, H2, H3)
    hardware_map = {0: "H0", 1: "H1", 2: "H2", 3: "H3"}
    # Apply the mapping to the "Hardware" column
    df["Hardware"] = df["Hardware"].map(hardware_map)
    # Convert Hardware to a categorical type
    df["Hardware"] = df["Hardware"].astype("category")

    # Define custom color mapping for Hardware
    color_map = {
        "H0": "#FF6347",  # Tomato
        "H1": "#4682B4",  # SteelBlue
        "H2": "#32CD32",  # LimeGreen
        "H3": "#FFD700",  # Gold
    }

    fig2 = px.scatter(
        df,
        x="x",
        y="y",
        color="Hardware",  # Color by Hardware
        symbol="mode",  # Different symbols for actual and predicted
        template="simple_white",
        error_y="error",
        opacity=0.8,
        category_orders={"Hardware": df["Hardware"].unique().tolist()},
        color_discrete_map=color_map,
    )

    # Update marker size and add black borders
    fig2.update_traces(
        marker=dict(
            size=8,  # Increase marker size (adjust as needed)
            line=dict(
                width=2, color="black"  # Width of the border  # Border color (black)
            ),
        )
    )

    # Rename the axes
    fig2.update_xaxes(title_text="Number of tasks")
    fig2.update_yaxes(title_text="Makespan (s)")

    # Save the figure
    fig2.write_html(f"{savedir}/cb_motivation.html")
    fig2.write_image(f"{savedir}/cb_motivation.png")


def run(
    n_sims: int,
    n_rounds: int,
    feature_cols: List[str],
    savedir: pathlib.Path,
    tolerance_ratio: Union[float, None] = None,
    tolerance_seconds: int = 0,
    model_enum: Model = Model.LINEAR_REGRESSION,
    model_kwargs: Dict = None,
):
    """
    Run a contextual bandit algorithm on the preprocessed data (from `preprocess.py`) n_sims times.
    Put the resulting figures and data into the specified `savedir` directory.

    Parameters:
        n_sims: how many simulations to run the contextual bandit algorithm on.
        n_rounds: how many rounds to run the contextual bandit algorithm for.
        feature_cols: which columns in the dataset to use as features for runtime prediction.
        savedir: the directory to save the resulting figures and data of the algorithm.
        tolerance_ratio: ratio of acceptable slowdown from the fastest runtime for choosing a less resource intensive hardware.
            * Ex: `tolerance_ratio=0.1` would choose a less intensive hardware if it was at most 10% slower than the fastest, more intensive hardware.
            * If tolerance_ratio = 0.0, it only chooses the less intensive hardware in a tie.
            * If tolerance_ratio = None, it doesn't re-analyze the runtimes to look for better options
            * If both tolerance_ratio and tolerance_seconds are specified, it chooses the larger acceptable tolerance.
        tolerance_seconds: additional seconds of acceptable slowdown from the fastest runtime for choosing a less resource intensive hardware.
            * Ex: `tolerance_seconds=60` would choose a less intensive hardware if it was at most 60s slower than the fastest, more intensive hardware.
            * If tolerance_seconds = 0.0, it only chooses the less intensive hardware in a tie.
            * If both tolerance_ratio and tolerance_seconds are specified, it chooses the larger acceptable tolerance.
        model_enum: which backend model to use for the prediction of runtimes based on features.
            The contextual bandit algorithm chooses which runs to train on for each hardware, and the backend models predict the relationship between features and runtimes per hardware.
        model_kwargs: any hyperparameters for the model to take in upon initialization.
    """
    dfs: List[pd.DataFrame] = []
    baseline_infos: List[Dict] = []
    for i in range(n_sims):
        print(f"Running simulation {i+1}/{n_sims}", end="\r")
        save_directory = (
            savedir if i == 0 else None
        )  # only save plots of one simulation
        df, baseline_info = run_sim(
            n_rounds=n_rounds,
            tolerance_ratio=tolerance_ratio,
            feature_cols=feature_cols,
            savedir=save_directory,
            tolerance_seconds=tolerance_seconds,
            model_enum=model_enum,
            model_kwargs=model_kwargs,
        )
        baseline_info["sim"] = i
        df["sim"] = i
        dfs.append(df)
        baseline_infos.append(baseline_info)
    print()

    df_sim = pd.concat(dfs)

    # assert that rmse is the same for all simulations
    rmse_full = np.mean([info["rmse"] for info in baseline_infos])
    # assert np.allclose([info["rmse"] for info in baseline_infos], rmse_full)
    # get accuracy of full data
    acc_full = np.mean([info["accuracy"] for info in baseline_infos])
    # assert np.allclose([info["accuracy"] for info in baseline_infos], acc_full)

    info_df = df_sim.copy()
    info_df["avg_rmse_full"] = rmse_full
    info_df["avg_acc_full"] = acc_full
    # info_df.to_csv('bp3d_cb.csv')

    fig = px.box(
        df_sim,
        x="round",
        y="rmse",
        title="RMSE over time",
        template="simple_white",
        labels={"x": "Round", "y": "RMSE"},
        points=False,
    )
    fig.add_hline(
        y=rmse_full,
        line_color="red",
        line_width=2,
    )
    print(f"Full fit RMSE: {rmse_full:.2f}")

    fig.write_html(savedir.joinpath("rmse.html"))
    fig.write_image(savedir.joinpath("rmse.png"))

    fig = px.line(
        df_sim,
        x="round",
        y="rmse",
        color="sim",
        title="RMSE over time",
        template="simple_white",
        labels={"x": "Round", "y": "RMSE"},
    )
    fig.add_hline(
        y=rmse_full,
        line_color="red",
        line_width=2,
    )
    # remove legend
    fig.for_each_trace(lambda t: t.update(showlegend=False))
    fig.write_html(savedir.joinpath("rmse_line.html"))
    fig.write_image(savedir.joinpath("rmse_line.png"))

    # Plot accuracy over time
    fig = px.line(
        df_sim,
        x="round",
        y="accuracy",
        color="sim",
        title="Accuracy over time",
        template="simple_white",
        labels={"x": "Round", "y": "Accuracy"},
    )

    fig.add_hline(
        y=acc_full,
        line_color="red",
        line_width=2,
    )
    fig.write_html(savedir.joinpath("accuracy_line.html"))
    fig.write_image(savedir.joinpath("accuracy_line.png"))

    # box
    fig = px.box(
        df_sim,
        x="round",
        y="accuracy",
        title="Accuracy over time",
        template="simple_white",
        labels={"x": "Round", "y": "Accuracy"},
        points=False,
    )
    fig.add_hline(
        y=acc_full,
        line_color="red",
        line_width=2,
    )
    fig.write_html(savedir.joinpath("accuracy.html"))
    fig.write_image(savedir.joinpath("accuracy.png"))

    checkpoint_rounds = [10, 100, n_rounds] if n_rounds > 100 else [10, n_rounds]
    # get average and std rmse at last round
    print("\n\nRMSE Stats:")
    print(f"Full fit RMSE:  {rmse_full:.2f}")
    for r in checkpoint_rounds:
        last_round_rmse = df_sim[df_sim["round"] == r - 1]
        avg_rmse = last_round_rmse["rmse"].mean()
        std_rmse = last_round_rmse["rmse"].std()
        print(f"Average RMSE at round {r}: {avg_rmse:.2f} ± {std_rmse:.2f}")

        # Print how many % better the full fit is compared to the last round
        improvement = 100 * (1 - rmse_full / avg_rmse)
        print(f"Full fit is {improvement:.2f}% better than the fit in round {r}")
    print("\n\nAccuracy Stats:")
    print(f"Full fit accuracy: {acc_full:.2f}")
    # print(f"Full fit accuracy: {avg_acc_full:.2f} ± {std_acc_full}")
    for r in checkpoint_rounds:
        last_round_acc = df_sim[df_sim["round"] == r - 1]
        avg_acc = last_round_acc["accuracy"].mean()
        std_acc = last_round_acc["accuracy"].std()
        print(f"Average Accuracy at round {r}: {avg_acc:.2f} ± {std_acc:.2f}")

        # Print how many % better the full fit accuracy is compared to the last round
        improvement = 100 * ((acc_full / avg_acc) - 1)
        print(f"Full fit is {improvement:.2f}% better than the fit in round {r}")
    print("\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the contextual bandit algorithm")
    parser.add_argument(
        "--n_sims", type=int, default=10, help="Number of simulations to run"
    )
    parser.add_argument(
        "--n_rounds", type=int, default=100, help="Number of rounds per simulation"
    )
    parser.add_argument("--savedir", type=str, help="Name of the savedir.")
    parser.add_argument(
        "--motivation",
        type=bool,
        help="Flag to set the plot for the motivation use-case",
    )
    parser.add_argument(
        "--model",
        choices=[m.name for m in Model],
        default="LINEAR_REGRESSION",
        help="Which regression model to use in the bandit (from models.Model)",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help="JSON dict of hyperparameters for the chosen model",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    n_sims = args.n_sims
    n_rounds = args.n_rounds
    savedir = this_dir if args.savedir is None else pathlib.Path(args.savedir)
    savedir = savedir.joinpath("results")
    savedir.mkdir(parents=True, exist_ok=True)
    motivation = args.motivation
    # Initialize HardwareManager with the preprocessed data
    data_file = savedir.joinpath("data/data.csv")
    df = pd.read_csv(data_file)
    # extra hardcoded preprocessing for BP3D. banditware.py does not have to do this, but cb.py needs features to be enumerated across all hardwares -> areas must be shared across hardwares.
    if "area" in df.columns:
        shared_areas = [1053216.0, 1854216.0, 1369900.0, 828144.0, 2543220.0]
        df = df[df["area"].isin(shared_areas)]
    df: pd.DataFrame = df.reset_index(drop=True)
    HardwareManager.init_manager(df)

    # tolerance_ratio is a float >= 0 to represent the amount of slowdown allowed for a less resource intensive hardware
    # when set to None, it selects the fastest hardware without reassessing in the case of a tie.
    # when set to 0, it selects the fastest hardware, preferring cheaper hardwares in the case of a tie.
    tolerance_ratio: Union[float, None] = 0.0
    # tolerance seconds represents how many seconds of slowdown is acceptable for a less resource intensive hardware
    tolerance_seconds: int = 0
    # feature_cols = ALL_FEATURE_COLS
    feature_cols = ["area"]
    feats = "_".join(feature_cols)
    if set(feature_cols) == set(ALL_FEATURE_COLS):
        feats = "all"
    results_dir = savedir.joinpath(feats)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_enum = Model[args.model]
    model_kwargs = json.loads(args.model_params)

    run(
        n_sims=n_sims,
        n_rounds=n_rounds,
        feature_cols=feature_cols,
        savedir=results_dir,
        tolerance_ratio=tolerance_ratio,
        tolerance_seconds=tolerance_seconds,
        model_enum=model_enum,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

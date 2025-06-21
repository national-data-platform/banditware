from functools import lru_cache, wraps
from itertools import product
import random
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import pathlib
import argparse
import plotly.express as px

from sklearn.linear_model import LinearRegression
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
def load_data(*feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data from the csv file"""
    feature_cols = list(feature_cols)
    _data = pd.read_csv(f"{this_dir}/results/data/data.csv")

    # Replace the hardware name with an integer identifier in hardware manager
    _data["hardware"] = _data["hardware"].apply(lambda x: int(HardwareManager.get_hardware_idx(x)))

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
            (np.logical_and.reduce([_data[col] == feature for col, feature in zip(feature_cols, features)])) &
            (_data["hardware"] == hardware)
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

def run_sim(n_rounds: int = 100,
            tolerance_ratio: Union[float, None] = None,
            tolerance_seconds: int = 0,
            e_start: float = 1,
            e_decay: float = 0.99,
            e_min: float = 0.0,
            feature_cols: List[str] = ["area"],
            savedir: pathlib.Path = None,
            motivation: bool = False) -> pd.DataFrame:
    """Run a single simulation of the contextual bandit algorithm"""        
    train_data, test_data = load_data(*feature_cols)
    unique_features = train_data[feature_cols].drop_duplicates().values
    unique_hardware = sorted(train_data["hardware"].unique())

    @lru_cache(maxsize=None)
    def get_truth() -> Tuple[List[np.ndarray], List[float], List[float]]:
        """Get the true coefficients and noise for each hardware"""	
        bias_truth = []
        std_truth = []
        coef_truth = []
        for hardware in unique_hardware:
            _data = train_data[train_data["hardware"] == hardware]
            X = _data[feature_cols].values
            y = _data["runtime"].values
            reg = LinearRegression().fit(X, y)
            coef_truth.append(reg.coef_)
            bias_truth.append(reg.intercept_)
            std_truth.append(np.std(reg.predict(X) - y))    
        return coef_truth, bias_truth, std_truth

    def get_runtimes(features: np.ndarray, hardware: int, df:pd.DataFrame = train_data) -> np.ndarray:
        """Get the runtime of a workflow on a specific hardware"""
        # get the runtimes from either the test or the train data
        filtered_data = df[
            (df["hardware"] == hardware) &
            np.logical_and.reduce([df[col] == feature for col, feature in zip(feature_cols, features)])
        ]

        return filtered_data["runtime"].values

    def sample_runtime(features: np.ndarray, hardware: int, df: pd.DataFrame) -> float:
        """Sample the runtime of a workflow on a specific hardware."""
        runtimes = get_runtimes(features, hardware, df)
        return random.choice(runtimes)

    def get_hardware_avg_runtimes(features:np.ndarray, hardware:List[int], df:pd.DataFrame=train_data) -> Dict[int, float]:
        avg_hardware_runtimes = {h: np.mean(get_runtimes(features, h, df))
                                for h in hardware}
        return avg_hardware_runtimes

    def get_best_hardware(avg_hardware_runtimes: Dict[int,float], tolerance_ratio: Union[float, None], tolerance_seconds: int = 0) -> int:
        """
        Determines the best hardware within an acceptable runtime tolerance, preferring options with fewer resources. 
        If tolerance_ratio is 0, only updates best_hardware in case of a tie for runtimes.
        If tolerace_ratio is None, returns the fastest hardware without updating

        Parameters:
            avg_hardware_runtimes: Dictionary of hardware configurations mapped to their average runtimes.
            tolerance_ratio: Fraction of acceptable runtime increase for choosing hardware with fewer resources. 
                Ex: tolerance_ratio=0.1 allows a 10% slower runtime if the hardware uses fewer resources
        
        Returns:
            The optimal hardware configuration, accounting for resource efficiency within the tolerance.
        """
        assert (tolerance_ratio is None) or (tolerance_ratio >= 0), "tolerance_ratio must be a float greater than 0 or None."
        assert tolerance_seconds >= 0, "tolerance seconds must be non-negative"

        # get information on the hardware that has the fastest runtime
        fastest_hardware = min(avg_hardware_runtimes, key=avg_hardware_runtimes.get)
        # if requestion no tolerance effects, just use the fastest hardware
        if tolerance_ratio is None:
            if tolerance_seconds == 0:
                return fastest_hardware
            # if tolerance seconds is specified without ratio, set ratio to 0.
            tolerance_ratio = 0
        
        fastest_runtime = avg_hardware_runtimes[fastest_hardware]
        base_cpu_count, base_memory_gb = HardwareManager.spec_from_hardware(HardwareManager.get_hardware(fastest_hardware))

        best_hardware = fastest_hardware
        tolerance_limit = (1 + tolerance_ratio) * fastest_runtime
        tolerance_limit = max(tolerance_limit, fastest_runtime + tolerance_seconds)
        max_resource_decrease = 0
        
        # potentially update the best hardware if a new one is fast enough with fewer resources
        for hardware, runtime in avg_hardware_runtimes.items():
            if runtime > tolerance_limit:
                continue
            
            cpu_count, memory_gb = HardwareManager.spec_from_hardware(HardwareManager.get_hardware(hardware))
            
            # calculate proportional decreases in CPU and memory
            cpu_decrease_ratio = (base_cpu_count - cpu_count) / base_cpu_count
            memory_decrease_ratio = (base_memory_gb - memory_gb) / base_memory_gb
            overall_resource_decrease = (cpu_decrease_ratio + memory_decrease_ratio) / 2
            
            # update best hardware if a larger average proportional decrease is found
            if overall_resource_decrease > max_resource_decrease:
                best_hardware = hardware
        
        return best_hardware
    
    @lru_cache(maxsize=None)
    def get_best_hardwares(tolerance_ratio: Union[float, None] = None, tolerance_seconds: int = 0) -> List[int]:
        best_hardwares = []
        for features in test_data[feature_cols].values:
            hardware_avg_runtimes = get_hardware_avg_runtimes(features, unique_hardware, df=test_data)
            best_hardware = get_best_hardware(hardware_avg_runtimes, tolerance_ratio, tolerance_seconds)
            best_hardwares.append(best_hardware)
        return best_hardwares
    
    def get_model_accuracy(coefs: Dict[int, List[float]], bias: Dict[int, float], std: Dict[int, float]) -> float:
        """Get the accuracy of the model on the test data - how often does it predict the best hardware"""
        # use get_best_hardware to get the ground truth for the test values
        truth = get_best_hardwares(tolerance_ratio=tolerance_ratio, tolerance_seconds=tolerance_seconds)
        # use the model to predict the best hardware
        predictions = []

        for features in test_data[feature_cols].values:
            predicted_runtimes = {h: np.dot(coefs[h], features) + bias[h] for h in unique_hardware}
            predicted_hardware = get_best_hardware(predicted_runtimes, tolerance_ratio)
            predictions.append(predicted_hardware)

        return np.mean([t == p for t, p in zip(truth, predictions)])
    
    def select_features_and_hardware(
            unexplored_feature_hardware_pairs: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
        assert len(unexplored_feature_hardware_pairs) > 0, "Nothing to choose from - list is empty"
        
        rand_index = random.randint(0, len(unexplored_feature_hardware_pairs)-1)
        # get a random choice from the unexplored (feature,hardware) pairs
        features, hardware = unexplored_feature_hardware_pairs.pop(rand_index)
        return features, hardware
        
    # Check that there are runtimes for all hardware/features in the training data
    for hardware, features in product(unique_hardware, unique_features):
        runtimes = get_runtimes(features, hardware, train_data)
        assert(len(runtimes) > 0)

    samples: Dict[int, List[Tuple[List[float], float]]] = {i: [] for i in unique_hardware}
     
    coefs: Dict[int, List[float]] = {i: np.zeros(len(feature_cols)) for i in unique_hardware}
    bias = {i: 0 for i in unique_hardware}
    std = {i: 0 for i in unique_hardware}
    rows_runtime = []
    e = e_start
    
    unexplored_feature_hardware_pairs = list(product(unique_features, unique_hardware))

    
    # random_feature_choices = rand_feature_choices()
    for i in range(n_rounds):
        if len(unexplored_feature_hardware_pairs) > 0:
            # coverage phase: make sure each combination gets explored at least once
            features, hardware = select_features_and_hardware(unexplored_feature_hardware_pairs) 
        else:
            # bandit phase: run the bandit algorithm as normal
            features = random.choice(unique_features)
            if np.random.rand() >= e:
                hardware = min(unique_hardware, key=lambda h: np.dot(coefs[h], features) + bias[h])
            else:
                hardware = random.choice(unique_hardware)
            
        # Sample the runtime of the workflow on the selected features and hardware
        runtime = sample_runtime(features, hardware, df=train_data)
        # print(f"features:\n{features}\nhardware: {hardware}\nruntime: {runtime}")
        samples[hardware].append((features, runtime)) 

        # Update the coefficients
        X, y = zip(*samples[hardware])
        X = np.array(X)
        y = np.array(y)
        reg = LinearRegression().fit(X, y)
        coefs[hardware] = reg.coef_
        bias[hardware] = reg.intercept_
        std[hardware] = np.std(reg.predict(X) - y)
        # noise_coefs[hardware] = (reg.intercept_, np.std(reg.predict(X) - y))

        # Calculate quality of the model on test data
        X_test = test_data[feature_cols].values
        y_test = test_data["runtime"].values
        # predicted runtimes
        y_pred = np.array([np.dot(coefs[h], x) + bias[h] 
                           for h, x in zip(test_data["hardware"], X_test)])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        acc = get_model_accuracy(coefs, bias, std)

        rows_runtime.append({
            "round": i,
            "runtime": runtime,
            "rmse": rmse,
            "accuracy": acc
        })

        # Decay epsilon
        e = max(e * e_decay, e_min)

    coef_truth, bias_truth, std_truth = get_truth()
    if savedir is not None and len(feature_cols) == 1: 
        savedir.mkdir(parents=True, exist_ok=True)   

        feature_col = feature_cols[0]
        rows = []
        for hardware_idx in unique_hardware:
            for x in np.linspace(train_data[feature_col].min(), train_data[feature_col].max(), 10):
                rows.append({
                    "x": x,
                    "y": coefs[hardware_idx][0] * x + bias[hardware_idx],
                    "mode": "Predicted",
                    "Hardware": hardware_idx,
                    "error": std[hardware_idx]
                })

                rows.append({
                    "x": x + 0.01,
                    "y": coef_truth[hardware_idx][0] * x + bias_truth[hardware_idx],
                    "mode": "Actual",
                    "Hardware": hardware_idx,
                    "error": std_truth[hardware_idx]
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
        
        # Rename the axis 
        fig.update_xaxes(title_text=feature_cols[0].capitalize())
        fig.update_yaxes(title_text="Runtime", matches="y")
        
        # Hide duplicate y-axis titles on other facets
        for axis in fig.layout:
            if axis.startswith("yaxis") and axis != "yaxis":
                fig.layout[axis].title.text = ""
        
        fig.write_html(f"{savedir}/cb_{feature_cols[0]}.html")
        fig.write_image(f"{savedir}/cb_{feature_cols[0]}.png")
        fig.write_image(f"{savedir}/cb_{feature_cols[0]}.pdf")

        if motivation:
            plot_motivation(df, feature_cols[0], savedir)


    # compute rmse on testing data
    X_test = test_data[feature_cols].values
    y_test = test_data["runtime"].values
    y_pred = np.array([np.dot(coef_truth[h], x) + bias_truth[h] for h, x in zip(test_data["hardware"], X_test)])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # compute accuracy on full data (excluding test data)
    acc = get_model_accuracy(coef_truth, bias_truth, std_truth)

    baseline_info = {
        "round": n_rounds,
        "rmse": rmse,
        "accuracy": acc
    }

    return pd.DataFrame(rows_runtime), baseline_info

def plot_motivation(df: pd.DataFrame, feature_cols: str, savedir: pathlib.Path):
    # Convert the Names of the hardwares 
    # Mapping from integers to custom labels (H0, H1, H2, H3)
    hardware_map = {
        0: "H0",
        1: "H1",
        2: "H2",
        3: "H3"
    }
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
        color="Hardware",    # Color by Hardware
        symbol="mode",       # Different symbols for actual and predicted
        template="simple_white",
        error_y="error",
        opacity=0.8,
        category_orders={"Hardware": df["Hardware"].unique().tolist()},
        color_discrete_map=color_map
    )

    # Update marker size and add black borders
    fig2.update_traces(marker=dict(
        size=8,                     # Increase marker size (adjust as needed)
        line=dict(
            width=2,                 # Width of the border
            color="black"            # Border color (black)
        )
    ))
    
    # Rename the axes
    fig2.update_xaxes(title_text="Number of tasks")
    fig2.update_yaxes(title_text="Makespan (s)")

    # Save the figure
    fig2.write_html(f"{savedir}/cb_motivation.html")
    fig2.write_image(f"{savedir}/cb_motivation.png")

def run(n_sims: int, 
        n_rounds: int,
        feature_cols: List[str],
        savedir: pathlib.Path,
        tolerance_ratio: Union[float, None] = None,
        tolerance_seconds: int = 0):
    dfs = []
    baseline_infos = []
    for i in range(n_sims):
        print(f"Running simulation {i+1}/{n_sims}", end="\r")
        df, baseline_info = run_sim(n_rounds=n_rounds, tolerance_ratio=tolerance_ratio, feature_cols=feature_cols, tolerance_seconds=tolerance_seconds)
        baseline_info["sim"] = i
        df["sim"] = i
        dfs.append(df)
        baseline_infos.append(baseline_info)
    print()
    
    df_sim = pd.concat(dfs)

    # assert that rmse is the same for all simulations
    print([info["rmse"] for info in baseline_infos])
    rmse_full = np.mean([info["rmse"] for info in baseline_infos])
    assert np.allclose([info["rmse"] for info in baseline_infos], rmse_full)
    # get accuracy of full data
    acc_full = np.mean([info["accuracy"] for info in baseline_infos])
    assert np.allclose([info["accuracy"] for info in baseline_infos], acc_full)

    info_df = df_sim.copy()
    info_df['avg_rmse_full'] = rmse_full
    info_df['avg_acc_full'] = acc_full
    # info_df.to_csv('bp3d_cb.csv')

    fig = px.box(
        df_sim, x="round", y="rmse",
        title="RMSE over time",
        template="simple_white",
        labels={"x": "Round", "y": "RMSE"},
        points=False
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
        df_sim, x="round", y="rmse", color="sim",
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
        df_sim, x="round", y="accuracy", color="sim",
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
        df_sim, x="round", y="accuracy",
        title="Accuracy over time",
        template="simple_white",
        labels={"x": "Round", "y": "Accuracy"},
        points=False
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
        print(
            f"Average Accuracy at round {r}: {avg_acc:.2f} ± {std_acc:.2f}")

        # Print how many % better the full fit accuracy is compared to the last round
        improvement = 100 * ((acc_full / avg_acc) - 1)
        print(f"Full fit is {improvement:.2f}% better than the fit in round {r}")
    print("\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the contextual bandit algorithm")
    parser.add_argument("--n_sims", type=int, default=10, help="Number of simulations to run")
    parser.add_argument("--n_rounds", type=int, default=100, help="Number of rounds per simulation")
    parser.add_argument("--savedir", type=str, help="Name of the savedir.")
    parser.add_argument("--motivation", type=bool, help="Flag to set the plot for the motivation use-case")


    
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
    # Initialize HardwareManager with the CSV file path
    HardwareManager.init_manager(savedir.joinpath("data/data.csv"))


    # tolerance_ratio is a float >= 0 to represent the amount of slowdown allowed for a less resource intensive hardware
    # when set to None, it selects the fastest hardware without reassessing in the case of a tie.
    # when set to 0, it selects the fastest hardware, preferring cheaper hardwares in the case of a tie.
    tolerance_ratio: Union[float, None] = 0.0
    # tolerance seconds represents how many seconds of slowdown is acceptable for a less resource intensive hardware
    tolerance_seconds: int = 0
    # feature_cols = ALL_FEATURE_COLS
    feature_cols = ['area']
    feats = '_'.join(feature_cols)
    results_dir = savedir.joinpath(feats)
    results_dir.mkdir(parents=True, exist_ok=True)


    run_sim(
        n_rounds=n_rounds,
        tolerance_ratio=tolerance_ratio,
        tolerance_seconds=tolerance_seconds,
        feature_cols=feature_cols,
        savedir=results_dir,
        motivation=motivation
    )

    run(
        n_sims=n_sims,
        n_rounds=n_rounds,
        feature_cols=feature_cols,
        savedir=results_dir,
        tolerance_ratio=tolerance_ratio,
        tolerance_seconds=tolerance_seconds
    )


if __name__ == "__main__":
    main()
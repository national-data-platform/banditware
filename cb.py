from functools import lru_cache, wraps
from itertools import product
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # for RMSE calculation


this_dir = pathlib.Path(__file__).parent.absolute()
results_dir = this_dir / "results"

@lru_cache(maxsize=None)
def load_data(*feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data from the csv file"""
    feature_cols = list(feature_cols)
    _data = pd.read_csv(f"{this_dir}/results/data/data.csv")
    unique_features = _data[feature_cols].drop_duplicates().values
    unique_hardware = sorted(_data["hardware"].unique())

    # set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # reserve 20% of the data for testing
    # must make sure that training data contains all feature/hardware combinations
    datas = []
    test_datas = []
    for features, hardware in product(unique_features, unique_hardware):
        data: pd.DataFrame = _data[
            (np.logical_and.reduce([_data[col] == feature for col, feature in zip(feature_cols, features)])) &
            (_data["hardware"] == hardware)
        ]
        test_data = data.sample(frac=0.2)
        data = data.drop(test_data.index)
        datas.append(data)
        test_datas.append(test_data)
    
    data = pd.concat(datas)
    test_data = pd.concat(test_datas)

    # rename hardware to integers 0, 1, 2, ...
    hardware_mapping = {hardware: i for i, hardware in enumerate(unique_hardware)}
    data["hardware"] = data["hardware"].map(hardware_mapping)
    

    return data, test_data

def run_sim(n_rounds: int = 100,
            e_start: float = 1,
            e_decay: float = 0.99,
            e_min: float = 0.0,
            feature_cols: List[str] = ["area"],
            savedir: pathlib.Path = None) -> pd.DataFrame:
    """Run a single simulation of the contextual bandit algorithm"""        
    data, test_data = load_data(*feature_cols)
    unique_features = data[feature_cols].drop_duplicates().values
    unique_hardware = sorted(data["hardware"].unique())

    
    @lru_cache(maxsize=None)
    def get_truth() -> Tuple[List[np.ndarray], List[float], List[float]]:
        """Get the true coefficients and noise for each hardware"""	
        bias_truth = []
        std_truth = []
        coef_truth = []
        for hardware in unique_hardware:
            _data = data[data["hardware"] == hardware]
            X = _data[feature_cols].values
            y = _data["runtime"].values
            reg = LinearRegression().fit(X, y)
            coef_truth.append(reg.coef_)
            bias_truth.append(reg.intercept_)
            std_truth.append(np.std(reg.predict(X) - y))
        return coef_truth, bias_truth, std_truth

    def get_runtimes(features: np.ndarray, hardware: int) -> np.ndarray:
        """Get the runtime of a workflow on a specific hardware"""
    
        filtered_data = data.loc[
            (data["hardware"] == hardware) &
            np.logical_and.reduce([data[col] == feature for col, feature in zip(feature_cols, features)])
        ]

        if filtered_data.empty:
            print(hardware, dict(zip(feature_cols, features)))

        return filtered_data["runtime"].values

    def sample_runtime(features: np.ndarray, hardware: int) -> float:
        """Sample the runtime of a workflow on a specific hardware"""
        runtimes = get_runtimes(features, hardware)
        return np.random.choice(runtimes)

    def get_best_hardware(features: np.ndarray, hardware: List[int]) -> int:
        # coef_truth, bias_truth, std_truth = get_truth()
        # return min(hardware, key=lambda h: np.dot(coef_truth[h], features) + bias_truth[h])

        # get hardware with lowest average runtime for the given features
        return min(hardware, key=lambda h: np.mean(get_runtimes(features, h)))
    
    @lru_cache(maxsize=None)
    def get_best_hardwares() -> List[int]:
        return [get_best_hardware(features, unique_hardware) for features in test_data[feature_cols].values]
    
    def get_model_accuracy(coefs: Dict[int, List[float]], bias: Dict[int, float], std: Dict[int, float]) -> float:
        """Get the accuracy of the model on the test data - how often does it predict the best hardware"""
        # use get_best_hardware to get the ground truth
        truth = get_best_hardwares()
        # use the model to predict the best hardware
        pred = [min(unique_hardware, key=lambda h: np.dot(coefs[h], features) + bias[h]) for features in test_data[feature_cols].values]
        return np.mean([t == p for t, p in zip(truth, pred)])
        
    # Check that there are runtimes for all hardware/features
    for hardware, features in product(unique_hardware, unique_features):
        runtimes = get_runtimes(features, hardware)
        assert(len(runtimes) > 0)

    samples: Dict[int, List[Tuple[List[float], float]]] = {i: [] for i in unique_hardware}
     
    coefs: Dict[int, List[float]] = {i: np.zeros(len(feature_cols)) for i in unique_hardware}
    bias = {i: 0 for i in unique_hardware}
    std = {i: 0 for i in unique_hardware}
    rows_runtime = []
    e = e_start
    
    # iterate through unique_features in a random order
    for i in range(n_rounds):
        features = random.choice(unique_features)
        if np.random.rand() < e:
            # Randomly select a hardware
            hardware = np.random.choice(unique_hardware)
        else:
            # Select the hardware with the best estimated runtime
            hardware = min(unique_hardware, key=lambda h: np.dot(coefs[h], features) + bias[h])

        # Sample the runtime of the workflow on the selected hardware
        runtime = sample_runtime(features, hardware)
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

        # Calculate quality of the model on *all* the data
        X_all = data[feature_cols].values
        y_all = data["runtime"].values
        y_pred = np.array([np.dot(coefs[h], x) + bias[h] for h, x in zip(data["hardware"], X_all)])
        
        rmse = np.sqrt(mean_squared_error(y_all, y_pred))
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
            for x in np.linspace(data[feature_col].min(), data[feature_col].max(), 10):
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
        fig.update_xaxes(title_text=feature_cols[0])
        fig.update_yaxes(title_text="Runtime", matches="y")

        # Hide duplicate y-axis titles on other facets
        for axis in fig.layout:
            if axis.startswith("yaxis") and axis != "yaxis":
                fig.layout[axis].title.text = ""

        fig.write_html(f"{savedir}/cb_{feature_cols[0]}.html")
        fig.write_image(f"{savedir}/cb_{feature_cols[0]}.png")

        plot_motivation(df, feature_cols[0], savedir)
        

    # compute rmse on full data (excluding test data)
    X_all = data[feature_cols].values
    y_all = data["runtime"].values
    y_pred = np.array([np.dot(coef_truth[h], x) + bias_truth[h] for h, x in zip(data["hardware"], X_all)])
    rmse = np.sqrt(mean_squared_error(y_all, y_pred))

    # compute accuracy on full data (excluding test data)
    acc = get_model_accuracy(coef_truth, bias_truth, std_truth)

    baseline_info = {
        "round": n_rounds,
        "rmse": rmse,
        "accuracy": acc
    }

    return pd.DataFrame(rows_runtime), baseline_info

def run(n_sims: int, 
        n_rounds: int,
        feature_cols: List[str],
        savedir: pathlib.Path):
    dfs = []
    baseline_infos = []
    for i in range(n_sims):
        print(f"Running simulation {i+1}/{n_sims}", end="\r")
        df, baseline_info = run_sim(n_rounds=n_rounds, feature_cols=feature_cols)
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

    savedir.mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{savedir}/rmse.html")
    fig.write_image(f"{savedir}/rmse.png")


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
    fig.write_html(f"{savedir}/rmse_line.html")
    fig.write_image(f"{savedir}/rmse_line.png")


    # get accuracy of full data
    acc_full = np.mean([info["accuracy"] for info in baseline_infos])
    assert np.allclose([info["accuracy"] for info in baseline_infos], acc_full)

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
    fig.write_html(f"{savedir}/accuracy_line.html")
    fig.write_image(f"{savedir}/accuracy_line.png")

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
    fig.write_html(f"{savedir}/accuracy.html")
    fig.write_image(f"{savedir}/accuracy.png")


    # get average and std rmse at last round
    for r in [10, n_rounds]:
        last_round_rmse = df_sim[df_sim["round"] == r - 1]
        avg_rmse = last_round_rmse["rmse"].mean()
        std_rmse = last_round_rmse["rmse"].std()
        print(f"Average RMSE at round {r}: {avg_rmse:.2f} ± {std_rmse:.2f}")

        # Print how many % better the full fit is compared to the last round
        improvement = 100 * (1 - avg_rmse / rmse_full)
        print(f"Full fit is {improvement:.2f}% better than the fit in round {r}")
    
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

def main():
    n_sims = 10
    n_rounds = 50
    

    run(
        n_sims=n_sims,
        n_rounds=n_rounds,
        feature_cols=["num_tasks"],
        savedir=results_dir / "num_tasks"
    )

    run_sim(
        n_rounds=n_rounds,
        feature_cols=["num_tasks"],
        savedir=results_dir / "num_tasks"
    )

    # run(
    #     n_sims=n_sims,
    #     n_rounds=n_rounds,
    #     feature_cols=["average_memory"],
    #     savedir=results_dir / "average_memory"
    # )
    # run(
    #     n_sims=n_sims,
    #     n_rounds=n_rounds,
    #     feature_cols=["average_cpu", "average_memory"],
    #     savedir=results_dir / "all"
    # )







if __name__ == "__main__":
    main()
from functools import lru_cache, wraps
from itertools import product
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px

from sklearn.linear_model import LinearRegression

this_dir = pathlib.Path(__file__).parent.absolute()
results_dir = this_dir / "results"

FEATURE_COLS = [
    # "canopy_moisture",
    # "run_max_mem_rss_bytes",
    # "sim_time",
    # "surface_moisture",
    # "threads",
    # "wind_direction",
    # "wind_speed",
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
def load_data() -> pd.DataFrame:
    """Load the data from the csv file"""
    data = pd.read_csv(f"{this_dir}/results/data/data.csv")
    return data

def get_runtimes(features: np.ndarray, hardware: int) -> np.ndarray:
    """Get the runtime of a workflow on a specific hardware"""
    data = load_data()

    filtered_data = data[
        (data["hardware"] == hardware) &
        np.logical_and.reduce([data[col] == feature for col, feature in zip(FEATURE_COLS, features)])
    ]

    return filtered_data["runtime"].values

def sample_runtime(features: np.ndarray, hardware: int) -> float:
    """Sample the runtime of a workflow on a specific hardware"""
    runtimes = get_runtimes(features, hardware)
    return np.random.choice(runtimes)

def get_best_hardware(features: np.ndarray, hardware: List[int]) -> int:
    """Get the hardware with the best average runtime for a specific set of features"""
    runtimes = [np.mean(get_runtimes(features, h)) for h in hardware]
    return hardware[np.argmin(runtimes)]

def sample_best_runtime(features: np.ndarray, hardware: List[int]) -> float:
    """Sample the runtime of a workflow on the best hardware for a specific set of features"""
    best_hardware = get_best_hardware(features, hardware)
    return sample_runtime(features, best_hardware)

def main():
    data = load_data()

    unique_features = data[FEATURE_COLS].drop_duplicates().values
    unique_hardware = data["hardware"].unique()

    print("Unique features:", unique_features)
    print("Unique hardware:", unique_hardware)

    for hardware, features in product(unique_hardware, unique_features):
        runtimes = get_runtimes(features, hardware)
        assert(len(runtimes) > 0)


    # fit a linear regression model to get the coefficients
    noise_mean = []
    noise_std = []
    coef_truth = []
    for hardware in unique_hardware:
        _data = data[data["hardware"] == hardware]
        X = _data[FEATURE_COLS].values
        y = _data["runtime"].values
        reg = LinearRegression().fit(X, y)
        coef_truth.append(reg.coef_)
        noise_mean.append(reg.intercept_)
        noise_std.append(np.std(reg.predict(X) - y))


    # Use e-greedy algo to find the best hardware
    N_ROUNDS = 30
    # Setting up the decaying epsilon
    e_start = 1 # epsilon
    e_decay = 0.99
    e_min = 0.0

    samples: Dict[int, List[Tuple[List[float], float]]] = {i: [] for i in unique_hardware}

     
    # Estimated coefficients (this is what we are trying to find)
    coefs: Dict[int, List[float]] = {i: np.zeros(len(FEATURE_COLS)) for i in unique_hardware}
    
    # Estimated noise coefficients (mean, std) 
    noise_coefs: Dict[int, Tuple[float, float]] = {i: (0, 0) for i in unique_hardware}
    rows_runtime = []
    e = e_start
    
    # iterate through unique_features in a random order
    for i in range(N_ROUNDS):
        print(f"Round {i+1}/{N_ROUNDS}")
        features = random.choice(unique_features)
        if np.random.rand() < e:
            # Randomly select a hardware
            hardware = np.random.choice(unique_hardware)
        else:
            # Select the hardware with the best estimated runtime
            hardware = get_best_hardware(features, unique_hardware)

        # Sample the runtime of the workflow on the selected hardware
        runtime = sample_runtime(features, hardware)
        samples[hardware].append((features, runtime)) 

        rows_runtime.append({
            "round": i,
            "runtime": runtime,
            "best_runtime": sample_best_runtime(features, unique_hardware)
        })

        # Update the coefficients
        X, y = zip(*samples[hardware])
        X = np.array(X)
        y = np.array(y)
        reg = LinearRegression().fit(X, y)
        coefs[hardware] = reg.coef_
        noise_coefs[hardware] = (reg.intercept_, np.std(reg.predict(X) - y))

        # Decay epsilon
        e = max(e * e_decay, e_min)
    
    
    # Print predicted and actual best hardware for each workflow
    rows_pred = []
    for idx, features in enumerate(unique_features):
        best_hardware = np.argmin([np.dot(coefs[h], features) + noise_coefs[h][0] for h in unique_hardware])
        actual_best_hardware = get_best_hardware(features, unique_hardware)
        rows_pred.append([idx, best_hardware, actual_best_hardware])
    df_pred = pd.DataFrame(rows_pred, columns=["workflow", "predicted_best_hardware", "actual_best_hardware"])
    print(df_pred)

    if len(FEATURE_COLS) == 1:
        feature_col = FEATURE_COLS[0]
        rows = []
        for hardware_idx in unique_hardware:
            for x in np.linspace(data[feature_col].min(), data[feature_col].max(), 10):
                rows.append({
                    "x": x,
                    "y": coefs[hardware_idx][0] * x + noise_coefs[hardware_idx][0],
                    "mode": "Predicted",
                    "Hardware": hardware_idx,
                    "error": noise_coefs[hardware_idx][1]
                })

                rows.append({
                    "x": x + 0.01,
                    "y": coef_truth[hardware_idx][0] * x + noise_mean[hardware_idx],
                    "mode": "Actual",
                    "Hardware": hardware_idx,
                    "error": noise_std[hardware_idx]
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

        fig.write_html(f"{results_dir}/cb.html")
        fig.write_image(f"{results_dir}/cb.png")



if __name__ == "__main__":
    main()
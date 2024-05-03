from typing import Dict, List, Tuple
import numpy as np
# for linear regression
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd


def main():
    # five different combos of hardware with N_FEATURES for mean
    N_HARDWARE = 3
    N_FEATURES = 10
    N_ROUNDS = 100
    e_start = 1 # epsilon
    e_decay = 0.9
    e_min = 0.0

    noise_mean, noise_std = np.random.random(N_HARDWARE) * 5, np.random.random(N_HARDWARE) * 0.1
    coef_truth = np.random.random(size=(N_HARDWARE,N_FEATURES)) * 4 - 2

    sample_runtime = lambda hardware_idx, workflow_features: (
        np.dot(coef_truth[hardware_idx], workflow_features) + 
        np.random.normal(noise_mean[hardware_idx], noise_std[hardware_idx])
    )

    
    sample_best_runtime = lambda workflow_features: (
        np.min([
            np.dot(coef_truth[hardware_idx], workflow_features) + noise_mean[hardware_idx]
            for hardware_idx in range(N_HARDWARE)
        ])
    )


    # Use e-greedy algo to find the best hardware for after 40 runs feeding the information back
    # to the algo
    # e-greedy algo
    # samples gathered over time
    # samples = {1: [(workflow, runtime), ...], ...}
    # where workflow is a list of workflow features (input features)
    samples: Dict[int, List[Tuple[List[float], float]]] = {i: [] for i in range(N_HARDWARE)}
    # Estimated coefficients
    coefs: Dict[int, List[float]] = {i: np.zeros(N_FEATURES) for i in range(N_HARDWARE)}
    # Estimated noise coefficients (mean, std)
    noise_coefs: Dict[int, Tuple[float, float]] = {i: (0, 0) for i in range(N_HARDWARE)}
    rows_predictions = []
    rows_runtime = []
    e = e_start
    for round_i in range(N_ROUNDS):
        workflow = np.random.random(N_FEATURES)
        if np.random.random() < e:
            hardware_idx = np.random.randint(N_HARDWARE)
        else:
            hardware_idx = np.argmin([
                np.dot(coefs[i], workflow) + noise_coefs[i][0] # predicted runtime
                for i in range(N_HARDWARE)
            ])

        # Update the samples
        actual_runtime = sample_runtime(hardware_idx, workflow)
        samples[hardware_idx].append((workflow, actual_runtime))

        best_runtime = sample_best_runtime(workflow)
        rows_runtime.append({
            "round": round_i,
            "runtime": actual_runtime,
            "best_runtime": best_runtime,
        })

        # Update the coefficients
        X, y = zip(*samples[hardware_idx])
        X = np.array(X)
        y = np.array(y)
        reg = LinearRegression().fit(X, y)
        coefs[hardware_idx] = reg.coef_
        noise_coefs[hardware_idx] = (reg.intercept_, np.std(reg.predict(X) - y))

        # Decay epsilon
        e = max(e * e_decay, e_min)

        # Evaluate how good the model is at this point
        for samples_idx in range(N_HARDWARE):
            for h_idx in range(N_HARDWARE):
                workflow = np.random.random(N_FEATURES)
                predicted_runtime = np.dot(coefs[h_idx], workflow) + noise_coefs[h_idx][0]
                actual_runtime = sample_runtime(h_idx, workflow)
                rows_predictions.append({
                    "round": round_i,
                    "hardware": h_idx,
                    "error": np.abs(predicted_runtime - actual_runtime)
                })

    df_predictions = pd.DataFrame(rows_predictions)
    df_predictions = df_predictions.groupby(["round", "hardware"]).mean().reset_index()
    fig = px.line(
        df_predictions, x="round", y="error", color="hardware",
        template="simple_white"
    )
    fig.write_html("predictions.html")
    fig.write_image("predictions.png")

    df_runtime = pd.DataFrame(rows_runtime)
    # plot actual - best runtime over rounds
    df_runtime["error"] = df_runtime["runtime"] - df_runtime["best_runtime"]
    fig = px.line(
        df_runtime, x="round", y="error",
        template="simple_white"
    )
    fig.write_html("error.html")
    fig.write_image("error.png")


    df_runtime = df_runtime.melt(id_vars=["round"], value_vars=["runtime", "best_runtime"])
    fig = px.line(
        df_runtime, x="round", y="value", color="variable",
        template="simple_white"
    )
    fig.write_html("runtime.html")
    fig.write_image("runtime.png")

    # Plot epsilon over rounds
    df_epsilon = pd.DataFrame({
        "round": range(N_ROUNDS),
        "epsilon": [max(e_start * e_decay ** i, e_min) for i in range(N_ROUNDS)]
    })
    fig = px.line(
        df_epsilon, x="round", y="epsilon",
        template="simple_white"
    )
    # set plot y axis to 0-1
    fig.update_yaxes(range=[0, 1])
    fig.write_html("epsilon.html")
    fig.write_image("epsilon.png")


    print(f"Predicted Coefs")
    for hardware_idx in range(N_HARDWARE):
        print(f"Hardware {hardware_idx}")
        print(f"Predicted Coefs: {coefs[hardware_idx]}")
        print(f"Predicted Noise Coefs: {noise_coefs[hardware_idx]}")
        print(f"Actual Coefs: {coef_truth[hardware_idx]}")
        print(f"Actual Noise Coefs: {noise_mean[hardware_idx], noise_std[hardware_idx]}")

    if N_FEATURES == 1:
        # Plot prediction vs actual 
        rows = []
        for hardware_idx in range(N_HARDWARE):
            for x in np.linspace(0, 1, 100):
                rows.append({
                    "x": x,
                    "y": coefs[hardware_idx][0] * x + noise_coefs[hardware_idx][0],
                    "mode": "Predicted",
                    "Hardware": hardware_idx
                })
                rows.append({
                    "x": x,
                    "y": coef_truth[hardware_idx][0] * x + noise_mean[hardware_idx],
                    "mode": "Actual",
                    "Hardware": hardware_idx
                })

        df = pd.DataFrame(rows)
        fig = px.scatter(
            df, x="x", y="y", color="mode", 
            facet_col="Hardware",
            template="simple_white"
        )
        fig.write_html("cb.html")
        fig.write_image("cb.png")


if __name__ == "__main__":
    main()
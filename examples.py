import pandas as pd
import numpy as np
from pre_process import preprocess
from models import Model
from banditware import BanditWare

np.random.seed(42)

def from_nothing(full_data, feature_cols, model_choice):
    """
    Represents General NDP Workflow: 
        * User start with no historical data and gets a hardware suggestion
        * User continuosly more historical data and gets hardware suggestions over time
    """
    print("BanditWare going from nothing to having training data")
    data = full_data.sample(frac=1, replace=False)
    data = data.reset_index(drop=True)
    one_row = data.iloc[[0]]
    two_rows = data.iloc[1:3]
    several_rows = data.iloc[4:30]
    bw = BanditWare(feature_cols=feature_cols, model_choice=model_choice)
    print("\tsuggested_hardware:", bw.suggest_hardware())
    bw.add_historical_data(one_row)
    print("\tsuggested_hardware:", bw.suggest_hardware())
    bw.add_historical_data(two_rows)
    print("\tsuggested_hardware:", bw.suggest_hardware())
    bw.add_historical_data(several_rows)
    print("\tsuggested_hardware:", bw.suggest_hardware())


def test_accuracy(data, feature_cols, model_choice):
    """
    Tests accuracy of BanditWare on a given datasest. 
    `bw.test_accuracy()` Uses newly trained models (only trained on the subset),
    even if `bw` is already trained.
    """
    bw = BanditWare(
        data = data,
        feature_cols = feature_cols,
        # feature_cols = all_bp3d_feature_cols,
        save_dir = "./bp3d",
        plot_motivation = False,
        # model_choice = Model.LINEAR_REGRESSION
        model_choice = model_choice
    )
    bw.test_accuracy(model_choice=model_choice, plot_runtime_predictions=False)

def main():
    bp3d_data = preprocess(base_path="data/bp3d_data")
    matmul_data = preprocess(data_file="data/matmul.csv")

    all_bp3d_feature_cols = ["area", "canopy_moisture", "run_max_mem_rss_bytes",
                             "sim_time","surface_moisture","wind_direction","wind_speed"]
    bp3d_subset_features = ["area", "canopy_moisture","wind_direction","wind_speed"]

    test_accuracy(bp3d_data, feature_cols=bp3d_subset_features, model_choice=Model.RANDOM_FOREST)
    from_nothing(bp3d_data, bp3d_subset_features, Model.LINEAR_REGRESSION)

if __name__ == "__main__":
    main()

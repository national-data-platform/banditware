import pandas as pd
import numpy as np
from pre_process import preprocess
from models import Model
from banditware import BanditWare

np.random.seed(42)

def from_nothing(full_data, feature_cols, model_choice):
    data = full_data.sample(frac=1, replace=False)
    data = data.reset_index(drop=True)
    one_row = data.iloc[0]
    two_rows = data.iloc[1:3]
    several_rows = data.iloc[4:30]
    bw = BanditWare(feature_cols=feature_cols, model_choice=model_choice)
    print(bw.suggest_hardware())
    bw.add_historical_data(one_row)
    print(bw.suggest_hardware())
    bw.add_historical_data(two_rows)
    print(bw.suggest_hardware())
    bw.add_historical_data(several_rows)
    print(bw.suggest_hardware())



def main():
    print("Preparing data")
    bp3d_data = preprocess(base_path="data/bp3d_data")
    matmul_data = preprocess(data_file="data/matmul.csv")
    

    all_bp3d_feature_cols = ["area", "canopy_moisture", "run_max_mem_rss_bytes", "sim_time","surface_moisture","wind_direction","wind_speed"]
    bw = BanditWare(
        data = bp3d_data,
        # feature_cols = ["area", "canopy_moisture","surface_moisture","wind_direction","wind_speed"],
        feature_cols = all_bp3d_feature_cols,
        save_dir = "./bp3d",
        plot_motivation = False,
        # model_choice = Model.LINEAR_REGRESSION
        model_choice = Model.DECISION_TREE
    )
    bw.train()
    bw.test_accuracy(model_choice=Model.RANDOM_FOREST, plot_runtime_predictions=False)

    # from_nothing(bp3d_data, all_bp3d_feature_cols, model_choice=Model.RANDOM_FOREST)

if __name__ == "__main__":
    main()
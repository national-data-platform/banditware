import pandas as pd
import numpy as np
from pre_process import preprocess
from models import Model
from banditware import BanditWare

np.random.seed(42)

def main():
    # bp3d_data = preprocess(base_path="data/bp3d_data")
    # matmul_data = preprocess(data_file="data/matmul.csv")
    bp3d_data = pd.read_csv("preprocessed_data/bp3d.csv")
    matmul_data = pd.read_csv("preprocessed_data/matmul.csv")

    matmul_feature_cols = ["size", "sparsity", "min", "max"]
    all_bp3d_feature_cols = ["area", "canopy_moisture", "run_max_mem_rss_bytes",
                             "sim_time","surface_moisture","wind_direction","wind_speed"]
    bp3d_subset_features = ["area", "canopy_moisture","wind_direction","wind_speed"]
    # test_accuracy(bp3d_data, feature_cols=bp3d_subset_features, model_choice=Model.RANDOM_FOREST)
    # from_nothing(bp3d_data, bp3d_subset_features, Model.LINEAR_REGRESSION)

    test_accuracy(
        matmul_data,
        feature_cols=matmul_feature_cols,
        model_choice=Model.DECISION_TREE,
        plot_predictions=True
        )
    from_nothing(
        matmul_data,
        feature_cols=matmul_feature_cols,
        model_choice=Model.RANDOM_FOREST,
        features=[12_500, 0.0, 1, 10000],
        prohibit_exploration=True
        )
    change_application(
        data1=bp3d_data,
        features1=bp3d_subset_features,
        save_dir1="./bp3d",
        model_choice1=Model.BAYESIAN_RIDGE,
        data2=matmul_data,
        features2=matmul_feature_cols,
        save_dir2="./matmul",
        model_choice2=Model.DECISION_TREE,
    )


def from_nothing(full_data, feature_cols, model_choice, features=None, prohibit_exploration=False):
    """
    Represents General NDP Workflow: 
        * User start with no historical data and gets a hardware suggestion
        * User continuosly creates more historical data and gets hardware suggestions over time
    """
    print_title("Starting from nothing then adding training data")
    data = full_data.sample(frac=1, replace=False)
    data = data.reset_index(drop=True)
    one_row = data.iloc[[0]]
    two_rows = data.iloc[1:3]
    several_rows = data.iloc[3:30]
    rest_of_rows = data.iloc[30:]
    incremental_updates = [one_row, two_rows, several_rows, rest_of_rows]
    save_dir = "./matmul" if "size" in data.columns else "./bp3d"
    
    bw = BanditWare(feature_cols=feature_cols, save_dir=save_dir, model_choice=model_choice)
    
    print("\tsuggested_hardware:", bw.suggest_hardware(prohibit_exploration=prohibit_exploration))
    for df in incremental_updates:
        bw.add_historical_data(
            df, update_saved_data=False, retrain=True)
        suggested_hardware = bw.suggest_hardware(
            features, prohibit_exploration=prohibit_exploration, smart_suggest=True)
        print("\tsuggested_hardware:", suggested_hardware)


def test_accuracy(data, feature_cols, model_choice, plot_predictions=False):
    """
    Tests accuracy of BanditWare on a given datasest. 
    `bw.test_accuracy()` Uses newly trained models (only trained on the subset),
    even if `bw` is already trained.
    """
    print_title("Testing Accuracy")
    save_dir = "./matmul" if "size" in data.columns else "./bp3d"
    bw = BanditWare(
        data = data,
        feature_cols = feature_cols,
        # feature_cols = all_bp3d_feature_cols,
        save_dir = save_dir,
        # model_choice = Model.LINEAR_REGRESSION
        model_choice = model_choice
    )
    bw.test_accuracy(model_choice=model_choice, plot_runtime_predictions=plot_predictions)

def change_application(data1, features1, save_dir1, model_choice1,
                       data2, features2, save_dir2, model_choice2):
    """
    Shows BanditWare's ability to change applications. 
    Tests accuracy on the first application, then switches to a second application.
    """
    print_title("Demonstrating Changing applications")
    print("BurnPro3D")
    bw = BanditWare(data=data1,
                    feature_cols=features1,
                    save_dir=save_dir1,
                    model_choice=model_choice1)
    bw.test_accuracy()
    print("\nMatrix Multiplication")
    bw.change_application(data2, save_dir2, features2, model_choice2)
    bw.test_accuracy()

def print_title(title):
    print("\n", "="*50, title, "="*50, "", sep="\n")

if __name__ == "__main__":
    main()

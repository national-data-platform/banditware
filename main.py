from models import Model
from banditware import BanditWare
import pandas as pd

pd.set_option("display.max_columns", None)

if __name__ == "__main__":
    data = pd.read_csv("preprocessed_data/matmul_w_time.csv")
    print(data)
    bw = BanditWare(
        data=data,  # Note: if you are running this after the first run was stopped halfway, comment this line out if you don't want the previous progress to be lost
        feature_cols=["size", "sparsity", "min", "max"],
        save_dir="mm_time_new",
        model_choice=Model.RANDOM_FOREST,
        ndp_username="rshende",
    )
    bw.query_performance_data(cpu=True, memory=True, gpu=False)
    bw.save_data()
    print(bw.get_historical_data())

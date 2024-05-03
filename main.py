import json
import pandas as pd
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    data_json = json.loads((thisdir / 'uniform-pgml-success.bp3d.json').read_text())

    df = pd.read_json(data_json["df"])

    print(df.columns)
    print(df['output'])
    

if __name__ == "__main__":
    main()
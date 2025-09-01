import pathlib
import json
import argparse
import pandas as pd
import enum
import os
import re
from typing import Dict, Union

this_dir = pathlib.Path(__file__).resolve().parent

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-process data for the integrated performance framework")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_path", type=pathlib.Path, help="Path to the head folder where data is stored.")
    group.add_argument("--data_file", type=pathlib.Path, help="Path to the data csv file.")
    return parser

def traverse_data(base_path: pathlib.Path) -> Dict[str, pd.DataFrame]:
    """Traverses the results folder and loads all data found in csv files"""
    # Traverse the directory structure
    dfs = {}
    folder_name = ""
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                
                # Extract the parent folder name
                folder_name = os.path.basename(root)
                
                # Read the CSV file into a DataFrame
                df = load_data(file_path)
                
                # Add the DataFrame to the dictionary under the folder name
                if folder_name not in dfs:
                    dfs[folder_name] = pd.DataFrame()
                
                dfs[folder_name] = pd.concat([dfs[folder_name], df], ignore_index=True)
    
    return dfs


def identify_hardware(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Identify the hardware used in the data"""
    ext_df = pd.DataFrame()
    for hardware_str, hardware_df in data.items():
        # Get the number of cpus and memory
        pattern = r"(\d+)cores_(\d+)gb"
        match = re.search(pattern, hardware_str)

        if match:
            cores = int(match.group(1))
            gb = int(match.group(2))
            # Check if value is a DataFrame
            if not isinstance(hardware_df, pd.DataFrame):
                print(f"Error: Expected a DataFrame, but got {type(hardware_df)} for key {hardware_str}")
                continue
            ext_df = extend_df(hardware_df, cores, gb, ext_df)
        else:
            print("Directory did not follow the expected pattern")

    return ext_df

def extend_df(df: pd.DataFrame, cpu_cores: int, mem_gb: int, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extend the dataframe with the hardware information"""
    # df["hardware"] = '_'.join(hardware.value[0])
    # cpus, mem = hardware.value[0]
    # df["hardware"] = f"{cpus}_{mem}"
    df["hardware"] = f"{cpu_cores}_{mem_gb}"
    # df["hardware"] = hardware.value[1]
    ext_df = pd.concat([ext_df, df], ignore_index=True)

    return ext_df   


def load_data(data_path: pathlib.Path) -> pd.DataFrame:
    """Load data from the data file"""
    try:
        # Data in a dataframe already
        df = pd.read_csv(data_path)
    except:
        # Data is in a json file
        with open(data_path, 'r') as f:
            info = json.load(f)
        df = pd.DataFrame(info)
    
    return df

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data"""

    # Remove unamed column
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # replace NaN with 0    
    data = data.fillna(0)
    data = data.reset_index(drop=True)

    return data

def preprocess(base_path:Union[str,None]=None, data_file:Union[str,None]=None) -> pd.DataFrame:
    correct_inputs = (base_path is not None) ^ (data_file is not None)
    if not correct_inputs:
        raise ValueError("Exactly one of base_path or data_file must be specified")
    if base_path is not None:
        dfs = traverse_data(pathlib.Path(base_path))
        full_df = identify_hardware(dfs)
        return clean_data(full_df)
    full_df = pd.read_csv(data_file)
    return clean_data(full_df)


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    full_df = None
    if args.base_path:
        dfs = traverse_data(pathlib.Path(args.base_path))
        full_df = identify_hardware(dfs)
    else:
        full_df = pd.read_csv(args.data_file)
    full_df = clean_data(full_df)


    # Save the data in a csv file 
    save_dir = this_dir.joinpath("results/data")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = f"{save_dir}/data.csv"
    full_df.to_csv(save_file, index=False)
    
    

if __name__ == "__main__":
    main()

# base_path = /home/tainagdcoleman/grafana_data_retrieval/data
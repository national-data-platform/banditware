import pathlib
import json
import argparse
import pandas as pd
import enum
import os
import re
from typing import Dict

this_dir = pathlib.Path(__file__).resolve().parent

class Hardware(enum.Enum):
    """Characterize hardware by the number of cpus and memory in a tuple (#cpu, mem (MB)), for example: H1 = (1, 16)"""

    H1 = [(2, 16), 0]
    H2 = [(3, 24), 1]
    H3 = [(4, 16), 2]  # Add more if needed

    @classmethod
    def from_spec(cls, cpus, mem):
        for hardware in cls:
            if hardware.value[0] == (cpus, mem):
                return hardware
        raise ValueError(f"No matching hardware for {cpus} CPUs and {mem} MB memory.")
    
    @classmethod
    def spec_from_hardware(cls, hardware_number: int) -> tuple[int, int]:
        # Given an int (between 0 and 4), return the (cpu, mem gb) tuple
        for hardware in cls:
            if hardware.value[1] == hardware_number:
                return hardware.value[0]
        raise ValueError(f"No hardware found for number {hardware_number}.")
    
def get_hardware(name: str) -> Hardware:
    """Get hardware values"""
    try:
        return Hardware[name.upper()]
    except KeyError:
        raise ValueError(f"Invalid hardware name: {name}") 

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-process data for the integrated performance framework")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_path", type=pathlib.Path, help="Path to the head folder where data is stored.")
    group.add_argument("--data_file", type=pathlib.Path, help="Path to the data csv file.")

    return parser

def traverse_data(base_path: pathlib.Path) -> pd.DataFrame:
    """Traverses the results folder and loads all data found in csv files"""
    # Traverse the directory structure
    dfs = {}
    folder_name = ""
    for root, dirs, files in os.walk(base_path):
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
    for key, value in data.items():
        # Get the number of cpus and memory
        pattern = r"(\d+)cores_(\d+)gb"
        match = re.search(pattern, key)

        if match:
            cores = match.group(1)
            gb = match.group(2)

            hardware = Hardware.from_spec(int(cores), int(gb))
            print(f"Hardware: {hardware.name} = {hardware.value}")
            
            # Check if value is a DataFrame
            if not isinstance(value, pd.DataFrame):
                print(f"Error: Expected a DataFrame, but got {type(value)} for key {key}")
                continue

            ext_df = extend_df(value, hardware, ext_df)
        else:
            print("Directory did not follow the expected pattern")

    return ext_df

def extend_df(df: pd.DataFrame, hardware: Hardware, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extend the dataframe with the hardware information"""
    # df["hardware"] = '_'.join(hardware.value[0])
    cpus, mem = hardware.value[0]
    df["hardware"] = f"{cpus}_{mem}"
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

    # have to do this for now
    shared_areas = [1053216.0, 1854216.0, 1369900.0, 828144.0, 2543220.0]
    data = data[data['area'].isin(shared_areas)]
    data = data.reset_index(drop=True)
    
    return data


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
    full_df.to_csv(f"{save_dir}/data.csv", index=False)
    
    

if __name__ == "__main__":
    main()

# base_path = /home/tainagdcoleman/grafana_data_retrieval/data
import pathlib 
import argparse
import pandas as pd
import re
import pprint as pp
from typing import Dict

def set_data_banditware(data: Dict):
    flattened_data = []
    for config, entries in data.items():
        for entry in entries:
            # Add the 'config' value as a new field in each dictionary
            flattened_data.append({**entry, 'hardware': config})


    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Save it to a CSV file
    df.to_csv("results.csv", index=False)
    

def get_parser():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--input", type=str, help="Root directory of the results")
    parser.add_argument("--name", type=str, help="Name of the experiment")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    root_dir = pathlib.Path(args.input)
    name = args.name
    # Initialize times dictionary
    times = {}

    # Define a dictionary to store the parsed data where the keys are the hardware names and the values are dataframes that contain the round, total runtime, average cpu usage and average memory usage
    parsed_data = {}


    # Iterate through directories and read execution_trace files
    for results_dir in root_dir.iterdir():
        results_dir_name = results_dir.name
        num_tasks = int(results_dir_name.split("_")[-1])
        results_dir_name = "_".join(results_dir_name.split("_")[:2])
        parsed_data.setdefault(results_dir_name, [])

        if results_dir.is_dir():
            # Add the directory name to times if it doesn't exist
            if results_dir_name not in times:
                times[results_dir_name] = []

            # Find and process each execution_trace file
            files = results_dir.glob("**/*execution_trace*")
            
            for file in files:
                print(f"Processing file: {file}")
                data = pd.read_csv(file, delimiter="\t").dropna()
                
                # Parse the duration column
                try:
                    data["duration"] = data["duration"].apply(
                        lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) / 1000 if 'ms' in x else float(re.search(r'\d+(\.\d+)?', x).group())
                    )
                except AttributeError:
                    print(f"Error parsing the duration column in file {file}")
                    continue
                # Sum duration and add to times dictionary
                time = data["duration"].sum()
                times[results_dir_name].append(float(time))


                # Get total runtime, average cpu usage and average memory usage for that round and store in parsed_data
                # Remove the % sign of the %cpu entries
                data["%cpu"] = data["%cpu"].str.replace("%", "")
                # Cast values to float
                data["%cpu"] = data["%cpu"].astype(float)
                average_cpu = data["%cpu"].mean()
                # Remove the MB of the peak_rss entries
                data["peak_rss"] = data["peak_rss"].str.replace("MB", "")
                # Cast values to float
                data["peak_rss"] = data["peak_rss"].astype(float)
                average_memory = data["peak_rss"].mean()
                
                round_data = {
                    "round": int(file.parent.stem),
                    "runtime": float(time),
                    "average_cpu": float(average_cpu),
                    "average_memory": float(average_memory),
                    "num_tasks": int(num_tasks)	
                }

                parsed_data[results_dir_name].append(round_data)



                

                
        
    # print(f"Times: {times}")

    # for key, value in times.items():
    #     # Return the average time for all the files
    #     average_time = sum(value) / len(value)        
        
    #     # Convert the time to minutes and seconds
    #     minutes = int(average_time // 60)
    #     seconds = round(average_time % 60)
    #     print(f"Average time for {key}: {minutes} minutes and {seconds} seconds")

    set_data_banditware(parsed_data)

if __name__ == "__main__":
    main()
# BanditWare

> An online recommendation system that dynamically selects the most suitable hardware for applications using a contextual multi-armed bandit algorithm.

## Setup
1. **Set Up a Virtual Environment:**
    - With Python's built-in venv:
      ```
      python3 -m venv env
      source env/bin/activate
      ```
2. **Install Dependencies:**
    - Install  required packages:
      ```
      pip3 install -r requirements.txt
      ```
3. **Run the Application:**
    - Once dependencies are installed, run:
      ```
      python main.py
      ```

## Running BanditWare:
1. Choose a dataset you would like to use. Choose a path to a folder or csv file that holds various runs of an application.
    - For a folder, label subfolders as follows: `{n}cores_{m}gb`, where `n` is the number of cpu cores and `m` is the number of gigabytes of RAM that the application was run on. Make sure all csv files in subfolders have a `runtime` column and any feature columns of the application.
    - For a csv file, make sure the columns contain `runtime` and any feature columns of the application.Encode the number of cpu cores and gigabytes of RAM in a `hardware` column with values like `"2_16"` to represent 2 cpu cores and 16 GB of RAM.
2. In the terminal, if you have a data folder, run `python3 pre_process.py --base_path={PATH_TO_DATA}` where `PATH_TO_DATA` is the path from step 1. If you have a single csv, run `python3 pre_process.py --data_file={PATH_TO_DATA_FILE}, where PATH_TO_DATA_FILE is the csv defined in part 1.
    - For example, `python3 pre_process.py --base_path=bp3d_data` or `python3 pre_process.py --base_path=bp3d_data.csv`
3. Finally, set the desired parameters for the contextual bandit in the `main()` function of `cb.py`, then run `python3 cb.py`.
4. To see results and preprocessed data, look in the `results` folder that was generated.

import pandas as pd
import numpy as np
import datetime
import json
from pathlib import Path

def flatten_sparse_json(json_list, nested_key="fields"):
    """
    Creates a Pandas DataFrame by flattening all top-level keys and
    all keys within a specified nested dictionary (the 'fields' key).
    Missing values are automatically set to NaN.
    """
    
    # Use pd.json_normalize on the entire list.
    # By default, it automatically flattens nested dictionaries and prefixes
    # the new columns with the name of the nested key, followed by a dot.
    # E.g., 'fields': {'r': 3} becomes the column 'fields.r'.
    df = pd.json_normalize(json_list)

    # 2.1. Clean up Column Names
    # We rename the flattened columns by removing the 'fields.' prefix.
    # This loop only affects columns that start with 'fields.'.
    df = df.rename(columns=lambda x: x.replace(f'{nested_key}.', '') 
                                     if x.startswith(f'{nested_key}.') else x)
    
    # 2.2. Drop the original nested column
    # The original 'fields' column itself contains the dict object and is now redundant.
    if nested_key in df.columns:
         df = df.drop(columns=[nested_key])

    return df 

def read_log_as_csv(path_to_logfile:str):
    with open(path_to_logfile) as f:
        lines = f.readlines()
        log_in_json = []
        for line in lines:
            json_obj = json.loads(line)
            log_in_json.append(json_obj)
    return flatten_sparse_json(log_in_json, nested_key="fields")


def get_workerlog_landlord_paths(path_to_log):
    # "." searches starting from your current folder
    workerlog_landlord_paths = []
    for path in Path(path_to_log).rglob("worker1.log"):
        a = str(path)
        if "landlord" in a.lower() and "precleanup" not in a.lower():
            workerlog_landlord_paths.append(a)
    return workerlog_landlord_paths
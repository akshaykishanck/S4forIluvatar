import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """
    Loads YAML configuration file and resolves paths based on output_path if subpaths are not specified.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    paths = config.get("paths", {})
    output_path = paths.get("output_path", "src/data/processed/results")

    # Derive subpaths based on output_path if not explicitly overridden
    paths.setdefault("plots_path", os.path.join(output_path, "reports", "plots"))
    paths.setdefault("models_path", os.path.join(output_path, "models"))
    paths.setdefault("reports_path", os.path.join(output_path, "reports"))
    paths.setdefault("error_data_path", os.path.join(output_path, "error_data"))
    paths.setdefault("logs_path", os.path.join(output_path, "logs"))

    config["paths"] = paths
    return config


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

def get_workerlog_all_paths(path_to_log):
    # "." searches starting from your current folder
    workerlog_all_paths = []
    for path in Path(path_to_log).rglob("worker1.log"):
        a = str(path)
        if "precleanup" not in a.lower() and "alwayscpu" not in a.lower():
            workerlog_all_paths.append(a)
    return workerlog_all_paths


def extract_policy_from_path(log_path_str):
    """
    Extract dispatch policy from file path string.
    """
    path_lower = str(log_path_str).lower()
    known_policies = ['landlord', 'alwaysgpu', 'random', 'roundrobin', 'leastloaded', 'greedy', 'warmonly', 'coldonly', 'mice', 'new_mice', 'speedup', 'weightedrandom']
    for p in known_policies:
        if p in path_lower:
            return p
    # Fallback to parent folder name
    parts = Path(log_path_str).parts
    if len(parts) >= 2:
        return parts[-2]
    return 'default'


def discover_and_load_logs(log_dir, policies=None, max_logs=10):
    """
    Stage 1: Recursively locate and load log files into a single raw DataFrame.
    Limits to max_logs (default: 10).
    """
    logger.info("==========================================")
    logger.info("STAGE 1: DISCOVERING & LOADING LOGS")
    logger.info("Search directory: %s", log_dir)
    logger.info("Max logs limit: %s", max_logs)
    logger.info("==========================================")

    log_paths = get_workerlog_all_paths(log_dir)

    if not log_paths:
        raise FileNotFoundError(f"No valid non-CPU log files found in directory: {log_dir}")

    total_discovered = len(log_paths)
    if max_logs is not None and max_logs > 0 and total_discovered > max_logs:
        logger.info("Found %d GPU log files. Limiting processing to first %d logs.", total_discovered, max_logs)
        log_paths = log_paths[:max_logs]
    else:
        logger.info("Found %d log files.", len(log_paths))

    dfs = []

    for idx, path_str in enumerate(log_paths):
        policy = extract_policy_from_path(path_str)
        
        if policies is not None:
            if policy not in policies:
                continue
                
        session_id = f"session_{idx+1}_{Path(path_str).parent.name}"
        
        try:
            df_log = read_log_as_csv(path_str)
            df_log['session'] = session_id
            df_log['dispatch_policy'] = policy
            df_log['log_path'] = path_str
            dfs.append(df_log)
        except Exception as e:
            logger.warning("Failed to read log file %s: %s", path_str, e)

    if not dfs:
        raise ValueError("Failed to parse any valid log files.")
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d total raw log records across %d sessions.", len(raw_df), len(dfs))
    return raw_df
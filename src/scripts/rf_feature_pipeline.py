import pandas as pd
import numpy as np
import src.scripts.s4_feature_pipeline as s4fp

def get_lagged_target_queue_len(df, fqdn_series, lag):
    """
    Solves the moving-target issue.
    Returns the queue length of the specific `fqdn` (from fqdn_series), 
    but evaluated at `lag` rows *prior* in the dataset.
    """
    # 1. Identify all pure queue length columns 
    q_len_cols = [c for c in df.columns if str(c).endswith('_len') and c not in ['target_queue_len', 'others_len_queue']]
    
    # 2. Shift the absolute state of all queues backwards by `lag` requests
    shifted_df = df[q_len_cols].shift(lag).fillna(0)
    
    # 3. Vectorized coordinate mapping: 
    # For each row, find the column index in `shifted_df` that corresponds to that row's target FQDN.
    col_map = {col: i for i, col in enumerate(shifted_df.columns)}
    col_indices = fqdn_series.map(lambda q: col_map.get(f"{q}_len", -1)).fillna(-1).astype(int).values
    
    shifted_values = shifted_df.values
    row_indices = np.arange(len(shifted_values))
    
    # 4. Extract the explicit values bridging time and moving targets
    valid_mask = col_indices != -1
    result = np.zeros(len(shifted_values))
    result[valid_mask] = shifted_values[row_indices[valid_mask], col_indices[valid_mask]]
    
    return result

def add_lagged_features(df, lags=[1, 3, 5]):
    """
    Takes the base final_features from s4_feature_pipeline and injects lagged memory.
    """
    # Ensure sequential time order to calculate lag correctly
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    elif 'invocation_timestamp' in df.columns:
        df = df.sort_values('invocation_timestamp').reset_index(drop=True)
        
    q_len_cols = [c for c in df.columns if str(c).endswith('_len') and c not in ['target_queue_len', 'others_len_queue']]
    total_q_lens = df[q_len_cols].sum(axis=1)
    
    new_cols = {}
    
    print(f"Generating rolling lag features across lags {lags}...")
    for lag in lags:
        # 1. Target Queue Length
        lagged_target = get_lagged_target_queue_len(df, df['fqdn'], lag)
        new_cols[f'target_queue_len_lag_{lag}'] = lagged_target
        
        # 2. Others Queue Length
        # (Total queue length at t-lag) - (Target queue length at t-lag)
        shifted_total = total_q_lens.shift(lag).fillna(0)
        lagged_others = shifted_total - lagged_target
        new_cols[f'others_len_queue_lag_{lag}'] = lagged_others
        
        # 3. Global properties
        if 'num_running_funcs_filled' in df.columns:
            new_cols[f'num_running_funcs_filled_lag_{lag}'] = df['num_running_funcs_filled'].shift(lag).fillna(0)
            
    # Concat all new features cleanly
    lag_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, lag_df], axis=1)

def generate_rf_features(raw_df, lags=[1, 3, 5]):
    """
    Main pipeline entrypoint for Random Forest specific processing.
    Delegates to s4_feature_pipeline for base extraction, then adds temporal lags.
    """
    print("Delegating baseline feature extraction to S4 Pipeline...")
    base_features = s4fp.generate_target_features(raw_df)
    
    print("Enhancing features with local temporal RF lags...")
    rf_features = add_lagged_features(base_features, lags=lags)
    
    print("RF Feature extraction complete!")
    return rf_features

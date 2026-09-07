from IPython.core import application
import logging
from joblib import compressor
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)

def get_time_of_invocation(df):
    if 'timestamp' in df.columns:
        df = df.sort_values(by='timestamp').reset_index(drop=True)
    if 'message' in df.columns:
        df = df[df['message']=='Handling invocation request'][['timestamp', 'tid']].rename({"timestamp":"invocation_timestamp"}, axis=1).reset_index(drop=True)
    return df

def calculate_iat(df):
    if 'timestamp' in df.columns and 'message' in df.columns:
        invocations = get_time_of_invocation(df)
    else:
        invocations = df.copy()

    invocations['invocation_timestamp'] = pd.to_datetime(invocations['invocation_timestamp'])
    invocations['last_timestamp'] = invocations['invocation_timestamp'].shift(1)
    invocations['iat'] = (invocations['invocation_timestamp'] - invocations['last_timestamp']).dt.total_seconds()
    invocations.loc[invocations['iat'].isna(), 'iat'] = 0
    invocations = invocations.drop(columns=['last_timestamp']).sort_values(by='invocation_timestamp').reset_index(drop=True)
    return invocations

def get_e2e_data(df):
    gpu_tids = df[(df['fqdn'].notna()) & (df['e2etime'].notna()) & (df['compute']=='GPU')]['tid'].unique()
    filtered_data = df[df['tid'].isin(gpu_tids)].reset_index(drop=True)
    e2e_data = filtered_data[filtered_data['e2etime'].notna()][['tid', 'fqdn', 'e2etime']].reset_index(drop=True)
    return e2e_data

def calculate_fqdn_iat(df):
    time_col = 'invocation_timestamp' if 'invocation_timestamp' in df.columns else 'timestamp'
    last_time = df.sort_values(by=['fqdn', time_col]).reset_index(drop=True)
    last_time['last_invocation_timestamp'] = last_time.groupby(['fqdn'])[time_col].shift(1)
    last_time['iat_fqdn'] = (pd.to_datetime(last_time[time_col]) - pd.to_datetime(last_time['last_invocation_timestamp'])).dt.total_seconds()
    last_time.loc[last_time['iat_fqdn'].isna(), 'iat_fqdn'] = 0
    last_time = last_time.drop(columns=['last_invocation_timestamp'])
    return last_time[['tid', 'iat_fqdn']].drop_duplicates()

def fill_realtime_running_funcs(df):
    num_req_cols = ['timestamp', 'message', 'fqdn', 'tid', 'num_running_funcs', 'e2etime']
    if 'remove_time' in df.columns:
        num_req_cols.append('remove_time')
    num_req_data = df[
        (df['num_running_funcs'].notna()) |
        (df['e2etime'].notna()) |
        (df['message'] == 'Item starting to execute') |
        (df['message'] == 'Handling invocation request') |
        (df['message'] == 'Invocation complete')          # needed so delta=-1 events reach fill_realtime_running_funcs
    ][[c for c in num_req_cols if c in df.columns]]

    df = num_req_data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    starts = df[df['message'] == 'Item starting to execute'].copy()
    if 'remove_time' in starts.columns:
        starts['remove_time'] = pd.to_datetime(starts['remove_time'])
    else:
        starts['remove_time'] = starts['timestamp']
    starts = starts[['tid', 'remove_time']].dropna(subset=['tid', 'remove_time']).drop_duplicates(subset=['tid'])
    
    # Extract end times from 'Invocation complete'
    ends = df[df['message'] == 'Invocation complete'][['tid', 'timestamp']].copy()
    ends = ends.rename(columns={'timestamp': 'complete_time'}).dropna(subset=['tid']).drop_duplicates(subset=['tid'])
    
    # Build intervals
    intervals = pd.merge(starts, ends, on='tid', how='inner')
    
    # Get invocation timestamps
    invocations = df[df['message'] == 'Handling invocation request'][['tid', 'timestamp']].copy()
    
    if len(intervals) == 0 or len(invocations) == 0:
        invocations['num_running_funcs_filled'] = 0
        return invocations[['tid', 'num_running_funcs_filled']].reset_index(drop=True)
        
    # Vectorized count of active intervals per invocation request
    remove_arr = intervals['remove_time'].values[:, None]
    ends_arr = intervals['complete_time'].values[:, None]
    ts_arr = invocations['timestamp'].values[None, :]
    
    active_mask = (remove_arr <= ts_arr) & (ends_arr >= ts_arr)
    counts = active_mask.sum(axis=0)
    
    invocations['num_running_funcs_filled'] = counts
    return invocations[['tid', 'num_running_funcs_filled']].reset_index(drop=True)

def get_queue_features_at_invocations(df, features_df):
    """
    Extracts queue features at invocation requests.
    Computes target_queue_len, target_queue_status (one-hot encoded), and others_len_queue,
    and drops individual FQDN-level queue length and status columns.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    tid_to_fqdn = df[['tid', 'fqdn']].dropna().drop_duplicates().set_index('tid')['fqdn'].to_dict()
    unique_queues = list(set([q for q in tid_to_fqdn.values() if isinstance(q, str)]))

    events = []
    for row in df.itertuples():
        if row.message == 'Switching state':
            events.append({
                'timestamp': row.timestamp, 
                'queue': getattr(row, 'queue', None) or getattr(row, 'fqdn', None),
                'type': 'set', 
                'val': getattr(row, 'queue_len', 0),
                'new_state': getattr(row, 'new_state', 'Active')
            })
        elif row.message == 'Item starting to execute':
            q = tid_to_fqdn.get(row.tid)
            if q and isinstance(q, str):
                events.append({
                    'timestamp': row.timestamp, 
                    'queue': q, 
                    'type': 'dec', 
                    'val': 1
                })
                
    q_len_state = {f"{q}_len": 0 for q in unique_queues}
    q_status_state = {f"{q}_status": "Inactive" for q in unique_queues}
    records = []
    
    for e in events:
        q = e['queue']
        if pd.isna(q):
            continue
            
        if q not in unique_queues:
            unique_queues.append(q)
            q_len_state[f"{q}_len"] = 0
            q_status_state[f"{q}_status"] = "Inactive"
            
        len_key = f"{q}_len"
        status_key = f"{q}_status"
        
        if e['type'] == 'set':
            q_len_state[len_key] = e['val']
            if 'new_state' in e and pd.notna(e['new_state']):
                q_status_state[status_key] = e['new_state']
        elif e['type'] == 'dec':
            q_len_state[len_key] = max(0, q_len_state[len_key] - 1)
        
        rec = {'timestamp': e['timestamp']}
        rec.update(q_len_state)
        rec.update(q_status_state)
        records.append(rec)
        
    state_history_df = pd.DataFrame(records)
    if len(state_history_df) == 0:
        return pd.DataFrame() 
        
    state_history_df = state_history_df.drop_duplicates(subset=['timestamp'], keep='last')
    state_history_df = state_history_df.set_index('timestamp')

    df_joined = df.join(state_history_df, on='timestamp')
    q_cols = [c for c in state_history_df.columns if c != 'timestamp']
    df_joined[q_cols] = df_joined[q_cols].ffill()
    
    # Fill NAs: numeric lengths with 0, Status with 'Inactive'
    len_cols = [c for c in q_cols if c.endswith('_len')]
    status_cols = [c for c in q_cols if c.endswith('_status')]
    df_joined[len_cols] = df_joined[len_cols].fillna(0)
    df_joined[status_cols] = df_joined[status_cols].fillna('Inactive')

    invocations = df_joined[df_joined['message'] == 'Handling invocation request'].copy()
    invocations = invocations[['timestamp', 'tid'] + q_cols].reset_index(drop=True)
    features_df = features_df.merge(invocations, on='tid')

    # Extract target FQDN's specific queue status and length dynamically
    def extract_invoked_target_status(row):
        target_q = row['fqdn']
        if pd.isna(target_q):
            return pd.Series({'target_queue_len': 0, 'target_queue_status': 'Inactive'})
            
        len_col = f"{target_q}_len"
        status_col = f"{target_q}_status"
        
        return pd.Series({
            'target_queue_len': row.get(len_col, 0),
            'target_queue_status': row.get(status_col, 'Inactive')
        })

    specific_queue_vars = features_df.apply(extract_invoked_target_status, axis=1)
    features_df = pd.concat([features_df, specific_queue_vars], axis=1)

    # Vectorized computation of others_len_queue
    all_q_len_cols = [c for c in features_df.columns if str(c).endswith('_len') and c != 'target_queue_len']
    features_df['others_len_queue'] = features_df[all_q_len_cols].sum(axis=1) - features_df['target_queue_len']

    # One-Hot Encode target_queue_status
    dummies = pd.get_dummies(features_df['target_queue_status'], prefix='is_status').astype(int)
    for state in ['Active', 'Inactive', 'Throttled']:
        if f'is_status_{state}' not in dummies.columns:
            dummies[f'is_status_{state}'] = 0
    features_df = pd.concat([features_df, dummies], axis=1)

    # Filter out FQDN-level queue length and status columns, keeping target and other summary columns
    keep_cols = ['timestamp', 'tid', 'target_queue_len', 'target_queue_status',   'others_len_queue', 'is_status_Active', 'is_status_Inactive', 'is_status_Throttled']
    existing_cols = [c for c in keep_cols if c in features_df.columns]
    return features_df[existing_cols].reset_index(drop=True)

def add_benchmark_features(df, json_path='worker_function_benchmarks.json'):
    """
    Reads the benchmark JSON and maps warm and cold execution times to the fqdn via base_function.
    """
    import os
    benchmarks = None
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                benchmarks = json.load(f)
        except Exception:
            raise
            
    if benchmarks is None:
        df['gpu_warm_results_sec'] = 0.0
        df['gpu_cold_results_sec'] = 0.0
        return df
        
    data = benchmarks.get('data', {})
    benchmark_features = []
    for base_func, info in data.items():
        try:
            gpu_data = info.get('resource_data', {}).get('GPU', {})
            warm_mean = np.mean(gpu_data.get('warm_results_sec', [0]))
            cold_mean = np.mean(gpu_data.get('cold_results_sec', [0]))
        except Exception:
            warm_mean, cold_mean = 0.0, 0.0
            
        benchmark_features.append({
            'base_function': base_func,
            'gpu_warm_results_sec': float(warm_mean),
            'gpu_cold_results_sec': float(cold_mean)
        })
        
    bench_df = pd.DataFrame(benchmark_features)
    df['base_function'] = df['fqdn'].str.extract(r'^([^0-9]+)')[0].str.strip('-')

    # Merge without dropping anything
    df = df.merge(bench_df, on='base_function', how='left')
    # Fill NAs for base functions that weren't in the JSON
    df['gpu_warm_results_sec'] = df['gpu_warm_results_sec'].fillna(0.0)
    df['gpu_cold_results_sec'] = df['gpu_cold_results_sec'].fillna(0.0)
    
    return df

def get_cold_gpu_tids(df):
    # Identify cold starts using the is_warm_gpu flag from 'Landlord Credit' log entries.
    # is_cold_start = 1 when is_warm_gpu is False (i.e. the GPU container was not warm).
    if 'message' in df.columns and 'is_warm_gpu' in df.columns:
        landlord_credit = df[df['message'] == 'Landlord Credit'][['tid', 'is_warm_gpu']].dropna(subset=['tid'])
        landlord_credit = landlord_credit.drop_duplicates(subset=['tid'], keep='first').copy()
        cold_tids = landlord_credit[landlord_credit['is_warm_gpu']==False]['tid'].unique()
        logger.info(f"Number of cold tids {len(cold_tids)}")
    elif 'message' in df.columns:
        cold_tids = df[df['message'] == 'Container cold start completed']['tid'].dropna().unique()
    else:
        cold_tids = []
    return cold_tids

def execution_start_status(df, cold_tids):
    df['is_cold_start'] = df['tid'].isin(cold_tids).astype(int)
    return df
    
def generate_target_features(raw_df, feature_config=None):
    """
    Main pipeline function that extracts features from the raw logs.

    Args:
        raw_df: Raw log DataFrame.
        feature_config: Dict with keys 'continuous' and 'categorical', each mapping
                        feature names to booleans. If None, all features are extracted.
    """
    df = raw_df.copy()
    df['e2etime'] = df['e2etime'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.loc[df['tid'].isna(), 'tid'] = df.get('span.tid', pd.NA)

    # Build flat toggle map: feature_name -> enabled (bool)
    cont = feature_config.get('continuous', {}) if feature_config else {}
    cat = feature_config.get('categorical', {}) if feature_config else {}
    toggles = {**cont, **cat}

    def is_on(*names):
        """Return True if any of the given feature names is explicitly enabled or feature_config is None."""
        if not feature_config:
            return True
        return any(toggles.get(n, False) for n in names)

    # Always extract IAT (used as base; drop if disabled at the end)
    features = calculate_iat(df)

    # Add e2etime targets (always required)
    e2e_data = get_e2e_data(df)
    features = features.merge(e2e_data, on=['tid'])

    # IAT per FQDN
    if is_on('iat_fqdn'):
        iat_fqdn_features = calculate_fqdn_iat(features)
        features = features.merge(iat_fqdn_features, on=['tid'])

    # Realtime running functions (contention)
    if is_on('num_running_funcs_filled'):
        running_funcs = fill_realtime_running_funcs(df)
        features = features.merge(running_funcs, on=['tid'])

    # Queue features (length, status, one-hot encoded status)
    queue_feats_needed = is_on('target_queue_len', 'others_len_queue',
                               'is_status_Active', 'is_status_Inactive', 'is_status_Throttled')
    if queue_feats_needed:
        logger.info("Extracting detailed queue state history... this may take a moment.")
        queue_features = get_queue_features_at_invocations(df, features)
        features = features.merge(queue_features, on=['tid'])

    # Benchmark features (warm/cold GPU exec times)
    if is_on('gpu_warm_results_sec', 'gpu_cold_results_sec'):
        features = add_benchmark_features(features, "src/data/raw/worker_function_benchmarks.json")

    # Cold start flag
    if is_on('is_cold_start'):
        cold_tids = get_cold_gpu_tids(df)
        features = execution_start_status(features, cold_tids)

    # --- Filter columns to only enabled features + identifier columns ---
    id_cols = ['tid', 'invocation_timestamp', 'e2etime', 'fqdn', 'session', 'dispatch_policy']
    if feature_config:
        all_toggles = {**cont, **cat}
        enabled_features = [f for f, enabled in all_toggles.items() if enabled]
        keep_cols = id_cols + [f for f in enabled_features if f in features.columns]
        # Deduplicate while preserving order
        seen = set()
        keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
        features = features[[c for c in keep_cols if c in features.columns]]
        logger.info("Feature toggles applied. Active features: %s", enabled_features)
    else:
        logger.info("No feature config provided — all features retained.")

    return features


def generate_rf_features(raw_df, feature_config=None):
    """
    Main pipeline entrypoint for Random Forest specific processing.
    """
    logger.info("Baseline feature extraction.")
    features = generate_target_features(raw_df, feature_config=feature_config)
    logger.info("RF Feature extraction complete!")
    return features
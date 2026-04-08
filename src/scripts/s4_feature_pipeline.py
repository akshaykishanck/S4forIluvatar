import pandas as pd
import numpy as np
import json

def calculate_iat(df):
    df['invocation_timestamp'] = pd.to_datetime(df['invocation_timestamp'])
    df['last_timestamp'] = df['invocation_timestamp'].shift(1)
    df['iat'] = (df['invocation_timestamp'] - df['last_timestamp']).dt.total_seconds()
    df.loc[df['iat'].isna(), 'iat'] = 0
    df = df.drop(columns=['last_timestamp'])
    return df

def calculate_fqdn_iat(df):
    last_time = df.sort_values(by=['fqdn', 'invocation_timestamp'])
    last_time['last_invocation_timestamp'] = last_time.groupby(['fqdn'])['invocation_timestamp'].shift(1)
    last_time['iat_fqdn'] = (pd.to_datetime(last_time['invocation_timestamp']) - pd.to_datetime(last_time['last_invocation_timestamp'])).dt.total_seconds()
    last_time.loc[last_time['iat_fqdn'].isna(), 'iat_fqdn'] = 0
    last_time = last_time.drop(columns=['last_invocation_timestamp'])
    return last_time[['tid', 'iat_fqdn']].drop_duplicates()

def fill_realtime_running_funcs(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['delta'] = 0
    df.loc[df['message'] == 'Handling invocation request', 'delta'] = 1
    df.loc[df['message'] == 'Invocation complete', 'delta'] = -1
    
    df['cumsum_delta'] = df['delta'].cumsum()
    df['baseline'] = df['num_running_funcs'] - df['cumsum_delta']
    df['baseline'] = df['baseline'].ffill().bfill()
    df['num_running_funcs_filled'] = df['cumsum_delta'] + df['baseline']
    
    df = df[df['message']=='Handling invocation request'][['tid', 'num_running_funcs_filled']].reset_index(drop=True)
    df['num_running_funcs_filled'] = df['num_running_funcs_filled'] - 1
    return df

def get_queue_features_at_invocations(df):
    """
    Extracts the detailed queue lengths for ALL fqdns and the 
    explicit queue states (Active, Inactive, Throttled).
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

    invocations = df_joined[df_joined['message'] == 'Handling invocation request']
    return invocations[['timestamp', 'tid'] + q_cols].reset_index(drop=True)

def add_benchmark_features(df, json_path='worker_function_benchmarks.json'):
    """
    Reads the benchmark JSON and maps warm and cold execution times to the fqdn via base_function.
    """
    try:
        with open(json_path, 'r') as f:
            benchmarks = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Giving benchmark features 0.0.")
        df['gpu_warm_results_sec'] = 0.0
        df['gpu_cold_results_sec'] = 0.0
        return df
        
    data = benchmarks.get('data', {})
    
    benchmark_features = []
    for base_func, info in data.items():
        try:
            gpu_data = info.get('resource_data', {}).get('gpu', {})
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
    
    if 'base_function' not in df.columns:
        df['base_function'] = df['fqdn'].str.extract(r'^([^0-9]+)')[0].str.strip('-')
        
    # Merge without dropping anything
    df = df.merge(bench_df, on='base_function', how='left')
    
    # Fill NAs for base functions that weren't in the JSON
    df['gpu_warm_results_sec'] = df['gpu_warm_results_sec'].fillna(0.0)
    df['gpu_cold_results_sec'] = df['gpu_cold_results_sec'].fillna(0.0)
    
    return df

def generate_target_features(raw_df):
    """
    Main pipeline function that extracts ALL S4 features from the raw logs.
    """
    df = raw_df.copy()
    df['e2etime'] = df['e2etime'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.loc[df['tid'].isna(), 'tid'] = df.get('span.tid', pd.NA)
    
    # Identify explicit cold starts logged by containermanager
    if 'message' in df.columns:
        cold_tids = df[df['message'] == 'Container cold start completed']['tid'].dropna().unique()
    else:
        cold_tids = []
    
    # Isolate GPU metrics
    gpu_tids = df[(df['fqdn'].notna()) & (df['e2etime'].notna()) & (df['compute']=='GPU')]['tid'].unique()
    filtered_data = df[df['tid'].isin(gpu_tids)].reset_index(drop=True)
    filtered_data = filtered_data.sort_values(by='timestamp').reset_index(drop=True)

    # 1. Base Invocation Data
    invocation_data = filtered_data[filtered_data['message']=='Handling invocation request'][['timestamp', 'tid']].rename({"timestamp":"invocation_timestamp"}, axis=1).reset_index(drop=True)
    
    # 2. Extract IAT
    base_features = calculate_iat(invocation_data)
    
    # Merge FQDN and e2etime targets
    e2e_data = filtered_data[filtered_data['e2etime'].notna()][['tid', 'fqdn', 'e2etime']].reset_index(drop=True)
    base_features = base_features.merge(e2e_data, on=['tid'], how='left')
    
    # 3. Extract IAT-FQDN
    iat_fqdn_feats = calculate_fqdn_iat(base_features)
    final_features = base_features.merge(iat_fqdn_feats, on=['tid'])
    
    # 4. Extract Realtime Running Funcs (Contention)
    num_req_data = df[((df['num_running_funcs'].notna()) | (df['e2etime'].notna())) | ((df['message']=='Item starting to execute') | (df['message']=='Handling invocation request'))][['timestamp', 'message', 'fqdn', 'tid','num_running_funcs','e2etime']]
    running_funcs = fill_realtime_running_funcs(num_req_data)
    final_features = final_features.merge(running_funcs, on=['tid'])

    # 5. Extract Queue Features (Length and Hardware Statuses)
    print("Extracting detailed queue state history... this may take a moment.")
    queue_features = get_queue_features_at_invocations(raw_df)
    final_features = final_features.merge(queue_features, on=['tid'])
    
    # 6. Extract target FQDN's specific queue status dynamically into single columns
    # Because there are 50 queues, we need to know exactly which queue we are entering!
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

    specific_queue_vars = final_features.apply(extract_invoked_target_status, axis=1)
    final_features = pd.concat([final_features, specific_queue_vars], axis=1)
    
    # Vectorized computation of others_len_queue
    all_q_len_cols = [c for c in final_features.columns if str(c).endswith('_len') and c != 'target_queue_len']
    final_features['others_len_queue'] = final_features[all_q_len_cols].sum(axis=1) - final_features['target_queue_len']
    
    # One-Hot Encode the specific target queue status for immediate S4 compatibility
    dummies = pd.get_dummies(final_features['target_queue_status'], prefix='is_status').astype(int)
    # Ensure all three exist in case the sample didn't capture them
    for state in ['Active', 'Inactive', 'Throttled']:
        if f'is_status_{state}' not in dummies.columns:
            dummies[f'is_status_{state}'] = 0
            
    final_features = pd.concat([final_features, dummies], axis=1)
    final_features['base_function'] = final_features['fqdn'].str.extract(r'^([^0-9]+)')[0].str.strip('-')
    
    # 7. Add internal benchmark representations explicitly requested (warm and cold times)
    final_features = add_benchmark_features(final_features, 'worker_function_benchmarks.json')
    
    final_features['is_cold_start'] = final_features['tid'].isin(cold_tids).astype(int)
    
    return final_features

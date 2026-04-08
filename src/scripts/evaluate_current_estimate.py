import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.scripts.utils import read_log_as_csv

def evaluate_current_estimate(path_to_log):
    df = read_log_as_csv(path_to_log)

    # 1. Securely identify purely GPU-bound TIDs
    gpu_tids = df[(df['e2etime'].notna()) & (df['compute'] == 'GPU')]['tid'].unique()

    # 2. Extract estimators safely formatting as numerics
    df['gpu_est_total'] = pd.to_numeric(df['gpu_est_total'], errors='coerce')
    df['e2etime'] = pd.to_numeric(df['e2etime'], errors='coerce')
    df['fqdn'] = df['fqdn'].astype(str)

    # 3. Flatten fragmented event logs down to the maximum resolved values
    unified = df.groupby('tid')[['gpu_est_total', 'e2etime', 'fqdn']].max().dropna(subset=['gpu_est_total', 'e2etime', 'fqdn'])
    unified = unified[unified['fqdn'] != 'nan']

    # 4. Strict Intersection Filter
    gpu_only = unified[unified.index.isin(gpu_tids)].copy()

    print(f'Total Unified TIDs found with estimator pairs: {len(unified)}')
    print(f'Strictly GPU-Bound TIDs: {len(gpu_only)}')

    # 5. Row-by-Row Error Calculations
    gpu_only['absolute_error'] = (gpu_only['e2etime'] - gpu_only['gpu_est_total']).abs()
    gpu_only['squared_error'] = (gpu_only['e2etime'] - gpu_only['gpu_est_total']) ** 2
    
    # Percentage relative error: ((actual - pred) / actual) * 100
    gpu_only['percentage_relative_error'] = ((gpu_only['e2etime'] - gpu_only['gpu_est_total']) / gpu_only['e2etime']) * 100

    # 6. Compute Global Benchmarks
    mse = gpu_only['squared_error'].mean()
    mae = gpu_only['absolute_error'].mean()
    mpre = gpu_only['percentage_relative_error'].mean()
    std_err = (gpu_only['gpu_est_total'] - gpu_only['e2etime']).std()

    print(f'TRUE Baseline MSE: {mse:.4f}')
    print(f'TRUE Baseline MAE: {mae:.4f}')
    print(f'TRUE Baseline Mean Perc. Relative Error: {mpre:.4f}%')
    print(f'TRUE Baseline Error StdDev: {std_err:.4f}')
    
    # 7. Format Return DataFrame
    return_df = gpu_only.reset_index()[['fqdn', 'tid', 'e2etime', 'gpu_est_total', 'absolute_error', 'squared_error', 'percentage_relative_error']]
    return return_df

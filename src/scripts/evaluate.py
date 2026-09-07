import os
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.scripts.utils import read_log_as_csv

logger = logging.getLogger(__name__)


def evaluate_current_estimate(path_or_df):
    """
    Evaluates current estimation model (gpu_est_total) on log file or raw dataframe.
    """
    if isinstance(path_or_df, str):
        df = read_log_as_csv(path_or_df)
    else:
        df = path_or_df.copy()

    # 1. Securely identify purely GPU-bound TIDs
    if 'compute' in df.columns:
        gpu_tids = df[(df['e2etime'].notna()) & (df['compute'] == 'GPU')]['tid'].unique()
    else:
        gpu_tids = df[(df['e2etime'].notna())]['tid'].unique()

    # 2. Extract estimators safely formatting as numerics
    df['gpu_est_total'] = pd.to_numeric(df['gpu_est_total'], errors='coerce')
    df['e2etime'] = pd.to_numeric(df['e2etime'], errors='coerce')
    df['fqdn'] = df['fqdn'].astype(str)

    # 3. Flatten fragmented event logs down to the maximum resolved values
    unified = df.groupby('tid')[['gpu_est_total', 'e2etime', 'fqdn']].max().dropna(subset=['gpu_est_total', 'e2etime', 'fqdn'])
    unified = unified[unified['fqdn'] != 'nan']

    # 4. Strict Intersection Filter
    gpu_only = unified[unified.index.isin(gpu_tids)].copy()

    logger.info(f'Total Unified TIDs found with estimator pairs: {len(unified)}')
    logger.info(f'Strictly GPU-Bound TIDs: {len(gpu_only)}')

    # 5. Row-by-Row Error Calculations
    gpu_only['absolute_error'] = (gpu_only['e2etime'] - gpu_only['gpu_est_total']).abs()
    gpu_only['squared_error'] = (gpu_only['e2etime'] - gpu_only['gpu_est_total']) ** 2
    
    # Percentage relative error: ((actual - pred) / actual) * 100
    gpu_only['percentage_relative_error'] = ((gpu_only['gpu_est_total'] - gpu_only['e2etime']) / gpu_only['e2etime']) * 100

    # 6. Compute Global Benchmarks
    if len(gpu_only) > 0:
        mse = gpu_only['squared_error'].mean()
        mae = gpu_only['absolute_error'].mean()
        mpre = gpu_only['percentage_relative_error'].mean()
        std_err = (gpu_only['gpu_est_total'] - gpu_only['e2etime']).std()

        logger.info(f'TRUE Baseline MSE: {mse:.4f}')
        logger.info(f'TRUE Baseline MAE: {mae:.4f}')
        logger.info(f'TRUE Baseline Mean Perc. Relative Error: {mpre:.4f}%')
        logger.info(f'TRUE Baseline Error StdDev: {std_err:.4f}')

    # 7. Format Return DataFrame
    return_df = gpu_only.reset_index()[['fqdn', 'tid', 'e2etime', 'gpu_est_total', 'absolute_error', 'squared_error', 'percentage_relative_error']]
    return return_df


def compare_with_current_estimate(test_df, raw_df, reports_dir):
    """
    Stage 7: Performance comparison between RF model and current estimate model (gpu_est_total).
    """
    logger.info("==========================================")
    logger.info("STAGE 7: MODEL COMPARISON (RF vs CURRENT ESTIMATE)")
    logger.info("==========================================")

    curr_df = evaluate_current_estimate(raw_df)
    
    merged = test_df.merge(
        curr_df[['tid', 'gpu_est_total', 'absolute_error', 'squared_error', 'percentage_relative_error']],
        on='tid', how='left', suffixes=('', '_curr')
    )
    
    if 'gpu_est_total' in merged.columns:
        merged['gpu_est_total'] = pd.to_numeric(merged['gpu_est_total'], errors='coerce')

    # Filter test set to instances with valid current estimates for a fair side-by-side comparison
    comp_df = merged.dropna(subset=['gpu_est_total', 'e2etime', 'rf_prediction']).copy()
    
    if len(comp_df) == 0:
        logger.warning("Warning: No matching test records found with both gpu_est_total and rf_prediction. Using test_df directly.")
        comp_df = test_df.dropna(subset=['e2etime', 'rf_prediction']).copy()
        if 'gpu_est_total' not in comp_df.columns:
            comp_df['gpu_est_total'] = comp_df['e2etime']

    # Error computations for Current Estimate
    comp_df['curr_abs_error'] = (comp_df['e2etime'] - comp_df['gpu_est_total']).abs()
    comp_df['curr_sq_error'] = (comp_df['e2etime'] - comp_df['gpu_est_total']) ** 2
    comp_df['curr_ape'] = (comp_df['curr_abs_error'] / comp_df['e2etime']) * 100

    # Metrics Overall
    overall_metrics = {
        'Model': ['Current Estimate (gpu_est_total)', 'Random Forest Model'],
        'MAE': [comp_df['curr_abs_error'].mean(), comp_df['rf_absolute_error'].mean()],
        'MSE': [comp_df['curr_sq_error'].mean(), comp_df['rf_squared_error'].mean()],
        'meanAPE (%)': [comp_df['curr_ape'].mean(), comp_df['rf_ape'].mean()],
        'medianAPE (%)': [comp_df['curr_ape'].median(), comp_df['rf_ape'].median()]
    }
    overall_summary_df = pd.DataFrame(overall_metrics)

    # Per FQDN comparison
    fqdn_comparison_rows = []
    if 'fqdn' in comp_df.columns:
        for fqdn, group in comp_df.groupby('fqdn'):
            if len(group) == 0:
                continue
            fqdn_comparison_rows.append({
                'fqdn': fqdn,
                'sample_count': len(group),
                'curr_MAE': group['curr_abs_error'].mean(),
                'rf_MAE': group['rf_absolute_error'].mean(),
                'curr_MSE': group['curr_sq_error'].mean(),
                'rf_MSE': group['rf_squared_error'].mean(),
                'curr_meanAPE': group['curr_ape'].mean(),
                'rf_meanAPE': group['rf_ape'].mean(),
                'curr_medianAPE': group['curr_ape'].median(),
                'rf_medianAPE': group['rf_ape'].median(),
            })

    fqdn_summary_df = pd.DataFrame(fqdn_comparison_rows)

    # Per Policy comparison
    policy_comparison_rows = []
    if 'dispatch_policy' in comp_df.columns:
        for policy, group in comp_df.groupby('dispatch_policy'):
            if len(group) == 0:
                continue
            policy_comparison_rows.append({
                'dispatch_policy': policy,
                'sample_count': len(group),
                'curr_MAE': group['curr_abs_error'].mean(),
                'rf_MAE': group['rf_absolute_error'].mean(),
                'curr_MSE': group['curr_sq_error'].mean(),
                'rf_MSE': group['rf_squared_error'].mean(),
                'curr_meanAPE': group['curr_ape'].mean(),
                'rf_meanAPE': group['rf_ape'].mean(),
                'curr_medianAPE': group['curr_ape'].median(),
                'rf_medianAPE': group['rf_ape'].median(),
            })

    policy_summary_df = pd.DataFrame(policy_comparison_rows)
    
    os.makedirs(reports_dir, exist_ok=True)
    comparison_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    
    # Save combined comparison CSV
    with open(comparison_csv_path, 'w') as f:
        f.write("# OVERALL MODEL COMPARISON METRICS\n")
        overall_summary_df.to_csv(f, index=False)
        f.write("\n# FQDN LEVEL MODEL COMPARISON METRICS\n")
        fqdn_summary_df.to_csv(f, index=False)
        f.write("\n# POLICY LEVEL MODEL COMPARISON METRICS\n")
        policy_summary_df.to_csv(f, index=False)
    logger.info("--- OVERALL COMPARISON ---")
    logger.info(overall_summary_df.to_string(index=False))

    logger.info(f"Saved performance comparison CSV to: {comparison_csv_path}")

    return overall_summary_df, fqdn_summary_df, policy_summary_df, comparison_csv_path, comp_df


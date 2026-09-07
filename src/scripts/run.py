#!/usr/bin/env python3
"""
GPU Latency Prediction Master Pipeline Script for iluvatar-faas

Driven by config.yaml. Run with:
    python3 src/scripts/pipeline_main.py [--config config.yaml]
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# Ensure src modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scripts.utils import discover_and_load_logs, load_config
from src.scripts.feature_engineering import generate_target_features
from src.scripts.train import get_stratified_session_split, train_and_evaluate_rf
from src.scripts.eda import compute_basic_statistics, perform_eda_and_save_plots
from src.scripts.evaluate import compare_with_current_estimate

logger = logging.getLogger(__name__)


def setup_logging(logs_path):
    """Configure root logger to write to both console and a timestamped log file."""
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f"pipeline.log")

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    return log_file


def extract_features(raw_df, output_dir, feature_config):
    """
    Stage 3: Feature extraction pipeline using feature_engineering_pipeline.
    feature_config is the training.features dict from config.yaml.
    """
    logger.info("==========================================")
    logger.info("STAGE 3: FEATURE EXTRACTION PIPELINE")
    logger.info("==========================================")

    features_df = generate_target_features(raw_df, feature_config=feature_config)

    # Preserve session and dispatch_policy in features_df if lost during merge
    if 'session' not in features_df.columns or 'dispatch_policy' not in features_df.columns:
        session_policy_map = (
            raw_df[['tid', 'session', 'dispatch_policy']]
            .dropna(subset=['tid'])
            .drop_duplicates(subset=['tid'])
        )
        features_df = features_df.merge(session_policy_map, on='tid', how='left')

    features_df['session'] = features_df.get('session', pd.Series('default_session', index=features_df.index)).fillna('default_session')
    features_df['dispatch_policy'] = features_df.get('dispatch_policy', pd.Series('default', index=features_df.index)).fillna('default')

    os.makedirs(output_dir, exist_ok=True)
    feature_csv_path = os.path.join(output_dir, "feature_data.csv")
    features_df.to_csv(feature_csv_path, index=False)
    logger.info("Features successfully extracted! Shape: %s", features_df.shape)
    logger.info("Saved feature dataset to: %s", feature_csv_path)

    return features_df, feature_csv_path


def save_model_and_errors(best_rf, test_df, model_dir, error_dir):
    """Stage 6: Save model binary and test error dataset."""
    logger.info("==========================================")
    logger.info("STAGE 6: SAVING MODEL & ERROR ARTIFACTS")
    logger.info("==========================================")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"rf_model.joblib")
    joblib.dump(best_rf, model_path)
    logger.info("Saved Random Forest model to: %s", model_path)

    error_path = os.path.join(error_dir, f"rf_test_errors.csv")
    save_cols = ['session', 'fqdn', 'tid', 'dispatch_policy', 'e2etime',
                 'rf_prediction', 'rf_absolute_error', 'rf_squared_error', 'rf_ape']
    test_df[[c for c in save_cols if c in test_df.columns]].to_csv(error_path, index=False)
    logger.info("Saved RF test error data to: %s", error_path)

    return model_path, error_path


def format_df_as_markdown(df, float_decimals=3):
    """Format DataFrame to markdown table with float columns rounded to float_decimals places."""
    if df is None or df.empty:
        return ""
    df = df.copy()
    float_cols = df.select_dtypes(include=['float64', 'float32', 'float16']).columns
    df[float_cols] = df[float_cols].round(float_decimals)
    try:
        return df.to_markdown(index=False, floatfmt=f'.{float_decimals}f')
    except Exception:
        headers = [str(c) for c in df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join([str(v) for v in row]) + " |" for row in df.values]
        return "\n".join([header_line, sep_line] + row_lines)


def generate_run_report(stats, eval_summary, overall_summary_df, fqdn_summary_df, policy_summary_df,
                        feature_csv_path, model_path, error_path, comparison_csv_path,
                        plot_paths, reports_dir, config):
    """Stage 8: Synthesize findings into a comprehensive markdown report."""
    logger.info("==========================================")
    logger.info("STAGE 8: GENERATING RUN REPORT")
    logger.info("==========================================")

    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"run_report.md")

    overall_table_md = format_df_as_markdown(overall_summary_df)
    fqdn_table_md = (format_df_as_markdown(fqdn_summary_df)
                     if fqdn_summary_df is not None and len(fqdn_summary_df) > 0
                     else "No FQDN breakdown available.")
    policy_table_md = (format_df_as_markdown(policy_summary_df)
                       if policy_summary_df is not None and len(policy_summary_df) > 0
                       else "No policy breakdown available.")

    e2e_stats = stats.get('e2e_stats', {})
    training_cfg = config.get('training', {})
    policies_used = training_cfg.get('policies', ['all'])
    feature_config = training_cfg.get('features', {})
    enabled_cont = [f for f, v in feature_config.get('continuous', {}).items() if v]
    enabled_cat = [f for f, v in feature_config.get('categorical', {}).items() if v]

    report_md = f"""# GPU Latency Prediction Pipeline Run Report


**Generated At:** `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`  
**Config:** `{config.get('_config_path', 'config.yaml')}`

---

## 1. Executive Summary & Workflow Overview

Automated GPU latency prediction pipeline for `iluvatar-faas`:
1. Recursive ingestion of FaaS session worker logs.
2. Invocation & policy level statistics.
3. Feature engineering (toggleable per config).
4. EDA visual plots.
5. Stratified session train-test split & Random Forest hyperparameter tuning.
6. Export of trained model and evaluation error datasets.
7. Side-by-side evaluation vs the current baseline estimate (`gpu_est_total`).

---

## 2. Configuration Summary

### Policies Used for Training
`{', '.join(policies_used)}`

### Active Features
| Type | Features |
| :--- | :--- |
| Continuous | `{', '.join(enabled_cont) or 'none'}` |
| Categorical | `{', '.join(enabled_cat) or 'none'}` |

---

## 3. Ingestion & Basic Statistics

### Session & Invocation Metrics
- **Total Log Sessions Processed:** `{stats['num_sessions']}`
- **Total Log Records:** `{stats['total_log_records']}`
- **Total Invocations:** `{stats['total_invocations']}`
- **Invocations per Session:** Mean `{stats['inv_mean']:.3f}` | Median `{stats['inv_median']:.3f}` | Min `{stats['inv_min']}` | Max `{stats['inv_max']}`

### Dispatch Policy Distribution
| Policy | Session Count | Invocation Count |
| :--- | :--- | :--- |
"""
    for pol, sess_cnt in stats['policy_session_counts'].items():
        inv_cnt = stats['policy_invocation_counts'].get(pol, 0)
        report_md += f"| `{pol}` | {sess_cnt} | {inv_cnt} |\n"

    report_md += f"""
### Latency (`e2etime`) Characterization
- **Mean:** `{e2e_stats.get('mean', 0):.3f} s` | **Std:** `{e2e_stats.get('std', 0):.3f} s`
- **P50:** `{e2e_stats.get('p50', 0):.3f} s` | **P90:** `{e2e_stats.get('p90', 0):.3f} s` | **P95:** `{e2e_stats.get('p95', 0):.3f} s` | **P99:** `{e2e_stats.get('p99', 0):.3f} s`
- **Range:** `[{e2e_stats.get('min', 0):.3f} s, {e2e_stats.get('max', 0):.3f} s]`

---

## 4. Feature Engineering & EDA

- **Feature Matrix:** [`{os.path.basename(feature_csv_path)}`](file://{feature_csv_path})
- **Active Feature Columns:** `{', '.join(eval_summary['feature_cols'])}`

### Visual Plots Generated
"""
    for p in plot_paths:
        report_md += f"- [{os.path.basename(p)}](file://{p})\n"

    report_md += f"""
---

## 5. Model Training & Evaluation

- **Model Type:** Random Forest Regressor
- **Target Variable:** `log1p(e2etime)`
- **Optimal Hyperparameters:** `{eval_summary['best_params']}`

### Test Set Performance Metrics
| Metric | Value |
| :--- | :--- |
| Log R² | `{eval_summary['test_r2']:.3f}` |
| MAE | `{eval_summary['test_mae']:.3f} s` |
| MSE | `{eval_summary['test_mse']:.3f}` |
| Mean APE | `{eval_summary['test_mean_ape']:.3f}%` |
| Median APE | `{eval_summary['test_med_ape']:.3f}%` |

---

## 6. Performance Comparison: RF Model vs Current Estimate

### Overall Model Comparison
{overall_table_md}

### Policy Level Model Comparison
{policy_table_md}

### FQDN Level Model Comparison
{fqdn_table_md}
---

## 7. Generated Output Artifacts

- **Processed Features:** [`{os.path.basename(feature_csv_path)}`](file://{feature_csv_path})
- **Trained Model Binary:** [`{os.path.basename(model_path)}`](file://{model_path})
- **RF Test Errors Dataset:** [`{os.path.basename(error_path)}`](file://{error_path})
- **Model Comparison CSV:** [`{os.path.basename(comparison_csv_path)}`](file://{comparison_csv_path})
- **Run Summary Report:** [`{os.path.basename(report_path)}`](file://{report_path})
"""

    with open(report_path, 'w') as f:
        f.write(report_md)

    logger.info("Run report successfully created at: %s", report_path)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="GPU Latency Prediction Master Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config(args.config)
    config['_config_path'] = args.config
    experiment_name = config.get("experiment_name", "default")

    paths = config.get("paths", {})
    training = config.get("training", {})

    input_path = paths["input_path"]
    feature_path = f"src/data/processed/{experiment_name}/"
    plots_path = f"src/plots/{experiment_name}/"
    models_path = f"src/models/{experiment_name}/"
    reports_path = f"src/reports/{experiment_name}/"
    error_data_path = f"src/data/processed/{experiment_name}/"
    logs_path = f"src/logs/{experiment_name}"

    max_logs = training.get("max_logs", 10)
    test_size = training.get("test_size", 0.2)
    random_state = training.get("random_state", 42)
    policies = training.get("policies", None)  # None means all policies
    feature_config = training.get("features", None)

    # --- Setup logging (file + console) ---
    log_file = setup_logging(logs_path)
    logger.info("Pipeline started. Config: %s", args.config)
    logger.info("Log file: %s", log_file)

    start_time = time.time()

    # Stage 1: Load logs
    raw_df = discover_and_load_logs(input_path, policies, max_logs=max_logs)

    # Stage 2: Basic statistics
    stats = compute_basic_statistics(raw_df)

    # Stage 3: Feature extraction (respects feature toggles)
    features_df, feature_csv_path = extract_features(
        raw_df, feature_path, feature_config
    )

    # Stage 4: EDA & plots
    plot_sub_dir = os.path.join(plots_path)
    plot_paths = perform_eda_and_save_plots(features_df, raw_df, plot_sub_dir)

    # Stage 5: Train & Evaluate RF Model
    best_rf, train_df, test_df, eval_summary = train_and_evaluate_rf(
        features_df, test_size=test_size, random_state=random_state
    )

    # Stage 6: Save Model & Error Data
    model_path, error_path = save_model_and_errors(
        best_rf, test_df, models_path, error_data_path
    )

    # Stage 7: Compare RF with Current Estimator
    overall_summary_df, fqdn_summary_df, policy_summary_df, comparison_csv_path, comp_df = compare_with_current_estimate(
        test_df, raw_df, reports_path
    )

    # Stage 8: Generate Report
    report_path = generate_run_report(
        stats, eval_summary, overall_summary_df, fqdn_summary_df, policy_summary_df, 
        feature_csv_path, model_path, error_path, comparison_csv_path,
        plot_paths, reports_path, config
    )

    elapsed_time = time.time() - start_time
    logger.info("==========================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed_time)
    logger.info("Run Report: %s", report_path)
    logger.info("==========================================")


if __name__ == "__main__":
    main()

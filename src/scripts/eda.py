import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_basic_statistics(raw_df):
    """
    Stage 2: Report basic statistics (sessions, invocations, policy counts, e2etime stats).
    """
    logger.info("==========================================")
    logger.info("STAGE 2: BASIC STATISTICS REPORTING")
    logger.info("==========================================")

    # Filter for handling invocation request or e2etime records
    invocation_df = raw_df[raw_df['message'] == 'Handling invocation request'] if 'message' in raw_df.columns else raw_df
    
    # Invocation counts per session
    invocations_per_session = invocation_df.groupby('session').size()
    
    stats = {
        'num_sessions': int(raw_df['session'].nunique()),
        'total_log_records': len(raw_df),
        'total_invocations': len(invocation_df),
        'inv_mean': float(invocations_per_session.mean()) if len(invocations_per_session) > 0 else 0,
        'inv_median': float(invocations_per_session.median()) if len(invocations_per_session) > 0 else 0,
        'inv_min': int(invocations_per_session.min()) if len(invocations_per_session) > 0 else 0,
        'inv_max': int(invocations_per_session.max()) if len(invocations_per_session) > 0 else 0,
    }

    # Policy counts
    policy_counts = raw_df.groupby('dispatch_policy')['session'].nunique().to_dict()
    policy_invocations = invocation_df.groupby('dispatch_policy').size().to_dict()
    
    stats['policy_session_counts'] = policy_counts
    stats['policy_invocation_counts'] = policy_invocations

    # e2etime statistics
    e2e_series = pd.to_numeric(raw_df['e2etime'], errors='coerce').dropna()
    if len(e2e_series) > 0:
        stats['e2e_stats'] = {
            'count': int(len(e2e_series)),
            'mean': float(e2e_series.mean()),
            'std': float(e2e_series.std()),
            'min': float(e2e_series.min()),
            'p25': float(e2e_series.quantile(0.25)),
            'p50': float(e2e_series.median()),
            'p75': float(e2e_series.quantile(0.75)),
            'p90': float(e2e_series.quantile(0.90)),
            'p95': float(e2e_series.quantile(0.95)),
            'p99': float(e2e_series.quantile(0.99)),
            'max': float(e2e_series.max()),
        }
    else:
        stats['e2e_stats'] = {}

    logger.info(f"Number of Sessions (Logs): {stats['num_sessions']}")
    logger.info(f"Invocations per Session - Mean: {stats['inv_mean']:.2f}, Median: {stats['inv_median']:.2f}, Min: {stats['inv_min']}, Max: {stats['inv_max']}")
    logger.info(f"Logs/Sessions per Policy: {policy_counts}")
    logger.info(f"Invocations per Policy: {policy_invocations}")
    if stats['e2e_stats']:
        logger.info(f"Latency (e2etime) - Mean: {stats['e2e_stats']['mean']:.4f}s, Median: {stats['e2e_stats']['p50']:.4f}s, P95: {stats['e2e_stats']['p95']:.4f}s, Max: {stats['e2e_stats']['max']:.4f}s")

    return stats


def perform_eda_and_save_plots(features_df, raw_df, plot_dir):
    """
    Stage 4: Basic EDA on features and e2etime with charts saved to reports/plots.
    """
    logger.info("==========================================")
    logger.info("STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("==========================================")

    os.makedirs(plot_dir, exist_ok=True)
    generated_plots = []
    sns.set_theme(style="whitegrid")

    # Plot 1: e2etime Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(features_df['e2etime'], kde=True, color='skyblue')
    plt.title('e2etime (Latency) Distribution')
    plt.xlabel('Latency (s)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(features_df['e2etime']), kde=True, color='teal')
    plt.title('Log1p(e2etime) Distribution')
    plt.xlabel('log1p(Latency)')
    
    p1 = os.path.join(plot_dir, 'e2etime_distribution.png')
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()
    generated_plots.append(p1)

    # Plot 2: Latency by Dispatch Policy
    if 'dispatch_policy' in features_df.columns and features_df['dispatch_policy'].nunique() > 1:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=features_df, x='dispatch_policy', y='e2etime', hue='dispatch_policy', palette='Set2', legend=False)
        plt.title('Latency (e2etime) by Dispatch Policy')
        plt.xlabel('Dispatch Policy')
        plt.ylabel('Latency (s)')
        p2 = os.path.join(plot_dir, 'e2etime_by_policy.png')
        plt.tight_layout()
        plt.savefig(p2, dpi=300)
        plt.close()
        generated_plots.append(p2)

    # Plot 3: Latency by FQDN
    if 'fqdn' in features_df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=features_df, x='fqdn', y='e2etime', hue='fqdn', palette='Set3', legend=False)
        plt.xticks(rotation=45, ha='right')
        plt.title('Latency (e2etime) by Function (FQDN)')
        plt.xlabel('Function FQDN')
        plt.ylabel('Latency (s)')
        p3 = os.path.join(plot_dir, 'e2etime_by_fqdn.png')
        plt.tight_layout()
        plt.savefig(p3, dpi=300)
        plt.close()
        generated_plots.append(p3)

    # Plot 4: Feature Correlation Heatmap
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['tid']]
    if len(feature_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = features_df[feature_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        p4 = os.path.join(plot_dir, 'feature_correlations.png')
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close()
        generated_plots.append(p4)

    logger.info(f"Generated {len(generated_plots)} EDA plots in {plot_dir}")
    return generated_plots
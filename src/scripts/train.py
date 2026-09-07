import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def get_stratified_session_split(df, session_col='session', stratify_col='dispatch_policy', test_size=0.2, random_state=42):
    """
    Splits the dataframe into train and test sets by guaranteeing two things:
    1. Independent sessions are kept perfectly intact (no data leakage).
    2. The proportion of each 'policy' is strictly maintained in both Train and Test sets.
    """
    logger.info(f"Original Dataset Size: {len(df)} rows")
    
    # 1. Extract the unique 1-to-1 mapping of session -> policy
    session_mapping = df[[session_col, stratify_col]].drop_duplicates()
    
    logger.info("Policy Distribution Across Total Sessions:")
    logger.info(session_mapping[stratify_col].value_counts().to_string())
    
    # 2. Perform a Stratified split ONLY on the session IDs
    train_sessions, test_sessions = train_test_split(
        session_mapping[session_col],
        test_size=test_size,
        stratify=session_mapping[stratify_col],
        random_state=random_state
    )
    
    # 3. Reconstruct the full datasets based on the isolated session IDs
    train_df = df[df[session_col].isin(train_sessions)].copy()
    test_df = df[df[session_col].isin(test_sessions)].copy()
    
    logger.info("Stratified Split Complete!")
    logger.info(f"Train Sessions: {len(train_sessions)} | Test Sessions: {len(test_sessions)}")
    
    return train_df, test_df


def train_and_evaluate_rf(features_df, test_size=0.2, random_state=42):
    """
    Perform session-stratified train-test split, train RF model with hyperparameter tuning, and evaluate.

    Args:
        features_df: Feature DataFrame.
        test_size: Fraction of sessions for the test set.
        random_state: Random seed.

    Returns: best_rf, train_df, test_df, eval_summary
    """
    logger.info("==========================================")
    logger.info("STAGE 5: MODEL TRAINING & EVALUATION")
    logger.info("==========================================")

    # --- Feature selection: use whatever columns are present, excluding metadata ---
    non_feature_cols = {'tid', 'invocation_timestamp', 'e2etime', 'fqdn', 'session',
                        'dispatch_policy', 'log_path', 'base_function', 'target_queue_status'}
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols
                    and features_df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8', 'bool']]

    target_col = 'e2etime'
    cols_to_check = [target_col, 'session', 'dispatch_policy'] + feature_cols
    df_clean = features_df.dropna(subset=[c for c in cols_to_check if c in features_df.columns]).copy()

    # Stratified session split
    try:
        train_df, test_df = get_stratified_session_split(
            df_clean, session_col='session', stratify_col='dispatch_policy',
            test_size=test_size, random_state=random_state
        )
    except Exception as e:
        logger.warning(f"Stratified split warning ({e}). Falling back to session train_test_split...")
        unique_sessions = df_clean['session'].unique()
        tr_sess, te_sess = train_test_split(unique_sessions, test_size=test_size, random_state=random_state)
        train_df = df_clean[df_clean['session'].isin(tr_sess)].copy()
        test_df = df_clean[df_clean['session'].isin(te_sess)].copy()

    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[target_col])
    X_test = test_df[feature_cols]
    y_test = np.log1p(test_df[target_col])

    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    logger.info(f"Features used ({len(feature_cols)}): {feature_cols}")

    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [25, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 1.0]
    }
    
    rf = RandomForestRegressor(random_state=random_state)
    cv_folds = min(3, len(train_df['session'].unique()))
    if cv_folds >= 2:
        random_search = RandomizedSearchCV(
            estimator=rf, param_distributions=param_dist, cv=cv_folds,
            scoring='neg_mean_squared_error', n_iter=6, n_jobs=-1, random_state=random_state
        )
        random_search.fit(X_train, y_train)
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_
    else:
        rf.fit(X_train, y_train)
        best_rf = rf
        best_params = rf.get_params()

    # Predict
    test_pred_log = best_rf.predict(X_test)
    test_pred_real = np.expm1(test_pred_log)
    real_y_test = np.expm1(y_test)

    # Metrics
    test_r2 = r2_score(y_test, test_pred_log)
    test_mse = mean_squared_error(real_y_test, test_pred_real)
    test_mae = mean_absolute_error(real_y_test, test_pred_real)
    test_mean_ape = np.mean(np.abs((real_y_test - test_pred_real) / real_y_test)) * 100
    test_med_ape = np.median(np.abs((real_y_test - test_pred_real) / real_y_test)) * 100

    logger.info("--- PERFORMANCE METRICS ---")
    logger.info("RF Model Test Metrics:")
    logger.info(f" - Log R2: {test_r2:.4f}")
    logger.info(f" - MAE: {test_mae:.4f}s")
    logger.info(f" - MSE: {test_mse:.4f}")
    logger.info(f" - Mean APE: {test_mean_ape:.2f}%")
    logger.info(f" - Median APE: {test_med_ape:.2f}%")

    test_df = test_df.copy()
    test_df['rf_prediction'] = test_pred_real
    test_df['rf_absolute_error'] = (test_df[target_col] - test_df['rf_prediction']).abs()
    test_df['rf_squared_error'] = (test_df[target_col] - test_df['rf_prediction']) ** 2
    test_df['rf_ape'] = (test_df['rf_absolute_error'] / test_df[target_col]) * 100

    eval_summary = {
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_mean_ape': test_mean_ape,
        'test_med_ape': test_med_ape,
        'best_params': best_params,
        'feature_cols': feature_cols
    }

    return best_rf, train_df, test_df, eval_summary


def run_stratified_rf_training(data_path_or_df):
    if isinstance(data_path_or_df, str):
        df = pd.read_csv(data_path_or_df)
    else:
        df = data_path_or_df
    best_rf, train_df, test_df, eval_summary = train_and_evaluate_rf(df)
    return best_rf, test_df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) > 1:
        run_stratified_rf_training(sys.argv[1])
    else:
        DATA_PATH = "src/data/processed/feature_data.csv"
        if os.path.exists(DATA_PATH):
            run_stratified_rf_training(DATA_PATH)


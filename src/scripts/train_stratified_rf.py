import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_stratified_session_split(df, session_col='session', stratify_col='dispatch_policy', test_size=0.2, random_state=42):
    """
    Splits the dataframe into train and test sets by guaranteeing two things:
    1. Independent sessions are kept perfectly intact (no data leakage).
    2. The proportion of each 'policy' is strictly maintained in both Train and Test sets.
    """
    print(f"Original Dataset Size: {len(df)} rows")
    
    # 1. Extract the unique 1-to-1 mapping of session -> policy
    session_mapping = df[[session_col, stratify_col]].drop_duplicates()
    
    print("\nPolicy Distribution Across Total Sessions:")
    print(session_mapping[stratify_col].value_counts())
    
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
    
    print(f"\nStratified Split Complete!")
    print(f"Train Sessions: {len(train_sessions)} | Test Sessions: {len(test_sessions)}")
    
    return train_df, test_df

def run_stratified_rf_training(df, base_cols=None, target_col='e2etime'):
    # print(f"Loading data from: {data_path}")
    # df = pd.read_csv(data_path)
    
    # Define features based on your previous rf_tuning structure
    if base_cols is None:
        base_cols = [
            'target_queue_len', 'others_len_queue', 'iat', 'iat_fqdn', 'num_running_funcs_filled',
            'gpu_warm_results_sec', 'gpu_cold_results_sec', 'is_cold_start'
        ]
    # base_cols = [ # 4 features
    #     'others_len_queue', 'gpu_warm_results_sec', 'gpu_cold_results_sec', 'is_cold_start'
    # ]
    lagged_cols = []#[c for c in df.columns if 'lag' in c]
    
    # We must also ensure we extract 'policy' as a categorical feature if the model should learn it natively!
    feature_cols = base_cols + lagged_cols
    
    # # Optional: One-hot encode the 'policy' column so the RF natively learns the policy rules
    # if 'dispatch_policy' in df.columns:
    #     print("One-hot encoding 'policy' variable for Random Forest...")
    #     policy_dummies = pd.get_dummies(df['dispatch_policy'], prefix='policy')
    #     df = pd.concat([df, policy_dummies], axis=1)
    #     feature_cols.extend(policy_dummies.columns.tolist())
    
    # Clean Data
    cols_to_check = [target_col, 'session', 'dispatch_policy'] + base_cols
    missing = [c for c in cols_to_check if c not in df.columns]
    if missing:
        print(f"WARNING: Missing expected columns for cleaning: {missing}")
        
    df_clean = df.dropna(subset=[c for c in cols_to_check if c in df.columns])
    
    # 1. SPLIT PROPORTIONATELY BY POLICY
    train_df, test_df = get_stratified_session_split(df_clean, session_col='session', stratify_col='dispatch_policy')
    
    # Verify all feature columns are present for the X payload
    final_features = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[final_features]
    y_train = np.log1p(train_df[target_col])
    
    X_test = test_df[final_features]
    y_test = np.log1p(test_df[target_col])
    
    # 2. TUNE AND TRAIN THE MODEL
    print("\n--- STARTING RANDOMIZED SEARCH CV TUNING ---")
    param_dist = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_leaf': [5, 10, 20, 30, 50],
        'max_features': ['sqrt', 'log2', 1.0]
    }
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    
    print(f"\nOptimal Hyperparameters Discovered:")
    for param, val in random_search.best_params_.items():
        print(f" - {param}: {val}")
        
    # 3. EVALUATE BEST MODEL
    train_pred = best_rf.predict(X_train)
    test_pred = best_rf.predict(X_test)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Tuned Train R2 (Log): {r2_score(y_train, train_pred):.4f}")
    print(f"Tuned Test R2  (Log): {r2_score(y_test, test_pred):.4f}")
    
    real_y_test = np.expm1(y_test)
    real_test_pred = np.expm1(test_pred)
    
    print(f"Tuned Test MSE (Real): {mean_squared_error(real_y_test, real_test_pred):.4f}")
    print(f"Tuned Test MAE (Real): {mean_absolute_error(real_y_test, real_test_pred):.4f}")
    
    # 4. EXTRACT FEATURE IMPORTANCES
    importances = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': final_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n--- FEATURE IMPORTANCES ---")
    print(importance_df.to_string(index=False))
    
    # 5. SESSION-WISE EVALUATION
    df_eval = df_clean.copy()
    
    # Model predictions (reverse the log scale)
    rf_pred_log = best_rf.predict(df_eval[final_features])
    df_eval['rf_prediction'] = np.expm1(rf_pred_log)
    
    # Error computations
    df_eval['absolute_error'] = (df_eval[target_col] - df_eval['rf_prediction']).abs()
    df_eval['squared_error'] = (df_eval[target_col] - df_eval['rf_prediction']) ** 2
    df_eval['percentage_relative_error'] = ((df_eval[target_col] - df_eval['rf_prediction']) / df_eval[target_col]) * 100
    df_eval['is_test'] = df_eval.index.isin(X_test.index)
    
    # Build results dataframe dynamically preserving what's available
    save_cols = []
    for c in ['session', 'fqdn', 'tid', 'dispatch_policy', 'policy']:
        if c in df_eval.columns:
            save_cols.append(c)
            
    save_cols.extend([target_col, 'rf_prediction', 'absolute_error', 'squared_error', 'percentage_relative_error', 'is_test'])
    
    # Keep only the columns that actually exist just in case
    save_cols = [c for c in save_cols if c in df_eval.columns]
    results_df = df_eval[save_cols].reset_index(drop=True)
    
    return best_rf, results_df

if __name__ == "__main__":
    DATA_PATH = "/Users/akshaykishan/PycharmProjects/iluvatar-faas/S4forIluvatar/src/data/processed_data/all_policies/all_policies_data_for_rf_training_w_lagged_features_20_coldstart.csv"
    run_stratified_rf_training(DATA_PATH)

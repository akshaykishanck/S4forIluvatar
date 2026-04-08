import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def split_rf_data(df_clean, feature_cols, target_col='e2etime', train_by_sessions=False, test_sessions=None, session_col='session'):
    """
    Independent functional block to handle splitting the dataset. 
    Supports classical randomized splits as well as grouped temporal sessions.
    """
    X = df_clean[feature_cols]
    y = np.log1p(df_clean[target_col])
    
    if train_by_sessions:
        if test_sessions is None:
            raise ValueError("You must provide 'test_sessions' as a list when 'train_by_sessions=True'")
            
        if session_col not in df_clean.columns:
            raise ValueError(f"Session column '{session_col}' not found in dataframe.")
            
        test_mask = df_clean[session_col].isin(test_sessions)
        
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        
        print(f"Session Splitting Active | Train Size: {len(X_train)} | Test Size: {len(X_test)} (Holdout Sessions: {test_sessions})")
    else:
        print(f"Default Random Split (80:20)... Total Valid Samples: {len(X)}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
    return X_train, X_test, y_train, y_test

def tune_rf_model(features_df, target_col='e2etime', train_by_sessions=False, test_sessions=None, session_col='session'):
    """
    Takes the deeply engineered RF feature dataframe, extracts training targets,
    sets up a consistent 80:20 split, tests a baseline, and then tunes the Random
    Forest parameters to prevent the massive overfitting seen initially.
    """
    # 1. Identify valid feature columns generically
    base_cols = [
        'target_queue_len', 'others_len_queue', 'is_status_Active', 'is_status_Inactive', 
        'is_status_Throttled', 'iat', 'iat_fqdn', 'num_running_funcs_filled',
        'gpu_warm_results_sec', 'gpu_cold_results_sec', 'is_cold_start'
    ]
    lagged_cols = [c for c in features_df.columns if 'lag' in c]
    feature_cols = base_cols + lagged_cols
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")
        
    # 2. Filter out NA rows
    # Include session_col in the dropna check if we are splitting securely by sessions
    cols_to_check = [target_col] + feature_cols
    if train_by_sessions and session_col in features_df.columns:
        cols_to_check.append(session_col)
        
    df_clean = features_df.dropna(subset=cols_to_check)
    
    # 3. Dynamic Data Splitting
    X_train, X_test, y_train, y_test = split_rf_data(
        df_clean=df_clean, 
        feature_cols=feature_cols, 
        target_col=target_col, 
        train_by_sessions=train_by_sessions,
        test_sessions=test_sessions,
        session_col=session_col
    )
    
    # 4. Baseline Evaluation (User's working configuration)
    print("\n" + "="*50)
    print("USER'S BASELINE MODEL (Overfit Corrected)")
    print("="*50)
    # The user identified that n_estimators=5 and min_samples_leaf=20 prevented massive overfitting
    base_rf = RandomForestRegressor(n_estimators=5, min_samples_leaf=20, random_state=42, n_jobs=-1)
    base_rf.fit(X_train, y_train)
    
    base_train_r2 = r2_score(y_train, base_rf.predict(X_train))
    base_test_r2 = r2_score(y_test, base_rf.predict(X_test))
    print(f"Baseline Train R2 (Log Scale): {base_train_r2:.4f}")
    print(f"Baseline Test R2  (Log Scale): {base_test_r2:.4f}")
    
    # 5. Automated Hyperparameter Tuning
    print("\n" + "="*50)
    print("STARTING RANDOMIZED SEARCH CV TUNING")
    print("="*50)
    print("Searching for optimal parameters bounded for robust generalization...")
    
    # Grid specifically focuses on constrained trees to avoid depth memorization
    param_dist = {
        'n_estimators': [50, 100],            # Don't need huge numbers of trees if they are shallow
        'max_depth': [10, 15, 20],         # Restrict depth to prevent memorizing exact training paths
        'min_samples_leaf': [5, 10, 20, 30, 50],         # The crucial regularizer discovered by the user
        'max_features': ['sqrt', 'log2', 1.0]        # Subsampling features helps decorrelate the trees (increasing robustness)
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    random_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_dist,
        # n_iter=15,          # 15 random combinations
        cv=3,               # 3-Fold Cross Validation
        scoring='neg_mean_squared_error',       # Optimize for mean squared error
        n_jobs=-1,          # Use all CPUs
        verbose=1       # Show progress
        # random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    
    print(f"\nOptimal Hyperparameters Discovered:")
    for param, val in random_search.best_params_.items():
        print(f" - {param}: {val}")
        
    print("\n" + "="*50)
    print("BEST TUNED MODEL EVALUATION")
    print("="*50)
    
    tuned_train_pred = best_rf.predict(X_train)
    tuned_test_pred = best_rf.predict(X_test)
    
    print(f"Tuned Train R2 (Log Scale): {r2_score(y_train, tuned_train_pred):.4f}")
    print(f"Tuned Test R2  (Log Scale): {r2_score(y_test, tuned_test_pred):.4f}")
    
    # Convert predictions and ground truth back from log scale for real-world MSE/MAE
    real_y_test = np.expm1(y_test)
    real_test_pred = np.expm1(tuned_test_pred)
    
    print(f"Tuned Test MSE (Real Units): {mean_squared_error(real_y_test, real_test_pred):.4f}")
    print(f"Tuned Test MAE (Real Units): {mean_absolute_error(real_y_test, real_test_pred):.4f}")
    
    # Extract Feature Importances from Tuned Model
    importances = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n[Tuned Model] Feature Importances:")
    print(importance_df.to_string(index=False))
    
    # 6. Session-wise Evaluation
    df_eval = df_clean.copy()
    
    # Model predictions (best_rf was trained on log transformed targets, so we reverse using expm1)
    rf_pred_log = best_rf.predict(df_eval[feature_cols])
    df_eval['rf_prediction'] = np.expm1(rf_pred_log)
    
    # Error computations
    df_eval['absolute_error'] = (df_eval[target_col] - df_eval['rf_prediction']).abs()
    df_eval['squared_error'] = (df_eval[target_col] - df_eval['rf_prediction']) ** 2
    # Ensure no division by zero issues
    df_eval['percentage_relative_error'] = ((df_eval[target_col] - df_eval['rf_prediction']) / df_eval[target_col]) * 100
    df_eval['is_test'] = df_eval.index.isin(X_test.index)
    
    # Build results dataframe dynamically preserving what's available
    save_cols = []
    if session_col in df_eval.columns:
        save_cols.append(session_col)
    if 'fqdn' in df_eval.columns:
        save_cols.append('fqdn')
    if 'tid' in df_eval.columns:
        save_cols.append('tid')
        
    save_cols.extend([target_col, 'rf_prediction', 'absolute_error', 'squared_error', 'percentage_relative_error', 'is_test'])
    
    # Keep only the columns that actually exist just in case
    save_cols = [c for c in save_cols if c in df_eval.columns]
    results_df = df_eval[save_cols].reset_index(drop=True)
    
    return best_rf, results_df
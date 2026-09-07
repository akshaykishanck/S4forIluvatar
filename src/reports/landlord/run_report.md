# GPU Latency Prediction Pipeline Run Report


**Generated At:** `2026-09-07 19:09:35`  
**Config:** `config.yaml`

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
`landlord`

### Active Features
| Type | Features |
| :--- | :--- |
| Continuous | `target_queue_len, others_len_queue, iat_fqdn, num_running_funcs_filled, gpu_warm_results_sec, gpu_cold_results_sec` |
| Categorical | `is_cold_start` |

---

## 3. Ingestion & Basic Statistics

### Session & Invocation Metrics
- **Total Log Sessions Processed:** `44`
- **Total Log Records:** `2059209`
- **Total Invocations:** `118190`
- **Invocations per Session:** Mean `2686.136` | Median `2349.000` | Min `1107` | Max `6153`

### Dispatch Policy Distribution
| Policy | Session Count | Invocation Count |
| :--- | :--- | :--- |
| `landlord` | 44 | 118190 |

### Latency (`e2etime`) Characterization
- **Mean:** `4.611 s` | **Std:** `24.876 s`
- **P50:** `0.676 s` | **P90:** `3.403 s` | **P95:** `7.076 s` | **P99:** `184.213 s`
- **Range:** `[0.000 s, 560.797 s]`

---

## 4. Feature Engineering & EDA

- **Feature Matrix:** [`feature_data.csv`](file://src/data/processed/landlord/feature_data.csv)
- **Active Feature Columns:** `target_queue_len, others_len_queue, iat_fqdn, num_running_funcs_filled, gpu_warm_results_sec, gpu_cold_results_sec, is_cold_start`

### Visual Plots Generated
- [e2etime_distribution.png](file://src/plots/landlord/e2etime_distribution.png)
- [e2etime_by_fqdn.png](file://src/plots/landlord/e2etime_by_fqdn.png)
- [feature_correlations.png](file://src/plots/landlord/feature_correlations.png)

---

## 5. Model Training & Evaluation

- **Model Type:** Random Forest Regressor
- **Target Variable:** `log1p(e2etime)`
- **Optimal Hyperparameters:** `{'n_estimators': 50, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10}`

### Test Set Performance Metrics
| Metric | Value |
| :--- | :--- |
| Log R² | `0.782` |
| MAE | `0.691 s` |
| MSE | `12.819` |
| Mean APE | `30.767%` |
| Median APE | `11.802%` |

---

## 6. Performance Comparison: RF Model vs Current Estimate

### Overall Model Comparison
| Model | MAE | MSE | meanAPE (%) | medianAPE (%) |
| --- | --- | --- | --- | --- |
| Current Estimate (gpu_est_total) | 1.763 | 31.693 | 200.67 | 78.64 |
| Random Forest Model | 0.666 | 12.471 | 30.973 | 11.167 |

### Policy Level Model Comparison
| dispatch_policy | sample_count | curr_MAE | rf_MAE | curr_MSE | rf_MSE | curr_meanAPE | rf_meanAPE | curr_medianAPE | rf_medianAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| landlord | 16447 | 1.763 | 0.666 | 31.693 | 12.471 | 200.67 | 30.973 | 78.64 | 11.167 |

### FQDN Level Model Comparison
| fqdn | sample_count | curr_MAE | rf_MAE | curr_MSE | rf_MSE | curr_meanAPE | rf_meanAPE | curr_medianAPE | rf_medianAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onnx-roberta-0-0.0.1 | 243 | 3.039 | 1.29 | 31.8 | 12.409 | 331.981 | 46.929 | 97.061 | 26.186 |
| onnx-roberta-1-0.0.1 | 169 | 3.803 | 1.128 | 43.629 | 9.689 | 606.312 | 71.905 | 293.919 | 34.648 |
| onnx-roberta-2-0.0.1 | 453 | 1.474 | 0.589 | 9.814 | 1.686 | 223.242 | 61.951 | 132.388 | 32.626 |
| onnx-roberta-3-0.0.1 | 264 | 1.923 | 0.93 | 22.769 | 8.327 | 166.18 | 45.965 | 72.954 | 37.283 |
| onnx-roberta-4-0.0.1 | 244 | 1.432 | 0.509 | 9.019 | 1.412 | 192.81 | 50.055 | 67.462 | 34.004 |
| pyhpc-eos-0-0.0.1 | 39 | 5.944 | 1.251 | 128.276 | 5.239 | 4940.82 | 176.618 | 84.154 | 11.93 |
| pyhpc-eos-1-0.0.1 | 85 | 5.944 | 2.136 | 116.851 | 22.585 | 2261.102 | 388.696 | 122.735 | 45.617 |
| pyhpc-eos-2-0.0.1 | 38 | 5.357 | 1.278 | 111.777 | 10.149 | 3155.583 | 159.684 | 118.994 | 36.075 |
| pyhpc-eos-3-0.0.1 | 61 | 4.913 | 2.074 | 50.051 | 17.204 | 2550.105 | 180.721 | 109.028 | 48.139 |
| pyhpc-eos-4-0.0.1 | 49 | 3.805 | 1.716 | 31.603 | 16.656 | 1574.485 | 270.869 | 69.464 | 32.483 |
| pyhpc-isoneural-0-0.0.1 | 59 | 9.405 | 3.64 | 217.16 | 87.246 | 2498.226 | 132.958 | 75.494 | 17.261 |
| pyhpc-isoneural-1-0.0.1 | 61 | 17.012 | 11.27 | 969.036 | 684.664 | 1543.104 | 101.504 | 94.466 | 57.535 |
| pyhpc-isoneural-2-0.0.1 | 17 | 10.336 | 4.039 | 199.893 | 79.386 | 2130.246 | 27.257 | 57.15 | 9.497 |
| pyhpc-isoneural-3-0.0.1 | 83 | 9.185 | 6.798 | 323.023 | 318.577 | 230.188 | 112.265 | 80.015 | 31.01 |
| pyhpc-isoneural-4-0.0.1 | 12 | 5.851 | 1.423 | 49.414 | 6.469 | 50.597 | 12.191 | 44.109 | 5.293 |
| rodinia-lavamd-0-0.0.1 | 95 | 1.323 | 0.348 | 5.336 | 0.77 | 106.426 | 16.241 | 79.766 | 13.197 |
| rodinia-lavamd-1-0.0.1 | 496 | 1.462 | 0.665 | 12.184 | 7.815 | 120.203 | 21.085 | 82.879 | 13.052 |
| rodinia-lavamd-2-0.0.1 | 389 | 0.99 | 0.299 | 4.078 | 0.556 | 105.215 | 22.865 | 83.407 | 13.957 |
| rodinia-lavamd-3-0.0.1 | 445 | 0.967 | 0.293 | 2.849 | 1.066 | 102.595 | 20.092 | 82.305 | 13.204 |
| rodinia-lavamd-4-0.0.1 | 544 | 1.155 | 0.456 | 11.701 | 5.158 | 99.383 | 19.341 | 77.799 | 13.327 |
| rodinia-lud-0-0.0.1 | 1018 | 0.86 | 0.299 | 3.832 | 1.275 | 113.963 | 23.164 | 82.063 | 10.972 |
| rodinia-lud-1-0.0.1 | 782 | 1.007 | 0.36 | 6.347 | 2.389 | 115.717 | 23.754 | 79.865 | 11.459 |
| rodinia-lud-2-0.0.1 | 250 | 0.881 | 0.325 | 1.586 | 0.546 | 130.252 | 25.021 | 88.447 | 11.382 |
| rodinia-lud-3-0.0.1 | 721 | 1.052 | 0.735 | 8.336 | 12.944 | 100.448 | 25.489 | 77.566 | 11.058 |
| rodinia-lud-4-0.0.1 | 894 | 0.824 | 0.367 | 3.264 | 1.958 | 103.279 | 22.533 | 79.199 | 10.341 |
| rodinia-myocyte-0-0.0.1 | 54 | 2.236 | 0.321 | 25.78 | 0.483 | 294.025 | 19.334 | 86.328 | 16.502 |
| rodinia-myocyte-1-0.0.1 | 48 | 1.937 | 0.385 | 16.281 | 0.543 | 227.461 | 21.857 | 74.147 | 16.071 |
| rodinia-myocyte-2-0.0.1 | 792 | 1.142 | 0.452 | 23.81 | 10.796 | 164.322 | 36.558 | 88.462 | 17.151 |
| rodinia-myocyte-3-0.0.1 | 154 | 1.098 | 0.326 | 3.248 | 1.609 | 239.648 | 20.617 | 165.366 | 14.543 |
| rodinia-myocyte-4-0.0.1 | 599 | 0.864 | 0.439 | 7.312 | 4.994 | 146.23 | 36.134 | 97.375 | 17.535 |
| rodinia-needle-2-0.0.1 | 959 | 1.583 | 0.693 | 8.184 | 10.114 | 78.993 | 12.623 | 71.773 | 5.683 |
| rodinia-needle-3-0.0.1 | 698 | 1.619 | 0.316 | 35.357 | 2.446 | 78.161 | 10.367 | 70.921 | 5.075 |
| rodinia-needle-4-0.0.1 | 1047 | 2.47 | 0.811 | 92.111 | 25.741 | 87.634 | 12.533 | 71.978 | 5.672 |
| rodinia-pathfinder-0-0.0.1 | 303 | 1.088 | 0.421 | 7.838 | 2.496 | 119.583 | 23.642 | 87.675 | 12.877 |
| rodinia-pathfinder-1-0.0.1 | 58 | 1.642 | 0.587 | 12.792 | 3.968 | 118.047 | 18.176 | 68.033 | 14.428 |
| rodinia-pathfinder-2-0.0.1 | 24 | 0.784 | 0.31 | 0.779 | 0.195 | 64.234 | 19.167 | 38.881 | 14.113 |
| rodinia-pathfinder-3-0.0.1 | 399 | 0.977 | 0.386 | 5.082 | 1.768 | 114.752 | 23.287 | 92.034 | 11.357 |
| rodinia-pathfinder-4-0.0.1 | 820 | 0.679 | 0.305 | 1.353 | 0.684 | 101.086 | 29.411 | 81.768 | 12.502 |
| rodinia-srad-0-0.0.1 | 800 | 2.603 | 0.819 | 59.362 | 23.576 | 86.414 | 9.662 | 72.632 | 4.092 |
| rodinia-srad-1-0.0.1 | 584 | 1.924 | 0.465 | 14.069 | 4.766 | 80.047 | 9.383 | 74.444 | 3.853 |
| rodinia-srad-2-0.0.1 | 219 | 1.917 | 0.413 | 7.244 | 1.565 | 85.455 | 9.954 | 82.875 | 4.55 |
| rodinia-srad-3-0.0.1 | 129 | 2.277 | 0.598 | 12.73 | 3.611 | 92.854 | 12.988 | 77.881 | 5.818 |
| rodinia-srad-4-0.0.1 | 552 | 1.95 | 0.391 | 8.955 | 1.905 | 87.773 | 10.095 | 75.086 | 4.178 |
| squeezenet-0-0.0.1 | 67 | 5.121 | 2.949 | 188.564 | 123.619 | 83.673 | 25.607 | 41.652 | 17.381 |
| squeezenet-1-0.0.1 | 146 | 6.142 | 1.476 | 492.683 | 36.839 | 233.938 | 31.529 | 72.289 | 21.719 |
| squeezenet-2-0.0.1 | 61 | 3.764 | 1.346 | 39.418 | 13.2 | 131.554 | 21.829 | 66.212 | 16.328 |
| squeezenet-4-0.0.1 | 134 | 1.851 | 0.759 | 8.156 | 3.492 | 95.98 | 26.856 | 56.443 | 24.737 |
| torch_rnn-0-0.0.1 | 68 | 7.107 | 2.425 | 133.928 | 24.865 | 1660.617 | 107.768 | 109.12 | 49.506 |
| torch_rnn-1-0.0.1 | 118 | 2.939 | 1.407 | 65.728 | 25.79 | 823.593 | 183.434 | 106.59 | 74.95 |
| torch_rnn-2-0.0.1 | 39 | 4.884 | 1.742 | 44.061 | 6.179 | 1158.118 | 240.563 | 69.684 | 30.536 |
| torch_rnn-3-0.0.1 | 24 | 2.72 | 0.859 | 9.471 | 1.979 | 1528.847 | 139.865 | 62.049 | 23.417 |
| torch_rnn-4-0.0.1 | 40 | 3.809 | 1.135 | 28.265 | 5.943 | 1290.961 | 147.143 | 83.617 | 33.157 |
---

## 7. Generated Output Artifacts

- **Processed Features:** [`feature_data.csv`](file://src/data/processed/landlord/feature_data.csv)
- **Trained Model Binary:** [`rf_model.joblib`](file://src/models/landlord/rf_model.joblib)
- **RF Test Errors Dataset:** [`rf_test_errors.csv`](file://src/data/processed/landlord/rf_test_errors.csv)
- **Model Comparison CSV:** [`model_comparison.csv`](file://src/reports/landlord/model_comparison.csv)
- **Run Summary Report:** [`run_report.md`](file://src/reports/landlord/run_report.md)

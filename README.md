# S4 for Ilúvatar FaaS: Latency Prediction

This repository implements predictive models to estimate the end-to-end execution latency (`e2etime`) of Function-as-a-Service (FaaS) invocations running on the Ilúvatar platform. We are aiming to first evaluate the current latency estimation strategy and explore robust machine learning models capable of handling highly variable load spikes.

## Workflow & Project Structure

The codebase is structured to guide you from evaluating the current heuristic baseline to exploring and training advanced machine learning models.

### 1. Evaluating the Current Strategy
Before exploring machine learning models, our first step is to establish a strong understanding of how the current static latency estimation (Kalman Filter) performs over real-world data:
- **`src/scripts/evaluate_current_estimate.py`**: This script analyzes chronological trace logs (`worker1.log`) and evaluates the absolute, squared, and percentage relative errors of the **current estimation strategy**. Run this script first to generate baseline dataframes and understand the limitations of the existing static approach.

### 2. Feature Engineering & Pipelines (`src/scripts/`)
Following the baseline evaluation, we use these feature engineering pipelines to explore ML models and uncover which features are necessary to accurately map the performance distribution:
- `s4_feature_pipeline.py`: Comprehensive feature extraction pipeline designed specifically for sequence modeling. Generates session-grouped features (queue lengths, GPU active/throttled states, IATs, and cold/warm execution results).
- `rf_feature_pipeline.py`: A feature extractor focused on single-row decoupled feature sets optimized for classical tree-based regressors, injecting lag-based historical features.
- `utils.py`: `pandas` utilities for ingesting and correctly typing JSON-formatted worker logs into structured DataFrames.

### 3. Modeling & Training 
Once features are engineered, two distinct model architectures are available for training and evaluating latency prediction:
- **Random Forest Baseline (`src/scripts/rf_tuning.py`)**: A classical ML baseline regressor that uses cross-validation and hyperparameter tuning to prevent overfitting on exact depth paths. Predicts latency based on carefully engineered lag features.
- **S4 (Structured State Spaces) (`s4_iluvatar.py`)**: An advanced deep learning architecture adapted to sequence-based tasks. It ingests independent execution sequences without padding, mapping dependencies across traces using bounded state spaces. Uses a local sub-module embedded in `s4/`.

### 4. Interactive Notebooks (`src/notebooks/`)
Jupyter environments (`rf_model_comparison.ipynb`, `feature_engineering_notebook.ipynb`, `data_analysis.ipynb`) track the exploratory data engineering workflow, visualization of true FaaS concurrency, burst variance, and raw model evaluation.
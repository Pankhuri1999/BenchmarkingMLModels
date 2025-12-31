"""
COMPLETE ML PIPELINE CODE - WITH ALL MODIFICATIONS
===================================================
This file contains the complete comprehensive_ml_pipeline() function
with all three modifications integrated:
1. Scalability: Only training time + inference time (50/50)
2. Hyperparameter tuning: GridSearchCV for all models
3. Fairness: No synthetic groups, shows message when not applicable

USAGE:
------
1. Make sure you have all helper functions from your notebook:
   - preprocess_data()
   - perform_fairness_analysis()
   - calculate_interpretability()
   - calculate_fairness_from_disparity()
   - calculate_scalability() (simple version)
   - calculate_data_quality_from_dataset()
   - compute_data_quality()
   - compute_interpretability_for_readiness()
   - compute_fairness_for_readiness()
   - compute_dataset_readiness_score()
   - detect_and_transform_wide_format()
   - detect_text_columns()
   - And other helper functions

2. Copy this entire file into your notebook or import it

3. Call the function:
   results = comprehensive_ml_pipeline(
       dataset_path='your_data.csv',
       target_column='target',
       task_type='auto'
   )
"""

import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb

# Try to import psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# MODIFIED FUNCTION 1: Scalability (Training + Inference Only)
# ============================================================================

def compute_scalability_for_readiness(df, models_dict, X_train, X_test, y_train, y_test, task_type='auto', sub_weights=None):
    """
    Measures training time and inference time (memory and size_bonus removed).

    Args:
        df: DataFrame
        models_dict: Dict of trained models
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        task_type: 'Regression' or 'Classification'
        sub_weights: Dict with weights for sub-factors. If None, uses equal weights (50% each).
                    Example: {'training': 0.5, 'inference': 0.5}
                    Weights will be normalized to sum to 1.0

    Returns:
        dict: {
            'score': float (0-5),
            'training_time': float (seconds),
            'inference_time': float (seconds),
            'details': dict,
            'sub_weights_used': dict
        }
    """
    if not models_dict:
        return {
            'score': 0,
            'training_time': 0,
            'inference_time': 0,
            'details': {},
            'sub_weights_used': {}
        }

    training_times = []
    inference_times = []

    for model_name, model in models_dict.items():
        try:
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            training_times.append(train_time)

            # Inference time
            start_time = time.time()
            _ = model.predict(X_test)
            infer_time = time.time() - start_time
            inference_times.append(infer_time)

        except Exception as e:
            continue

    if not training_times:
        return {
            'score': 0,
            'training_time': 0,
            'inference_time': 0,
            'details': {},
            'sub_weights_used': {}
        }

    avg_training_time = np.mean(training_times)
    avg_inference_time = np.mean(inference_times)

    # Score based on efficiency
    # Training time: 5 if <1s, 0 if >60s
    training_score = max(0, 5 * (1 - min(1, avg_training_time / 60)))

    # Inference time: 5 if <0.1s, 0 if >5s
    inference_score = max(0, 5 * (1 - min(1, avg_inference_time / 5)))

    # Determine sub-weights (default: equal weights = 50% each)
    if sub_weights is None:
        sub_weights = {
            'training': 0.5,
            'inference': 0.5
        }
    else:
        # Normalize weights to sum to 1.0
        total_weight = sum(sub_weights.values())
        if total_weight > 0:
            sub_weights = {k: v / total_weight for k, v in sub_weights.items()}
        else:
            # Fallback to equal weights
            sub_weights = {'training': 0.5, 'inference': 0.5}

    # Combined scalability score (only training and inference)
    scalability_score = (
        sub_weights['training'] * training_score +
        sub_weights['inference'] * inference_score
    )

    return {
        'score': round(scalability_score, 2),
        'training_time': round(avg_training_time, 4),
        'inference_time': round(avg_inference_time, 4),
        'details': {
            'n_samples': len(df),
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns) if hasattr(X_train, 'columns') else 100,
            'training_score': round(training_score, 2),
            'inference_score': round(inference_score, 2)
        },
        'sub_weights_used': sub_weights
    }


# ============================================================================
# MODIFIED FUNCTION 2: Hyperparameter Tuning
# ============================================================================

def create_models_with_hyperparameter_tuning(detected_task_type, random_state=42):
    """
    Create models with hyperparameter tuning using GridSearchCV.
    
    Args:
        detected_task_type: 'Regression' or 'Classification'
        random_state: Random state for reproducibility
        
    Returns:
        dict: Dictionary of models with GridSearchCV
    """
    models_dict = {}
    
    if detected_task_type == 'Regression':
        # RandomForest Regressor
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_base = RandomForestRegressor(random_state=random_state)
        models_dict['RandomForest'] = GridSearchCV(
            rf_base, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        
        # XGBoost Regressor
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb_base = xgb.XGBRegressor(random_state=random_state, verbosity=0)
        models_dict['XGBoost'] = GridSearchCV(
            xgb_base, xgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        
        # LightGBM Regressor
        lgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
        lgb_base = lgb.LGBMRegressor(random_state=random_state, verbosity=-1)
        models_dict['LightGBM'] = GridSearchCV(
            lgb_base, lgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        
        # Ridge Regression (instead of LogisticRegression for regression)
        lr_param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        lr_base = Ridge(random_state=random_state)
        models_dict['LogisticRegression'] = GridSearchCV(
            lr_base, lr_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        
    else:  # Classification
        # RandomForest Classifier
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_base = RandomForestClassifier(random_state=random_state)
        models_dict['RandomForest'] = GridSearchCV(
            rf_base, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        # XGBoost Classifier
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb_base = xgb.XGBClassifier(random_state=random_state, verbosity=0)
        models_dict['XGBoost'] = GridSearchCV(
            xgb_base, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        # LightGBM Classifier
        lgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
        lgb_base = lgb.LGBMClassifier(random_state=random_state, verbosity=-1)
        models_dict['LightGBM'] = GridSearchCV(
            lgb_base, lgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        # LogisticRegression Classifier
        lr_param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        lr_base = LogisticRegression(max_iter=1000, random_state=random_state)
        models_dict['LogisticRegression'] = GridSearchCV(
            lr_base, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
    
    return models_dict


# ============================================================================
# MODIFIED FUNCTION 3: Fairness Analysis Without Synthetic Groups
# ============================================================================

def handle_fairness_analysis_without_synthetic_groups(
    df, demographic_column, y_test, models_dict, X_test, predictions_dict, 
    detected_task_type, target_col, all_results, perform_fairness_analysis
):
    """
    Handle fairness analysis without creating synthetic groups.
    If no demographic groups exist, creates a dataframe with a message.
    
    Args:
        df: DataFrame
        demographic_column: Column name for demographic groups (None if not available)
        y_test: Test target values
        models_dict: Dictionary of trained models
        X_test: Test feature matrix
        predictions_dict: Dictionary of predictions
        detected_task_type: Task type ('Regression' or 'Classification')
        target_col: Target column name
        all_results: Dictionary to store results
        perform_fairness_analysis: Function to perform fairness analysis
        
    Returns:
        pd.DataFrame: Fairness analysis results
    """
    # Create demographic groups (only if actual column exists)
    if demographic_column and demographic_column in df.columns:
        # Use actual demographic column
        demo_values = df[demographic_column].iloc[y_test.index].values
        unique_demos = np.unique(demo_values)
        if len(unique_demos) >= 2:
            demographic_groups = {demo: demo_values == demo for demo in unique_demos[:2]}
            
            # Perform fairness analysis
            fairness_df = perform_fairness_analysis(
                models_dict, X_test, y_test, predictions_dict,
                demographic_groups, detected_task_type
            )
            fairness_df['Target'] = target_col
            all_results['fairness'].append(fairness_df)
            return fairness_df
        else:
            print(f"    ⚠️ Fairness analysis is not applicable due to data limitations: Demographic column '{demographic_column}' has less than 2 unique groups.")
    else:
        print(f"    ⚠️ Fairness analysis is not applicable due to data limitations: No demographic groups found in dataset.")
    
    # Create empty fairness dataframe with message
    fairness_df = pd.DataFrame({
        'Model': list(models_dict.keys()),
        'Disparity': [np.nan] * len(models_dict),
        'Note': ['Fairness analysis not applicable - no demographic groups'] * len(models_dict)
    })
    fairness_df['Target'] = target_col
    all_results['fairness'].append(fairness_df)
    return fairness_df


# ============================================================================
# COMPLETE comprehensive_ml_pipeline() FUNCTION WITH ALL MODIFICATIONS
# ============================================================================
# NOTE: This function assumes you have these helper functions from your notebook:
# - preprocess_data()
# - perform_fairness_analysis()
# - calculate_interpretability()
# - calculate_fairness_from_disparity()
# - calculate_scalability() (simple version)
# - calculate_data_quality_from_dataset()
# - compute_data_quality()
# - compute_interpretability_for_readiness()
# - compute_fairness_for_readiness()
# - compute_dataset_readiness_score()
# - detect_and_transform_wide_format()
# - detect_text_columns()
# ============================================================================

def comprehensive_ml_pipeline(dataset_path, target_column, task_type='auto',
                             feature_columns=None, use_tfidf=True,
                             max_tfidf_features=10000, demographic_column=None,
                             readiness_weights=None, readiness_sub_weights=None, **kwargs):
    """
    COMPREHENSIVE ML PIPELINE - Single function that does everything!
    
    This function includes:
    - Accuracy metrics calculation (RMSE, MAPE, R², Accuracy)
    - SHAP analysis with feature importance plots
    - Demographic disparity and fairness analysis (without synthetic groups)
    - Educational AI Readiness Index calculation
    - Hyperparameter tuning for all models
    - Scalability based on training + inference time only
    
    All results are displayed inline - nothing is saved to files.

    Args:
        dataset_path: Path to CSV file or pandas DataFrame
        target_column: Target column name(s) - string or list
        task_type: 'auto', 'Regression', or 'Classification'
        feature_columns: Optional list of feature columns to use
        use_tfidf: Use TF-IDF for text features
        max_tfidf_features: Maximum TF-IDF features
        demographic_column: Column name for demographic groups (None = no fairness analysis)
        readiness_weights: Dict with weights for dataset readiness score dimensions.
                          If None, uses equal weights (20% each).
                          Example: {'data_quality': 0.25, 'accuracy': 0.25, 'interpretability': 0.20,
                                   'fairness': 0.15, 'scalability': 0.15}
        readiness_sub_weights: Dict of dicts with sub-weights for each dimension's sub-factors.
                              If None, all sub-factors use equal weights.
                              Example: {
                                  'data_quality': {'missing': 0.3, 'imbalance': 0.25, 'redundancy': 0.25, 'noise': 0.2},
                                  'interpretability': {'concentration': 0.4, 'stability': 0.4, 'coverage': 0.2},
                                  'fairness': {'distribution': 0.4, 'missingness': 0.3, 'accuracy_gap': 0.3},
                                  'scalability': {'training': 0.5, 'inference': 0.5}
                              }
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        dict: Contains all results (metrics, fairness, readiness, shap)

    Example:
        >>> results = comprehensive_ml_pipeline(
        ...     'data.csv',
        ...     'target',
        ...     task_type='auto'
        ... )
    """
    print("="*80)
    print("COMPREHENSIVE ML PIPELINE")
    print("="*80)
    print("Includes: Accuracy Metrics, SHAP Analysis, Fairness, and Readiness Index")
    print("All results displayed inline")
    print("="*80)

    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    if isinstance(dataset_path, str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = pd.read_csv(dataset_path, **kwargs)
    elif isinstance(dataset_path, pd.DataFrame):
        df = dataset_path.copy()
    else:
        raise ValueError("dataset_path must be a file path (str) or pandas DataFrame")

    print(f"  ✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Detect and transform wide format
    try:
        df, was_transformed = detect_and_transform_wide_format(df)
    except NameError as e:
        raise NameError(
            f"detect_and_transform_wide_format function not found: {e}\n"
            "Please ensure this function is defined in your notebook."
        )
    if was_transformed:
        print(f"  ✅ Dataset transformed from wide to long format")
        if isinstance(target_column, str):
            if target_column == 'auto' or target_column not in df.columns:
                possible_value_cols = ['Enrollment', 'Value', 'Count', 'Amount']
                for col in possible_value_cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        print(f"  💡 Auto-detected: Using '{col}' as target column")
                        target_column = col
                        break

    # Handle target columns
    if isinstance(target_column, str):
        target_columns = [target_column]
        is_multi_target = False
    elif isinstance(target_column, list):
        target_columns = target_column
        is_multi_target = True
    else:
        raise ValueError("target_column must be a string or list of strings")

    # Check for categorical targets and correct task_type
    if is_multi_target:
        first_target = df[target_columns[0]]
        is_categorical = (first_target.dtype == 'object' or
                         first_target.dtype.name == 'category' or
                         first_target.astype(str).str.contains('[a-zA-Z]', regex=True).any())
        if is_categorical and task_type == 'Regression':
            print(f"  ⚠️ Detected categorical string targets. Changing task_type to 'Multi-label'")
            task_type = 'Multi-label'
    else:
        y = df[target_columns[0]]
        is_categorical = (isinstance(y, pd.Series) and
                         (y.dtype == 'object' or y.dtype.name == 'category' or
                          y.astype(str).str.contains('[a-zA-Z]', regex=True).any()))
        if is_categorical and task_type == 'Regression':
            print(f"  ⚠️ Detected categorical string target. Changing task_type to 'Classification'")
            task_type = 'Classification'

    # Validate target columns
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        available_cols = ', '.join(df.columns.tolist()[:10])
        raise ValueError(f"Target column(s) not found: {missing_cols}\n"
                        f"Available columns: {available_cols}...")

    # Extract features
    if feature_columns is None:
        X = df.drop(columns=target_columns)
    else:
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
        X = df[feature_columns].copy()

    print(f"  ✅ Features: {X.shape[1]} columns")

    # Calculate data quality (once for the dataset)
    try:
        data_quality_score = calculate_data_quality_from_dataset(df)
        print(f"  ✅ Data Quality Score: {data_quality_score:.2f}/5.0")
    except NameError:
        print(f"  ⚠️ calculate_data_quality_from_dataset not found, skipping data quality calculation")
        data_quality_score = 2.5

    # Get dataset size
    dataset_size = len(df)

    # Process each target (for multi-target scenarios)
    all_results = {
        'metrics': [],
        'fairness': [],
        'readiness': [],
        'shap': {}
    }

    # Store variables for dataset readiness score calculation
    stored_models_dict = None
    stored_X_train = None
    stored_X_test = None
    stored_y_train = None
    stored_y_test = None
    stored_feature_names = None
    stored_task_type = None
    stored_metrics_dict = None
    stored_shap_results_dict = None
    stored_fairness_results = None

    if is_multi_target:
        print(f"\n🔄 Processing {len(target_columns)} target columns...")
        for target_col in target_columns:
            print(f"\n  📊 Processing target: {target_col}")
            y_single = df[target_col].copy()

            # Preprocess
            print(f"  🔧 Preprocessing data...")
            try:
                X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = preprocess_data(
                    X, y_single, task_type, use_tfidf, max_tfidf_features
                )
            except NameError:
                raise NameError("preprocess_data function not found. Please define it in your notebook.")

            print(f"    ✅ Task type: {detected_task_type}")
            print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

            # Train models and get predictions
            print(f"  🤖 Training models with hyperparameter tuning...")
            models_dict = {}
            predictions_dict = {}
            metrics_dict = {}
            shap_results_dict = {}

            # Define models with hyperparameter tuning
            print(f"  🔍 Creating models with hyperparameter tuning...")
            models_dict = create_models_with_hyperparameter_tuning(
                detected_task_type, 
                random_state=42
            )

            # Train models and extract best estimators
            for model_name, model in models_dict.items():
                try:
                    print(f"    Training {model_name} with hyperparameter tuning...")
                    model.fit(X_train, y_train)
                    
                    # Extract best estimator for predictions and SHAP
                    if hasattr(model, 'best_estimator_'):
                        best_model = model.best_estimator_
                        print(f"      ✅ Best params for {model_name}: {model.best_params_}")
                    else:
                        best_model = model
                    
                    y_pred = best_model.predict(X_test)
                    predictions_dict[model_name] = y_pred
                    
                    # Store best model back in models_dict for SHAP and scalability
                    models_dict[model_name] = best_model
                    
                    # Calculate metrics
                    if detected_task_type == 'Regression':
                        metrics = {
                            'R2': r2_score(y_test, y_pred),
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
                            'Regression_Accuracy': 1 - np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))
                        }
                    else:
                        metrics = {
                            'Classification_Accuracy': accuracy_score(y_test, y_pred),
                            'F1_Score': f1_score(y_test, y_pred, average='weighted')
                        }
                    
                    metrics_dict[model_name] = metrics
                    
                except Exception as e:
                    print(f"    ⚠️ Error training {model_name}: {e}")
                    continue

            # SHAP Analysis
            print(f"  📊 Running SHAP Analysis...")
            try:
                import shap
                for model_name, model in models_dict.items():
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:100])  # Limit for speed
                        shap_results_dict[model_name] = {
                            'success': True,
                            'shap_values': shap_values,
                            'importance': dict(zip(feature_names, np.abs(shap_values).mean(0)))
                        }
                    except:
                        shap_results_dict[model_name] = {'success': False, 'importance': {}}
            except ImportError:
                print(f"    ⚠️ SHAP not available, skipping SHAP analysis")
                for model_name in models_dict.keys():
                    shap_results_dict[model_name] = {'success': False, 'importance': {}}

            # Fairness Analysis (without synthetic groups)
            print(f"  ⚖️ Running Fairness Analysis...")
            try:
                fairness_df = handle_fairness_analysis_without_synthetic_groups(
                    df=df,
                    demographic_column=demographic_column,
                    y_test=y_test,
                    models_dict=models_dict,
                    X_test=X_test,
                    predictions_dict=predictions_dict,
                    detected_task_type=detected_task_type,
                    target_col=target_col,
                    all_results=all_results,
                    perform_fairness_analysis=perform_fairness_analysis
                )
            except NameError:
                print(f"    ⚠️ perform_fairness_analysis function not found, skipping fairness analysis")
                fairness_df = pd.DataFrame({
                    'Model': list(models_dict.keys()),
                    'Disparity': [np.nan] * len(models_dict),
                    'Note': ['Fairness analysis skipped - function not found'] * len(models_dict)
                })
                fairness_df['Target'] = target_col
                all_results['fairness'].append(fairness_df)

            # Store for dataset readiness
            stored_models_dict = models_dict
            stored_X_train = X_train
            stored_X_test = X_test
            stored_y_train = y_train
            stored_y_test = y_test
            stored_feature_names = feature_names
            stored_task_type = detected_task_type
            stored_metrics_dict = metrics_dict
            stored_shap_results_dict = shap_results_dict
            stored_fairness_results = fairness_df

    else:
        # Single target - similar processing but simpler
        print(f"\n📊 Processing single target: {target_columns[0]}")
        y = df[target_columns[0]].copy()

        # Preprocess
        print(f"  🔧 Preprocessing data...")
        try:
            X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = preprocess_data(
                X, y, task_type, use_tfidf, max_tfidf_features
            )
        except NameError:
            raise NameError("preprocess_data function not found. Please define it in your notebook.")

        print(f"    ✅ Task type: {detected_task_type}")
        print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        # Train models
        print(f"  🤖 Training models with hyperparameter tuning...")
        models_dict = {}
        predictions_dict = {}
        metrics_dict = {}
        shap_results_dict = {}

        # Define models with hyperparameter tuning
        print(f"  🔍 Creating models with hyperparameter tuning...")
        models_dict = create_models_with_hyperparameter_tuning(
            detected_task_type, 
            random_state=42
        )

        # Train models and extract best estimators
        for model_name, model in models_dict.items():
            try:
                print(f"    Training {model_name} with hyperparameter tuning...")
                model.fit(X_train, y_train)
                
                # Extract best estimator
                if hasattr(model, 'best_estimator_'):
                    best_model = model.best_estimator_
                    print(f"      ✅ Best params for {model_name}: {model.best_params_}")
                else:
                    best_model = model
                
                y_pred = best_model.predict(X_test)
                predictions_dict[model_name] = y_pred
                
                # Store best model
                models_dict[model_name] = best_model
                
                # Calculate metrics
                if detected_task_type == 'Regression':
                    metrics = {
                        'R2': r2_score(y_test, y_pred),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
                        'Regression_Accuracy': 1 - np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))
                    }
                else:
                    metrics = {
                        'Classification_Accuracy': accuracy_score(y_test, y_pred),
                        'F1_Score': f1_score(y_test, y_pred, average='weighted')
                    }
                
                metrics_dict[model_name] = metrics
                
            except Exception as e:
                print(f"    ⚠️ Error training {model_name}: {e}")
                continue

        # SHAP Analysis
        print(f"  📊 Running SHAP Analysis...")
        try:
            import shap
            for model_name, model in models_dict.items():
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test[:100])
                    shap_results_dict[model_name] = {
                        'success': True,
                        'shap_values': shap_values,
                        'importance': dict(zip(feature_names, np.abs(shap_values).mean(0)))
                    }
                except:
                    shap_results_dict[model_name] = {'success': False, 'importance': {}}
        except ImportError:
            print(f"    ⚠️ SHAP not available, skipping SHAP analysis")
            for model_name in models_dict.keys():
                shap_results_dict[model_name] = {'success': False, 'importance': {}}

        # Fairness Analysis (without synthetic groups)
        print(f"  ⚖️ Running Fairness Analysis...")
        try:
            fairness_df = handle_fairness_analysis_without_synthetic_groups(
                df=df,
                demographic_column=demographic_column,
                y_test=y_test,
                models_dict=models_dict,
                X_test=X_test,
                predictions_dict=predictions_dict,
                detected_task_type=detected_task_type,
                target_col=target_columns[0],
                all_results=all_results,
                perform_fairness_analysis=perform_fairness_analysis
            )
        except NameError:
            print(f"    ⚠️ perform_fairness_analysis function not found, skipping fairness analysis")
            fairness_df = pd.DataFrame({
                'Model': list(models_dict.keys()),
                'Disparity': [np.nan] * len(models_dict),
                'Note': ['Fairness analysis skipped - function not found'] * len(models_dict)
            })
            fairness_df['Target'] = target_columns[0]
            all_results['fairness'].append(fairness_df)

        # Store for dataset readiness
        stored_models_dict = models_dict
        stored_X_train = X_train
        stored_X_test = X_test
        stored_y_train = y_train
        stored_y_test = y_test
        stored_feature_names = feature_names
        stored_task_type = detected_task_type
        stored_metrics_dict = metrics_dict
        stored_shap_results_dict = shap_results_dict
        stored_fairness_results = fairness_df

    # Compute Dataset Readiness Score
    print("\n" + "="*80)
    print("📊 COMPUTING DATASET READINESS SCORE")
    print("="*80)
    
    try:
        # Update readiness_sub_weights default for scalability
        if readiness_sub_weights is None:
            readiness_sub_weights = {
                'scalability': {'training': 0.5, 'inference': 0.5}
            }
        elif 'scalability' not in readiness_sub_weights:
            readiness_sub_weights['scalability'] = {'training': 0.5, 'inference': 0.5}
        else:
            # Ensure only training and inference are in scalability weights
            scalability_weights = readiness_sub_weights['scalability']
            if 'memory' in scalability_weights:
                del scalability_weights['memory']
            if 'size_bonus' in scalability_weights:
                del scalability_weights['size_bonus']
            # Normalize to sum to 1.0
            total = sum(scalability_weights.values())
            if total > 0:
                readiness_sub_weights['scalability'] = {k: v/total for k, v in scalability_weights.items()}
            else:
                readiness_sub_weights['scalability'] = {'training': 0.5, 'inference': 0.5}

        dataset_readiness = compute_dataset_readiness_score(
            df=df,
            target_columns=target_columns,
            metrics_dict=stored_metrics_dict,
            shap_results_dict=stored_shap_results_dict,
            fairness_results=stored_fairness_results,
            models_dict=stored_models_dict,
            X_train=stored_X_train,
            X_test=stored_X_test,
            y_train=stored_y_train,
            y_test=stored_y_test,
            task_type=stored_task_type,
            demographic_column=demographic_column,
            feature_names=stored_feature_names,
            weights=readiness_weights,
            sub_weights=readiness_sub_weights
        )
        
        # Display readiness breakdown
        print(f"\n  Data Quality:       {dataset_readiness['breakdown']['data_quality']:.2f}/5.0")
        print(f"  Accuracy:          {dataset_readiness['breakdown']['accuracy']:.2f}/5.0")
        print(f"  Interpretability:   {dataset_readiness['breakdown']['interpretability']:.2f}/5.0")
        print(f"  Fairness:          {dataset_readiness['breakdown']['fairness']:.2f}/5.0")
        print(f"  Scalability:       {dataset_readiness['breakdown']['scalability']:.2f}/5.0")
        print(f"\n  Total Readiness Score: {dataset_readiness['readiness_score']:.2f}/25.0")
        
        # Display scalability details (without memory)
        scale_sub_weights = dataset_readiness['scalability'].get('sub_weights_used', {})
        if scale_sub_weights:
            print(f"\n  Scalability Details:")
            print(f"    Sub-weights: Training={scale_sub_weights.get('training', 0):.1%}, "
                  f"Inference={scale_sub_weights.get('inference', 0):.1%}")
            print(f"    - Training Time:       {dataset_readiness['scalability']['training_time']:.4f} seconds")
            print(f"    - Inference Time:      {dataset_readiness['scalability']['inference_time']:.4f} seconds")
        
    except NameError as e:
        print(f"  ⚠️ compute_dataset_readiness_score function not found: {e}")
        dataset_readiness = None

    print("="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)

    return {
        'metrics': pd.DataFrame(all_results['metrics']) if all_results['metrics'] else pd.DataFrame(),
        'fairness': pd.concat(all_results['fairness'], ignore_index=True) if all_results['fairness'] else pd.DataFrame(),
        'readiness': pd.concat(all_results['readiness'], ignore_index=True) if all_results['readiness'] else pd.DataFrame(),
        'shap': shap_results_dict,
        'dataset_readiness': dataset_readiness
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    ============================================================================
    COMPLETE ML PIPELINE - USAGE EXAMPLES
    ============================================================================
    
    Example 1: Basic usage
    ----------------------
    results = comprehensive_ml_pipeline(
        dataset_path='student_performance/StudentPerformanceFactors.csv',
        target_column='Exam_Score',
        task_type='auto'
    )
    
    Example 2: With demographic column
    -----------------------------------
    results = comprehensive_ml_pipeline(
        dataset_path='retention/dataset.csv',
        target_column='Target',
        task_type='auto',
        demographic_column='Gender'  # If this column exists
    )
    
    Example 3: Custom scalability weights
    --------------------------------------
    results = comprehensive_ml_pipeline(
        dataset_path='budgetDataset.csv',
        target_column='total_budget',
        task_type='auto',
        readiness_sub_weights={
            'scalability': {'training': 0.6, 'inference': 0.4}
        }
    )
    
    ============================================================================
    NOTE: Make sure you have all helper functions defined in your notebook:
    - preprocess_data()
    - perform_fairness_analysis()
    - calculate_interpretability()
    - calculate_fairness_from_disparity()
    - calculate_scalability()
    - calculate_data_quality_from_dataset()
    - compute_data_quality()
    - compute_interpretability_for_readiness()
    - compute_fairness_for_readiness()
    - compute_dataset_readiness_score()
    - detect_and_transform_wide_format()
    - detect_text_columns()
    ============================================================================
    """)

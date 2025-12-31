"""
Modified Functions for notebook.ipynb
=====================================
Copy these modified functions into your notebook to implement the three changes:
1. Scalability calculation (training + inference only)
2. Hyperparameter tuning for models
3. Fairness analysis without synthetic groups
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Try to import psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


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
        
        # LogisticRegression doesn't do regression - skip or use different model
        # For regression, you might want to use LinearRegression or Ridge instead
        from sklearn.linear_model import Ridge
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


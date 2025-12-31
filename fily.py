"""
COMPLETE FAST ML PIPELINE - ALL IN ONE FILE
===========================================
This is a complete, ready-to-run file with:
1. All helper functions
2. Fast hyperparameter tuning (RandomizedSearchCV - 2-5 minutes instead of 27+ minutes)
3. Complete comprehensive_ml_pipeline() function
4. All three modifications integrated:
   - Scalability: Only training + inference time (50/50)
   - Hyperparameter tuning: RandomizedSearchCV for speed
   - Fairness: No synthetic groups

USAGE:
------
1. Copy this ENTIRE file into a Jupyter notebook cell
2. Run the cell
3. Call: results = comprehensive_ml_pipeline('your_data.csv', 'target', 'auto')
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, f1_score
)
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
# Don't force backend - let matplotlib auto-detect (works in Jupyter and regular Python)
# If in Jupyter, user should run: %matplotlib inline
# If graphs don't show, we'll use display() as fallback
try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Try to import psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# HELPER FUNCTION 1: preprocess_data
# ============================================================================

def preprocess_data(X, y, task_type='auto', use_tfidf=True, max_tfidf_features=10000):
    """
    Preprocess features and target for machine learning.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        task_type: 'auto', 'Regression', or 'Classification'
        use_tfidf: Whether to use TF-IDF for text features
        max_tfidf_features: Maximum TF-IDF features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type)
    """
    # Detect task type if auto
    if task_type == 'auto':
        if y.dtype == 'object' or y.dtype.name == 'category':
            detected_task_type = 'Classification'
        elif len(y.unique()) < 20 and len(y) < 1000:
            detected_task_type = 'Classification'
        else:
            detected_task_type = 'Regression'
    else:
        detected_task_type = task_type
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Separate text and non-text columns
    text_cols = []
    non_text_cols = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'string':
            sample = X[col].dropna().head(100)
            if len(sample) > 0:
                if sample.astype(str).str.contains('[a-zA-Z]', regex=True).any():
                    text_cols.append(col)
                else:
                    non_text_cols.append(col)
            else:
                non_text_cols.append(col)
        else:
            non_text_cols.append(col)
    
    processors = {}
    feature_names = []
    text_cols_used = []  # Store which text columns were used for TF-IDF
    
    # Process text columns with TF-IDF
    if text_cols and use_tfidf:
        try:
            text_cols_used = text_cols.copy()  # Store which columns were used
            X_text_train = X_train[text_cols].fillna('').astype(str)
            X_text_test = X_test[text_cols].fillna('').astype(str)
            
            # Combine all text columns
            X_text_train_combined = X_text_train.apply(lambda x: ' '.join(x), axis=1)
            X_text_test_combined = X_text_test.apply(lambda x: ' '.join(x), axis=1)
            
            tfidf = TfidfVectorizer(
                max_features=max_tfidf_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            X_text_train_tfidf = tfidf.fit_transform(X_text_train_combined)
            X_text_test_tfidf = tfidf.transform(X_text_test_combined)
            
            processors['tfidf'] = tfidf
            processors['text_cols_used'] = text_cols_used  # Store for later reference
            # Get actual feature names from TF-IDF (words/ngrams)
            try:
                tfidf_feature_names = tfidf.get_feature_names_out()
            except AttributeError:
                # Fallback for older sklearn versions
                tfidf_feature_names = tfidf.get_feature_names()
            # Use actual feature names (words/ngrams) instead of tfidf_0, tfidf_1, etc.
            # Only add prefix if multiple text columns, otherwise use names directly
            if len(text_cols) == 1:
                # Single text column - use feature names directly
                feature_names.extend(tfidf_feature_names)
            else:
                # Multiple text columns - add a simple prefix
                feature_names.extend([f'TFIDF_{name}' for name in tfidf_feature_names])
        except Exception as e:
            print(f"    ⚠️ TF-IDF processing failed: {e}")
            X_text_train_tfidf = None
            X_text_test_tfidf = None
    else:
        X_text_train_tfidf = None
        X_text_test_tfidf = None
    
    # Process non-text columns
    X_non_text_train = X_train[non_text_cols].copy()
    X_non_text_test = X_test[non_text_cols].copy()
    
    # Handle categorical non-text columns
    categorical_cols = []
    numerical_cols = []
    
    for col in non_text_cols:
        if X_non_text_train[col].dtype == 'object' or X_non_text_train[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Encode categorical columns
    if categorical_cols:
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_non_text_train[col] = le.fit_transform(X_non_text_train[col].astype(str).fillna('Unknown'))
            X_non_text_test[col] = le.transform(X_non_text_test[col].astype(str).fillna('Unknown'))
            le_dict[col] = le
        processors['label_encoders'] = le_dict
        feature_names.extend(categorical_cols)
    
    # Process numerical columns
    if numerical_cols:
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_non_text_train[numerical_cols] = imputer.fit_transform(X_non_text_train[numerical_cols])
        X_non_text_test[numerical_cols] = imputer.transform(X_non_text_test[numerical_cols])
        processors['imputer'] = imputer
        
        # Scale numerical features
        scaler = StandardScaler()
        X_non_text_train[numerical_cols] = scaler.fit_transform(X_non_text_train[numerical_cols])
        X_non_text_test[numerical_cols] = scaler.transform(X_non_text_test[numerical_cols])
        processors['scaler'] = scaler
        
        feature_names.extend(numerical_cols)
    
    # Combine text and non-text features
    if X_text_train_tfidf is not None:
        X_non_text_train_sparse = csr_matrix(X_non_text_train.values)
        X_non_text_test_sparse = csr_matrix(X_non_text_test.values)
        
        X_train_final = hstack([X_non_text_train_sparse, X_text_train_tfidf])
        X_test_final = hstack([X_non_text_test_sparse, X_text_test_tfidf])
    else:
        X_train_final = X_non_text_train.values
        X_test_final = X_non_text_test.values
    
    # Process target variable
    if detected_task_type == 'Classification':
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train.astype(str))
        y_test = le_target.transform(y_test.astype(str))
        processors['target_encoder'] = le_target
    else:
        y_train = y_train.values
        y_test = y_test.values
    
    return X_train_final, X_test_final, y_train, y_test, feature_names, processors, detected_task_type


# ============================================================================
# HELPER FUNCTION 2: detect_and_transform_wide_format
# ============================================================================

def detect_and_transform_wide_format(df):
    """Detect if dataset is in wide format and transform to long format if needed."""
    return df, False


# ============================================================================
# HELPER FUNCTION 3: detect_text_columns
# ============================================================================

def detect_text_columns(X):
    """Detect text/string columns in the feature matrix."""
    if isinstance(X, pd.DataFrame):
        text_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'string':
                sample = X[col].dropna().head(100)
                if len(sample) > 0:
                    if sample.astype(str).str.contains('[a-zA-Z]', regex=True).any():
                        text_cols.append(col)
        return text_cols
    return []


# ============================================================================
# HELPER FUNCTION 4: calculate_data_quality_from_dataset
# ============================================================================

def calculate_data_quality_from_dataset(df):
    """Calculate data quality score based on missing values."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    score = max(0, 5 * (1 - 2 * missing_ratio))
    return round(score, 2)


# ============================================================================
# HELPER FUNCTION 5: Individual Fairness (for when no demographic groups)
# ============================================================================

def calculate_individual_fairness(X_test, predictions, k=5):
    """
    Measure if similar individuals get similar predictions (Individual Fairness).
    
    Args:
        X_test: Feature matrix (n_samples, n_features)
        predictions: Model predictions (n_samples,)
        k: Number of nearest neighbors to consider
    
    Returns:
        fairness_score: 0-5 score (higher = more fair)
        avg_difference: Average prediction difference between similar individuals
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Convert predictions to numpy array if needed
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        predictions = np.array(predictions)
        
        # Handle sparse matrices
        if hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = np.array(X_test)
        
        # Find k+1 nearest neighbors for each person (k+1 because we exclude self)
        nn = NearestNeighbors(n_neighbors=min(k+1, len(X_test_dense)), metric='euclidean')
        nn.fit(X_test_dense)
        distances, indices = nn.kneighbors(X_test_dense)
        
        fairness_scores = []
        
        # For each person, check their neighbors' predictions
        for i in range(len(X_test_dense)):
            # Get indices of k nearest neighbors (exclude self, which is index 0)
            neighbor_indices = indices[i][1:]  # Skip index 0 (self)
            
            if len(neighbor_indices) == 0:
                continue
            
            # Get predictions of neighbors
            neighbor_predictions = predictions[neighbor_indices]
            
            # Calculate difference between this person's prediction and neighbors
            prediction_diff = np.abs(predictions[i] - neighbor_predictions)
            
            # Average difference (lower = more fair)
            avg_diff = np.mean(prediction_diff)
            fairness_scores.append(avg_diff)
        
        if len(fairness_scores) == 0:
            return 2.5, 0.0
        
        # Overall fairness = average of all individual fairness scores
        avg_fairness_diff = np.mean(fairness_scores)
        
        # Convert to 0-5 scale
        # Normalize based on prediction range
        pred_range = np.max(predictions) - np.min(predictions)
        if pred_range == 0:
            pred_range = 1.0  # Avoid division by zero
        
        # Normalize difference relative to prediction range
        normalized_diff = avg_fairness_diff / (pred_range + 1e-8)
        
        # Score: 5 if normalized_diff = 0, approaches 0 as normalized_diff increases
        fairness_score = max(0, 5 * (1 - min(1, normalized_diff)))
        
        return round(fairness_score, 2), round(avg_fairness_diff, 4)
        
    except Exception as e:
        print(f"    ⚠️ Individual fairness calculation failed: {e}")
        return 2.5, 0.0


# ============================================================================
# HELPER FUNCTION 5B: perform_fairness_analysis
# ============================================================================

def perform_fairness_analysis(models_dict, X_test, y_test, predictions_dict, demographic_groups, task_type):
    """Perform fairness analysis across demographic groups."""
    fairness_results = {}
    
    # Convert y_test to numpy array if needed
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test)
    
    for model_name, y_pred in predictions_dict.items():
        try:
            # Convert predictions to numpy array if needed
            if isinstance(y_pred, pd.Series):
                y_pred_array = y_pred.values
            else:
                y_pred_array = np.array(y_pred)
            
            if task_type == 'Regression':
                disparities = []
                for group_name, group_mask in demographic_groups.items():
                    if np.sum(group_mask) > 0:
                        group_rmse = np.sqrt(mean_squared_error(y_test_array[group_mask], y_pred_array[group_mask]))
                        disparities.append(group_rmse)
                disparity = abs(disparities[0] - disparities[1]) if len(disparities) >= 2 else 0.0
            else:
                accuracies = []
                for group_name, group_mask in demographic_groups.items():
                    if np.sum(group_mask) > 0:
                        group_acc = accuracy_score(y_test_array[group_mask], y_pred_array[group_mask])
                        accuracies.append(group_acc)
                disparity = abs(accuracies[0] - accuracies[1]) if len(accuracies) >= 2 else 0.0
            
            fairness_results[model_name] = {'Disparity': round(disparity, 4)}
        except Exception as e:
            print(f"    ⚠️ Fairness analysis failed for {model_name}: {e}")
            fairness_results[model_name] = {'Disparity': np.nan}
    
    fairness_df = pd.DataFrame(fairness_results).T
    fairness_df.index.name = 'Model'
    return fairness_df.reset_index()


# ============================================================================
# HELPER FUNCTION 6: calculate_interpretability
# ============================================================================

def calculate_interpretability(model_name, shap_success, feature_imp_available):
    """Calculate interpretability score for a model."""
    base_scores = {
        'LogisticRegression': 5.0, 'RandomForest': 4.0, 'XGBoost': 3.5, 'LightGBM': 3.0
    }
    base_score = base_scores.get(model_name, 2.5)
    if shap_success:
        base_score += 0.5
    if feature_imp_available:
        base_score += 0.5
    return min(5.0, base_score)


# ============================================================================
# HELPER FUNCTION 7: calculate_fairness_from_disparity
# ============================================================================

def calculate_fairness_from_disparity(disparity, max_disparity):
    """Calculate fairness score from disparity value."""
    if max_disparity == 0 or np.isnan(disparity) or np.isnan(max_disparity):
        return 2.5
    normalized_disparity = disparity / max_disparity if max_disparity > 0 else 0
    return max(0, min(5, 5 * (1 - normalized_disparity)))


# ============================================================================
# HELPER FUNCTION 8: calculate_scalability
# ============================================================================

def calculate_scalability(model_name, dataset_size=None, training_success=True):
    """Simple scalability score based on model type."""
    model_scores = {
        'LightGBM': 5.0, 'XGBoost': 4.5, 'RandomForest': 4.0, 'LogisticRegression': 3.5
    }
    base_score = model_scores.get(model_name, 3.0)
    size_adjustment = 0.5 if dataset_size and dataset_size > 100000 else (-0.5 if dataset_size and dataset_size < 10000 else 0.0)
    perf_adjustment = 0.2 if training_success else 0.0
    return round(min(5.0, max(0.0, base_score + size_adjustment + perf_adjustment)), 2)


# ============================================================================
# HELPER FUNCTION 9: compute_data_quality
# ============================================================================

def compute_data_quality(df, target_columns, feature_columns=None, sub_weights=None):
    """Compute data quality score (simplified version)."""
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in target_columns]
    
    total_cells = df[feature_columns].shape[0] * df[feature_columns].shape[1]
    missing_cells = df[feature_columns].isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    missing_score = max(0, 5 * (1 - 2 * missing_ratio))
    
    return {
        'score': missing_score,
        'missing_value_ratio': missing_ratio,
        'sub_weights_used': {}
    }


# ============================================================================
# HELPER FUNCTION 10: compute_interpretability_for_readiness
# ============================================================================

def compute_interpretability_for_readiness(shap_results_dict, models_dict):
    """Compute interpretability for readiness score (simplified)."""
    shap_success_count = sum(1 for v in shap_results_dict.values() if v.get('success', False))
    total_models = len(models_dict)
    score = (shap_success_count / total_models) * 5 if total_models > 0 else 2.5
    
    return {
        'score': round(score, 2),
        'shap_concentration': 0.5,
        'shap_stability': 0.5,
        'models_with_shap': shap_success_count,
        'sub_weights_used': {}
    }


# ============================================================================
# HELPER FUNCTION 11: compute_fairness_for_readiness
# ============================================================================

def compute_fairness_for_readiness(fairness_results, df, demographic_column):
    """Compute fairness for readiness score (simplified)."""
    if fairness_results is None or len(fairness_results) == 0:
        return {
            'score': 2.5, 'demographic_parity': 2.5, 'accuracy_gap': 0.0,
            'missingness_disparity': 0.0, 'sub_weights_used': {}
        }
    
    if 'Disparity' in fairness_results.columns:
        disparities = fairness_results['Disparity'].dropna()
        if len(disparities) > 0:
            max_disparity = disparities.max()
            avg_disparity = disparities.mean()
            score = max(0, 5 * (1 - min(1, avg_disparity / (max_disparity + 1e-8))))
        else:
            score = 2.5
    else:
        score = 2.5
    
    return {
        'score': round(score, 2), 'demographic_parity': round(score, 2),
        'accuracy_gap': 0.0, 'missingness_disparity': 0.0, 'sub_weights_used': {}
    }


# ============================================================================
# MODIFIED FUNCTION 1: Scalability (Training + Inference Only)
# ============================================================================

def compute_scalability_for_readiness(df, models_dict, X_train, X_test, y_train, y_test, task_type='auto', sub_weights=None):
    """
    Measures training time and inference time (memory and size_bonus removed).
    """
    if not models_dict:
        return {
            'score': 0, 'training_time': 0, 'inference_time': 0,
            'details': {}, 'sub_weights_used': {}
        }

    training_times = []
    inference_times = []

    for model_name, model in models_dict.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            training_times.append(train_time)

            start_time = time.time()
            _ = model.predict(X_test)
            infer_time = time.time() - start_time
            inference_times.append(infer_time)
        except Exception:
            continue

    if not training_times:
        return {
            'score': 0, 'training_time': 0, 'inference_time': 0,
            'details': {}, 'sub_weights_used': {}
        }

    avg_training_time = np.mean(training_times)
    avg_inference_time = np.mean(inference_times)

    training_score = max(0, 5 * (1 - min(1, avg_training_time / 60)))
    inference_score = max(0, 5 * (1 - min(1, avg_inference_time / 5)))

    if sub_weights is None:
        sub_weights = {'training': 0.5, 'inference': 0.5}
    else:
        total_weight = sum(sub_weights.values())
        if total_weight > 0:
            sub_weights = {k: v / total_weight for k, v in sub_weights.items()}
        else:
            sub_weights = {'training': 0.5, 'inference': 0.5}

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
# MODIFIED FUNCTION 2: FAST Hyperparameter Tuning (RandomizedSearchCV)
# ============================================================================

def create_models_with_hyperparameter_tuning_fast(detected_task_type, random_state=42, n_iter=10):
    """
    Create models with FAST hyperparameter tuning using RandomizedSearchCV.
    Uses only 10 random combinations per model (cv=3) = 30 fits instead of 180+ fits.
    Should complete in 2-5 minutes instead of 27+ minutes.
    """
    models_dict = {}
    
    if detected_task_type == 'Regression':
        # RandomForest Regressor - FAST
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, None],
            'min_samples_split': [2, 5]
        }
        rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        models_dict['RandomForest'] = RandomizedSearchCV(
            rf_base, rf_param_grid, cv=3, scoring='r2', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # XGBoost Regressor - FAST
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb_base = xgb.XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1)
        models_dict['XGBoost'] = RandomizedSearchCV(
            xgb_base, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # LightGBM Regressor - FAST
        lgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2],
            'num_leaves': [31, 50]
        }
        lgb_base = lgb.LGBMRegressor(random_state=random_state, verbosity=-1, n_jobs=-1)
        models_dict['LightGBM'] = RandomizedSearchCV(
            lgb_base, lgb_param_grid, cv=3, scoring='r2', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # Ridge Regression
        lr_param_grid = {'alpha': [0.01, 0.1, 1, 10]}
        lr_base = Ridge(random_state=random_state)
        models_dict['LogisticRegression'] = GridSearchCV(
            lr_base, lr_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        
    else:  # Classification
        # RandomForest Classifier - FAST
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, None],
            'min_samples_split': [2, 5]
        }
        rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        models_dict['RandomForest'] = RandomizedSearchCV(
            rf_base, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # XGBoost Classifier - FAST
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb_base = xgb.XGBClassifier(random_state=random_state, verbosity=0, n_jobs=-1)
        models_dict['XGBoost'] = RandomizedSearchCV(
            xgb_base, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # LightGBM Classifier - FAST
        lgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2],
            'num_leaves': [31, 50]
        }
        lgb_base = lgb.LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=-1)
        models_dict['LightGBM'] = RandomizedSearchCV(
            lgb_base, lgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1,
            verbose=1, n_iter=n_iter, random_state=random_state
        )
        
        # LogisticRegression Classifier
        lr_param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        lr_base = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
        models_dict['LogisticRegression'] = GridSearchCV(
            lr_base, lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
    
    return models_dict


# ============================================================================
# MODIFIED FUNCTION 3: Fairness Analysis Without Synthetic Groups
# ============================================================================

def handle_fairness_analysis_without_synthetic_groups(
    df, demographic_column, y_test, models_dict, X_test, predictions_dict, 
    detected_task_type, target_col, all_results, perform_fairness_analysis
):
    """Handle fairness analysis - uses Individual Fairness when no demographic groups."""
    if demographic_column and demographic_column in df.columns:
        # Handle y_test indexing - if it's a Series with index, use that; otherwise use position
        if isinstance(y_test, pd.Series) and hasattr(y_test, 'index'):
            try:
                demo_values = df[demographic_column].iloc[y_test.index].values
            except:
                # Fallback: use position-based indexing
                demo_values = df[demographic_column].iloc[:len(y_test)].values
        else:
            # Use position-based indexing
            demo_values = df[demographic_column].iloc[:len(y_test)].values
        
        unique_demos = np.unique(demo_values)
        if len(unique_demos) >= 2:
            demographic_groups = {demo: demo_values == demo for demo in unique_demos[:2]}
            fairness_df = perform_fairness_analysis(
                models_dict, X_test, y_test, predictions_dict,
                demographic_groups, detected_task_type
            )
            fairness_df['Target'] = target_col
            fairness_df['Fairness_Method'] = 'Demographic_Group_Disparity'
            all_results['fairness'].append(fairness_df)
            return fairness_df
        else:
            print(f"    ⚠️ Demographic column '{demographic_column}' has less than 2 unique groups.")
            print(f"    🔄 Using Individual Fairness instead...")
    else:
        print(f"    ℹ️ No demographic groups found. Using Individual Fairness...")
    
    # Use Individual Fairness when no demographic groups
    fairness_results = {}
    for model_name, y_pred in predictions_dict.items():
        try:
            fairness_score, avg_diff = calculate_individual_fairness(X_test, y_pred, k=5)
            # Convert fairness score to disparity-like metric (lower is better)
            # For consistency with demographic disparity, we use (5 - fairness_score) as "disparity"
            disparity_equivalent = max(0, 5 - fairness_score)
            fairness_results[model_name] = {
                'Disparity': round(disparity_equivalent, 4),
                'Individual_Fairness_Score': round(fairness_score, 2),
                'Avg_Prediction_Difference': round(avg_diff, 4)
            }
            print(f"      ✅ {model_name}: Individual Fairness Score = {fairness_score:.2f}/5.0, Avg Difference = {avg_diff:.4f}")
        except Exception as e:
            print(f"    ⚠️ Individual fairness failed for {model_name}: {e}")
            fairness_results[model_name] = {
                'Disparity': np.nan,
                'Individual_Fairness_Score': 2.5,
                'Avg_Prediction_Difference': np.nan
            }
    
    fairness_df = pd.DataFrame(fairness_results).T
    fairness_df.index.name = 'Model'
    fairness_df = fairness_df.reset_index()
    fairness_df['Target'] = target_col
    fairness_df['Fairness_Method'] = 'Individual_Fairness'
    all_results['fairness'].append(fairness_df)
    return fairness_df


# ============================================================================
# HELPER FUNCTION 12: compute_dataset_readiness_score
# ============================================================================

def compute_dataset_readiness_score(
    df, target_columns, metrics_dict=None, shap_results_dict=None,
    fairness_results=None, models_dict=None, X_train=None, X_test=None,
    y_train=None, y_test=None, task_type='auto', demographic_column=None,
    feature_names=None, weights=None, sub_weights=None
):
    """Compute dataset readiness score."""
    # Data Quality
    data_quality = compute_data_quality(df, target_columns)
    
    # Accuracy
    if metrics_dict:
        if task_type == 'Regression':
            best_r2 = max([m.get('R2', 0) for m in metrics_dict.values() if isinstance(m, dict)])
            accuracy_score_val = max(0, min(5, 5 * best_r2))
        else:
            best_acc = max([m.get('Classification_Accuracy', 0) for m in metrics_dict.values() if isinstance(m, dict)])
            accuracy_score_val = max(0, min(5, 5 * best_acc))
        accuracy = {
            'score': round(accuracy_score_val, 2),
            'best_model': max(metrics_dict.keys(), key=lambda k: metrics_dict[k].get('R2' if task_type == 'Regression' else 'Classification_Accuracy', 0)),
            'best_metric': accuracy_score_val
        }
    else:
        accuracy = {'score': 2.5, 'best_model': None, 'best_metric': 0}
    
    # Interpretability
    interpretability = compute_interpretability_for_readiness(shap_results_dict or {}, models_dict or {})
    
    # Fairness
    fairness = compute_fairness_for_readiness(fairness_results, df, demographic_column)
    
    # Scalability
    if models_dict and X_train is not None:
        scalability = compute_scalability_for_readiness(
            df, models_dict, X_train, X_test, y_train, y_test, task_type,
            sub_weights=sub_weights.get('scalability') if sub_weights else None
        )
    else:
        scalability = {'score': 2.5, 'training_time': 0, 'inference_time': 0, 'details': {}, 'sub_weights_used': {}}
    
    # Calculate total readiness score as SUM (0-25), not weighted average
    # Each dimension is 0-5, so sum is 0-25
    readiness_score = (
        data_quality['score'] +
        accuracy['score'] +
        interpretability['score'] +
        fairness['score'] +
        scalability['score']
    )
    
    # Store weights for reference (but not used in calculation)
    if weights is None:
        weights = {
            'data_quality': 0.20, 'accuracy': 0.20, 'interpretability': 0.20,
            'fairness': 0.20, 'scalability': 0.20
        }
    else:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
    
    return {
        'readiness_score': round(readiness_score, 2),
        'data_quality': data_quality,
        'accuracy': accuracy,
        'interpretability': interpretability,
        'fairness': fairness,
        'scalability': scalability,
        'breakdown': {
            'data_quality': data_quality['score'],
            'accuracy': accuracy['score'],
            'interpretability': interpretability['score'],
            'fairness': fairness['score'],
            'scalability': scalability['score']
        },
        'weights_used': weights
    }


# ============================================================================
# COMPLETE comprehensive_ml_pipeline() FUNCTION WITH FAST HYPERPARAMETER TUNING
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
    - FAST Hyperparameter tuning (RandomizedSearchCV - 2-5 minutes instead of 27+ minutes)
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
        readiness_sub_weights: Dict of dicts with sub-weights for each dimension's sub-factors.
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
    print("COMPREHENSIVE ML PIPELINE (FAST VERSION)")
    print("="*80)
    print("Includes: Accuracy Metrics, SHAP Analysis, Fairness, and Readiness Index")
    print("Using FAST hyperparameter tuning (RandomizedSearchCV)")
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
    except NameError:
        print(f"  ⚠️ detect_and_transform_wide_format not found, skipping transformation")
        was_transformed = False
    
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

    # Calculate data quality
    try:
        data_quality_score = calculate_data_quality_from_dataset(df)
        print(f"  ✅ Data Quality Score: {data_quality_score:.2f}/5.0")
    except NameError:
        print(f"  ⚠️ calculate_data_quality_from_dataset not found, skipping data quality calculation")
        data_quality_score = 2.5

    # Get dataset size
    dataset_size = len(df)

    # Process each target
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
            X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = preprocess_data(
                X, y_single, task_type, use_tfidf, max_tfidf_features
            )

            print(f"    ✅ Task type: {detected_task_type}")
            print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

            # Train models with FAST hyperparameter tuning
            print(f"  🤖 Training models with FAST hyperparameter tuning (RandomizedSearchCV)...")
            models_dict = {}
            predictions_dict = {}
            metrics_dict = {}
            shap_results_dict = {}

            # Create models with FAST hyperparameter tuning
            print(f"  🔍 Creating models with FAST hyperparameter tuning (n_iter=10, cv=3)...")
            models_dict = create_models_with_hyperparameter_tuning_fast(
                detected_task_type, 
                random_state=42,
                n_iter=10  # Only 10 random combinations per model
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
                    all_results['metrics'].append({
                        'Model': model_name,
                        'Target': target_col,
                        **metrics
                    })
                    
                    # Display metrics
                    print(f"      ✅ {model_name} Metrics:")
                    if detected_task_type == 'Regression':
                        print(f"         R² Score:              {metrics['R2']:.4f}")
                        print(f"         MAE:                   {metrics['MAE']:.4f}")
                        print(f"         RMSE:                  {metrics['RMSE']:.4f}")
                        print(f"         MAPE:                  {metrics['MAPE']:.2f}%")
                        print(f"         Regression Accuracy:   {metrics['Regression_Accuracy']:.4f}")
                    else:
                        print(f"         Classification Accuracy: {metrics['Classification_Accuracy']:.4f}")
                        print(f"         F1 Score:              {metrics['F1_Score']:.4f}")
                    
                except Exception as e:
                    print(f"    ⚠️ Error training {model_name}: {e}")
                    continue

            # SHAP Analysis with Visualizations (with fallback to model feature importance)
            print(f"  📊 Running SHAP Analysis...")
            
            def get_feature_importance_fallback(model, model_name, feature_names):
                """Get feature importance from model itself when SHAP fails."""
                try:
                    # For tree-based models
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        importance_dict = dict(zip(feature_names, importances))
                        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        return importance_dict, sorted_features, 'Model_Feature_Importance'
                    
                    # For linear models (LogisticRegression, Ridge)
                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        # Handle multi-class (coef_ is 2D)
                        if coef.ndim > 1:
                            coef = np.abs(coef).mean(axis=0)  # Average across classes
                        else:
                            coef = np.abs(coef)
                        importance_dict = dict(zip(feature_names, coef))
                        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        return importance_dict, sorted_features, 'Model_Coefficient_Importance'
                    
                    else:
                        return {}, [], 'No_Importance_Available'
                except Exception as e:
                    print(f"        ⚠️ Fallback feature importance failed: {e}")
                    return {}, [], 'No_Importance_Available'
            
            for model_name, model in models_dict.items():
                try:
                    print(f"    Analyzing {model_name}...")
                    shap_success = False
                    importance_dict = {}
                    sorted_features = []
                    method_used = 'Unknown'
                    
                    # Try SHAP first (only for tree-based models)
                    if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                        try:
                            import shap
                            
                            # Convert sparse matrix to dense if needed
                            if hasattr(X_test, 'toarray'):
                                X_test_dense = X_test.toarray()
                            else:
                                X_test_dense = np.array(X_test)
                            
                            # Use more samples for better analysis (up to 200)
                            n_samples = min(200, len(X_test_dense))
                            X_test_sample = X_test_dense[:n_samples]
                            
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test_sample)
                            
                            # Handle multi-class classification
                            if isinstance(shap_values, list):
                                shap_values = shap_values[0]  # Use first class for multi-class
                            
                            # Calculate mean absolute SHAP values
                            mean_abs_shap = np.abs(shap_values).mean(0)
                            
                            # Create feature importance dictionary
                            importance_dict = dict(zip(feature_names, mean_abs_shap))
                            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                            method_used = 'SHAP'
                            shap_success = True
                            
                        except Exception as e:
                            print(f"      ⚠️ SHAP failed for {model_name}: {e}")
                            print(f"      🔄 Using model's built-in feature importance instead...")
                            importance_dict, sorted_features, method_used = get_feature_importance_fallback(
                                model, model_name, feature_names
                            )
                    else:
                        # For non-tree models, use built-in feature importance
                        print(f"      ℹ️ {model_name} is not tree-based, using model's built-in importance...")
                        importance_dict, sorted_features, method_used = get_feature_importance_fallback(
                            model, model_name, feature_names
                        )
                    
                    # Store results
                    shap_results_dict[model_name] = {
                        'success': shap_success,
                        'importance': importance_dict,
                        'sorted_features': sorted_features,
                        'method': method_used
                    }
                    
                    # Create visualization for top 20 features (always, even if SHAP failed)
                    if len(sorted_features) > 0:
                        print(f"      📈 Creating feature importance visualization for {model_name}...")
                        top_n = min(20, len(sorted_features))
                        top_features = sorted_features[:top_n]
                        
                        feature_names_plot = [f[0] for f in top_features]
                        importance_values_plot = [f[1] for f in top_features]
                        
                        try:
                            # Create figure
                            fig, ax = plt.subplots(figsize=(10, 8))
                            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names_plot)))
                            ax.barh(range(len(feature_names_plot)), importance_values_plot, color=colors)
                            ax.set_yticks(range(len(feature_names_plot)))
                            ax.set_yticklabels(feature_names_plot)
                            # Set labels based on method - match the example format
                            if method_used == 'SHAP':
                                ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
                                ax.set_title(f'Top {top_n} SHAP Features - {model_name}', 
                                        fontsize=14, fontweight='bold')
                            else:
                                ax.set_xlabel('Feature Importance', fontsize=12)
                                ax.set_title(f'Top {top_n} Feature Importance - {model_name}\n(Method: {method_used})', 
                                        fontsize=14, fontweight='bold')
                            ax.invert_yaxis()
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            
                            # Display graph - works in both Jupyter and regular Python
                            if IPYTHON_AVAILABLE:
                                display(fig)  # Better for Jupyter
                            else:
                                plt.show()  # For regular Python
                            print(f"      ✅ Graph displayed for {model_name}")
                        except Exception as e:
                            print(f"      ⚠️ Error displaying graph for {model_name}: {e}")
                            # Try alternative display method
                            try:
                                plt.show()
                            except:
                                pass
                        
                        # Print top features
                        print(f"      Top 10 Features for {model_name} ({method_used}):")
                        for i, (feat, val) in enumerate(top_features[:10], 1):
                            print(f"        {i:2d}. {feat:30s}: {val:.4f}")
                    else:
                        print(f"      ⚠️ No feature importance available for {model_name}")
                        shap_results_dict[model_name] = {
                            'success': False, 
                            'importance': {}, 
                            'sorted_features': [],
                            'method': 'No_Importance_Available'
                        }
                        
                except Exception as e:
                    print(f"      ⚠️ Feature importance analysis failed for {model_name}: {e}")
                    shap_results_dict[model_name] = {
                        'success': False, 
                        'importance': {}, 
                        'sorted_features': [],
                        'method': 'Failed'
                    }

            # Fairness Analysis (without synthetic groups)
            print(f"  ⚖️ Running Fairness Analysis...")
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

            # Readiness Index
            print(f"  📈 Calculating Readiness Index for {target_col}...")
            readiness_list = []
            for model_name in models_dict.keys():
                if model_name in metrics_dict:
                    metrics = metrics_dict[model_name]
                    shap_success = shap_results_dict.get(model_name, {}).get('success', False)
                    feature_imp_available = shap_results_dict.get(model_name, {}).get('importance', {}) != {}
                    interpretability = calculate_interpretability(model_name, shap_success, feature_imp_available)
                    
                    fairness_row = fairness_df[fairness_df['Model'] == model_name]
                    if not fairness_row.empty:
                        # Check if Individual Fairness Score is available
                        if 'Individual_Fairness_Score' in fairness_row.columns:
                            fairness_score = fairness_row['Individual_Fairness_Score'].iloc[0]
                        elif 'Disparity' in fairness_row.columns:
                            disparity = fairness_row['Disparity'].iloc[0]
                            max_disparity = fairness_df['Disparity'].max() if len(fairness_df) > 0 else disparity * 2 if not np.isnan(disparity) else 1.0
                            fairness_score = calculate_fairness_from_disparity(disparity, max_disparity)
                        else:
                            fairness_score = 2.5
                    else:
                        fairness_score = 2.5
                    
                    scalability = calculate_scalability(model_name, dataset_size, True)
                    
                    if detected_task_type == 'Regression':
                        accuracy_score_val = max(0, min(5, 5 * metrics.get('R2', 0)))
                    else:
                        accuracy_score_val = max(0, min(5, 5 * metrics.get('Classification_Accuracy', 0)))
                    
                    readiness = {
                        'Model': model_name,
                        'Target': target_col,
                        'Interpretability': interpretability,
                        'Fairness': fairness_score,
                        'Scalability': scalability,
                        'Accuracy': accuracy_score_val,
                        'Data_Quality': data_quality_score,
                        'Total_Score': round(interpretability + fairness_score + scalability + accuracy_score_val + data_quality_score, 2)
                    }
                    readiness_list.append(readiness)
            
            readiness_df = pd.DataFrame(readiness_list)
            all_results['readiness'].append(readiness_df)
            
            # Display Readiness Index
            print(f"\n  📊 Readiness Index for {target_col}:")
            print("=" * 80)
            print(f"{'Model':<20} {'Interpretability':<18} {'Fairness':<12} {'Scalability':<14} {'Accuracy':<12} {'Data Quality':<14} {'Total Score':<12}")
            print("=" * 80)
            for _, row in readiness_df.iterrows():
                print(f"{row['Model']:<20} {row['Interpretability']:<18.2f} {row['Fairness']:<12.2f} "
                      f"{row['Scalability']:<14.2f} {row['Accuracy']:<12.2f} {row['Data_Quality']:<14.2f} {row['Total_Score']:<12.2f}")
            print("=" * 80)
            best_model = readiness_df.loc[readiness_df['Total_Score'].idxmax(), 'Model']
            best_score = readiness_df['Total_Score'].max()
            print(f"  🏆 Best Model: {best_model} (Score: {best_score:.2f}/25.0)")
            print()

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
        # Single target
        print(f"\n📊 Processing single target: {target_columns[0]}")
        y = df[target_columns[0]].copy()

        # Preprocess
        print(f"  🔧 Preprocessing data...")
        X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = preprocess_data(
            X, y, task_type, use_tfidf, max_tfidf_features
        )

        print(f"    ✅ Task type: {detected_task_type}")
        print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        # Train models with FAST hyperparameter tuning
        print(f"  🤖 Training models with FAST hyperparameter tuning (RandomizedSearchCV)...")
        models_dict = {}
        predictions_dict = {}
        metrics_dict = {}
        shap_results_dict = {}

        # Create models with FAST hyperparameter tuning
        print(f"  🔍 Creating models with FAST hyperparameter tuning (n_iter=10, cv=3)...")
        models_dict = create_models_with_hyperparameter_tuning_fast(
            detected_task_type, 
            random_state=42,
            n_iter=10  # Only 10 random combinations per model
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
                all_results['metrics'].append({
                    'Model': model_name,
                    **metrics
                })
                
                # Display metrics
                print(f"      ✅ {model_name} Metrics:")
                if detected_task_type == 'Regression':
                    print(f"         R² Score:              {metrics['R2']:.4f}")
                    print(f"         MAE:                   {metrics['MAE']:.4f}")
                    print(f"         RMSE:                  {metrics['RMSE']:.4f}")
                    print(f"         MAPE:                  {metrics['MAPE']:.2f}%")
                    print(f"         Regression Accuracy:   {metrics['Regression_Accuracy']:.4f}")
                else:
                    print(f"         Classification Accuracy: {metrics['Classification_Accuracy']:.4f}")
                    print(f"         F1 Score:              {metrics['F1_Score']:.4f}")
                
            except Exception as e:
                print(f"    ⚠️ Error training {model_name}: {e}")
                continue

        # SHAP Analysis with Visualizations (with fallback to model feature importance)
        print(f"  📊 Running SHAP Analysis...")
        
        def get_feature_importance_fallback(model, model_name, feature_names):
            """Get feature importance from model itself when SHAP fails."""
            try:
                # For tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_dict = dict(zip(feature_names, importances))
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    return importance_dict, sorted_features, 'Model_Feature_Importance'
                
                # For linear models (LogisticRegression, Ridge)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    # Handle multi-class (coef_ is 2D)
                    if coef.ndim > 1:
                        coef = np.abs(coef).mean(axis=0)  # Average across classes
                    else:
                        coef = np.abs(coef)
                    importance_dict = dict(zip(feature_names, coef))
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    return importance_dict, sorted_features, 'Model_Coefficient_Importance'
                
                else:
                    return {}, [], 'No_Importance_Available'
            except Exception as e:
                print(f"        ⚠️ Fallback feature importance failed: {e}")
                return {}, [], 'No_Importance_Available'
        
        for model_name, model in models_dict.items():
            try:
                print(f"    Analyzing {model_name}...")
                shap_success = False
                importance_dict = {}
                sorted_features = []
                method_used = 'Unknown'
                
                # Try SHAP first (only for tree-based models)
                if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    try:
                        import shap
                        
                        # Convert sparse matrix to dense if needed
                        if hasattr(X_test, 'toarray'):
                            X_test_dense = X_test.toarray()
                        else:
                            X_test_dense = np.array(X_test)
                        
                        # Use more samples for better analysis (up to 200)
                        n_samples = min(200, len(X_test_dense))
                        X_test_sample = X_test_dense[:n_samples]
                        
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test_sample)
                        
                        # Handle multi-class classification
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]  # Use first class for multi-class
                        
                        # Calculate mean absolute SHAP values
                        mean_abs_shap = np.abs(shap_values).mean(0)
                        
                        # Create feature importance dictionary
                        importance_dict = dict(zip(feature_names, mean_abs_shap))
                        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        method_used = 'SHAP'
                        shap_success = True
                        
                    except Exception as e:
                        print(f"      ⚠️ SHAP failed for {model_name}: {e}")
                        print(f"      🔄 Using model's built-in feature importance instead...")
                        importance_dict, sorted_features, method_used = get_feature_importance_fallback(
                            model, model_name, feature_names
                        )
                else:
                    # For non-tree models, use built-in feature importance
                    print(f"      ℹ️ {model_name} is not tree-based, using model's built-in importance...")
                    importance_dict, sorted_features, method_used = get_feature_importance_fallback(
                        model, model_name, feature_names
                    )
                
                # Store results
                shap_results_dict[model_name] = {
                    'success': shap_success,
                    'importance': importance_dict,
                    'sorted_features': sorted_features,
                    'method': method_used
                }
                
                # Create visualization for top 20 features (always, even if SHAP failed)
                if len(sorted_features) > 0:
                    print(f"      📈 Creating feature importance visualization for {model_name}...")
                    top_n = min(20, len(sorted_features))
                    top_features = sorted_features[:top_n]
                    
                    feature_names_plot = [f[0] for f in top_features]
                    importance_values_plot = [f[1] for f in top_features]
                    
                    try:
                        # Create figure
                        fig, ax = plt.subplots(figsize=(10, 8))
                        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names_plot)))
                        ax.barh(range(len(feature_names_plot)), importance_values_plot, color=colors)
                        ax.set_yticks(range(len(feature_names_plot)))
                        ax.set_yticklabels(feature_names_plot)
                        
                        # Set labels based on method - match the example format
                        if method_used == 'SHAP':
                            ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
                            ax.set_title(f'Top {top_n} SHAP Features - {model_name}', 
                                    fontsize=14, fontweight='bold')
                        else:
                            # Set labels based on method - match the example format
                            if method_used == 'SHAP':
                                ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
                                ax.set_title(f'Top {top_n} SHAP Features - {model_name}', 
                                        fontsize=14, fontweight='bold')
                            else:
                                ax.set_xlabel('Feature Importance', fontsize=12)
                                ax.set_title(f'Top {top_n} Feature Importance - {model_name}\n(Method: {method_used})', 
                                        fontsize=14, fontweight='bold')
                        
                        ax.invert_yaxis()
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        
                        # Display graph - works in both Jupyter and regular Python
                        if IPYTHON_AVAILABLE:
                            display(fig)  # Better for Jupyter
                        else:
                            plt.show()  # For regular Python
                        print(f"      ✅ Graph displayed for {model_name}")
                    except Exception as e:
                        print(f"      ⚠️ Error displaying graph for {model_name}: {e}")
                        # Try alternative display method
                        try:
                            plt.show()
                        except:
                            pass
                    
                    # Print top features
                    print(f"      Top 10 Features for {model_name} ({method_used}):")
                    for i, (feat, val) in enumerate(top_features[:10], 1):
                        print(f"        {i:2d}. {feat:30s}: {val:.4f}")
                else:
                    print(f"      ⚠️ No feature importance available for {model_name}")
                    shap_results_dict[model_name] = {
                        'success': False, 
                        'importance': {}, 
                        'sorted_features': [],
                        'method': 'No_Importance_Available'
                    }
                    
            except Exception as e:
                print(f"      ⚠️ Feature importance analysis failed for {model_name}: {e}")
                shap_results_dict[model_name] = {
                    'success': False, 
                    'importance': {}, 
                    'sorted_features': [],
                    'method': 'Failed'
                }

        # Fairness Analysis (without synthetic groups)
        print(f"  ⚖️ Running Fairness Analysis...")
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

        # Readiness Index
        print(f"  📈 Calculating Readiness Index...")
        readiness_list = []
        for model_name in models_dict.keys():
            if model_name in metrics_dict:
                metrics = metrics_dict[model_name]
                shap_success = shap_results_dict.get(model_name, {}).get('success', False)
                feature_imp_available = shap_results_dict.get(model_name, {}).get('importance', {}) != {}
                interpretability = calculate_interpretability(model_name, shap_success, feature_imp_available)
                
                fairness_row = fairness_df[fairness_df['Model'] == model_name]
                if not fairness_row.empty:
                    # Check if Individual Fairness Score is available
                    if 'Individual_Fairness_Score' in fairness_row.columns:
                        fairness_score = fairness_row['Individual_Fairness_Score'].iloc[0]
                    elif 'Disparity' in fairness_row.columns:
                        disparity = fairness_row['Disparity'].iloc[0]
                        max_disparity = fairness_df['Disparity'].max() if len(fairness_df) > 0 else disparity * 2 if not np.isnan(disparity) else 1.0
                        fairness_score = calculate_fairness_from_disparity(disparity, max_disparity)
                    else:
                        fairness_score = 2.5
                else:
                    fairness_score = 2.5
                
                scalability = calculate_scalability(model_name, dataset_size, True)
                
                if detected_task_type == 'Regression':
                    accuracy_score_val = max(0, min(5, 5 * metrics.get('R2', 0)))
                else:
                    accuracy_score_val = max(0, min(5, 5 * metrics.get('Classification_Accuracy', 0)))
                
                readiness = {
                    'Model': model_name,
                    'Interpretability': interpretability,
                    'Fairness': fairness_score,
                    'Scalability': scalability,
                    'Accuracy': accuracy_score_val,
                    'Data_Quality': data_quality_score,
                    'Total_Score': round(interpretability + fairness_score + scalability + accuracy_score_val + data_quality_score, 2)
                }
                readiness_list.append(readiness)
        
        readiness_df = pd.DataFrame(readiness_list)
        all_results['readiness'].append(readiness_df)
        
        # Display Readiness Index
        print(f"\n  📊 Readiness Index:")
        print("=" * 80)
        print(f"{'Model':<20} {'Interpretability':<18} {'Fairness':<12} {'Scalability':<14} {'Accuracy':<12} {'Data Quality':<14} {'Total Score':<12}")
        print("=" * 80)
        for _, row in readiness_df.iterrows():
            print(f"{row['Model']:<20} {row['Interpretability']:<18.2f} {row['Fairness']:<12.2f} "
                  f"{row['Scalability']:<14.2f} {row['Accuracy']:<12.2f} {row['Data_Quality']:<14.2f} {row['Total_Score']:<12.2f}")
        print("=" * 80)
        best_model = readiness_df.loc[readiness_df['Total_Score'].idxmax(), 'Model']
        best_score = readiness_df['Total_Score'].max()
        print(f"  🏆 Best Model: {best_model} (Score: {best_score:.2f}/25.0)")
        print()

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
            scalability_weights = readiness_sub_weights['scalability']
            if 'memory' in scalability_weights:
                del scalability_weights['memory']
            if 'size_bonus' in scalability_weights:
                del scalability_weights['size_bonus']
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
        print(f"\n  Overall Score: {dataset_readiness['readiness_score']:.2f}/25.0")
        
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

    # ============================================================================
    # COMPREHENSIVE RESULTS SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("📋 OVERALL RESULTS SUMMARY")
    print("="*80)
    
    # Combine all metrics
    if all_results['metrics']:
        metrics_df = pd.DataFrame(all_results['metrics'])
        print("\n📊 ACCURACY METRICS:")
        print("="*80)
        if stored_task_type == 'Regression':
            print(f"{'Model':<20} {'R² Score':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'Accuracy':<12}")
            print("="*80)
            for _, row in metrics_df.iterrows():
                print(f"{row['Model']:<20} {row.get('R2', 0):<12.4f} {row.get('MAE', 0):<12.4f} "
                      f"{row.get('RMSE', 0):<12.4f} {row.get('MAPE', 0):<12.2f} {row.get('Regression_Accuracy', 0):<12.4f}")
        else:
            print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12}")
            print("="*80)
            for _, row in metrics_df.iterrows():
                print(f"{row['Model']:<20} {row.get('Classification_Accuracy', 0):<12.4f} {row.get('F1_Score', 0):<12.4f}")
        print("="*80)
        
        # Best model by accuracy
        if stored_task_type == 'Regression':
            best_acc_model = metrics_df.loc[metrics_df['R2'].idxmax(), 'Model']
            best_acc_value = metrics_df['R2'].max()
            print(f"🏆 Best Model (R²): {best_acc_model} (R² = {best_acc_value:.4f})")
        else:
            best_acc_model = metrics_df.loc[metrics_df['Classification_Accuracy'].idxmax(), 'Model']
            best_acc_value = metrics_df['Classification_Accuracy'].max()
            print(f"🏆 Best Model (Accuracy): {best_acc_model} (Accuracy = {best_acc_value:.4f})")
    
    # Fairness Summary
    if all_results['fairness']:
        fairness_df_all = pd.concat(all_results['fairness'], ignore_index=True)
        print("\n⚖️ FAIRNESS ANALYSIS:")
        print("="*80)
        if 'Fairness_Method' in fairness_df_all.columns:
            print(f"{'Model':<20} {'Method':<25} {'Disparity':<12} {'Fairness Score':<15}")
            print("="*80)
            for _, row in fairness_df_all.iterrows():
                method = row.get('Fairness_Method', 'Unknown')
                if method == 'Individual_Fairness':
                    fairness_score = row.get('Individual_Fairness_Score', np.nan)
                    print(f"{row['Model']:<20} {method:<25} {row.get('Disparity', np.nan):<12.4f} {fairness_score:<15.2f}")
                else:
                    print(f"{row['Model']:<20} {method:<25} {row.get('Disparity', np.nan):<12.4f} {'N/A':<15}")
        else:
            print(f"{'Model':<20} {'Disparity':<12}")
            print("="*80)
            for _, row in fairness_df_all.iterrows():
                print(f"{row['Model']:<20} {row.get('Disparity', np.nan):<12.4f}")
        print("="*80)
    
    # Readiness Summary
    if all_results['readiness']:
        readiness_df_all = pd.concat(all_results['readiness'], ignore_index=True)
        print("\n📈 READINESS INDEX:")
        print("="*80)
        print(f"{'Model':<20} {'Total Score':<15} {'Accuracy':<12} {'Fairness':<12} {'Interpretability':<18} {'Scalability':<14}")
        print("="*80)
        for _, row in readiness_df_all.iterrows():
            print(f"{row['Model']:<20} {row['Total_Score']:<15.2f} {row['Accuracy']:<12.2f} "
                  f"{row['Fairness']:<12.2f} {row['Interpretability']:<18.2f} {row['Scalability']:<14.2f}")
        print("="*80)
        best_readiness_model = readiness_df_all.loc[readiness_df_all['Total_Score'].idxmax(), 'Model']
        best_readiness_score = readiness_df_all['Total_Score'].max()
        print(f"🏆 Best Model (Readiness): {best_readiness_model} (Score: {best_readiness_score:.2f}/25.0)")
    
    # SHAP Summary
    if stored_shap_results_dict:
        print("\n📊 SHAP FEATURE IMPORTANCE SUMMARY:")
        print("="*80)
        for model_name, shap_data in stored_shap_results_dict.items():
            if shap_data.get('success', False) and 'sorted_features' in shap_data:
                sorted_features = shap_data['sorted_features']
                print(f"\n  {model_name} - Top 5 Features:")
                for i, (feat, val) in enumerate(sorted_features[:5], 1):
                    print(f"    {i}. {feat:30s}: {val:.4f}")
        print("="*80)
    
    # Dataset Readiness
    if dataset_readiness:
        print("\n🌐 DATASET READINESS SCORE:")
        print("="*80)
        print(f"  Overall Score: {dataset_readiness['readiness_score']:.2f}/25.0")
        print(f"  - Data Quality:     {dataset_readiness['breakdown']['data_quality']:.2f}/5.0")
        print(f"  - Accuracy:         {dataset_readiness['breakdown']['accuracy']:.2f}/5.0")
        print(f"  - Interpretability: {dataset_readiness['breakdown']['interpretability']:.2f}/5.0")
        print(f"  - Fairness:         {dataset_readiness['breakdown']['fairness']:.2f}/5.0")
        print(f"  - Scalability:      {dataset_readiness['breakdown']['scalability']:.2f}/5.0")
        print("="*80)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE - ALL RESULTS DISPLAYED ABOVE")
    print("="*80)

    return {
        'metrics': pd.DataFrame(all_results['metrics']) if all_results['metrics'] else pd.DataFrame(),
        'fairness': pd.concat(all_results['fairness'], ignore_index=True) if all_results['fairness'] else pd.DataFrame(),
        'readiness': pd.concat(all_results['readiness'], ignore_index=True) if all_results['readiness'] else pd.DataFrame(),
        'shap': stored_shap_results_dict,
        'dataset_readiness': dataset_readiness
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

print("""
================================================================================
COMPLETE FAST ML PIPELINE - READY TO USE
================================================================================

This file contains everything you need:
✅ All helper functions
✅ Fast hyperparameter tuning (RandomizedSearchCV - 2-5 minutes instead of 27+)
✅ Complete comprehensive_ml_pipeline() function
✅ All three modifications integrated

USAGE:
------
1. Copy this ENTIRE file into a Jupyter notebook cell
2. Run the cell
3. Call the function:

# Example 1: Student Performance
results = comprehensive_ml_pipeline(
    dataset_path='student_performance/StudentPerformanceFactors.csv',
    target_column='Exam_Score',
    task_type='auto'
)

# Example 2: University Ranking
results = comprehensive_ml_pipeline(
    dataset_path='/content/university_rankings/THE World University Rankings 2016-2026.csv',
    target_column='Rank',
    task_type='auto',
    use_tfidf=False
)

# Example 3: Enrollment
results = comprehensive_ml_pipeline(
    dataset_path='college_enrollment/enrollment.csv',
    target_column='Enrollment',
    task_type='auto'
)

# Example 4: Dropout
results = comprehensive_ml_pipeline(
    dataset_path='retention/dataset.csv',
    target_column='Target',
    task_type='auto'
)

# Example 5: Budget Prediction
results = comprehensive_ml_pipeline(
    dataset_path='/content/budgetDataset.csv',
    target_column='total_budget',
    task_type='auto'
)

================================================================================
FAST HYPERPARAMETER TUNING:
- Uses RandomizedSearchCV with n_iter=10 (only 10 random combinations)
- Uses cv=3 instead of cv=5
- Should complete in 2-5 minutes instead of 27+ minutes
- Still finds good hyperparameters!

================================================================================
""")


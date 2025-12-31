"""
ALL REQUIRED HELPER FUNCTIONS FOR comprehensive_ml_pipeline()
==============================================================
Copy all these functions into your notebook BEFORE calling comprehensive_ml_pipeline()
These are essential functions that the pipeline needs to work.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ESSENTIAL FUNCTION 1: preprocess_data
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
    
    # Process text columns with TF-IDF
    if text_cols and use_tfidf:
        try:
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
            feature_names.extend([f'tfidf_{i}' for i in range(X_text_train_tfidf.shape[1])])
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
        from scipy.sparse import csr_matrix
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
# ESSENTIAL FUNCTION 2: detect_and_transform_wide_format
# ============================================================================

def detect_and_transform_wide_format(df):
    """
    Detect if dataset is in wide format and transform to long format if needed.
    For now, returns dataset as-is.
    """
    return df, False


# ============================================================================
# ESSENTIAL FUNCTION 3: detect_text_columns
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
# ESSENTIAL FUNCTION 4: calculate_data_quality_from_dataset
# ============================================================================

def calculate_data_quality_from_dataset(df):
    """Calculate data quality score based on missing values."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    score = max(0, 5 * (1 - 2 * missing_ratio))
    return round(score, 2)


# ============================================================================
# ESSENTIAL FUNCTION 5: perform_fairness_analysis
# ============================================================================

def perform_fairness_analysis(models_dict, X_test, y_test, predictions_dict, demographic_groups, task_type):
    """
    Perform fairness analysis across demographic groups.
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        predictions_dict: Dictionary of predictions
        demographic_groups: Dictionary of demographic group masks
        task_type: 'Regression' or 'Classification'
        
    Returns:
        pd.DataFrame: Fairness metrics for each model
    """
    from sklearn.metrics import mean_squared_error, accuracy_score
    
    fairness_results = {}
    
    for model_name, y_pred in predictions_dict.items():
        try:
            if task_type == 'Regression':
                # For regression: use RMSE for each group
                group_metrics = {}
                disparities = []
                
                for group_name, group_mask in demographic_groups.items():
                    if np.sum(group_mask) > 0:
                        group_rmse = np.sqrt(mean_squared_error(
                            y_test[group_mask],
                            y_pred[group_mask]
                        ))
                        group_metrics[f'{group_name}_RMSE'] = round(group_rmse, 4)
                        disparities.append(group_rmse)
                
                # Calculate disparity (difference between groups)
                if len(disparities) >= 2:
                    disparity = abs(disparities[0] - disparities[1])
                else:
                    disparity = 0.0
                
                group_metrics['Disparity'] = round(disparity, 4)
                fairness_results[model_name] = group_metrics
                
            else:  # Classification
                # For classification: use accuracy for each group
                group_metrics = {}
                accuracies = []
                
                for group_name, group_mask in demographic_groups.items():
                    if np.sum(group_mask) > 0:
                        group_acc = accuracy_score(
                            y_test[group_mask],
                            y_pred[group_mask]
                        )
                        group_metrics[f'{group_name}_Accuracy'] = round(group_acc, 4)
                        accuracies.append(group_acc)
                
                # Calculate disparity
                if len(accuracies) >= 2:
                    disparity = abs(accuracies[0] - accuracies[1])
                else:
                    disparity = 0.0
                
                group_metrics['Disparity'] = round(disparity, 4)
                fairness_results[model_name] = group_metrics
                
        except Exception as e:
            print(f"    ⚠️ Fairness analysis failed for {model_name}: {e}")
            fairness_results[model_name] = {'Disparity': np.nan}
    
    fairness_df = pd.DataFrame(fairness_results).T
    fairness_df.index.name = 'Model'
    fairness_df = fairness_df.reset_index()
    
    return fairness_df


# ============================================================================
# ESSENTIAL FUNCTION 6: calculate_interpretability
# ============================================================================

def calculate_interpretability(model_name, shap_success, feature_imp_available):
    """Calculate interpretability score for a model."""
    base_scores = {
        'LogisticRegression': 5.0,
        'RandomForest': 4.0,
        'XGBoost': 3.5,
        'LightGBM': 3.0
    }
    base_score = base_scores.get(model_name, 2.5)
    if shap_success:
        base_score += 0.5
    if feature_imp_available:
        base_score += 0.5
    return min(5.0, base_score)


# ============================================================================
# ESSENTIAL FUNCTION 7: calculate_fairness_from_disparity
# ============================================================================

def calculate_fairness_from_disparity(disparity, max_disparity):
    """Calculate fairness score from disparity value."""
    if max_disparity == 0 or np.isnan(disparity) or np.isnan(max_disparity):
        return 2.5
    normalized_disparity = disparity / max_disparity if max_disparity > 0 else 0
    score = 5 * (1 - normalized_disparity)
    return max(0, min(5, score))


# ============================================================================
# ESSENTIAL FUNCTION 8: calculate_scalability
# ============================================================================

def calculate_scalability(model_name, dataset_size=None, training_success=True):
    """Simple scalability score based on model type."""
    model_scores = {
        'LightGBM': 5.0,
        'XGBoost': 4.5,
        'RandomForest': 4.0,
        'LogisticRegression': 3.5
    }
    base_score = model_scores.get(model_name, 3.0)
    size_adjustment = 0.5 if dataset_size and dataset_size > 100000 else (-0.5 if dataset_size and dataset_size < 10000 else 0.0)
    perf_adjustment = 0.2 if training_success else 0.0
    return round(min(5.0, max(0.0, base_score + size_adjustment + perf_adjustment)), 2)


# ============================================================================
# OPTIONAL FUNCTIONS (for advanced features)
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


def compute_fairness_for_readiness(fairness_results, df, demographic_column):
    """Compute fairness for readiness score (simplified)."""
    if fairness_results is None or len(fairness_results) == 0:
        return {
            'score': 2.5,
            'demographic_parity': 2.5,
            'accuracy_gap': 0.0,
            'missingness_disparity': 0.0,
            'sub_weights_used': {}
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
        'score': round(score, 2),
        'demographic_parity': round(score, 2),
        'accuracy_gap': 0.0,
        'missingness_disparity': 0.0,
        'sub_weights_used': {}
    }


def compute_dataset_readiness_score(
    df, target_columns, metrics_dict=None, shap_results_dict=None,
    fairness_results=None, models_dict=None, X_train=None, X_test=None,
    y_train=None, y_test=None, task_type='auto', demographic_column=None,
    feature_names=None, weights=None, sub_weights=None
):
    """
    Compute dataset readiness score (simplified version).
    This is a basic implementation - you may want to enhance it.
    """
    from modified_functions import compute_scalability_for_readiness
    
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
    
    # Default weights
    if weights is None:
        weights = {
            'data_quality': 0.20,
            'accuracy': 0.20,
            'interpretability': 0.20,
            'fairness': 0.20,
            'scalability': 0.20
        }
    else:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
    
    # Calculate total readiness score
    readiness_score = (
        weights['data_quality'] * data_quality['score'] +
        weights['accuracy'] * accuracy['score'] +
        weights['interpretability'] * interpretability['score'] +
        weights['fairness'] * fairness['score'] +
        weights['scalability'] * scalability['score']
    )
    
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
# USAGE INSTRUCTIONS
# ============================================================================

print("""
================================================================================
INSTRUCTIONS
================================================================================

1. Copy ALL the functions above into your notebook

2. Make sure you also have the modified functions from modified_functions.py:
   - compute_scalability_for_readiness()
   - create_models_with_hyperparameter_tuning()
   - handle_fairness_analysis_without_synthetic_groups()

3. Then you can call comprehensive_ml_pipeline() normally:

   results = comprehensive_ml_pipeline(
       dataset_path='your_data.csv',
       target_column='target',
       task_type='auto'
   )

================================================================================
""")


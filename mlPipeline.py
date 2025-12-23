"""
COMPREHENSIVE ML PIPELINE
=========================
Complete pipeline that extends generic_accuracy_calculator.py to include:
- Accuracy metrics calculation
- SHAP analysis with feature importance plots
- Demographic disparity and fairness analysis
- Educational AI Readiness Index calculation

This is a single function that does everything!
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from scipy.sparse import hstack, csr_matrix
from scipy.stats import entropy
import xgboost as xgb
import lightgbm as lgb

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Warning: psutil not available. Memory tracking will be limited.")

warnings.filterwarnings('ignore')

# Import functions from generic_accuracy_calculator
try:
    from generic_accuracy_calculator import (
        detect_and_transform_wide_format,
        detect_text_columns,
        detect_task_type,
        preprocess_data,
        calculate_all_metrics,
        train_and_evaluate_models
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import from generic_accuracy_calculator: {e}")
    print("   Attempting to import functions individually...")
    try:
        import generic_accuracy_calculator as gac
        detect_and_transform_wide_format = gac.detect_and_transform_wide_format
        detect_text_columns = gac.detect_text_columns
        detect_task_type = gac.detect_task_type
        preprocess_data = gac.preprocess_data
        calculate_all_metrics = gac.calculate_all_metrics
        train_and_evaluate_models = gac.train_and_evaluate_models
    except Exception as e2:
        raise ImportError(f"Could not import required functions from generic_accuracy_calculator: {e2}")


def calculate_interpretability(model_name, shap_success, feature_importance_available):
    """
    Calculate Interpretability Score (0-5) based on:
    1. Model type (linear models are more interpretable)
    2. SHAP availability and success
    3. Feature importance availability
    
    Returns: float (0-5)
    """
    # Base score by model type
    model_scores = {
        'LogisticRegression': 4.0,
        'RandomForest': 3.5,
        'XGBoost': 3.0,
        'LightGBM': 3.0
    }
    
    base_score = model_scores.get(model_name, 2.5)
    
    # SHAP bonus
    shap_bonus = 0.5 if shap_success else 0.0
    
    # Feature importance bonus
    feature_imp_bonus = 0.5 if feature_importance_available else 0.0
    
    total_score = min(5.0, base_score + shap_bonus + feature_imp_bonus)
    return round(total_score, 2)


def calculate_fairness_from_disparity(disparity, max_disparity=None):
    """
    Calculate Fairness Score (0-5) from disparity metrics.
    Lower disparity = Higher fairness score.
    
    Returns: float (0-5)
    """
    if pd.isna(disparity) or disparity is None:
        return 2.5  # Default
    
    if max_disparity is None or max_disparity == 0:
        max_disparity = disparity * 2 if disparity > 0 else 1.0
    
    # Normalize: lower disparity = higher score
    normalized_score = 5.0 * (1.0 - min(disparity / max_disparity, 1.0))
    return round(max(0.0, min(5.0, normalized_score)), 2)


def calculate_scalability(model_name, dataset_size=None, training_success=True):
    """
    Calculate Scalability Score (0-5) based on model type and dataset size.
    
    Returns: float (0-5)
    """
    model_scores = {
        'LightGBM': 5.0,
        'XGBoost': 4.5,
        'RandomForest': 4.0,
        'LogisticRegression': 3.5
    }
    
    base_score = model_scores.get(model_name, 3.0)
    
    # Dataset size adjustment
    size_adjustment = 0.0
    if dataset_size is not None:
        if dataset_size > 100000:
            size_adjustment = 0.5
        elif dataset_size < 10000:
            size_adjustment = -0.5
    
    # Performance-based adjustment
    perf_adjustment = 0.2 if training_success else 0.0
    
    total_score = min(5.0, base_score + size_adjustment + perf_adjustment)
    return round(max(0.0, total_score), 2)


def calculate_data_quality_from_dataset(df):
    """
    Calculate Data Quality Score (0-5) from dataset characteristics.
    
    Returns: float (0-5)
    """
    try:
        # Sample for efficiency
        df_sample = df.head(10000) if len(df) > 10000 else df
        
        # Missing values analysis
        total_cells = df_sample.shape[0] * df_sample.shape[1]
        missing_cells = df_sample.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        missing_score = 5.0 * (1.0 - min(missing_percentage / 50.0, 1.0))
        
        # Data completeness
        complete_rows = df_sample.dropna().shape[0]
        completeness_ratio = complete_rows / df_sample.shape[0] if df_sample.shape[0] > 0 else 0
        completeness_score = 5.0 * completeness_ratio
        
        # Feature diversity
        diversity_scores = []
        for col in df_sample.select_dtypes(include=[np.number]).columns:
            unique_ratio = df_sample[col].nunique() / len(df_sample) if len(df_sample) > 0 else 0
            if 0.3 <= unique_ratio <= 0.7:
                diversity_scores.append(5.0)
            elif unique_ratio < 0.1 or unique_ratio > 0.9:
                diversity_scores.append(2.0)
            else:
                diversity_scores.append(3.5)
        
        diversity_score = np.mean(diversity_scores) if diversity_scores else 3.0
        
        # Final score
        final_score = (missing_score + completeness_score + diversity_score) / 3.0
        return round(max(0.0, min(5.0, final_score)), 2)
    
    except Exception as e:
        print(f"  ⚠️ Error calculating data quality: {e}")
        return 3.0


def perform_shap_analysis(model, X_train, X_test, model_name, task_type, 
                         feature_names, shap_sample_size=500):
    """
    Perform SHAP analysis and display plots inline.
    
    Returns:
        bool: Whether SHAP succeeded
        dict: Feature importance dictionary
        pd.DataFrame: Feature importance dataframe
    """
    shap_success = False
    feature_importance_dict = {}
    importance_df = None
    
    try:
        print(f"    🧠 Computing SHAP values for {model_name}...")
        
        # Sample data for SHAP (to speed up computation)
        # Properly handle sparse matrices
        if hasattr(X_test, 'toarray'):
            # Sparse matrix - convert to dense numpy array
            X_test_sample = np.array(X_test[:shap_sample_size].toarray())
            X_train_sample = np.array(X_train[:min(100, shap_sample_size)].toarray())
        else:
            # Already dense - ensure it's a numpy array
            X_test_sample = np.array(X_test[:shap_sample_size])
            X_train_sample = np.array(X_train[:min(100, shap_sample_size)])
        
        # Ensure 2D arrays
        if X_test_sample.ndim == 1:
            X_test_sample = X_test_sample.reshape(1, -1)
        if X_train_sample.ndim == 1:
            X_train_sample = X_train_sample.reshape(1, -1)
        
        # Choose appropriate SHAP explainer
        X_test_shap_used = None  # Store the sample used for SHAP
        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            # Tree explainers can work with sparse matrices, but convert to dense for consistency
            explainer = shap.TreeExplainer(model)
            # Use a smaller sample for tree explainers to avoid memory issues
            sample_size = min(shap_sample_size, 100)
            if hasattr(X_test, 'toarray'):
                # Convert sparse to dense for SHAP
                X_test_shap_used = np.array(X_test[:sample_size].toarray())
            else:
                X_test_shap_used = X_test_sample[:sample_size]
            shap_values = explainer.shap_values(X_test_shap_used)
        else:  # LogisticRegression
            if task_type == 'Regression':
                background = X_train_sample[:50]
                explainer = shap.KernelExplainer(model.predict, background)
                X_test_shap_used = X_test_sample[:50]
                shap_values = explainer.shap_values(X_test_shap_used, nsamples=100)
            else:
                # For LogisticRegression with sparse data, use KernelExplainer
                # LinearExplainer has issues with sparse matrices
                try:
                    # Try LinearExplainer first (faster if it works)
                    background = X_train_sample[:50]
                    explainer = shap.LinearExplainer(model, background)
                    X_test_shap_used = X_test_sample[:min(50, len(X_test_sample))]
                    shap_values = explainer.shap_values(X_test_shap_used)
                except Exception as e:
                    # Fallback to KernelExplainer
                    background = X_train_sample[:50]
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    X_test_shap_used = X_test_sample[:min(50, len(X_test_sample))]
                    shap_values = explainer.shap_values(X_test_shap_used, nsamples=100)
        
        # Handle multi-class SHAP outputs
        # Convert to numpy array if needed (handle sparse outputs)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        elif hasattr(shap_values, 'toarray'):
            shap_values = shap_values.toarray()
        else:
            shap_values = np.array(shap_values)
        
        # Calculate mean absolute SHAP values
        if shap_values.ndim == 3:
            # Multi-class: (n_classes, n_samples, n_features)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
        elif shap_values.ndim == 2:
            # Binary or regression: (n_samples, n_features)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            # 1D array
            mean_abs_shap = np.abs(shap_values)
        
        # Ensure we have the right number of features
        if len(mean_abs_shap) > len(feature_names):
            mean_abs_shap = mean_abs_shap[:len(feature_names)]
        elif len(mean_abs_shap) < len(feature_names):
            mean_abs_shap = np.pad(mean_abs_shap, (0, len(feature_names) - len(mean_abs_shap)))
        
        # Create feature importance dictionary and dataframe
        feature_importance_dict = dict(zip(feature_names, mean_abs_shap))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Display feature importance table
        print(f"\n    📊 SHAP Feature Importance for {model_name}:")
        print(importance_df.head(20).to_string(index=False))
        
        # Bar plot - Top 20 features (display inline)
        plt.figure(figsize=(10, 6))
        top_df = importance_df.head(20)
        plt.barh(range(len(top_df)), top_df['mean_abs_shap'].values[::-1], 
                color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_df)), top_df['feature'].values[::-1])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Top 20 SHAP Features - {model_name}')
        plt.tight_layout()
        plt.show()
        
        # SHAP summary plot (display inline)
        try:
            if X_test_shap_used is not None:
                plt.figure(figsize=(10, 6))
                # Handle different shap_values formats
                if shap_values.ndim == 3:
                    # Multi-class: use first class or average
                    shap_values_plot = shap_values[0] if shap_values.shape[0] > 0 else np.mean(shap_values, axis=0)
                else:
                    shap_values_plot = shap_values
                
                # Ensure features match shap_values dimensions
                n_features_plot = shap_values_plot.shape[1] if shap_values_plot.ndim > 1 else len(shap_values_plot)
                n_samples_plot = min(len(X_test_shap_used), shap_values_plot.shape[0])
                features_plot = X_test_shap_used[:n_samples_plot]
                shap_values_plot = shap_values_plot[:n_samples_plot]
                feature_names_plot = feature_names[:n_features_plot] if n_features_plot <= len(feature_names) else feature_names
                
                shap.summary_plot(
                    shap_values_plot,
                    features=features_plot,
                    feature_names=feature_names_plot[:n_features_plot],
                    show=False,
                    max_display=20
                )
                plt.title(f'SHAP Summary Plot - {model_name}')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"      ⚠️ Summary plot failed: {e}")
        
        shap_success = True
        print(f"    ✅ SHAP analysis completed for {model_name}\n")
        
    except Exception as e:
        print(f"    ⚠️ SHAP failed for {model_name}: {e}")
        # Fallback to model feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) <= len(feature_names):
                feature_importance_dict = dict(zip(feature_names[:len(importances)], importances))
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],
                    'mean_abs_shap': importances
                }).sort_values('mean_abs_shap', ascending=False)
                shap_success = True
                print(f"    ✅ Using model feature importances as fallback\n")
        elif hasattr(model, "coef_"):
            coef_vals = np.abs(model.coef_).ravel()
            if len(coef_vals) <= len(feature_names):
                feature_importance_dict = dict(zip(feature_names[:len(coef_vals)], coef_vals))
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(coef_vals)],
                    'mean_abs_shap': coef_vals
                }).sort_values('mean_abs_shap', ascending=False)
                shap_success = True
                print(f"    ✅ Using model coefficients as fallback\n")
    
    return shap_success, feature_importance_dict, importance_df


def perform_fairness_analysis(models_dict, X_test, y_test, predictions_dict, 
                              demographic_groups, task_type):
    """
    Perform fairness analysis across demographic groups and display results.
    
    Returns:
        pd.DataFrame: Fairness metrics for each model
    """
    fairness_results = {}
    
    print(f"\n  ⚖️ Running Fairness Analysis...")
    
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
    
    # Display fairness results
    print(f"\n    📊 Fairness Analysis Results:")
    print(fairness_df.to_string(index=False))
    
    return fairness_df


def calculate_readiness_index(model_name, interpretability, fairness_score, 
                             scalability, accuracy_score, data_quality):
    """
    Calculate Educational AI Readiness Index.
    
    Returns: dict with all readiness dimensions
    """
    total_score = interpretability + fairness_score + scalability + accuracy_score + data_quality
    
    return {
        'Model': model_name,
        'Interpretability': interpretability,
        'Fairness': fairness_score,
        'Scalability': scalability,
        'Accuracy': accuracy_score,
        'Data_Quality': data_quality,
        'Total_Score': round(total_score, 2)
    }


# ============================================================================
# DATASET READINESS SCORE ALGORITHM
# ============================================================================

def compute_data_quality(df, target_columns, feature_columns=None):
    """
    Computes dataset-level data quality score (0-5) based on:
    1. Missing value ratio
    2. Class imbalance (for classification)
    3. Feature redundancy (correlation)
    4. Noise sensitivity (performance drop after noise injection)
    
    Args:
        df: DataFrame
        target_columns: List of target column names
        feature_columns: List of feature column names (if None, auto-detect)
    
    Returns:
        dict: {
            'score': float (0-5),
            'missing_value_ratio': float,
            'class_imbalance_score': float,
            'feature_redundancy_score': float,
            'noise_sensitivity_score': float,
            'details': dict
        }
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in target_columns]
    
    # 1. Missing Value Ratio (0-5 scale)
    total_cells = df[feature_columns].shape[0] * df[feature_columns].shape[1]
    missing_cells = df[feature_columns].isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    
    # Score: 5 if no missing, 0 if >50% missing
    missing_score = max(0, 5 * (1 - 2 * missing_ratio))
    
    # 2. Class Imbalance (for classification targets)
    class_imbalance_scores = []
    for target_col in target_columns:
        if target_col in df.columns:
            target_series = df[target_col].dropna()
            if target_series.dtype == 'object' or len(target_series.unique()) < 20:
                # Classification task
                value_counts = target_series.value_counts()
                if len(value_counts) > 1:
                    # Calculate entropy (higher = more balanced)
                    probs = value_counts / len(target_series)
                    ent = entropy(probs)
                    max_ent = np.log(len(value_counts))
                    balance_ratio = ent / max_ent if max_ent > 0 else 0
                    # Score: 5 if perfectly balanced, 0 if highly imbalanced
                    class_imbalance_score = 5 * balance_ratio
                else:
                    class_imbalance_score = 0  # Only one class
            else:
                # Regression task - check value distribution
                if len(target_series) > 0:
                    cv = target_series.std() / target_series.mean() if target_series.mean() != 0 else 0
                    # Higher CV = more spread = better for regression
                    class_imbalance_score = min(5, 2.5 + 2.5 * min(1, cv))
                else:
                    class_imbalance_score = 2.5
            class_imbalance_scores.append(class_imbalance_score)
    
    avg_class_imbalance_score = np.mean(class_imbalance_scores) if class_imbalance_scores else 2.5
    
    # 3. Feature Redundancy (correlation)
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr().abs()
        # Remove diagonal
        np.fill_diagonal(corr_matrix.values, 0)
        # Average correlation (higher = more redundant)
        avg_corr = corr_matrix.values[corr_matrix.values > 0].mean() if (corr_matrix.values > 0).any() else 0
        # Score: 5 if no redundancy, 0 if highly redundant
        redundancy_score = max(0, 5 * (1 - avg_corr))
    else:
        redundancy_score = 2.5  # Neutral if can't compute
    
    # 4. Noise Sensitivity
    # Add small random noise and measure performance drop
    try:
        # Simple heuristic: if dataset is too small or has too many missing, skip
        if len(df) < 50 or missing_ratio > 0.5:
            noise_sensitivity_score = 2.5
        else:
            # Use a simple model to test noise sensitivity
            test_target = target_columns[0]
            if test_target in df.columns:
                # Prepare data
                X = df[feature_columns].select_dtypes(include=[np.number]).fillna(0)
                y = df[test_target].dropna()
                
                if len(X) > 0 and len(y) > 0 and len(X) == len(y):
                    # Align indices
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(common_idx) >= 20:
                        # Determine task type
                        is_classification = (y.dtype == 'object' or len(y.unique()) < 20)
                        
                        # Baseline model
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=42
                        )
                        
                        if len(X_train) > 0 and len(X_test) > 0:
                            if is_classification:
                                le = LabelEncoder()
                                y_train_enc = le.fit_transform(y_train)
                                y_test_enc = le.transform(y_test)
                                model = RandomForestClassifier(n_estimators=50, random_state=42)
                                model.fit(X_train, y_train_enc)
                                baseline_pred = model.predict(X_test)
                                baseline_score = accuracy_score(y_test_enc, baseline_pred)
                            else:
                                model = RandomForestRegressor(n_estimators=50, random_state=42)
                                model.fit(X_train, y_train)
                                baseline_pred = model.predict(X_test)
                                baseline_score = r2_score(y_test, baseline_pred)
                            
                            # Add noise (5% of std)
                            X_test_noisy = X_test.copy()
                            for col in X_test_noisy.columns:
                                std_val = X_test_noisy[col].std()
                                if std_val > 0:
                                    noise = np.random.normal(0, 0.05 * std_val, size=len(X_test_noisy))
                                    X_test_noisy[col] = X_test_noisy[col] + noise
                            
                            # Test with noise
                            if is_classification:
                                noisy_pred = model.predict(X_test_noisy)
                                noisy_score = accuracy_score(y_test_enc, noisy_pred)
                            else:
                                noisy_pred = model.predict(X_test_noisy)
                                noisy_score = r2_score(y_test, noisy_pred)
                            
                            # Performance drop
                            if baseline_score > 0:
                                performance_drop = abs(baseline_score - noisy_score) / baseline_score
                                # Score: 5 if no drop, 0 if >50% drop
                                noise_sensitivity_score = max(0, 5 * (1 - 2 * performance_drop))
                            else:
                                noise_sensitivity_score = 2.5
                        else:
                            noise_sensitivity_score = 2.5
                    else:
                        noise_sensitivity_score = 2.5
                else:
                    noise_sensitivity_score = 2.5
            else:
                noise_sensitivity_score = 2.5
    except Exception as e:
        noise_sensitivity_score = 2.5  # Default if computation fails
    
    # Weighted average
    data_quality_score = (
        0.3 * missing_score +
        0.25 * avg_class_imbalance_score +
        0.25 * redundancy_score +
        0.2 * noise_sensitivity_score
    )
    
    return {
        'score': round(data_quality_score, 2),
        'missing_value_ratio': round(missing_ratio, 4),
        'class_imbalance_score': round(avg_class_imbalance_score, 2),
        'feature_redundancy_score': round(redundancy_score, 2),
        'noise_sensitivity_score': round(noise_sensitivity_score, 2),
        'details': {
            'missing_score': round(missing_score, 2),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'avg_correlation': round(avg_corr, 4) if len(numeric_features) > 1 else None
        }
    }


def compute_accuracy_for_readiness(metrics_dict, task_type='auto'):
    """
    Computes accuracy-related metrics and returns a single score (0-5).
    
    Args:
        metrics_dict: Dict with model names as keys and metrics dicts as values
        task_type: 'Regression', 'Classification', or 'auto'
    
    Returns:
        dict: {
            'score': float (0-5),
            'best_model': str,
            'best_metric': float,
            'details': dict
        }
    """
    if not metrics_dict:
        return {'score': 0, 'best_model': None, 'best_metric': None, 'details': {}}
    
    # Determine task type from metrics
    if task_type == 'auto':
        first_metrics = list(metrics_dict.values())[0]
        if 'R2' in first_metrics:
            task_type = 'Regression'
        elif 'Classification_Accuracy' in first_metrics:
            task_type = 'Classification'
        else:
            task_type = 'Regression'  # Default
    
    # Collect all accuracy metrics
    accuracy_scores = {}
    for model_name, metrics in metrics_dict.items():
        if task_type == 'Regression':
            # Use R² as primary metric (scale 0-1 to 0-5)
            r2 = metrics.get('R2', 0)
            # Normalize R² to 0-5 scale
            r2_score_scaled = max(0, 5 * r2) if r2 >= 0 else 0
            accuracy_scores[model_name] = {
                'score': round(r2_score_scaled, 2),
                'R2': r2,
                'MAE': metrics.get('MAE', 0),
                'RMSE': metrics.get('RMSE', 0)
            }
        else:
            # Classification: Use accuracy and F1
            acc = metrics.get('Classification_Accuracy', 0)
            f1 = metrics.get('F1_Score', 0)
            # Weighted average: 60% accuracy, 40% F1
            combined_score = 0.6 * acc + 0.4 * f1
            accuracy_scores[model_name] = {
                'score': round(5 * combined_score, 2),  # Scale to 0-5
                'Accuracy': acc,
                'F1': f1
            }
    
    # Find best model
    best_model = max(accuracy_scores.keys(), key=lambda k: accuracy_scores[k]['score'])
    best_score = accuracy_scores[best_model]['score']
    
    # Average across all models
    avg_score = np.mean([acc['score'] for acc in accuracy_scores.values()])
    
    return {
        'score': round(avg_score, 2),
        'best_model': best_model,
        'best_metric': round(best_score, 2),
        'details': accuracy_scores
    }


def compute_interpretability_for_readiness(shap_results_dict, feature_names):
    """
    Computes interpretability using SHAP concentration + stability.
    Average SHAP output by all 4 models on each dataset and then compare.
    
    Args:
        shap_results_dict: Dict with model names as keys and SHAP results as values
        feature_names: List of feature names
    
    Returns:
        dict: {
            'score': float (0-5),
            'shap_concentration': float,
            'shap_stability': float,
            'models_with_shap': int,
            'details': dict
        }
    """
    if not shap_results_dict:
        return {
            'score': 0,
            'shap_concentration': 0,
            'shap_stability': 0,
            'models_with_shap': 0,
            'details': {}
        }
    
    # Collect SHAP importance from all successful models
    shap_importances = []
    successful_models = []
    
    for model_name, shap_result in shap_results_dict.items():
        if shap_result.get('success', False) and shap_result.get('importance') is not None:
            shap_importances.append(shap_result['importance'])
            successful_models.append(model_name)
    
    if not shap_importances:
        return {
            'score': 0,
            'shap_concentration': 0,
            'shap_stability': 0,
            'models_with_shap': 0,
            'details': {}
        }
    
    # 1. SHAP Concentration
    # Measure how concentrated feature importance is (fewer important features = more interpretable)
    concentration_scores = []
    for shap_imp in shap_importances:
        if isinstance(shap_imp, dict):
            importances = list(shap_imp.values())
        elif isinstance(shap_imp, (list, np.ndarray)):
            importances = shap_imp
        else:
            continue
        
        if len(importances) > 0:
            importances = np.array(importances)
            importances = np.abs(importances)
            importances = importances / (importances.sum() + 1e-10)  # Normalize
            
            # Calculate entropy (lower entropy = more concentrated)
            entropy_val = entropy(importances)
            max_entropy = np.log(len(importances))
            
            # Concentration: 1 - normalized_entropy (higher = more concentrated)
            concentration = 1 - (entropy_val / max_entropy) if max_entropy > 0 else 0
            concentration_scores.append(concentration)
    
    avg_concentration = np.mean(concentration_scores) if concentration_scores else 0
    
    # 2. SHAP Stability
    # Measure consistency of feature importance across models
    if len(shap_importances) > 1:
        # Convert all to same format and align features
        aligned_importances = []
        all_features = set()
        
        for shap_imp in shap_importances:
            if isinstance(shap_imp, dict):
                all_features.update(shap_imp.keys())
            elif isinstance(shap_imp, (list, np.ndarray)) and len(shap_imp) == len(feature_names):
                all_features.update(feature_names[:len(shap_imp)])
        
        # Create aligned importance vectors
        for shap_imp in shap_importances:
            aligned_dict = {}
            if isinstance(shap_imp, dict):
                aligned_dict = shap_imp
            elif isinstance(shap_imp, (list, np.ndarray)) and len(shap_imp) == len(feature_names):
                aligned_dict = dict(zip(feature_names[:len(shap_imp)], shap_imp))
            
            # Normalize
            total = sum(abs(v) for v in aligned_dict.values())
            if total > 0:
                aligned_dict = {k: abs(v) / total for k, v in aligned_dict.items()}
            aligned_importances.append(aligned_dict)
        
        # Calculate pairwise correlations
        if len(aligned_importances) >= 2:
            stability_scores = []
            for i in range(len(aligned_importances)):
                for j in range(i + 1, len(aligned_importances)):
                    imp1 = aligned_importances[i]
                    imp2 = aligned_importances[j]
                    
                    # Get common features
                    common_features = set(imp1.keys()) & set(imp2.keys())
                    if len(common_features) > 0:
                        vec1 = [imp1.get(f, 0) for f in common_features]
                        vec2 = [imp2.get(f, 0) for f in common_features]
                        
                        # Correlation
                        if np.std(vec1) > 0 and np.std(vec2) > 0:
                            corr = np.corrcoef(vec1, vec2)[0, 1]
                            stability_scores.append(corr)
            
            avg_stability = np.mean(stability_scores) if stability_scores else 0
        else:
            avg_stability = 0.5  # Neutral if only one model
    else:
        avg_stability = 0.5  # Neutral if only one model
    
    # Combined interpretability score
    # Concentration (40%) + Stability (40%) + Model coverage (20%)
    model_coverage = len(successful_models) / 4.0  # 4 models total
    interpretability_score = (
        0.4 * avg_concentration * 5 +  # Scale to 0-5
        0.4 * avg_stability * 5 +
        0.2 * model_coverage * 5
    )
    
    return {
        'score': round(interpretability_score, 2),
        'shap_concentration': round(avg_concentration, 4),
        'shap_stability': round(avg_stability, 4),
        'models_with_shap': len(successful_models),
        'details': {
            'successful_models': successful_models,
            'concentration_scores': [round(c, 4) for c in concentration_scores]
        }
    }


def compute_fairness_for_readiness(df, target_columns, demographic_column=None, fairness_results=None):
    """
    Computes demographic parity and accuracy gap.
    Considers:
    - Distribution per category in target variable
    - Missingness & Data Quality Disparities
    - Missing values per category of target variable
    
    Args:
        df: DataFrame
        target_columns: List of target column names
        demographic_column: Name of demographic column (if None, use target categories)
        fairness_results: Optional fairness analysis results from pipeline
    
    Returns:
        dict: {
            'score': float (0-5),
            'demographic_parity': float,
            'accuracy_gap': float,
            'missingness_disparity': float,
            'details': dict
        }
    """
    fairness_scores = []
    
    for target_col in target_columns:
        if target_col not in df.columns:
            continue
        
        target_series = df[target_col].dropna()
        
        # 1. Distribution per category in target variable
        if target_series.dtype == 'object' or len(target_series.unique()) < 20:
            # Classification: check class balance
            value_counts = target_series.value_counts()
            if len(value_counts) > 1:
                probs = value_counts / len(target_series)
                ent = entropy(probs)
                max_ent = np.log(len(value_counts))
                balance_ratio = ent / max_ent if max_ent > 0 else 0
                distribution_score = 5 * balance_ratio
            else:
                distribution_score = 0
        else:
            # Regression: check value distribution fairness
            # Higher variance = more diverse = potentially more fair
            cv = target_series.std() / target_series.mean() if target_series.mean() != 0 else 0
            distribution_score = min(5, 2.5 + 2.5 * min(1, cv))
        
        # 2. Missingness & Data Quality Disparities per category
        if target_series.dtype == 'object' or len(target_series.unique()) < 20:
            # For each category, check missing values in features
            categories = target_series.unique()
            missing_ratios_per_category = []
            
            feature_cols = [col for col in df.columns if col != target_col]
            for category in categories:
                category_mask = df[target_col] == category
                category_data = df.loc[category_mask, feature_cols]
                if len(category_data) > 0:
                    missing_ratio = category_data.isnull().sum().sum() / (len(category_data) * len(feature_cols))
                    missing_ratios_per_category.append(missing_ratio)
            
            if len(missing_ratios_per_category) > 1:
                # Disparity = std of missing ratios across categories
                missingness_disparity = np.std(missing_ratios_per_category)
                # Score: 5 if no disparity, 0 if high disparity
                missingness_score = max(0, 5 * (1 - 5 * missingness_disparity))
            else:
                missingness_score = 2.5
        else:
            # For regression, use demographic column if available
            if demographic_column and demographic_column in df.columns:
                demo_values = df[demographic_column].dropna()
                if len(demo_values.unique()) > 1:
                    missing_ratios_per_demo = []
                    feature_cols = [col for col in df.columns if col not in [target_col, demographic_column]]
                    for demo_val in demo_values.unique():
                        demo_mask = df[demographic_column] == demo_val
                        demo_data = df.loc[demo_mask, feature_cols]
                        if len(demo_data) > 0:
                            missing_ratio = demo_data.isnull().sum().sum() / (len(demo_data) * len(feature_cols))
                            missing_ratios_per_demo.append(missing_ratio)
                    
                    if len(missing_ratios_per_demo) > 1:
                        missingness_disparity = np.std(missing_ratios_per_demo)
                        missingness_score = max(0, 5 * (1 - 5 * missingness_disparity))
                    else:
                        missingness_score = 2.5
                else:
                    missingness_score = 2.5
            else:
                missingness_score = 2.5
        
        # 3. Accuracy Gap (from fairness_results if available)
        if fairness_results is not None:
            # Extract accuracy gap from fairness analysis
            if isinstance(fairness_results, pd.DataFrame):
                if 'Accuracy_Gap' in fairness_results.columns:
                    avg_accuracy_gap = fairness_results['Accuracy_Gap'].abs().mean()
                elif 'Disparity' in fairness_results.columns:
                    avg_accuracy_gap = fairness_results['Disparity'].abs().mean()
                else:
                    avg_accuracy_gap = 0
            else:
                avg_accuracy_gap = 0
            
            # Score: 5 if no gap, 0 if large gap
            # Normalize gap (assuming max gap of 1.0 for classification, or relative for regression)
            accuracy_gap_score = max(0, 5 * (1 - min(1, avg_accuracy_gap)))
        else:
            accuracy_gap_score = 2.5  # Neutral if no fairness results
        
        # Combined fairness score for this target
        target_fairness = (
            0.4 * distribution_score +
            0.3 * missingness_score +
            0.3 * accuracy_gap_score
        )
        fairness_scores.append(target_fairness)
    
    # Average across all targets
    avg_fairness_score = np.mean(fairness_scores) if fairness_scores else 2.5
    
    # Get missingness_disparity for return (use last computed value)
    missingness_disparity = 0
    if 'missingness_disparity' in locals():
        pass  # Already computed
    else:
        missingness_disparity = 0
    
    # Get accuracy_gap for return
    avg_accuracy_gap = 0
    if 'avg_accuracy_gap' in locals():
        pass  # Already computed
    else:
        avg_accuracy_gap = 0
    
    return {
        'score': round(avg_fairness_score, 2),
        'demographic_parity': round(distribution_score, 2) if 'distribution_score' in locals() else 2.5,
        'accuracy_gap': round(avg_accuracy_gap, 4),
        'missingness_disparity': round(missingness_disparity, 4),
        'details': {
            'fairness_scores_per_target': [round(s, 2) for s in fairness_scores]
        }
    }


def compute_scalability_for_readiness(df, models_dict, X_train, X_test, y_train, y_test, task_type='auto'):
    """
    Measures training time, inference time, and memory usage.
    
    Args:
        df: DataFrame
        models_dict: Dict of trained models
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        task_type: 'Regression' or 'Classification'
    
    Returns:
        dict: {
            'score': float (0-5),
            'training_time': float (seconds),
            'inference_time': float (seconds),
            'memory_usage_mb': float,
            'details': dict
        }
    """
    if not models_dict:
        return {
            'score': 0,
            'training_time': 0,
            'inference_time': 0,
            'memory_usage_mb': 0,
            'details': {}
        }
    
    # Get process for memory tracking
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
    else:
        process = None
    
    training_times = []
    inference_times = []
    memory_usages = []
    
    for model_name, model in models_dict.items():
        try:
            # Memory before
            if process:
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
            else:
                mem_before = 0
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            training_times.append(train_time)
            
            # Memory after training
            if process:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_used = mem_after - mem_before
                memory_usages.append(mem_used)
            else:
                memory_usages.append(0)
            
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
            'memory_usage_mb': 0,
            'details': {}
        }
    
    avg_training_time = np.mean(training_times)
    avg_inference_time = np.mean(inference_times)
    avg_memory = np.mean(memory_usages) if memory_usages else 0
    
    # Dataset size factors
    n_samples = len(df)
    n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns) if hasattr(X_train, 'columns') else 100
    
    # Score based on efficiency
    # Training time: 5 if <1s, 0 if >60s
    training_score = max(0, 5 * (1 - min(1, avg_training_time / 60)))
    
    # Inference time: 5 if <0.1s, 0 if >5s
    inference_score = max(0, 5 * (1 - min(1, avg_inference_time / 5)))
    
    # Memory: 5 if <100MB, 0 if >2GB
    memory_score = max(0, 5 * (1 - min(1, avg_memory / 2000))) if avg_memory > 0 else 2.5
    
    # Dataset size bonus: larger datasets get slight bonus if still fast
    size_bonus = 0
    if n_samples > 1000 and avg_training_time < 10:
        size_bonus = 0.5
    elif n_samples > 10000 and avg_training_time < 30:
        size_bonus = 1.0
    
    # Combined scalability score
    scalability_score = (
        0.4 * training_score +
        0.3 * inference_score +
        0.2 * memory_score +
        0.1 * min(5, size_bonus * 5)
    )
    
    return {
        'score': round(scalability_score, 2),
        'training_time': round(avg_training_time, 4),
        'inference_time': round(avg_inference_time, 4),
        'memory_usage_mb': round(avg_memory, 2),
        'details': {
            'n_samples': n_samples,
            'n_features': n_features,
            'training_score': round(training_score, 2),
            'inference_score': round(inference_score, 2),
            'memory_score': round(memory_score, 2)
        }
    }


def compute_dataset_readiness_score(
    df,
    target_columns,
    metrics_dict=None,
    shap_results_dict=None,
    fairness_results=None,
    models_dict=None,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    task_type='auto',
    demographic_column=None,
    feature_names=None
):
    """
    Main function to compute comprehensive dataset readiness score.
    
    Args:
        df: DataFrame
        target_columns: List of target column names
        metrics_dict: Dict of model metrics (from pipeline)
        shap_results_dict: Dict of SHAP results (from pipeline)
        fairness_results: Fairness analysis results (from pipeline)
        models_dict: Dict of trained models (for scalability)
        X_train, X_test: Feature matrices (for scalability)
        y_train, y_test: Target vectors (for scalability)
        task_type: 'Regression', 'Classification', or 'auto'
        demographic_column: Name of demographic column
        feature_names: List of feature names (for interpretability)
    
    Returns:
        dict: {
            'readiness_score': float (0-5),
            'data_quality': dict,
            'accuracy': dict,
            'interpretability': dict,
            'fairness': dict,
            'scalability': dict,
            'breakdown': dict
        }
    """
    # 1. Data Quality
    data_quality = compute_data_quality(df, target_columns)
    
    # 2. Accuracy
    if metrics_dict:
        accuracy = compute_accuracy_for_readiness(metrics_dict, task_type)
    else:
        accuracy = {'score': 0, 'best_model': None, 'best_metric': None, 'details': {}}
    
    # 3. Interpretability
    if shap_results_dict and feature_names:
        interpretability = compute_interpretability_for_readiness(shap_results_dict, feature_names)
    else:
        interpretability = {
            'score': 0,
            'shap_concentration': 0,
            'shap_stability': 0,
            'models_with_shap': 0,
            'details': {}
        }
    
    # 4. Fairness
    fairness = compute_fairness_for_readiness(df, target_columns, demographic_column, fairness_results)
    
    # 5. Scalability
    if models_dict and X_train is not None and X_test is not None:
        scalability = compute_scalability_for_readiness(df, models_dict, X_train, X_test, y_train, y_test, task_type)
    else:
        scalability = {
            'score': 0,
            'training_time': 0,
            'inference_time': 0,
            'memory_usage_mb': 0,
            'details': {}
        }
    
    # Final Readiness Score (weighted average)
    readiness_score = (
        0.25 * data_quality['score'] +
        0.25 * accuracy['score'] +
        0.20 * interpretability['score'] +
        0.15 * fairness['score'] +
        0.15 * scalability['score']
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
        }
    }


def comprehensive_ml_pipeline(dataset_path, target_column, task_type='auto',
                             feature_columns=None, use_tfidf=True,
                             max_tfidf_features=10000, demographic_column=None, **kwargs):
    """
    COMPREHENSIVE ML PIPELINE - Single function that does everything!
    
    This function extends generic_accuracy_calculator to include:
    - Accuracy metrics calculation (RMSE, MAPE, R², Accuracy)
    - SHAP analysis with feature importance plots (for all models and targets)
    - Demographic disparity and fairness analysis
    - Educational AI Readiness Index calculation
    
    All results are displayed inline - nothing is saved to files.
    
    Args:
        dataset_path: Path to CSV file or pandas DataFrame
        target_column: Target column name(s) - string or list
        task_type: 'auto', 'Regression', 'Classification', or 'Multi-label'
        feature_columns: List of feature columns (None = all except target)
        use_tfidf: Whether to use TF-IDF for text columns
        max_tfidf_features: Maximum TF-IDF features
        demographic_column: Column name for demographic groups (None = synthetic)
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
            "Please ensure generic_accuracy_calculator.py is in the same directory and can be imported.\n"
            "You may need to restart your Python kernel or reload the module."
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
    data_quality_score = calculate_data_quality_from_dataset(df)
    print(f"  ✅ Data Quality Score: {data_quality_score:.2f}/5.0")
    
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
            X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = \
                preprocess_data(X, y_single, task_type, use_tfidf, max_tfidf_features)
            
            print(f"    ✅ Task type: {detected_task_type}")
            print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
            
            # Train models and get predictions
            print(f"  🤖 Training models...")
            models_dict = {}
            predictions_dict = {}
            metrics_dict = {}
            shap_results_dict = {}
            
            # Define models
            if detected_task_type == 'Regression':
                models_dict = {
                    'LogisticRegression': LogisticRegression(max_iter=1000),
                    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
                    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                               random_state=42, verbosity=0),
                    'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                 random_state=42, verbose=-1)
                }
            else:
                models_dict = {
                    'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs'),
                    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
                    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                use_label_encoder=False, eval_metric='logloss',
                                                random_state=42, verbosity=0),
                    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                   random_state=42, verbose=-1)
                }
            
            # Train and evaluate each model
            for model_name, model in models_dict.items():
                try:
                    print(f"    🔹 Training {model_name}...")
                    
                    # Handle sparse matrices
                    if hasattr(X_train, 'toarray'):
                        X_train_model = X_train.toarray() if model_name == 'LogisticRegression' else X_train
                        X_test_model = X_test.toarray() if model_name == 'LogisticRegression' else X_test
                    else:
                        X_train_model = X_train
                        X_test_model = X_test
                    
                    # Special handling for LogisticRegression on regression
                    if detected_task_type == 'Regression' and model_name == 'LogisticRegression':
                        median_y = np.median(y_train)
                        y_train_class = (y_train > median_y).astype(int)
                        y_test_class = (y_test > median_y).astype(int)
                        model.fit(X_train_model, y_train_class)
                        y_pred_class = model.predict(X_test_model)
                        y_pred = np.where(y_pred_class == 1, median_y * 1.1, median_y * 0.9)
                    else:
                        model.fit(X_train_model, y_train)
                        y_pred = model.predict(X_test_model)
                    
                    predictions_dict[model_name] = y_pred
                    
                    # Calculate metrics
                    metrics = calculate_all_metrics(y_test, y_pred, detected_task_type)
                    metrics['Model'] = model_name
                    metrics['Target'] = target_col
                    metrics['Dataset'] = os.path.basename(dataset_path) if isinstance(dataset_path, str) else 'DataFrame'
                    metrics_dict[model_name] = metrics
                    all_results['metrics'].append(metrics)
                    
                    # SHAP analysis
                    shap_success, shap_importance, shap_df = perform_shap_analysis(
                        model, X_train_model, X_test_model, model_name,
                        detected_task_type, feature_names
                    )
                    shap_results_dict[model_name] = {
                        'success': shap_success,
                        'importance': shap_importance,
                        'dataframe': shap_df
                    }
                    
                    print(f"      ✅ {model_name}: ", end="")
                    if detected_task_type == 'Regression':
                        print(f"R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
                    else:
                        print(f"Acc={metrics['Classification_Accuracy']:.4f}, F1={metrics['F1_Score']:.4f}")
                
                except Exception as e:
                    print(f"      ❌ Error training {model_name}: {e}")
                    continue
            
            # Display metrics table for this target
            print(f"\n    📊 Accuracy Metrics for {target_col}:")
            metrics_table = pd.DataFrame([metrics_dict[m] for m in models_dict.keys() if m in metrics_dict])
            metric_cols = ['Model']
            if detected_task_type == 'Regression':
                metric_cols.extend(['R2', 'MAE', 'RMSE', 'MAPE', 'Regression_Accuracy'])
            else:
                metric_cols.extend(['Classification_Accuracy', 'F1_Score'])
            available_cols = [col for col in metric_cols if col in metrics_table.columns]
            print(metrics_table[available_cols].to_string(index=False))
            
            # Fairness analysis
            # Create demographic groups (synthetic if not provided)
            if demographic_column and demographic_column in df.columns:
                # Use actual demographic column
                demo_values = df[demographic_column].iloc[y_test.index].values
                unique_demos = np.unique(demo_values)
                demographic_groups = {demo: demo_values == demo for demo in unique_demos[:2]}
            else:
                # Synthetic demographic groups
                np.random.seed(42)
                demo_values = np.random.choice(['GroupA', 'GroupB'], size=len(y_test))
                demographic_groups = {
                    'GroupA': demo_values == 'GroupA',
                    'GroupB': demo_values == 'GroupB'
                }
            
            fairness_df = perform_fairness_analysis(
                models_dict, X_test, y_test, predictions_dict,
                demographic_groups, detected_task_type
            )
            fairness_df['Target'] = target_col
            all_results['fairness'].append(fairness_df)
            
            # Readiness Index
            print(f"  📈 Calculating Readiness Index for {target_col}...")
            readiness_list = []
            
            for model_name in models_dict.keys():
                if model_name in metrics_dict:
                    metrics = metrics_dict[model_name]
                    
                    # Interpretability
                    shap_success = shap_results_dict.get(model_name, {}).get('success', False)
                    feature_imp_available = shap_results_dict.get(model_name, {}).get('importance', {}) != {}
                    interpretability = calculate_interpretability(
                        model_name, shap_success, feature_imp_available
                    )
                    
                    # Fairness
                    fairness_row = fairness_df[fairness_df['Model'] == model_name]
                    if not fairness_row.empty and 'Disparity' in fairness_row.columns:
                        disparity = fairness_row['Disparity'].iloc[0]
                        max_disparity = fairness_df['Disparity'].max() if len(fairness_df) > 0 else disparity * 2
                        fairness_score = calculate_fairness_from_disparity(disparity, max_disparity)
                    else:
                        fairness_score = 2.5
                    
                    # Scalability
                    scalability = calculate_scalability(model_name, dataset_size, True)
                    
                    # Accuracy (normalized to 0-5)
                    if detected_task_type == 'Regression':
                        accuracy_score_val = round(metrics['R2'] * 5.0, 2)
                    else:
                        accuracy_score_val = round(metrics['Classification_Accuracy'] * 5.0, 2)
                    accuracy_score_val = max(0.0, min(5.0, accuracy_score_val))
                    
                    # Readiness index
                    readiness = calculate_readiness_index(
                        model_name, interpretability, fairness_score,
                        scalability, accuracy_score_val, data_quality_score
                    )
                    readiness['Target'] = target_col
                    readiness_list.append(readiness)
            
            readiness_df = pd.DataFrame(readiness_list)
            all_results['readiness'].append(readiness_df)
            
            # Display readiness index for this target
            print(f"\n    📈 Readiness Index for {target_col}:")
            print(readiness_df[['Model', 'Interpretability', 'Fairness', 'Scalability', 
                               'Accuracy', 'Data_Quality', 'Total_Score']].to_string(index=False))
            
            # Store for dataset readiness score (use last target's data)
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
        X_train, X_test, y_train, y_test, feature_names, processors, detected_task_type = \
            preprocess_data(X, y, task_type, use_tfidf, max_tfidf_features)
        
        print(f"    ✅ Task type: {detected_task_type}")
        print(f"    ✅ Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Train models
        print(f"  🤖 Training models...")
        models_dict = {}
        predictions_dict = {}
        metrics_dict = {}
        shap_results_dict = {}
        
        # Define models
        if detected_task_type == 'Regression':
            models_dict = {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbosity=0),
                'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                             random_state=42, verbose=-1)
            }
        else:
            models_dict = {
                'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs'),
                'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                            use_label_encoder=False, eval_metric='logloss',
                                            random_state=42, verbosity=0),
                'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                               random_state=42, verbose=-1)
            }
        
        # Train and evaluate
        for model_name, model in models_dict.items():
            try:
                print(f"    🔹 Training {model_name}...")
                
                # Handle sparse
                if hasattr(X_train, 'toarray'):
                    X_train_model = X_train.toarray() if model_name == 'LogisticRegression' else X_train
                    X_test_model = X_test.toarray() if model_name == 'LogisticRegression' else X_test
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Special handling for LogisticRegression on regression
                if detected_task_type == 'Regression' and model_name == 'LogisticRegression':
                    median_y = np.median(y_train)
                    y_train_class = (y_train > median_y).astype(int)
                    model.fit(X_train_model, y_train_class)
                    y_pred_class = model.predict(X_test_model)
                    y_pred = np.where(y_pred_class == 1, median_y * 1.1, median_y * 0.9)
                else:
                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_test_model)
                
                predictions_dict[model_name] = y_pred
                
                # Metrics
                metrics = calculate_all_metrics(y_test, y_pred, detected_task_type)
                metrics['Model'] = model_name
                metrics['Target'] = target_columns[0]
                metrics['Dataset'] = os.path.basename(dataset_path) if isinstance(dataset_path, str) else 'DataFrame'
                metrics_dict[model_name] = metrics
                all_results['metrics'].append(metrics)
                
                # SHAP
                shap_success, shap_importance, shap_df = perform_shap_analysis(
                    model, X_train_model, X_test_model, model_name,
                    detected_task_type, feature_names
                )
                shap_results_dict[model_name] = {
                    'success': shap_success,
                    'importance': shap_importance,
                    'dataframe': shap_df
                }
                
                print(f"      ✅ {model_name}: ", end="")
                if detected_task_type == 'Regression':
                    print(f"R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
                else:
                    print(f"Acc={metrics['Classification_Accuracy']:.4f}, F1={metrics['F1_Score']:.4f}")
            
            except Exception as e:
                print(f"      ❌ Error training {model_name}: {e}")
                continue
        
        # Display metrics table
        print(f"\n    📊 Accuracy Metrics:")
        metrics_table = pd.DataFrame([metrics_dict[m] for m in models_dict.keys() if m in metrics_dict])
        metric_cols = ['Model']
        if detected_task_type == 'Regression':
            metric_cols.extend(['R2', 'MAE', 'RMSE', 'MAPE', 'Regression_Accuracy'])
        else:
            metric_cols.extend(['Classification_Accuracy', 'F1_Score'])
        available_cols = [col for col in metric_cols if col in metrics_table.columns]
        print(metrics_table[available_cols].to_string(index=False))
        
        # Fairness
        if demographic_column and demographic_column in df.columns:
            demo_values = df[demographic_column].iloc[y_test.index].values
            unique_demos = np.unique(demo_values)
            demographic_groups = {demo: demo_values == demo for demo in unique_demos[:2]}
        else:
            np.random.seed(42)
            demo_values = np.random.choice(['GroupA', 'GroupB'], size=len(y_test))
            demographic_groups = {
                'GroupA': demo_values == 'GroupA',
                'GroupB': demo_values == 'GroupB'
            }
        
        fairness_df = perform_fairness_analysis(
            models_dict, X_test, y_test, predictions_dict,
            demographic_groups, detected_task_type
        )
        all_results['fairness'].append(fairness_df)
        
        # Readiness Index
        print(f"  📈 Calculating Readiness Index...")
        readiness_list = []
        
        for model_name in models_dict.keys():
            if model_name in metrics_dict:
                metrics = metrics_dict[model_name]
                
                # Interpretability
                shap_success = shap_results_dict.get(model_name, {}).get('success', False)
                feature_imp_available = shap_results_dict.get(model_name, {}).get('importance', {}) != {}
                interpretability = calculate_interpretability(
                    model_name, shap_success, feature_imp_available
                )
                
                # Fairness
                fairness_row = fairness_df[fairness_df['Model'] == model_name]
                if not fairness_row.empty and 'Disparity' in fairness_row.columns:
                    disparity = fairness_row['Disparity'].iloc[0]
                    max_disparity = fairness_df['Disparity'].max() if len(fairness_df) > 0 else disparity * 2
                    fairness_score = calculate_fairness_from_disparity(disparity, max_disparity)
                else:
                    fairness_score = 2.5
                
                # Scalability
                scalability = calculate_scalability(model_name, dataset_size, True)
                
                # Accuracy
                if detected_task_type == 'Regression':
                    accuracy_score_val = round(metrics['R2'] * 5.0, 2)
                else:
                    accuracy_score_val = round(metrics['Classification_Accuracy'] * 5.0, 2)
                accuracy_score_val = max(0.0, min(5.0, accuracy_score_val))
                
                # Readiness
                readiness = calculate_readiness_index(
                    model_name, interpretability, fairness_score,
                    scalability, accuracy_score_val, data_quality_score
                )
                readiness['Target'] = target_columns[0]  # Add target column for consistency
                readiness_list.append(readiness)
        
        readiness_df = pd.DataFrame(readiness_list)
        all_results['readiness'].append(readiness_df)
        
        # Display readiness index
        print(f"\n    📈 Readiness Index:")
        print(readiness_df[['Model', 'Interpretability', 'Fairness', 'Scalability', 
                           'Accuracy', 'Data_Quality', 'Total_Score']].to_string(index=False))
        
        # Store for dataset readiness score
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
    print("Calculating comprehensive readiness based on:")
    print("  - Data Quality (missing values, class imbalance, redundancy, noise sensitivity)")
    print("  - Accuracy (model performance across all models)")
    print("  - Interpretability (SHAP concentration and stability)")
    print("  - Fairness (demographic parity, accuracy gaps, missingness disparities)")
    print("  - Scalability (training time, inference time, memory usage)")
    print("="*80)
    
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
        feature_names=stored_feature_names
    )
    
    # Display Dataset Readiness Score
    print(f"\n🎯 DATASET READINESS SCORE: {dataset_readiness['readiness_score']:.2f}/5.0")
    print("="*80)
    print("\n📊 Breakdown by Dimension:")
    print("-" * 80)
    print(f"  Data Quality:      {dataset_readiness['breakdown']['data_quality']:.2f}/5.0")
    print(f"    - Missing Value Ratio: {dataset_readiness['data_quality']['missing_value_ratio']:.4f}")
    print(f"    - Class Imbalance:      {dataset_readiness['data_quality']['class_imbalance_score']:.2f}/5.0")
    print(f"    - Feature Redundancy:   {dataset_readiness['data_quality']['feature_redundancy_score']:.2f}/5.0")
    print(f"    - Noise Sensitivity:   {dataset_readiness['data_quality']['noise_sensitivity_score']:.2f}/5.0")
    print(f"\n  Accuracy:          {dataset_readiness['breakdown']['accuracy']:.2f}/5.0")
    if dataset_readiness['accuracy']['best_model']:
        print(f"    - Best Model: {dataset_readiness['accuracy']['best_model']}")
        print(f"    - Best Score: {dataset_readiness['accuracy']['best_metric']:.2f}/5.0")
    print(f"\n  Interpretability:   {dataset_readiness['breakdown']['interpretability']:.2f}/5.0")
    print(f"    - SHAP Concentration: {dataset_readiness['interpretability']['shap_concentration']:.4f}")
    print(f"    - SHAP Stability:      {dataset_readiness['interpretability']['shap_stability']:.4f}")
    print(f"    - Models with SHAP:    {dataset_readiness['interpretability']['models_with_shap']}/4")
    print(f"\n  Fairness:          {dataset_readiness['breakdown']['fairness']:.2f}/5.0")
    print(f"    - Demographic Parity:  {dataset_readiness['fairness']['demographic_parity']:.2f}/5.0")
    print(f"    - Accuracy Gap:        {dataset_readiness['fairness']['accuracy_gap']:.4f}")
    print(f"    - Missingness Disparity: {dataset_readiness['fairness']['missingness_disparity']:.4f}")
    print(f"\n  Scalability:       {dataset_readiness['breakdown']['scalability']:.2f}/5.0")
    print(f"    - Training Time:       {dataset_readiness['scalability']['training_time']:.4f} seconds")
    print(f"    - Inference Time:      {dataset_readiness['scalability']['inference_time']:.4f} seconds")
    print(f"    - Memory Usage:        {dataset_readiness['scalability']['memory_usage_mb']:.2f} MB")
    print("="*80)
    
    # Display all results inline
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL RESULTS")
    print("="*80)
    
    # Display all accuracy metrics
    if all_results['metrics']:
        metrics_df = pd.DataFrame(all_results['metrics'])
        print("\n📊 ACCURACY METRICS (All Models & Targets):")
        print("="*80)
        
        # Display all metrics columns
        metric_cols = ['Model', 'Target']
        if 'R2' in metrics_df.columns:
            metric_cols.extend(['R2', 'MAE', 'RMSE', 'MAPE', 'Regression_Accuracy'])
        if 'Classification_Accuracy' in metrics_df.columns:
            metric_cols.extend(['Classification_Accuracy', 'F1_Score'])
        
        available_cols = [col for col in metric_cols if col in metrics_df.columns]
        print(metrics_df[available_cols].to_string(index=False))
    
    # Display fairness analysis
    if all_results['fairness']:
        fairness_df = pd.concat(all_results['fairness'], ignore_index=True)
        print("\n⚖️ FAIRNESS ANALYSIS (All Models & Targets):")
        print("="*80)
        print(fairness_df.to_string(index=False))
    
    # Display readiness index
    if all_results['readiness']:
        readiness_df = pd.concat(all_results['readiness'], ignore_index=True)
        print("\n📈 READINESS INDEX (All Models & Targets):")
        print("="*80)
        # Include 'Target' column only if it exists
        cols = ['Model', 'Interpretability', 'Fairness', 'Scalability', 
                'Accuracy', 'Data_Quality', 'Total_Score']
        if 'Target' in readiness_df.columns:
            cols.insert(1, 'Target')  # Insert 'Target' after 'Model'
        print(readiness_df[cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ ALL RESULTS DISPLAYED ABOVE")
    print("="*80)
    print("Note: SHAP plots and feature importance tables were displayed during processing.")
    
    return {
        'metrics': pd.DataFrame(all_results['metrics']) if all_results['metrics'] else pd.DataFrame(),
        'fairness': pd.concat(all_results['fairness'], ignore_index=True) if all_results['fairness'] else pd.DataFrame(),
        'readiness': pd.concat(all_results['readiness'], ignore_index=True) if all_results['readiness'] else pd.DataFrame(),
        'shap': shap_results_dict,
        'dataset_readiness': dataset_readiness
    }


if __name__ == "__main__":
    # Example usage
    print("""
    ============================================================================
    COMPREHENSIVE ML PIPELINE - USAGE EXAMPLES
    ============================================================================
    
    Example 1: Student Performance
    ------------------------------
    from comprehensive_ml_pipeline import comprehensive_ml_pipeline
    
    results = comprehensive_ml_pipeline(
        dataset_path='student_performance/StudentPerformanceFactors.csv',
        target_column='Exam_Score',
        task_type='auto'
    )
    # All results (metrics, SHAP plots, fairness, readiness) displayed inline
    
    Example 2: Budget Prediction (Multi-target)
    -------------------------------------------
    results = comprehensive_ml_pipeline(
        dataset_path='education/TrainingData.csv',
        target_column=['Function', 'Object_Type', 'Operating_Status'],
        task_type='auto',
        use_tfidf=True
    )
    # SHAP plots and metrics displayed for each target variable
    
    Example 3: Enrollment Forecasting
    ----------------------------------
    results = comprehensive_ml_pipeline(
        dataset_path='college_enrollment/enrollment.csv',
        target_column='Enrollment',
        task_type='auto'
    )
    # All results displayed inline - nothing saved to files
    """)


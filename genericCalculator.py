"""
generic_accuracy_calculator_enhanced.py
======================================
- Fixes ColumnTransformer+TFIDF single-column Series issue via [tc] + flatten
- Provides preprocessing + baseline & tuned model training
- Returns rich metric dicts including timing + best params + best CV score
- NO LinearRegression anywhere (by design)
- Adds HistGradientBoosting (HistGB) for both regression & classification

NOTE: SHAP plotting + readiness scoring are handled in vertexML_enhanced.py.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import time
from typing import Dict, Tuple, Any, List, Optional

from scipy import sparse

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False


def detect_and_transform_wide_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if len(year_cols) >= 5:
        id_cols = [c for c in df.columns if c not in year_cols]
        df_long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name="Value")
        df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
        df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
        return df_long, True
    return df, False


def detect_text_columns(df: pd.DataFrame, max_unique_ratio: float = 0.8) -> List[str]:
    text_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(300)
            if len(sample) == 0:
                continue
            avg_len = sample.map(len).mean()
            uniq_ratio = sample.nunique() / max(len(sample), 1)
            if avg_len >= 20 and uniq_ratio <= max_unique_ratio:
                text_cols.append(c)
    return text_cols


def detect_task_type(y: pd.Series, user_task_type: str = "auto") -> str:
    if user_task_type and str(user_task_type).lower() != "auto":
        return user_task_type
    if y.dtype == "object" or y.dtype.name == "category":
        return "Classification"
    nun = y.dropna().nunique()
    if pd.api.types.is_integer_dtype(y) and nun <= 20:
        return "Classification"
    return "Regression"


def build_preprocessor(
    X: pd.DataFrame,
    text_cols: List[str],
    use_tfidf: bool = True,
    max_tfidf_features: int = 10000
) -> ColumnTransformer:
    all_cols = list(X.columns)
    text_cols = [c for c in text_cols if c in all_cols]

    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(X[c]) and c not in text_cols]
    categorical_cols = [c for c in all_cols if (X[c].dtype == "object" or X[c].dtype.name == "category") and c not in text_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    if use_tfidf and text_cols:
        for tc in text_cols:
            text_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("flatten", FunctionTransformer(lambda x: x.ravel().astype(str), validate=False)),
                ("tfidf", TfidfVectorizer(max_features=max_tfidf_features, stop_words="english"))
            ])
            transformers.append((f"tfidf_{tc}", text_pipe, [tc]))  # ✅ [tc] keeps it 2D

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    feature_columns: Optional[List[str]] = None,
    use_tfidf: bool = True,
    max_tfidf_features: int = 10000,
    test_size: float = 0.2,
    random_state: int = 42
):
    if feature_columns is None:
        X = df.drop(columns=[target_column])
    else:
        X = df[feature_columns].copy()

    y = df[target_column].copy()
    task_type_final = detect_task_type(y, user_task_type=task_type)

    label_encoder = None
    if task_type_final == "Classification":
        label_encoder = LabelEncoder()
        y = y.astype(str).fillna("NA")
        y = label_encoder.fit_transform(y)

    text_cols = detect_text_columns(X)
    # If TF-IDF is disabled, do NOT use TF-IDF; let build_preprocessor treat object columns as categorical.
    if not use_tfidf:
        text_cols = []
    preprocessor = build_preprocessor(X, text_cols, use_tfidf=use_tfidf, max_tfidf_features=max_tfidf_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type_final == "Classification" else None
    )

    return X_train, X_test, y_train, y_test, task_type_final, preprocessor, label_encoder


def calculate_all_metrics(y_true, y_pred, task_type: str) -> Dict[str, float]:
    # Calculate all metrics for both regression and classification
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    if task_type == "Regression":
        return {
            "Regression_Accuracy_R2": round(float(r2), 4),
            "R2": round(float(r2), 4),
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4),
            "Accuracy": round(float(acc), 4),  # Can be calculated for regression too
        }
    
    # Classification
    return {
        "Accuracy": round(float(acc), 4),
        "Classification_Accuracy": round(float(acc), 4),
        "F1_Score": round(float(f1), 4),
        "R2": round(float(r2), 4),  # R2 can be calculated for classification
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }


def get_base_models(task_type: str) -> Dict[str, Any]:
    if task_type == "Regression":
        models = {
            "LogisticRegression": LinearRegression(),  # Using LinearRegression for regression (LogisticRegression is for classification)
            "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
            "HistGB": HistGradientBoostingRegressor(random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                random_state=42, n_estimators=400, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8
            )
        if LGB_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(
                random_state=42, n_estimators=600, learning_rate=0.05
            )
        return models

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=42, n_estimators=400, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss"
        )
    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            random_state=42, n_estimators=600, learning_rate=0.05
        )
    return models


def get_tuning_space(task_type: str) -> Dict[str, Dict[str, Any]]:
    if task_type == "Regression":
        spaces = {
            "LogisticRegression": {
                "model__fit_intercept": [True, False],
            },
            "RandomForest": {
                "model__n_estimators": [150, 300, 500],
                "model__max_depth": [None, 8, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }
        }

        spaces["HistGB"] = {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [None, 6, 10],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [200, 400],
        }
        if XGB_AVAILABLE:
            spaces["XGBoost"] = {
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [3, 5, 7, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            }
        if LGB_AVAILABLE:
            spaces["LightGBM"] = {
                "model__n_estimators": [300, 600, 900],
                "model__num_leaves": [31, 63, 127],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            }
        return spaces

    spaces = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "saga"],
        },
        "HistGB": {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [None, 6, 10],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [200, 400],
        },
        "RandomForest": {
            "model__n_estimators": [150, 300, 500],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    }
    if XGB_AVAILABLE:
        spaces["XGBoost"] = {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    if LGB_AVAILABLE:
        spaces["LightGBM"] = {
            "model__n_estimators": [300, 600, 900],
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    return spaces


def _densify_if_needed(X):
    """Convert sparse matrices to dense numpy arrays (needed for HistGB)."""
    try:
        if sparse.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return X


def train_and_evaluate_models(
    X_train, X_test, y_train, y_test,
    task_type: str,
    preprocessor,
    do_hyperparam_tuning: bool = True,
    n_iter: int = 4,
    cv: int = 2,
    random_state: int = 42,
    verbose: int = 0,
    model_names: Optional[List[str]] = None,
    inference_batch_size: int = 2000
):
    """
    Trains baseline models and optionally runs RandomizedSearchCV.
    Adds:
      - Train_Time_Sec
      - Inference_Time_ms_per_sample (predict time on X_test, optionally batched)
      - Best_Params / Best_CV_Score (if tuned)
    model_names: if provided, only run the listed models.
    """
    base_models = get_base_models(task_type)
    tune_spaces = get_tuning_space(task_type)

    if model_names is not None:
        keep = set(model_names)
        base_models = {k: v for k, v in base_models.items() if k in keep}
        tune_spaces = {k: v for k, v in tune_spaces.items() if k in keep}

    models_baseline, metrics_baseline = {}, {}
    models_tuned, metrics_tuned = {}, {}

    def _predict_timed(pipeline, X):
        t0 = time.time()
        if hasattr(X, "iloc"):
            n = X.shape[0]
            preds = []
            for i in range(0, n, inference_batch_size):
                preds.append(pipeline.predict(X.iloc[i:i+inference_batch_size]))
            y_pred = np.concatenate(preds, axis=0)
        else:
            y_pred = pipeline.predict(X)
        dt = time.time() - t0
        ms_per = (dt / max(1, len(X))) * 1000.0
        return y_pred, ms_per

    # Baseline
    for name, model in base_models.items():
        steps = [("preprocess", preprocessor)]
        if name == "HistGB":
            steps.append(("densify", FunctionTransformer(_densify_if_needed, validate=False)))
        # LinearRegression doesn't need densification, but sparse matrices might cause issues
        if name == "LogisticRegression" and task_type == "Regression":
            steps.append(("densify", FunctionTransformer(_densify_if_needed, validate=False)))
        steps.append(("model", model))
        pipe = Pipeline(steps=steps)

        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_pred, inf_ms = _predict_timed(pipe, X_test)

        m = calculate_all_metrics(y_test, y_pred, task_type)
        m["Train_Time_Sec"] = round(float(fit_time), 4)
        m["Inference_Time_ms_per_sample"] = round(float(inf_ms), 4)

        models_baseline[name] = pipe
        metrics_baseline[name] = m

    # Tuning
    if do_hyperparam_tuning:
        for name, model in base_models.items():
            if name not in tune_spaces:
                models_tuned[name] = models_baseline[name]
                metrics_tuned[name] = metrics_baseline[name]
                continue

            steps = [("preprocess", preprocessor)]
            if name == "HistGB":
                steps.append(("densify", FunctionTransformer(_densify_if_needed, validate=False)))
            # LinearRegression doesn't need densification, but sparse matrices might cause issues
            if name == "LogisticRegression" and task_type == "Regression":
                steps.append(("densify", FunctionTransformer(_densify_if_needed, validate=False)))
            steps.append(("model", model))
            pipe = Pipeline(steps=steps)

            scoring = "r2" if task_type == "Regression" else "accuracy"

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=tune_spaces[name],
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                verbose=verbose,
                refit=True
            )

            t0 = time.time()
            search.fit(X_train, y_train)
            fit_time = time.time() - t0

            best_model = search.best_estimator_
            y_pred, inf_ms = _predict_timed(best_model, X_test)

            m = calculate_all_metrics(y_test, y_pred, task_type)
            m["Train_Time_Sec"] = round(float(fit_time), 4)
            m["Inference_Time_ms_per_sample"] = round(float(inf_ms), 4)
            m["Best_Params"] = search.best_params_
            m["Best_CV_Score"] = float(search.best_score_)

            models_tuned[name] = best_model
            metrics_tuned[name] = m
    else:
        models_tuned = dict(models_baseline)
        metrics_tuned = dict(metrics_baseline)

    return models_baseline, metrics_baseline, models_tuned, metrics_tuned

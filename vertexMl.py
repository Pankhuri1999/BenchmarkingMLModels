"""
vertexML_enhanced.py
====================
Adds:
- SHAP feature importance plots (bar + summary) displayed and saved
- Supports HistGB (HistGradientBoosting) models
- Educational AI Readiness Score printed per model
- Full metrics printed: Accuracy, Regression Accuracy (R2), RMSE, MAPE, MAE, R2
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import shap

# from generic_accuracy_calculator_enhanced_histgb_v12_stable import (
#     detect_and_transform_wide_format,
#     preprocess_data,
#     train_and_evaluate_models
# )


def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def _ensure_shap_js():
    try:
        shap.initjs()
    except Exception:
        pass


# -------------------------
# Readiness score helpers
# -------------------------

def _score_data_quality(df_features: pd.DataFrame) -> float:
    total = df_features.shape[0] * max(df_features.shape[1], 1)
    miss = int(df_features.isna().sum().sum())
    ratio = miss / max(total, 1)
    score = 5.0 * (1.0 - min(1.0, 2.0 * ratio))
    return float(max(0.0, min(5.0, score)))


def _score_accuracy(task_type: str, metrics: dict) -> float:
    if task_type == "Regression":
        r2 = float(metrics.get("R2", 0.0) or 0.0)
        mapped = 2.5 * (r2 + 1.0)  # [-1,1] -> [0,5]
        return float(max(0.0, min(5.0, mapped)))
    acc = float(metrics.get("Accuracy", metrics.get("Classification_Accuracy", 0.0)) or 0.0)
    return float(max(0.0, min(5.0, 5.0 * acc)))


def _score_training_time(train_time_sec: float) -> float:
    """Smaller training time -> higher score (0..5), log mapping."""
    t = float(train_time_sec or 0.0)
    score = 5.0 - np.log10(1.0 + t) * 1.5
    return float(max(0.0, min(5.0, score)))


def _score_inference_time(inf_ms_per_sample: float) -> float:
    """Smaller inference latency -> higher score (0..5), log mapping on milliseconds."""
    ms = float(inf_ms_per_sample or 0.0)
    score = 5.0 - np.log10(1.0 + ms) * 1.5
    return float(max(0.0, min(5.0, score)))


def _score_scalability(metrics: dict) -> float:
    """Combine training + inference into one scalability score."""
    st = _score_training_time(metrics.get("Train_Time_Sec", 0.0))
    si = _score_inference_time(metrics.get("Inference_Time_ms_per_sample", 0.0))
    return float(max(0.0, min(5.0, 0.6 * st + 0.4 * si)))



def _score_fairness_from_shap(shap_values_2d: np.ndarray, preds_1d: np.ndarray, n_bins: int = 4) -> float:
    """
    Demographic-free fairness using SHAP stability across prediction strata.
    Bin samples by prediction quantiles, compare mean(|SHAP|) distributions across bins.
    Higher = fairer (more consistent).
    """
    sv = np.array(shap_values_2d, dtype=float)
    if sv.ndim != 2 or sv.shape[0] < 20:
        return 0.0

    preds = np.array(preds_1d, dtype=float).reshape(-1)
    m = min(len(preds), sv.shape[0])
    preds = preds[:m]
    sv = sv[:m, :]

    if np.nanstd(preds) < 1e-12:
        return 0.0

    try:
        bins = pd.qcut(preds, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(preds, bins=n_bins, labels=False)

    bins = np.array(bins)
    uniq = [b for b in np.unique(bins) if pd.notna(b)]
    if len(uniq) < 2:
        return 0.0

    vecs = []
    for b in uniq:
        idx = np.where(bins == b)[0]
        if len(idx) < 5:
            continue
        v = np.mean(np.abs(sv[idx, :]), axis=0)
        v = np.maximum(v, 0.0)
        s = v.sum()
        if s <= 0:
            continue
        vecs.append(v / s)

    if len(vecs) < 2:
        return 0.0

    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(float(np.sum(np.abs(vecs[i] - vecs[j]))))
    l1 = float(np.mean(dists))  # 0..2
    fairness = 5.0 * (1.0 - min(1.0, max(0.0, l1 / 2.0)))
    return float(max(0.0, min(5.0, fairness)))

def _score_interpretability_from_shap(mean_abs_shap: np.ndarray) -> float:
    v = np.array(mean_abs_shap, dtype=float)
    v = np.maximum(v, 0.0)
    if v.sum() <= 0:
        return 0.0
    p = v / v.sum()
    eps = 1e-12
    entropy = -np.sum(p * np.log(p + eps))
    max_entropy = np.log(len(p) + eps)
    concentration = 1.0 - (entropy / max_entropy)
    return float(max(0.0, min(5.0, 5.0 * concentration)))


def compute_readiness_score(df_features: pd.DataFrame, task_type: str, metrics: dict, shap_bundle) -> dict:
    """
    Weights:
      Accuracy 25%
      Fairness 20% (SHAP-based, demographic-free)
      Interpretability 20% (SHAP concentration)
      Data Quality 20%
      Scalability 15% (train + inference)
    """
    dq = _score_data_quality(df_features)
    acc = _score_accuracy(task_type, metrics)
    sc = _score_scalability(metrics)

    interp = 0.0
    fair = 0.0
    if shap_bundle is not None and shap_bundle.get("success"):
        mean_abs = shap_bundle.get("mean_abs_shap")
        sv_plot = shap_bundle.get("sv_plot")
        preds = shap_bundle.get("preds")
        if mean_abs is not None:
            interp = _score_interpretability_from_shap(mean_abs)
        if sv_plot is not None and preds is not None:
            fair = _score_fairness_from_shap(sv_plot, preds, n_bins=4)

    total = (
        0.25 * acc +
        0.20 * fair +
        0.20 * interp +
        0.20 * dq +
        0.15 * sc
    )

    return {
        "DataQuality": round(dq, 3),
        "Accuracy": round(acc, 3),
        "Fairness": round(fair, 3),
        "Interpretability": round(interp, 3),
        "Scalability": round(sc, 3),
        "ReadinessScore": round(float(total), 3)
    }


# -------------------------
# SHAP
# -------------------------



def _clean_feature_name(s: str) -> str:
    s = str(s)
    s = s.replace("num__", "").replace("cat__", "")
    s = s.replace("num_", "").replace("cat_", "")
    # ColumnTransformer may emit "remainder__<col>" too
    s = s.replace("remainder__", "")
    return s

def _clean_shap_feature_names(model, n_features):
    """Map transformed feature names back to clean dataset-level names."""
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(n_features)]

    cleaned = []
    for i, f in enumerate(names):
        f = f.replace("num__", "").replace("cat__", "")
        f = f.replace("num_", "").replace("cat_", "")
        if "Unnamed" in f:
            f = f"feature_{i}"
        cleaned.append(f)
    return cleaned


def _shap_to_2d(shap_values, task_type: str = "Regression"):
    """Normalize SHAP outputs to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        arrs = [np.array(a) for a in shap_values if a is not None]
        if len(arrs) == 0:
            return None
        # keep only 2D arrays
        arrs2 = []
        for a in arrs:
            if a.ndim == 2:
                arrs2.append(a)
            elif a.ndim == 3:
                # (n,f,c) -> choose last class
                arrs2.append(a[:, :, -1])
            else:
                arrs2.append(a.reshape(a.shape[0], -1))
        if len(arrs2) == 2 and task_type != "Regression":
            return arrs2[1]  # positive class
        return np.mean(np.stack(arrs2, axis=2), axis=2)

    sv = np.array(shap_values)
    if sv.dtype == object:
        try:
            sv = sv.astype(float)
        except Exception:
            pass
    if sv.ndim == 3:
        if task_type != "Regression" and sv.shape[2] >= 2:
            return sv[:, :, -1]
        return np.mean(sv, axis=2)
    if sv.ndim == 2:
        return sv
    if sv.ndim == 1:
        return sv.reshape(-1, 1)
    return None

def perform_shap_analysis_pipeline_model(
    trained_pipeline,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    model_name: str,
    task_type: str,
    out_dir: str = "shap_outputs",
    shap_sample_size: int = 200
):
    _safe_makedirs(out_dir)
    _ensure_shap_js()

    result = {
        "success": False,
        "importance_df_path": None,
        "summary_plot_path": None,
        "bar_plot_path": None,
        "mean_abs_shap": None,
        "sv_plot": None,
        "preds": None,
        "error": None
    }

    try:
        preprocess = trained_pipeline.named_steps["preprocess"]
        model = trained_pipeline.named_steps["model"]

        X_train = preprocess.transform(X_train_raw)
        X_test = preprocess.transform(X_test_raw)

        n = min(shap_sample_size, X_test.shape[0])
        X_test_s = X_test[:n]
        X_train_s = X_train[:min(50, X_train.shape[0])]

        if hasattr(X_test_s, "toarray"):
            X_test_s_dense = X_test_s.toarray()
        else:
            X_test_s_dense = np.array(X_test_s)

        if hasattr(X_train_s, "toarray"):
            X_train_s_dense = X_train_s.toarray()
        else:
            X_train_s_dense = np.array(X_train_s)

        try:
            feature_names = preprocess.get_feature_names_out()
            feature_names = [_clean_feature_name(x) for x in feature_names]
        except Exception:
            feature_names = [f"f{i}" for i in range(X_test_s_dense.shape[1])]

        # Drop common CSV index column names (e.g., Unnamed: 0) from labels only
        feature_names = [("Index" if ("Unnamed" in str(n)) else str(n)) for n in feature_names]

        # Explain (TreeExplainer preferred)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_s_dense)
        except Exception:
            # Kernel fallback
            if hasattr(model, "predict_proba") and task_type != "Regression":
                explainer = shap.KernelExplainer(model.predict_proba, X_train_s_dense[:50])
                shap_values = explainer.shap_values(X_test_s_dense[:50], nsamples=100)
            else:
                explainer = shap.KernelExplainer(model.predict, X_train_s_dense[:50])
                shap_values = explainer.shap_values(X_test_s_dense[:50], nsamples=100)

        sv_plot = _shap_to_2d(shap_values, task_type)

        if sv_plot is None:
            raise ValueError("SHAP returned an unsupported shape")
        mean_abs = np.mean(np.abs(sv_plot), axis=0)
        if getattr(mean_abs, 'ndim', 1) != 1:
            mean_abs = np.mean(mean_abs, axis=-1)
        result["mean_abs_shap"] = mean_abs
        result["sv_plot"] = sv_plot

        # predictions for the same samples (for SHAP-fairness)
        try:
            preds = model.predict(X_test_s_dense[:sv_plot.shape[0]])
        except Exception:
            preds = np.zeros((sv_plot.shape[0],), dtype=float)
        result["preds"] = np.array(preds).reshape(-1)

        raw_feats = feature_names[:len(mean_abs)]
        base_feats = [str(f).split("_", 1)[0] for f in raw_feats]
        imp_df = (
            pd.DataFrame({
                "feature": base_feats,
                "mean_abs_shap": mean_abs[:len(base_feats)]
            })
            .groupby("feature", as_index=False)["mean_abs_shap"].sum()
            .sort_values("mean_abs_shap", ascending=False)
        )

        imp_path = os.path.join(out_dir, f"{model_name}_shap_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        result["importance_df_path"] = imp_path

        # Bar plot (Top 20)
        top = imp_df.head(20).iloc[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean |SHAP|")
        plt.title(f"Top 20 SHAP Features - {model_name}")
        plt.tight_layout()
        bar_path = os.path.join(out_dir, f"{model_name}_shap_bar.png")
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()
        result["bar_plot_path"] = bar_path

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_plot,
            features=X_test_s_dense[:sv_plot.shape[0]],
            feature_names=feature_names,
            max_display=20,
            show=False
        )
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.tight_layout()
        sum_path = os.path.join(out_dir, f"{model_name}_shap_summary.png")
        plt.savefig(sum_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()
        result["summary_plot_path"] = sum_path

        result["success"] = True
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


# -------------------------
# Printing
# -------------------------

def _print_metrics(task_type: str, metrics: dict):
    if task_type == "Regression":
        print(f"Accuracy: {metrics.get('Accuracy')} | Regression Accuracy (R2): {metrics.get('R2')}")
        print(f"RMSE: {metrics.get('RMSE')} | MAE: {metrics.get('MAE')} | MAPE: {metrics.get('MAPE')} | R2: {metrics.get('R2')}")
    else:
        print(f"Accuracy: {metrics.get('Accuracy')} | F1: {metrics.get('F1_Score')}")
        print(f"R2: {metrics.get('R2')} | RMSE: {metrics.get('RMSE')} | MAE: {metrics.get('MAE')} | MAPE: {metrics.get('MAPE')}")
    print(f"TrainTime(s): {metrics.get('Train_Time_Sec')}")
    print(f"Inference(ms/sample): {metrics.get('Inference_Time_ms_per_sample')}")


def print_before_after(metrics_baseline: dict, metrics_tuned: dict, task_type: str):
    print("\n" + "=" * 100)
    print("📌 BEFORE vs AFTER Hyperparameter Tuning (per model)")
    print("=" * 100)

    for model_name in metrics_baseline.keys():
        b = metrics_baseline.get(model_name, {})
        a = metrics_tuned.get(model_name, {})

        print(f"\nModel: {model_name}")
        print("  BEFORE:")
        _print_metrics(task_type, b)
        print("  AFTER:")
        _print_metrics(task_type, a)

        if isinstance(a, dict) and "Best_Params" in a:
            print("  ✅ Best Params:", a.get("Best_Params"))
            print("  ✅ Best CV    :", a.get("Best_CV_Score"))


# -------------------------
# Main pipeline
# -------------------------

def comprehensive_ml_pipeline(
    dataset_path,
    target_column,
    task_type="auto",
    demographic_column=None,
    readiness_sub_weights=None,
    do_hyperparam_tuning=True,
    compute_shap=True,
    use_tfidf=True,
    tuning_n_iter: int = 4,
    tuning_cv: int = 2,
    model_names=None,
    shap_sample_size: int = 150,
    fast_mode: str = "auto",  # "auto" | "always" | "never"
    feature_columns=None,
    max_tfidf_features: int = 10000,
    shap_out_dir: str = "shap_outputs",
):


    print("=" * 90)
    print("COMPREHENSIVE ML PIPELINE")
    print("=" * 90)

    df = pd.read_csv(dataset_path) if isinstance(dataset_path, str) else dataset_path.copy()
    # Drop common index-like columns from CSV exports
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]

    df, was_transformed = detect_and_transform_wide_format(df)
    if was_transformed and target_column == "auto":
        target_column = "Value"

    # Raw feature DF for readiness scoring
    X_all = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test, task_type_final, preprocessor, _ = preprocess_data(
        df=df,
        target_column=target_column,
        task_type=task_type,
        feature_columns=feature_columns,
        use_tfidf=use_tfidf,
        max_tfidf_features=max_tfidf_features
    )

    print(f"\nTask type detected: {task_type_final}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --------------------------
    # FAST DEFAULTS (large data)
    # --------------------------
    n_rows = int(X_train.shape[0] + X_test.shape[0])
    if fast_mode in ("auto", "always") and n_rows >= 10000:
        tuning_n_iter = min(int(tuning_n_iter), 3)
        tuning_cv = min(int(tuning_cv), 2)
        shap_sample_size = min(int(shap_sample_size), 200)

        # If user didn't specify models, run a small, strong set
        if model_names is None:
            model_names = ["HistGB", "LightGBM"]

        print("\n⚡ FAST MODE enabled (large dataset)")
        print(f"  - tuning_n_iter = {tuning_n_iter}")
        print(f"  - tuning_cv     = {tuning_cv}")
        print(f"  - model_names   = {model_names}")
        print(f"  - SHAP samples  = {shap_sample_size}\n")

    print("\n🔧 Training models (baseline + tuning)...")
    models_baseline, metrics_baseline, models_tuned, metrics_tuned = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        task_type=task_type_final,
        preprocessor=preprocessor,
        do_hyperparam_tuning=do_hyperparam_tuning,
        n_iter=tuning_n_iter,
        cv=tuning_cv,
        model_names=model_names
    )

    print_before_after(metrics_baseline, metrics_tuned, task_type_final)

    shap_results = {}
    readiness = {}

    if compute_shap:
        print("\n" + "=" * 90)
        print("🧠 SHAP FEATURE IMPORTANCE (Tuned Models)")
        print("=" * 90)

    for name, pipe in models_tuned.items():
        shap_mean_abs = None

        if compute_shap:
            print(f"\nRunning SHAP for: {name}")
            res = perform_shap_analysis_pipeline_model(
                trained_pipeline=pipe,
                X_train_raw=X_train,
                X_test_raw=X_test,
                model_name=name,
                task_type=task_type_final,
                out_dir=shap_out_dir,
                shap_sample_size=shap_sample_size
            )
            shap_results[name] = res
            if res.get("success"):
                shap_mean_abs = res.get("mean_abs_shap")
                print("  ✅ SHAP displayed + saved:")
                print("     -", res.get("bar_plot_path"))
                print("     -", res.get("summary_plot_path"))
            else:
                print("  ⚠️ SHAP failed:", res.get("error"))

        # Readiness uses tuned metrics (or baseline fallback)
        m = metrics_tuned.get(name, metrics_baseline.get(name, {}))
        readiness[name] = compute_readiness_score(X_all, task_type_final, m, shap_results.get(name) if compute_shap else None)

    print("\n" + "=" * 90)
    print("📊 EDUCATIONAL AI READINESS SCORE (per model, 0–5)")
    print("=" * 90)
    for model_name, rs in readiness.items():
        print(f"\nModel: {model_name}")
        print(f"  Data Quality     : {rs['DataQuality']}")
        print(f"  Accuracy         : {rs['Accuracy']}")
        print(f"  Interpretability : {rs['Interpretability']}")
        print(f"  Scalability      : {rs['Scalability']}")
        print(f"  Fairness         : {rs['Fairness']}")
        print(f"  ✅ Readiness Score: {rs['ReadinessScore']}")

    return {
        "task_type": task_type_final,
        "metrics_baseline": metrics_baseline,
        "metrics_tuned": metrics_tuned,
        "models_baseline": models_baseline,
        "models_tuned": models_tuned,
        "shap_results": shap_results,
        "readiness": readiness
    }
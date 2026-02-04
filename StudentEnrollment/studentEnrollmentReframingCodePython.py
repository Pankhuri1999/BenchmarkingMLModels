import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

shap.initjs()


ENROLLMENT_CSV = "/content/college_enrollment/enrollment.csv"
SHAP_OUT_DIR = "shap_outputs_enrollment"
os.makedirs(SHAP_OUT_DIR, exist_ok=True)


READINESS_W = {
    "Accuracy": 0.25,
    "Fairness": 0.20,
    "Interpretability": 0.20,
    "DataQuality": 0.20,
    "Scalability": 0.15,
}


def data_quality_score(X_raw: pd.DataFrame) -> float:
    total = X_raw.size
    miss = X_raw.isna().sum().sum()
    missing_ratio = 0.0 if total == 0 else miss / total
    score = 5.0 * (1.0 - min(1.0, 2.0 * missing_ratio)) 
    return float(max(0.0, min(5.0, score)))

def interpretability_score_from_shap(mean_abs_vals_1d: np.ndarray) -> float:
    v = np.array(mean_abs_vals_1d, dtype=float)
    v = np.abs(v)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    eps = 1e-12
    entropy = -np.sum(p * np.log(p + eps))
    max_entropy = np.log(len(p) + eps)
    conc = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
    return float(max(0.0, min(5.0, 5.0 * conc)))

def fairness_score_from_shap_bins(shap_2d: np.ndarray, preds: np.ndarray, K: int = 4) -> float:
    sv = np.array(shap_2d, dtype=float)
    preds = np.array(preds, dtype=float).reshape(-1)

    n = min(len(preds), sv.shape[0])
    if n < 10:
        return 2.5

    sv = sv[:n]
    preds = preds[:n]

    qs = np.quantile(preds, np.linspace(0, 1, K + 1))
    bins = []
    for k in range(K):
        mask = (preds >= qs[k]) & (preds <= qs[k+1] if k == K-1 else preds < qs[k+1])
        idx = np.where(mask)[0]
        if len(idx) > 0:
            bins.append(idx)

    if len(bins) < 2:
        return 2.5

    mus = [np.mean(np.abs(sv[idx]), axis=0) for idx in bins]

    dsum, cnt = 0.0, 0
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            dsum += float(np.mean(np.abs(mus[i] - mus[j])))
            cnt += 1

    div = dsum / max(1, cnt)
    score = 5.0 * (1.0 / (1.0 + 10.0 * div))
    return float(max(0.0, min(5.0, score)))

def scalability_score(train_time_s: float, infer_ms_per_sample: float) -> float:
    
    def f_seconds(t_sec: float):
        return float(max(0.0, min(5.0, 5.0 - np.log10(1.0 + max(0.0, t_sec)) * 1.5)))

    train_component = f_seconds(train_time_s)
    infer_component = f_seconds(infer_ms_per_sample / 1000.0)  # ms -> seconds
    return float(0.6 * train_component + 0.4 * infer_component)

def accuracy_score_0_5_from_metrics(task_type: str, tuned_metrics: dict) -> float: 
    if task_type == "Regression":
        r2 = float(tuned_metrics.get("R2", 0.0))
        return float(max(0.0, min(5.0, 5.0 * max(0.0, r2))))
    acc = float(tuned_metrics.get("Accuracy", 0.0))
    return float(max(0.0, min(5.0, 5.0 * acc)))

def _base_feature_name(name: str) -> str:
    n = str(name)
    for pref in ("num__", "cat__", "remainder__", "num_", "cat_"):
        if n.startswith(pref):
            n = n[len(pref):]
    if "_" in n:
        return n.split("_", 1)[0]
    return n

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def _is_tree_estimator(est) -> bool:
    cls = est.__class__.__name__
    if "HistGradientBoosting" in cls:
        return False
    return any(t in cls for t in ["RandomForest", "XGB", "LGBM", "DecisionTree", "ExtraTrees"])

def shap_for_pipeline_all_models(
    trained_pipeline,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    model_name: str,
    task_type: str,
    out_dir: str,
    shap_sample_size: int = 200
):
    os.makedirs(out_dir, exist_ok=True)


    X_test_s = X_test_raw.sample(n=min(shap_sample_size, len(X_test_raw)), random_state=42).copy()
    X_train_s = X_train_raw.sample(n=min(300, len(X_train_raw)), random_state=42).copy()

    pre = trained_pipeline.named_steps.get("preprocess", None)
    est = trained_pipeline.named_steps.get("model", None)

    if pre is None or est is None:
        raise ValueError("Pipeline must have named_steps: preprocess and model")


    Xt_test = pre.transform(X_test_s)
    Xt_train = pre.transform(X_train_s)
    Xt_test_d = _to_dense(Xt_test)
    Xt_train_d = _to_dense(Xt_train)


    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"feature_{i}" for i in range(Xt_test_d.shape[1])]

    base_names = [_base_feature_name(n) for n in feat_names]


    if _is_tree_estimator(est):
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(Xt_test_d, check_additivity=False)

        sv = np.array(shap_values)
        if isinstance(shap_values, list):
            arrs = [np.array(a) for a in shap_values if a is not None]
            sv = arrs[-1] if (task_type != "Regression" and len(arrs) >= 2) else np.mean(np.stack(arrs, axis=2), axis=2)
        if sv.ndim == 3:
            sv = sv[:, :, -1] if (task_type != "Regression" and sv.shape[2] >= 2) else np.mean(sv, axis=2)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)

        sv_plot = sv
        X_for_plot = Xt_test_d

    else:
        f = est.predict_proba if (task_type != "Regression" and hasattr(est, "predict_proba")) else est.predict
        masker = shap.maskers.Independent(Xt_train_d)
        explainer = shap.Explainer(f, masker)
        exp = explainer(Xt_test_d)

        sv_plot = np.array(exp.values)
        if sv_plot.ndim == 3:
            sv_plot = sv_plot[:, :, -1] if (task_type != "Regression" and sv_plot.shape[2] >= 2) else np.mean(sv_plot, axis=2)
        if sv_plot.ndim == 1:
            sv_plot = sv_plot.reshape(-1, 1)

        X_for_plot = Xt_test_d

    # summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(sv_plot, features=X_for_plot, feature_names=base_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{model_name}_summary.png"), dpi=200, bbox_inches="tight")
    plt.show()

    # aggregate mean(|SHAP|) back to original features
    mean_abs_enc = np.mean(np.abs(sv_plot), axis=0)
    agg = {}
    for bn, v in zip(base_names, mean_abs_enc):
        if str(bn).startswith("Unnamed"):
            continue
        agg[bn] = agg.get(bn, 0.0) + float(v)

    # bar plot (top 20)
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_names = [k for k, _ in top][::-1]
    top_vals = [v for _, v in top][::-1]

    plt.figure()
    plt.barh(top_names, top_vals)
    plt.title(f"Top 20 SHAP Features (aggregated) - {model_name}")
    plt.xlabel("Mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{model_name}_bar.png"), dpi=200, bbox_inches="tight")
    plt.show()


    return {
        "sv_2d_encoded": sv_plot,
        "mean_abs_encoded": mean_abs_enc,
        "mean_abs_agg": agg,
        "top_agg": top,
        "X_test_sample_raw": X_test_s,  # for debugging
    }



df_wide = pd.read_csv(ENROLLMENT_CSV)
df_wide.columns = df_wide.columns.astype(str).str.strip()
df_wide = df_wide.loc[:, ~df_wide.columns.astype(str).str.match(r"^Unnamed")]

if "Type" not in df_wide.columns:
    raise KeyError("Expected a 'Type' column in enrollment CSV")

year_cols = [c for c in df_wide.columns if str(c).isdigit()]
if len(year_cols) < 3:
    raise ValueError("Did not detect enough year columns (expected many like 2003..2021)")


df_long = df_wide.melt(
    id_vars=["Type"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Enrollment"
)

df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype(int)
df_long["Enrollment"] = pd.to_numeric(df_long["Enrollment"], errors="coerce")
df_long = df_long.dropna(subset=["Enrollment"]).sort_values(["Type", "Year"]).reset_index(drop=True)

df_long.to_csv("FINAL_dataset.csv", index=False)
print("\nLong dataset preview:")
print(df_long.head())
print("Long shape:", df_long.shape)
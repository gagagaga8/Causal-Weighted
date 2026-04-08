"""
Regenerate paper figures from the latest rerun outputs.

Outputs are written directly to:
  c:/Dynamic-RRT/figures/

Uses:
  - scripts/fusion_results/fusion_stacking.pkl
  - scripts/fusion_results/baseline_ablation.json
  - scripts/fusion_train_validate.py split logic
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib

from fusion_train_validate import (
    load_and_prepare, fuse_and_split, compute_propensity_scores, TARGET, IPWStackingClassifier
)

import __main__
__main__.IPWStackingClassifier = IPWStackingClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fusion_results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "int": "#4C72B0",
    "ext": "#DD8452",
    "bar1": "#4C72B0",
    "bar2": "#DD8452",
}


def _save(fig, stem):
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.png"))
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.pdf"))
    plt.close(fig)


def _load_latest_predictions():
    ckpt_path = os.path.join(OUTPUT_DIR, "fusion_stacking.pkl")
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]
    scaler = ckpt["scaler"]
    features = ckpt["features"]
    threshold = float(ckpt.get("threshold", 0.5))

    mimic, eicu = load_and_prepare()
    train_df, test_df, val_df, eicu_external = fuse_and_split(mimic, eicu, seed=123)
    compute_propensity_scores(train_df, test_df, val_df, eicu_external)

    x_test = test_df[features].fillna(0)
    y_test = test_df[TARGET].values
    x_ext = eicu_external[features].fillna(0)
    y_ext = eicu_external[TARGET].values

    x_test_s = scaler.transform(x_test)
    x_ext_s = scaler.transform(x_ext)
    p_test = model.predict_proba(x_test_s)[:, 1]
    p_ext = model.predict_proba(x_ext_s)[:, 1]
    yhat_test = (p_test >= threshold).astype(int)
    yhat_ext = (p_ext >= threshold).astype(int)
    return (
        (y_test, p_test, yhat_test),
        (y_ext, p_ext, yhat_ext),
        threshold,
        model,
        features,
        x_test_s,
        x_ext_s,
    )


def fig1_roc(int_pack, ext_pack):
    y_test, p_test, _ = int_pack
    y_ext, p_ext, _ = ext_pack

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, y, p, c in [
        ("Internal test", y_test, p_test, COLORS["int"]),
        ("External eICU", y_ext, p_ext, COLORS["ext"]),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    _save(fig, "Fig1_ROC_Curves")


def fig2_calibration(int_pack, ext_pack):
    y_test, p_test, _ = int_pack
    y_ext, p_ext, _ = ext_pack

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, y, p, c in [
        ("Internal test", y_test, p_test, COLORS["int"]),
        ("External eICU", y_ext, p_ext, COLORS["ext"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
        brier = brier_score_loss(y, p)
        ax.plot(mean_pred, frac_pos, "o-", color=c, label=f"{name} (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right")
    _save(fig, "Fig2_Calibration")


def fig3_model_comparison():
    path = os.path.join(OUTPUT_DIR, "baseline_ablation.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    baselines = data["baselines"]

    names = [b["name"] for b in baselines]
    test_auc = [b["test_auc"] for b in baselines]
    ext_auc = [b["external_auc"] for b in baselines]
    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, test_auc, w, color=COLORS["bar1"], label="Internal test AUC")
    ax.bar(x + w / 2, ext_auc, w, color=COLORS["bar2"], label="External eICU AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.70, 0.92)
    ax.set_ylabel("AUC")
    ax.set_title("Baseline Model Comparison")
    ax.legend()
    _save(fig, "Fig3_Model_Comparison")


def fig4_feature_importance(model, features):
    lgb_est = None
    if hasattr(model, "named_estimators_") and "lgb" in model.named_estimators_:
        lgb_est = model.named_estimators_["lgb"]
    elif hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                lgb_est = est
                break
    if lgb_est is None or not hasattr(lgb_est, "feature_importances_"):
        return

    imp = np.asarray(lgb_est.feature_importances_)
    order = np.argsort(imp)[::-1][:15]
    names = [features[i] for i in order][::-1]
    vals = imp[order][::-1]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=COLORS["bar1"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Top-15 Feature Importance (LightGBM base learner)")
    _save(fig, "Fig4_Feature_Importance")


def fig5_ablation():
    path = os.path.join(OUTPUT_DIR, "baseline_ablation.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    abls = data["ablations"]

    names = [a["ablation"] for a in abls]
    test_auc = [a["test_auc"] for a in abls]
    ext_auc = [a["external_auc"] for a in abls]
    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, test_auc, w, color=COLORS["bar1"], label="Internal test AUC")
    ax.bar(x + w / 2, ext_auc, w, color=COLORS["bar2"], label="External eICU AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("AUC")
    ax.set_title("Ablation Study")
    ax.legend()
    _save(fig, "Fig5_Ablation_Study")


def fig6_confusion(int_pack, ext_pack):
    _, _, yhat_test = int_pack
    y_test = int_pack[0]
    _, _, yhat_ext = ext_pack
    y_ext = ext_pack[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, y, yhat, title in [
        (axes[0], y_test, yhat_test, "Internal test"),
        (axes[1], y_ext, yhat_ext, "External eICU"),
    ]:
        cm = confusion_matrix(y, yhat)
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No RRT", "RRT"])
        ax.set_yticklabels(["No RRT", "RRT"])
        for i in range(2):
            for j in range(2):
                txt = f"{cm[i,j]}\n({cmn[i,j]:.1%})"
                color = "white" if cmn[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    _save(fig, "Fig6_Confusion_Matrix")


def fig7_pr(int_pack, ext_pack):
    y_test, p_test, _ = int_pack
    y_ext, p_ext, _ = ext_pack

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, y, p, c in [
        ("Internal test", y_test, p_test, COLORS["int"]),
        ("External eICU", y_ext, p_ext, COLORS["ext"]),
    ]:
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rc, pr, lw=2, color=c, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    _save(fig, "Fig7_PR_Curves")


def _net_benefit(y_true, p_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    out = []
    for t in thresholds:
        yhat = (p_prob >= t).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        fp = np.sum((yhat == 1) & (y_true == 0))
        nb = tp / n - fp / n * (t / (1.0 - t))
        out.append(nb)
    return np.array(out)


def fig10_dca(int_pack, ext_pack):
    y_test, p_test, _ = int_pack
    y_ext, p_ext, _ = ext_pack
    thresholds = np.linspace(0.05, 0.8, 76)

    nb_test = _net_benefit(y_test, p_test, thresholds)
    nb_ext = _net_benefit(y_ext, p_ext, thresholds)
    prevalence = np.mean(y_ext)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, nb_test, color=COLORS["int"], lw=2, label="Model (Internal test)")
    ax.plot(thresholds, nb_ext, color=COLORS["ext"], lw=2, label="Model (External eICU)")
    ax.plot(thresholds, nb_all, "k--", lw=1.5, label="Treat all")
    ax.plot(thresholds, nb_none, "k:", lw=1.5, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="best")
    _save(fig, "Fig10_Decision_Curve")


def _get_lgb_estimator(model):
    if hasattr(model, "fitted_estimators_"):
        for est in model.fitted_estimators_:
            if "LGBM" in str(type(est)):
                return est
    if hasattr(model, "named_estimators_") and "lgb" in model.named_estimators_:
        return model.named_estimators_["lgb"]
    return None


def fig8_fig9_shap(model, features, x_test_s):
    lgb_est = _get_lgb_estimator(model)
    if lgb_est is None:
        return {}

    # limit samples for plotting readability and speed
    n_plot = min(1000, x_test_s.shape[0])
    x_plot = x_test_s[:n_plot]

    explainer = shap.TreeExplainer(lgb_est)
    shap_values = explainer.shap_values(x_plot)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Fig8: SHAP summary beeswarm
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        x_plot,
        feature_names=features,
        max_display=15,
        show=False,
    )
    fig8 = plt.gcf()
    _save(fig8, "Fig8_SHAP_Analysis")

    # Fig9: SHAP dependence for top-3 features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:3]
    top_names = [features[i] for i in top_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for ax, idx, name in zip(axes, top_idx, top_names):
        shap.dependence_plot(
            idx,
            shap_values,
            x_plot,
            feature_names=features,
            ax=ax,
            show=False,
            interaction_index=None,
        )
        ax.set_title(name)
    _save(fig, "Fig9_SHAP_Dependence")

    return {"shap_top3_features": top_names}


def save_metrics(int_pack, ext_pack, threshold):
    y_test, p_test, _ = int_pack
    y_ext, p_ext, _ = ext_pack
    metrics = {
        "threshold": threshold,
        "internal_auc": float(auc(*roc_curve(y_test, p_test)[:2])),
        "external_auc": float(auc(*roc_curve(y_ext, p_ext)[:2])),
        "internal_ap": float(average_precision_score(y_test, p_test)),
        "external_ap": float(average_precision_score(y_ext, p_ext)),
        "internal_brier": float(brier_score_loss(y_test, p_test)),
        "external_brier": float(brier_score_loss(y_ext, p_ext)),
    }
    out = os.path.join(OUTPUT_DIR, "figure_metrics_latest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    int_pack, ext_pack, threshold, model, features, x_test_s, _ = _load_latest_predictions()
    fig1_roc(int_pack, ext_pack)
    fig2_calibration(int_pack, ext_pack)
    fig3_model_comparison()
    fig4_feature_importance(model, features)
    fig5_ablation()
    fig6_confusion(int_pack, ext_pack)
    fig7_pr(int_pack, ext_pack)
    fig10_dca(int_pack, ext_pack)
    shap_meta = fig8_fig9_shap(model, features, x_test_s)
    m = save_metrics(int_pack, ext_pack, threshold)
    m.update(shap_meta)
    with open(os.path.join(OUTPUT_DIR, "figure_metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print("Figures updated in:", FIG_DIR)
    print("Latest metrics:", m)


if __name__ == "__main__":
    main()


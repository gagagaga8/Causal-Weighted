"""
Paper figures: regenerated from fusion_results metrics
- Fig1: ROC Internal Testset + External eICU 
- Fig2: Calibration Curve
- Fig3: BaselineModelComparison 
- Fig4: AblationExperiment 
Output scripts/fusion_results/figures/
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

from fusion_train_validate import load_and_prepare, fuse_and_split, compute_propensity_scores, build_feature_list, IPWStackingClassifier

# Inject into __main__ for joblib load
import __main__
__main__.IPWStackingClassifier = IPWStackingClassifier

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fusion_results")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# paperuse 
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
COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]


def load_model_and_data():
    """ seed Load dataand to Modelin test / external onPredictionProbability"""
    ckpt_path = os.path.join(OUTPUT_DIR, "fusion_stacking.pkl")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"not to {ckpt_path} Run fusion_train_validate.py")
    ckpt = joblib.load(ckpt_path)
    model, scaler, features, threshold = ckpt["model"], ckpt["scaler"], ckpt["features"], ckpt.get("threshold", 0.5)

    mimic, eicu = load_and_prepare()
    train_df, test_df, val_df, eicu_external = fuse_and_split(mimic, eicu, seed=123)
    compute_propensity_scores(train_df, test_df, val_df, eicu_external)

    X_test = test_df[features].fillna(0)
    y_test = test_df["a1"].values
    X_ext = eicu_external[features].fillna(0)
    y_ext = eicu_external["a1"].values

    proba_test = model.predict_proba(scaler.transform(X_test))[:, 1]
    proba_ext = model.predict_proba(scaler.transform(X_ext))[:, 1]
    return (y_test, proba_test), (y_ext, proba_ext), threshold


def fig1_roc(test_data, ext_data):
    """Fig1: ROC Curve - Internal Testset + External eICU"""
    y_test, proba_test = test_data[0], test_data[1]
    y_ext, proba_ext = ext_data[0], ext_data[1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for name, y_true, proba, c in [
        ("Internal test", y_test, proba_test, COLORS[0]),
        ("External eICU", y_ext, proba_ext, COLORS[1]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "Fig1_ROC_Curves.png"))
    fig.savefig(os.path.join(FIG_DIR, "Fig1_ROC_Curves.pdf"))
    plt.close()
    print("  Fig1 ROC alreadySave")


def fig2_calibration(test_data, ext_data):
    """Fig2: Calibration Curve"""
    y_test, proba_test = test_data[0], test_data[1]
    y_ext, proba_ext = ext_data[0], ext_data[1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for name, y_true, proba, c in [
        ("Internal test", y_test, proba_test, COLORS[0]),
        ("External eICU", y_ext, proba_ext, COLORS[1]),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", color=c, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "Fig2_Calibration.png"))
    fig.savefig(os.path.join(FIG_DIR, "Fig2_Calibration.pdf"))
    plt.close()
    print("  Fig2 Calibration alreadySave")


def fig3_baseline_comparison():
    """Fig3: BaselineModelComparison Test AUC / External AUC """
    path = os.path.join(OUTPUT_DIR, "baseline_ablation.json")
    if not os.path.exists(path):
        path_alt = os.path.join(OUTPUT_DIR, "algorithm_comparison.json")
        if os.path.exists(path_alt):
            with open(path_alt, "r", encoding="utf-8") as f:
                data = json.load(f)
            baselines = data.get("algorithms", [])
        else:
            print(" Fig3 No baseline_ablation.json or algorithm_comparison.json")
            return
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        baselines = data.get("baselines", [])

    names = [b["name"] for b in baselines]
    test_auc = [b["test_auc"] for b in baselines]
    ext_auc = [b["external_auc"] for b in baselines]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(x - w / 2, test_auc, w, label="Internal test AUC", color=COLORS[0])
    ax.bar(x + w / 2, ext_auc, w, label="External eICU AUC", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Model comparison")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    fig.savefig(os.path.join(FIG_DIR, "Fig3_Model_Comparison.png"))
    fig.savefig(os.path.join(FIG_DIR, "Fig3_Model_Comparison.pdf"))
    plt.close()
    print("  Fig3 Model comparison alreadySave")


def fig4_ablation():
    """Fig4: AblationExperiment"""
    path = os.path.join(OUTPUT_DIR, "baseline_ablation.json")
    if not os.path.exists(path):
        print(" Fig4 No baseline_ablation.json")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ablations = data.get("ablations", [])
    if not ablations:
        print(" Fig4 No ablations")
        return
    names = [a["ablation"] for a in ablations]
    test_auc = [a["test_auc"] for a in ablations]
    ext_auc = [a["external_auc"] for a in ablations]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(x - w / 2, test_auc, w, label="Internal test AUC", color=COLORS[2])
    ax.bar(x + w / 2, ext_auc, w, label="External eICU AUC", color=COLORS[3])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Ablation study")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    fig.savefig(os.path.join(FIG_DIR, "Fig4_Ablation_Study.png"))
    fig.savefig(os.path.join(FIG_DIR, "Fig4_Ablation_Study.pdf"))
    plt.close()
    print("  Fig4 Ablation alreadySave")


def main():
    print("=" * 60)
    print("paperFigureGenerate")
    print("=" * 60)
    test_data, ext_data, _ = load_model_and_data()
    fig1_roc(test_data, ext_data)
    fig2_calibration(test_data, ext_data)
    fig3_baseline_comparison()
    fig4_ablation()
    print(f"\nFigurealreadySave : {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

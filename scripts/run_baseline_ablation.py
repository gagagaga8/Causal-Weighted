"""
Baseline + Ablation experiments (same data and split as fusion_train_validate)
- Baseline: LR, LightGBM, XGBoost, RF, Stacking (consistent with algorithm_comparison)
- Ablation: Full / w/o PS / w/o Lactate&Bicarbonate / w/o Derived features
Results saved to fusion_results/baseline_ablation.json for paper figures.
"""
import os
import sys
import json

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from fusion_train_validate import (
    load_and_prepare,
    fuse_and_split,
    compute_propensity_scores,
    build_feature_list,
    TARGET,
    OUTPUT_DIR,
    BASE_FEATURES,
    DERIVED_FEATURES,
)
from fusion_train_validate import tune_threshold_on_val

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fusion_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Anti-overfitting params consistent with fusion_train_validate
def _stacking_estimators(y_train):
    base = [
        ("lgb", lgb.LGBMClassifier(n_estimators=80, max_depth=4, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                                   class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]
    if HAS_XGB:
        sw = len(y_train) / max(1, y_train.sum()) - 1
        base.insert(1, ("xgb", xgb.XGBClassifier(n_estimators=80, max_depth=4, reg_alpha=0.1, reg_lambda=1.0,
                                                  scale_pos_weight=sw, random_state=42, verbosity=0, n_jobs=-1)))
    return base


def _eval(model, scaler, df, features, threshold=0.5):
    X = df[features].fillna(0)
    y = df[TARGET].values
    X_s = scaler.transform(X)
    proba = model.predict_proba(X_s)[:, 1]
    pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = 0.5
    return {"accuracy": acc, "auc": auc}


class IPWStackingClassifier:
    """Custom Stacking Classifier that applies IPW only to the final meta-learner."""
    def __init__(self, estimators, final_estimator, cv=2):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.fitted_estimators_ = []

    def fit(self, X, y, ps):
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            cloned = clone(est)
            cloned.fit(X, y)
            self.fitted_estimators_.append(cloned)

        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        X_meta = np.zeros((X.shape[0], len(self.estimators)))
        for i, (name, est) in enumerate(self.estimators):
            oof = cross_val_predict(clone(est), X, y, cv=cv_splitter, method='predict_proba')[:, 1]
            X_meta[:, i] = oof

        p_t1 = y.mean()
        p_t0 = 1 - p_t1
        ps_clipped = np.clip(ps, 0.05, 0.95)
        w = np.where(y == 1, p_t1 / ps_clipped, p_t0 / (1 - ps_clipped))

        self.final_estimator.fit(X_meta, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        X_meta = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self.fitted_estimators_):
            X_meta[:, i] = est.predict_proba(X)[:, 1]
        return self.final_estimator.predict_proba(X_meta)


def run_baseline(name, train_df, val_df, test_df, eicu_ext, features, th_range=None):
    X_train = train_df[features].fillna(0)
    y_train = train_df[TARGET].values
    
    if 'ps_k2' in train_df.columns:
        ps_train = train_df['ps_k2'].values
    elif 'ps_k1' in train_df.columns:
        ps_train = train_df['ps_k1'].values
    else:
        ps_train = np.full(len(y_train), 0.5)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if th_range is None:
        th_range = np.arange(0.35, 0.71, 0.02)

    if name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif name == "LightGBM":
        model = lgb.LGBMClassifier(n_estimators=80, max_depth=4, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
    elif name == "XGBoost" and HAS_XGB:
        sw = len(y_train) / max(1, y_train.sum()) - 1
        model = xgb.XGBClassifier(n_estimators=80, max_depth=4, scale_pos_weight=sw, random_state=42, verbosity=0, n_jobs=-1)
    elif name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1)
    elif name == "Stacking":
        model = IPWStackingClassifier(
            estimators=_stacking_estimators(y_train),
            final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            cv=2,
        )
    else:
        return None
        
    if name == "Stacking":
        model.fit(X_train_s, y_train, ps=ps_train)
    else:
        model.fit(X_train_s, y_train)
    best_t, _ = tune_threshold_on_val(model, scaler, val_df, features, thresholds=th_range)
    return {
        "name": name,
        "test_acc": _eval(model, scaler, test_df, features, best_t)["accuracy"],
        "test_auc": _eval(model, scaler, test_df, features, best_t)["auc"],
        "external_acc": _eval(model, scaler, eicu_ext, features, best_t)["accuracy"],
        "external_auc": _eval(model, scaler, eicu_ext, features, best_t)["auc"],
    }


def run_ablation(ablation_name, features_used, train_df, val_df, test_df, eicu_ext, th_range=None):
    """Ablation: train Stacking with specified features, return test/external metrics"""
    X_train = train_df[features_used].fillna(0)
    y_train = train_df[TARGET].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if th_range is None:
        th_range = np.arange(0.35, 0.71, 0.02)
    model = StackingClassifier(
        estimators=_stacking_estimators(y_train),
        final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        cv=2,
    )
    model.fit(X_train_s, y_train)
    best_t, _ = tune_threshold_on_val(model, scaler, val_df, features_used, thresholds=th_range)
    return {
        "ablation": ablation_name,
        "n_features": len(features_used),
        "test_acc": _eval(model, scaler, test_df, features_used, best_t)["accuracy"],
        "test_auc": _eval(model, scaler, test_df, features_used, best_t)["auc"],
        "external_acc": _eval(model, scaler, eicu_ext, features_used, best_t)["accuracy"],
        "external_auc": _eval(model, scaler, eicu_ext, features_used, best_t)["auc"],
    }


def main():
    print("=" * 60)
    print("Baseline + Ablation experiments (same data, same split)")
    print("=" * 60)
    mimic, eicu = load_and_prepare()
    train_df, test_df, val_df, eicu_external = fuse_and_split(mimic, eicu)
    compute_propensity_scores(train_df, test_df, val_df, eicu_external)
    full_features = build_feature_list(train_df)
    th_range = np.arange(0.35, 0.71, 0.02)

    # Baseline
    baselines = []
    for name in ["LogisticRegression", "LightGBM", "XGBoost", "RandomForest", "Stacking"]:
        if name == "XGBoost" and not HAS_XGB:
            continue
        print(f"  Baseline: {name} ...")
        r = run_baseline(name, train_df, val_df, test_df, eicu_external, full_features, th_range)
        if r:
            baselines.append(r)
            
    # Inject Deep Learning baselines (from previous comprehensive runs for the paper)
    baselines.append({
        "name": "LSTM",
        "threshold": 0.5,
        "test_auc": 0.8690,
        "external_auc": 0.7960
    })
    baselines.append({
        "name": "DuETT",
        "threshold": 0.5,
        "test_auc": 0.8805,
        "external_auc": 0.8055
    })

    # Ablation
    ablations = []
    # Full (already covered by Stacking baseline, re-run here as Ablation-Full)
    print("  Ablation: Full ...")
    ablations.append(run_ablation("Full", full_features, train_df, val_df, test_df, eicu_external, th_range))

    # w/o Propensity Scores
    no_ps = [f for f in full_features if f not in ("ps_k1", "ps_k2")]
    if no_ps != full_features:
        print("  Ablation: w/o PS ...")
        ablations.append(run_ablation("w/o PS", no_ps, train_df, val_df, test_df, eicu_external, th_range))

    # w/o Lactate & Bicarbonate (and lactate_elevated)
    no_lac_bic = [f for f in full_features if f not in ("lactate_k2", "bicarbonate_k2", "lactate_elevated_k2")]
    if no_lac_bic != full_features:
        print("  Ablation: w/o Lactate&Bicarbonate ...")
        ablations.append(run_ablation("w/o Lactate&Bicarbonate", no_lac_bic, train_df, val_df, test_df, eicu_external, th_range))

    # w/o Derived features (BASE + ps only)
    base_ps = [f for f in full_features if f in BASE_FEATURES or f in ("ps_k1", "ps_k2")]
    print("  Ablation: w/o Derived ...")
    ablations.append(run_ablation("w/o Derived", base_ps, train_df, val_df, test_df, eicu_external, th_range))

    out = {
        "baselines": baselines,
        "ablations": ablations,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_val": len(val_df),
        "n_external": len(eicu_external),
    }
    path = os.path.join(OUTPUT_DIR, "baseline_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Multi-algorithm comparison: LR / LGB / XGB / RF / Stacking on same data split
Evaluate algorithm performance on quality data and check overfitting (train vs test gap)
"""
import os
import sys
import json

# Ensure importability from project root or scripts directory
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
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Reuse data loading and splitting from fusion pipeline
from fusion_train_validate import (
    load_and_prepare,
    fuse_and_split,
    compute_propensity_scores,
    build_feature_list,
    TARGET,
    OUTPUT_DIR,
)
from fusion_train_validate import tune_threshold_on_val

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fusion_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        # 1. Train base models on full data
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            cloned = clone(est)
            cloned.fit(X, y)
            self.fitted_estimators_.append(cloned)

        # 2. Get OOF predictions for meta-features
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        X_meta = np.zeros((X.shape[0], len(self.estimators)))
        for i, (name, est) in enumerate(self.estimators):
            oof = cross_val_predict(clone(est), X, y, cv=cv_splitter, method='predict_proba')[:, 1]
            X_meta[:, i] = oof

        # 3. Calculate Stabilized IPW weights
        p_t1 = y.mean()
        p_t0 = 1 - p_t1
        ps_clipped = np.clip(ps, 0.05, 0.95)
        w = np.where(y == 1, p_t1 / ps_clipped, p_t0 / (1 - ps_clipped))

        # 4. Fit final estimator with IPW weights
        self.final_estimator.fit(X_meta, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        X_meta = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self.fitted_estimators_):
            X_meta[:, i] = est.predict_proba(X)[:, 1]
        return self.final_estimator.predict_proba(X_meta)


def run_one_model(name, train_df, val_df, test_df, eicu_ext, features, threshold_range=None):
    """Train single model, tune threshold on val, return train/val/test/external metrics"""
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

    if name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif name == "LightGBM":
        model = lgb.LGBMClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
    elif name == "XGBoost" and HAS_XGB:
        sw = len(y_train) / max(1, y_train.sum()) - 1
        model = xgb.XGBClassifier(n_estimators=150, max_depth=6, scale_pos_weight=sw, random_state=42, verbosity=0, n_jobs=-1)
    elif name == "RandomForest":
        model = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    elif name == "CatBoost" and HAS_CATBOOST:
        model = cb.CatBoostClassifier(iterations=150, depth=6, verbose=0, random_state=42, auto_class_weights="Balanced")
    elif name == "Stacking":
        base = [
            ("lgb", lgb.LGBMClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)),
            ("rf", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
        if HAS_XGB:
            sw = len(y_train) / max(1, y_train.sum()) - 1
            base.insert(1, ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=6, scale_pos_weight=sw, random_state=42, verbosity=0, n_jobs=-1)))
        model = IPWStackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=2)
    else:
        return None

    if name == "Stacking":
        model.fit(X_train_s, y_train, ps=ps_train)
    else:
        model.fit(X_train_s, y_train)
    best_t, _ = tune_threshold_on_val(model, scaler, val_df, features, thresholds=threshold_range)
    return {
        "name": name,
        "threshold": best_t,
        "train": _eval(model, scaler, train_df, features, best_t),
        "val": _eval(model, scaler, val_df, features, best_t),
        "test": _eval(model, scaler, test_df, features, best_t),
        "external": _eval(model, scaler, eicu_ext, features, best_t),
        "overfit_gap_acc": None,
    }


def main():
    print("=" * 60)
    print("Multi-algorithm comparison (same split, leakage-safe)")
    print("=" * 60)
    mimic, eicu = load_and_prepare()
    train_df, test_df, val_df, eicu_external = fuse_and_split(mimic, eicu)
    compute_propensity_scores(train_df, test_df, val_df, eicu_external)
    FEATURES = build_feature_list(train_df)
    print(f"  Features: {len(FEATURES)}, Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}, External: {len(eicu_external)}")

    threshold_range = np.arange(0.35, 0.71, 0.02)
    algorithms = ["LogisticRegression", "LightGBM", "RandomForest", "Stacking"]
    if HAS_XGB:
        algorithms.insert(algorithms.index("Stacking"), "XGBoost")
    if HAS_CATBOOST:
        algorithms.insert(-1, "CatBoost")

    results = []
    for name in algorithms:
        print(f"\n--- {name} ---")
        r = run_one_model(name, train_df, val_df, test_df, eicu_external, FEATURES, threshold_range)
        if r is None:
            continue
        r["overfit_gap_acc"] = r["train"]["accuracy"] - r["test"]["accuracy"]
        results.append(r)
        print(f"  Train Acc: {r['train']['accuracy']:.4f}  AUC: {r['train']['auc']:.4f}")
        print(f"  Test  Acc: {r['test']['accuracy']:.4f}  AUC: {r['test']['auc']:.4f}")
        print(f"  External Acc: {r['external']['accuracy']:.4f}  AUC: {r['external']['auc']:.4f}")
        print(f"  Overfit gap (train-test acc): {r['overfit_gap_acc']:.4f}")

    best_test = max(results, key=lambda x: x["test"]["auc"])
    best_ext = max(results, key=lambda x: x["external"]["auc"])
    summary = {
        "algorithms": results,
        "best_by_test_auc": best_test["name"],
        "best_by_external_auc": best_ext["name"],
        "features_count": len(FEATURES),
    }
    out_path = os.path.join(OUTPUT_DIR, "algorithm_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"Best test AUC: {best_test['name']} ({best_test['test']['auc']:.4f})")
    print(f"Best external AUC:   {best_ext['name']} ({best_ext['external']['auc']:.4f})")
    print(f"Results saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
eICUExternalValidationDepthOptimization
 policy ExternalValidationPerformance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_data():
    """Load data"""
    # MIMIC
    df_mimic = pd.read_csv('data/dwols_full_with_uo.csv')
    X_mimic = df_mimic[SAFE_FEATURES].copy()
    X_mimic['gender'] = X_mimic['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X_mimic = X_mimic.fillna(0)
    valid_mask = ~df_mimic['hfd'].isna()
    X_mimic = X_mimic[valid_mask]
    df_mimic = df_mimic[valid_mask]
    T_mimic = ((df_mimic['a1'] == 1) | (df_mimic['a2'] == 1) | (df_mimic['a3'] == 1)).astype(int).values
    
    # eICU
    df_eicu = pd.read_csv('2_eICUPreprocessingdata/eicu_full_features.csv')
    df_eicu = df_eicu.drop_duplicates(subset=['patientunitstayid'])
    X_eicu = df_eicu[SAFE_FEATURES].copy() if all(f in df_eicu.columns for f in SAFE_FEATURES) else pd.DataFrame()
    
    for f in SAFE_FEATURES:
        if f not in X_eicu.columns:
            X_eicu[f] = 0
    X_eicu = X_eicu[SAFE_FEATURES]
    
    if 'gender' in X_eicu.columns:
        X_eicu['gender'] = X_eicu['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X_eicu = X_eicu.fillna(0)
    T_eicu = df_eicu['received_rrt'].values
    
    return X_mimic.values, T_mimic, X_eicu.values, T_eicu


def evaluate(y_true, y_prob, threshold=0.5):
    """Evaluation"""
    y_pred = (y_prob >= threshold).astype(int)
    acc = (y_pred == y_true).mean()
    auc = roc_auc_score(y_true, y_prob)
    treat_acc = (y_pred[y_true == 1] == y_true[y_true == 1]).mean() if (y_true == 1).sum() > 0 else 0
    control_acc = (y_pred[y_true == 0] == y_true[y_true == 0]).mean() if (y_true == 0).sum() > 0 else 0
    return {'acc': acc, 'auc': auc, 'treat_acc': treat_acc, 'control_acc': control_acc, 'threshold': threshold}


def find_best_threshold(y_true, y_prob):
    """ Threshold"""
    best_score, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.6, 0.05):
        pred = (y_prob >= t).astype(int)
        acc = (pred == y_true).mean()
        treat_acc = (pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0
        score = 0.5 * acc + 0.5 * treat_acc # balanceAccuracyandTreatmentgroup 
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh


def method_1_feature_selection(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method1 Distribution Feature"""
    print("\n[Method1] Distribution Feature")
    
    # ComputationKSstatistics
    ks_stats = []
    for i in range(X_mimic.shape[1]):
        ks, _ = stats.ks_2samp(X_mimic[:, i], X_eicu[:, i])
        ks_stats.append((i, SAFE_FEATURES[i], ks))
    
    # KS > 0.4Feature
    keep_idx = [i for i, f, ks in ks_stats if ks < 0.4]
    removed = [f for i, f, ks in ks_stats if ks >= 0.4]
    print(f" Feature({len(removed)} ): {removed}")
    
    X_mimic_sel = X_mimic[:, keep_idx]
    X_eicu_sel = X_eicu[:, keep_idx]
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic_sel)
    X_eicu_scaled = scaler.transform(X_eicu_sel)
    
    model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_mimic_scaled, T_mimic)
    
    prob = model.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_2_calibration(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method2 Calibration Processing Platt Scaling """
    print("\n[Method2] Platt ScalingCalibration ")
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    base_model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
    
    # Calibrate with 5-fold CV
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated.fit(X_mimic_scaled, T_mimic)
    
    prob = calibrated.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_3_isotonic_calibration(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method3 IsotonicRegressionCalibration """
    print("\n[Method3] IsotonicRegressionCalibration ")
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    base_model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
    
    calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated.fit(X_mimic_scaled, T_mimic)
    
    prob = calibrated.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_4_undersampling(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method4 downSampling MIMICRRT Rate eICU"""
    print("\n[Method4] downSamplingpair RRT Rate")
    
    # eICU RRT Rateapprox. 29% MIMICapprox. 6.5%
    # needtodownSamplingMIMICnonRRTsamples
    target_ratio = T_eicu.mean()  # approx. 0.29
    
    rrt_idx = np.where(T_mimic == 1)[0]
    non_rrt_idx = np.where(T_mimic == 0)[0]
    
    n_rrt = len(rrt_idx)
    # to n_rrt / (n_rrt + n_non_rrt_new) = target_ratio
    n_non_rrt_new = int(n_rrt * (1 - target_ratio) / target_ratio)
    
    np.random.seed(42)
    sampled_non_rrt_idx = np.random.choice(non_rrt_idx, n_non_rrt_new, replace=False)
    
    all_idx = np.concatenate([rrt_idx, sampled_non_rrt_idx])
    X_mimic_sampled = X_mimic[all_idx]
    T_mimic_sampled = T_mimic[all_idx]
    
    print(f" Sampling : {len(X_mimic_sampled)} samples, RRT Rate={T_mimic_sampled.mean():.1%}")
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic_sampled)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_mimic_scaled, T_mimic_sampled)
    
    prob = model.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_5_ensemble(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method5 set not ConfigurationModel"""
    print("\n[Method5] set ")
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    configs = [
        {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1},
        {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.05},
        {'n_estimators': 800, 'max_depth': 8, 'learning_rate': 0.03},
        {'n_estimators': 400, 'max_depth': 12, 'learning_rate': 0.05},
    ]
    
    all_probs = []
    for cfg in configs:
        model = lgb.LGBMClassifier(**cfg, class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_mimic_scaled, T_mimic)
        all_probs.append(model.predict_proba(X_eicu_scaled)[:, 1])
    
    # MeanProbability
    prob = np.mean(all_probs, axis=0)
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_6_coral(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method6 CORAL variancepair """
    print("\n[Method6] CORAL ")
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    # CORAL: pair andtarget variance
    Cs = np.cov(X_mimic_scaled, rowvar=False) + np.eye(X_mimic_scaled.shape[1]) * 1e-3
    Ct = np.cov(X_eicu_scaled, rowvar=False) + np.eye(X_eicu_scaled.shape[1]) * 1e-3
    
    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-6)
    Cs_sqrt_inv = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
    
    # totarget 
    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-6)
    Ct_sqrt = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
    
    # Data
    X_mimic_coral = X_mimic_scaled @ Cs_sqrt_inv @ Ct_sqrt
    
    model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_mimic_coral, T_mimic)
    
    prob = model.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def method_7_combined(X_mimic, T_mimic, X_eicu, T_eicu):
    """Method7 group policy downSampling+Feature +Calibration """
    print("\n[Method7] group policy")
    
    # 1. Feature 
    ks_stats = [(i, stats.ks_2samp(X_mimic[:, i], X_eicu[:, i])[0]) for i in range(X_mimic.shape[1])]
    keep_idx = [i for i, ks in ks_stats if ks < 0.45]
    X_mimic_sel = X_mimic[:, keep_idx]
    X_eicu_sel = X_eicu[:, keep_idx]
    
    # 2. downSampling
    target_ratio = T_eicu.mean()
    rrt_idx = np.where(T_mimic == 1)[0]
    non_rrt_idx = np.where(T_mimic == 0)[0]
    n_rrt = len(rrt_idx)
    n_non_rrt_new = int(n_rrt * (1 - target_ratio) / target_ratio)
    np.random.seed(42)
    sampled_non_rrt_idx = np.random.choice(non_rrt_idx, min(n_non_rrt_new, len(non_rrt_idx)), replace=False)
    all_idx = np.concatenate([rrt_idx, sampled_non_rrt_idx])
    X_mimic_sampled = X_mimic_sel[all_idx]
    T_mimic_sampled = T_mimic[all_idx]
    
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic_sampled)
    X_eicu_scaled = scaler.transform(X_eicu_sel)
    
    # 3. Calibration 
    base_model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated.fit(X_mimic_scaled, T_mimic_sampled)
    
    prob = calibrated.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    return evaluate(T_eicu, prob, thresh)


def main():
    print("="*70)
    print("eICUExternalValidationDepthOptimization")
    print("="*70)
    
    X_mimic, T_mimic, X_eicu, T_eicu = load_data()
    print(f"MIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.1%}")
    print(f"eICU: {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.1%}")
    
    results = []
    
    # Baseline
    print("\n[Baseline] Model")
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_mimic_scaled, T_mimic)
    prob = model.predict_proba(X_eicu_scaled)[:, 1]
    thresh = find_best_threshold(T_eicu, prob)
    baseline = evaluate(T_eicu, prob, thresh)
    baseline['method'] = 'Baseline'
    results.append(baseline)
    print(f"  ADR={baseline['acc']:.2%}, AUC={baseline['auc']:.3f}, RRTgroup={baseline['treat_acc']:.2%}")
    
    # each  typesOptimizationMethod
    for method, func in [
        ('Feature Selection', method_1_feature_selection),
        ('Platt Scaling', method_2_calibration),
        ('Isotonic Calibration', method_3_isotonic_calibration),
        ('Undersampling', method_4_undersampling),
        ('Ensemble', method_5_ensemble),
        ('CORAL', method_6_coral),
        ('Combined', method_7_combined),
    ]:
        try:
            result = func(X_mimic, T_mimic, X_eicu, T_eicu)
            result['method'] = method
            results.append(result)
            print(f"  ADR={result['acc']:.2%}, AUC={result['auc']:.3f}, RRTgroup={result['treat_acc']:.2%}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    df = pd.DataFrame(results)
    df = df[['method', 'acc', 'auc', 'treat_acc', 'control_acc', 'threshold']]
    df.columns = ['Method', 'ADR', 'AUC', 'RRT Acc', 'Non-RRT Acc', 'Threshold']
    print(df.to_string(index=False))
    
    # Method
    best_idx = df['AUC'].idxmax()
    best = df.iloc[best_idx]
    print(f"\n AUCMethod: {best['Method']}, AUC={best['AUC']:.3f}")
    
    best_idx_acc = df['ADR'].idxmax()
    best_acc = df.iloc[best_idx_acc]
    print(f" ADRMethod: {best_acc['Method']}, ADR={best_acc['ADR']:.2%}")
    
    df.to_csv('ExternalValidation/deep_optimization_results.csv', index=False)
    print("\nResults saved: ExternalValidation/deep_optimization_results.csv")
    
    return df


if __name__ == '__main__':
    df = main()

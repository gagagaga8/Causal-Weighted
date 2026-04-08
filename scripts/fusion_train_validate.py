"""
MIMIC + eICU fusion training and validation pipeline (Stacking version)

Study time node: k2 (0-24h post KDIGO3) - target a1
1. Causal inference: propensity scores ps_k1, ps_k2 (fit on train only, leakage-safe)
2. Feature Feature bun_creat_ratio ph_change acidosis hyperk 
3. Model Stacking (LGB + XGB + RF → LR) ThresholdinValidationset notuseTestset 
4. Overfitting scaler/PS only fit Trainingset Thresholduse val Final Evaluationonlyuse test
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Path configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIMIC_PATH = os.path.join(PROJECT_ROOT, "03_DataSplit", "data", "mimic_preprocessed.csv")
EICU_PATH = os.path.join(PROJECT_ROOT, "2_eICUPreprocessing ", "data", "eicu_preprocessed.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fusion_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# k2 only k1 Baseline + k2 when status
BASE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'lactate_k2', 'bicarbonate_k2', # /sepsis-AKI Metric
]
# Feature use No fit No 
DERIVED_FEATURES = [
    'bun_creat_ratio_k1', 'bun_creat_ratio_k2', 'ph_change', 'uo_change',
    'acidosis_k2', 'hyperk_k2',
    'lactate_elevated_k2', # lactate>2 mmol/L Metric
    'egfr_k1', 'egfr_k2', 'egfr_decline_k2', # creat/eGFR to
    'oliguria_k2', 'anuria_k2', 'complication_score_k2', # KDIGO and 
]
# CausalInference
PS_FEATURES_K1 = ['bun_k1', 'ph_k1', 'pot_k1']
PS_FEATURES_K2 = ['bun_k2', 'ph_k2', 'pot_k2', 'uo_k2', 'lactate_k2', 'bicarbonate_k2']

# target Variable a1 = 0-24h start RRT
TARGET = 'a1'


def _calc_egfr(creat, age, gender_val):
    """CKD-EPI gender_val: 1=M, 0=F"""
    if pd.isna(creat) or pd.isna(age) or creat <= 0:
        return np.nan
    creat = max(creat, 0.1)
    age = min(max(age, 1), 120)
    if gender_val == 1:
        k, a = (0.9, -0.411) if creat <= 0.9 else (0.9, -1.209)
    else:
        k, a = (0.7, -0.329) if creat <= 0.7 else (0.7, -1.209)
    return 141 * (creat / k) ** a * (0.993 ** age) * (1.012 if gender_val == 0 else 1)


def add_derived_features(df):
    """ Feature use No fit No Data """
    d = df.copy()
    g = d.get('gender', pd.Series(0, index=d.index))
    if hasattr(g, 'map') and g.dtype == object:
        g = g.map({'M': 1, 'Male': 1, 'F': 0, 'Female': 0}).fillna(0)
    g = pd.to_numeric(g, errors='coerce').fillna(0).astype(int)

    if 'bun_k1' in d.columns and 'creat_k1' in d.columns:
        d['bun_creat_ratio_k1'] = (d['bun_k1'].fillna(0) + 1) / (d['creat_k1'].fillna(0) + 1)
    if 'bun_k2' in d.columns and 'creat_k2' in d.columns:
        d['bun_creat_ratio_k2'] = (d['bun_k2'].fillna(0) + 1) / (d['creat_k2'].fillna(0) + 1)
    if 'ph_k1' in d.columns and 'ph_k2' in d.columns:
        d['ph_change'] = np.abs(d['ph_k2'].fillna(0) - d['ph_k1'].fillna(0))
    if 'uo_k1' in d.columns and 'uo_k2' in d.columns:
        d['uo_change'] = d['uo_k2'].fillna(0) - d['uo_k1'].fillna(0)
    if 'ph_k2' in d.columns:
        d['acidosis_k2'] = (d['ph_k2'].fillna(7.4) < 7.2).astype(float)
    if 'pot_k2' in d.columns:
        d['hyperk_k2'] = (d['pot_k2'].fillna(4) > 5.5).astype(float)
    if 'lactate_k2' in d.columns:
        d['lactate_elevated_k2'] = (d['lactate_k2'].fillna(1) > 2.0).astype(float)

    # eGFR Baseline creat/eGFR as to 
    age_arr = d['admission_age'].fillna(65).values if 'admission_age' in d.columns else np.full(len(d), 65)
    for col, out in [('creat_k1', 'egfr_k1'), ('creat_k2', 'egfr_k2')]:
        if col in d.columns:
            d[out] = [_calc_egfr(float(x) if not pd.isna(x) else 2, float(age_arr[i]), int(g.iloc[i]) if i < len(g) else 0) for i, x in enumerate(d[col])]
    if 'egfr_k1' in d.columns and 'egfr_k2' in d.columns:
        d['egfr_decline_k2'] = d['egfr_k1'].fillna(100) - d['egfr_k2'].fillna(100)

    # oliguria (<0.5 mL/kg/h), anuria (<0.05)
    if 'uo_k2' in d.columns:
        u = d['uo_k2'].fillna(10)
        d['oliguria_k2'] = (u < 0.5).astype(float)
        d['anuria_k2'] = (u < 0.05).astype(float)
    a2 = d.get('acidosis_k2', pd.Series(0, index=d.index))
    h2 = d.get('hyperk_k2', pd.Series(0, index=d.index))
    o2 = d.get('oliguria_k2', pd.Series(0, index=d.index))
    d['complication_score_k2'] = a2.fillna(0) + h2.fillna(0) + o2.fillna(0)
    return d


def build_feature_list(train_df):
    """according toTrainingset Column FeatureColumnTable"""
    # Remove 'ps_k1', 'ps_k2' from the features used by tree models to prevent feature masking/shortcuts
    feats = BASE_FEATURES.copy()
    for x in DERIVED_FEATURES:
        if x in train_df.columns:
            feats.append(x)
    return feats


def align_columns(df, source='mimic'):
    """pair Column with """
    df = df.copy()
    # PatientID
    id_col = 'icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid'
    df['patient_id'] = df[id_col].astype(str) + f'_{source}'

    # Sex
    if 'gender' in df.columns:
        if df['gender'].dtype == object or df['gender'].dtype.name == 'object':
            df['gender'] = df['gender'].map({'M': 1, 'Male': 1, 'F': 0, 'Female': 0}).fillna(0)
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(0)

    # target a1 k2 decision point 0-24h is start RRT 
    if 'a1' in df.columns:
        df['a1'] = pd.to_numeric(df['a1'], errors='coerce').fillna(0).astype(int)
    elif 'received_rrt' in df.columns:
        # eICU No a1 use received_rrt 
        df['a1'] = df['received_rrt'].astype(int)
    else:
        df['a1'] = 0

    # fillMissingFeature use 0 or to fill k1->k2->k3 
    base_ps = BASE_FEATURES + ['ps_k1', 'ps_k2']
    for col in base_ps:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # to fill k2 Missing bun/pot/ph use k1 
    for base in ['bun', 'pot', 'ph']:
        c, c1 = f'{base}_k2', f'{base}_k1'
        if c in df.columns and c1 in df.columns:
            df[c] = df[c].fillna(df[c1])
    # lactate/bicarbonate use k1 fill k2 Missing else use Normalvalue
    if 'lactate_k2' in df.columns:
        if 'lactate_k1' in df.columns:
            df['lactate_k2'] = df['lactate_k2'].fillna(df['lactate_k1'])
        df['lactate_k2'] = df['lactate_k2'].fillna(2.0)
    if 'bicarbonate_k2' in df.columns:
        if 'bicarbonate_k1' in df.columns:
            df['bicarbonate_k2'] = df['bicarbonate_k2'].fillna(df['bicarbonate_k1'])
        df['bicarbonate_k2'] = df['bicarbonate_k2'].fillna(22.0)

    for c in BASE_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    # range 
    for c in ['uo_k1', 'uo_k2', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr']:
        if c in df.columns:
            df[c] = df[c].clip(0, 10)
    if 'sofa_24hours' in df.columns:
        df['sofa_24hours'] = df['sofa_24hours'].clip(0, 24)

    return df


def load_and_prepare():
    """Loadingand pair MIMIC eICU Data"""
    print("=" * 60)
    print("1. Load data")
    print("=" * 60)

    if not os.path.exists(MIMIC_PATH):
        raise FileNotFoundError(f"MIMIC Datanot in: {MIMIC_PATH}\n Run 1_mimicPreprocessing and Outputto 03_DataSplit/data/")
    if not os.path.exists(EICU_PATH):
        raise FileNotFoundError(f"eICU Datanot in: {EICU_PATH}\n Run 2_eICUPreprocessing ")

    mimic = pd.read_csv(MIMIC_PATH)
    eicu = pd.read_csv(EICU_PATH)

    # eICU Patientcancan Row 
    eicu = eicu.drop_duplicates(subset=['patientunitstayid'], keep='first').reset_index(drop=True)

    print(f" MIMIC: {len(mimic)} ")
    print(f" eICU: {len(eicu)} ")

    mimic = align_columns(mimic, 'mimic')
    eicu = align_columns(eicu, 'eicu')
    mimic = add_derived_features(mimic)
    eicu = add_derived_features(eicu)

    return mimic, eicu


def fuse_and_split(mimic, eicu, seed=123):
    """eICU Half fusion, half external validation; fusion split 7:2:1"""
    print("\n" + "=" * 60)
    print("2. withSplit")
    print("=" * 60)

    np.random.seed(seed)

    # eICU 50/50
    n_eicu = len(eicu)
    idx = np.random.permutation(n_eicu)
    n_half = n_eicu // 2
    eicu_fuse = eicu.iloc[idx[:n_half]].copy()
    eicu_external = eicu.iloc[idx[n_half:]].copy()

    print(f" eICU : {len(eicu_fuse)} ")
    print(f" eICU ExternalValidation: {len(eicu_external)} ")

    common_cols = [c for c in mimic.columns if c in eicu_fuse.columns]
    fused = pd.concat([
        mimic[common_cols],
        eicu_fuse[common_cols]
    ], ignore_index=True)

    fused['dataset'] = ['MIMIC'] * len(mimic) + ['eICU'] * len(eicu_fuse)
    print(f" : {len(fused)} (MIMIC {len(mimic)} + eICU {len(eicu_fuse)})")

    # 7:2:1 Split
    n = len(fused)
    shuf = np.random.permutation(n)
    n_train = int(0.7 * n)
    n_test = int(0.2 * n)
    n_val = n - n_train - n_test

    train_df = fused.iloc[shuf[:n_train]].reset_index(drop=True)
    test_df = fused.iloc[shuf[n_train:n_train + n_test]].reset_index(drop=True)
    val_df = fused.iloc[shuf[n_train + n_test:]].reset_index(drop=True)

    print(f" Trainingset: {len(train_df)} ")
    print(f" Testset: {len(test_df)} ")
    print(f" Validationset: {len(val_df)} ")

    return train_df, test_df, val_df, eicu_external


def compute_propensity_scores(train_df, test_df, val_df, eicu_external):
    """
    CausalInference Computationpropensity score ps_k1, ps_k2 dWOLS 
    inTrainingsetonfit PS Model again useto set 
    Return (ps_models, ps_cols_k1, ps_cols_k2) Inference use
    """
    ps_models = {}
    ps_cols_k1, ps_cols_k2 = [], []
    for df in [train_df, test_df, val_df, eicu_external]:
        df['ps_k1'] = 0.0
        df['ps_k2'] = 0.0

    # ps_k1: P(RRT|bun_k1,ph_k1,pot_k1)
    ps_cols_k1 = [c for c in PS_FEATURES_K1 if c in train_df.columns]
    if ps_cols_k1:
        X_ps = train_df[ps_cols_k1].fillna(0)
        y_k1 = train_df[TARGET].values
        if len(np.unique(y_k1)) > 1:
            m1 = LogisticRegression(max_iter=1000, random_state=42)
            m1.fit(X_ps, y_k1)
            ps_models['k1'] = (m1, ps_cols_k1)
            for name, df in [('train', train_df), ('test', test_df), ('val', val_df), ('external', eicu_external)]:
                df['ps_k1'] = m1.predict_proba(df[ps_cols_k1].fillna(0))[:, 1]

    # ps_k2
    ps_cols_k2 = [c for c in PS_FEATURES_K2 if c in train_df.columns]
    if ps_cols_k2:
        X_ps = train_df[ps_cols_k2].fillna(0)
        y_k2 = train_df[TARGET].values
        if len(np.unique(y_k2)) > 1:
            m2 = LogisticRegression(max_iter=1000, random_state=42)
            m2.fit(X_ps, y_k2)
            ps_models['k2'] = (m2, ps_cols_k2)
            for name, df in [('train', train_df), ('test', test_df), ('val', val_df), ('external', eicu_external)]:
                df['ps_k2'] = m2.predict_proba(df[ps_cols_k2].fillna(0))[:, 1]

    return ps_models


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


def train_model(train_df, val_df, features):
    """
    Training Stacking Model LGB + XGB + RF → LR (IPW Weighted)
    - scaler only fit Trainingset 
    - use cv=2 Generate Feature not 
    - will ps_k2 use meta-learner IPW Feature 
    """
    print("\n" + "=" * 60)
    print("3. Training Stacking Model LGB + XGB + RF → LR with IPW ")
    print("=" * 60)

    X_train = train_df[features].fillna(0)
    y_train = train_df[TARGET].values
    
    # Extract propensity scores for IPW
    if 'ps_k2' in train_df.columns:
        ps_train = train_df['ps_k2'].values
    elif 'ps_k1' in train_df.columns:
        ps_train = train_df['ps_k1'].values
    else:
        ps_train = np.full(len(y_train), 0.5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training set only

    # Overfitting Increaseregularize
    base = [
        ('lgb', lgb.LGBMClassifier(n_estimators=80, max_depth=4, min_child_samples=20,
                                   reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
                                   random_state=42, verbose=-1, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
    ]
    if HAS_XGB:
        sw = len(y_train) / max(1, y_train.sum()) - 1
        base.insert(1, ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=4,
                                                  reg_alpha=0.1, reg_lambda=1.0,
                                                  scale_pos_weight=sw, random_state=42, verbosity=0, n_jobs=-1)))

    model = IPWStackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42), # C<1 L2 regularize
        cv=2  # Cross-validation meta-feature generation, no leakage
    )
    model.fit(X_train_scaled, y_train, ps=ps_train)
    return model, scaler


def tune_threshold_on_val(model, scaler, val_df, features, thresholds=None):
    """
    inValidationseton Threshold useTestset 
    """
    if thresholds is None:
        thresholds = np.arange(0.35, 0.71, 0.02)
    X_val = val_df[features].fillna(0)
    y_val = val_df[TARGET].values
    X_val_s = scaler.transform(X_val)
    proba = model.predict_proba(X_val_s)[:, 1]
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def evaluate(model, scaler, df, features, name="Set", threshold=0.5):
    """EvaluationAccuracywith AUC can Threshold """
    X = df[features].fillna(0)
    y = df[TARGET].values
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)  # Use tuned threshold

    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, proba)
    except:
        auc = 0.5
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:    {auc:.4f}")
    print("\nconfusion matrix:")
    print(confusion_matrix(y, pred))
    print("\nClassificationreport:")
    print(classification_report(y, pred, target_names=['No RRT', 'RRT']))
    return {'accuracy': acc, 'auc': auc, 'n': len(df)}


def main():
    mimic, eicu = load_and_prepare()
    train_df, test_df, val_df, eicu_external = fuse_and_split(mimic, eicu)

    # CausalInference propensity score onlyuseTrainingsetfit 
    print("\n" + "=" * 60)
    print("2.5 CausalInference Computationpropensity score ps_k1, ps_k2 fit only train ")
    print("=" * 60)
    ps_models = compute_propensity_scores(train_df, test_df, val_df, eicu_external)
    print(" ps_k1, ps_k2 already ")

    # FeatureColumnTable Feature 
    FEATURES = build_feature_list(train_df)
    print(f"  Features: {len(FEATURES)} ({', '.join([f for f in FEATURES if f not in BASE_FEATURES][:6])}...)")

    # Training onlyuse train scaler only fit train 
    model, scaler = train_model(train_df, val_df, FEATURES)

    # Threshold onlyinValidationset useTestset 
    print("\n" + "=" * 60)
    print("3.5 Threshold onlyuseValidationset Overfitting ")
    print("=" * 60)
    best_threshold, val_acc = tune_threshold_on_val(model, scaler, val_df, FEATURES)
    print(f" Threshold: {best_threshold:.2f} (Validationset acc={val_acc:.4f})")

    # SaveModel Threshold PS Model Inference 
    import joblib
    ckpt = {
        'model': model, 'scaler': scaler, 'features': FEATURES, 'threshold': best_threshold,
        'ps_models': ps_models # {'k1':(m,cols), 'k2':(m,cols)} use Inference Computation ps
    }
    joblib.dump(ckpt, os.path.join(OUTPUT_DIR, 'fusion_stacking.pkl'))
    print(f"  alreadySave: fusion_stacking.pkl")

    # TrainingsetEvaluation onlyuse Overfitting not with 
    print("\n" + "=" * 60)
    print("3.9 Overfitting TrainingsetTable ")
    print("=" * 60)
    train_metrics = evaluate(model, scaler, train_df, FEATURES, "Trainingset", threshold=best_threshold)
    # Internal Evaluation Testset times use No 
    print("\n" + "=" * 60)
    print("4. Internal Validation Testset not withTraining/ ")
    print("=" * 60)
    internal = evaluate(model, scaler, test_df, FEATURES, "Testset", threshold=best_threshold)

    # ExternalEvaluation
    print("\n" + "=" * 60)
    print("5. ExternalValidation eICU 50% ")
    print("=" * 60)
    external = evaluate(model, scaler, eicu_external, FEATURES, "eICU ExternalValidationset", threshold=best_threshold)

    # Summary TrainingsetMetric Overfitting 
    results = {
        'train': train_metrics,
        'internal_test': internal,
        'external_eicu': external,
        'threshold': best_threshold,
        'val_accuracy_at_threshold': val_acc,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'val_size': len(val_df),
        'eicu_external_size': len(eicu_external),
        'fusion_total': len(train_df) + len(test_df) + len(val_df),
        'features': FEATURES,
        'overfit_gap': train_metrics['accuracy'] - internal['accuracy'],
    }
    with open(os.path.join(OUTPUT_DIR, 'eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ResultsSummary")
    print("=" * 60)
    print(f"Internal Testset - Accuracy: {internal['accuracy']:.4f}, AUC: {internal['auc']:.4f}")
    print(f"External eICU  - Accuracy: {external['accuracy']:.4f}, AUC: {external['auc']:.4f}")
    print(f"\nResults saved : {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

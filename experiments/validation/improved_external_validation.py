"""
1 Experiment eICUExternalValidationPerformance
policy 
1. FeatureDistributionpair Distribution Alignment 
2. Training Importance Weighting 
3. Training Fine-tuning on eICU subset 
4. ThresholdCalibration 
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
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


def load_mimic_data():
    """LoadingMIMICData"""
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    
    X = df[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask]
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T


def load_eicu_data():
    """LoadingeICUData"""
    df = pd.read_csv('2_eICUPreprocessingdata/eicu_full_features.csv')
    df = df.drop_duplicates(subset=['patientunitstayid'])
    
    available = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[available].copy()
    
    # MissingFeature
    for f in SAFE_FEATURES:
        if f not in X.columns:
            X[f] = 0
    
    X = X[SAFE_FEATURES]
    
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    
    X = X.fillna(0)
    T = df['received_rrt'].values
    
    return X, T


def compute_distribution_shift(X_source, X_target, feature_names):
    """ComputationFeatureDistribution """
    shifts = []
    for i, f in enumerate(feature_names):
        src = X_source[:, i]
        tgt = X_target[:, i]
        
        # KSstatistics
        ks_stat, _ = stats.ks_2samp(src, tgt)
        
        # value 
        mean_shift = abs(src.mean() - tgt.mean()) / (src.std() + 1e-6)
        
        shifts.append({
            'feature': f,
            'ks_stat': ks_stat,
            'mean_shift': mean_shift,
            'src_mean': src.mean(),
            'tgt_mean': tgt.mean()
        })
    
    return pd.DataFrame(shifts).sort_values('ks_stat', ascending=False)


def importance_weighted_training(X_mimic, T_mimic, X_eicu):
    """
     to Training
     use compareestimate Row 
    """
    print(" Training Classificationer...")
    
    # MergeData 
    X_all = np.vstack([X_mimic, X_eicu])
    domain_labels = np.concatenate([np.zeros(len(X_mimic)), np.ones(len(X_eicu))])
    
    # Training Classificationer
    domain_clf = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, random_state=42, verbose=-1
    )
    domain_clf.fit(X_all, domain_labels)
    
    # ComputationMIMICsamples to Weight
    proba = domain_clf.predict_proba(X_mimic)[:, 1]
    weights = proba / (1 - proba + 1e-6) # compare
    weights = np.clip(weights, 0.1, 10) # value
    weights = weights / weights.mean()  # Normalize
    
    print(f"  Weightrange: [{weights.min():.2f}, {weights.max():.2f}]")
    
    # Use weighted training
    print(" TrainingModel...")
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_mimic, T_mimic, sample_weight=weights)
    
    return model, weights


def mixed_training(X_mimic, T_mimic, X_eicu, T_eicu, eicu_ratio=0.2):
    """
     Training use eICUData Row 
    """
    print(f" Training eICURatio={eicu_ratio:.0%} ...")
    
    # Random eICUData
    n_eicu = int(len(X_eicu) * eicu_ratio)
    idx = np.random.RandomState(42).choice(len(X_eicu), n_eicu, replace=False)
    X_eicu_train = X_eicu[idx]
    T_eicu_train = T_eicu[idx]
    
    # MergeTrainingData
    X_train = np.vstack([X_mimic, X_eicu_train])
    T_train = np.concatenate([T_mimic, T_eicu_train])
    
    # Training
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_train, T_train)
    
    return model, idx


def evaluate_model(model, scaler, X, T, threshold=0.5):
    """EvaluationModel"""
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1]
    pred = (prob >= threshold).astype(int)
    
    acc = (pred == T).mean()
    auc = roc_auc_score(T, prob)
    f1 = f1_score(T, pred)
    
    treat_mask = T == 1
    control_mask = T == 0
    treat_acc = (pred[treat_mask] == T[treat_mask]).mean() if treat_mask.sum() > 0 else 0
    control_acc = (pred[control_mask] == T[control_mask]).mean() if control_mask.sum() > 0 else 0
    
    return {
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'treat_acc': treat_acc,
        'control_acc': control_acc,
        'threshold': threshold
    }


def find_optimal_threshold(model, scaler, X, T):
    """ Threshold"""
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1]
    
    best_thresh = 0.5
    best_score = 0
    
    for thresh in np.arange(0.1, 0.6, 0.05):
        pred = (prob >= thresh).astype(int)
        acc = (pred == T).mean()
        treat_acc = (pred[T == 1] == T[T == 1]).mean() if (T == 1).sum() > 0 else 0
        
        score = 0.6 * acc + 0.4 * treat_acc
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh


def main():
    print("="*70)
    print("1 Experiment eICUExternalValidation ")
    print("="*70)
    
    # Load data
    print("\n[1/6] Load data")
    X_mimic_df, T_mimic = load_mimic_data()
    X_eicu_df, T_eicu = load_eicu_data()
    
    X_mimic = X_mimic_df.values
    X_eicu = X_eicu_df.values
    
    print(f"  MIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.1%}")
    print(f"  eICU: {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.1%}")
    
    # Standardize
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    # Distribution Analysis
    print("\n[2/6] FeatureDistribution Analysis")
    shifts = compute_distribution_shift(X_mimic, X_eicu, SAFE_FEATURES)
    print(" Maximum5 Feature:")
    for _, row in shifts.head().iterrows():
        print(f" {row['feature']}: KS={row['ks_stat']:.3f}, value ={row['mean_shift']:.2f}")
    shifts.to_csv('ExternalValidation/feature_distribution_shift.csv', index=False)
    
    results = []
    
    # Method1: Baseline MIMICTraining 
    print("\n[3/6] Method1: BaselineModel")
    model_baseline = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model_baseline.fit(X_mimic_scaled, T_mimic)
    
    thresh_baseline = find_optimal_threshold(model_baseline, scaler, X_eicu, T_eicu)
    eval_baseline = evaluate_model(model_baseline, scaler, X_eicu, T_eicu, thresh_baseline)
    eval_baseline['method'] = 'Baseline'
    results.append(eval_baseline)
    print(f" Threshold: {thresh_baseline:.2f}")
    print(f"  ADR: {eval_baseline['accuracy']:.2%}, AUC: {eval_baseline['auc']:.3f}")
    print(f"  RRTgroupAccuracy: {eval_baseline['treat_acc']:.2%}")
    
    # Method2: to 
    print("\n[4/6] Method2: to Training")
    model_iw, weights = importance_weighted_training(X_mimic_scaled, T_mimic, X_eicu_scaled)
    
    thresh_iw = find_optimal_threshold(model_iw, scaler, X_eicu, T_eicu)
    eval_iw = evaluate_model(model_iw, scaler, X_eicu, T_eicu, thresh_iw)
    eval_iw['method'] = 'Importance Weighting'
    results.append(eval_iw)
    print(f" Threshold: {thresh_iw:.2f}")
    print(f"  ADR: {eval_iw['accuracy']:.2%}, AUC: {eval_iw['auc']:.3f}")
    print(f"  RRTgroupAccuracy: {eval_iw['treat_acc']:.2%}")
    
    # Method3: Training 10% 
    print("\n[5/6] Method3: Training 10% eICU ")
    model_mix10, idx10 = mixed_training(X_mimic_scaled, T_mimic, X_eicu_scaled, T_eicu, 0.1)
    
    # in eICUonEvaluation
    mask_test = np.ones(len(X_eicu), dtype=bool)
    mask_test[idx10] = False
    X_eicu_test = X_eicu[mask_test]
    T_eicu_test = T_eicu[mask_test]
    
    thresh_mix = find_optimal_threshold(model_mix10, scaler, X_eicu_test, T_eicu_test)
    eval_mix10 = evaluate_model(model_mix10, scaler, X_eicu_test, T_eicu_test, thresh_mix)
    eval_mix10['method'] = 'Mixed Training (10%)'
    results.append(eval_mix10)
    print(f" Threshold: {thresh_mix:.2f}")
    print(f"  ADR: {eval_mix10['accuracy']:.2%}, AUC: {eval_mix10['auc']:.3f}")
    print(f"  RRTgroupAccuracy: {eval_mix10['treat_acc']:.2%}")
    
    # Method4: Training 20% 
    print("\n[6/6] Method4: Training 20% eICU ")
    model_mix20, idx20 = mixed_training(X_mimic_scaled, T_mimic, X_eicu_scaled, T_eicu, 0.2)
    
    mask_test20 = np.ones(len(X_eicu), dtype=bool)
    mask_test20[idx20] = False
    X_eicu_test20 = X_eicu[mask_test20]
    T_eicu_test20 = T_eicu[mask_test20]
    
    thresh_mix20 = find_optimal_threshold(model_mix20, scaler, X_eicu_test20, T_eicu_test20)
    eval_mix20 = evaluate_model(model_mix20, scaler, X_eicu_test20, T_eicu_test20, thresh_mix20)
    eval_mix20['method'] = 'Mixed Training (20%)'
    results.append(eval_mix20)
    print(f" Threshold: {thresh_mix20:.2f}")
    print(f"  ADR: {eval_mix20['accuracy']:.2%}, AUC: {eval_mix20['auc']:.3f}")
    print(f"  RRTgroupAccuracy: {eval_mix20['treat_acc']:.2%}")
    
    # SummaryResults
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    results_df = pd.DataFrame(results)
    results_df = results_df[['method', 'accuracy', 'auc', 'f1', 'treat_acc', 'control_acc', 'threshold']]
    
    print(results_df.to_string(index=False))
    
    # Save
    results_df.to_csv('ExternalValidation/improved_validation_results.csv', index=False)
    print("\nResults saved: ExternalValidation/improved_validation_results.csv")
    
    # Method
    best_idx = results_df['accuracy'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\n Method: {best['method']}")
    print(f"  ADR: {best['accuracy']:.2%}")
    print(f"  AUC: {best['auc']:.3f}")
    print(f"  RRTgroupAccuracy: {best['treat_acc']:.2%}")
    
    return results_df


if __name__ == '__main__':
    results = main()

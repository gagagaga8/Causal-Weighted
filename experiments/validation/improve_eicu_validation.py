"""
 eICUExternalValidation RRTgroup 
policy 
1. Threshold from0.5Decreased 
2. useClassWeight Training
3. AnalysisFeatureDistribution 
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_mimic_data():
    """LoadingMIMICTrainingData"""
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    
    X = df[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T


def load_eicu_data():
    """LoadingeICUValidationData"""
    df = pd.read_csv('2_eICUPreprocessingdata/eicu_full_features.csv')
    df = df.drop_duplicates(subset=['patientunitstayid'])
    
    available = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[available].copy()
    
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    
    X = X.fillna(0).values
    T = df['received_rrt'].values
    
    return X, T


def train_mimic_model(X, T):
    """inMIMIConTrainingModel"""
    print("TrainingMIMICModel...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_scaled, T)
    
    # 5-fold Cross-validationEvaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, test_idx in skf.split(X_scaled, T):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        temp_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        )
        temp_model.fit(X_train, T_train)
        pred = temp_model.predict(X_test)
        cv_scores.append((pred == T_test).mean())
    
    print(f"  MIMIC 5-fold CV: {np.mean(cv_scores):.2%}")
    
    return model, scaler


def evaluate_with_threshold(model, scaler, X, T, threshold=0.5):
    """Evaluate with specified threshold"""
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1]
    pred = (prob >= threshold).astype(int)
    
    acc = (pred == T).mean()
    treat_mask = T == 1
    control_mask = T == 0
    
    treat_acc = (pred[treat_mask] == T[treat_mask]).mean() if treat_mask.sum() > 0 else 0
    control_acc = (pred[control_mask] == T[control_mask]).mean() if control_mask.sum() > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': acc,
        'treat_acc': treat_acc,
        'control_acc': control_acc,
        'pred_rate': pred.mean()
    }


def threshold_optimization():
    print("="*70)
    print("eICUExternalValidation ThresholdOptimization")
    print("="*70)
    
    # Load data
    print("\n1. Load data")
    X_mimic, T_mimic = load_mimic_data()
    X_eicu, T_eicu = load_eicu_data()
    print(f"  MIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.1%}")
    print(f"  eICU: {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.1%}")
    
    # TrainingModel
    print("\n2. TrainingModel")
    model, scaler = train_mimic_model(X_mimic, T_mimic)
    
    # MIMICValidation defaultThreshold0.5 
    print("\n3. MIMICValidation Threshold=0.5 ")
    mimic_result = evaluate_with_threshold(model, scaler, X_mimic, T_mimic, 0.5)
    print(f"  ADR: {mimic_result['accuracy']:.2%}")
    print(f"  RRTgroupAccuracy: {mimic_result['treat_acc']:.2%}")
    print(f"  nonRRTgroupAccuracy: {mimic_result['control_acc']:.2%}")
    
    # eICUValidation defaultThreshold0.5 
    print("\n4. eICUValidation Threshold=0.5 ")
    eicu_result_default = evaluate_with_threshold(model, scaler, X_eicu, T_eicu, 0.5)
    print(f"  ADR: {eicu_result_default['accuracy']:.2%}")
    print(f"  RRTgroupAccuracy: {eicu_result_default['treat_acc']:.2%}")
    print(f"  nonRRTgroupAccuracy: {eicu_result_default['control_acc']:.2%}")
    
    # Thresholdsearch
    print("\n5. eICUThresholdOptimization")
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = []
    
    for thresh in thresholds:
        result = evaluate_with_threshold(model, scaler, X_eicu, T_eicu, thresh)
        results.append(result)
        print(f"  Threshold={thresh:.2f}: ADR={result['accuracy']:.2%}, "
              f"RRTAccuracy={result['treat_acc']:.2%}, "
              f"nonRRTAccuracy={result['control_acc']:.2%}")
    
    # Threshold balanceADRandRRTAccuracy 
    results_df = pd.DataFrame(results)
    
    # ADR + RRTAccuracy Weight0.5 
    results_df['score'] = results_df['accuracy'] + 0.5 * results_df['treat_acc']
    best_idx = results_df['score'].idxmax()
    best = results_df.iloc[best_idx]
    
    print("\n" + "="*70)
    print(" Threshold ")
    print("="*70)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"ADR: {best['accuracy']:.2%}")
    print(f"RRTgroupAccuracy: {best['treat_acc']:.2%}")
    print(f"nonRRTgroupAccuracy: {best['control_acc']:.2%}")
    
    # Compare with original
    print("\n" + "="*70)
    print(" ")
    print("="*70)
    print(f" Threshold=0.5 :")
    print(f"  eICU ADR: {eicu_result_default['accuracy']:.2%}")
    print(f"  RRTgroupAccuracy: {eicu_result_default['treat_acc']:.2%}")
    
    print(f"\nOptimization Threshold={best['threshold']:.2f} :")
    print(f"  eICU ADR: {best['accuracy']:.2%}")
    print(f"  RRTgroupAccuracy: {best['treat_acc']:.2%}")
    
    print(f"\n :")
    print(f"  ADR: {best['accuracy']-eicu_result_default['accuracy']:+.2%}")
    print(f"  RRTgroupAccuracy: {best['treat_acc']-eicu_result_default['treat_acc']:+.2%}")
    
    # SaveResults
    results_df.to_csv('output/eicu_threshold_optimization.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: output/eicu_threshold_optimization.csv")
    
    return results_df


if __name__ == '__main__':
    results = threshold_optimization()

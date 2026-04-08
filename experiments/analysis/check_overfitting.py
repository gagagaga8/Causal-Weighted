"""
ModelOverfitting 
 LightGBMModelTrainingsetvsTestsetTable 
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# allFeature
SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

def load_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    available = [c for c in SAFE_FEATURES if c in df.columns]
    
    X = df[available].copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T

def check_overfitting():
    print("="*60)
    print("ModelOverfitting ")
    print("="*60)
    
    X, T = load_data()
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} Feature")
    print(f"Treatmentgroup: {T.sum()} ({T.mean():.2%})")
    
    # 1. Trainingset vs TestsetComparison
    print("\n" + "-"*60)
    print("1. Trainingset vs TestsetComparison (85/15Split)")
    print("-"*60)
    
    X_train, X_test, T_train, T_test = train_test_split(
        X, T, test_size=0.15, random_state=42, stratify=T
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1
    )
    model.fit(X_train_scaled, T_train)
    
    train_acc = (model.predict(X_train_scaled) == T_train).mean()
    test_acc = (model.predict(X_test_scaled) == T_test).mean()
    gap = train_acc - test_acc
    
    print(f"  Training accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    print(f"  Gap: {gap:.2%}")
    
    if gap > 0.10:
        print(f" [WARNING] inOverfittingRisk (Gap > 10%)")
    elif gap > 0.05:
        print(f" [ ] Slight overfitting (Gap 5-10%)")
    else:
        print(f" [Normal] No Overfitting (Gap < 5%)")
    
    # 2. 5-fold Cross-validation
    print("\n" + "-"*60)
    print("2. 5-fold Cross-validationstable ")
    print("-"*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, test_scores = [], []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, T)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        
        model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1
        )
        model.fit(X_tr, T[tr_idx])
        
        tr_acc = (model.predict(X_tr) == T[tr_idx]).mean()
        te_acc = (model.predict(X_te) == T[te_idx]).mean()
        train_scores.append(tr_acc)
        test_scores.append(te_acc)
        print(f"  Fold {fold+1}: Training={tr_acc:.2%}, Test={te_acc:.2%}, Gap={tr_acc-te_acc:.2%}")
    
    print(f"\n Trainingset value: {np.mean(train_scores):.2%} +/- {np.std(train_scores):.2%}")
    print(f" Testset value: {np.mean(test_scores):.2%} +/- {np.std(test_scores):.2%}")
    print(f"  MeanGap: {np.mean(train_scores) - np.mean(test_scores):.2%}")
    
    # 3. CurveAnalysis
    print("\n" + "-"*60)
    print("3. Curve (not TrainingData )")
    print("-"*60)
    
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        n_samples = int(len(X) * frac * 0.85)
        X_sub, _, T_sub, _ = train_test_split(X, T, train_size=frac*0.85, random_state=42, stratify=T)
        X_tr, X_te, T_tr, T_te = train_test_split(X_sub, T_sub, test_size=0.15, random_state=42, stratify=T_sub)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1
        )
        model.fit(X_tr_s, T_tr)
        
        tr_acc = (model.predict(X_tr_s) == T_tr).mean()
        te_acc = (model.predict(X_te_s) == T_te).mean()
        print(f"  {int(frac*100)}%Data ({len(X_tr)}samples): Training={tr_acc:.2%}, Test={te_acc:.2%}")
    
    # 4. Summary
    print("\n" + "="*60)
    print("Overfitting Summary")
    print("="*60)
    avg_gap = np.mean(train_scores) - np.mean(test_scores)
    
    if avg_gap > 0.10:
        status = "Severe overfitting"
        suggestion = " : Increaseregularization DecreaseModel IncreaseData"
    elif avg_gap > 0.05:
        status = "Slight overfitting"
        suggestion = " : canAccept butcan or Decreased Depth"
    else:
        status = "No Overfitting"
        suggestion = "Model can "
    
    print(f"  status: {status}")
    print(f"  Training accuracy: {np.mean(train_scores):.2%}")
    print(f"  Test accuracy: {np.mean(test_scores):.2%}")
    print(f" Gap: {avg_gap:.2%}")
    print(f"  {suggestion}")

if __name__ == '__main__':
    check_overfitting()

"""
set policyComparisonExperiment
Comparisonnot set policypairModelPerformance 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier, BaggingClassifier,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
import xgboost as xgb
import lightgbm as lgb
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
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    X = df[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    return X, T


def evaluate_ensemble(X, T, model, model_name):
    """Evaluationset Model"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, T_train)
        pred = model.predict(X_test_scaled)
        scores.append((pred == T_test).mean())
    
    return {
        'strategy': model_name,
        'adr': np.mean(scores),
        'adr_std': np.std(scores)
    }


def run_ensemble_comparison():
    print("="*70)
    print("set policyComparisonExperiment")
    print("="*70)
    
    X, T = load_data()
    print(f"\nDataset: {len(X)} samples\n")
    
    results = []
    
    # 1. LightGBM 
    print("[1/8] LightGBM ...")
    model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                class_weight='balanced', random_state=42, verbose=-1)
    result = evaluate_ensemble(X, T, model, 'Single LightGBM')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 2. Bagging LightGBM
    print("[2/8] Bagging LightGBM...")
    base_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.05,
                                   class_weight='balanced', random_state=42, verbose=-1)
    model = BaggingClassifier(estimator=base_lgb, n_estimators=10, random_state=42, n_jobs=-1)
    result = evaluate_ensemble(X, T, model, 'Bagging LightGBM')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 3. AdaBoost
    print("[3/8] AdaBoost...")
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    result = evaluate_ensemble(X, T, model, 'AdaBoost')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 4. set (LR + RF + XGB + LGBM)
    print("[4/8] set ...")
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, verbose=-1))
    ]
    model = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    result = evaluate_ensemble(X, T, model, 'Hard Voting (4 models)')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 5. set 
    print("[5/8] set ...")
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, verbose=-1))
    ]
    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    result = evaluate_ensemble(X, T, model, 'Soft Voting (4 models)')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 6. Stacking (LR er)
    print("[6/8] Stacking (LR er)...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=8, random_state=42, verbose=-1))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
        cv=3, n_jobs=-1
    )
    result = evaluate_ensemble(X, T, model, 'Stacking (LR meta)')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 7. Stacking (LGBM er)
    print("[7/8] Stacking (LGBM er)...")
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', verbose=-1),
        cv=3, n_jobs=-1
    )
    result = evaluate_ensemble(X, T, model, 'Stacking (LGBM meta)')
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 8. Gradient set 
    print("[8/8] Gradient set typesConfigurationMean ...")
    # Average predictions from multiple LGBMs with different hyperparams
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Training not ConfigurationModel
        models = [
            lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, class_weight='balanced', random_state=42, verbose=-1),
            lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, class_weight='balanced', random_state=43, verbose=-1),
            lgb.LGBMClassifier(n_estimators=700, max_depth=12, learning_rate=0.03, class_weight='balanced', random_state=44, verbose=-1)
        ]
        
        preds = []
        for m in models:
            m.fit(X_train_scaled, T_train)
            preds.append(m.predict_proba(X_test_scaled)[:, 1])
        
        # MeanProbabilityPrediction
        avg_proba = np.mean(preds, axis=0)
        final_pred = (avg_proba > 0.5).astype(int)
        scores.append((final_pred == T_test).mean())
    
    result = {'strategy': 'Multi-LGBM Average', 'adr': np.mean(scores), 'adr_std': np.std(scores)}
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # SummaryResults
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('adr', ascending=False)
    
    print("\n" + "="*70)
    print("set policyComparisonResults byADRSort ")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # SaveResults
    results_df.to_csv('Experiment/ensemble_comparison.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/ensemble_comparison.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_ensemble_comparison()

"""
Ablation 5: Feature type ablation
Evaluate contribution of different feature types
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Feature groups (by type)
DEMOGRAPHIC = ['admission_age', 'gender', 'weight']
SEVERITY = ['sofa_24hours', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo']
RENAL = ['creat', 'creat_k1', 'creat_k2', 'creat_k3']
URINE = ['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr', 'uo_k1', 'uo_k2', 'uo_k3']
METABOLIC = ['bun_k1', 'bun_k2', 'bun_k3', 'pot_k1', 'pot_k2', 'pot_k3']
ACID_BASE = ['ph_k1', 'ph_k2', 'ph_k3']

ALL_FEATURES = DEMOGRAPHIC + SEVERITY + RENAL + URINE + METABOLIC + ACID_BASE


def load_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    valid_mask = ~df['hfd'].isna()
    df = df[valid_mask]
    T = ((df['a1'] == 1) | (df['a2'] == 1) | (df['a3'] == 1)).astype(int).values
    return df, T


def evaluate_features(df, T, features, name):
    """Evaluation Featuregroup """
    X = df[features].copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X_scaled, T):
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                   class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_scaled[train_idx], T[train_idx])
        pred = model.predict(X_scaled[test_idx])
        scores.append((pred == T[test_idx]).mean())
    
    return {'feature_type': name, 'n_features': len(features),
            'adr': np.mean(scores), 'std': np.std(scores)}


def main():
    print("="*70)
    print("Ablation 5: Feature type ablation")
    print("="*70)
    
    df, T = load_data()
    print(f"samples : {len(df)}, RRT Rate: {T.mean():.1%}")
    
    results = []
    
    # Full Model
    results.append(evaluate_features(df, T, ALL_FEATURES, 'Full Model'))
    print(f"Full Model ({len(ALL_FEATURES)}Feature): ADR={results[-1]['adr']:.2%}")
    
    # each type
    feature_groups = [
        (DEMOGRAPHIC, 'Demographic Only'),
        (SEVERITY, 'Severity Only'),
        (RENAL, 'Renal Only'),
        (URINE, 'Urine Only'),
        (METABOLIC, 'Metabolic Only'),
        (ACID_BASE, 'Acid-Base Only'),
    ]
    
    for features, name in feature_groups:
        result = evaluate_features(df, T, features, name)
        results.append(result)
        print(f"{name} ({len(features)}Feature): ADR={result['adr']:.2%}")
    
    # each type
    print("\n each type:")
    removal_groups = [
        ([f for f in ALL_FEATURES if f not in DEMOGRAPHIC], 'w/o Demographic'),
        ([f for f in ALL_FEATURES if f not in SEVERITY], 'w/o Severity'),
        ([f for f in ALL_FEATURES if f not in RENAL], 'w/o Renal'),
        ([f for f in ALL_FEATURES if f not in URINE], 'w/o Urine'),
        ([f for f in ALL_FEATURES if f not in METABOLIC], 'w/o Metabolic'),
        ([f for f in ALL_FEATURES if f not in ACID_BASE], 'w/o Acid-Base'),
    ]
    
    for features, name in removal_groups:
        result = evaluate_features(df, T, features, name)
        results.append(result)
        print(f"{name}: ADR={result['adr']:.2%}")
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df['drop'] = results_df['adr'].iloc[0] - results_df['adr']
    results_df.to_csv('Experiment /ablation_feature_type.csv', index=False)
    
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    print(results_df.to_string(index=False))
    print("\nResults saved: Experiment /ablation_feature_type.csv")


if __name__ == '__main__':
    main()

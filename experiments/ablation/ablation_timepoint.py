"""
Ablation 1: Time point ablation
Evaluate contribution of three time points k1/k2/k3
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Feature groups
BASELINE_FEATURES = ['admission_age', 'gender', 'weight', 'sofa_24hours',
                     'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
                     'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr']

K1_FEATURES = ['uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1']
K2_FEATURES = ['uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2']
K3_FEATURES = ['uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3']


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
    
    return {'config': name, 'n_features': len(features), 
            'adr': np.mean(scores), 'std': np.std(scores)}


def main():
    print("="*70)
    print("Ablation 1: Time point ablation")
    print("="*70)
    
    df, T = load_data()
    print(f"samples : {len(df)}, RRT Rate: {T.mean():.1%}")
    
    results = []
    
    # Full Model
    all_features = BASELINE_FEATURES + K1_FEATURES + K2_FEATURES + K3_FEATURES
    results.append(evaluate_features(df, T, all_features, 'Full Model'))
    print(f"Full Model: ADR={results[-1]['adr']:.2%}")
    
    # Baseline only
    results.append(evaluate_features(df, T, BASELINE_FEATURES, 'Baseline Only'))
    print(f"Baseline Only: ADR={results[-1]['adr']:.2%}")
    
    # Baseline + k1
    results.append(evaluate_features(df, T, BASELINE_FEATURES + K1_FEATURES, 'Baseline + k1'))
    print(f"Baseline + k1: ADR={results[-1]['adr']:.2%}")
    
    # Baseline + k1 + k2
    results.append(evaluate_features(df, T, BASELINE_FEATURES + K1_FEATURES + K2_FEATURES, 'Baseline + k1 + k2'))
    print(f"Baseline + k1 + k2: ADR={results[-1]['adr']:.2%}")
    
    # each time point
    results.append(evaluate_features(df, T, BASELINE_FEATURES + K2_FEATURES + K3_FEATURES, 'w/o k1'))
    print(f"w/o k1: ADR={results[-1]['adr']:.2%}")
    
    results.append(evaluate_features(df, T, BASELINE_FEATURES + K1_FEATURES + K3_FEATURES, 'w/o k2'))
    print(f"w/o k2: ADR={results[-1]['adr']:.2%}")
    
    results.append(evaluate_features(df, T, BASELINE_FEATURES + K1_FEATURES + K2_FEATURES, 'w/o k3'))
    print(f"w/o k3: ADR={results[-1]['adr']:.2%}")
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df['drop'] = results_df['adr'].iloc[0] - results_df['adr']
    results_df.to_csv('Experiment /ablation_timepoint.csv', index=False)
    
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    print(results_df.to_string(index=False))
    print("\nResults saved: Experiment /ablation_timepoint.csv")


if __name__ == '__main__':
    main()

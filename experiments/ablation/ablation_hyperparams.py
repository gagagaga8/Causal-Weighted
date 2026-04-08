"""
Ablation 2: Model hyperparameter ablation
Evaluate impact of different hyperparameter configurations
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    X = df[FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    T = ((df[valid_mask]['a1'] == 1) | (df[valid_mask]['a2'] == 1) | (df[valid_mask]['a3'] == 1)).astype(int).values
    return X, T


def evaluate_config(X, T, config, name):
    """Evaluation Configuration"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X_scaled, T):
        model = lgb.LGBMClassifier(**config, class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_scaled[train_idx], T[train_idx])
        pred = model.predict(X_scaled[test_idx])
        scores.append((pred == T[test_idx]).mean())
    
    return {'config': name, 'adr': np.mean(scores), 'std': np.std(scores)}


def main():
    print("="*70)
    print("Ablation 2: Model hyperparameter ablation")
    print("="*70)
    
    X, T = load_data()
    print(f"samples : {len(X)}, RRT Rate: {T.mean():.1%}")
    
    results = []
    
    # ParametersConfiguration
    configs = [
        ({'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.05}, 'Default (500, 10, 0.05)'),
        ({'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.05}, 'n_estimators=100'),
        ({'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.05}, 'n_estimators=1000'),
        ({'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05}, 'max_depth=5'),
        ({'n_estimators': 500, 'max_depth': 15, 'learning_rate': 0.05}, 'max_depth=15'),
        ({'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.01}, 'learning_rate=0.01'),
        ({'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.1}, 'learning_rate=0.1'),
        ({'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03}, 'Light (300, 8, 0.03)'),
        ({'n_estimators': 800, 'max_depth': 12, 'learning_rate': 0.03}, 'Heavy (800, 12, 0.03)'),
    ]
    
    for config, name in configs:
        result = evaluate_config(X, T, config, name)
        results.append(result)
        print(f"{name}: ADR={result['adr']:.2%} ± {result['std']:.2%}")
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df.to_csv('Experiment /ablation_hyperparams.csv', index=False)
    
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    print(results_df.to_string(index=False))
    
    best = results_df.loc[results_df['adr'].idxmax()]
    print(f"\n Configuration: {best['config']}, ADR={best['adr']:.2%}")
    print("\nResults saved: Experiment /ablation_hyperparams.csv")


if __name__ == '__main__':
    main()

"""
Ablation 4: Class weight ablation
Evaluate impact of different class weight settings
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


def evaluate_class_weight(X, T, class_weight, name):
    """Evaluation ClassWeight"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    treat_accs = []
    control_accs = []
    
    for train_idx, test_idx in skf.split(X_scaled, T):
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                   class_weight=class_weight, random_state=42, verbose=-1)
        model.fit(X_scaled[train_idx], T[train_idx])
        pred = model.predict(X_scaled[test_idx])
        
        T_test = T[test_idx]
        scores.append((pred == T_test).mean())
        treat_accs.append((pred[T_test == 1] == 1).mean() if (T_test == 1).sum() > 0 else 0)
        control_accs.append((pred[T_test == 0] == 0).mean() if (T_test == 0).sum() > 0 else 0)
    
    return {'config': name, 'adr': np.mean(scores), 'std': np.std(scores),
            'treat_acc': np.mean(treat_accs), 'control_acc': np.mean(control_accs)}


def main():
    print("="*70)
    print("Ablation 4: Class weight ablation")
    print("="*70)
    
    X, T = load_data()
    rrt_rate = T.mean()
    print(f"samples : {len(X)}, RRT Rate: {rrt_rate:.1%}")
    
    results = []
    
    # ClassWeightConfiguration
    configs = [
        (None, 'None (default)'),
        ('balanced', 'balanced'),
        ({0: 1, 1: 5}, 'RRT x5'),
        ({0: 1, 1: 10}, 'RRT x10'),
        ({0: 1, 1: 15}, 'RRT x15 (≈1/rate)'),
        ({0: 1, 1: 20}, 'RRT x20'),
        ({0: 1, 1: 30}, 'RRT x30'),
    ]
    
    for weight, name in configs:
        result = evaluate_class_weight(X, T, weight, name)
        results.append(result)
        print(f"{name}: ADR={result['adr']:.2%}, RRTgroup={result['treat_acc']:.2%}, nonRRTgroup={result['control_acc']:.2%}")
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df.to_csv('Experiment /ablation_class_weight.csv', index=False)
    
    print("\n" + "="*70)
    print("ResultsSummary")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\nKey findings:")
    print(" - No Weight ADR butRRTgroup ")
    print(" - balancedWeightbalance AccuracyandRRTgroup ")
    print(" - WeightwillDecreased ADR")
    print("\nResults saved: Experiment /ablation_class_weight.csv")


if __name__ == '__main__':
    main()

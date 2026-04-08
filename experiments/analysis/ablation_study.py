"""
Ablation study: analyze feature importance
Stepwise feature removal to observe model performance changes
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Feature groups
FEATURE_GROUPS = {
    'demographic': ['admission_age', 'gender', 'weight'],
    'severity': ['sofa_24hours', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo'],
    'baseline': ['creat', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
    'k1': ['uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1'],
    'k2': ['uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2'],
    'k3': ['uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3']
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def load_data(data_path):
    df = pd.read_csv(data_path)
    
    X = df[ALL_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask]
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T


def evaluate_features(X, T, features_to_use):
    """Evaluation FeatureModelPerformance"""
    X_subset = X[features_to_use].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X_subset, T):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_train_scaled, T_train)
        pred = model.predict(X_test_scaled)
        scores.append((pred == T_test).mean())
    
    return np.mean(scores), np.std(scores)


def ablation_study(X, T):
    print("="*70)
    print("AblationExperiment Feature to Analysis")
    print("="*70)
    
    # 1. Full Model
    print("\n1. Full Model All Feature ")
    full_mean, full_std = evaluate_features(X, T, ALL_FEATURES)
    print(f"   ADR: {full_mean:.2%} +/- {full_std:.2%}")
    
    results = [{'feature_set': 'Full Model', 'n_features': len(ALL_FEATURES), 
                'adr': full_mean, 'std': full_std, 'drop': 0.0}]
    
    # 2. each Featuregroup
    print("\n2. Featuregroup")
    for group_name, group_features in FEATURE_GROUPS.items():
        remaining = [f for f in ALL_FEATURES if f not in group_features]
        mean, std = evaluate_features(X, T, remaining)
        drop = full_mean - mean
        
        print(f" {group_name:12s}: ADR={mean:.2%} +/- {std:.2%}, decline={drop:.2%}")
        results.append({
            'feature_set': f'w/o {group_name}',
            'n_features': len(remaining),
            'adr': mean,
            'std': std,
            'drop': drop
        })
    
    # 3. only useeach Featuregroup
    print("\n3. only use Featuregroup")
    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in X.columns]
        if len(available) == 0:
            continue
        mean, std = evaluate_features(X, T, available)
        
        print(f"   only {group_name:12s}: ADR={mean:.2%} +/- {std:.2%}")
        results.append({
            'feature_set': f'only {group_name}',
            'n_features': len(available),
            'adr': mean,
            'std': std,
            'drop': full_mean - mean
        })
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('drop', ascending=False)
    
    print("\n" + "="*70)
    print(" to by PerformancedeclineSort ")
    print("="*70)
    print(results_df[results_df['feature_set'].str.startswith('w/o')].to_string(index=False))
    
    results_df.to_csv('output/ablation_results.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: output/ablation_results.csv")
    
    return results_df


if __name__ == '__main__':
    X, T = load_data('data/dwols_full_with_uo.csv')
    print(f"\nData: {len(X)} samples, {len(ALL_FEATURES)} Feature")
    print(f"Treatmentgroup: {T.sum()} ({T.mean():.1%})")
    
    results = ablation_study(X, T)

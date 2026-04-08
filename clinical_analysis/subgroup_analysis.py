"""
Subgroup analysis: model performance across patient subgroups
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_full_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    
    X = df[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask]
    df_valid = df[valid_mask].copy()
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T, df_valid


def evaluate_subgroup(X, T, mask, subgroup_name):
    """Evaluate specific subgroup"""
    X_sub = X[mask].values
    T_sub = T[mask]
    
    if len(X_sub) < 100:
        return None
    
    skf = StratifiedKFold(n_splits=min(5, len(T_sub[T_sub==1]) // 2), shuffle=True, random_state=42)
    scores = []
    
    try:
        for train_idx, test_idx in skf.split(X_sub, T_sub):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            T_train, T_test = T_sub[train_idx], T_sub[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.05,
                class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
            )
            model.fit(X_train_scaled, T_train)
            pred = model.predict(X_test_scaled)
            scores.append((pred == T_test).mean())
        
        return {
            'subgroup': subgroup_name,
            'n': len(X_sub),
            'treat_rate': T_sub.mean(),
            'adr': np.mean(scores),
            'std': np.std(scores)
        }
    except:
        return None


def subgroup_analysis():
    print("="*70)
    print("Subgroup analysis")
    print("="*70)
    
    X, T, df = load_full_data()
    print(f"\nOverall: {len(X)} samples, Treatmentgroup: {T.sum()} ({T.mean():.1%})\n")
    
    results = []
    
    # 1. byageGroup
    print("1. ageGroup")
    age_groups = [
        (df['admission_age'] < 50, '< 50 '),
        ((df['admission_age'] >= 50) & (df['admission_age'] < 65), '50-65 '),
        ((df['admission_age'] >= 65) & (df['admission_age'] < 75), '65-75 '),
        (df['admission_age'] >= 75, '≥ 75 ')
    ]
    
    for mask, name in age_groups:
        result = evaluate_subgroup(X, T, mask, f'Age: {name}')
        if result:
            print(f"   {name:12s}: n={result['n']:4d}, Treatment={result['treat_rate']:.1%}, ADR={result['adr']:.2%} +/- {result['std']:.2%}")
            results.append(result)
    
    # 2. bySexGroup
    print("\n2. SexGroup")
    gender_groups = [
        (df['gender'] == 'M', ' '),
        (df['gender'] == 'F', ' ')
    ]
    
    for mask, name in gender_groups:
        result = evaluate_subgroup(X, T, mask, f'Gender: {name}')
        if result:
            print(f"   {name:12s}: n={result['n']:4d}, Treatment={result['treat_rate']:.1%}, ADR={result['adr']:.2%} +/- {result['std']:.2%}")
            results.append(result)
    
    # 3. by Severe Group (SOFAScore)
    print("\n3. SOFAScoreGroup")
    sofa_groups = [
        (df['sofa_24hours'] < 5, 'SOFA < 5'),
        ((df['sofa_24hours'] >= 5) & (df['sofa_24hours'] < 10), 'SOFA 5-10'),
        (df['sofa_24hours'] >= 10, 'SOFA ≥ 10')
    ]
    
    for mask, name in sofa_groups:
        result = evaluate_subgroup(X, T, mask, f'SOFA: {name}')
        if result:
            print(f"   {name:12s}: n={result['n']:4d}, Treatment={result['treat_rate']:.1%}, ADR={result['adr']:.2%} +/- {result['std']:.2%}")
            results.append(result)
    
    # 4. byAKISevere Group
    print("\n4. AKI Group")
    aki_groups = [
        (df['aki_stage'] == 1, 'AKI Stage 1'),
        (df['aki_stage'] == 2, 'AKI Stage 2'),
        (df['aki_stage'] == 3, 'AKI Stage 3')
    ]
    
    for mask, name in aki_groups:
        result = evaluate_subgroup(X, T, mask, f'AKI: {name}')
        if result:
            print(f"   {name:12s}: n={result['n']:4d}, Treatment={result['treat_rate']:.1%}, ADR={result['adr']:.2%} +/- {result['std']:.2%}")
            results.append(result)
    
    # 5. byUrine outputGroup
    print("\n5. Urine output Group")
    uo_groups = [
        (df['uo_rt_24hr'] < 0.5, 'Urine output < 0.5'),
        ((df['uo_rt_24hr'] >= 0.5) & (df['uo_rt_24hr'] < 1.0), 'Urine output 0.5-1.0'),
        (df['uo_rt_24hr'] >= 1.0, 'Urine output ≥ 1.0')
    ]
    
    for mask, name in uo_groups:
        result = evaluate_subgroup(X, T, mask, f'UO: {name}')
        if result:
            print(f"   {name:12s}: n={result['n']:4d}, Treatment={result['treat_rate']:.1%}, ADR={result['adr']:.2%} +/- {result['std']:.2%}")
            results.append(result)
    
    # SaveResults
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("Subgroup performance summary")
    print("="*70)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('output/subgroup_analysis.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: output/subgroup_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    results = subgroup_analysis()

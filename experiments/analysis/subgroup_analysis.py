"""
Subgroup analysis experiment
EvaluationModelinnot Patient groupinPerformance Analysis 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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


def load_data_with_subgroups():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    valid_mask = ~df['hfd'].isna()
    df_valid = df[valid_mask].copy()
    
    X = df_valid[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    # Create groupLabel
    subgroups = {}
    
    # ageGroup
    age = df_valid['admission_age'].values
    subgroups['age_<50'] = age < 50
    subgroups['age_50-65'] = (age >= 50) & (age < 65)
    subgroups['age_65-75'] = (age >= 65) & (age < 75)
    subgroups['age_≥75'] = age >= 75
    
    # SexGroup
    gender = X['gender'].values
    subgroups['Male'] = gender == 1
    subgroups['Female'] = gender == 0
    
    # SOFAScoreGroup
    sofa = df_valid['sofa_24hours'].values
    subgroups['SOFA_<5'] = sofa < 5
    subgroups['SOFA_5-10'] = (sofa >= 5) & (sofa < 10)
    subgroups['SOFA_≥10'] = sofa >= 10
    
    # AKI Group
    aki = df_valid['aki_stage'].values
    subgroups['AKI_Stage1'] = aki == 1
    subgroups['AKI_Stage2'] = aki == 2
    subgroups['AKI_Stage3'] = aki == 3
    
    return X.values, T, subgroups


def run_subgroup_analysis():
    print("="*70)
    print("Subgroup analysis (fairness evaluation)")
    print("="*70)
    
    X, T, subgroups = load_data_with_subgroups()
    print(f"\n Sample Size: {len(X)}, RRT Rate: {T.mean():.1%}\n")
    
    # TrainingFull Modeland Prediction
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_pred = np.zeros(len(T))
    all_proba = np.zeros(len(T))
    
    print("TrainingModel...")
    for train_idx, test_idx in skf.split(X, T):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train, T[train_idx])
        all_pred[test_idx] = model.predict(X_test)
        all_proba[test_idx] = model.predict_proba(X_test)[:, 1]
    
    # Analysiseach groupPerformance
    results = []
    
    print("\n" + "-"*80)
    print(f"{' group':15s} | {'Sample Size':>8s} | {'RRT Rate':>8s} | {'ADR':>8s} | {'Recall':>8s} | {'Specificity':>8s}")
    print("-"*80)
    
    # byClassGroup 
    categories = {
        'age ': ['age_<50', 'age_50-65', 'age_65-75', 'age_≥75'],
        'Sex ': ['Male', 'Female'],
        'SOFA ': ['SOFA_<5', 'SOFA_5-10', 'SOFA_≥10'],
        'AKI ': ['AKI_Stage1', 'AKI_Stage2', 'AKI_Stage3']
    }
    
    for cat_name, groups in categories.items():
        print(f"\n {cat_name} ")
        for group_name in groups:
            mask = subgroups[group_name]
            n = mask.sum()
            if n < 10:
                continue
                
            T_sub = T[mask]
            pred_sub = all_pred[mask]
            
            rrt_rate = T_sub.mean()
            adr = (pred_sub == T_sub).mean()
            
            # Recall RRTgroup 
            rrt_mask = T_sub == 1
            recall = (pred_sub[rrt_mask] == 1).mean() if rrt_mask.sum() > 0 else 0
            
            # Specificity nonRRTgroupCorrect 
            non_rrt_mask = T_sub == 0
            spec = (pred_sub[non_rrt_mask] == 0).mean() if non_rrt_mask.sum() > 0 else 0
            
            results.append({
                'category': cat_name,
                'subgroup': group_name,
                'n': n,
                'rrt_rate': rrt_rate,
                'adr': adr,
                'recall': recall,
                'specificity': spec
            })
            
            print(f"{group_name:15s} | {n:>8d} | {rrt_rate:>8.1%} | {adr:>8.2%} | {recall:>8.2%} | {spec:>8.2%}")
    
    print("-"*80)
    
    # Analysis
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(" Metric")
    print("="*70)
    
    adr_values = results_df['adr'].values
    print(f"\nADRrange: {adr_values.min():.2%} - {adr_values.max():.2%}")
    print(f"ADR : {adr_values.max() - adr_values.min():.2%}")
    print(f"ADRStandard deviation: {adr_values.std():.2%}")
    
    # each Class Maximum 
    for cat_name in categories.keys():
        cat_df = results_df[results_df['category'] == cat_name]
        if len(cat_df) > 1:
            gap = cat_df['adr'].max() - cat_df['adr'].min()
            print(f"{cat_name} ADR : {gap:.2%}")
    
    # SaveResults
    results_df.to_csv('Experiment/subgroup_analysis.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/subgroup_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_subgroup_analysis()

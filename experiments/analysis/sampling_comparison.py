"""
SamplingpolicyComparisonExperiment
Comparisonnot ProcessingClassnotbalanceMethodpairModelPerformance 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sampling 
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imblearnnotInstall willonly use Method")

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


def evaluate_with_sampling(X, T, sampler_name, sampler=None):
    """Evaluate model with specified sampling strategy"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    rrt_recalls = []
    non_rrt_specs = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # useSamplingpolicy
        if sampler is not None:
            try:
                X_resampled, T_resampled = sampler.fit_resample(X_train_scaled, T_train)
            except Exception as e:
                X_resampled, T_resampled = X_train_scaled, T_train
        else:
            X_resampled, T_resampled = X_train_scaled, T_train
        
        # TrainingModel
        if sampler_name == 'Class Weight (balanced)':
            model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                        class_weight='balanced', random_state=42, verbose=-1)
        else:
            model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                        random_state=42, verbose=-1)
        
        model.fit(X_resampled, T_resampled)
        pred = model.predict(X_test_scaled)
        
        # ComputationMetric
        scores.append((pred == T_test).mean())
        
        # RRTgroupRecall
        rrt_mask = T_test == 1
        if rrt_mask.sum() > 0:
            rrt_recalls.append((pred[rrt_mask] == 1).mean())
        
        # nonRRTgroupSpecificity
        non_rrt_mask = T_test == 0
        if non_rrt_mask.sum() > 0:
            non_rrt_specs.append((pred[non_rrt_mask] == 0).mean())
    
    return {
        'strategy': sampler_name,
        'adr': np.mean(scores),
        'adr_std': np.std(scores),
        'rrt_recall': np.mean(rrt_recalls) if rrt_recalls else 0,
        'non_rrt_spec': np.mean(non_rrt_specs) if non_rrt_specs else 0
    }


def run_sampling_comparison():
    print("="*70)
    print("SamplingpolicyComparisonExperiment")
    print("="*70)
    
    X, T = load_data()
    print(f"\nDataset: {len(X)} samples")
    print(f"ClassDistribution: {Counter(T)}")
    print(f"RRT Rate: {T.mean():.1%}\n")
    
    results = []
    
    # 1. No Sampling 
    print("[1/8] No Sampling ...")
    result = evaluate_with_sampling(X, T, 'No Sampling', None)
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    # 2. ClassWeightbalance
    print("[2/8] ClassWeightbalance...")
    result = evaluate_with_sampling(X, T, 'Class Weight (balanced)', None)
    results.append(result)
    print(f"      ADR: {result['adr']:.2%}")
    
    if HAS_IMBLEARN:
        # 3. Random Sampling
        print("[3/8] Random Sampling...")
        sampler = RandomOverSampler(random_state=42)
        result = evaluate_with_sampling(X, T, 'Random OverSampling', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 4. SMOTE
        print("[4/8] SMOTE...")
        sampler = SMOTE(random_state=42)
        result = evaluate_with_sampling(X, T, 'SMOTE', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 5. ADASYN
        print("[5/8] ADASYN...")
        sampler = ADASYN(random_state=42)
        result = evaluate_with_sampling(X, T, 'ADASYN', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 6. Random Sampling
        print("[6/8] Random Sampling...")
        sampler = RandomUnderSampler(random_state=42)
        result = evaluate_with_sampling(X, T, 'Random UnderSampling', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 7. SMOTE + Tomek
        print("[7/8] SMOTE + Tomek Links...")
        sampler = SMOTETomek(random_state=42)
        result = evaluate_with_sampling(X, T, 'SMOTE + Tomek', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 8. SMOTE + ENN
        print("[8/8] SMOTE + ENN...")
        sampler = SMOTEENN(random_state=42)
        result = evaluate_with_sampling(X, T, 'SMOTE + ENN', sampler)
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
    
    # SummaryResults
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('adr', ascending=False)
    
    print("\n" + "="*70)
    print("SamplingpolicyComparisonResults byADRSort ")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # SaveResults
    results_df.to_csv('Experiment/sampling_comparison.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/sampling_comparison.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_sampling_comparison()

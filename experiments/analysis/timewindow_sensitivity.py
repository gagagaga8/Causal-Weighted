"""
 Experiment
Testnotsimultaneously PartitionpairModelPerformance 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Feature not time pointFeature 
BASE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'
]

# time pointFeature 
TIMEPOINT_FEATURES = ['uo', 'bun', 'pot', 'ph', 'creat']


def load_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    valid_mask = ~df['hfd'].isna()
    df_valid = df[valid_mask].copy()
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    return df_valid, T


def get_features_for_config(config_name):
    """according toConfiguration FeatureColumnTable"""
    features = BASE_FEATURES.copy()
    
    if config_name == 'Baseline Only':
        pass # onlyuse Feature
    elif config_name == 'k1 Only':
        features.extend([f'{f}_k1' for f in TIMEPOINT_FEATURES])
    elif config_name == 'k2 Only':
        features.extend([f'{f}_k2' for f in TIMEPOINT_FEATURES])
    elif config_name == 'k3 Only':
        features.extend([f'{f}_k3' for f in TIMEPOINT_FEATURES])
    elif config_name == 'k1 + k2':
        features.extend([f'{f}_k1' for f in TIMEPOINT_FEATURES])
        features.extend([f'{f}_k2' for f in TIMEPOINT_FEATURES])
    elif config_name == 'k2 + k3':
        features.extend([f'{f}_k2' for f in TIMEPOINT_FEATURES])
        features.extend([f'{f}_k3' for f in TIMEPOINT_FEATURES])
    elif config_name == 'k1 + k3':
        features.extend([f'{f}_k1' for f in TIMEPOINT_FEATURES])
        features.extend([f'{f}_k3' for f in TIMEPOINT_FEATURES])
    elif config_name == 'Full (k1+k2+k3)':
        features.extend([f'{f}_k1' for f in TIMEPOINT_FEATURES])
        features.extend([f'{f}_k2' for f in TIMEPOINT_FEATURES])
        features.extend([f'{f}_k3' for f in TIMEPOINT_FEATURES])
    
    return features


def evaluate_timewindow(df, T, config_name):
    """Evaluation Configuration"""
    features = get_features_for_config(config_name)
    
    X = df[features].copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0).values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train_scaled, T_train)
        pred = model.predict(X_test_scaled)
        scores.append((pred == T_test).mean())
    
    return {
        'config': config_name,
        'n_features': len(features),
        'adr': np.mean(scores),
        'adr_std': np.std(scores)
    }


def run_timewindow_sensitivity():
    print("="*70)
    print(" Experiment")
    print("="*70)
    
    df, T = load_data()
    print(f"\nDataset: {len(df)} samples\n")
    
    # All Configuration
    configs = [
        'Baseline Only',
        'k1 Only',
        'k2 Only', 
        'k3 Only',
        'k1 + k2',
        'k2 + k3',
        'k1 + k3',
        'Full (k1+k2+k3)'
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config}...")
        result = evaluate_timewindow(df, T, config)
        results.append(result)
        print(f"      Features: {result['n_features']}, ADR: {result['adr']:.2%}")
    
    # SummaryResults
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('adr', ascending=False)
    
    print("\n" + "="*70)
    print(" ConfigurationComparisonResults byADRSort ")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Computationeach time point 
    print("\n" + "="*70)
    print("each time point Analysis")
    print("="*70)
    
    baseline_adr = results_df[results_df['config'] == 'Baseline Only']['adr'].values[0]
    full_adr = results_df[results_df['config'] == 'Full (k1+k2+k3)']['adr'].values[0]
    
    for tp in ['k1', 'k2', 'k3']:
        tp_only_adr = results_df[results_df['config'] == f'{tp} Only']['adr'].values[0]
        marginal = tp_only_adr - baseline_adr
        print(f" {tp} : +{marginal:.2%}")
    
    print(f"\n  Full Model vs Baseline: +{full_adr - baseline_adr:.2%}")
    
    # SaveResults
    results_df.to_csv('Experiment/timewindow_sensitivity.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/timewindow_sensitivity.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_timewindow_sensitivity()

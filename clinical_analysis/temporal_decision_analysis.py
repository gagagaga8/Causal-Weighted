"""
Temporal decision point analysis: treatment decision performance at k1/k2/k3
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Features for three decision points
K1_FEATURES = ['admission_age', 'gender', 'weight', 'sofa_24hours',
               'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
               'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
               'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1']

K2_FEATURES = K1_FEATURES + ['uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2']

K3_FEATURES = K2_FEATURES + ['uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3']


def load_data():
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    
    X = df[K3_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask]
    df_valid = df[valid_mask].copy()
    
    # Treatment labels for three time points
    T1 = (df_valid['a1'] == 1).astype(int).values
    T2 = (df_valid['a2'] == 1).astype(int).values
    T3 = (df_valid['a3'] == 1).astype(int).values
    T_any = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T1, T2, T3, T_any


def evaluate_timepoint(X, T, features, timepoint_name):
    """Evaluation time pointPredictionPerformance"""
    X_subset = X[features].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    treat_accs = []
    no_treat_accs = []
    
    for train_idx, test_idx in skf.split(X_subset, T):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_train_scaled, T_train)
        pred = model.predict(X_test_scaled)
        
        # Accuracy
        scores.append((pred == T_test).mean())
        
        # Treatmentgroup and control group Accuracy
        treat_mask = T_test == 1
        no_treat_mask = T_test == 0
        
        if treat_mask.sum() > 0:
            treat_accs.append((pred[treat_mask] == T_test[treat_mask]).mean())
        if no_treat_mask.sum() > 0:
            no_treat_accs.append((pred[no_treat_mask] == T_test[no_treat_mask]).mean())
    
    return {
        'timepoint': timepoint_name,
        'n_features': len(features),
        'treat_rate': T.mean(),
        'adr': np.mean(scores),
        'adr_std': np.std(scores),
        'treat_acc': np.mean(treat_accs) if treat_accs else 0,
        'no_treat_acc': np.mean(no_treat_accs) if no_treat_accs else 0
    }


def temporal_analysis():
    print("="*70)
    print("Temporal decision pointAnalysis")
    print("="*70)
    
    X, T1, T2, T3, T_any = load_data()
    print(f"\nData: {len(X)} samples")
    print(f"k1Treatment : {T1.mean():.1%} ({T1.sum()})")
    print(f"k2Treatment : {T2.mean():.1%} ({T2.sum()})")
    print(f"k3Treatment : {T3.mean():.1%} ({T3.sum()})")
    print(f"Any time point: {T_any.mean():.1%} ({T_any.sum()})\n")
    
    results = []
    
    # 1. k1decision point onlyusek1FeaturePredictionk1Treatment 
    print("1. k1decision pointPrediction")
    result_k1 = evaluate_timepoint(X, T1, K1_FEATURES, 'k1')
    print(f" Features={result_k1['n_features']}, Treatment ={result_k1['treat_rate']:.1%}")
    print(f"   ADR={result_k1['adr']:.2%} +/- {result_k1['adr_std']:.2%}")
    print(f" TreatmentgroupAccuracy={result_k1['treat_acc']:.2%}, pair groupAccuracy={result_k1['no_treat_acc']:.2%}")
    results.append(result_k1)
    
    # 2. k2decision point usek1+k2FeaturePredictionk2Treatment 
    print("\n2. k2decision pointPrediction")
    result_k2 = evaluate_timepoint(X, T2, K2_FEATURES, 'k2')
    print(f" Features={result_k2['n_features']}, Treatment ={result_k2['treat_rate']:.1%}")
    print(f"   ADR={result_k2['adr']:.2%} +/- {result_k2['adr_std']:.2%}")
    print(f" TreatmentgroupAccuracy={result_k2['treat_acc']:.2%}, pair groupAccuracy={result_k2['no_treat_acc']:.2%}")
    results.append(result_k2)
    
    # 3. k3decision point usek1+k2+k3FeaturePredictionk3Treatment 
    print("\n3. k3decision pointPrediction")
    result_k3 = evaluate_timepoint(X, T3, K3_FEATURES, 'k3')
    print(f" Features={result_k3['n_features']}, Treatment ={result_k3['treat_rate']:.1%}")
    print(f"   ADR={result_k3['adr']:.2%} +/- {result_k3['adr_std']:.2%}")
    print(f" TreatmentgroupAccuracy={result_k3['treat_acc']:.2%}, pair groupAccuracy={result_k3['no_treat_acc']:.2%}")
    results.append(result_k3)
    
    # 4. Any time point useall FeaturePredictionis needtoTreatment 
    print("\n4. Treatment prediction at any time point (overall decision)")
    result_any = evaluate_timepoint(X, T_any, K3_FEATURES, 'any')
    print(f" Features={result_any['n_features']}, Treatment ={result_any['treat_rate']:.1%}")
    print(f"   ADR={result_any['adr']:.2%} +/- {result_any['adr_std']:.2%}")
    print(f" TreatmentgroupAccuracy={result_any['treat_acc']:.2%}, pair groupAccuracy={result_any['no_treat_acc']:.2%}")
    results.append(result_any)
    
    # SaveResults
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("Temporal Summary")
    print("="*70)
    print(results_df[['timepoint', 'treat_rate', 'adr', 'treat_acc', 'no_treat_acc']].to_string(index=False))
    
    results_df.to_csv('output/temporal_analysis.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: output/temporal_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    results = temporal_analysis()

"""
Bootstrap confidence interval: evaluate statistical significance of model performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tqdm import tqdm

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


def bootstrap_iteration(X, T, random_state):
    """Single bootstrap iteration"""
    n_samples = len(X)
    
    # BootstrapSampling
    indices = np.random.RandomState(random_state).choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    T_boot = T[indices]
    
    # OOBsamples notbySamplingsamples 
    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[indices] = False
    X_oob = X[oob_mask]
    T_oob = T[oob_mask]
    
    if len(X_oob) < 10: # OOBsamples 
        return None
    
    # TrainingModel
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)
    X_oob_scaled = scaler.transform(X_oob)
    
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=random_state, verbose=-1, n_jobs=1
    )
    model.fit(X_boot_scaled, T_boot)
    
    # OOBPrediction
    pred = model.predict(X_oob_scaled)
    acc = (pred == T_oob).mean()
    
    # Treatmentgroup and control group Accuracy
    treat_mask = T_oob == 1
    no_treat_mask = T_oob == 0
    
    treat_acc = (pred[treat_mask] == T_oob[treat_mask]).mean() if treat_mask.sum() > 0 else np.nan
    no_treat_acc = (pred[no_treat_mask] == T_oob[no_treat_mask]).mean() if no_treat_mask.sum() > 0 else np.nan
    
    return {'adr': acc, 'treat_acc': treat_acc, 'no_treat_acc': no_treat_acc}


def bootstrap_analysis(n_iterations=1000):
    print("="*70)
    print(f"BootstrapConfidence intervalComputation (n={n_iterations})")
    print("="*70)
    
    X, T = load_data()
    print(f"\nData: {len(X)} samples, Treatmentgroup: {T.sum()} ({T.mean():.1%})")
    print(f"StartBootstrapSampling...\n")
    
    results = []
    for i in tqdm(range(n_iterations), desc="Bootstrap "):
        result = bootstrap_iteration(X, T, random_state=i)
        if result is not None:
            results.append(result)
    
    # statisticsResults
    adrs = [r['adr'] for r in results]
    treat_accs = [r['treat_acc'] for r in results if not np.isnan(r['treat_acc'])]
    no_treat_accs = [r['no_treat_acc'] for r in results if not np.isnan(r['no_treat_acc'])]
    
    # ComputationConfidence interval
    adr_mean = np.mean(adrs)
    adr_std = np.std(adrs)
    adr_ci_lower = np.percentile(adrs, 2.5)
    adr_ci_upper = np.percentile(adrs, 97.5)
    
    treat_mean = np.mean(treat_accs) if treat_accs else np.nan
    treat_std = np.std(treat_accs) if treat_accs else np.nan
    treat_ci_lower = np.percentile(treat_accs, 2.5) if treat_accs else np.nan
    treat_ci_upper = np.percentile(treat_accs, 97.5) if treat_accs else np.nan
    
    no_treat_mean = np.mean(no_treat_accs) if no_treat_accs else np.nan
    no_treat_std = np.std(no_treat_accs) if no_treat_accs else np.nan
    no_treat_ci_lower = np.percentile(no_treat_accs, 2.5) if no_treat_accs else np.nan
    no_treat_ci_upper = np.percentile(no_treat_accs, 97.5) if no_treat_accs else np.nan
    
    print("\n" + "="*70)
    print("BootstrapResults")
    print("="*70)
    print(f"\nOverallADR:")
    print(f" value: {adr_mean:.2%} ± {adr_std:.2%}")
    print(f"  95% CI: [{adr_ci_lower:.2%}, {adr_ci_upper:.2%}]")
    
    print(f"\nTreatmentgroupAccuracy:")
    print(f" value: {treat_mean:.2%} ± {treat_std:.2%}")
    print(f"  95% CI: [{treat_ci_lower:.2%}, {treat_ci_upper:.2%}]")
    
    print(f"\npair groupAccuracy:")
    print(f" value: {no_treat_mean:.2%} ± {no_treat_std:.2%}")
    print(f"  95% CI: [{no_treat_ci_lower:.2%}, {no_treat_ci_upper:.2%}]")
    
    # SaveResults
    summary = pd.DataFrame([
        {
            'metric': 'Overall ADR',
            'mean': adr_mean,
            'std': adr_std,
            'ci_lower': adr_ci_lower,
            'ci_upper': adr_ci_upper
        },
        {
            'metric': 'Treatment Group Acc',
            'mean': treat_mean,
            'std': treat_std,
            'ci_lower': treat_ci_lower,
            'ci_upper': treat_ci_upper
        },
        {
            'metric': 'Control Group Acc',
            'mean': no_treat_mean,
            'std': no_treat_std,
            'ci_lower': no_treat_ci_lower,
            'ci_upper': no_treat_ci_upper
        }
    ])
    
    summary.to_csv('output/bootstrap_ci.csv', index=False, encoding='utf-8-sig')
    
    # SaveAll IterationResults
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/bootstrap_iterations.csv', index=False, encoding='utf-8-sig')
    
    print("\nResults saved:")
    print("  - output/bootstrap_ci.csv")
    print("  - output/bootstrap_iterations.csv")
    
    return summary


if __name__ == '__main__':
    summary = bootstrap_analysis(n_iterations=1000)

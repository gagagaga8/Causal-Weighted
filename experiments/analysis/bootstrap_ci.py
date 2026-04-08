"""
BootstrapConfidence intervalAnalysis
ComputationModelPerformanceMetricstatisticsSignificance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy import stats
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


def bootstrap_cv(X, T, n_bootstrap=1000):
    """BootstrapCross-validation"""
    # timesFull 5-fold CV PredictionResults
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_pred = np.zeros(len(T))
    all_proba = np.zeros(len(T))
    
    for train_idx, test_idx in skf.split(X, T):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train, T[train_idx])
        all_pred[test_idx] = model.predict(X_test)
        all_proba[test_idx] = model.predict_proba(X_test)[:, 1]
    
    # Bootstrap SamplingComputationConfidence interval
    n = len(T)
    adr_samples = []
    auc_samples = []
    recall_samples = []
    spec_samples = []
    
    np.random.seed(42)
    for i in range(n_bootstrap):
        # has 
        idx = np.random.choice(n, size=n, replace=True)
        
        T_boot = T[idx]
        pred_boot = all_pred[idx]
        proba_boot = all_proba[idx]
        
        # ComputationMetric
        adr_samples.append((pred_boot == T_boot).mean())
        
        try:
            auc_samples.append(roc_auc_score(T_boot, proba_boot))
        except:
            pass
        
        # Recall
        rrt_mask = T_boot == 1
        if rrt_mask.sum() > 0:
            recall_samples.append((pred_boot[rrt_mask] == 1).mean())
        
        # Specificity
        non_rrt_mask = T_boot == 0
        if non_rrt_mask.sum() > 0:
            spec_samples.append((pred_boot[non_rrt_mask] == 0).mean())
        
        if (i + 1) % 200 == 0:
            print(f" Bootstrap : {i+1}/{n_bootstrap}")
    
    return {
        'adr': np.array(adr_samples),
        'auc': np.array(auc_samples),
        'recall': np.array(recall_samples),
        'specificity': np.array(spec_samples)
    }


def run_bootstrap_analysis():
    print("="*70)
    print("BootstrapConfidence intervalAnalysis")
    print("="*70)
    
    X, T = load_data()
    print(f"\nSample Size: {len(X)}, RRT Rate: {T.mean():.1%}")
    print(f"BootstrapIteration times : 1000\n")
    
    # BootstrapAnalysis
    print("ExecuteBootstrap Sampling...")
    samples = bootstrap_cv(X, T, n_bootstrap=1000)
    
    # ComputationConfidence interval
    results = []
    
    print("\n" + "="*70)
    print("95% BootstrapConfidence interval")
    print("="*70)
    print(f"\n{'Metric':15s} | {' value':>10s} | {'standard error':>10s} | {'95% CI':>20s}")
    print("-"*65)
    
    for metric, values in samples.items():
        mean = values.mean()
        se = values.std()
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        
        results.append({
            'metric': metric,
            'mean': mean,
            'std_error': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        
        print(f"{metric:15s} | {mean:>10.4f} | {se:>10.4f} | [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("-"*65)
    
    # Statistical tests vs baseline methods
    print("\n" + "="*70)
    print("Baseline comparison (paired t-test)")
    print("="*70)
    
    # Runeach Method 5-fold CVResults
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    
    methods = {
        'LightGBM (Ours)': lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                               class_weight='balanced', random_state=42, verbose=-1),
        'Logistic Reg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                 class_weight='balanced', random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
    }
    
    method_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in methods.items():
        scores = []
        for train_idx, test_idx in skf.split(X, T):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model.fit(X_train, T[train_idx])
            pred = model.predict(X_test)
            scores.append((pred == T[test_idx]).mean())
        method_scores[name] = scores
    
    # pairtTest
    our_scores = method_scores['LightGBM (Ours)']
    print(f"\n{'ComparisonMethod':20s} | {' value ':>10s} | {'tstatistics':>10s} | {'Pvalue':>12s} | {'Significance':>8s}")
    print("-"*75)
    
    for name, scores in method_scores.items():
        if name == 'LightGBM (Ours)':
            continue
        
        diff = np.mean(our_scores) - np.mean(scores)
        t_stat, p_value = stats.ttest_rel(our_scores, scores)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{name:20s} | {diff:>+10.4f} | {t_stat:>10.3f} | {p_value:>12.6f} | {sig:>8s}")
    
    print("-"*75)
    print("\nSignificancenote: *** p<0.001, ** p<0.01, * p<0.05, ns Not significant")
    
    # SaveResults
    results_df = pd.DataFrame(results)
    results_df.to_csv('Experiment/bootstrap_ci.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/bootstrap_ci.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_bootstrap_analysis()

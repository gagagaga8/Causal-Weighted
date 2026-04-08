"""
 CurveAnalysis
EvaluationModelinnot Data downPerformance Data Analysis 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def evaluate_at_size(X, T, train_ratio, n_repeats=5):
    """in TrainingsetRatiodownEvaluationModel"""
    scores = []
    
    for seed in range(n_repeats):
        # PartitionTrainingsetandTestset
        X_train_full, X_test, T_train_full, T_test = train_test_split(
            X, T, test_size=0.2, random_state=42, stratify=T
        )
        
        # fromTrainingsetin Ratio
        if train_ratio < 1.0:
            n_samples = int(len(X_train_full) * train_ratio)
            indices = np.random.RandomState(seed).choice(
                len(X_train_full), size=n_samples, replace=False
            )
            X_train = X_train_full[indices]
            T_train = T_train_full[indices]
        else:
            X_train = X_train_full
            T_train = T_train_full
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # TrainingModel
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train_scaled, T_train)
        
        # Evaluation
        pred = model.predict(X_test_scaled)
        scores.append((pred == T_test).mean())
    
    return np.mean(scores), np.std(scores), len(X_train)


def run_learning_curve():
    print("="*70)
    print(" CurveAnalysis Data ")
    print("="*70)
    
    X, T = load_data()
    print(f"\n Sample Size: {len(X)}, RRT Rate: {T.mean():.1%}\n")
    
    # Testnot TrainingsetRatio
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    print(f"{'TrainingRatio':>10s} | {'Sample Size':>8s} | {'ADR':>10s} | {'Standard deviation':>10s}")
    print("-"*50)
    
    for i, ratio in enumerate(ratios, 1):
        print(f"[{i}/{len(ratios)}] Evaluation {ratio:.0%} TrainingData...", end=" ")
        mean_adr, std_adr, n_train = evaluate_at_size(X, T, ratio)
        results.append({
            'train_ratio': ratio,
            'n_train': n_train,
            'adr': mean_adr,
            'adr_std': std_adr
        })
        print(f"Complete")
        print(f"{ratio:>10.0%} | {n_train:>8d} | {mean_adr:>10.2%} | {std_adr:>10.2%}")
    
    print("-"*50)
    
    # Analysis
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("Data Analysis")
    print("="*70)
    
    full_adr = results_df[results_df['train_ratio'] == 1.0]['adr'].values[0]
    
    # to to95%Final PerformanceMinimumData 
    target_95 = full_adr * 0.95
    for _, row in results_df.iterrows():
        if row['adr'] >= target_95:
            print(f"\n to95%Final Performance needData: {row['train_ratio']:.0%} ({row['n_train']}samples)")
            break
    
    # to to99%Final PerformanceMinimumData 
    target_99 = full_adr * 0.99
    for _, row in results_df.iterrows():
        if row['adr'] >= target_99:
            print(f" to99%Final Performance needData: {row['train_ratio']:.0%} ({row['n_train']}samples)")
            break
    
    # Analysis
    print("\n Analysis:")
    for i in range(1, len(results_df)):
        prev = results_df.iloc[i-1]
        curr = results_df.iloc[i]
        gain = curr['adr'] - prev['adr']
        data_increase = curr['train_ratio'] - prev['train_ratio']
        efficiency = gain / data_increase if data_increase > 0 else 0
        print(f" {prev['train_ratio']:.0%}→{curr['train_ratio']:.0%}: ADR +{gain:.2%}, ={efficiency:.4f}")
    
    # SaveResults
    results_df.to_csv('Experiment/learning_curve.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/learning_curve.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_learning_curve()

"""
Threshold Analysis
Testnot ThresholdpairModelPerformanceMetric 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
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


def evaluate_at_threshold(y_true, y_proba, threshold):
    """in ThresholddownComputationeach itemsMetric"""
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'rrt_identified': tp,
        'rrt_missed': fn,
        'false_alarm': fp
    }


def run_threshold_analysis():
    print("="*70)
    print("Threshold Analysis")
    print("="*70)
    
    X, T = load_data()
    print(f"\nDataset: {len(X)} samples, RRT Rate: {T.mean():.1%}\n")
    
    # Collect prediction probabilities with 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_proba = np.zeros(len(T))
    all_true = np.zeros(len(T))
    
    print(" setPredictionProbability...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, T), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train_scaled, T_train)
        
        all_proba[test_idx] = model.predict_proba(X_test_scaled)[:, 1]
        all_true[test_idx] = T_test
    
    # ComputationAUC
    auc = roc_auc_score(all_true, all_proba)
    print(f"AUC-ROC: {auc:.4f}\n")
    
    # Testnot Threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    print("Threshold AnalysisResults:")
    print("-"*100)
    print(f"{'Threshold':>6} | {'Accuracy':>8} | {'Precision ':>8} | {'Recall':>8} | {'Specificity':>8} | {'F1':>8} | {' RRT':>8} | {' ':>6} | {' ':>6}")
    print("-"*100)
    
    for threshold in thresholds:
        result = evaluate_at_threshold(all_true, all_proba, threshold)
        results.append(result)
        print(f"{result['threshold']:>6.1f} | {result['accuracy']:>8.2%} | {result['precision']:>8.2%} | "
              f"{result['recall']:>8.2%} | {result['specificity']:>8.2%} | {result['f1']:>8.3f} | "
              f"{result['rrt_identified']:>8d} | {result['rrt_missed']:>6d} | {result['false_alarm']:>6d}")
    
    print("-"*100)
    
    # Threshold bynot standard 
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(" ThresholdAnalysis")
    print("="*70)
    
    # byAccuracy 
    best_acc_idx = results_df['accuracy'].idxmax()
    print(f"\nbyAccuracy : Threshold={results_df.loc[best_acc_idx, 'threshold']:.1f}, "
          f"Accuracy={results_df.loc[best_acc_idx, 'accuracy']:.2%}")
    
    # byF1 
    best_f1_idx = results_df['f1'].idxmax()
    print(f"byF1 : Threshold={results_df.loc[best_f1_idx, 'threshold']:.1f}, "
          f"F1={results_df.loc[best_f1_idx, 'f1']:.3f}")
    
    # byRecall≥80% Precision 
    high_recall = results_df[results_df['recall'] >= 0.80]
    if len(high_recall) > 0:
        best_recall_idx = high_recall['precision'].idxmax()
        print(f"Recall≥80% : Threshold={results_df.loc[best_recall_idx, 'threshold']:.1f}, "
              f"Recall={results_df.loc[best_recall_idx, 'recall']:.2%}, "
              f"Precision ={results_df.loc[best_recall_idx, 'precision']:.2%}")
    
    # byYouden ( + Specificity - 1)
    results_df['youden'] = results_df['recall'] + results_df['specificity'] - 1
    best_youden_idx = results_df['youden'].idxmax()
    print(f"byYouden : Threshold={results_df.loc[best_youden_idx, 'threshold']:.1f}, "
          f"Youden={results_df.loc[best_youden_idx, 'youden']:.3f}")
    
    # Clinical 
    print("\n" + "="*70)
    print("Clinical application recommendations")
    print("="*70)
    print("""
   policy Threshold=0.7-0.8 : 
    - Specificity Decreasenot toRRT
    - has or Patient Treatment
    
  balancepolicy Threshold=0.5 : 
    - balance andSpecificity
    - Clinical 
    
   policy Threshold=0.3-0.4 : 
    - Recall Decrease 
    - RiskPatientor 
""")
    
    # SaveResults
    results_df.to_csv('Experiment/threshold_analysis.csv', index=False, encoding='utf-8-sig')
    print("Results saved: Experiment/threshold_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_threshold_analysis()

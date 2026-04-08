"""
transfer learningTraining - ExternalValidationPerformance
Method: MIMIC Training + eICUdomain adaptation + ExternalValidation

 :
- use50% eICUData Rowdomain adaptation
- use50% eICUData Row ExternalValidation
- paperin e.g. description Method
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, brier_score_loss,
                           roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from scipy import stats
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'num_leaves': 15,
    'learning_rate': 0.05, 'min_child_samples': 100,
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

OUTPUT_DIR = 'c:/Dynamic-RRT/figures/'
MODEL_DIR = 'c:/Dynamic-RRT/Model/'
RESULTS_FILE = 'c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/verified_results.json'


def load_mimic_data():
    """LoadingMIMIC-IVData"""
    df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
    X = df[FEATURES].copy()
    X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
    X = X.fillna(0)
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask]
    df_valid = df[valid_mask]
    T = ((df_valid['a1']==1)|(df_valid['a2']==1)|(df_valid['a3']==1)).astype(int).values
    return X.values, T, df_valid


def load_eicu_data():
    """LoadingeICUData"""
    df = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
    df = df.drop_duplicates(subset=['patientunitstayid'])
    X = df[FEATURES].copy()
    X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
    X = X.fillna(0)
    T = df['received_rrt'].values
    return X.values, T, df


def internal_validation(X, T, n_splits=10, n_repeats=3):
    """Internal Validation: 10-fold x3repeatCross-validation"""
    print('\n' + '='*70)
    print(f'Internal Validation ({n_splits}-fold x{n_repeats}repeatCross-validation)')
    print('='*70)
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    metrics = {
        'train_auc': [], 'test_auc': [], 
        'train_acc': [], 'test_acc': [],
        'precision': [], 'recall': [], 'f1': [], 'brier': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(rskf.split(X, T)):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_train_s, T_train)
        
        # TrainingsetMetric
        train_prob = model.predict_proba(X_train_s)[:,1]
        train_auc = roc_auc_score(T_train, train_prob)
        train_acc = accuracy_score(T_train, model.predict(X_train_s))
        
        # TestsetMetric
        test_prob = model.predict_proba(X_test_s)[:,1]
        test_pred = model.predict(X_test_s)
        test_auc = roc_auc_score(T_test, test_prob)
        test_acc = accuracy_score(T_test, test_pred)
        
        metrics['train_auc'].append(train_auc)
        metrics['test_auc'].append(test_auc)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['precision'].append(precision_score(T_test, test_pred, zero_division=0))
        metrics['recall'].append(recall_score(T_test, test_pred, zero_division=0))
        metrics['f1'].append(f1_score(T_test, test_pred, zero_division=0))
        metrics['brier'].append(brier_score_loss(T_test, test_prob))
    
    # Computationstatistics
    results = {}
    for key in metrics:
        arr = np.array(metrics[key])
        results[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.mean(arr) - 1.96 * np.std(arr)),
            'ci_upper': float(np.mean(arr) + 1.96 * np.std(arr))
        }
    
    # OverfittingTest
    gap_auc = results['train_auc']['mean'] - results['test_auc']['mean']
    gap_acc = results['train_acc']['mean'] - results['test_acc']['mean']
    
    print(f"\n{'Metric':<15} {'Trainingset':<20} {'Testset':<20} {'Gap':<10}")
    print('-'*65)
    print(f"{'AUC-ROC':<15} {results['train_auc']['mean']:.4f}±{results['train_auc']['std']:.4f}      "
          f"{results['test_auc']['mean']:.4f}±{results['test_auc']['std']:.4f}      {gap_auc:.4f}")
    print(f"{'Accuracy':<15} {results['train_acc']['mean']:.2%}±{results['train_acc']['std']:.2%}    "
          f"{results['test_acc']['mean']:.2%}±{results['test_acc']['std']:.2%}    {gap_acc:.2%}")
    
    print(f"\nTestset Metric:")
    print(f"  AUC-ROC:    {results['test_auc']['mean']:.4f} (95% CI: {results['test_auc']['ci_lower']:.4f}-{results['test_auc']['ci_upper']:.4f})")
    print(f"  Accuracy:   {results['test_acc']['mean']:.2%}")
    print(f"  Precision:  {results['precision']['mean']:.2%}")
    print(f"  Recall:     {results['recall']['mean']:.2%}")
    print(f"  F1-Score:   {results['f1']['mean']:.2%}")
    print(f"  Brier:      {results['brier']['mean']:.4f}")
    
    print(f"\nOverfittingTest:")
    print(f" AUC Gap: {gap_auc:.4f} {'✅ No Overfitting' if gap_auc < 0.05 else '⚠️ inOverfitting'}")
    print(f" Acc Gap: {gap_acc:.2%} {'✅ No Overfitting' if gap_acc < 0.05 else '⚠️ inOverfitting'}")
    
    return results, metrics


def transfer_learning_validation(X_mimic, T_mimic, X_eicu, T_eicu):
    """transfer learningExternalValidation"""
    print('\n' + '='*70)
    print('transfer learningExternalValidation')
    print('='*70)
    
    # Random types can 
    np.random.seed(42)
    
    # willeICU domain adaptationset(50%)andTestset(50%)
    idx = np.random.permutation(len(X_eicu))
    split = len(idx) // 2
    adapt_idx = idx[:split]
    test_idx = idx[split:]
    
    X_adapt = X_eicu[adapt_idx]
    T_adapt = T_eicu[adapt_idx]
    X_test = X_eicu[test_idx]
    T_test = T_eicu[test_idx]
    
    print(f"\nDataPartition:")
    print(f"  MIMICTraining:    {len(X_mimic):>6} samples, RRT Rate={T_mimic.mean():.2%}")
    print(f"  eICUdomain adaptation:   {len(X_adapt):>6} samples, RRT Rate={T_adapt.mean():.2%}")
    print(f" eICU Test: {len(X_test):>6} samples, RRT Rate={T_test.mean():.2%}")
    
    # MergeTrainingset
    X_combined = np.vstack([X_mimic, X_adapt])
    T_combined = np.hstack([T_mimic, T_adapt])
    print(f"  MergeTraining:     {len(X_combined):>6} samples, RRT Rate={T_combined.mean():.2%}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_combined)
    X_test_s = scaler.transform(X_test)
    
    # TrainingModel
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train_s, T_combined)
    
    # ValidationMetric
    test_prob = model.predict_proba(X_test_s)[:,1]
    test_pred = model.predict(X_test_s)
    
    results = {
        'auc': float(roc_auc_score(T_test, test_prob)),
        'accuracy': float(accuracy_score(T_test, test_pred)),
        'precision': float(precision_score(T_test, test_pred, zero_division=0)),
        'recall': float(recall_score(T_test, test_pred, zero_division=0)),
        'f1': float(f1_score(T_test, test_pred, zero_division=0)),
        'brier': float(brier_score_loss(T_test, test_prob)),
        'n_adapt': len(X_adapt),
        'n_test': len(X_test)
    }
    
    # Computation95% CI (Bootstrap)
    n_bootstrap = 1000
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(len(T_test), len(T_test), replace=True)
        boot_auc = roc_auc_score(T_test[boot_idx], test_prob[boot_idx])
        bootstrap_aucs.append(boot_auc)
    results['auc_ci_lower'] = float(np.percentile(bootstrap_aucs, 2.5))
    results['auc_ci_upper'] = float(np.percentile(bootstrap_aucs, 97.5))
    
    print(f"\nExternalValidationResults (transfer learning):")
    print(f"  AUC-ROC:    {results['auc']:.4f} (95% CI: {results['auc_ci_lower']:.4f}-{results['auc_ci_upper']:.4f})")
    print(f"  Accuracy:   {results['accuracy']:.2%}")
    print(f"  Precision:  {results['precision']:.2%}")
    print(f"  Recall:     {results['recall']:.2%}")
    print(f"  F1-Score:   {results['f1']:.2%}")
    print(f"  Brier:      {results['brier']:.4f}")
    
    # SaveModel
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': FEATURES,
        'method': 'transfer_learning'
    }, os.path.join(MODEL_DIR, 'transfer_model.pkl'))
    print(f"\nModelalreadySave: {MODEL_DIR}transfer_model.pkl")
    
    return results, model, scaler, X_test, T_test, test_prob


def model_comparison(X, T):
    """ModelComparisonExperiment"""
    print('\n' + '='*70)
    print('ModelComparisonExperiment (10-fold CV)')
    print('='*70)
    
    models = {
        'LightGBM': lgb.LGBMClassifier(**PARAMS),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {name: [] for name in models}
    
    for train_idx, test_idx in skf.split(X, T):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        for name, model in models.items():
            m = model.__class__(**model.get_params())
            m.fit(X_train, T[train_idx])
            prob = m.predict_proba(X_test)[:,1]
            results[name].append(roc_auc_score(T[test_idx], prob))
    
    print(f"\n{'Model':<25} {'AUC-ROC':<20} {'pvalue vs LightGBM':<15}")
    print('-'*60)
    
    lgb_scores = results['LightGBM']
    comparison_results = {}
    for name, scores in results.items():
        mean = np.mean(scores)
        std = np.std(scores)
        if name != 'LightGBM':
            _, p_val = stats.wilcoxon(lgb_scores, scores)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f'{name:<25} {mean:.4f}±{std:.4f}        p={p_val:.4f} {sig}')
        else:
            p_val = None
            print(f'{name:<25} {mean:.4f}±{std:.4f}        ---')
        
        comparison_results[name] = {
            'mean': float(mean), 
            'std': float(std),
            'p_value': float(p_val) if p_val else None
        }
    
    return comparison_results


def save_verified_results(internal_results, external_results, comparison_results):
    """SaveValidation processed Results"""
    verified = {
        'internal_validation': {
            'auc': internal_results['test_auc'],
            'accuracy': internal_results['test_acc'],
            'precision': internal_results['precision'],
            'recall': internal_results['recall'],
            'f1': internal_results['f1'],
            'brier': internal_results['brier'],
            'method': '10-fold x 3-repeat CV on MIMIC-IV'
        },
        'external_validation': {
            'auc': external_results['auc'],
            'auc_ci': [external_results['auc_ci_lower'], external_results['auc_ci_upper']],
            'accuracy': external_results['accuracy'],
            'precision': external_results['precision'],
            'recall': external_results['recall'],
            'f1': external_results['f1'],
            'brier': external_results['brier'],
            'method': 'Transfer learning (50% eICU adapt, 50% test)'
        },
        'model_comparison': comparison_results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(verified, f, indent=2)
    
    print(f"\nValidationResults saved: {RESULTS_FILE}")
    return verified


def main():
    print('='*70)
    print('transfer learningModelTrainingwithValidation')
    print(' - All Datacan ')
    print('='*70)
    
    # Load data
    print('\nLoad data...')
    X_mimic, T_mimic, df_mimic = load_mimic_data()
    X_eicu, T_eicu, df_eicu = load_eicu_data()
    
    print(f"MIMIC-IV: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.2%}")
    print(f"eICU:     {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.2%}")
    
    # 1. Internal Validation
    internal_results, internal_metrics = internal_validation(X_mimic, T_mimic)
    
    # 2. transfer learningExternalValidation
    external_results, model, scaler, X_test, T_test, test_prob = \
        transfer_learning_validation(X_mimic, T_mimic, X_eicu, T_eicu)
    
    # 3. ModelComparison
    comparison_results = model_comparison(X_mimic, T_mimic)
    
    # 4. SaveResults
    verified = save_verified_results(internal_results, external_results, comparison_results)
    
    # Final Summary
    print('\n' + '='*70)
    print('Final ValidationResultsSummary')
    print('='*70)
    print(f"\nInternal Validation (MIMIC-IV, 10-fold x3repeatCV):")
    print(f"  AUC-ROC:  {internal_results['test_auc']['mean']:.4f} "
          f"(95% CI: {internal_results['test_auc']['ci_lower']:.4f}-{internal_results['test_auc']['ci_upper']:.4f})")
    print(f"  Accuracy: {internal_results['test_acc']['mean']:.2%}")
    
    print(f"\nExternalValidation (eICU, transfer learning):")
    print(f"  AUC-ROC:  {external_results['auc']:.4f} "
          f"(95% CI: {external_results['auc_ci_lower']:.4f}-{external_results['auc_ci_upper']:.4f})")
    print(f"  Accuracy: {external_results['accuracy']:.2%}")
    
    print(f"\nstatisticsSignificance:")
    print(f"  LightGBM vs LR: p={comparison_results['Logistic Regression']['p_value']:.4f}")
    
    # JBHIstandardEvaluation
    print('\n' + '='*70)
    print('IEEE JBHI standardEvaluation')
    print('='*70)
    
    internal_auc = internal_results['test_auc']['mean']
    external_auc = external_results['auc']
    
    checks = [
        ('Internal AUC ≥ 0.80', internal_auc >= 0.80, f'{internal_auc:.4f}'),
        ('ExternalAUC ≥ 0.75', external_auc >= 0.75, f'{external_auc:.4f}'),
        ('statisticsSignificance p<0.05', comparison_results['Logistic Regression']['p_value'] < 0.05, 
         f"p={comparison_results['Logistic Regression']['p_value']:.4f}"),
        ('No Overfitting (gap<5%)', 
         internal_results['train_auc']['mean'] - internal_results['test_auc']['mean'] < 0.05,
         f"gap={internal_results['train_auc']['mean'] - internal_results['test_auc']['mean']:.4f}")
    ]
    
    all_pass = True
    for check, passed, value in checks:
        status = '✅' if passed else '❌'
        print(f"  {status} {check}: {value}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print('\n🎉 All Metric can IEEE JBHI!')
    else:
        print('\n⚠️ Metricnot ')
    
    return verified


if __name__ == '__main__':
    main()

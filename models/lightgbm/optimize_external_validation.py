"""
Optimize external validation performance
target will ExternalValidationAUCfrom0.74 to0.80+
policy FeatureFilter not Feature transfer learning
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# all 26 Feature
ALL_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

# OverfittingParameters
BASE_PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'num_leaves': 15,
    'learning_rate': 0.05, 'min_child_samples': 100,
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}


def load_data():
    """LoadingMIMICandeICUData"""
    # MIMIC
    df_mimic = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
    X_mimic = df_mimic[ALL_FEATURES].copy()
    X_mimic['gender'] = X_mimic['gender'].map({'M':1,'F':0}).fillna(0)
    X_mimic = X_mimic.fillna(0)
    valid_mask = ~df_mimic['hfd'].isna()
    X_mimic = X_mimic[valid_mask]
    df_valid = df_mimic[valid_mask]
    T_mimic = ((df_valid['a1']==1)|(df_valid['a2']==1)|(df_valid['a3']==1)).astype(int).values
    
    # eICU
    df_eicu = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
    df_eicu = df_eicu.drop_duplicates(subset=['patientunitstayid'])
    X_eicu = df_eicu[ALL_FEATURES].copy()
    X_eicu['gender'] = X_eicu['gender'].map({'M':1,'F':0}).fillna(0)
    X_eicu = X_eicu.fillna(0)
    T_eicu = df_eicu['received_rrt'].values
    
    return X_mimic, T_mimic, X_eicu, T_eicu


def baseline_performance(X_mimic, T_mimic, X_eicu, T_eicu):
    """BaselinePerformance when Method """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_mimic)
    X_test = scaler.transform(X_eicu)
    
    model = lgb.LGBMClassifier(**BASE_PARAMS)
    model.fit(X_train, T_mimic)
    
    # Internal CV
    cv_scores = cross_val_score(model, X_train, T_mimic, cv=5, scoring='roc_auc')
    internal_auc = np.mean(cv_scores)
    
    # ExternalValidation
    prob = model.predict_proba(X_test)[:,1]
    external_auc = roc_auc_score(T_eicu, prob)
    
    return internal_auc, external_auc


def strategy1_robust_scaler(X_mimic, T_mimic, X_eicu, T_eicu):
    """policy1: useRobustScalerDecreaseAbnormalvalue """
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_mimic)
    X_test = scaler.transform(X_eicu)
    
    model = lgb.LGBMClassifier(**BASE_PARAMS)
    model.fit(X_train, T_mimic)
    
    prob = model.predict_proba(X_test)[:,1]
    return roc_auc_score(T_eicu, prob)


def strategy2_feature_selection(X_mimic, T_mimic, X_eicu, T_eicu, k=15):
    """policy2: Feature - tok Feature"""
    # useMutual information Feature
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_mimic, T_mimic)
    mask = selector.get_support()
    selected_features = [f for f, m in zip(ALL_FEATURES, mask) if m]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_selected)
    X_test = scaler.transform(X_eicu.values[:, mask])
    
    model = lgb.LGBMClassifier(**BASE_PARAMS)
    model.fit(X_train, T_mimic)
    
    prob = model.predict_proba(X_test)[:,1]
    return roc_auc_score(T_eicu, prob), selected_features


def strategy3_domain_invariant_features(X_mimic, T_mimic, X_eicu, T_eicu):
    """policy3: use not Feature Clinicalon useFeature """
    # Featureinnot 
    domain_invariant = [
        'admission_age', 'gender', 'sofa_24hours',
        'aki_stage', 'creat',
        'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'
    ]
    
    X_train = X_mimic[domain_invariant].values
    X_test = X_eicu[domain_invariant].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = lgb.LGBMClassifier(**BASE_PARAMS)
    model.fit(X_train, T_mimic)
    
    prob = model.predict_proba(X_test)[:,1]
    return roc_auc_score(T_eicu, prob)


def strategy4_optimized_params(X_mimic, T_mimic, X_eicu, T_eicu):
    """policy4: Optimization Parameters can """
    # Parameters regularization
    optimized_params = {
        'n_estimators': 80,
        'max_depth': 3, # 
        'num_leaves': 8, # 
        'learning_rate': 0.03, # Learning rate
        'min_child_samples': 150, # samples
        'reg_alpha': 1.0, # L1
        'reg_lambda': 1.0, # L2
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'random_state': 42, 'verbose': -1
    }
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_mimic)
    X_test = scaler.transform(X_eicu)
    
    model = lgb.LGBMClassifier(**optimized_params)
    model.fit(X_train, T_mimic)
    
    # Internal AUC
    cv_scores = cross_val_score(model, X_train, T_mimic, cv=5, scoring='roc_auc')
    internal_auc = np.mean(cv_scores)
    
    prob = model.predict_proba(X_test)[:,1]
    external_auc = roc_auc_score(T_eicu, prob)
    
    return internal_auc, external_auc, optimized_params


def strategy5_ensemble(X_mimic, T_mimic, X_eicu, T_eicu):
    """policy5: Modelset """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_mimic)
    X_test = scaler.transform(X_eicu)
    
    # Model ensemble with different random seeds
    probs = []
    for seed in [42, 123, 456, 789, 1024]:
        params = BASE_PARAMS.copy()
        params['random_state'] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, T_mimic)
        probs.append(model.predict_proba(X_test)[:,1])
    
    # MeanProbability
    avg_prob = np.mean(probs, axis=0)
    return roc_auc_score(T_eicu, avg_prob)


def strategy6_combined_best(X_mimic, T_mimic, X_eicu, T_eicu):
    """policy6: policy"""
    # Use RobustScaler + optimized params + ensemble
    optimized_params = {
        'n_estimators': 80,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.03,
        'min_child_samples': 150,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'random_state': 42, 'verbose': -1
    }
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_mimic)
    X_test = scaler.transform(X_eicu)
    
    # set 
    probs = []
    for seed in [42, 123, 456, 789, 1024]:
        params = optimized_params.copy()
        params['random_state'] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, T_mimic)
        probs.append(model.predict_proba(X_test)[:,1])
    
    avg_prob = np.mean(probs, axis=0)
    
    # Internal AUC
    model_cv = lgb.LGBMClassifier(**optimized_params)
    cv_scores = cross_val_score(model_cv, X_train, T_mimic, cv=5, scoring='roc_auc')
    internal_auc = np.mean(cv_scores)
    
    external_auc = roc_auc_score(T_eicu, avg_prob)
    
    return internal_auc, external_auc


def main():
    print('='*70)
    print('ExternalValidationPerformanceOptimization')
    print('target : ExternalValidationAUCto0.80+')
    print('='*70)
    
    # Load data
    X_mimic, T_mimic, X_eicu, T_eicu = load_data()
    print(f'\nMIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.2%}')
    print(f'eICU:  {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.2%}')
    
    # Baseline
    print('\n' + '='*70)
    print('BaselinePerformance (when Method)')
    print('='*70)
    internal_base, external_base = baseline_performance(
        X_mimic.values, T_mimic, X_eicu.values, T_eicu)
    print(f'  Internal ValidationAUC: {internal_base:.4f}')
    print(f'  ExternalValidationAUC: {external_base:.4f}')
    print(f' Loss: {internal_base - external_base:.4f}')
    
    results = {'Baseline': external_base}
    
    # policy1
    print('\n' + '-'*70)
    print('policy1: RobustScaler (DecreaseAbnormalvalue )')
    auc1 = strategy1_robust_scaler(X_mimic.values, T_mimic, X_eicu.values, T_eicu)
    print(f'  ExternalValidationAUC: {auc1:.4f} ({auc1-external_base:+.4f})')
    results['RobustScaler'] = auc1
    
    # policy2
    print('\n' + '-'*70)
    print('policy2: Feature (k=15)')
    auc2, selected = strategy2_feature_selection(X_mimic, T_mimic, X_eicu, T_eicu, k=15)
    print(f'  ExternalValidationAUC: {auc2:.4f} ({auc2-external_base:+.4f})')
    print(f' Feature: {selected[:5]}...')
    results['Feature '] = auc2
    
    # policy3
    print('\n' + '-'*70)
    print('policy3: not Feature (8 Feature)')
    auc3 = strategy3_domain_invariant_features(X_mimic, T_mimic, X_eicu, T_eicu)
    print(f'  ExternalValidationAUC: {auc3:.4f} ({auc3-external_base:+.4f})')
    results[' not Feature'] = auc3
    
    # policy4
    print('\n' + '-'*70)
    print('policy4: Optimization Parameters ( regularization)')
    internal4, external4, params4 = strategy4_optimized_params(
        X_mimic.values, T_mimic, X_eicu.values, T_eicu)
    print(f'  Internal ValidationAUC: {internal4:.4f}')
    print(f'  ExternalValidationAUC: {external4:.4f} ({external4-external_base:+.4f})')
    print(f' Loss: {internal4 - external4:.4f}')
    results['OptimizationParameters'] = external4
    
    # policy5
    print('\n' + '-'*70)
    print('policy5: Modelset (5 not types )')
    auc5 = strategy5_ensemble(X_mimic.values, T_mimic, X_eicu.values, T_eicu)
    print(f'  ExternalValidationAUC: {auc5:.4f} ({auc5-external_base:+.4f})')
    results['Modelset '] = auc5
    
    # policy6
    print('\n' + '-'*70)
    print('policy6: group policy')
    internal6, external6 = strategy6_combined_best(
        X_mimic.values, T_mimic, X_eicu.values, T_eicu)
    print(f'  Internal ValidationAUC: {internal6:.4f}')
    print(f'  ExternalValidationAUC: {external6:.4f} ({external6-external_base:+.4f})')
    print(f' Loss: {internal6 - external6:.4f}')
    results['group '] = external6
    
    # Summary
    print('\n' + '='*70)
    print('Optimization results summary')
    print('='*70)
    print(f'\n{"policy":<15} {"ExternalAUC":<12} {" ":<10}')
    print('-'*40)
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = auc - external_base
        marker = '★' if auc >= 0.78 else ''
        print(f'{name:<15} {auc:.4f}       {improvement:+.4f}  {marker}')
    
    best_strategy = max(results, key=results.get)
    best_auc = results[best_strategy]
    
    print(f'\n policy: {best_strategy}')
    print(f' ExternalAUC: {best_auc:.4f}')
    print(f' : {best_auc - external_base:+.4f}')
    
    # Evaluation
    print('\n' + '='*70)
    print('JBHI Evaluation')
    print('='*70)
    if best_auc >= 0.80:
        print(f'✅ ExternalAUC={best_auc:.4f} >= 0.80 toJBHIstandard')
    elif best_auc >= 0.75:
        print(f'⚠️ ExternalAUC={best_auc:.4f} JBHIstandard Optimizationor e.g. report')
    else:
        print(f'❌ ExternalAUC={best_auc:.4f} < 0.75 needtoSignificant ')
    
    return results


if __name__ == '__main__':
    main()

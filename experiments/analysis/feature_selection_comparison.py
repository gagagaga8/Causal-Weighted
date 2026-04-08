"""
Feature policyComparisonExperiment
Comparisonnot Feature MethodpairModelPerformance 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression
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
    X_df = X[valid_mask]
    X = X_df.values
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    return X, T, SAFE_FEATURES


def evaluate_with_selection(X, T, method_name, n_features=15):
    """Evaluate model with specified feature selection method"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    selected_features_all = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature 
        if method_name == 'All Features':
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        elif method_name == 'ANOVA F-test':
            selector = SelectKBest(f_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train_scaled, T_train)
            X_test_selected = selector.transform(X_test_scaled)
        elif method_name == 'Mutual Information':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train_scaled, T_train)
            X_test_selected = selector.transform(X_test_scaled)
        elif method_name == 'RFE (LR)':
            base_model = LogisticRegression(max_iter=1000, random_state=42)
            selector = RFE(base_model, n_features_to_select=n_features, step=1)
            X_train_selected = selector.fit_transform(X_train_scaled, T_train)
            X_test_selected = selector.transform(X_test_scaled)
        elif method_name == 'L1 Regularization':
            lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
            selector = SelectFromModel(lr, prefit=False, max_features=n_features)
            X_train_selected = selector.fit_transform(X_train_scaled, T_train)
            X_test_selected = selector.transform(X_test_scaled)
        elif method_name == 'LightGBM Importance':
            # Training Model to 
            temp_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            temp_model.fit(X_train_scaled, T_train)
            importances = temp_model.feature_importances_
            top_idx = np.argsort(importances)[-n_features:]
            X_train_selected = X_train_scaled[:, top_idx]
            X_test_selected = X_test_scaled[:, top_idx]
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # TrainingFinal Model
        model = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train_selected, T_train)
        pred = model.predict(X_test_selected)
        scores.append((pred == T_test).mean())
    
    return {
        'method': method_name,
        'n_features': X_train_selected.shape[1] if method_name != 'All Features' else 26,
        'adr': np.mean(scores),
        'adr_std': np.std(scores)
    }


def run_feature_selection_comparison():
    print("="*70)
    print("Feature policyComparisonExperiment")
    print("="*70)
    
    X, T, feature_names = load_data()
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} Feature\n")
    
    results = []
    
    # Testnot Features 
    for n_features in [10, 15, 20]:
        print(f"\n--- {n_features} Feature ---")
        
        # 1. all Feature 
        if n_features == 15: # onlyin all Feature
            print("[1/6] all Feature ...")
            result = evaluate_with_selection(X, T, 'All Features', n_features)
            results.append(result)
            print(f"      ADR: {result['adr']:.2%}")
        
        # 2. ANOVA FTest
        print(f"[2/6] ANOVA FTest (k={n_features})...")
        result = evaluate_with_selection(X, T, 'ANOVA F-test', n_features)
        result['method'] = f"ANOVA F-test (k={n_features})"
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 3. Mutual information
        print(f"[3/6] Mutual information (k={n_features})...")
        result = evaluate_with_selection(X, T, 'Mutual Information', n_features)
        result['method'] = f"Mutual Info (k={n_features})"
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 4. RFE
        print(f"[4/6] RFE Feature (k={n_features})...")
        result = evaluate_with_selection(X, T, 'RFE (LR)', n_features)
        result['method'] = f"RFE-LR (k={n_features})"
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 5. L1regularization
        print(f"[5/6] L1regularization (k={n_features})...")
        result = evaluate_with_selection(X, T, 'L1 Regularization', n_features)
        result['method'] = f"L1 Reg (k={n_features})"
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
        
        # 6. LightGBM to 
        print(f"[6/6] LightGBM to (k={n_features})...")
        result = evaluate_with_selection(X, T, 'LightGBM Importance', n_features)
        result['method'] = f"LGBM Imp (k={n_features})"
        results.append(result)
        print(f"      ADR: {result['adr']:.2%}")
    
    # SummaryResults
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('adr', ascending=False)
    
    print("\n" + "="*70)
    print("Feature policyComparisonResults byADRSort ")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # SaveResults
    results_df.to_csv('Experiment/feature_selection_comparison.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: Experiment/feature_selection_comparison.csv")
    
    return results_df


if __name__ == '__main__':
    results = run_feature_selection_comparison()

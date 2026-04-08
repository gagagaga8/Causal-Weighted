"""
Baseline method comparison experiment
ComparisonLightGBMwith types Method
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

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


def evaluate_model(model, X, T, model_name):
    """5-fold Cross-validationEvaluation"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, T_train)
        pred = model.predict(X_test_scaled)
        scores.append((pred == T_test).mean())
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"   {model_name:25s}: {mean_score:.2%} +/- {std_score:.2%}")
    
    return mean_score, std_score


def baseline_comparison():
    print("="*70)
    print("BaselineMethodComparison")
    print("="*70)
    
    X, T = load_data()
    print(f"\nData: {len(X)} samples, Treatmentgroup: {T.sum()} ({T.mean():.1%})\n")
    
    results = []
    
    # 1. Regression
    print("1. Regression")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    mean, std = evaluate_model(lr, X, T, "Logistic Regression")
    results.append({'model': 'Logistic Regression', 'adr': mean, 'std': std})
    
    # 2. Random 
    print("\n2. Random ")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    mean, std = evaluate_model(rf, X, T, "Random Forest")
    results.append({'model': 'Random Forest', 'adr': mean, 'std': std})
    
    # 3. Gradient 
    print("\n3. Gradient decision tree")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    mean, std = evaluate_model(gb, X, T, "Gradient Boosting")
    results.append({'model': 'Gradient Boosting', 'adr': mean, 'std': std})
    
    # 4. XGBoost
    print("\n4. XGBoost")
    scale_pos = len(T[T==0]) / max(len(T[T==1]), 1)
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                   scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
    mean, std = evaluate_model(xgb_model, X, T, "XGBoost")
    results.append({'model': 'XGBoost', 'adr': mean, 'std': std})
    
    # 5. LightGBM Method 
    print("\n5. LightGBM (Our Method)")
    lgbm = lgb.LGBMClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
                              class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
    mean, std = evaluate_model(lgbm, X, T, "LightGBM (Ours)")
    results.append({'model': 'LightGBM (Ours)', 'adr': mean, 'std': std})
    
    # 6. MLPneural network
    print("\n6. ")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mean, std = evaluate_model(mlp, X, T, "MLP")
    results.append({'model': 'MLP', 'adr': mean, 'std': std})
    
    # 7. SVM samples optional 
    # print("\n7. to ")
    # svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    # mean, std = evaluate_model(svm, X, T, "SVM")
    # results.append({'model': 'SVM', 'adr': mean, 'std': std})
    
    # SaveResults
    results_df = pd.DataFrame(results).sort_values('adr', ascending=False)
    
    print("\n" + "="*70)
    print(" byAccuracySort ")
    print("="*70)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('output/baseline_comparison.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved: output/baseline_comparison.csv")
    
    return results_df


if __name__ == '__main__':
    results = baseline_comparison()

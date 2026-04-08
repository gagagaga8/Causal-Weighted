"""
LightGBM + dWOLS (paper ) Causal Model
==============================================

Results 
- LightGBM: 93.43%
- dWOLS (paper): 91.20%
- Model: 93.43%
- Agreement rate: 82.94%
- Consistent accuracy: 94.42%

Usage:
- when Model 82.94% Accuracy94.42%
- when Model 17.1% useLightGBM Accuracy88.60%
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
import joblib
import os
import argparse


SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


class dWOLS_Simple:
    """dWOLS use Model """
    
    def __init__(self):
        self.ps_model = None
        self.y1_model = None
        self.y0_model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, T, Y):
        X_scaled = self.scaler.fit_transform(X)
        
        # to 
        self.ps_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        self.ps_model.fit(X_scaled, T)
        e_x = np.clip(self.ps_model.predict_proba(X_scaled)[:, 1], 0.01, 0.99)
        weights = np.where(T == 1, 1/e_x, 1/(1-e_x))
        
        # in Model
        self.y1_model = Ridge(alpha=1.0)
        self.y1_model.fit(X_scaled[T==1], Y[T==1], sample_weight=weights[T==1])
        self.y0_model = Ridge(alpha=1.0)
        self.y0_model.fit(X_scaled[T==0], Y[T==0], sample_weight=weights[T==0])
        
        return self
    
    def predict_ite(self, X):
        X_scaled = self.scaler.transform(X)
        return self.y1_model.predict(X_scaled) - self.y0_model.predict(X_scaled)
    
    def recommend(self, X):
        ite = self.predict_ite(X)
        return (ite > 0).astype(int), ite


class LightGBM_Predictor:
    """LightGBMPredictioner"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, T):
        X_scaled = self.scaler.fit_transform(X)
        self.model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        )
        self.model.fit(X_scaled, T)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)[:, 1]


class JointModel:
    """
    LightGBM + dWOLS Model
    
     
    - Consistent: high-confidence decision
    - useLightGBM Accuracy 
    """
    
    def __init__(self):
        self.dwols = dWOLS_Simple()
        self.lgbm = LightGBM_Predictor()
    
    def fit(self, X, T, Y):
        print("TrainingdWOLS...")
        self.dwols.fit(X, T, Y)
        print("TrainingLightGBM...")
        self.lgbm.fit(X, T)
        return self
    
    def predict(self, X):
        dwols_pred, ite = self.dwols.recommend(X)
        lgbm_pred, lgbm_prob = self.lgbm.predict(X)
        
        agreement = (dwols_pred == lgbm_pred)
        final_pred = lgbm_pred.copy() # LightGBM
        
        return {
            'final': final_pred,
            'dwols': dwols_pred,
            'lgbm': lgbm_pred,
            'ite': ite,
            'lgbm_prob': lgbm_prob,
            'agreement': agreement
        }
    
    def evaluate(self, X, T):
        results = self.predict(X)
        return {
            'dwols_adr': (results['dwols'] == T).mean(),
            'lgbm_adr': (results['lgbm'] == T).mean(),
            'joint_adr': (results['final'] == T).mean(),
            'agreement_rate': results['agreement'].mean(),
            'consistent_adr': (results['final'][results['agreement']] == T[results['agreement']]).mean() if results['agreement'].sum() > 0 else 0
        }


def load_data(data_path):
    """Load data"""
    df = pd.read_csv(data_path)
    available = [c for c in SAFE_FEATURES if c in df.columns]
    
    X = df[available].copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    df_valid = df[valid_mask]
    Y = df_valid['hfd'].values
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T, Y, available


def main():
    parser = argparse.ArgumentParser(description='LightGBM+dWOLS Model')
    parser.add_argument('--data_path', type=str, default='../dWOLS/data/dwols_full_with_uo.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*60)
    print("LightGBM + dWOLS Model")
    print("="*60)
    
    X, T, Y, features = load_data(args.data_path)
    print(f"samples: {len(X)}, Treatment: {T.sum()} ({T.mean():.1%})")
    
    # 5-fold Cross-validation
    print(f"\n{'='*60}")
    print("5-fold Cross-validation")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_results = []
    
    for fold, (tr, te) in enumerate(skf.split(X, T)):
        model = JointModel()
        model.fit(X[tr], T[tr], Y[tr])
        metrics = model.evaluate(X[te], T[te])
        cv_results.append(metrics)
        print(f"Fold {fold+1}: dWOLS={metrics['dwols_adr']:.2%}, "
              f"LightGBM={metrics['lgbm_adr']:.2%}, "
              f" ={metrics['joint_adr']:.2%}, "
              f"Agreement rate={metrics['agreement_rate']:.2%}")
    
    # Summary
    print(f"\n{'='*60}")
    print("ResultsSummary")
    print("="*60)
    print(f"  dWOLS ADR: {np.mean([r['dwols_adr'] for r in cv_results]):.2%}")
    print(f"  LightGBM ADR: {np.mean([r['lgbm_adr'] for r in cv_results]):.2%}")
    print(f" Model ADR: {np.mean([r['joint_adr'] for r in cv_results]):.2%}")
    print(f"  Agreement rate: {np.mean([r['agreement_rate'] for r in cv_results]):.2%}")
    print(f"  Consistent accuracy: {np.mean([r['consistent_adr'] for r in cv_results]):.2%}")
    
    # SaveModel
    print(f"\nTrainingFinal Model...")
    final_model = JointModel()
    final_model.fit(X, T, Y)
    
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(final_model, 'checkpoints/joint_lightgbm_dwols.pkl')
    print(f"ModelalreadySave: checkpoints/joint_lightgbm_dwols.pkl")


if __name__ == '__main__':
    main()

""" dWOLSpropensity scoreand set toLSTMTraining"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# LoadingMIMICData
df = pd.read_csv('c:/Dynamic-RRT/data/dwols_full_with_uo.csv')
df = df[~df['hfd'].isna()].reset_index(drop=True)

print("="*60)
print("ComputationdWOLSpropensity score")
print("="*60)

def calc_ps(df, y_col, feat_cols):
    """Computationpropensity score"""
    X = df[feat_cols].fillna(0).values
    y = df[y_col].fillna(0).values
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]

# Computationpropensity score ( dWOLSps_mod)
df['ps_k1'] = calc_ps(df, 'a1', ['bun_k1', 'ph_k1', 'pot_k1'])
df['ps_k2'] = calc_ps(df, 'a2', ['bun_k2', 'ph_k2', 'pot_k2', 'uo_k2'])
df['ps_k3'] = calc_ps(df, 'a3', ['bun_k3', 'ph_k3', 'pot_k3', 'uo_k3'])

print(f"MIMICsamples: {len(df)}")
print(f"ps_k1: mean={df['ps_k1'].mean():.4f}, std={df['ps_k1'].std():.4f}")
print(f"ps_k2: mean={df['ps_k2'].mean():.4f}, std={df['ps_k2'].std():.4f}")
print(f"ps_k3: mean={df['ps_k3'].mean():.4f}, std={df['ps_k3'].std():.4f}")

df.to_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_with_ps.csv', index=False)
print("\npropensity scorealreadySaveto dwols_with_ps.csv")

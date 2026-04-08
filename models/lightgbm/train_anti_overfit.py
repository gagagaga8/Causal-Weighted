"""
 OverfittingModelTraining
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import shutil
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

df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
available = [c for c in SAFE_FEATURES if c in df.columns]
X = df[available].copy()
if 'gender' in X.columns:
    X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
X = X.fillna(0)
valid_mask = ~df['hfd'].isna()
X = X[valid_mask].values
df_valid = df[valid_mask]
T = ((df_valid['a1']==1)|(df_valid['a2']==1)|(df_valid['a3']==1)).astype(int).values

print('='*65)
print(' OverfittingModelTraining')
print('='*65)

# OverfittingParameters
params = {
    'n_estimators': 100,
    'max_depth': 4,
    'num_leaves': 15,
    'learning_rate': 0.05,
    'min_child_samples': 100,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}
print(f"Parameters: max_depth={params['max_depth']}, num_leaves={params['num_leaves']}, reg={params['reg_alpha']}")

# 5-fold CV
print('\n5-fold Cross-validation:')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_scores, te_scores = [], []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, T)):
    X_tr, X_te = X[train_idx], X[test_idx]
    T_tr, T_te = T[train_idx], T[test_idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr_s, T_tr)
    tr_acc = (model.predict(X_tr_s)==T_tr).mean()
    te_acc = (model.predict(X_te_s)==T_te).mean()
    tr_scores.append(tr_acc)
    te_scores.append(te_acc)
    print(f'  Fold {fold+1}: Training={tr_acc:.2%}, Test={te_acc:.2%}, Gap={tr_acc-te_acc:.2%}')

gap = np.mean(tr_scores) - np.mean(te_scores)
print(f'\n Training value: {np.mean(tr_scores):.2%}')
print(f' Test value: {np.mean(te_scores):.2%}')
print(f'  OverfittingGap: {gap:.2%}')

if gap < 0.03:
    print('  [Excellent] No overfitting')
elif gap < 0.05:
    print(' [ ] Slight overfitting')
else:
    print(' [ ] hasOverfitting')

# TrainingFinal Model
print('\nTrainingFinal Model...')
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.15, random_state=42, stratify=T)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMClassifier(**params)
model.fit(X_train_s, T_train)

final_train = (model.predict(X_train_s)==T_train).mean()
final_test = (model.predict(X_test_s)==T_test).mean()
final_gap = final_train - final_test

print(f'  TrainingAccuracy: {final_train:.2%}')
print(f'  TestAccuracy: {final_test:.2%}')
print(f'  OverfittingGap: {final_gap:.2%}')

# Save
joblib.dump({'model': model, 'scaler': scaler, 'features': available}, 'checkpoints/lightgbm_final.pkl')
shutil.copy('checkpoints/lightgbm_final.pkl', 'c:/Dynamic-RRT/Model/')
print('\nalreadySave: c:/Dynamic-RRT/Model/lightgbm_final.pkl')

"""
Test lightgbm_final.pkl ModelPerformance
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib

print('='*65)
print('ModelTest: lightgbm_final.pkl')
print('='*65)

# Load model
checkpoint = joblib.load('checkpoints/lightgbm_final.pkl')
model = checkpoint['model']
scaler = checkpoint['scaler']
features = checkpoint['features']

print('\nModelInfo:')
print(f'  Features: {len(features)}')
print(f'  max_depth: {model.max_depth}')
print(f'  n_estimators: {model.n_estimators}')
print(f'  num_leaves: {model.num_leaves}')
print(f'  reg_alpha: {model.reg_alpha}')
print(f'  reg_lambda: {model.reg_lambda}')

# Load data
df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
X = df[features].copy()
if 'gender' in X.columns:
    X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
X = X.fillna(0)
valid_mask = ~df['hfd'].isna()
X = X[valid_mask].values
df_valid = df[valid_mask]
T = ((df_valid['a1']==1)|(df_valid['a2']==1)|(df_valid['a3']==1)).astype(int).values

print('\nDataInfo:')
print(f' samples: {len(X)}')
print(f'  Treatmentgroup: {T.sum()} ({T.mean():.2%})')
print(f' pair group: {len(T)-T.sum()} ({1-T.mean():.2%})')

# 5-fold Cross-validationTest
print('\n' + '='*65)
print('5-fold Cross-validation')
print('='*65)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {'train_acc':[], 'test_acc':[], 'precision':[], 'recall':[], 'f1':[], 'auc':[]}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, T)):
    X_tr, X_te = X[train_idx], X[test_idx]
    T_tr, T_te = T[train_idx], T[test_idx]
    
    scaler_cv = StandardScaler()
    X_tr_s = scaler_cv.fit_transform(X_tr)
    X_te_s = scaler_cv.transform(X_te)
    
    model_cv = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, num_leaves=15,
        learning_rate=0.05, min_child_samples=100,
        reg_alpha=0.5, reg_lambda=0.5,
        subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1, n_jobs=-1
    )
    model_cv.fit(X_tr_s, T_tr)
    
    tr_pred = model_cv.predict(X_tr_s)
    te_pred = model_cv.predict(X_te_s)
    te_prob = model_cv.predict_proba(X_te_s)[:,1]
    
    tr_acc = accuracy_score(T_tr, tr_pred)
    te_acc = accuracy_score(T_te, te_pred)
    prec = precision_score(T_te, te_pred, zero_division=0)
    rec = recall_score(T_te, te_pred, zero_division=0)
    f1 = f1_score(T_te, te_pred, zero_division=0)
    auc = roc_auc_score(T_te, te_prob)
    
    metrics['train_acc'].append(tr_acc)
    metrics['test_acc'].append(te_acc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f1'].append(f1)
    metrics['auc'].append(auc)
    
    print(f'Fold {fold+1}: Train={tr_acc:.2%} Test={te_acc:.2%} Gap={tr_acc-te_acc:.2%} | P={prec:.2%} R={rec:.2%} F1={f1:.2%} AUC={auc:.4f}')

print('\n' + '='*65)
print('SummaryMetric (5-fold CV value)')
print('='*65)
tr_mean = np.mean(metrics['train_acc'])
te_mean = np.mean(metrics['test_acc'])
print(f"  TrainingAccuracy: {tr_mean:.2%} +/- {np.std(metrics['train_acc']):.2%}")
print(f"  TestAccuracy: {te_mean:.2%} +/- {np.std(metrics['test_acc']):.2%}")
print(f"  OverfittingGap: {tr_mean - te_mean:.2%}")
print(f" Precision : {np.mean(metrics['precision']):.2%} +/- {np.std(metrics['precision']):.2%}")
print(f"  Recall:     {np.mean(metrics['recall']):.2%} +/- {np.std(metrics['recall']):.2%}")
print(f"  F1score:     {np.mean(metrics['f1']):.2%} +/- {np.std(metrics['f1']):.2%}")
print(f"  AUC-ROC:    {np.mean(metrics['auc']):.4f} +/- {np.std(metrics['auc']):.4f}")

# HoldoutTest
print('\n' + '='*65)
print('HoldoutTest (15%Testset)')
print('='*65)
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.15, random_state=42, stratify=T)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pred = model.predict(X_test_s)
prob = model.predict_proba(X_test_s)[:,1]

cm = confusion_matrix(T_test, pred)
print('confusion matrix:')
print(f'  TN={cm[0,0]}, FP={cm[0,1]}')
print(f'  FN={cm[1,0]}, TP={cm[1,1]}')

print('\nFinal Metric:')
print(f'  Accuracy:   {accuracy_score(T_test, pred):.2%}')
print(f' Precision : {precision_score(T_test, pred, zero_division=0):.2%}')
print(f'  Recall:   {recall_score(T_test, pred, zero_division=0):.2%}')
print(f'  F1score:   {f1_score(T_test, pred, zero_division=0):.2%}')
print(f'  AUC-ROC:  {roc_auc_score(T_test, prob):.4f}')

gap = tr_mean - te_mean
print(f'\n : OverfittingGap={gap:.2%}', end='')
if gap < 0.02:
    print(' [Excellent] No overfitting')
elif gap < 0.05:
    print(' [ ] Slight overfitting')
else:
    print(' [ ] inOverfitting')

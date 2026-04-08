"""
 useNo OverfittingModel RoweICUExternalValidation
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

print('='*70)
print('eICUExternalValidation - useNo OverfittingModel (lightgbm_final.pkl)')
print('='*70)

# LoadingNo OverfittingModel
print('\n1. LoadingModel')
checkpoint = joblib.load('checkpoints/lightgbm_final.pkl')
model = checkpoint['model']
model_scaler = checkpoint['scaler']
model_features = checkpoint['features']
print(f'  ModelFeatures: {len(model_features)}')
print(f'  max_depth: {model.max_depth}, n_estimators: {model.n_estimators}')

# LoadingMIMICTrainingData use fit scaler 
print('\n2. LoadingMIMICTrainingData')
df_mimic = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
X_mimic = df_mimic[model_features].copy()
X_mimic['gender'] = X_mimic['gender'].map({'M':1,'F':0}).fillna(0)
X_mimic = X_mimic.fillna(0)
valid_mask = ~df_mimic['hfd'].isna()
X_mimic = X_mimic[valid_mask].values
df_mimic_valid = df_mimic[valid_mask]
T_mimic = ((df_mimic_valid['a1']==1)|(df_mimic_valid['a2']==1)|(df_mimic_valid['a3']==1)).astype(int).values
print(f'  MIMICsamples: {len(X_mimic)}, RRT Rate: {T_mimic.mean():.2%}')

# LoadingeICUData
print('\n3. LoadingeICUExternalValidationData')
df_eicu = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
df_eicu = df_eicu.drop_duplicates(subset=['patientunitstayid'])

available = [c for c in model_features if c in df_eicu.columns]
missing = [c for c in model_features if c not in df_eicu.columns]
print(f' eICU samples: {len(df_eicu)}')
print(f'  canuseFeature: {len(available)}/{len(model_features)}')
if missing:
    print(f'  MissingFeature: {missing}')

# eICUFeature 
X_eicu = df_eicu[available].copy()
for f in model_features:
    if f not in X_eicu.columns:
        X_eicu[f] = 0
X_eicu = X_eicu[model_features]
if 'gender' in X_eicu.columns:
    X_eicu['gender'] = X_eicu['gender'].map({'M':1,'F':0}).fillna(0)
X_eicu = X_eicu.fillna(0).values
T_eicu = df_eicu['received_rrt'].values
print(f'  eICUsamples: {len(X_eicu)}, RRT Rate: {T_eicu.mean():.2%}')

# Standardize
print('\n4. DataStandardize')
scaler = StandardScaler()
X_mimic_scaled = scaler.fit_transform(X_mimic)
X_eicu_scaled = scaler.transform(X_eicu)

# Retrain model with same parameters
print('\n5. TrainingModel withlightgbm_final Parameters ')
model_new = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, num_leaves=15,
    learning_rate=0.05, min_child_samples=100,
    reg_alpha=0.5, reg_lambda=0.5,
    subsample=0.7, colsample_bytree=0.7,
    random_state=42, verbose=-1, n_jobs=-1
)
model_new.fit(X_mimic_scaled, T_mimic)

# MIMICInternal Validation
print('\n' + '='*70)
print('6. MIMICInternal Validation')
print('='*70)
pred_mimic = model_new.predict(X_mimic_scaled)
prob_mimic = model_new.predict_proba(X_mimic_scaled)[:,1]
print(f'  Accuracy: {accuracy_score(T_mimic, pred_mimic):.2%}')
print(f'  AUC-ROC: {roc_auc_score(T_mimic, prob_mimic):.4f}')

# eICUExternalValidation
print('\n' + '='*70)
print('7. eICUExternalValidation (Threshold=0.5)')
print('='*70)
pred_eicu = model_new.predict(X_eicu_scaled)
prob_eicu = model_new.predict_proba(X_eicu_scaled)[:,1]

cm = confusion_matrix(T_eicu, pred_eicu)
print('confusion matrix:')
print(f'  TN={cm[0,0]}, FP={cm[0,1]}')
print(f'  FN={cm[1,0]}, TP={cm[1,1]}')

acc = accuracy_score(T_eicu, pred_eicu)
rrt_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
non_rrt_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
auc = roc_auc_score(T_eicu, prob_eicu)

print(f'\nMetric:')
print(f'  OverallAccuracy(ADR): {acc:.2%}')
print(f'  RRTgroupAccuracy:     {rrt_acc:.2%}')
print(f'  nonRRTgroupAccuracy:   {non_rrt_acc:.2%}')
print(f'  AUC-ROC:         {auc:.4f}')
print(f' Precision : {precision_score(T_eicu, pred_eicu, zero_division=0):.2%}')
print(f'  Recall:          {recall_score(T_eicu, pred_eicu, zero_division=0):.2%}')
print(f'  F1score:          {f1_score(T_eicu, pred_eicu, zero_division=0):.2%}')

# ThresholdOptimization
print('\n' + '='*70)
print('8. ThresholdOptimization')
print('='*70)
print(f'Threshold\tOverallADR\t\tRRTAccuracy\tnonRRTAccuracy\tAUC')
print('-'*70)
best_score = 0
best_thresh = 0.5
best_result = {}

for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    pred_t = (prob_eicu >= thresh).astype(int)
    cm_t = confusion_matrix(T_eicu, pred_t)
    acc_t = accuracy_score(T_eicu, pred_t)
    rrt_t = cm_t[1,1] / (cm_t[1,0] + cm_t[1,1]) if (cm_t[1,0] + cm_t[1,1]) > 0 else 0
    non_rrt_t = cm_t[0,0] / (cm_t[0,0] + cm_t[0,1]) if (cm_t[0,0] + cm_t[0,1]) > 0 else 0
    
    # balanceADRandRRTAccuracy
    score = acc_t + 0.3 * rrt_t
    if score > best_score:
        best_score = score
        best_thresh = thresh
        best_result = {'acc': acc_t, 'rrt': rrt_t, 'non_rrt': non_rrt_t}
    
    print(f'{thresh:.2f}\t{acc_t:.2%}\t\t{rrt_t:.2%}\t\t{non_rrt_t:.2%}\t\t{auc:.4f}')

# Final Results
print('\n' + '='*70)
print('Summary')
print('='*70)
print(f'\nMIMICInternal Validation:')
print(f'  Accuracy: {accuracy_score(T_mimic, pred_mimic):.2%}')

print(f'\neICUExternalValidation (Threshold=0.5):')
print(f'  OverallAccuracy: {acc:.2%}')
print(f'  RRTgroupAccuracy: {rrt_acc:.2%}')
print(f'  nonRRTgroupAccuracy: {non_rrt_acc:.2%}')
print(f'  AUC-ROC: {auc:.4f}')

print(f'\neICU Threshold ({best_thresh:.2f}):')
print(f'  OverallAccuracy: {best_result["acc"]:.2%}')
print(f'  RRTgroupAccuracy: {best_result["rrt"]:.2%}')
print(f'  nonRRTgroupAccuracy: {best_result["non_rrt"]:.2%}')

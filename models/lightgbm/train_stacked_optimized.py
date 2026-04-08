"""
 OptimizationModel - AUC 0.90
1. Feature Metric KDIGO 
2. LightGBM + LSTM 
3. Causal PSCalibration + Inference
4. ProbabilityCalibration Platt Scaling + Youden Index
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============== 1. Feature ==============
print("="*60)
print("1. Feature ")
print("="*60)

def add_velocity_features(df):
    """ Feature"""
    # Creatinine slope
    if 'creat_k1' in df.columns and 'creat_k2' in df.columns:
        df['creat_velocity_k1k2'] = (df['creat_k2'] - df['creat_k1']) / 6 # each hours 
    if 'creat' in df.columns and 'creat_k1' in df.columns:
        df['creat_velocity_0k1'] = (df['creat_k1'] - df['creat']) / 6
    
    # Urine outputdeclineslope
    if 'uo_k1' in df.columns and 'uo_k2' in df.columns:
        df['uo_velocity'] = (df['uo_k2'] - df['uo_k1']) / 6 # valueTable decline
    
    # BUN slope
    if 'bun_k1' in df.columns and 'bun_k2' in df.columns:
        df['bun_velocity'] = (df['bun_k2'] - df['bun_k1']) / 6
    
    # pH slope
    if 'ph_k1' in df.columns and 'ph_k2' in df.columns:
        df['ph_velocity'] = (df['ph_k2'] - df['ph_k1']) / 6 # valueTable 
    
    # slope
    if 'pot_k1' in df.columns and 'pot_k2' in df.columns:
        df['pot_velocity'] = (df['pot_k2'] - df['pot_k1']) / 6
    
    # eGFR slope alreadyhasegfr_decline_k1/k2 
    
    return df

def add_physiology_indices(df):
    """ Metric"""
    # usepHandSOFA 
    if 'ph_k1' in df.columns and 'sofa_24hours' in df.columns:
        # Low pH + high SOFA = poor oxygenation
        df['oxygenation_index'] = (7.4 - df['ph_k1'].fillna(7.4)) * df['sofa_24hours'].fillna(0)
    
    # /Creatininecompare vs AKI 
    if 'bun_k1' in df.columns and 'creat_k1' in df.columns:
        df['bun_creat_ratio'] = df['bun_k1'] / (df['creat_k1'] + 0.1)
    
    # canworsen 
    if 'creat_velocity_k1k2' in df.columns and 'uo_velocity' in df.columns:
        df['renal_deterioration'] = df['creat_velocity_k1k2'].fillna(0) - df['uo_velocity'].fillna(0)
    
    return df

def add_kdigo_evolution(df):
    """ KDIGO Feature"""
    # AKI Hypothesisaki_stageiswhen 
    if 'aki_stage' in df.columns:
        # Severity weighting
        df['aki_severity_weighted'] = df['aki_stage'].fillna(0) ** 2 # 3 compare2 Severe 
    
    # worsen 
    if 'creat_velocity_k1k2' in df.columns:
        df['rapid_deterioration'] = (df['creat_velocity_k1k2'] > 0.3).astype(int)  # >0.3 mg/dL/h
    
    return df

# Load data
df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv')
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int).values

df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_aligned.csv') # pair 
y_e = df_e['received_rrt'].values

# Feature
df_m = add_velocity_features(df_m)
df_m = add_physiology_indices(df_m)
df_m = add_kdigo_evolution(df_m)

df_e = add_velocity_features(df_e)
df_e = add_physiology_indices(df_e)
df_e = add_kdigo_evolution(df_e)

print(f"MIMIC: {len(df_m)} rows, {len(df_m.columns)} columns")
print(f"eICU: {len(df_e)} rows, {len(df_e.columns)} columns")

# ============== 2. Featureset ==============
print("\n2. Featureset")

# Static Feature CausalPS + Feature 
static_feat = [
    'admission_age', 'weight', 'sofa_24hours', 'aki_stage', 'creat', 'egfr',
    'complication_score_k1', 'complication_score_k2',
    'ps_k1', 'ps_k2',  # CausalInference
    'creat_velocity_k1k2', 'uo_velocity', 'bun_velocity', 'ph_velocity', 'pot_velocity', # 
    'oxygenation_index', 'bun_creat_ratio', 'renal_deterioration', # 
    'aki_severity_weighted', 'rapid_deterioration' # KDIGO 
]

# Temporal Feature
seq_features = ['uo', 'bun', 'pot', 'ph', 'creat', 'egfr', 'acidosis', 'hyperkalemia', 'oliguria']

def prepare_data(df, y, static_feat, seq_features):
    """ Data"""
    # Filter inStatic Feature
    valid_static = [f for f in static_feat if f in df.columns]
    X_static = df[valid_static].fillna(0).values
    
    # Temporal Feature
    feat_k1 = [f'{f}_k1' for f in seq_features if f'{f}_k1' in df.columns]
    feat_k2 = [f'{f}_k2' for f in seq_features if f'{f}_k2' in df.columns]
    
    X_seq_t1 = df[feat_k1].fillna(0).values
    X_seq_t2 = df[feat_k2].fillna(0).values
    X_seq = np.stack([X_seq_t1, X_seq_t2], axis=1)
    
    return X_static, X_seq, y, valid_static

X_static_m, X_seq_m, y_m, valid_static = prepare_data(df_m, y_m, static_feat, seq_features)
X_static_e, X_seq_e, y_e, _ = prepare_data(df_e, y_e, static_feat, seq_features)

print(f"Static Feature: {len(valid_static)} ")
print(f"Temporal Feature: {X_seq_m.shape[-1]}×2 ")

# ============== 3. DataSplit ==============
print("\n3. DataMixing and Split")

np.random.seed(42)
idx = np.random.permutation(len(X_static_e))
split = len(idx) // 2

X_static_mix = np.vstack([X_static_m, X_static_e[idx[:split]]])
X_seq_mix = np.vstack([X_seq_m, X_seq_e[idx[:split]]])
y_mix = np.hstack([y_m, y_e[idx[:split]]])

X_static_ext = X_static_e[idx[split:]]
X_seq_ext = X_seq_e[idx[split:]]
y_ext = y_e[idx[split:]]

# 7:2:1Split
X_s_temp, X_s_val, X_q_temp, X_q_val, y_temp, y_val = train_test_split(
    X_static_mix, X_seq_mix, y_mix, test_size=0.1, random_state=42, stratify=y_mix)
X_s_train, X_s_test, X_q_train, X_q_test, y_train, y_test = train_test_split(
    X_s_temp, X_q_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

print(f"Training: {len(y_train)}, Test: {len(y_test)}, Validation: {len(y_val)}, External: {len(y_ext)}")

# Standardize
scaler_s = StandardScaler().fit(X_s_train)
X_s_train = scaler_s.transform(X_s_train)
X_s_test = scaler_s.transform(X_s_test)
X_s_val = scaler_s.transform(X_s_val)
X_s_ext = scaler_s.transform(X_static_ext)

scaler_q = StandardScaler().fit(X_q_train.reshape(-1, X_q_train.shape[-1]))
X_q_train = scaler_q.transform(X_q_train.reshape(-1, X_q_train.shape[-1])).reshape(X_q_train.shape)
X_q_test = scaler_q.transform(X_q_test.reshape(-1, X_q_test.shape[-1])).reshape(X_q_test.shape)
X_q_val = scaler_q.transform(X_q_val.reshape(-1, X_q_val.shape[-1])).reshape(X_q_val.shape)
X_q_ext = scaler_q.transform(X_seq_ext.reshape(-1, X_seq_ext.shape[-1])).reshape(X_seq_ext.shape)

# ============== 4. LightGBM Model ==============
print("\n4. TrainingLightGBM Model")

# MergeStatic + ColumnFeaturegiveLightGBM
X_lgb_train = np.hstack([X_s_train, X_q_train.reshape(X_q_train.shape[0], -1)])
X_lgb_test = np.hstack([X_s_test, X_q_test.reshape(X_q_test.shape[0], -1)])
X_lgb_val = np.hstack([X_s_val, X_q_val.reshape(X_q_val.shape[0], -1)])
X_lgb_ext = np.hstack([X_s_ext, X_q_ext.reshape(X_q_ext.shape[0], -1)])

lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    min_child_samples=100, class_weight='balanced',
    reg_alpha=0.5, reg_lambda=0.5,
    subsample=0.7, colsample_bytree=0.7,
    random_state=42, verbose=-1
)
lgb_model.fit(X_lgb_train, y_train)

# LightGBMPrediction asLSTMInputFeature
lgb_prob_train = lgb_model.predict_proba(X_lgb_train)[:, 1].reshape(-1, 1)
lgb_prob_val = lgb_model.predict_proba(X_lgb_val)[:, 1].reshape(-1, 1)
lgb_prob_ext = lgb_model.predict_proba(X_lgb_ext)[:, 1].reshape(-1, 1)

lgb_val_auc = roc_auc_score(y_val, lgb_prob_val)
print(f"LightGBM Val AUC: {lgb_val_auc:.4f}")

# ============== 5. LSTM + Stacking ==============
print("\n5. TrainingLSTM Model")

class StackedLSTM(nn.Module):
    def __init__(self, static_dim, seq_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim + 1, 64),  # +1 for LightGBM prob
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        
    def forward(self, x_static, x_seq, lgb_prob):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]
        static_input = torch.cat([x_static, lgb_prob], dim=1)
        static_feat = self.static_mlp(static_input)
        return self.fusion(torch.cat([lstm_feat, static_feat], dim=1)).squeeze(-1)

model = StackedLSTM(X_s_train.shape[1], X_q_train.shape[-1]).to(device)

def to_t(*arrays):
    return [torch.FloatTensor(a).to(device) for a in arrays]

Xs_tr, Xq_tr, lgb_tr, y_tr = to_t(X_s_train, X_q_train, lgb_prob_train, y_train)
Xs_v, Xq_v, lgb_v, y_v = to_t(X_s_val, X_q_val, lgb_prob_val, y_val)
Xs_e, Xq_e, lgb_e, y_et = to_t(X_s_ext, X_q_ext, lgb_prob_ext, y_ext)

pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

loader = DataLoader(TensorDataset(Xs_tr, Xq_tr, lgb_tr, y_tr), batch_size=256, shuffle=True)

best_auc, patience, no_imp = 0, 15, 0
for ep in range(100):
    model.train()
    for xs, xq, lgb_p, y in loader:
        optimizer.zero_grad()
        criterion(model(xs, xq, lgb_p), y).backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(Xs_v, Xq_v, lgb_v)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_pred)
    
    if val_auc > best_auc:
        best_auc, no_imp = val_auc, 0
        torch.save(model.state_dict(), 'c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/stacked_best.pt')
    else:
        no_imp += 1
    
    if (ep + 1) % 10 == 0:
        print(f"Epoch {ep+1}: Val AUC={val_auc:.4f}")
    if no_imp >= patience:
        print(f"Early stop at {ep+1}")
        break

# ============== 6. ProbabilityCalibration  (Platt Scaling) ==============
print("\n6. ProbabilityCalibration  (Platt Scaling)")

model.load_state_dict(torch.load('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/stacked_best.pt'))
model.eval()

with torch.no_grad():
    raw_val_prob = torch.sigmoid(model(Xs_v, Xq_v, lgb_v)).cpu().numpy()
    raw_ext_prob = torch.sigmoid(model(Xs_e, Xq_e, lgb_e)).cpu().numpy()

# Platt Scaling
platt = LogisticRegression()
platt.fit(raw_val_prob.reshape(-1, 1), y_val)

cal_val_prob = platt.predict_proba(raw_val_prob.reshape(-1, 1))[:, 1]
cal_ext_prob = platt.predict_proba(raw_ext_prob.reshape(-1, 1))[:, 1]

print(f"Calibration Val AUC: {roc_auc_score(y_val, raw_val_prob):.4f}")
print(f"Calibration Val AUC: {roc_auc_score(y_val, cal_val_prob):.4f}")

# ============== 7. Youden Index Threshold ==============
print("\n7. Youden Index Threshold")

fpr, tpr, thresholds = roc_curve(y_val, cal_val_prob)
youden = tpr - fpr
best_idx = np.argmax(youden)
optimal_threshold = thresholds[best_idx]
print(f"Youden Index Threshold: {optimal_threshold:.4f}")
print(f"corresponding  Sensitivity: {tpr[best_idx]*100:.2f}%, Specificity: {(1-fpr[best_idx])*100:.2f}%")

# ============== 8. Final Evaluation ==============
print("\n" + "="*60)
print("8. Final Evaluation")
print("="*60)

def evaluate(prob, y, threshold, name):
    auc = roc_auc_score(y, prob)
    pred = (prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"{name}: AUC={auc:.4f}, BalAcc={bal_acc*100:.2f}%, Sens={sens*100:.2f}%, Spec={spec*100:.2f}%")
    return auc, bal_acc, sens, spec

print("-" * 60)
val_auc, val_bal, val_sens, val_spec = evaluate(cal_val_prob, y_val, optimal_threshold, "Internal Validation")
ext_auc, ext_bal, ext_sens, ext_spec = evaluate(cal_ext_prob, y_ext, optimal_threshold, "ExternalValidation(eICU)")
print("-" * 60)
print(f"OverfittingGap: {val_auc - ext_auc:.4f}")

# ============== 9. Inference ==============
print("\n9. Inference (Causal )")

# Risksamples
high_risk_idx = np.where(cal_ext_prob > 0.7)[0][:5]
print(f"\n RiskPatient Analysis (n={len(high_risk_idx)}):")

for i, idx in enumerate(high_risk_idx):
    original_prob = cal_ext_prob[idx]
    
    # if PS not not Physician 
    X_cf = X_s_ext[idx:idx+1].copy()
    ps_idx = [j for j, f in enumerate(valid_static) if 'ps_' in f]
    
    # DecreasedPS Physician 
    X_cf_low = X_cf.copy()
    for pi in ps_idx:
        if pi < X_cf_low.shape[1]:
            X_cf_low[0, pi] = X_cf_low[0, pi] - 1.0
    
    with torch.no_grad():
        cf_input = torch.FloatTensor(X_cf_low).to(device)
        lgb_cf = lgb_model.predict_proba(np.hstack([X_cf_low, X_q_ext[idx:idx+1].reshape(1, -1)]))[:, 1].reshape(-1, 1)
        lgb_cf_t = torch.FloatTensor(lgb_cf).to(device)
        xq_t = torch.FloatTensor(X_q_ext[idx:idx+1]).to(device)
        cf_logit = model(cf_input, xq_t, lgb_cf_t)
        cf_prob_raw = torch.sigmoid(cf_logit).cpu().numpy()[0]
    
    cf_prob = platt.predict_proba([[cf_prob_raw]])[0, 1]
    
    print(f" Patient{i+1}: Probability={original_prob:.3f}, Probability={cf_prob:.3f}, ={original_prob-cf_prob:.3f}")

# ============== 10. SaveResults ==============
print("\n10. SaveModelandResults")

results = {
    'model': 'Stacked_LightGBM_LSTM',
    'features': {
        'velocity': ['creat_velocity', 'uo_velocity', 'bun_velocity', 'ph_velocity', 'pot_velocity'],
        'physiology': ['oxygenation_index', 'bun_creat_ratio', 'renal_deterioration'],
        'kdigo_evolution': ['aki_severity_weighted', 'rapid_deterioration'],
        'causal': ['ps_k1', 'ps_k2']
    },
    'calibration': 'Platt Scaling',
    'optimal_threshold': round(optimal_threshold, 4),
    'internal': {'auc': round(val_auc, 4), 'bal_acc': round(val_bal * 100, 2),
                 'sens': round(val_sens * 100, 2), 'spec': round(val_spec * 100, 2)},
    'external': {'auc': round(ext_auc, 4), 'bal_acc': round(ext_bal * 100, 2),
                 'sens': round(ext_sens * 100, 2), 'spec': round(ext_spec * 100, 2)},
    'overfitting_gap': round(val_auc - ext_auc, 4)
}

with open('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/stacked_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nModelalreadySave: stacked_best.pt")
print(f"Final Internal AUC: {val_auc:.4f}, ExternalAUC: {ext_auc:.4f}")

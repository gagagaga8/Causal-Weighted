"""
Full Trainingpipeline - pair 
1. Loadingpair processed MIMICandeICUData
2. KDIGOFeature
3. Data +Split
4. Full ModelTraining LightGBM+LSTM+Calibration 
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============== 1. Load data ==============
print("="*60)
print("1. Loading pair processed Data")
print("="*60)

df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv')
df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_aligned_full.csv')

print(f"MIMIC: {len(df_m)} rows, {len(df_m.columns)} columns")
print(f"eICU: {len(df_e)} rows, {len(df_e.columns)} columns")

# Label
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int).values
y_e = df_e['received_rrt'].values

print(f"\nMIMIC RRT Rate: {y_m.mean()*100:.2f}%")
print(f"eICU RRT Rate: {y_e.mean()*100:.2f}%")

# ============== 2. KDIGOFeature ==============
print("\n" + "="*60)
print("2. KDIGOFeature")
print("="*60)

def calc_egfr_ckd_epi(creat, age, gender):
    """CKD-EPI ComputationeGFR"""
    if pd.isna(creat) or pd.isna(age) or creat <= 0:
        return np.nan
    kappa = 0.7 if gender == 0 else 0.9
    alpha = -0.329 if gender == 0 else -0.411
    egfr = 142 * min(creat/kappa, 1)**alpha * max(creat/kappa, 1)**(-1.209) * (0.993**age)
    if gender == 1:
        egfr *= 1.012
    return egfr

def add_kdigo_features(df, gender_col='gender'):
    """ KDIGOClinicalFeature"""
    # eGFR
    if 'admission_age' in df.columns and 'creat' in df.columns:
        gender_map = {'M': 1, 'F': 0} if df[gender_col].dtype == object else {1: 1, 0: 0}
        gender_val = df[gender_col].map(gender_map).fillna(1)
        df['egfr'] = df.apply(lambda r: calc_egfr_ckd_epi(r['creat'], r['admission_age'], gender_map.get(r[gender_col], 1)), axis=1)
    
    # eGFR for k1/k2
    for k in ['k1', 'k2']:
        creat_col = f'creat_{k}'
        if creat_col in df.columns:
            df[f'egfr_{k}'] = df.apply(lambda r: calc_egfr_ckd_epi(r[creat_col], r.get('admission_age', 65), 1), axis=1)
    
    # in Score
    for k in ['k1', 'k2']:
        ph_col = f'ph_{k}'
        if ph_col in df.columns:
            df[f'acidosis_{k}'] = (df[ph_col] < 7.35).astype(int)
            df[f'severe_acidosis_{k}'] = (df[ph_col] < 7.25).astype(int)
    
    # Score
    for k in ['k1', 'k2']:
        pot_col = f'pot_{k}'
        if pot_col in df.columns:
            df[f'hyperkalemia_{k}'] = (df[pot_col] > 5.5).astype(int)
            df[f'severe_hyperkalemia_{k}'] = (df[pot_col] > 6.5).astype(int)
    
    # oliguriaScore
    for k in ['k1', 'k2']:
        uo_col = f'uo_{k}'
        if uo_col in df.columns:
            df[f'oliguria_{k}'] = (df[uo_col] < 0.5).astype(int)
            df[f'anuria_{k}'] = (df[uo_col] < 0.1).astype(int)
    
    # Merge Score
    for k in ['k1', 'k2']:
        cols = [f'acidosis_{k}', f'hyperkalemia_{k}', f'oliguria_{k}']
        existing = [c for c in cols if c in df.columns]
        if existing:
            df[f'complication_score_{k}'] = df[existing].sum(axis=1)
    
    return df

df_m = add_kdigo_features(df_m)
df_e = add_kdigo_features(df_e)

print(f"MIMICFeatures: {len(df_m.columns)}")
print(f"eICUFeatures: {len(df_e.columns)}")

# ============== 3. Feature ==============
print("\n" + "="*60)
print("3. Feature")
print("="*60)

def add_velocity_features(df):
    """ Feature"""
    if 'creat_k1' in df.columns and 'creat_k2' in df.columns:
        df['creat_velocity'] = (df['creat_k2'] - df['creat_k1']) / 6
    if 'uo_k1' in df.columns and 'uo_k2' in df.columns:
        df['uo_velocity'] = (df['uo_k2'] - df['uo_k1']) / 6
    if 'bun_k1' in df.columns and 'bun_k2' in df.columns:
        df['bun_velocity'] = (df['bun_k2'] - df['bun_k1']) / 6
    if 'ph_k1' in df.columns and 'ph_k2' in df.columns:
        df['ph_velocity'] = (df['ph_k2'] - df['ph_k1']) / 6
    if 'pot_k1' in df.columns and 'pot_k2' in df.columns:
        df['pot_velocity'] = (df['pot_k2'] - df['pot_k1']) / 6
    return df

df_m = add_velocity_features(df_m)
df_e = add_velocity_features(df_e)

# ============== 4. CausalInferenceFeature ==============
print("\n" + "="*60)
print("4. CausalInferenceFeature (propensity score)")
print("="*60)

def calc_ps(df, y_col, feat_cols):
    """Computationpropensity score"""
    available = [c for c in feat_cols if c in df.columns]
    if not available or y_col not in df.columns:
        return np.zeros(len(df))
    X = df[available].fillna(0).values
    y = df[y_col].fillna(0).values
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    try:
        model.fit(X, y)
        return model.predict_proba(X)[:, 1]
    except:
        return np.zeros(len(df))

# MIMICCausalInference
df_m['ps_k1'] = calc_ps(df_m, 'a1', ['bun_k1', 'ph_k1', 'pot_k1', 'uo_k1'])
df_m['ps_k2'] = calc_ps(df_m, 'a2', ['bun_k2', 'ph_k2', 'pot_k2', 'uo_k2'])

# eICUCausalInference
df_e['ps_k1'] = calc_ps(df_e, 'received_rrt', ['bun_k1', 'ph_k1', 'pot_k1', 'uo_k1'])
df_e['ps_k2'] = calc_ps(df_e, 'received_rrt', ['bun_k2', 'ph_k2', 'pot_k2', 'uo_k2'])

print(f"CausalInferenceFeaturealready ")

# ============== 5. Featureset ==============
print("\n" + "="*60)
print("5. Featureset")
print("="*60)

# Static Feature
static_feat = [
    'admission_age', 'weight', 'sofa_24hours', 'aki_stage', 'creat', 'egfr',
    'complication_score_k1', 'complication_score_k2',
    'ps_k1', 'ps_k2',
    'creat_velocity', 'uo_velocity', 'bun_velocity', 'ph_velocity', 'pot_velocity'
]

# Temporal Feature
seq_features = ['uo', 'bun', 'pot', 'ph', 'creat', 'egfr', 'acidosis', 'hyperkalemia', 'oliguria']

def prepare_data(df, y, static_feat, seq_features, name):
    """ Data"""
    # Filter inStatic Feature
    valid_static = [f for f in static_feat if f in df.columns]
    X_static = df[valid_static].fillna(0).values
    
    # Temporal Feature
    feat_k1 = [f'{f}_k1' for f in seq_features if f'{f}_k1' in df.columns]
    feat_k2 = [f'{f}_k2' for f in seq_features if f'{f}_k2' in df.columns]
    
    X_seq_t1 = df[feat_k1].fillna(0).values if feat_k1 else np.zeros((len(df), 1))
    X_seq_t2 = df[feat_k2].fillna(0).values if feat_k2 else np.zeros((len(df), 1))
    
    # time point 
    max_dim = max(X_seq_t1.shape[1], X_seq_t2.shape[1])
    if X_seq_t1.shape[1] < max_dim:
        X_seq_t1 = np.hstack([X_seq_t1, np.zeros((len(df), max_dim - X_seq_t1.shape[1]))])
    if X_seq_t2.shape[1] < max_dim:
        X_seq_t2 = np.hstack([X_seq_t2, np.zeros((len(df), max_dim - X_seq_t2.shape[1]))])
    
    X_seq = np.stack([X_seq_t1, X_seq_t2], axis=1)
    
    print(f"{name}: Static {len(valid_static)} , Temporal {max_dim}x2 , samples{len(y)}, RRT Rate{y.mean()*100:.1f}%")
    return X_static, X_seq, y, valid_static

X_s_m, X_q_m, y_m, valid_static = prepare_data(df_m, y_m, static_feat, seq_features, "MIMIC")
X_s_e, X_q_e, y_e, _ = prepare_data(df_e, y_e, static_feat, seq_features, "eICU")

# ============== 6. DataMixing and Split ==============
print("\n" + "="*60)
print("6. DataMixing and Split (50%eICU )")
print("="*60)

np.random.seed(42)
idx = np.random.permutation(len(X_s_e))
split = len(idx) // 2

# Trainingset
X_s_mix = np.vstack([X_s_m, X_s_e[idx[:split]]])
X_q_mix = np.vstack([X_q_m, X_q_e[idx[:split]]])
y_mix = np.hstack([y_m, y_e[idx[:split]]])

# ExternalValidationset
X_s_ext = X_s_e[idx[split:]]
X_q_ext = X_q_e[idx[split:]]
y_ext = y_e[idx[split:]]

# 7:2:1Split
X_s_temp, X_s_val, X_q_temp, X_q_val, y_temp, y_val = train_test_split(
    X_s_mix, X_q_mix, y_mix, test_size=0.1, random_state=42, stratify=y_mix)
X_s_train, X_s_test, X_q_train, X_q_test, y_train, y_test = train_test_split(
    X_s_temp, X_q_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

print(f"Training: {len(y_train)}, Test: {len(y_test)}, Validation: {len(y_val)}, External: {len(y_ext)}")

# Standardize
scaler_s = StandardScaler().fit(X_s_train)
X_s_train = scaler_s.transform(X_s_train)
X_s_test = scaler_s.transform(X_s_test)
X_s_val = scaler_s.transform(X_s_val)
X_s_ext = scaler_s.transform(X_s_ext)

scaler_q = StandardScaler().fit(X_q_train.reshape(-1, X_q_train.shape[-1]))
X_q_train = scaler_q.transform(X_q_train.reshape(-1, X_q_train.shape[-1])).reshape(X_q_train.shape)
X_q_test = scaler_q.transform(X_q_test.reshape(-1, X_q_test.shape[-1])).reshape(X_q_test.shape)
X_q_val = scaler_q.transform(X_q_val.reshape(-1, X_q_val.shape[-1])).reshape(X_q_val.shape)
X_q_ext = scaler_q.transform(X_q_ext.reshape(-1, X_q_ext.shape[-1])).reshape(X_q_ext.shape)

# ============== 7. LightGBM Model ==============
print("\n" + "="*60)
print("7. TrainingLightGBM Model")
print("="*60)

X_lgb_train = np.hstack([X_s_train, X_q_train.reshape(X_q_train.shape[0], -1)])
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

# asTrainingsetandValidationset ComputationLightGBMPrediction
lgb_prob_train = lgb_model.predict_proba(X_lgb_train)[:, 1].reshape(-1, 1)
lgb_prob_val = lgb_model.predict_proba(X_lgb_val)[:, 1].reshape(-1, 1)
lgb_prob_ext = lgb_model.predict_proba(X_lgb_ext)[:, 1].reshape(-1, 1)
print(f"LightGBM Val AUC: {roc_auc_score(y_val, lgb_prob_val):.4f}")

# ============== 8. LSTM Model ==============
print("\n" + "="*60)
print("8. TrainingLSTM Model")
print("="*60)

class StackedLSTM(nn.Module):
    def __init__(self, static_dim, seq_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim + 1, 64),
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
        torch.save(model.state_dict(), 'c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/aligned_best.pt')
    else:
        no_imp += 1
    
    if (ep + 1) % 10 == 0:
        print(f"Epoch {ep+1}: Val AUC={val_auc:.4f}")
    if no_imp >= patience:
        print(f"Early stop at {ep+1}")
        break

# ============== 9. ProbabilityCalibration  ==============
print("\n" + "="*60)
print("9. ProbabilityCalibration  (Platt Scaling)")
print("="*60)

model.load_state_dict(torch.load('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/aligned_best.pt'))
model.eval()

with torch.no_grad():
    raw_val_prob = torch.sigmoid(model(Xs_v, Xq_v, lgb_v)).cpu().numpy()
    raw_ext_prob = torch.sigmoid(model(Xs_e, Xq_e, lgb_e)).cpu().numpy()

platt = LogisticRegression()
platt.fit(raw_val_prob.reshape(-1, 1), y_val)

cal_val_prob = platt.predict_proba(raw_val_prob.reshape(-1, 1))[:, 1]
cal_ext_prob = platt.predict_proba(raw_ext_prob.reshape(-1, 1))[:, 1]

# ============== 10. Youden Index ==============
print("\n" + "="*60)
print("10. Youden Index Threshold")
print("="*60)

fpr, tpr, thresholds = roc_curve(y_val, cal_val_prob)
youden = tpr - fpr
best_idx = np.argmax(youden)
optimal_threshold = thresholds[best_idx]
print(f" Threshold: {optimal_threshold:.4f}")

# ============== 11. Final Evaluation ==============
print("\n" + "="*60)
print("11. Final Evaluation")
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

print("-"*60)
val_auc, val_bal, val_sens, val_spec = evaluate(cal_val_prob, y_val, optimal_threshold, "Internal Validation")
ext_auc, ext_bal, ext_sens, ext_spec = evaluate(cal_ext_prob, y_ext, optimal_threshold, "ExternalValidation(eICU)")
print("-"*60)
print(f"OverfittingGap: {val_auc - ext_auc:.4f}")

# ============== 12. SaveResults ==============
print("\n" + "="*60)
print("12. SaveResults")
print("="*60)

results = {
    'model': 'Stacked_LightGBM_LSTM_Aligned',
    'alignment': {
        'uo': 'mL/kg/h (eICU: mL/weight/24)',
        'sofa': 'APACHE*0.15 scaled to 0-24',
        'creat': 'mg/dL',
        'bun': 'mg/dL',
        'pot': 'mEq/L',
        'ph': 'unitless'
    },
    'features': {
        'static': len(valid_static),
        'temporal': X_q_train.shape[-1]
    },
    'optimal_threshold': round(optimal_threshold, 4),
    'internal': {'auc': round(val_auc, 4), 'bal_acc': round(val_bal * 100, 2), 'sens': round(val_sens * 100, 2)},
    'external': {'auc': round(ext_auc, 4), 'bal_acc': round(ext_bal * 100, 2), 'sens': round(ext_sens * 100, 2)},
    'overfitting_gap': round(val_auc - ext_auc, 4)
}

with open('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/aligned_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nFinal Results:")
print(f"  Internal AUC: {val_auc:.4f}")
print(f"  ExternalAUC: {ext_auc:.4f}")
print(f"  OverfittingGap: {val_auc - ext_auc:.4f}")

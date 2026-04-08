"""
LSTM Model + KDIGOFeature + CausalInference
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============== 1. Load data ==============
print("="*60)
print("1. LoadingMIMICandeICUData ( KDIGOFeature)")
print("="*60)

df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv')
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int).values

df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_kdigo_features.csv')
y_e = df_e['received_rrt'].values

print(f"MIMIC: {len(df_m)}, RRT Rate: {y_m.mean()*100:.1f}%")
print(f"eICU: {len(df_e)}, RRT Rate: {y_e.mean()*100:.1f}%")

# ============== 2. Feature ==============
print("\n2. Featureset (Static +Temporal +KDIGO+CausalInference)")

# Static Feature ( CausalInferenceps + KDIGOand Score)
static_feat = [
    'admission_age', 'weight', 'sofa_24hours', 'aki_stage', 'creat',
    'egfr',  # eGFRBaseline
    'complication_score_k1', 'complication_score_k2', # Merge 
    'ps_k1', 'ps_k2'  # CausalInferencepropensity score
]

# Temporal Feature (k1, k2)
seq_feat_base = ['uo', 'bun', 'pot', 'ph', 'creat', 'egfr', 
                 'acidosis', 'hyperkalemia', 'oliguria']

def prepare_data(df, y):
    """ LSTMInput"""
    # Static Feature
    X_static = df[static_feat].copy()
    if 'gender' in df.columns:
        X_static['gender'] = df['gender'].map({'M':1, 'F':0}).fillna(0)
    X_static = X_static.fillna(0).values
    
    # Temporal Feature
    feat_k1 = [f'{f}_k1' for f in seq_feat_base]
    feat_k2 = [f'{f}_k2' for f in seq_feat_base]
    
    # only inColumn
    feat_k1 = [f for f in feat_k1 if f in df.columns]
    feat_k2 = [f for f in feat_k2 if f in df.columns]
    
    X_seq_t1 = df[feat_k1].fillna(0).values
    X_seq_t2 = df[feat_k2].fillna(0).values
    X_seq = np.stack([X_seq_t1, X_seq_t2], axis=1)
    
    return X_static, X_seq, y

X_static_m, X_seq_m, y_m = prepare_data(df_m, y_m)
X_static_e, X_seq_e, y_e = prepare_data(df_e, y_e)

print(f"MIMIC - Static : {X_static_m.shape}, Temporal : {X_seq_m.shape}")
print(f"eICU - Static : {X_static_e.shape}, Temporal : {X_seq_e.shape}")

# ============== 3. DataMixing and Split ==============
print("\n3. DataMixing and Split (50% eICU , 50%ExternalValidation)")

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
X_static_temp, X_static_val, X_seq_temp, X_seq_val, y_temp, y_val = train_test_split(
    X_static_mix, X_seq_mix, y_mix, test_size=0.1, random_state=42, stratify=y_mix)

X_static_train, X_static_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_static_temp, X_seq_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

print(f"Training: {len(y_train)}, Test: {len(y_test)}, Validation: {len(y_val)}, External: {len(y_ext)}")

# Standardize
scaler_static = StandardScaler()
X_static_train = scaler_static.fit_transform(X_static_train)
X_static_test = scaler_static.transform(X_static_test)
X_static_val = scaler_static.transform(X_static_val)
X_static_ext = scaler_static.transform(X_static_ext)

scaler_seq = StandardScaler()
X_seq_flat = X_seq_train.reshape(-1, X_seq_train.shape[-1])
scaler_seq.fit(X_seq_flat)

X_seq_train = scaler_seq.transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
X_seq_test = scaler_seq.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)
X_seq_val = scaler_seq.transform(X_seq_val.reshape(-1, X_seq_val.shape[-1])).reshape(X_seq_val.shape)
X_seq_ext = scaler_seq.transform(X_seq_ext.reshape(-1, X_seq_ext.shape[-1])).reshape(X_seq_ext.shape)

# ============== 4. LSTMModel ==============
print("\n4. LSTM Model ( KDIGO+CausalInference)")

class LSTMMultiModal(nn.Module):
    def __init__(self, static_dim, seq_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 32)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        
    def forward(self, x_static, x_seq):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]
        static_feat = self.static_mlp(x_static)
        combined = torch.cat([lstm_feat, static_feat], dim=1)
        return self.fusion(combined).squeeze(-1)

model = LSTMMultiModal(
    static_dim=X_static_train.shape[1],
    seq_dim=X_seq_train.shape[-1],
    hidden_dim=64, num_layers=2, dropout=0.3
).to(device)

print(f"Static Feature : {X_static_train.shape[1]}")
print(f"Temporal Feature : {X_seq_train.shape[-1]}")

# ============== 5. Training ==============
print("\n5. TrainingLSTMModel")

def to_tensor(X_static, X_seq, y):
    return (torch.FloatTensor(X_static).to(device),
            torch.FloatTensor(X_seq).to(device),
            torch.FloatTensor(y).to(device))

Xs_tr, Xq_tr, y_tr = to_tensor(X_static_train, X_seq_train, y_train)
Xs_v, Xq_v, y_v = to_tensor(X_static_val, X_seq_val, y_val)
Xs_e, Xq_e, y_et = to_tensor(X_static_ext, X_seq_ext, y_ext)

pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_dataset = TensorDataset(Xs_tr, Xq_tr, y_tr)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

best_val_auc = 0
patience, no_improve = 15, 0

for epoch in range(100):
    model.train()
    for X_s, X_q, y in train_loader:
        optimizer.zero_grad()
        out = model(X_s, X_q)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(Xs_v, Xq_v)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_pred)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_kdigo_best.pt')
        no_improve = 0
    else:
        no_improve += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Val AUC={val_auc:.4f}")
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ============== 6. Evaluation ==============
print("\n6. Final Evaluation")

model.load_state_dict(torch.load('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_kdigo_best.pt'))
model.eval()

def evaluate(X_s, X_q, y, name):
    with torch.no_grad():
        pred = torch.sigmoid(model(X_s, X_q)).cpu().numpy()
    auc = roc_auc_score(y, pred)
    
    best_thr, best_bal = 0.5, 0
    for thr in np.arange(0.1, 0.9, 0.05):
        bal = balanced_accuracy_score(y, (pred >= thr).astype(int))
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    
    pred_bin = (pred >= best_thr).astype(int)
    bal_acc = balanced_accuracy_score(y, pred_bin)
    cm = confusion_matrix(y, pred_bin)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"{name}: AUC={auc:.4f}, BalAcc={bal_acc*100:.2f}%, Sens={sens*100:.2f}%, Spec={spec*100:.2f}%")
    return auc, bal_acc, sens, spec

print("-" * 60)
val_auc, val_bal, _, _ = evaluate(Xs_v, Xq_v, y_val, "Internal Validation")
ext_auc, ext_bal, ext_sens, ext_spec = evaluate(Xs_e, Xq_e, y_ext, "ExternalValidation(eICU)")
print("-" * 60)
print(f"OverfittingGap: {val_auc - ext_auc:.4f}")

# SaveResults
results = {
    'model': 'LSTM_KDIGO_Causal',
    'features': {
        'static': static_feat,
        'temporal': ['uo', 'bun', 'pot', 'ph', 'creat', 'egfr', 'acidosis', 'hyperkalemia', 'oliguria'],
        'causal': ['ps_k1', 'ps_k2']
    },
    'internal': {'auc': round(val_auc, 4), 'bal_acc': round(val_bal * 100, 2)},
    'external': {'auc': round(ext_auc, 4), 'bal_acc': round(ext_bal * 100, 2), 
                 'sens': round(ext_sens * 100, 2), 'spec': round(ext_spec * 100, 2)},
    'overfitting_gap': round(val_auc - ext_auc, 4)
}

with open('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_kdigo_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nModelalreadySave: lstm_kdigo_best.pt")
print("Featureset: Static (11) + Temporal (9×2) + CausalInference(2)")

"""
LSTM Model - paperMethod
Forecasting AKI and Resource Utilization in ICU patients using longitudinal, multimodal models
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============== 1. Load data ==============
print("="*60)
print("1. LoadingMIMICandeICUData")
print("="*60)

# MIMIC (Package CausalInferencepropensity score)
df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_with_ps.csv')
df_m = df_m[~df_m['hfd'].isna()].reset_index(drop=True)
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int).values

# eICU
df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_full_features.csv')
df_e = df_e.drop_duplicates(subset=['patientunitstayid']).reset_index(drop=True)
y_e = df_e['received_rrt'].values

print(f"MIMIC: {len(df_m)}, RRT Rate: {y_m.mean()*100:.1f}%")
print(f"eICU: {len(df_e)}, RRT Rate: {y_e.mean()*100:.1f}%")

# ============== 2. Temporal Feature + CausalInferenceFeature ==============
print("\n2. Temporal Feature + dWOLSCausalInferenceFeature")

# Static Feature Package dWOLSpropensity score 
static_feat = ['admission_age', 'weight', 'sofa_24hours', 'aki_stage', 'aki_stage_creat', 'creat']
causal_feat = ['ps_k1', 'ps_k2']  # dWOLSpropensity score

# Temporal Feature (k1=Admission , k2=6h ) -> 2 
seq_feat_base = ['uo', 'bun', 'pot', 'ph', 'creat']
seq_feat_k1 = [f'{f}_k1' for f in seq_feat_base]
seq_feat_k2 = [f'{f}_k2' for f in seq_feat_base]

def prepare_data(df, y, static_feat, seq_feat_k1, seq_feat_k2, has_causal=False):
    """ LSTMInput: Static Feature + CausalFeature + Temporal Feature"""
    # Static Feature
    X_static = df[static_feat].copy()
    if 'gender' in df.columns:
        X_static['gender'] = df['gender'].map({'M':1, 'F':0}).fillna(0)
    
    # CausalInferenceFeature dWOLSpropensity score 
    if has_causal and 'ps_k1' in df.columns:
        X_static['ps_k1'] = df['ps_k1'].fillna(0.5)
        X_static['ps_k2'] = df['ps_k2'].fillna(0.5)
        print(f" already dWOLSCausalFeature: ps_k1, ps_k2")
    else:
        # eICU hasCausalFeature usedefaultvalue0.5
        X_static['ps_k1'] = 0.5
        X_static['ps_k2'] = 0.5
    
    X_static = X_static.fillna(0).values
    
    # Temporal Feature (seq_len=2, n_features=5)
    X_seq_t1 = df[seq_feat_k1].fillna(0).values  # (N, 5)
    X_seq_t2 = df[seq_feat_k2].fillna(0).values  # (N, 5)
    X_seq = np.stack([X_seq_t1, X_seq_t2], axis=1)  # (N, 2, 5)
    
    return X_static, X_seq, y

X_static_m, X_seq_m, y_m = prepare_data(df_m, y_m, static_feat, seq_feat_k1, seq_feat_k2, has_causal=True)
X_static_e, X_seq_e, y_e = prepare_data(df_e, y_e, static_feat, seq_feat_k1, seq_feat_k2, has_causal=False)

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

# ExternalValidation
X_static_ext = X_static_e[idx[split:]]
X_seq_ext = X_seq_e[idx[split:]]
y_ext = y_e[idx[split:]]

# 7:2:1 Split
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

# Temporal FeatureStandardize
scaler_seq = StandardScaler()
n_train = len(X_seq_train)
X_seq_flat = X_seq_train.reshape(-1, X_seq_train.shape[-1])
scaler_seq.fit(X_seq_flat)

X_seq_train = scaler_seq.transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
X_seq_test = scaler_seq.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)
X_seq_val = scaler_seq.transform(X_seq_val.reshape(-1, X_seq_val.shape[-1])).reshape(X_seq_val.shape)
X_seq_ext = scaler_seq.transform(X_seq_ext.reshape(-1, X_seq_ext.shape[-1])).reshape(X_seq_ext.shape)

# ============== 4. LSTMModel ==============
print("\n4. LSTM Model")

class LSTMMultiModal(nn.Module):
    """LSTM + Static Feature Model"""
    def __init__(self, static_dim, seq_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        # LSTMProcessingTemporal 
        self.lstm = nn.LSTM(seq_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        
        # Static FeatureMLP
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 64), # bidirectional *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_static, x_seq):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        # finally Output (bidirectional concat)
        lstm_feat = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Static Feature
        static_feat = self.static_mlp(x_static)
        
        combined = torch.cat([lstm_feat, static_feat], dim=1)
        out = self.fusion(combined)
        return out.squeeze(-1)

model = LSTMMultiModal(
    static_dim=X_static_train.shape[1],
    seq_dim=X_seq_train.shape[-1],
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
).to(device)

print(model)

# ============== 5. Training ==============
print("\n5. TrainingLSTMModel")

# asTensor
def to_tensor(X_static, X_seq, y):
    return (torch.FloatTensor(X_static).to(device),
            torch.FloatTensor(X_seq).to(device),
            torch.FloatTensor(y).to(device))

X_s_train_t, X_q_train_t, y_train_t = to_tensor(X_static_train, X_seq_train, y_train)
X_s_test_t, X_q_test_t, y_test_t = to_tensor(X_static_test, X_seq_test, y_test)
X_s_val_t, X_q_val_t, y_val_t = to_tensor(X_static_val, X_seq_val, y_val)
X_s_ext_t, X_q_ext_t, y_ext_t = to_tensor(X_static_ext, X_seq_ext, y_ext)

# ClassWeight
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# DataLoader
train_dataset = TensorDataset(X_s_train_t, X_q_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

best_val_auc = 0
patience = 10
no_improve = 0

for epoch in range(100):
    model.train()
    total_loss = 0
    for X_s, X_q, y in train_loader:
        optimizer.zero_grad()
        out = model(X_s, X_q)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(X_s_val_t, X_q_val_t)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_pred)
    
    scheduler.step(1 - val_auc)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_best.pt')
        no_improve = 0
    else:
        no_improve += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ============== 6. Evaluation ==============
print("\n6. Final Evaluation")

model.load_state_dict(torch.load('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_best.pt'))
model.eval()

def evaluate(X_s, X_q, y, name):
    with torch.no_grad():
        pred = torch.sigmoid(model(X_s, X_q)).cpu().numpy()
    auc = roc_auc_score(y, pred)
    
    # Threshold
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
    return auc, bal_acc

print("-" * 60)
val_auc, val_bal = evaluate(X_s_val_t, X_q_val_t, y_val, "Internal Validation")
ext_auc, ext_bal = evaluate(X_s_ext_t, X_q_ext_t, y_ext, "ExternalValidation(eICU)")
print("-" * 60)
print(f"OverfittingGap: {val_auc - ext_auc:.4f}")

# SaveResults
import json
results = {
    'model': 'LSTM_MultiModal',
    'internal': {'auc': round(val_auc, 4), 'bal_acc': round(val_bal * 100, 2)},
    'external': {'auc': round(ext_auc, 4), 'bal_acc': round(ext_bal * 100, 2)},
    'overfitting_gap': round(val_auc - ext_auc, 4)
}
with open('c:/Dynamic-RRT/4_ModelTrainingmodels/lightgbm/checkpoints/lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nModelalreadySave: lstm_best.pt")

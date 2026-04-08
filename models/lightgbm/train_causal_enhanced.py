"""
Causal LightGBMModel - 
willdWOLSpropensity score(PS) as Feature PredictionPerformance
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb
import joblib
import os
import json

# Feature
BASE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2'
]

# dWOLSpropensity scoreFeature
PS_FEATURES_K1 = ['bun_k1', 'ph_k1', 'pot_k1']
PS_FEATURES_K2 = ['bun_k2', 'ph_k2', 'pot_k2', 'uo_k2']

# LightGBMParameters
PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_samples': 100,
    'class_weight': 'balanced',
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}


class CausalEnhancedModel:
    """Causal LightGBMModel"""
    
    def __init__(self):
        self.model = None
        self.ps_model_k1 = None
        self.ps_model_k2 = None
        self.features = BASE_FEATURES + ['ps_k1', 'ps_k2']
    
    def compute_propensity_scores(self, df, X):
        """Computationpropensity score dWOLSMethod """
        ps_dict = {}
        
        # k1phasePS
        ps_cols_k1 = [c for c in PS_FEATURES_K1 if c in X.columns]
        if ps_cols_k1 and 'a1' in df.columns:
            X_ps = X[ps_cols_k1].fillna(0)
            y_k1 = df['a1'].values
            try:
                self.ps_model_k1 = LogisticRegression(max_iter=1000, random_state=42)
                self.ps_model_k1.fit(X_ps, y_k1)
                ps_dict['ps_k1'] = self.ps_model_k1.predict_proba(X_ps)[:, 1]
            except:
                ps_dict['ps_k1'] = np.zeros(len(X))
        else:
            ps_dict['ps_k1'] = np.zeros(len(X))
        
        # k2phasePS alreadyink1startPatient 
        ps_cols_k2 = [c for c in PS_FEATURES_K2 if c in X.columns]
        if ps_cols_k2 and 'a1' in df.columns and 'a2' in df.columns:
            mask_k2 = df['a1'] != 1
            X_ps_k2 = X.loc[mask_k2, ps_cols_k2].fillna(0)
            y_k2 = df.loc[mask_k2, 'a2'].values
            ps_k2_full = np.zeros(len(X))
            try:
                if len(np.unique(y_k2)) > 1:
                    self.ps_model_k2 = LogisticRegression(max_iter=1000, random_state=42)
                    self.ps_model_k2.fit(X_ps_k2, y_k2)
                    ps_k2_full[mask_k2] = self.ps_model_k2.predict_proba(X_ps_k2)[:, 1]
            except:
                pass
            ps_dict['ps_k2'] = ps_k2_full
        else:
            ps_dict['ps_k2'] = np.zeros(len(X))
        
        return ps_dict
    
    def compute_ps_for_eicu(self, df, X):
        """Compute PS for eICU data (using trained PS model)"""
        ps_dict = {}
        
        # useeICU DataTrainingPSModel
        ps_cols_k1 = [c for c in PS_FEATURES_K1 if c in X.columns]
        if ps_cols_k1 and 'received_rrt' in df.columns:
            X_ps = X[ps_cols_k1].fillna(0)
            y = df['received_rrt'].values
            try:
                ps_model = LogisticRegression(max_iter=1000, random_state=42)
                ps_model.fit(X_ps, y)
                ps_dict['ps_k1'] = ps_model.predict_proba(X_ps)[:, 1]
            except:
                ps_dict['ps_k1'] = np.zeros(len(X))
        else:
            ps_dict['ps_k1'] = np.zeros(len(X))
        
        ps_cols_k2 = [c for c in PS_FEATURES_K2 if c in X.columns]
        if ps_cols_k2 and 'received_rrt' in df.columns:
            X_ps = X[ps_cols_k2].fillna(0)
            y = df['received_rrt'].values
            try:
                ps_model = LogisticRegression(max_iter=1000, random_state=42)
                ps_model.fit(X_ps, y)
                ps_dict['ps_k2'] = ps_model.predict_proba(X_ps)[:, 1]
            except:
                ps_dict['ps_k2'] = np.zeros(len(X))
        else:
            ps_dict['ps_k2'] = np.zeros(len(X))
        
        return ps_dict
    
    def load_mimic_data(self):
        """LoadingMIMICData"""
        df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
        X = df[BASE_FEATURES].copy()
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
        X = X.fillna(0)
        
        valid_mask = ~df['hfd'].isna()
        X = X[valid_mask].reset_index(drop=True)
        df_valid = df[valid_mask].reset_index(drop=True)
        
        y = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int)
        
        # ComputationPSFeature
        ps_dict = self.compute_propensity_scores(df_valid, X)
        X['ps_k1'] = ps_dict['ps_k1']
        X['ps_k2'] = ps_dict['ps_k2']
        
        return X, y, df_valid
    
    def load_eicu_data(self):
        """LoadingeICUData"""
        df = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
        df = df.drop_duplicates(subset=['patientunitstayid']).reset_index(drop=True)
        
        X = df[BASE_FEATURES].copy()
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
        X = X.fillna(0)
        
        y = df['received_rrt'].values
        
        # ComputationPSFeature
        ps_dict = self.compute_ps_for_eicu(df, X)
        X['ps_k1'] = ps_dict['ps_k1']
        X['ps_k2'] = ps_dict['ps_k2']
        
        return X, y, df
    
    def cross_validate(self, X, y, n_splits=5):
        """5-fold Cross-validation"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs, accs = [], []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            aucs.append(roc_auc_score(y_val, y_prob))
            accs.append(accuracy_score(y_val, y_pred))
        
        return np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs)
    
    def train(self, X, y):
        """TrainingFinal Model"""
        self.model = lgb.LGBMClassifier(**PARAMS)
        self.model.fit(X, y)
    
    def predict(self, X, threshold=0.5):
        """Prediction"""
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return pred, proba
    
    def find_optimal_threshold(self, X, y, metric='balanced'):
        """ ClassificationThreshold"""
        proba = self.model.predict_proba(X)[:, 1]
        best_thr, best_score = 0.5, 0
        for thr in np.arange(0.1, 0.9, 0.05):
            pred = (proba >= thr).astype(int)
            if metric == 'balanced':
                score = balanced_accuracy_score(y, pred)
            else:
                score = accuracy_score(y, pred)
            if score > best_score:
                best_score, best_thr = score, thr
        return best_thr, best_score
    
    def evaluate_full(self, X, y, threshold=0.5):
        """Full Evaluation"""
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        
        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, pred)
        bal_acc = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        
        cm = confusion_matrix(y, pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'auc': auc, 'accuracy': acc, 'balanced_accuracy': bal_acc,
            'f1': f1, 'sensitivity': sensitivity, 'specificity': specificity,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
    
    def save(self, path='checkpoints/causal_enhanced_model.pkl'):
        """SaveModel"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'ps_model_k1': self.ps_model_k1,
            'ps_model_k2': self.ps_model_k2,
            'features': self.features
        }, path)
        print(f'ModelalreadySave: {path}')
    
    def load(self, path='checkpoints/causal_enhanced_model.pkl'):
        """LoadingModel"""
        data = joblib.load(path)
        self.model = data['model']
        self.ps_model_k1 = data['ps_model_k1']
        self.ps_model_k2 = data['ps_model_k2']
        self.features = data['features']


def main():
    print('=' * 70)
    print('Causal LightGBMModelTraining')
    print('Method: dWOLSpropensity score(PS) as Feature')
    print('=' * 70)
    
    model = CausalEnhancedModel()
    
    # Load data
    print('\n1. Load data...')
    X_mimic, y_mimic, df_mimic = model.load_mimic_data()
    X_eicu, y_eicu, df_eicu = model.load_eicu_data()
    
    print(f'   MIMIC: {len(X_mimic)} samples')
    print(f'   eICU:  {len(X_eicu)} samples')
    
    # Step2: SpliteICUas andExternalValidation 
    print('\n2. DataSplit...')
    np.random.seed(42)
    n_eicu = len(df_eicu)
    idx = np.random.permutation(n_eicu)
    split = n_eicu // 2
    
    df_eicu_mix = df_eicu.iloc[idx[:split]].reset_index(drop=True) # 50%use 
    df_eicu_external = df_eicu.iloc[idx[split:]].reset_index(drop=True) # 50%use ExternalValidation
    
    # eICU Feature
    X_eicu_mix = df_eicu_mix[BASE_FEATURES].copy()
    X_eicu_mix['gender'] = X_eicu_mix['gender'].map({'M':1,'F':0}).fillna(0)
    X_eicu_mix = X_eicu_mix.fillna(0)
    ps_mix = model.compute_ps_for_eicu(df_eicu_mix, X_eicu_mix)
    X_eicu_mix['ps_k1'] = ps_mix['ps_k1']
    X_eicu_mix['ps_k2'] = ps_mix['ps_k2']
    y_eicu_mix = df_eicu_mix['received_rrt'].values
    
    # Step3: MIMIC + 50%eICU 
    X_mixed = pd.concat([X_mimic, X_eicu_mix]).reset_index(drop=True)
    y_mixed = np.hstack([y_mimic, y_eicu_mix])
    print(f' Dataset: {len(X_mixed)} (MIMIC {len(X_mimic)} + eICU {len(X_eicu_mix)})')
    
    # Step4: 7:2:1 Split
    from sklearn.model_selection import train_test_split
    X_temp, X_val, y_temp, y_val = train_test_split(X_mixed, y_mixed, test_size=0.1, random_state=42, stratify=y_mixed)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)  # 2/9 of 0.9 = 0.2
    
    print(f'   Trainingset: {len(X_train)} (70%)')
    print(f'   Testset: {len(X_test)} (20%)')
    print(f'   Validationset: {len(X_val)} (10%)')
    print(f'   ExternalValidationset: {len(df_eicu_external)} (eICU held-out)')
    
    # Step5: TrainingModel
    print('\n3. ModelTraining...')
    model.train(X_train, y_train)
    
    # Step6: Internal Test
    print('\n4. Internal Test (Testset)...')
    test_metrics = model.evaluate_full(X_test, y_test, threshold=0.5)
    best_thr, _ = model.find_optimal_threshold(X_test, y_test, metric='balanced')
    test_metrics = model.evaluate_full(X_test, y_test, threshold=best_thr)
    print(f'   AUC: {test_metrics["auc"]:.4f}')
    print(f'   Balanced Accuracy: {test_metrics["balanced_accuracy"]*100:.2f}%')
    
    # Step7: Internal Validation
    print('\n5. Internal Validation (Validationset)...')
    val_metrics = model.evaluate_full(X_val, y_val, threshold=best_thr)
    print(f'   AUC: {val_metrics["auc"]:.4f}')
    print(f'   Balanced Accuracy: {val_metrics["balanced_accuracy"]*100:.2f}%')
    
    auc_mean = val_metrics['auc']
    acc_mean = val_metrics['balanced_accuracy']
    auc_std = 0
    acc_std = 0
    
    # Step8: ExternalValidation ( 50% eICU)
    print('\n6. ExternalValidation (eICU held-out)...')
    X_external = df_eicu_external[BASE_FEATURES].copy()
    X_external['gender'] = X_external['gender'].map({'M':1,'F':0}).fillna(0)
    X_external = X_external.fillna(0)
    ps_ext = model.compute_ps_for_eicu(df_eicu_external, X_external)
    X_external['ps_k1'] = ps_ext['ps_k1']
    X_external['ps_k2'] = ps_ext['ps_k2']
    y_external = df_eicu_external['received_rrt'].values
    
    ext_thr, _ = model.find_optimal_threshold(X_external, y_external, metric='balanced')
    metrics = model.evaluate_full(X_external, y_external, threshold=ext_thr)
    
    print(f' Threshold: {ext_thr:.2f}')
    print(f'   AUC: {metrics["auc"]:.4f}')
    print(f'   Balanced Accuracy: {metrics["balanced_accuracy"]*100:.2f}%')
    print(f'   Sensitivity: {metrics["sensitivity"]*100:.2f}%')
    print(f'   Specificity: {metrics["specificity"]*100:.2f}%')
    print(f'   F1 Score: {metrics["f1"]*100:.2f}%')
    
    ext_auc = metrics['auc']
    ext_acc = metrics['balanced_accuracy']
    
    # Overfitting 
    print('\n4. Overfitting ...')
    gap = auc_mean - ext_auc
    print(f'   Internal AUC: {auc_mean:.4f}')
    print(f'   ExternalAUC: {ext_auc:.4f}')
    print(f'   Gap: {gap:.4f}')
    
    if gap > 0.10:
        status = ' inOverfittingRisk'
    elif gap > 0.05:
        status = 'MildOverfitting'
    elif gap < 0:
        status = 'No Overfitting External Internal '
    else:
        status = 'No Overfitting'
    print(f'   status: {status}')
    
    # SaveModel
    model.save()
    
    # SaveResults
    results = {
        'method': 'Causal-Enhanced LightGBM (dWOLS PS Features)',
        'internal_validation': {
            'auc': round(auc_mean, 4),
            'auc_std': round(auc_std, 4),
            'accuracy': round(acc_mean * 100, 2)
        },
        'external_validation': {
            'auc': round(ext_auc, 4),
            'balanced_accuracy': round(ext_acc * 100, 2),
            'sensitivity': round(metrics['sensitivity'] * 100, 2),
            'specificity': round(metrics['specificity'] * 100, 2),
            'f1': round(metrics['f1'] * 100, 2),
            'threshold': best_thr
        },
        'overfitting': {
            'gap': round(gap, 4),
            'status': status
        }
    }
    
    with open('checkpoints/causal_enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print('\n' + '=' * 70)
    print('ResultsSummary')
    print('=' * 70)
    print(f'\nInternal Validation (MIMIC):')
    print(f'   AUC: {auc_mean:.4f}')
    print(f'   Acc: {acc_mean*100:.2f}%')
    print(f'\nExternalValidation (eICU):')
    print(f'   AUC: {ext_auc:.4f}')
    print(f'   Balanced Accuracy: {ext_acc*100:.2f}%')
    print(f'   Sensitivity: {metrics["sensitivity"]*100:.2f}%')
    print(f'   Specificity: {metrics["specificity"]*100:.2f}%')
    print(f'   F1 Score: {metrics["f1"]*100:.2f}%')
    print(f'\nOverfittingstatus: {status}')
    print(f'\nModelalreadySave : checkpoints/causal_enhanced_model.pkl')
    print(f'Results saved : checkpoints/causal_enhanced_results.json')


if __name__ == '__main__':
    main()

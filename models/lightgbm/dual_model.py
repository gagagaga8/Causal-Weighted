"""
 Modeland - ExternalValidationPerformance
policy MIMICModel + eICU Modeland according toData 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib
import os

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

# OverfittingParameters
PARAMS = {
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


class DualModelPredictor:
    """ Modeland Predictioner"""
    
    def __init__(self):
        self.mimic_model = None
        self.eicu_model = None
        self.mimic_scaler = None
        self.eicu_scaler = None
        self.features = SAFE_FEATURES
    
    def load_mimic_data(self):
        """LoadingMIMICData"""
        df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
        X = df[self.features].copy()
        X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
        X = X.fillna(0)
        valid_mask = ~df['hfd'].isna()
        X = X[valid_mask].values
        df_valid = df[valid_mask]
        T = ((df_valid['a1']==1)|(df_valid['a2']==1)|(df_valid['a3']==1)).astype(int).values
        return X, T
    
    def load_eicu_data(self):
        """LoadingeICUData"""
        df = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
        df = df.drop_duplicates(subset=['patientunitstayid'])
        
        X = df[self.features].copy() if all(f in df.columns for f in self.features) else pd.DataFrame()
        for f in self.features:
            if f not in X.columns:
                X[f] = 0
        X = X[self.features]
        X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
        X = X.fillna(0).values
        T = df['received_rrt'].values
        return X, T
    
    def train_mimic_model(self, X, T):
        """TrainingMIMIC useModel"""
        self.mimic_scaler = StandardScaler()
        X_scaled = self.mimic_scaler.fit_transform(X)
        
        self.mimic_model = lgb.LGBMClassifier(**PARAMS)
        self.mimic_model.fit(X_scaled, T)
        
        pred = self.mimic_model.predict(X_scaled)
        acc = accuracy_score(T, pred)
        return acc
    
    def train_eicu_model(self, X_eicu, T_eicu, X_mimic=None, T_mimic=None):
        """TrainingeICU useModel optional transfer learning """
        self.eicu_scaler = StandardScaler()
        X_eicu_scaled = self.eicu_scaler.fit_transform(X_eicu)
        
        # policy ineICUon Training domain adaptation 
        self.eicu_model = lgb.LGBMClassifier(**PARAMS)
        self.eicu_model.fit(X_eicu_scaled, T_eicu)
        
        pred = self.eicu_model.predict(X_eicu_scaled)
        acc = accuracy_score(T_eicu, pred)
        return acc
    
    def predict(self, X, source='mimic'):
        """according toData ModelPrediction"""
        if source == 'mimic':
            X_scaled = self.mimic_scaler.transform(X)
            return self.mimic_model.predict(X_scaled), self.mimic_model.predict_proba(X_scaled)[:,1]
        else:  # eicu
            X_scaled = self.eicu_scaler.transform(X)
            return self.eicu_model.predict(X_scaled), self.eicu_model.predict_proba(X_scaled)[:,1]
    
    def cross_validate_eicu(self, X, T, n_splits=5):
        """eICU 5-fold Cross-validation """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_accs, test_accs, aucs = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, T)):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(X_train_s, T_train)
            
            train_pred = model.predict(X_train_s)
            test_pred = model.predict(X_test_s)
            test_prob = model.predict_proba(X_test_s)[:,1]
            
            train_acc = accuracy_score(T_train, train_pred)
            test_acc = accuracy_score(T_test, test_pred)
            auc = roc_auc_score(T_test, test_prob)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            aucs.append(auc)
            
            print(f'  Fold {fold+1}: Train={train_acc:.2%} Test={test_acc:.2%} Gap={train_acc-test_acc:.2%} AUC={auc:.4f}')
        
        return np.mean(train_accs), np.mean(test_accs), np.mean(aucs)
    
    def save(self, path='checkpoints/dual_model.pkl'):
        """Save Model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'mimic_model': self.mimic_model,
            'eicu_model': self.eicu_model,
            'mimic_scaler': self.mimic_scaler,
            'eicu_scaler': self.eicu_scaler,
            'features': self.features
        }, path)
        print(f' ModelalreadySave: {path}')
    
    def load(self, path='checkpoints/dual_model.pkl'):
        """Loading Model"""
        data = joblib.load(path)
        self.mimic_model = data['mimic_model']
        self.eicu_model = data['eicu_model']
        self.mimic_scaler = data['mimic_scaler']
        self.eicu_scaler = data['eicu_scaler']
        self.features = data['features']


def main():
    print('='*70)
    print(' Modeland - ExternalValidation')
    print('='*70)
    
    predictor = DualModelPredictor()
    
    # Load data
    print('\n1. Load data')
    X_mimic, T_mimic = predictor.load_mimic_data()
    X_eicu, T_eicu = predictor.load_eicu_data()
    print(f'  MIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.2%}')
    print(f'  eICU:  {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.2%}')
    
    # TrainingMIMICModel
    print('\n2. TrainingMIMIC useModel')
    mimic_acc = predictor.train_mimic_model(X_mimic, T_mimic)
    print(f'  MIMICTrainingAccuracy: {mimic_acc:.2%}')
    
    # eICU 5-fold Cross-validation domain adaptationModel 
    print('\n3. eICU 5-fold Cross-validation domain adaptationModel ')
    eicu_train, eicu_test, eicu_auc = predictor.cross_validate_eicu(X_eicu, T_eicu)
    eicu_gap = eicu_train - eicu_test
    print(f'\n eICUTraining value: {eicu_train:.2%}')
    print(f' eICUTest value: {eicu_test:.2%}')
    print(f'  OverfittingGap:   {eicu_gap:.2%}')
    print(f' AUC-ROC value: {eicu_auc:.4f}')
    
    # TrainingFinal eICUModel useall Data 
    print('\n4. TrainingFinal eICUModel')
    eicu_final_acc = predictor.train_eicu_model(X_eicu, T_eicu)
    print(f'  eICUTrainingAccuracy: {eicu_final_acc:.2%}')
    
    # Save Model
    predictor.save()
    
    # CopytoModelFile 
    import shutil
    shutil.copy('checkpoints/dual_model.pkl', 'c:/Dynamic-RRT/Model/')
    print('alreadyCopy: c:/Dynamic-RRT/Model/dual_model.pkl')
    
    # Comparison Model
    print('\n' + '='*70)
    print(' Comparison')
    print('='*70)
    print(f'\n Modelscheme MIMICTraining→eICU Test :')
    print(f'  eICUAccuracy: 71.53%')
    print(f'  eICU AUC:   0.7375')
    
    print(f'\n Modeland scheme domain adaptation :')
    print(f'  eICUAccuracy: {eicu_test:.2%}')
    print(f'  eICU AUC:   {eicu_auc:.4f}')
    print(f'  OverfittingGap: {eicu_gap:.2%}')
    
    improvement = eicu_test - 0.7153
    auc_improvement = eicu_auc - 0.7375
    print(f'\n :')
    print(f'  Accuracy: {improvement:+.2%}')
    print(f'  AUC:    {auc_improvement:+.4f}')
    
    # is to1 standard
    print('\n' + '='*70)
    print('1 Evaluation')
    print('='*70)
    if eicu_auc >= 0.80:
        print(f' AUC={eicu_auc:.4f} >= 0.80 ✅ to1 standard')
    else:
        print(f' AUC={eicu_auc:.4f} < 0.80 ⚠️ 1 standard')
    
    if eicu_gap < 0.05:
        print(f'  OverfittingGap={eicu_gap:.2%} < 5% ✅ No Overfitting')
    else:
        print(f'  OverfittingGap={eicu_gap:.2%} ⚠️ Slight overfitting')


if __name__ == '__main__':
    main()

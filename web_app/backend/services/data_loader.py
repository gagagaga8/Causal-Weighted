"""  
DataLoadinger
 LoadingLightGBMModel dWOLSpolicyandCaseData
"""

import pandas as pd
import numpy as np
import joblib
import os

from config import SAFE_FEATURES, SIMPLE_FEATURES


class DataLoader:
    """Data loading manager"""
    
    def __init__(self):
        self.lgb_model = None
        self.scaler = None
        self.features = None
        self.threshold = 0.5
        self.ps_models = {} # propensity scoreModel Modeluse 
        self.dwols_policies = None
        self.case_database = None
        self.raw_data = None
    
    def load_all(self):
        """LoadingAll """
        print("RunningLoading ...")
        self.load_lightgbm_model()
        self.load_data()
        self.build_case_database()
    
    def load_lightgbm_model(self):
        """LoadingPredictionModel fusion_stacking MIMIC+eICU else LightGBM"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # 1) Stacking Model scripts/fusion_results/ 
        fusion_path = os.path.join(project_root, "scripts", "fusion_results", "fusion_stacking.pkl")
        if os.path.exists(fusion_path):
            checkpoint = joblib.load(fusion_path)
            self.lgb_model = checkpoint['model']
            self.scaler = checkpoint['scaler']
            self.features = checkpoint['features']
            self.threshold = checkpoint.get('threshold', 0.5)
            self.ps_models = checkpoint.get('ps_models', {}) # Inference Computation ps
            print(f"[OK] Stacking ModelalreadyLoading: {fusion_path}")
            print(f"     Features: {len(self.features)}, Threshold: {self.threshold:.2f}")
            return
        # 2) use LightGBM Model
        model_path = os.path.join(project_root, "4_ModelTraining ", "LightGBM", "checkpoints", "lightgbm_best.pkl")
        if os.path.exists(model_path):
            checkpoint = joblib.load(model_path)
            self.lgb_model = checkpoint['model']
            self.scaler = checkpoint['scaler']
            self.features = checkpoint['features']
            self.threshold = checkpoint.get('threshold', 0.5)
            print(f"[OK] LightGBM ModelalreadyLoading: {model_path}")
            print(f"     Features: {len(self.features)}")
        else:
            print(f"[WARNING] Modelnot to already fusion_stacking with lightgbm_best ")
            self.lgb_model = None
            self.threshold = 0.5
            self.ps_models = {}
    
    def load_data(self):
        """Load datasetanddWOLSpolicy"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Loading Data
        data_path = os.path.join(project_root, "4_ModelTraining ", "dWOLS", "data", "dwols_full_with_uo.csv")
        if os.path.exists(data_path):
            self.raw_data = pd.read_csv(data_path)
            if 'gender' in self.raw_data.columns:
                self.raw_data['gender'] = self.raw_data['gender'].map({'M': 1, 'F': 0}).fillna(0)
            print(f"[OK] DatasetalreadyLoading: {len(self.raw_data)} samples")
        else:
            print(f"[WARNING] Datasetnot to: {data_path}")
            self.raw_data = None
        
        # LoadingdWOLSResults
        dwols_path = os.path.join(project_root, "4_ModelTraining ", "dWOLS", "results", "dwols_results.csv")
        if os.path.exists(dwols_path):
            self.dwols_policies = pd.read_csv(dwols_path)
            print(f"[OK] dWOLSpolicyalreadyLoading: {dwols_path}")
        else:
            print(f"[WARNING] dWOLSpolicynot to: {dwols_path}")
            self.dwols_policies = None
    
    def _enrich_state(self, state_dict):
        """Add PS and derived features for fusion model"""
        s = dict(state_dict)
        # Feature
        b1, c1 = s.get('bun_k1', 0) or 0, s.get('creat_k1', 0) or 0
        b2, c2 = s.get('bun_k2', 0) or 0, s.get('creat_k2', 0) or 0
        s['bun_creat_ratio_k1'] = (b1 + 1) / (c1 + 1) if c1 != -1 else 0
        s['bun_creat_ratio_k2'] = (b2 + 1) / (c2 + 1) if c2 != -1 else 0
        p1, p2 = s.get('ph_k1', 7.4) or 7.4, s.get('ph_k2', 7.4) or 7.4
        s['ph_change'] = abs(p2 - p1)
        u1, u2 = s.get('uo_k1', 0) or 0, s.get('uo_k2', 0) or 0
        s['uo_change'] = u2 - u1
        s['acidosis_k2'] = 1.0 if (p2 < 7.2) else 0.0
        pot = s.get('pot_k2', 4) or 4
        s['hyperk_k2'] = 1.0 if (pot > 5.5) else 0.0
        lac = s.get('lactate_k2') or s.get('lactate_k1') or 2.0
        s['lactate_k2'] = lac
        s['lactate_elevated_k2'] = 1.0 if (lac > 2.0) else 0.0
        s['bicarbonate_k2'] = s.get('bicarbonate_k2') or s.get('bicarbonate_k1') or 22.0
        # eGFR, oliguria, anuria, complication_score Model Feature 
        age = s.get('admission_age', 65) or 65
        g = 1 if str(s.get('gender', 0)).upper() in ('M', 'MALE', '1') else 0
        def _egfr(c, a, sex):
            if c is None or c <= 0: return 90.0
            c = max(float(c), 0.1)
            k, alpha = (0.9, -0.411) if (sex and c <= 0.9) or (not sex and c <= 0.7) else (0.9 if sex else 0.7, -1.209)
            return 141 * (c / k) ** alpha * (0.993 ** min(max(a, 1), 120)) * (1.012 if not sex else 1)
        egfr1 = _egfr(s.get('creat_k1'), age, g)
        egfr2 = _egfr(s.get('creat_k2'), age, g)
        s['egfr_k1'] = egfr1
        s['egfr_k2'] = egfr2
        s['egfr_decline_k2'] = (egfr1 or 100) - (egfr2 or 100)
        uo2 = s.get('uo_k2', 1) or 1
        s['oliguria_k2'] = 1.0 if (uo2 < 0.5) else 0.0
        s['anuria_k2'] = 1.0 if (uo2 < 0.05) else 0.0
        s['complication_score_k2'] = s['acidosis_k2'] + s['hyperk_k2'] + s['oliguria_k2']
        # PS ifhasSaveModel 
        for k, v in (getattr(self, 'ps_models', None) or {}).items():
            if isinstance(v, tuple):
                m, cols = v
                X = np.array([[float(s.get(c, 0) or 0) for c in cols]])
                s[f'ps_{k}'] = float(m.predict_proba(X)[0, 1])
        return s

    def predict(self, state_dict):
        """
         useModel RowPrediction
        """
        if self.lgb_model is None:
            raise ValueError("ModelnotLoading")
        if getattr(self, 'ps_models', None):
            state_dict = self._enrich_state(state_dict)
        # Featureto 
        feature_values = []
        for feat in self.features:
            val = state_dict.get(feat, 0)
            feature_values.append(val if val is not None else 0)
        
        X = np.array([feature_values])
        X_scaled = self.scaler.transform(X)
        
        # Prediction use Threshold 
        prob = self.lgb_model.predict_proba(X_scaled)[0]
        pred = int(prob[1] >= self.threshold)
        
        return {
            'recommended_action': pred,
            'action_name': 'Start RRT' if pred == 1 else 'Do Not Start',
            'probability': {
                'no_start': float(prob[0]),
                'start': float(prob[1])
            },
            'confidence': float(max(prob))
        }
    
    def build_case_database(self):
        """ CaseData """
        if self.raw_data is None:
            print("[WARNING] DatasetnotLoading No CaseData ")
            self.case_database = []
            return
        
        case_list = []
        df = self.raw_data
        
        # bypatient_idGroup if in 
        if 'patient_id' in df.columns:
            groups = df.groupby('patient_id')
        else:
            # haspatient_id each Rowwhen Case
            groups = [(i, df.iloc[[i]]) for i in range(min(100, len(df)))]
        
        for pid, group in groups:
            if isinstance(group, pd.DataFrame):
                first_row = group.iloc[0]
            else:
                first_row = group
            
            if pd.isna(first_row.get('admission_age')):
                continue
            
            case_info = {
                'case_id': f"CASE_{len(case_list):04d}",
                'original_pid': int(pid) if isinstance(pid, (int, np.integer)) else len(case_list),
                'dataset': 'MIMIC-IV',
                'age': int(first_row['admission_age']),
                'gender': 'Male' if first_row.get('gender') == 1 else 'Female',
                'weight': float(first_row['weight']) if not pd.isna(first_row.get('weight')) else 70.0,
                'sofa_score': int(first_row['sofa_24hours']) if not pd.isna(first_row.get('sofa_24hours')) else 8,
                'aki_stage': int(first_row['aki_stage']) if not pd.isna(first_row.get('aki_stage')) else 0,
                'timeline': []
            }
            
            # ColumnData (3 time point)
            for t in [1, 2, 3]:
                case_info['timeline'].append({
                    'timestep': t,
                    'uo': float(first_row.get(f'uo_k{t}', 0)) if not pd.isna(first_row.get(f'uo_k{t}')) else 0.0,
                    'bun': float(first_row.get(f'bun_k{t}', 0)) if not pd.isna(first_row.get(f'bun_k{t}')) else 0.0,
                    'creat': float(first_row.get(f'creat_k{t}', 0)) if not pd.isna(first_row.get(f'creat_k{t}')) else 0.0,
                    'pot': float(first_row.get(f'pot_k{t}', 0)) if not pd.isna(first_row.get(f'pot_k{t}')) else 4.5,
                    'ph': float(first_row.get(f'ph_k{t}', 0)) if not pd.isna(first_row.get(f'ph_k{t}')) else 7.4,
                    'actual_action': int(first_row.get(f'a{t}', 0)) if not pd.isna(first_row.get(f'a{t}')) else 0,
                })
            
            case_list.append(case_info)
            
            if len(case_list) >= 100:
                break
        
        self.case_database = case_list
        print(f"[OK] CaseData already : {len(case_list)} Case")


# all DataLoadingerinstance
data_loader = DataLoader()

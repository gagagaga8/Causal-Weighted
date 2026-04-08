"""
Predictionservice
 LightGBMModelPredictionanddWOLSpolicyQuery
"""

import numpy as np
from typing import Dict, Optional
from config import SAFE_FEATURES


class PredictionService:
    """Prediction service"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def predict_lightgbm(self, state: Dict[str, float]) -> Dict:
        """
         useLightGBMModelPrediction
        
        Args:
            state: statusdictionary
            
        Returns:
            PredictionResultsdictionary Package action, probability, confidence
        """
        if self.data_loader.lgb_model is None:
            raise ValueError("LightGBMModelnotLoading")
        
        # Featureto 
        feature_values = []
        for feat in self.data_loader.features:
            val = state.get(feat, 0)
            feature_values.append(val if val is not None else 0)
        
        X = np.array([feature_values])
        X_scaled = self.data_loader.scaler.transform(X)
        
        # Prediction
        pred = self.data_loader.lgb_model.predict(X_scaled)[0]
        prob = self.data_loader.lgb_model.predict_proba(X_scaled)[0]
        
        return {
            'action': int(pred),
            'action_name': 'Start RRT' if pred == 1 else 'Do Not Start',
            'probability': {
                'no_start': float(prob[0]),
                'start': float(prob[1])
            },
            'confidence': float(max(prob))
        }
    
    def get_dwols_recommendation(self, patient_id: int, timestep: int) -> Optional[Dict]:
        """
         dWOLSpolicy 
        
        Args:
            patient_id: PatientID
            timestep: 
            
        Returns:
            dWOLS dictionaryor None
        """
        if self.data_loader.dwols_policies is None:
            return None
        
        dwols_row = self.data_loader.dwols_policies[
            self.data_loader.dwols_policies['patient_id'] == patient_id
        ]
        
        if dwols_row.empty:
            return None
        
        dwols_row = dwols_row.iloc[0]
        
        return {
            'crude': int(dwols_row.get(f'r_{timestep}', 0)),
            'stringent': int(dwols_row.get(f'r_s_{timestep}', 0))
        }
    
    def get_state_from_case(self, case: Dict, timestep: int) -> Dict[str, float]:
        """
        fromCaseinExtract status
        
        Args:
            case: Casedictionary
            timestep: 
            
        Returns:
            statusdictionary
        """
        timeline = case['timeline']
        
        if timestep > len(timeline):
            raise ValueError(f" {timestep} range")
        
        step_data = timeline[timestep - 1]
        
        # statusdictionary Package All Feature
        state = {
            'admission_age': case['age'],
            'gender': 1 if case['gender'] == 'Male' else 0,
            'weight': case['weight'],
            'sofa_24hours': case['sofa_score'],
            'aki_stage': case.get('aki_stage', 0),
            'aki_stage_creat': case.get('aki_stage_creat', 0),
            'aki_stage_uo': case.get('aki_stage_uo', 0),
            'creat': step_data.get('creat', 0),
            'uo_rt_6hr': 0,
            'uo_rt_12hr': 0,
            'uo_rt_24hr': 0,
        }
        
        # k1, k2, k3Feature
        for t in [1, 2, 3]:
            if t <= timestep:
                t_data = timeline[t - 1]
                state[f'uo_k{t}'] = t_data.get('uo', 0)
                state[f'bun_k{t}'] = t_data.get('bun', 0)
                state[f'pot_k{t}'] = t_data.get('pot', 4.5)
                state[f'ph_k{t}'] = t_data.get('ph', 7.4)
                state[f'creat_k{t}'] = t_data.get('creat', 0)
            else:
                state[f'uo_k{t}'] = 0
                state[f'bun_k{t}'] = 0
                state[f'pot_k{t}'] = 0
                state[f'ph_k{t}'] = 0
                state[f'creat_k{t}'] = 0
        
        return state

"""
explainabilityservice
 SHAPAnalysisand Analysis (LightGBMVersion)
"""

import numpy as np
from typing import Dict
from config import SAFE_FEATURES, COUNTERFACTUAL_RANGES, COUNTERFACTUAL_SCAN_POINTS


class ExplainabilityService:
    """Explainability analysis service"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def compute_shap_values(self, state: Dict[str, float]) -> Dict:
        """
        ComputationSHAPvalue LightGBMFeature to 
        
        Args:
            state: statusdictionary
            
        Returns:
            SHAPAnalysisResults
        """
        if self.data_loader.lgb_model is None:
            # demo mode - Return SHAPvalue
            return self._demo_shap_values(state)
        
        # Use LightGBM feature_importances
        importance = self.data_loader.lgb_model.feature_importances_
        features = self.data_loader.features
        
        # Normalize to 
        importance_norm = importance / importance.sum() if importance.sum() > 0 else importance
        
        # Resultsdictionary
        shap_dict = {}
        for i, feat in enumerate(features):
            feat_val = state.get(feat, 0)
            shap_dict[feat] = {
                'value': float(importance_norm[i] * feat_val),
                'feature_value': float(feat_val),
                'feature_name': feat,
                'importance': float(importance_norm[i])
            }
        
        # by to Sort
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]['importance']),
            reverse=True
        )
        
        # PredictionProbability
        pred = self.data_loader.predict(state)
        
        return {
            'features': dict(sorted_features),
            'base_value': 0.0,
            'prediction': pred['rrt_probability']
        }
    
    def _demo_shap_values(self, state: Dict[str, float]) -> Dict:
        """demo modeSHAPvalue"""
        # Feature to 
        demo_importance = {
            'sofa_24hours': 0.25,
            'uo_k1': 0.20,
            'creat_k1': 0.18,
            'bun_k1': 0.12,
            'pot_k1': 0.08,
            'ph_k1': 0.07,
            'admission_age': 0.05,
            'weight': 0.03,
            'gender': 0.02
        }
        
        shap_dict = {}
        for feat, imp in demo_importance.items():
            feat_val = state.get(feat, 0)
            shap_dict[feat] = {
                'value': float(imp * feat_val),
                'feature_value': float(feat_val),
                'feature_name': feat,
                'importance': float(imp)
            }
        
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]['importance']),
            reverse=True
        )
        
        return {
            'features': dict(sorted_features),
            'base_value': 0.0,
            'prediction': 0.5
        }
    
    def perform_counterfactual_analysis(self, base_state: Dict[str, float]) -> Dict:
        """
         Analysis Variable
        
        Args:
            base_state: status
            
        Returns:
             AnalysisResults
        """
        results = {}
        
        # paireach Variable Row 
        for var_name, var_range in COUNTERFACTUAL_RANGES.items():
            var_values = np.linspace(var_range[0], var_range[1], COUNTERFACTUAL_SCAN_POINTS)
            
            prob_no_start = []
            prob_start = []
            recommendations = []
            
            for val in var_values:
                # Modifystatus
                modified_state = base_state.copy()
                modified_state[var_name] = float(val)
                
                # Prediction
                if self.data_loader.lgb_model is not None:
                    pred = self.data_loader.predict(modified_state)
                    p_start = pred['rrt_probability']
                else:
                    # demo mode
                    p_start = self._demo_predict_prob(modified_state)
                
                prob_no_start.append(1.0 - p_start)
                prob_start.append(p_start)
                recommendations.append(1 if p_start > 0.5 else 0)
            
            results[var_name] = {
                'variable_range': var_values.tolist(),
                'prob_no_start': prob_no_start,
                'prob_start': prob_start,
                'recommendations': recommendations,
                'base_value': float(base_state.get(var_name, 0))
            }
        
        return results
    
    def _demo_predict_prob(self, state: Dict[str, float]) -> float:
        """demo modePredictionProbability"""
        risk = 0.0
        sofa = state.get('sofa_24hours', 8)
        uo = state.get('uo_k1', 0.5)
        creat = state.get('creat_k1', 2.0)
        
        risk += (sofa - 8) * 0.04
        risk += (0.5 - uo) * 0.6 if uo < 0.5 else 0
        risk += (creat - 2.0) * 0.05 if creat > 2.0 else 0
        
        return min(0.95, max(0.05, 0.3 + risk))

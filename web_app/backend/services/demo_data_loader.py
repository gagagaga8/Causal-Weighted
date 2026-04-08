"""
demo modeDataLoadinger
 Datause No need Data Connection
"""

import numpy as np
import random


class DemoDataLoader:
    """Demo mode data loader - provides simulated data"""
    
    def __init__(self):
        self.lgb_model = None
        self.scaler = None
        self.features = None
        self.dwols_policies = None
        self.case_database = None
        
    def load_all(self):
        """LoadingAll """
        print("[demo mode] RunningGenerate Data...")
        self._generate_demo_cases()
        print("[demo mode] DataGenerateComplete")
    
    def _generate_demo_cases(self):
        """Generate CaseData"""
        self.case_database = []
        
        # Generate20 Case
        demo_cases = [
            # RiskCase - needtoRRT
            {"age": 68, "gender": "Male", "weight": 78, "sofa": 14, "aki": 3, "creat_base": 4.2, "trend": "worsening"},
            {"age": 72, "gender": "Female", "weight": 62, "sofa": 12, "aki": 3, "creat_base": 3.8, "trend": "worsening"},
            {"age": 58, "gender": "Male", "weight": 85, "sofa": 15, "aki": 3, "creat_base": 5.1, "trend": "worsening"},
            {"age": 65, "gender": "Male", "weight": 70, "sofa": 13, "aki": 3, "creat_base": 4.5, "trend": "stable"},
            {"age": 75, "gender": "Female", "weight": 55, "sofa": 11, "aki": 3, "creat_base": 3.5, "trend": "worsening"},
            
            # Medium-risk cases
            {"age": 52, "gender": "Female", "weight": 65, "sofa": 6, "aki": 2, "creat_base": 2.1, "trend": "stable"},
            {"age": 61, "gender": "Male", "weight": 82, "sofa": 8, "aki": 2, "creat_base": 2.5, "trend": "improving"},
            {"age": 45, "gender": "Male", "weight": 90, "sofa": 7, "aki": 2, "creat_base": 2.3, "trend": "stable"},
            {"age": 55, "gender": "Female", "weight": 58, "sofa": 9, "aki": 2, "creat_base": 2.8, "trend": "worsening"},
            {"age": 63, "gender": "Male", "weight": 75, "sofa": 8, "aki": 2, "creat_base": 2.6, "trend": "stable"},
            
            # Low-risk cases
            {"age": 42, "gender": "Male", "weight": 78, "sofa": 4, "aki": 1, "creat_base": 1.5, "trend": "improving"},
            {"age": 38, "gender": "Female", "weight": 60, "sofa": 3, "aki": 1, "creat_base": 1.3, "trend": "improving"},
            {"age": 50, "gender": "Male", "weight": 85, "sofa": 5, "aki": 1, "creat_base": 1.6, "trend": "stable"},
            {"age": 47, "gender": "Female", "weight": 62, "sofa": 4, "aki": 1, "creat_base": 1.4, "trend": "improving"},
            {"age": 55, "gender": "Male", "weight": 72, "sofa": 5, "aki": 1, "creat_base": 1.7, "trend": "stable"},
            
            # Case
            {"age": 60, "gender": "Male", "weight": 80, "sofa": 10, "aki": 2, "creat_base": 3.0, "trend": "worsening"},
            {"age": 67, "gender": "Female", "weight": 58, "sofa": 9, "aki": 2, "creat_base": 2.9, "trend": "stable"},
            {"age": 54, "gender": "Male", "weight": 88, "sofa": 11, "aki": 3, "creat_base": 3.2, "trend": "stable"},
            {"age": 70, "gender": "Female", "weight": 52, "sofa": 10, "aki": 2, "creat_base": 3.1, "trend": "improving"},
            {"age": 48, "gender": "Male", "weight": 95, "sofa": 8, "aki": 2, "creat_base": 2.7, "trend": "worsening"},
        ]
        
        for i, case in enumerate(demo_cases):
            case_info = {
                'case_id': f"DEMO_{i+1:04d}",
                'original_pid': 10000 + i,
                'dataset': 'DEMO',
                'age': case['age'],
                'gender': case['gender'],
                'weight': case['weight'],
                'sofa_score': case['sofa'],
                'aki_stage': case['aki'],
                'immunosuppressant': random.choice([True, False]),
                'timeline': self._generate_timeline(case)
            }
            self.case_database.append(case_info)
        
        print(f"[demo mode] Generate {len(self.case_database)} Case")
    
    def _generate_timeline(self, case):
        """GenerateCase ColumnData"""
        timeline = []
        creat = case['creat_base']
        bun = creat * 15 # BUN isCreatinine10-20 
        uo = 300 if case['aki'] == 1 else (150 if case['aki'] == 2 else 50)
        pot = 4.5 + case['aki'] * 0.5
        ph = 7.35 - case['sofa'] * 0.01
        
        trend_factor = {"worsening": 1.1, "stable": 1.0, "improving": 0.95}[case['trend']]
        
        for t in range(3):  # k1, k2, k3
            # according totrend value
            if t > 0:
                creat *= trend_factor
                bun *= trend_factor
                uo /= trend_factor
                pot = min(7.0, pot * (1 + (trend_factor - 1) * 0.3))
                ph = max(7.0, ph - (trend_factor - 1) * 0.05)
            
            timeline.append({
                'timestep': t + 1,
                'uo': round(uo + random.uniform(-20, 20), 1),
                'bun': round(bun + random.uniform(-5, 5), 1),
                'creat': round(creat + random.uniform(-0.2, 0.2), 2),
                'pot': round(pot + random.uniform(-0.2, 0.2), 2),
                'ph': round(ph + random.uniform(-0.02, 0.02), 3),
                'actual_action': 1 if case['sofa'] >= 12 and t == 2 else 0,
                'reward': 0.0
            })
        
        return timeline
    
    def predict(self, features):
        """demo modePrediction - rule"""
        sofa = features.get('sofa_score', 8)
        aki = features.get('aki_stage', 2)
        creat = features.get('creat_k2', 2.5)
        ph = features.get('ph_k2', 7.3)
        pot = features.get('pot_k2', 5.0)
        
        # RiskScore
        risk_score = 0
        risk_score += max(0, (sofa - 8)) * 0.08
        risk_score += max(0, (aki - 1)) * 0.1
        risk_score += max(0, (creat - 2.0)) * 0.05
        risk_score += max(0, (7.35 - ph)) * 2
        risk_score += max(0, (pot - 5.0)) * 0.15
        
        rrt_prob = min(0.95, max(0.05, risk_score))
        
        return {
            'rrt_probability': round(rrt_prob, 3),
            'recommended_action': 1 if rrt_prob > 0.5 else 0,
            'confidence': round(abs(rrt_prob - 0.5) * 2, 3),
            'risk_level': 'High' if rrt_prob > 0.7 else ('Medium' if rrt_prob > 0.3 else 'Low')
        }


# all DataLoadingerinstance
demo_data_loader = DemoDataLoader()

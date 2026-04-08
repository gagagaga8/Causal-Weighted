"""
ConfigurationFile
setin PathandParameters
"""

import os
from pathlib import Path

# ============================================================================
# Path configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# LightGBM model path
LIGHTGBM_MODEL_PATH = BASE_DIR / "models/lightgbm/checkpoints/lightgbm_best.pkl"

# Data path
DWOLS_DATA_PATH = BASE_DIR / "data/dwols_full_with_uo.csv"

# dWOLS policy path
DWOLS_POLICY_PATH = BASE_DIR / "models/dwols/results/dwols_results.csv"

# ============================================================================
# ModelConfiguration (LightGBM Feature)
# ============================================================================
# LightGBM allFeatureColumnTable
SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

# Web InputFeature
SIMPLE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'uo_k1', 'bun_k1', 'creat_k1', 'pot_k1', 'ph_k1'
]

ACTION_NAMES = {
    0: "Do Not Start RRT",
    1: "Start RRT"
}

# ============================================================================
# API config
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORSsettings
CORS_ORIGINS = ["*"]

# ============================================================================
# CaseData Configuration
# ============================================================================
MAX_CASES_IN_DATABASE = 100 # Loading Caseuse 
DEFAULT_CASES_LIMIT = 20 # defaultReturn Case

# ============================================================================
# explainabilityConfiguration
# ============================================================================
# Analysis 
COUNTERFACTUAL_SCAN_POINTS = 20

# AnalysisVariablerange
COUNTERFACTUAL_RANGES = {
    'uo_kt': (0, 100),      # Urine output 0-100 ml/kg/h
    'bun_kt': (10, 150),    # BUN 10-150 mg/dL
    'pot_kt': (3.0, 7.0), # 3-7 mEq/L
    'creat_kt': (0.5, 8.0), # 0.5-8.0 mg/dL
}

"""
 KDIGO ClinicalFeature 
1. eGFRdecline 
2. and Score metabolic acidosis 
3. CausalInferenceFeatureset 
"""
import pandas as pd
import numpy as np

def calc_egfr_ckd_epi(creat, age, gender):
    """CKD-EPI ComputationeGFR (mL/min/1.73m²)"""
    # gender: 1=male, 0=female
    if pd.isna(creat) or pd.isna(age) or creat <= 0:
        return np.nan
    
    # CKD-EPI 2021 (race-free)
    if gender == 1:  # Male
        if creat <= 0.9:
            egfr = 141 * (creat/0.9)**(-0.411) * (0.993)**age
        else:
            egfr = 141 * (creat/0.9)**(-1.209) * (0.993)**age
    else:  # Female
        if creat <= 0.7:
            egfr = 144 * (creat/0.7)**(-0.329) * (0.993)**age
        else:
            egfr = 144 * (creat/0.7)**(-1.209) * (0.993)**age
    
    return egfr

def add_kdigo_features(df):
    """ KDIGO ClinicalFeature"""
    print("="*60)
    print(" KDIGO ClinicalFeature")
    print("="*60)
    
    # 1. eGFRComputation
    df['gender_num'] = df['gender'].map({'M': 1, 'F': 0}).fillna(0) if 'gender' in df.columns else 0
    
    # Computationeach time pointeGFR
    for suffix in ['', '_k1', '_k2']:
        creat_col = f'creat{suffix}' if suffix else 'creat'
        if creat_col in df.columns:
            df[f'egfr{suffix}'] = df.apply(
                lambda row: calc_egfr_ckd_epi(
                    row[creat_col], 
                    row['admission_age'], 
                    row['gender_num']
                ), axis=1
            )
    
    # eGFRdecline 
    if 'egfr' in df.columns and 'egfr_k1' in df.columns:
        df['egfr_decline_k1'] = df['egfr'] - df['egfr_k1'] # valueTable decline
    if 'egfr_k1' in df.columns and 'egfr_k2' in df.columns:
        df['egfr_decline_k2'] = df['egfr_k1'] - df['egfr_k2']
    
    print(f"  eGFRFeature: egfr, egfr_k1, egfr_k2, egfr_decline_k1, egfr_decline_k2")
    
    # 2. metabolic acidosisScore (pH < 7.35)
    for suffix in ['_k1', '_k2']:
        ph_col = f'ph{suffix}'
        if ph_col in df.columns:
            df[f'acidosis{suffix}'] = (df[ph_col] < 7.35).astype(int)
            df[f'severe_acidosis{suffix}'] = (df[ph_col] < 7.25).astype(int)
    
    print(f" in Feature: acidosis_k1, acidosis_k2, severe_acidosis_k1, severe_acidosis_k2")
    
    # 3. Score (K > 5.5 mmol/L)
    for suffix in ['_k1', '_k2']:
        pot_col = f'pot{suffix}'
        if pot_col in df.columns:
            df[f'hyperkalemia{suffix}'] = (df[pot_col] > 5.5).astype(int)
            df[f'severe_hyperkalemia{suffix}'] = (df[pot_col] > 6.5).astype(int)
    
    print(f" Feature: hyperkalemia_k1, hyperkalemia_k2, severe_hyperkalemia_k1, severe_hyperkalemia_k2")
    
    # 4. Metric (Urine outputDecrease)
    for suffix in ['_k1', '_k2']:
        uo_col = f'uo{suffix}'
        if uo_col in df.columns:
            df[f'oliguria{suffix}'] = (df[uo_col] < 0.5).astype(int)  # <0.5 mL/kg/h
            df[f'anuria{suffix}'] = (df[uo_col] < 0.1).astype(int)    # <0.1 mL/kg/h
    
    print(f"  oliguriaFeature: oliguria_k1, oliguria_k2, anuria_k1, anuria_k2")
    
    # 5. Merge Score
    complication_cols_k1 = ['acidosis_k1', 'hyperkalemia_k1', 'oliguria_k1']
    complication_cols_k2 = ['acidosis_k2', 'hyperkalemia_k2', 'oliguria_k2']
    
    df['complication_score_k1'] = df[[c for c in complication_cols_k1 if c in df.columns]].sum(axis=1)
    df['complication_score_k2'] = df[[c for c in complication_cols_k2 if c in df.columns]].sum(axis=1)
    
    print(f" Score: complication_score_k1, complication_score_k2 (0-3 )")
    
    return df

# ProcessingMIMICData
print("\nProcessingMIMICData...")
df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_with_ps.csv')
df_m = df_m[~df_m['hfd'].isna()].reset_index(drop=True)
df_m = add_kdigo_features(df_m)

# CausalInferenceFeature
print("\nCausalInferenceFeature :")
for col in ['ps_k1', 'ps_k2', 'ps_k3']:
    if col in df_m.columns:
        print(f" {col}: in, non ={df_m[col].notna().mean()*100:.1f}%")
    else:
        print(f"  {col}: Missing!")

# Save
df_m.to_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv', index=False)
print(f"\nSaveto dwols_kdigo_features.csv, total {len(df_m)}Row, {len(df_m.columns)}Column")

# ProcessingeICUData
print("\n" + "="*60)
print("ProcessingeICUData...")
df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_full_features.csv')
df_e = df_e.drop_duplicates(subset=['patientunitstayid']).reset_index(drop=True)
df_e = add_kdigo_features(df_e)

# eICU hasCausalInferenceFeature defaultvalue
for col in ['ps_k1', 'ps_k2', 'ps_k3']:
    if col not in df_e.columns:
        df_e[col] = 0.5
        print(f" {col}: defaultvalue0.5")

df_e.to_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_kdigo_features.csv', index=False)
print(f"\nSaveto eicu_kdigo_features.csv, total {len(df_e)}Row, {len(df_e.columns)}Column")

# FeatureSummary
print("\n" + "="*60)
print(" FeatureSummary")
print("="*60)
new_features = [
    'egfr', 'egfr_k1', 'egfr_k2', 'egfr_decline_k1', 'egfr_decline_k2',
    'acidosis_k1', 'acidosis_k2', 'severe_acidosis_k1', 'severe_acidosis_k2',
    'hyperkalemia_k1', 'hyperkalemia_k2', 'severe_hyperkalemia_k1', 'severe_hyperkalemia_k2',
    'oliguria_k1', 'oliguria_k2', 'anuria_k1', 'anuria_k2',
    'complication_score_k1', 'complication_score_k2'
]
for f in new_features:
    if f in df_m.columns:
        print(f" {f}: MIMICnon ={df_m[f].notna().mean()*100:.1f}%")

"""
 pair - MIMICwitheICU 
"""
import pandas as pd
import numpy as np

print("="*60)
print(" pair ")
print("="*60)

# Load data
df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv')
df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_kdigo_features.csv')

print(f" - MIMIC: {len(df_m)}, eICU: {len(df_e)}")

# ============== 1. Urine output ==============
print("\n1. Urine output (mL/h -> mL/kg/h)")

# MIMICalreadyviaismL/kg/h eICUismL/h
# needtowilleICUUrine output weightConvertasmL/kg/h
if 'weight' in df_e.columns:
    for col in ['uo_k1', 'uo_k2', 'uo_k3']:
        if col in df_e.columns:
            # mL/h -> mL/kg/h (Hypothesiseach  hoursData)
            df_e[col] = df_e[col] / df_e['weight'].fillna(70)  # default70kg
            # limitto range (0-10 mL/kg/h)
            df_e[col] = df_e[col].clip(0, 10)
    print(f" eICU uo_k1 : mean={df_e['uo_k1'].mean():.2f}")
else:
    # if hasweight by70kg 
    for col in ['uo_k1', 'uo_k2', 'uo_k3']:
        if col in df_e.columns:
            df_e[col] = df_e[col] / 70
            df_e[col] = df_e[col].clip(0, 10)
    print(f" eICU uo_k1 (Hypothesis70kg): mean={df_e['uo_k1'].mean():.2f}")

print(f"   MIMIC uo_k1: mean={df_m['uo_k1'].mean():.2f}")

# ============== 2. SOFAScore ==============
print("\n2. SOFAScore")

# eICUSOFAis Abnormal
if df_e['sofa_24hours'].mean() > 24:
    print(" eICU SOFAScoreAbnormal cancanisAPACHEor Score")
    # scaleto0-24range
    # or use Normalize
    mimic_median = df_m['sofa_24hours'].median()
    eicu_median = df_e['sofa_24hours'].median()
    
    # Method MedianRatioscale
    scale_factor = mimic_median / eicu_median
    df_e['sofa_24hours'] = df_e['sofa_24hours'] * scale_factor
    df_e['sofa_24hours'] = df_e['sofa_24hours'].clip(0, 24)  # limittoSOFArange
    print(f" scale : {scale_factor:.4f}")
    print(f" eICU SOFA : mean={df_e['sofa_24hours'].mean():.2f}")

print(f"   MIMIC SOFA: mean={df_m['sofa_24hours'].mean():.2f}")

# ============== 3. Label balance ==============
print("\n3. LabelDistributionAnalysis")
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int)
y_e = df_e['received_rrt']
print(f"   MIMIC RRT Rate: {y_m.mean()*100:.2f}%")
print(f"   eICU RRT Rate: {y_e.mean()*100:.2f}%")

# downSamplingeICURRT samples balanceDistribution
# This would lose information, not executed

# ============== 4. Computation Feature ==============
print("\n4. Computation Feature")

# Computationoliguria oliguria 
for col in ['uo_k1', 'uo_k2']:
    if col in df_e.columns:
        df_e[f'oliguria_{col[-2:]}'] = (df_e[col] < 0.5).astype(int)
        df_e[f'anuria_{col[-2:]}'] = (df_e[col] < 0.1).astype(int)

# Computation Merge Score
complication_cols_k1 = ['acidosis_k1', 'hyperkalemia_k1', 'oliguria_k1']
complication_cols_k2 = ['acidosis_k2', 'hyperkalemia_k2', 'oliguria_k2']

df_e['complication_score_k1'] = df_e[[c for c in complication_cols_k1 if c in df_e.columns]].sum(axis=1)
df_e['complication_score_k2'] = df_e[[c for c in complication_cols_k2 if c in df_e.columns]].sum(axis=1)

# ============== 5. Save processed Data ==============
print("\n5. Save processed Data")

df_e.to_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_aligned.csv', index=False)
print(f"   Saveto eicu_aligned.csv")

# Validation 
print("\n" + "="*60)
print(" DistributionComparison")
print("="*60)
features = ['uo_k1', 'sofa_24hours', 'creat', 'bun_k1']
for col in features:
    if col in df_m.columns and col in df_e.columns:
        m_mean = df_m[col].dropna().mean()
        e_mean = df_e[col].dropna().mean()
        ratio = m_mean / e_mean if e_mean != 0 else float('inf')
        print(f"{col:15s}: MIMIC={m_mean:8.2f}, eICU={e_mean:8.2f}, ratio={ratio:.2f}")

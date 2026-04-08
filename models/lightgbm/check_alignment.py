"""
 pair with 
"""
import pandas as pd
import numpy as np

# Load data
df_m = pd.read_csv('c:/Dynamic-RRT/4_ModelTrainingmodels/dwols/data/dwols_kdigo_features.csv')
df_e = pd.read_csv('c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_kdigo_features.csv')

print("="*60)
print("1. Creatinine (mg/dL vs μmol/L)")
print("="*60)
print(f"MIMIC creat: mean={df_m['creat'].mean():.2f}, median={df_m['creat'].median():.2f}")
print(f"eICU creat: mean={df_e['creat'].mean():.2f}, median={df_e['creat'].median():.2f}")

# mg/dLNormal range0.7-1.3, μmol/LNormal range62-115
# AKI Stage 3: creat >= 4.0 mg/dL or  >= 354 μmol/L
mimic_unit = "μmol/L" if df_m['creat'].mean() > 50 else "mg/dL"
eicu_unit = "μmol/L" if df_e['creat'].mean() > 50 else "mg/dL"
print(f"\nMIMIC : {mimic_unit}")
print(f"eICU : {eicu_unit}")

# if not RowConvert
if mimic_unit != eicu_unit:
    print("\n⚠️ not needtoConvert")
    # Convert : μmol/L = mg/dL × 88.4
    if mimic_unit == "mg/dL" and eicu_unit == "μmol/L":
        print("eICUneedtofromμmol/LConvertasmg/dL")
        for col in ['creat', 'creat_k1', 'creat_k2']:
            if col in df_e.columns:
                df_e[col] = df_e[col] / 88.4
else:
    print("\n✅ ")

print("\n" + "="*60)
print("2. FeatureDistributionComparison")
print("="*60)
features = ['creat', 'bun_k1', 'pot_k1', 'ph_k1', 'uo_k1', 'sofa_24hours']
for col in features:
    if col in df_m.columns and col in df_e.columns:
        m_mean = df_m[col].dropna().mean()
        e_mean = df_e[col].dropna().mean()
        m_std = df_m[col].dropna().std()
        e_std = df_e[col].dropna().std()
        ratio = m_mean / e_mean if e_mean != 0 else float('inf')
        print(f"{col:15s}: MIMIC={m_mean:8.2f}±{m_std:6.2f}, eICU={e_mean:8.2f}±{e_std:6.2f}, ratio={ratio:.2f}")

print("\n" + "="*60)
print("3. Distribution (KL )")
print("="*60)

def compute_distribution_shift(arr1, arr2, bins=50):
    """Computation DistributionJensen-Shannon """
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]
    
    # range
    min_val = min(arr1.min(), arr2.min())
    max_val = max(arr1.max(), arr2.max())
    
    hist1, _ = np.histogram(arr1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(arr2, bins=bins, range=(min_val, max_val), density=True)
    
    # smooth
    hist1 = (hist1 + 1e-10) / (hist1 + 1e-10).sum()
    hist2 = (hist2 + 1e-10) / (hist2 + 1e-10).sum()
    
    # JS 
    m = (hist1 + hist2) / 2
    js = 0.5 * (np.sum(hist1 * np.log(hist1/m)) + np.sum(hist2 * np.log(hist2/m)))
    return js

for col in features:
    if col in df_m.columns and col in df_e.columns:
        js_div = compute_distribution_shift(df_m[col].values, df_e[col].values)
        shift_level = " " if js_div < 0.1 else ("in" if js_div < 0.3 else " ")
        print(f"{col:15s}: JS ={js_div:.4f} ({shift_level} )")

print("\n" + "="*60)
print("4. RRTLabelDistribution")
print("="*60)
y_m = ((df_m['a1']==1)|(df_m['a2']==1)|(df_m['a3']==1)).astype(int)
y_e = df_e['received_rrt']
print(f"MIMIC RRT Rate: {y_m.mean()*100:.2f}%")
print(f"eICU RRT Rate: {y_e.mean()*100:.2f}%")
print(f" : {abs(y_m.mean() - y_e.mean())*100:.2f}%")

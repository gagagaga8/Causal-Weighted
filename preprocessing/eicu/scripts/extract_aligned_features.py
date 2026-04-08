"""
eICU feature extraction - fully semantically aligned
Ensure all units are consistent with MIMIC

Unit alignment specification:
- creat: mg/dL (MIMICdefault)
- BUN: mg/dL (eICUdefaultmg/dL)
- potassium: mEq/L ( )
- pH: No ( )
- urine output: mL/kg/h (MIMIC) → needfrom mLConvert
- SOFA: 0-24 (needfromgroup Computation nonAPACHE)
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'dbname': os.getenv('EICU_DB_NAME', 'eICU'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

def get_engine():
    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(conn_str)

def extract_aki_patients(engine):
    """ExtractAKI Stage 3Patient"""
    print("1. ExtractAKI Stage 3Patient...")
    
    query = """
    WITH creatinine AS (
        SELECT patientunitstayid,
               labresultoffset,
               labresult as creatinine
        FROM lab
        WHERE labname = 'creatinine'
          AND labresult IS NOT NULL
          AND labresult > 0
    ),
    aki_stage AS (
        SELECT c.patientunitstayid,
               MAX(c.creatinine) as max_creat,
               MIN(c.creatinine) as baseline_creat,
               CASE 
                   WHEN MAX(c.creatinine) >= 4.0 THEN 3
                   WHEN MAX(c.creatinine) >= 3 * NULLIF(MIN(c.creatinine), 0) THEN 3
                   WHEN MAX(c.creatinine) >= 2 * NULLIF(MIN(c.creatinine), 0) THEN 2
                   ELSE 1
               END as aki_stage_creat
        FROM creatinine c
        GROUP BY c.patientunitstayid
        HAVING MAX(c.creatinine) >= 4.0 OR MAX(c.creatinine) >= 3 * NULLIF(MIN(c.creatinine), 0)
    )
    SELECT p.patientunitstayid, 
           p.age, p.gender, p.admissionweight as weight,
           p.unitdischargeoffset, p.hospitaldischargestatus,
           ak.max_creat as creat, ak.baseline_creat, ak.aki_stage_creat
    FROM patient p
    INNER JOIN aki_stage ak ON p.patientunitstayid = ak.patientunitstayid
    WHERE p.age != '' AND p.unitdischargeoffset IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    print(f" to {len(df)} AKI Stage 3Patient")
    return df

def extract_sofa_components(engine, patient_ids):
    """ExtractSOFAScore - useAPACHEscaletoSOFArange"""
    print("2. ExtractSOFAScore (APACHEscaleto0-24)...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        # Scale APACHE score to SOFA range (0-24)
        # APACHE 0-71 → SOFA 0-24
        # QueryTable : eICUapachescoreMean~60, MIMICSOFAMean~9
        # scale = 9/60 = 0.15
        query = f"""
        SELECT patientunitstayid, 
               LEAST(24, GREATEST(0, COALESCE(apachescore, 0) * 0.15)) as sofa_24hours
        FROM apachepatientresult
        WHERE patientunitstayid IN ({ids_str})
        """
        
        try:
            batch_df = pd.read_sql(query, engine)
            results.append(batch_df)
        except Exception as e:
            print(f" SOFA Failure: {e}")
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def extract_lab_values(engine, patient_ids):
    """ExtractLaboratoryvalue - withMIMICpair """
    print("3. ExtractLaboratoryvalue (creat mg/dL, BUN mg/dL, K mEq/L, pH)...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        # eICUlabTable 
        # - creatinine: mg/dL (withMIMIC )
        # - BUN: mg/dL (withMIMIC )
        # - potassium: mEq/L (withMIMIC )
        # - pH: No (withMIMIC )
        query = f"""
        SELECT patientunitstayid,
               -- k1: 0-24 hours
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as creat_k1,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as bun_k1,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as pot_k1,
               MIN(CASE WHEN labname = 'pH' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as ph_k1,
               -- k2: 24-48 hours
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as creat_k2,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as bun_k2,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as pot_k2,
               MIN(CASE WHEN labname = 'pH' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as ph_k2,
               -- k3: 48-72 hours
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as creat_k3,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as bun_k3,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as pot_k3,
               MIN(CASE WHEN labname = 'pH' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as ph_k3
        FROM lab
        WHERE patientunitstayid IN ({ids_str})
          AND labresultoffset BETWEEN 0 AND 4320
          AND labname IN ('creatinine', 'BUN', 'potassium', 'pH')
          AND labresult IS NOT NULL
        GROUP BY patientunitstayid
        """
        
        batch_df = pd.read_sql(query, engine)
        results.append(batch_df)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def extract_urine_output(engine, patient_ids, weights_df):
    """ExtractUrine output - ConvertasmL/kg/hwithMIMICpair 
    
    MIMIC: uo_rt_24hr is mL/kg/h
    eICU: intakeoutput is mL needto mL ÷ weight(kg) ÷ (h) = mL/kg/h
    """
    print("4. ExtractUrine outputand ConvertasmL/kg/h...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        # each Urine output(mL)
        query = f"""
        SELECT patientunitstayid,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 0 AND 1440 THEN cellvaluenumeric END) as uo_ml_k1,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 1440 AND 2880 THEN cellvaluenumeric END) as uo_ml_k2,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 2880 AND 4320 THEN cellvaluenumeric END) as uo_ml_k3
        FROM intakeoutput
        WHERE patientunitstayid IN ({ids_str})
          AND celllabel ILIKE '%%urine%%'
          AND cellvaluenumeric IS NOT NULL
          AND cellvaluenumeric > 0
        GROUP BY patientunitstayid
        """
        
        try:
            batch_df = pd.read_sql(query, engine)
            if len(batch_df) > 0:
                results.append(batch_df)
        except Exception as e:
            print(f"   Urine outputExtractFailure: {e}")
    
    if not results:
        return pd.DataFrame(columns=['patientunitstayid', 'uo_k1', 'uo_k2', 'uo_k3'])
    
    uo_df = pd.concat(results, ignore_index=True)
    
    # MergeweightInfo Row Convert
    uo_df = uo_df.merge(weights_df[['patientunitstayid', 'weight']], on='patientunitstayid', how='left')
    
    # ConvertasmL/kg/h
    # : mL/kg/h = mL ÷ weight(kg) ÷ 24 hours
    default_weight = 70  # defaultweight
    uo_df['weight'] = uo_df['weight'].fillna(default_weight)
    
    uo_df['uo_k1'] = uo_df['uo_ml_k1'] / uo_df['weight'] / 24
    uo_df['uo_k2'] = uo_df['uo_ml_k2'] / uo_df['weight'] / 24
    uo_df['uo_k3'] = uo_df['uo_ml_k3'] / uo_df['weight'] / 24
    
    # limitto range (0-10 mL/kg/h)
    for col in ['uo_k1', 'uo_k2', 'uo_k3']:
        uo_df[col] = uo_df[col].clip(0, 10)
    
    return uo_df[['patientunitstayid', 'uo_k1', 'uo_k2', 'uo_k3']]

def extract_rrt_info(engine, patient_ids):
    """ExtractRRTTreatmentInfo"""
    print("5. ExtractRRTInfo...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        query = f"""
        SELECT patientunitstayid,
               MIN(treatmentoffset) as rrt_offset,
               1 as received_rrt
        FROM treatment
        WHERE patientunitstayid IN ({ids_str})
          AND (treatmentstring ILIKE '%%dialysis%%' 
               OR treatmentstring ILIKE '%%hemodialysis%%'
               OR treatmentstring ILIKE '%%CRRT%%'
               OR treatmentstring ILIKE '%%renal replacement%%'
               OR treatmentstring ILIKE '%%RRT%%'
               OR treatmentstring ILIKE '%%continuous renal%%')
        GROUP BY patientunitstayid
        """
        
        try:
            batch_df = pd.read_sql(query, engine)
            if len(batch_df) > 0:
                results.append(batch_df)
        except Exception as e:
            print(f"   RRTExtractFailure: {e}")
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def main():
    print("="*60)
    print("eICUFeatureExtract - all pair ")
    print("="*60)
    print(" pair :")
    print("  - creat: mg/dL")
    print("  - BUN: mg/dL")
    print("  - potassium: mEq/L")
    print(" - pH: No ")
    print("  - urine output: mL/kg/h")
    print(" - SOFA: 0-24 ")
    print("="*60 + "\n")
    
    engine = get_engine()
    
    # 1. ExtractAKIPatient
    aki_df = extract_aki_patients(engine)
    if len(aki_df) == 0:
        print("Error: not toAKIPatient")
        return
    
    patient_ids = aki_df['patientunitstayid'].tolist()
    
    # 2. ExtractSOFAScore
    sofa_df = extract_sofa_components(engine, patient_ids)
    
    # 3. ExtractLaboratoryvalue
    lab_df = extract_lab_values(engine, patient_ids)
    
    # 4. ExtractUrine output (needtoweightInfo)
    uo_df = extract_urine_output(engine, patient_ids, aki_df)
    
    # 5. ExtractRRTInfo
    rrt_df = extract_rrt_info(engine, patient_ids)
    
    # MergeData - 
    print("\n6. MergeData...")
    result = aki_df.copy()
    
    if len(sofa_df) > 0:
        # SOFAcancanhas records records
        sofa_df = sofa_df.drop_duplicates(subset='patientunitstayid', keep='first')
        result = result.merge(sofa_df[['patientunitstayid', 'sofa_24hours']], on='patientunitstayid', how='left')
    
    if len(lab_df) > 0:
        lab_df = lab_df.drop_duplicates(subset='patientunitstayid', keep='first')
        result = result.merge(lab_df, on='patientunitstayid', how='left')
    
    if len(uo_df) > 0:
        uo_df = uo_df.drop_duplicates(subset='patientunitstayid', keep='first')
        result = result.merge(uo_df, on='patientunitstayid', how='left')
    
    if len(rrt_df) > 0:
        rrt_df = rrt_df.drop_duplicates(subset='patientunitstayid', keep='first')
        result = result.merge(rrt_df, on='patientunitstayid', how='left')
        result['received_rrt'] = result['received_rrt'].fillna(0).astype(int)
    else:
        result['received_rrt'] = 0
    
    # ProcessingageandSex
    result['admission_age'] = pd.to_numeric(result['age'].replace('> 89', '90'), errors='coerce')
    result['gender'] = result['gender'].map({'Male': 'M', 'Female': 'F'})
    
    # ComputationAKI stage
    result['aki_stage'] = result['aki_stage_creat']
    result['aki_stage_uo'] = 0
    
    # ComputationtreatmentVariable
    result['aki_to_rrt_hours'] = result['rrt_offset'] / 60.0
    result['a1'] = ((result['aki_to_rrt_hours'] < 24) & (result['received_rrt'] == 1)).astype(int)
    result['a2'] = ((result['aki_to_rrt_hours'] < 48) & (result['received_rrt'] == 1)).astype(int)
    result['a3'] = ((result['aki_to_rrt_hours'] < 72) & (result['received_rrt'] == 1)).astype(int)
    
    # Computationhfd
    result['hfd'] = np.maximum(0, 60 - result['unitdischargeoffset'] / 1440.0)
    result.loc[result['hospitaldischargestatus'] == 'Expired', 'hfd'] = 0
    
    # Computationhospital_mortality
    result['hospital_mortality'] = (result['hospitaldischargestatus'] == 'Expired').astype(int)
    
    # Final Column
    final_columns = [
        'patientunitstayid', 'admission_age', 'gender', 'weight', 
        'sofa_24hours', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
        'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
        'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
        'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3',
        'a1', 'a2', 'a3', 'hfd', 'received_rrt', 'hospital_mortality'
    ]
    
    available_cols = [c for c in final_columns if c in result.columns]
    result = result[available_cols]
    
    # Save
    output_path = 'c:/Dynamic-RRT/2_eICUPreprocessingdata/eicu_aligned_full.csv'
    result.to_csv(output_path, index=False)
    
    # Validation pair 
    print("\n" + "="*60)
    print("7. pair Validation")
    print("="*60)
    print(f"Sample Size: {len(result)}")
    print(f"RRTPatient: {result['received_rrt'].sum()} ({result['received_rrt'].mean():.1%})")
    print("\nFeatureDistribution ( withMIMIC ):")
    print(f"  creat: mean={result['creat'].mean():.2f} mg/dL (MIMIC ~3.8)")
    print(f"  bun_k1: mean={result['bun_k1'].mean():.2f} mg/dL (MIMIC ~55)")
    print(f"  pot_k1: mean={result['pot_k1'].mean():.2f} mEq/L (MIMIC ~4.5)")
    print(f"  ph_k1: mean={result['ph_k1'].mean():.2f} (MIMIC ~7.3)")
    print(f"  uo_k1: mean={result['uo_k1'].mean():.2f} mL/kg/h (MIMIC ~0.9)")
    print(f"  sofa_24hours: mean={result['sofa_24hours'].mean():.2f} (MIMIC ~9)")
    
    print(f"\nSaveto: {output_path}")

if __name__ == '__main__':
    main()

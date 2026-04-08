"""
eICU full feature extraction
Extract features matching MIMIC training data from PostgreSQL
use LightGBMExternalValidation
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
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
    """Create database connection"""
    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(conn_str)

def extract_aki_patients(engine):
    """ExtractAKI Stage 3Patient"""
    print("ExtractAKI Stage 3Patient...")
    
    # DiagnosisasAKI Stage 3Patient ( DiagnosisTableor CreatinineElevated)
    query = """
    WITH aki_patients AS (
        SELECT DISTINCT p.patientunitstayid, p.patienthealthsystemstayid,
               p.age, p.gender, p.admissionweight as weight,
               p.unitdischargeoffset, p.hospitaldischargestatus
        FROM patient p
        WHERE p.age != ''
          AND p.unitdischargeoffset IS NOT NULL
    ),
    -- Creatininevalue AKI
    creatinine AS (
        SELECT patientunitstayid,
               labresultoffset,
               labresult as creatinine
        FROM lab
        WHERE labname = 'creatinine'
          AND labresult IS NOT NULL
          AND labresult > 0
    ),
    -- ComputationAKI Stage ( Creatinine >= 3xBaselineor >= 4 mg/dL)
    aki_stage AS (
        SELECT c.patientunitstayid,
               MAX(c.creatinine) as max_creat,
               MIN(c.creatinine) as baseline_creat,
               CASE 
                   WHEN MAX(c.creatinine) >= 4.0 THEN 3
                   WHEN MAX(c.creatinine) >= 3 * MIN(c.creatinine) THEN 3
                   WHEN MAX(c.creatinine) >= 2 * MIN(c.creatinine) THEN 2
                   WHEN MAX(c.creatinine) >= 1.5 * MIN(c.creatinine) THEN 1
                   ELSE 0
               END as aki_stage_creat
        FROM creatinine c
        GROUP BY c.patientunitstayid
        HAVING MAX(c.creatinine) >= 4.0 OR MAX(c.creatinine) >= 3 * MIN(c.creatinine)
    )
    SELECT ap.*, 
           ak.max_creat as creat,
           ak.baseline_creat,
           ak.aki_stage_creat
    FROM aki_patients ap
    INNER JOIN aki_stage ak ON ap.patientunitstayid = ak.patientunitstayid
    """
    
    df = pd.read_sql(query, engine)
    print(f" to {len(df)} AKI Stage 3Patient")
    return df

def extract_sofa_scores(engine, patient_ids):
    """ExtractSOFAScore"""
    print("ExtractSOFAScore...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        query = f"""
        SELECT patientunitstayid,
               actualhospitalmortality,
               apachescore as sofa_24hours
        FROM apachepatientresult
        WHERE patientunitstayid IN ({ids_str})
        """
        batch_df = pd.read_sql(query, engine)
        results.append(batch_df)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def extract_lab_values(engine, patient_ids):
    """ExtractLaboratoryvalue (creat, bun, potassium, ph) in3 """
    print("ExtractLaboratoryvalue...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        query = f"""
        SELECT patientunitstayid,
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as creat_k1,
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as creat_k2,
               MAX(CASE WHEN labname = 'creatinine' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as creat_k3,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as bun_k1,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as bun_k2,
               MAX(CASE WHEN labname = 'BUN' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as bun_k3,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as pot_k1,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as pot_k2,
               MAX(CASE WHEN labname = 'potassium' AND labresultoffset BETWEEN 2880 AND 4320 THEN labresult END) as pot_k3,
               MIN(CASE WHEN labname = 'pH' AND labresultoffset BETWEEN 0 AND 1440 THEN labresult END) as ph_k1,
               MIN(CASE WHEN labname = 'pH' AND labresultoffset BETWEEN 1440 AND 2880 THEN labresult END) as ph_k2,
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

def extract_urine_output(engine, patient_ids):
    """ExtractUrine outputData"""
    print("ExtractUrine outputData...")
    
    # Processing SQL 
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        query = f"""
        SELECT patientunitstayid,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 0 AND 360 THEN cellvaluenumeric END) / 6.0 as uo_rt_6hr,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 0 AND 720 THEN cellvaluenumeric END) / 12.0 as uo_rt_12hr,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 0 AND 1440 THEN cellvaluenumeric END) / 24.0 as uo_rt_24hr,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 0 AND 1440 THEN cellvaluenumeric END) as uo_k1,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 1440 AND 2880 THEN cellvaluenumeric END) as uo_k2,
               SUM(CASE WHEN intakeoutputoffset BETWEEN 2880 AND 4320 THEN cellvaluenumeric END) as uo_k3
        FROM intakeoutput
        WHERE patientunitstayid IN ({ids_str})
          AND celllabel ILIKE '%%urine%%'
          AND cellvaluenumeric IS NOT NULL
          AND cellvaluenumeric > 0
        GROUP BY patientunitstayid
        """
        
        batch_df = pd.read_sql(query, engine)
        results.append(batch_df)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def extract_rrt_info(engine, patient_ids):
    """ExtractRRTTreatmentInfo"""
    print("ExtractRRTInfo...")
    
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
               OR treatmentstring ILIKE '%%RRT%%')
        GROUP BY patientunitstayid
        """
        
        batch_df = pd.read_sql(query, engine)
        results.append(batch_df)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def extract_outcomes(engine, patient_ids):
    """Extract Data"""
    print("Extract Data...")
    
    batch_size = 1000
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        ids_str = ','.join(map(str, batch_ids))
        
        query = f"""
        SELECT patientunitstayid,
               unitdischargeoffset,
               hospitaldischargestatus,
               CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END as hospital_mortality
        FROM patient
        WHERE patientunitstayid IN ({ids_str})
        """
        
        batch_df = pd.read_sql(query, engine)
        results.append(batch_df)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def main():
    print("="*60)
    print("eICUFull FeatureExtract")
    print("="*60)
    
    engine = get_engine()
    
    # 1. ExtractAKIPatient Info
    aki_df = extract_aki_patients(engine)
    if len(aki_df) == 0:
        print("Error: not toAKIPatient")
        return
    
    patient_ids = aki_df['patientunitstayid'].tolist()
    
    # 2. ExtractSOFAScore
    sofa_df = extract_sofa_scores(engine, patient_ids)
    
    # 3. ExtractLaboratoryvalue
    lab_df = extract_lab_values(engine, patient_ids)
    
    # 4. ExtractUrine output
    uo_df = extract_urine_output(engine, patient_ids)
    
    # 5. ExtractRRTInfo
    rrt_df = extract_rrt_info(engine, patient_ids)
    
    # 6. Extract 
    outcome_df = extract_outcomes(engine, patient_ids)
    
    # MergeAll Data
    print("\nMergeData...")
    result = aki_df.copy()
    
    if len(sofa_df) > 0:
        result = result.merge(sofa_df, on='patientunitstayid', how='left')
    
    if len(lab_df) > 0:
        result = result.merge(lab_df, on='patientunitstayid', how='left')
    
    if len(uo_df) > 0:
        result = result.merge(uo_df, on='patientunitstayid', how='left')
    
    if len(rrt_df) > 0:
        result = result.merge(rrt_df, on='patientunitstayid', how='left')
        result['received_rrt'] = result['received_rrt'].fillna(0).astype(int)
    else:
        result['received_rrt'] = 0
    
    # Processingage
    result['admission_age'] = pd.to_numeric(result['age'].replace('> 89', '90'), errors='coerce')
    
    # ProcessingSex
    result['gender'] = result['gender'].map({'Male': 'M', 'Female': 'F'})
    
    # ComputationAKI stage ( )
    result['aki_stage'] = result['aki_stage_creat']
    result['aki_stage_uo'] = 0 # Processing
    
    # ComputationtreatmentVariable (a1, a2, a3)
    result['aki_to_rrt_hours'] = result['rrt_offset'] / 60.0  # Convertas hours
    result['a1'] = ((result['aki_to_rrt_hours'] < 24) & (result['received_rrt'] == 1)).astype(int)
    result['a2'] = ((result['aki_to_rrt_hours'] < 48) & (result['received_rrt'] == 1)).astype(int)
    result['a3'] = ((result['aki_to_rrt_hours'] < 72) & (result['received_rrt'] == 1)).astype(int)
    
    # Computationhfd (hospital free days, as useICUDischarge )
    result['hfd'] = np.maximum(0, 60 - result['unitdischargeoffset'] / 1440.0)
    result.loc[result['hospitaldischargestatus'] == 'Expired', 'hfd'] = 0
    
    # Final Column
    final_columns = [
        'patientunitstayid', 'admission_age', 'gender', 'weight', 
        'sofa_24hours', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
        'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
        'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
        'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
        'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3',
        'a1', 'a2', 'a3', 'hfd', 'received_rrt', 'hospital_mortality'
    ]
    
    # only inColumn
    available_cols = [c for c in final_columns if c in result.columns]
    result = result[available_cols]
    
    # SaveResults
    output_path = 'data/eicu_full_features.csv'
    result.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("Complete")
    print(f"{'='*60}")
    print(f"Sample Size: {len(result)}")
    print(f"Features: {len(result.columns)}")
    print(f"RRTPatient: {result['received_rrt'].sum()} ({result['received_rrt'].mean():.1%})")
    print(f"\nSaveto: {output_path}")
    print(f"\nColumn : {result.columns.tolist()}")
    
    # Missingvalue
    print(f"\nMissingvaluestatistics:")
    missing = result.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]} ({missing[col]/len(result):.1%})")

if __name__ == '__main__':
    main()

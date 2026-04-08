-- ========================================
-- MIMIC-IV remaining issues fix script
-- Migrated from temp code, one-time fix during MIMIC preprocessing
-- microbiologyevents type issues and some concept table dependencies.
-- ========================================

-- 1. microbiologyevents Table suspicion_of_infection Dependency 
DROP VIEW IF EXISTS mimiciv_hosp.microbiologyevents CASCADE;
CREATE VIEW mimiciv_hosp.microbiologyevents AS
SELECT
    microevent_id::INT,
    subject_id::INT,
    hadm_id::INT,
    micro_specimen_id::INT,
    chartdate::DATE,
    charttime::TIMESTAMP,
    spec_itemid::INT,
    spec_type_desc,
    test_seq::INT,
    storedate::DATE,
    storetime::TIMESTAMP,
    test_itemid::INT,
    test_name,
    org_itemid::INT, -- as INT
    org_name,
    isolate_num::INT,
    quantity,
    ab_itemid::INT,
    ab_name,
    dilution_text,
    dilution_comparison,
    CASE WHEN dilution_value ~ '^[0-9\.-]+$' THEN dilution_value::DOUBLE PRECISION ELSE NULL END AS dilution_value,
    interpretation,
    comments
FROM mimic_hosp.microbiologyevents;

-- 2. encode meld and sirs SQL
-- due to meld.sql and sirs.sql FilecancanPackage GBK encode in UTF-8 Data inwill 
-- e.g. need use Table use erwill SQL as UTF-8 againExecute 

-- 3. Create antibiotic Table encode in 
-- Full SQL only down Dependency 
DROP TABLE IF EXISTS mimiciv_derived.antibiotic CASCADE;
CREATE TABLE mimiciv_derived.antibiotic AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    starttime,
    stoptime,
    drug
FROM (
    SELECT NULL::INT AS subject_id,
           NULL::INT AS hadm_id,
           NULL::INT AS stay_id,
           NULL::TIMESTAMP AS starttime,
           NULL::TIMESTAMP AS stoptime,
           NULL::TEXT AS drug
    LIMIT 0
) AS placeholder;

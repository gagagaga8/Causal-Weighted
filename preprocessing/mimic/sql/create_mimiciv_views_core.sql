-- ========================================
-- MIMIC-IV type conversion and cleaning views (core subset)
-- Migrated from temp code, used in MIMIC preprocessing to unify numeric types/time formats,
-- ensuring official MIMIC-IV concept SQL runs on local database.
-- ========================================

-- 1. chartevents
DROP VIEW IF EXISTS mimiciv_icu.chartevents CASCADE;
CREATE VIEW mimiciv_icu.chartevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    caregiver_id::INT,
    charttime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid::INT,
    value,
    CASE WHEN valuenum ~ '^[0-9\.-]+$' THEN valuenum::DOUBLE PRECISION ELSE NULL END AS valuenum,
    valueuom
FROM mimic_icu.chartevents;

-- 2. icustays
DROP VIEW IF EXISTS mimiciv_icu.icustays CASCADE;
CREATE VIEW mimiciv_icu.icustays AS
SELECT * FROM mimic_icu.icustays;

-- 3. outputevents
DROP VIEW IF EXISTS mimiciv_icu.outputevents CASCADE;
CREATE VIEW mimiciv_icu.outputevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    charttime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid::INT,
    CASE WHEN value ~ '^[0-9\.-]+$' THEN value::DOUBLE PRECISION ELSE NULL END AS value,
    valueuom
FROM mimic_icu.outputevents;

-- 4. admissions
DROP VIEW IF EXISTS mimiciv_hosp.admissions CASCADE;
CREATE VIEW mimiciv_hosp.admissions AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    admittime::TIMESTAMP,
    dischtime::TIMESTAMP,
    CASE WHEN deathtime = '' THEN NULL ELSE deathtime::TIMESTAMP END AS deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    language,
    marital_status,
    race,
    CASE WHEN edregtime = '' THEN NULL ELSE edregtime::TIMESTAMP END AS edregtime,
    CASE WHEN edouttime = '' THEN NULL ELSE edouttime::TIMESTAMP END AS edouttime,
    hospital_expire_flag::SMALLINT
FROM mimic_hosp.admissions;

-- 5. labevents
DROP VIEW IF EXISTS mimiciv_hosp.labevents CASCADE;
CREATE VIEW mimiciv_hosp.labevents AS
SELECT
    labevent_id::BIGINT,
    subject_id::INT,
    hadm_id::INT,
    specimen_id::BIGINT,
    itemid::INT,
    order_provider_id,
    charttime::TIMESTAMP,
    storetime::TIMESTAMP,
    value,
    CASE WHEN valuenum ~ '^[0-9\.-]+$' THEN valuenum::DOUBLE PRECISION ELSE NULL END AS valuenum,
    valueuom,
    CASE WHEN ref_range_lower ~ '^[0-9\.-]+$' THEN ref_range_lower::DOUBLE PRECISION ELSE NULL END AS ref_range_lower,
    CASE WHEN ref_range_upper ~ '^[0-9\.-]+$' THEN ref_range_upper::DOUBLE PRECISION ELSE NULL END AS ref_range_upper,
    flag,
    priority,
    comments
FROM mimic_hosp.labevents;

-- 6. patients
DROP VIEW IF EXISTS mimiciv_hosp.patients CASCADE;
CREATE VIEW mimiciv_hosp.patients AS
SELECT
    subject_id::INT,
    gender,
    anchor_age::INT,
    anchor_year::INT,
    anchor_year_group,
    CASE WHEN dod = '' THEN NULL ELSE dod::TIMESTAMP END AS dod
FROM mimic_hosp.patients;

-- 7. diagnoses_icd
DROP VIEW IF EXISTS mimiciv_hosp.diagnoses_icd CASCADE;
CREATE VIEW mimiciv_hosp.diagnoses_icd AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    seq_num::INT,
    icd_code,
    icd_version::SMALLINT
FROM mimic_hosp.diagnoses_icd;

-- 8. d_icd_diagnoses
DROP VIEW IF EXISTS mimiciv_hosp.d_icd_diagnoses CASCADE;
CREATE VIEW mimiciv_hosp.d_icd_diagnoses AS
SELECT
    icd_code,
    icd_version::SMALLINT,
    long_title
FROM mimic_hosp.d_icd_diagnoses;

-- 9. inputevents ifCSV Normal can notRun use fix_inputevents_mapping.sql 
-- typeConvertVersion ComparisonDebug
DROP VIEW IF EXISTS mimiciv_icu.inputevents CASCADE;
CREATE VIEW mimiciv_icu.inputevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    starttime::TIMESTAMP,
    endtime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid::INT,
    CASE WHEN amount ~ '^[0-9\.-]+$' THEN amount::DOUBLE PRECISION ELSE NULL END AS amount,
    amountuom,
    CASE WHEN rate ~ '^[0-9\.-]+$' THEN rate::DOUBLE PRECISION ELSE NULL END AS rate,
    rateuom,
    orderid::INT,
    linkorderid::INT,
    ordercategoryname,
    secondaryordercategoryname,
    ordercomponenttypedescription,
    ordercategorydescription,
    CASE WHEN patientweight ~ '^[0-9\.-]+$' THEN patientweight::DOUBLE PRECISION ELSE NULL END AS patientweight,
    CASE WHEN totalamount ~ '^[0-9\.-]+$' THEN totalamount::DOUBLE PRECISION ELSE NULL END AS totalamount,
    totalamountuom,
    CASE WHEN isopenbag ~ '^[0-1]$' THEN isopenbag::SMALLINT ELSE NULL END AS isopenbag,
    continueinnextdept::SMALLINT,
    statusdescription
FROM mimic_icu.inputevents;

-- 10. procedureevents ifCSV Normal can notRun use fix_procedureevents_mapping.sql 
DROP VIEW IF EXISTS mimiciv_icu.procedureevents CASCADE;
CREATE VIEW mimiciv_icu.procedureevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    starttime::TIMESTAMP,
    endtime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid::INT,
    CASE WHEN value ~ '^[0-9\.-]+$' THEN value::DOUBLE PRECISION ELSE NULL END AS value,
    valueuom,
    location,
    locationcategory,
    orderid::INT,
    linkorderid::INT,
    ordercategoryname,
    ordercategorydescription,
    CASE WHEN patientweight ~ '^[0-9\.-]+$' THEN patientweight::DOUBLE PRECISION ELSE NULL END AS patientweight,
    CASE WHEN isopenbag ~ '^[0-1]$' THEN isopenbag::SMALLINT ELSE NULL END AS isopenbag,
    continueinnextdept::SMALLINT,
    statusdescription
FROM mimic_icu.procedureevents;

-- 11. d_items
DROP VIEW IF EXISTS mimiciv_icu.d_items CASCADE;
CREATE VIEW mimiciv_icu.d_items AS
SELECT
    itemid::INT,
    label,
    abbreviation,
    linksto,
    category,
    unitname,
    param_type,
    CASE WHEN lownormalvalue ~ '^[0-9\.-]+$' THEN lownormalvalue::DOUBLE PRECISION ELSE NULL END AS lownormalvalue,
    CASE WHEN highnormalvalue ~ '^[0-9\.-]+$' THEN highnormalvalue::DOUBLE PRECISION ELSE NULL END AS highnormalvalue
FROM mimic_icu.d_items;

-- 12. services
DROP VIEW IF EXISTS mimiciv_hosp.services CASCADE;
CREATE VIEW mimiciv_hosp.services AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    transfertime::TIMESTAMP,
    prev_service,
    curr_service
FROM mimic_hosp.services;

-- 13. datetimeevents
DROP VIEW IF EXISTS mimiciv_icu.datetimeevents CASCADE;
CREATE VIEW mimiciv_icu.datetimeevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    charttime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid::INT,
    value::TIMESTAMP,
    valueuom
FROM mimic_icu.datetimeevents
WHERE EXISTS (SELECT 1 FROM mimic_icu.datetimeevents LIMIT 1);

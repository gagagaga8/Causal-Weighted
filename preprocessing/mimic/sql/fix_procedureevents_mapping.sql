-- Fix procedureevents table column mapping errors
-- Cause: missing caregiver_id column during import, causing column misalignment
-- Fixes procedureevents view in MIMIC preprocessing for invasive procedures and RRT concept tables

DROP VIEW IF EXISTS mimiciv_icu.procedureevents CASCADE;

CREATE VIEW mimiciv_icu.procedureevents AS
SELECT
    subject_id::INT,
    hadm_id::INT,
    stay_id::INT,
    NULL::INT AS caregiver_id,  -- MissingColumn
    starttime::TIMESTAMP,
    endtime::TIMESTAMP,
    storetime::TIMESTAMP,
    itemid2::INT AS itemid, -- itemid in itemid2 Column
    CASE WHEN value ~ '^[0-9\.-]+$' THEN value::DOUBLE PRECISION ELSE NULL END AS value,
    valueuom,
    location,
    locationcategory,
    orderid::INT,
    linkorderid::INT,
    ordercategoryname,
    ordercategorydescription,
    CASE WHEN patientweight ~ '^[0-9\.-]+$' THEN patientweight::DOUBLE PRECISION ELSE NULL END AS patientweight,
    isopenbag::SMALLINT,
    continueinnextdept::SMALLINT,
    statusdescription,
    CASE WHEN cancelreason ~ '^[0-9\.-]+$' THEN cancelreason::DOUBLE PRECISION ELSE NULL END AS originalamount,
    CASE WHEN comments_date ~ '^[0-9\.-]+$' THEN comments_date::DOUBLE PRECISION ELSE NULL END AS originalrate
FROM mimic_icu.procedureevents;

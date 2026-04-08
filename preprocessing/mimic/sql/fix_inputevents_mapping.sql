-- Fix inputevents table column mapping errors
-- Cause: missing caregiver_id column during import, causing column misalignment
-- Fixes inputevents view in MIMIC preprocessing, ensuring concept table compatibility with official SQL

-- Delete Error 
DROP VIEW IF EXISTS mimiciv_icu.inputevents CASCADE;

-- CreateCorrectmapping 
CREATE VIEW mimiciv_icu.inputevents AS
SELECT
    subject_id::INT AS subject_id,
    hadm_id::INT AS hadm_id,
    stay_id::INT AS stay_id,
    NULL::INT AS caregiver_id, -- TableMissing Column fill NULL
    endtime::TIMESTAMP AS starttime, -- Table starttime is itemid
    storetime::TIMESTAMP AS endtime, -- Table endtime is starttime
    itemid::TIMESTAMP AS storetime, -- Table storetime is endtime
    amount::INT AS itemid, -- Table itemid is storetime amount is itemid
    CASE WHEN amountuom ~ '^[0-9\.-]+$' THEN amountuom::DOUBLE PRECISION ELSE NULL END AS amount, -- Table amountuom is amount
    rate AS amountuom, -- Table rate is amountuom
    CASE WHEN rateuom ~ '^[0-9\.-]+$' THEN rateuom::DOUBLE PRECISION ELSE NULL END AS rate, -- Table rateuom is rate
    orderid AS rateuom, -- Table orderid is rateuom
    linkorderid::BIGINT AS orderid,
    ordercategoryname::BIGINT AS linkorderid,
    secondaryordercategoryname AS ordercategoryname,
    ordercomponenttypedescription AS secondaryordercategoryname,
    ordercategorydescription AS ordercomponenttypedescription,
    patientweight AS ordercategorydescription,
    CASE WHEN totalamount ~ '^[0-9\.-]+$' THEN totalamount::DOUBLE PRECISION ELSE NULL END AS patientweight,
    CASE WHEN totalamountuom ~ '^[0-9\.-]+$' THEN totalamountuom::DOUBLE PRECISION ELSE NULL END AS totalamount,
    isopenbag AS totalamountuom,
    continueinnextdept::SMALLINT AS isopenbag,
    cancelreason::SMALLINT AS continueinnextdept,
    statusdescription AS cancelreason,
    CASE WHEN originalamount ~ '^[0-9\.-]+$' THEN originalamount::DOUBLE PRECISION ELSE NULL END AS statusdescription,
    CASE WHEN originalrate ~ '^[0-9\.-]+$' THEN originalrate::DOUBLE PRECISION ELSE NULL END AS originalamount,
    starttime::DOUBLE PRECISION AS originalrate -- Table starttime is itemid 
FROM mimic_icu.inputevents;

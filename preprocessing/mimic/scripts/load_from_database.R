# ============================================
# MIMIC-IV data preprocessing (read directly from PostgreSQL)
# Based on 63 pre-generated concept tables
# ============================================

library(dplyr)
library(RPostgreSQL)

# Load database connection config
source("../config/db_config.R")

# ConnectionData 
con <- create_mimic_connection()

message("StartfromMIMIC-IVData Data...")

# 1. KDIGO Data
message(" kdigo_stages...")
kdigo_stages <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.kdigo_stages")
# mapping stay_id -> icustay_id
if("stay_id" %in% colnames(kdigo_stages)) {
  kdigo_stages <- kdigo_stages %>% rename(icustay_id = stay_id)
}

# 2. ICUHospitalization Info
message(" icustay_detail...")
icustay_detail <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.icustay_detail")
# mapping stay_id -> icustay_id
if("stay_id" %in% colnames(icustay_detail)) {
  icustay_detail <- icustay_detail %>% rename(icustay_id = stay_id)
}

# 3. RRTCorrelationData
message(" rrt...")
rrt_data <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.rrt")
# Convertaspivoted_rrt rrtTablealreadyPackage charttimeanddialysis_active 
pivoted_rrt <- rrt_data %>%
  select(stay_id, charttime, dialysis_active, dialysis_type, dialysis_present)
if("stay_id" %in% colnames(pivoted_rrt)) {
  pivoted_rrt <- pivoted_rrt %>% rename(icustay_id = stay_id)
}

# 4. CRRT 
message(" crrt...")
crrt_data <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.crrt")
crrt_durations <- crrt_data
if("stay_id" %in% colnames(crrt_durations)) {
  crrt_durations <- crrt_durations %>% rename(icustay_id = stay_id)
}

# 5. Laboratory Data
message(" first_day_lab...")
first_day_lab <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.first_day_lab")
if("stay_id" %in% colnames(first_day_lab)) {
  first_day_lab <- first_day_lab %>% rename(icustay_id = stay_id)
}

# 6. AnalysisData
message(" first_day_bg and first_day_bg_art...")
first_day_bg <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.first_day_bg")
first_day_bg_art <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.first_day_bg_art")
if("stay_id" %in% colnames(first_day_bg)) {
  first_day_bg <- first_day_bg %>% rename(icustay_id = stay_id)
}
if("stay_id" %in% colnames(first_day_bg_art)) {
  first_day_bg_art <- first_day_bg_art %>% rename(icustay_id = stay_id)
}
# Merge Data
pivoted_bg <- bind_rows(first_day_bg, first_day_bg_art)
pivoted_lab <- first_day_lab

# 7. ESRDPatient
message(" esrd Patient from sepsis3 and suspicion_of_infection ...")
# MIMIC-IV has ESRDTable needtofromICDDiagnosisinExtract
esrd_patients <- dbGetQuery(con, "
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE d.icd_code IN ('5856', '5859', 'N186', '585.6', '585.9')
     OR d.icd_version = 9 AND d.icd_code LIKE '585%'
")

# 8. useData
message(" vasoactive_agent...")
vasoactive_agent <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.vasoactive_agent")
vasopressor_durations <- vasoactive_agent
if("stay_id" %in% colnames(vasopressor_durations)) {
  vasopressor_durations <- vasopressor_durations %>% rename(icustay_id = stay_id)
}

# 9. Data
message(" ventilation...")
ventilation_data <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.ventilation")
ventilation_durations <- ventilation_data
if("stay_id" %in% colnames(ventilation_durations)) {
  ventilation_durations <- ventilation_durations %>% rename(icustay_id = stay_id)
}

# 10. HospitalizationInfo
message(" admissions...")
admissions <- dbGetQuery(con, "SELECT * FROM mimiciv_hosp.admissions")

# 11. weightData
message(" weight_durations...")
weight_durations <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.weight_durations")
if("stay_id" %in% colnames(weight_durations)) {
  weight_durations <- weight_durations %>% rename(icustay_id = stay_id)
}

# 12. SOFAScore
message(" first_day_sofa...")
first_day_sofa <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.first_day_sofa")
if("stay_id" %in% colnames(first_day_sofa)) {
  first_day_sofa <- first_day_sofa %>% rename(icustay_id = stay_id)
}
# first_day_sofaTableColumn alreadyisstandard use
pivoted_sofa <- first_day_sofa

# 13. LaboratoryTemporal Data use Extract k2/k3 bun, pot, ph 
# Query only KDIGO3 patient hadm_ids, greatly reduce labevents scan range
message(" labevents Temporal bun/potassium/ph ...")
pivoted_lab_timeseries <- dbGetQuery(con, "
  WITH kdigo3_hadm AS (
    SELECT DISTINCT ie.hadm_id
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_derived.kdigo_stages k ON ie.stay_id = k.stay_id
    WHERE k.aki_stage = 3
  )
  SELECT 
    le.hadm_id,
    le.charttime,
    MAX(CASE WHEN le.itemid = 51006 THEN le.valuenum END) AS bun,
    MAX(CASE WHEN le.itemid = 50971 THEN le.valuenum END) AS potassium,
    MAX(CASE WHEN le.itemid = 50820 THEN le.valuenum END) AS ph,
    MAX(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate,
    MAX(CASE WHEN le.itemid = 50803 THEN le.valuenum END) AS bicarbonate
  FROM mimiciv_hosp.labevents le
  INNER JOIN kdigo3_hadm kh ON le.hadm_id = kh.hadm_id
  WHERE le.itemid IN (51006, 50971, 50820, 50813, 50803)
    AND le.valuenum IS NOT NULL
  GROUP BY le.hadm_id, le.charttime
")
message(paste0("  labevents Temporal : ", nrow(pivoted_lab_timeseries), " Row"))

# 14. Data fromantibioticandnsaid 
message(" antibiotic and nsaid as ...")
antibiotic_data <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.antibiotic")
nsaid_data <- dbGetQuery(con, "SELECT * FROM mimiciv_derived.nsaid")

# antibiotichasstay_id mapping
if("stay_id" %in% colnames(antibiotic_data)) {
  antibiotic_data <- antibiotic_data %>% rename(icustay_id = stay_id)
}

# nsaid hasstay_id needtothroughsubject_id+hadm_idAssociationicustay_detail icustay_id
if(!"stay_id" %in% colnames(nsaid_data) && "subject_id" %in% colnames(nsaid_data)) {
  nsaid_data <- nsaid_data %>%
    left_join(icustay_detail %>% select(subject_id, hadm_id, icustay_id), 
              by = c("subject_id", "hadm_id"))
}

# MergeData
immunosuppressant <- bind_rows(
  antibiotic_data,
  nsaid_data
)

message("Data Complete ")
message(paste0("kdigo_stages: ", nrow(kdigo_stages), " Row"))
message(paste0("icustay_detail: ", nrow(icustay_detail), " Row"))
message(paste0("pivoted_rrt: ", nrow(pivoted_rrt), " Row"))
message(paste0("vasopressor_durations: ", nrow(vasopressor_durations), " Row"))
message(paste0("ventilation_durations: ", nrow(ventilation_durations), " Row"))

# Close database connection
dbDisconnect(con)
message("Data Connectionalreadyshutdown")

# Savetowhen Preprocessing use
message("DataalreadyLoadingtoR can ExecutePreprocessingpipeline")

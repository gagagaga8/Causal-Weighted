# ============================================
# eICU-CRD data loading (from PostgreSQL)
# Extract AKI/RRT features from eICU-CRD standard table structure
# ============================================

library(dplyr)
library(RPostgreSQL)

# Load database connection config
source("../config/db_config.R")

# ConnectionData 
con <- create_eicu_connection()

message("StartfromeICU-CRDData Data...")

# 1. Patient Info
message(" patient Table...")
patient <- dbGetQuery(con, "SELECT * FROM patient")

# 1.1 weightInfo
message(" patient Table Filterweight ...")
patient_weight <- dbGetQuery(con, "
  SELECT patientunitstayid, 
         admissionweight as weight,
         admissionheight as height
  FROM patient
  WHERE admissionweight IS NOT NULL AND admissionweight > 0
")

# 2. Laboratory Creatinine 
message(" lab Table Filtercreatinine ...")
lab_creatinine <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labname, labresult
  FROM lab
  WHERE LOWER(labname) LIKE '%creatinine%'
    AND labresult IS NOT NULL
  ORDER BY patientunitstayid, labresultoffset
")

# 2.1 BUN
message(" lab Table FilterBUN ...")
lab_bun <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labresult as bun
  FROM lab
  WHERE LOWER(labname) LIKE '%bun%' OR LOWER(labname) LIKE '%urea nitrogen%'
  ORDER BY patientunitstayid, labresultoffset
")

# 2.2 Potassium
message(" lab Table FilterPotassium ...")
lab_potassium <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labresult as potassium
  FROM lab
  WHERE LOWER(labname) LIKE '%potassium%'
  ORDER BY patientunitstayid, labresultoffset
")

# 2.3 pH
message(" lab Table FilterpH ...")
lab_ph <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labresult as ph
  FROM lab
  WHERE LOWER(labname) = 'ph' AND labresult BETWEEN 6.5 AND 8.0
  ORDER BY patientunitstayid, labresultoffset
")

# 2.4 Lactate /sepsis-AKI Metric 
message(" lab Table FilterLactate ...")
lab_lactate <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labresult as lactate
  FROM lab
  WHERE LOWER(labname) LIKE '%lactate%' AND labresult IS NOT NULL
  ORDER BY patientunitstayid, labresultoffset
")

# 2.5 Bicarbonate Total CO2 or bicarbonate 
message(" lab Table FilterBicarbonate ...")
lab_bicarbonate <- dbGetQuery(con, "
  SELECT patientunitstayid, labresultoffset, labresult as bicarbonate
  FROM lab
  WHERE (LOWER(labname) LIKE '%bicarbonate%' OR LOWER(labname) LIKE '%total co2%')
    AND labresult IS NOT NULL AND labresult BETWEEN 5 AND 50
  ORDER BY patientunitstayid, labresultoffset
")

# 3. Urine outputData
message(" intakeoutput Table Filterurine output ...")
urine_output <- dbGetQuery(con, "
  SELECT patientunitstayid, intakeoutputoffset, cellpath, cellvaluenumeric
  FROM intakeoutput
  WHERE LOWER(cellpath) LIKE '%urine%' 
    OR LOWER(cellpath) LIKE '%output%'
  ORDER BY patientunitstayid, intakeoutputoffset
")

# 4. DiagnosisInfo FilterAKI/ESRD 
message(" diagnosis Table FilterAKI/ESRD ...")
diagnosis_aki <- dbGetQuery(con, "
  SELECT patientunitstayid, diagnosisoffset, diagnosisstring, icd9code
  FROM diagnosis
  WHERE LOWER(diagnosisstring) LIKE '%renal%'
     OR LOWER(diagnosisstring) LIKE '%kidney%'
     OR LOWER(diagnosisstring) LIKE '%aki%'
     OR icd9code IN ('5856', '5859', '584', '584.5', '584.6', '584.7', '584.8', '584.9')
  ORDER BY patientunitstayid, diagnosisoffset
")

# 5. TreatmentInfo FilterRRT 
message(" treatment Table FilterRRT ...")
treatment_rrt <- dbGetQuery(con, "
  SELECT patientunitstayid, treatmentoffset, treatmentstring
  FROM treatment
  WHERE LOWER(treatmentstring) LIKE '%dialysis%'
     OR LOWER(treatmentstring) LIKE '%hemodialysis%'
     OR LOWER(treatmentstring) LIKE '%hemofiltration%'
     OR LOWER(treatmentstring) LIKE '%crrt%'
     OR LOWER(treatmentstring) LIKE '%cvvh%'
     OR LOWER(treatmentstring) LIKE '%renal replacement%'
  ORDER BY patientunitstayid, treatmentoffset
")

# 6. APACHEScore
message(" apachepatientresult Table...")
apache <- dbGetQuery(con, "SELECT * FROM apachepatientresult")

# 7. use Info Medication 
# MedicationFilter records 
message(" medication Table Filter and Medication ...")
medication <- dbGetQuery(con, "
  SELECT patientunitstayid, drugstartoffset, drugstopoffset, drugname, drughiclseqno
  FROM medication
  WHERE LOWER(drugname) LIKE '%immunosuppress%'
     OR LOWER(drugname) LIKE '%steroid%'
     OR LOWER(drugname) LIKE '%predniso%'
     OR LOWER(drugname) LIKE '%norepinephrine%'
     OR LOWER(drugname) LIKE '%levophed%'
     OR LOWER(drugname) LIKE '%epinephrine%'
     OR LOWER(drugname) LIKE '%adrenalin%'
     OR LOWER(drugname) LIKE '%dopamine%'
     OR LOWER(drugname) LIKE '%vasopressin%'
     OR LOWER(drugname) LIKE '%dobutamine%'
     OR LOWER(drugname) LIKE '%phenylephrine%'
     OR LOWER(drugname) LIKE '%milrinone%'
  ORDER BY patientunitstayid, drugstartoffset
")

# 8. mechanical ventilationInfo
# Optimization only PatientIDColumnTable allTable
message(" respiratorycharting Table only PatientID ...")
ventilation <- dbGetQuery(con, "
  SELECT DISTINCT patientunitstayid
  FROM respiratorycharting
")
ventilation$respchartoffset <- 0 # 

# fromrespiratorycareTable mechanical ventilationPatient
message(" respiratorycare Table...")
ventilation_care <- dbGetQuery(con, "
  SELECT DISTINCT patientunitstayid
  FROM respiratorycare
")
ventilation_care$respchartoffset <- 0

# Mergemechanical ventilationData
ventilation <- rbind(ventilation, ventilation_care) %>%
  distinct(patientunitstayid)

message("Data Complete ")
message(paste0("patient: ", nrow(patient), " Row"))
message(paste0("patient_weight: ", nrow(patient_weight), " Row"))
message(paste0("lab_creatinine: ", nrow(lab_creatinine), " Row"))
message(paste0("treatment_rrt: ", nrow(treatment_rrt), " Row"))
message(paste0("apache: ", nrow(apache), " Row"))

# Close database connection
dbDisconnect(con)
message("Data Connectionalreadyshutdown")

message("DataalreadyLoadingtoR can ExecutePreprocessingpipeline")

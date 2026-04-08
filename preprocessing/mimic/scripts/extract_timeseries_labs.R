# ============================================
# Extract time-series lab data and urine output from MIMIC-IV
# For dWOLS model k1/k2/k3 decision points
# ============================================

library(dplyr)
library(RPostgreSQL)

source("../config/db_config.R")
con <- create_mimic_connection()

cat("==============================================\n")
cat("Extracting time-series lab data from MIMIC-IV\n")
cat("==============================================\n\n")

# Loading hasPreprocessingData Package KDIGO3 
if(!file.exists("../data/mimic_processed.rds")) {
  stop("Error: Run run_preprocessing.R Generate Data")
}

base_data <- readRDS("../data/mimic_processed.rds")
cat(" Data:", nrow(base_data), "Patient\n\n")

# PatientColumnTable stay_idandKDIGO3 
patients <- base_data %>%
  select(stay_id, hadm_id, subject_id, charttime_kdigo3) %>%
  mutate(charttime_kdigo3 = as.POSIXct(charttime_kdigo3))

cat("StartExtractLaboratoryData...\n")

# ===== BUN (Blood Urea Nitrogen) =====
cat("ExtractBUNData...\n")
# Query IN 
hadm_ids <- unique(patients$hadm_id)
batch_size <- 1000
n_batches <- ceiling(length(hadm_ids) / batch_size)

bun_data <- data.frame()
for(i in 1:n_batches) {
  start_idx <- (i-1) * batch_size + 1
  end_idx <- min(i * batch_size, length(hadm_ids))
  batch_ids <- hadm_ids[start_idx:end_idx]
  
  bun_query <- sprintf("
  SELECT 
    le.subject_id,
    le.hadm_id,
    le.charttime,
    le.valuenum as bun
  FROM mimiciv_hosp.labevents le
  WHERE le.hadm_id IN (%s)
    AND le.itemid IN (51006)
    AND le.valuenum IS NOT NULL
    AND le.valuenum > 0
  ", paste(batch_ids, collapse=","))
  
  batch_data <- dbGetQuery(con, bun_query)
  bun_data <- bind_rows(bun_data, batch_data)
  cat(sprintf(" times %d/%d: %d records \n", i, n_batches, nrow(batch_data)))
}
cat("total ", nrow(bun_data), " recordsBUN \n")

# ===== (Potassium) =====
cat("Extract Data...\n")
pot_data <- data.frame()
for(i in 1:n_batches) {
  start_idx <- (i-1) * batch_size + 1
  end_idx <- min(i * batch_size, length(hadm_ids))
  batch_ids <- hadm_ids[start_idx:end_idx]
  
  pot_query <- sprintf("
  SELECT 
    le.subject_id,
    le.hadm_id,
    le.charttime,
    le.valuenum as potassium
  FROM mimiciv_hosp.labevents le
  WHERE le.hadm_id IN (%s)
    AND le.itemid IN (50971, 50822)
    AND le.valuenum IS NOT NULL
    AND le.valuenum > 0
  ", paste(batch_ids, collapse=","))
  
  batch_data <- dbGetQuery(con, pot_query)
  pot_data <- bind_rows(pot_data, batch_data)
  cat(sprintf(" times %d/%d: %d records \n", i, n_batches, nrow(batch_data)))
}
cat("total ", nrow(pot_data), " records \n")

# ===== pHvalue ( ) =====
cat("ExtractpHData...\n")
ph_data <- data.frame()
for(i in 1:n_batches) {
  start_idx <- (i-1) * batch_size + 1
  end_idx <- min(i * batch_size, length(hadm_ids))
  batch_ids <- hadm_ids[start_idx:end_idx]
  
  ph_query <- sprintf("
  SELECT 
    le.subject_id,
    le.hadm_id,
    le.charttime,
    le.valuenum as ph
  FROM mimiciv_hosp.labevents le
  WHERE le.hadm_id IN (%s)
    AND le.itemid IN (50820)
    AND le.valuenum IS NOT NULL
    AND le.valuenum BETWEEN 6.5 AND 8.0
  ", paste(batch_ids, collapse=","))
  
  batch_data <- dbGetQuery(con, ph_query)
  ph_data <- bind_rows(ph_data, batch_data)
  cat(sprintf(" times %d/%d: %d records \n", i, n_batches, nrow(batch_data)))
}
cat("total ", nrow(ph_data), " recordspH \n")

# ===== Urine output (Urine Output from outputevents) =====
cat("ExtractUrine outputData...\n")
stay_ids <- unique(patients$stay_id)
stay_batch_size <- 1000
n_stay_batches <- ceiling(length(stay_ids) / stay_batch_size)

uo_data <- data.frame()
for(i in 1:n_stay_batches) {
  start_idx <- (i-1) * stay_batch_size + 1
  end_idx <- min(i * stay_batch_size, length(stay_ids))
  batch_ids <- stay_ids[start_idx:end_idx]
  
  # Use actual itemids (no filter, take all urine output records)
  uo_query <- sprintf("
  SELECT 
    oe.stay_id,
    oe.charttime,
    oe.value as urine_output
  FROM mimiciv_icu.outputevents oe
  WHERE oe.stay_id IN (%s)
    AND oe.value IS NOT NULL
    AND oe.value > 0
  ", paste(batch_ids, collapse=","))
  
  batch_data <- dbGetQuery(con, uo_query)
  uo_data <- bind_rows(uo_data, batch_data)
  cat(sprintf(" times %d/%d: %d records \n", i, n_stay_batches, nrow(batch_data)))
}
cat("total ", nrow(uo_data), " recordsUrine output \n")

# ===== SOFAScore from first_day_sofa =====
cat("ExtractSOFAScore...\n")
sofa_data <- data.frame()
for(i in 1:n_stay_batches) {
  start_idx <- (i-1) * stay_batch_size + 1
  end_idx <- min(i * stay_batch_size, length(stay_ids))
  batch_ids <- stay_ids[start_idx:end_idx]
  
  sofa_query <- sprintf("
  SELECT 
    stay_id,
    sofa as sofa_24hours
  FROM mimiciv_derived.first_day_sofa
  WHERE stay_id IN (%s)
  ", paste(batch_ids, collapse=","))
  
  batch_data <- dbGetQuery(con, sofa_query)
  sofa_data <- bind_rows(sofa_data, batch_data)
  cat(sprintf(" times %d/%d: %d records \n", i, n_stay_batches, nrow(batch_data)))
}
cat("total ", nrow(sofa_data), " recordsSOFA \n")

dbDisconnect(con)
cat("\nData Connectionalreadyshutdown\n")

# ===== DataProcessingFunction =====
get_closest_value <- function(patient_data, lab_data, time_col, value_col, 
                               hours_before=0, hours_after=24, prefix="") {
  merged <- patient_data %>%
    left_join(lab_data, by=c("hadm_id")) %>%
    mutate(
      lab_time = as.POSIXct(!!sym(time_col)),
      time_diff = as.numeric(difftime(lab_time, charttime_kdigo3, units="hours"))
    ) %>%
    filter(time_diff >= hours_before, time_diff < hours_after, !is.na(!!sym(value_col))) %>%
    group_by(stay_id) %>%
    arrange(stay_id, desc(time_diff)) %>%
    slice(1) %>%
    ungroup() %>%
    select(stay_id, !!sym(value_col))
  
  colnames(merged)[2] <- prefix
  return(merged)
}

get_closest_uo <- function(patient_data, uo_data, hours_before=0, hours_after=24, prefix="uo") {
  # Urine outputneedtofromstay_idAssociation
  merged <- patient_data %>%
    left_join(uo_data, by="stay_id") %>%
    mutate(
      uo_time = as.POSIXct(charttime),
      time_diff = as.numeric(difftime(uo_time, charttime_kdigo3, units="hours"))
    ) %>%
    filter(time_diff >= hours_before, time_diff < hours_after, !is.na(urine_output)) %>%
    group_by(stay_id) %>%
    summarise(total_uo = sum(urine_output, na.rm=TRUE)) %>%
    ungroup()
  
  colnames(merged)[2] <- paste0(prefix, "_total")
  return(merged)
}

cat("\nStartmatchdecision pointData...\n")

# k1: KDIGO3 before 
cat(" k1 (KDIGO3 )...\n")
bun_k1 <- get_closest_value(patients, bun_data, "charttime", "bun", -Inf, 0, "bun_k1")
pot_k1 <- get_closest_value(patients, pot_data, "charttime", "potassium", -Inf, 0, "pot_k1")
ph_k1 <- get_closest_value(patients, ph_data, "charttime", "ph", -Inf, 0, "ph_k1")
uo_k1 <- get_closest_uo(patients, uo_data, -24, 0, "uo_k1")

# k2: KDIGO30-24h post
cat(" k2 (KDIGO3 0-24h)...\n")
bun_k2 <- get_closest_value(patients, bun_data, "charttime", "bun", 0, 24, "bun_k2")
pot_k2 <- get_closest_value(patients, pot_data, "charttime", "potassium", 0, 24, "pot_k2")
ph_k2 <- get_closest_value(patients, ph_data, "charttime", "ph", 0, 24, "ph_k2")
uo_k2 <- get_closest_uo(patients, uo_data, 0, 24, "uo_k2")

# k3: KDIGO324-48h post
cat(" k3 (KDIGO3 24-48h)...\n")
bun_k3 <- get_closest_value(patients, bun_data, "charttime", "bun", 24, 48, "bun_k3")
pot_k3 <- get_closest_value(patients, pot_data, "charttime", "potassium", 24, 48, "pot_k3")
ph_k3 <- get_closest_value(patients, ph_data, "charttime", "ph", 24, 48, "ph_k3")
uo_k3 <- get_closest_uo(patients, uo_data, 24, 48, "uo_k3")

# SOFA_24hours
sofa_merged <- patients %>%
  left_join(sofa_data, by="stay_id")

# MergeAll Data
cat("\nMergeData...\n")
# Deletebase_datain Urine outputColumn if in 
if("uo_k1" %in% colnames(base_data)) {
  base_data <- base_data %>% select(-uo_k1, -uo_k2, -uo_k3)
}

enhanced_data <- base_data %>%
  left_join(bun_k1, by="stay_id") %>%
  left_join(pot_k1, by="stay_id") %>%
  left_join(ph_k1, by="stay_id") %>%
  left_join(uo_k1, by="stay_id") %>%
  left_join(bun_k2, by="stay_id") %>%
  left_join(pot_k2, by="stay_id") %>%
  left_join(ph_k2, by="stay_id") %>%
  left_join(uo_k2, by="stay_id") %>%
  left_join(bun_k3, by="stay_id") %>%
  left_join(pot_k3, by="stay_id") %>%
  left_join(ph_k3, by="stay_id") %>%
  left_join(uo_k3, by="stay_id") %>%
  left_join(sofa_merged %>% select(stay_id, sofa_24hours), by="stay_id")

# RenameUrine outputColumnasuo_k1/k2/k3
if("uo_k1_total" %in% colnames(enhanced_data)) {
  enhanced_data <- enhanced_data %>%
    rename(uo_k1 = uo_k1_total, uo_k2 = uo_k2_total, uo_k3 = uo_k3_total)
}

# Save Data
saveRDS(enhanced_data, "../data/mimic_processed_enhanced.rds")
write.csv(enhanced_data, "../data/mimic_processed_enhanced.csv", row.names=FALSE)

cat("\n DataSaveComplete \n")
cat("File: ../data/mimic_processed_enhanced.rds\n")
cat(" Patient :", nrow(enhanced_data), "\n")

# statisticsMissing 
cat("\nVariableFull statistics:\n")
key_vars <- c("bun_k1", "bun_k2", "bun_k3", "pot_k1", "pot_k2", "pot_k3",
              "ph_k1", "ph_k2", "ph_k3", "uo_k1", "uo_k2", "uo_k3", "sofa_24hours")
for(v in key_vars) {
  if(v %in% colnames(enhanced_data)) {
    n_missing <- sum(is.na(enhanced_data[[v]]))
    pct <- round(100 * n_missing / nrow(enhanced_data), 1)
    cat(sprintf("  %15s: %4d Missing (%.1f%%)\n", v, n_missing, pct))
  }
}

cat("\nComplete \n")

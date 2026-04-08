# ============================================================================
# dWOLS preprocessing script (adapted for MIMIC-IV)
# Based on dataprep_dev.R, adapted for MIMIC-IV database structure
# ============================================================================

library(dplyr)
library(RPostgreSQL)

# Database connection config
source("../config/db_config.R")
con <- create_mimic_connection()

cat(rep("=", 80), "\n")
cat("dWOLS Preprocessingpipeline MIMIC-IV Version \n")
cat(rep("=", 80), "\n\n")

# ============================================================================
# step 1: Extract Table fromData 
# ============================================================================
cat("step 1: Extract Table...\n")

# MIMIC-IV use stay_id andnon icustay_id
cat("  - Loading ICU stays...\n")
icustay_detail <- dbGetQuery(con, "
  SELECT 
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    EXTRACT(YEAR FROM ie.intime) - p.anchor_year + p.anchor_age AS admission_age,
    ie.intime,
    ie.outtime,
    p.gender,
    p.dod
  FROM mimiciv_icu.icustays ie
  INNER JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
")
cat("    ICU stays:", nrow(icustay_detail), "\n")

# KDIGO stages
cat("  - Loading KDIGO stages...\n")
kdigo_stages <- dbGetQuery(con, "
  SELECT 
    stay_id,
    charttime,
    aki_stage,
    aki_stage_creat,
    aki_stage_uo,
    creat,
    uo_rt_6hr,
    uo_rt_12hr,
    uo_rt_24hr
  FROM mimiciv_derived.kdigo_stages
")
cat("    KDIGO records:", nrow(kdigo_stages), "\n")

# RRT events (from procedureevents)
cat("  - Loading RRT events...\n")
pivoted_rrt <- dbGetQuery(con, "
  SELECT 
    pe.stay_id,
    pe.starttime AS charttime,
    1 AS dialysis_active,
    pe.itemid,
    di.label AS dialysis_type
  FROM mimiciv_icu.procedureevents pe
  LEFT JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
  WHERE pe.itemid IN (
    225805,  -- Peritoneal Dialysis
    225809,  -- CRRT Filter Change
    225955,  -- Dialysis Catheter
    225441,  -- Hemodialysis
    226499   -- Dialysis - CRRT
  )
")
cat("    RRT events:", nrow(pivoted_rrt), "\n")

# Laboratory 
cat(" - LoadingLaboratory ...\n")
pivoted_lab <- dbGetQuery(con, "
  SELECT 
    le.subject_id,
    le.hadm_id,
    le.charttime,
    MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine,
    MAX(CASE WHEN le.itemid = 51006 THEN le.valuenum END) AS bun,
    MAX(CASE WHEN le.itemid = 50971 THEN le.valuenum END) AS potassium,
    MAX(CASE WHEN le.itemid = 50820 THEN le.valuenum END) AS ph
  FROM mimiciv_hosp.labevents le
  WHERE le.itemid IN (50912, 51006, 50971, 50820)
    AND le.valuenum IS NOT NULL
  GROUP BY le.subject_id, le.hadm_id, le.charttime
")
cat("    Lab events:", nrow(pivoted_lab), "\n")

# Weight durations
cat("  - LoadingweightData...\n")
weight_durations <- dbGetQuery(con, "
  SELECT 
    stay_id,
    starttime,
    endtime,
    weight
  FROM mimiciv_derived.weight_durations
")
cat("    Weight records:", nrow(weight_durations), "\n")

# SOFA scores  
cat("  - Loading SOFA Score...\n")
pivoted_sofa <- dbGetQuery(con, "
  SELECT
    stay_id,
    starttime,
    endtime,
    sofa_24hours
  FROM mimiciv_derived.sofa
")
cat("    SOFA records:", nrow(pivoted_sofa), "\n")

# Vasopressor durations
cat(" - Loading ...\n")
vasopressor_durations <- as.data.frame(dbGetQuery(con, "
  SELECT 
    stay_id,
    starttime,
    endtime,
    vaso_rate,
    vaso_amount
  FROM mimiciv_derived.vasopressin
  UNION ALL
  SELECT
    stay_id,
    starttime,
    endtime,
    rate_norepinephrine AS vaso_rate,
    amount_norepinephrine AS vaso_amount
  FROM mimiciv_derived.norepinephrine
  UNION ALL
  SELECT
    stay_id,
    starttime,
    endtime,
    rate_epinephrine AS vaso_rate,
    amount_epinephrine AS vaso_amount
  FROM mimiciv_derived.epinephrine
"))
cat("    Vasopressor records:", nrow(vasopressor_durations), "\n")

# Ventilation durations
cat("  - Loadingmechanical ventilation...\n")
ventilation_durations <- dbGetQuery(con, "
  SELECT
    stay_id,
    starttime,
    endtime,
    ventilation_status
  FROM mimiciv_derived.ventilation
  WHERE ventilation_status != 'None'
")
cat("    Ventilation records:", nrow(ventilation_durations), "\n")

cat("\n")

# ============================================================================
# step 2: PatientFilterpipeline Copy 
# ============================================================================
cat("step 2: PatientFilter...\n")

# age >= 18 
age_eligible_icustays <- icustay_detail %>% 
  filter(admission_age >= 18) %>% 
  select(stay_id, hadm_id, subject_id, admission_age)
cat(" - age>=18 :", nrow(age_eligible_icustays), "\n")

# KDIGO3 Patient times KDIGO3 
kdigo_stages$charttime <- as.POSIXct(kdigo_stages$charttime)
kdigo_eligible_icustays <- kdigo_stages %>% 
  filter(aki_stage == 3) %>% 
  group_by(stay_id) %>% 
  arrange(charttime) %>%
  filter(row_number() == 1) %>% 
  rename(charttime.kdigo3 = charttime) %>%
  ungroup()
cat("  - KDIGO3Patient:", nrow(kdigo_eligible_icustays), "\n")

# Merge
eligible_icustays1 <- age_eligible_icustays %>% 
  filter(stay_id %in% kdigo_eligible_icustays$stay_id) %>%
  left_join(kdigo_eligible_icustays, by = "stay_id")

# needtoinKDIGO3 hasVMor 
icustay_detail$intime <- as.POSIXct(icustay_detail$intime)
ventilation_durations$starttime <- as.POSIXct(ventilation_durations$starttime)

# vasopressor_durationsis hasData
if(nrow(vasopressor_durations) > 0) {
  vasopressor_durations$starttime <- as.POSIXct(vasopressor_durations$starttime)
  vaso_eligible <- eligible_icustays1 %>%
    left_join(vasopressor_durations, by = "stay_id") %>%
    mutate(vaso_to_aki_hours = as.numeric(difftime(starttime, charttime.kdigo3, units = "hours"))) %>%
    filter(!is.na(vaso_to_aki_hours))
} else {
  cat(" Warning: Dataas only usemechanical ventilation records \n")
  vaso_eligible <- data.frame(stay_id = integer(0))
}

vm_eligible <- eligible_icustays1 %>%
  left_join(ventilation_durations, by = "stay_id") %>%
  mutate(vm_to_aki_hours = as.numeric(difftime(starttime, charttime.kdigo3, units = "hours"))) %>%
  filter(!is.na(vm_to_aki_hours))

eligible_icustays2 <- eligible_icustays1 %>%
  filter(stay_id %in% vaso_eligible$stay_id | stay_id %in% vm_eligible$stay_id)
cat(" - KDIGO3 hasVMor :", nrow(eligible_icustays2), "\n")

# each Patientonly times records ICU stay
eligible_icustays3 <- eligible_icustays2 %>%
  left_join(icustay_detail %>% select(stay_id, intime), by = "stay_id") %>%
  group_by(subject_id) %>%
  arrange(subject_id, intime) %>%
  filter(row_number() == 1) %>%
  ungroup()
cat(" - each Patient timesICU stay:", nrow(eligible_icustays3), "\n")

# KDIGO3 alreadystart RRT Patient
pivoted_rrt$charttime <- as.POSIXct(pivoted_rrt$charttime)
rrt_before_aki <- eligible_icustays3 %>%
  left_join(pivoted_rrt, by = "stay_id") %>%
  filter(charttime < charttime.kdigo3, dialysis_active == 1) %>%
  group_by(subject_id) %>%
  arrange(subject_id, charttime) %>%
  filter(row_number() == 1) %>%
  ungroup()

eligible_icustays4 <- eligible_icustays3 %>%
  filter(!stay_id %in% rrt_before_aki$stay_id)
cat(" - KDIGO3 RRT:", nrow(eligible_icustays4), "\n")

# times RRT 
firstrrt <- pivoted_rrt %>%
  rename(charttime.rrt = charttime) %>%
  filter(dialysis_active == 1) %>%
  group_by(stay_id) %>%
  arrange(stay_id, charttime.rrt) %>%
  filter(row_number() == 1) %>%
  ungroup()

# Mergeto Dataset
icustay_detail$dod <- as.POSIXct(icustay_detail$dod)
dat <- eligible_icustays4 %>%
  left_join(firstrrt, by = "stay_id") %>%
  left_join(icustay_detail %>% select(stay_id, dod, gender), by = "stay_id") %>%
  mutate(
    aki_to_rrt_hours = as.numeric(difftime(charttime.rrt, charttime.kdigo3, units = "hours")),
    aki_to_death_days = as.numeric(difftime(dod, charttime.kdigo3, units = "days"))
  )

cat(" - Final Column:", nrow(dat), "Patient\n")
cat(" - RRTstart:", sum(!is.na(dat$aki_to_rrt_hours)), " \n\n")

# ============================================================================
# step 3: ExtractFeature k=1, k=2, k=3 
# ============================================================================
cat("step 3: Extractdecision pointFeature...\n")

# Weight (most recent before KDIGO3)
weight_durations$starttime <- as.POSIXct(weight_durations$starttime)
temp_weight <- dat %>%
  left_join(weight_durations, by = "stay_id") %>%
  mutate(weight_to_aki = as.numeric(difftime(starttime, charttime.kdigo3, units = "hours"))) %>%
  filter(weight_to_aki <= 0) %>%
  group_by(stay_id) %>%
  arrange(stay_id, desc(weight_to_aki)) %>%
  filter(row_number() == 1) %>%
  ungroup() %>%
  select(stay_id, weight)

dat <- dat %>% left_join(temp_weight, by = "stay_id")
cat("  - Weight:", sum(!is.na(dat$weight)), "/", nrow(dat), "\n")

# SOFAScore
pivoted_sofa$starttime <- as.POSIXct(pivoted_sofa$starttime)
temp_sofa <- dat %>%
  left_join(pivoted_sofa, by = "stay_id") %>%
  mutate(sofa_to_aki = as.numeric(difftime(starttime, charttime.kdigo3, units = "hours"))) %>%
  filter(sofa_to_aki <= 0) %>%
  group_by(stay_id) %>%
  arrange(stay_id, desc(sofa_to_aki)) %>%
  filter(row_number() == 1) %>%
  ungroup() %>%
  select(stay_id, sofa_24hours)

dat <- dat %>% left_join(temp_sofa, by = "stay_id")
cat("  - SOFA:", sum(!is.na(dat$sofa_24hours)), "/", nrow(dat), "\n")

# MergeLaboratory 
pivoted_lab$charttime <- as.POSIXct(pivoted_lab$charttime)

# Function Extractktime pointLaboratoryvalue
get_lab_feature <- function(dat, kdigo_stages, pivoted_lab, 
                            time_start, time_end, 
                            feature_name) {
  
  # Extract urine output from kdigo_stages
  if (grepl("uo_", feature_name)) {
    uo_col <- paste0("uo_rt_24hr")
    kdigo_stages$charttime <- as.POSIXct(kdigo_stages$charttime)
    
    temp <- dat %>%
      select(stay_id, charttime.kdigo3) %>%
      left_join(kdigo_stages %>% select(stay_id, charttime, !!sym(uo_col)), 
                by = "stay_id") %>%
      mutate(
        time_diff = as.numeric(difftime(charttime, charttime.kdigo3, units = "hours"))
      ) %>%
      filter(time_diff >= time_start, time_diff < time_end, !is.na(!!sym(uo_col))) %>%
      group_by(stay_id) %>%
      arrange(stay_id, desc(time_diff)) %>%
      filter(row_number() == 1) %>%
      ungroup() %>%
      select(stay_id, value = !!sym(uo_col))
    
    names(temp)[2] <- feature_name
    return(temp)
  }
  
  # Extract other indicators from pivoted_lab
  lab_mapping <- c(
    bun = "bun",
    pot = "potassium", 
    ph = "ph",
    creat = "creatinine"
  )
  
  base_name <- gsub("_k[0-9]", "", feature_name)
  lab_col <- lab_mapping[[base_name]]
  
  if (is.null(lab_col)) return(NULL)
  
  temp <- dat %>%
    select(hadm_id, charttime.kdigo3) %>%
    left_join(pivoted_lab, by = "hadm_id") %>%
    mutate(
      time_diff = as.numeric(difftime(charttime, charttime.kdigo3, units = "hours"))
    ) %>%
    filter(time_diff >= time_start, time_diff < time_end, !is.na(!!sym(lab_col))) %>%
    group_by(hadm_id) %>%
    arrange(hadm_id, desc(time_diff)) %>%
    filter(row_number() == 1) %>%
    ungroup() %>%
    select(hadm_id, value = !!sym(lab_col))
  
  names(temp)[2] <- feature_name
  return(temp)
}

# k=1 Feature KDIGO3 andbefore 
cat("  - Extract k=1 Feature...\n")
for (feat in c("uo_k1", "bun_k1", "pot_k1", "ph_k1", "creat_k1")) {
  temp <- get_lab_feature(dat, kdigo_stages, pivoted_lab, -Inf, 0, feat)
  if (!is.null(temp)) {
    if (feat == "uo_k1") {
      dat <- dat %>% left_join(temp, by = "stay_id")
    } else {
      dat <- dat %>% left_join(temp, by = "hadm_id")
    }
  }
}

# k=2 Feature KDIGO30-24h post 
cat("  - Extract k=2 Feature...\n")
for (feat in c("uo_k2", "bun_k2", "pot_k2", "ph_k2", "creat_k2")) {
  temp <- get_lab_feature(dat, kdigo_stages, pivoted_lab, 0, 24, feat)
  if (!is.null(temp)) {
    if (feat == "uo_k2") {
      dat <- dat %>% left_join(temp, by = "stay_id")
    } else {
      dat <- dat %>% left_join(temp, by = "hadm_id")
    }
  }
}

# k=3 Feature KDIGO324-48h post 
cat("  - Extract k=3 Feature...\n")
for (feat in c("uo_k3", "bun_k3", "pot_k3", "ph_k3", "creat_k3")) {
  temp <- get_lab_feature(dat, kdigo_stages, pivoted_lab, 24, 48, feat)
  if (!is.null(temp)) {
    if (feat == "uo_k3") {
      dat <- dat %>% left_join(temp, by = "stay_id")
    } else {
      dat <- dat %>% left_join(temp, by = "hadm_id")
    }
  }
}

# ============================================================================
# step 4: decision pointData a1, a2, a3 
# ============================================================================
cat("\nstep 4: decision pointData...\n")

# RRT aki_to_rrt_hours 
dat <- dat %>%
  mutate(
    a1 = as.integer(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours < 24),
    a2 = as.integer(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours >= 24 & aki_to_rrt_hours < 48),
    a3 = as.integer(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours >= 48 & aki_to_rrt_hours < 72),
    hfd = pmax(0, 90 - pmin(90, as.numeric(aki_to_death_days)))
  )

cat("  - a1 (0-24h RRT):", sum(dat$a1), "/", nrow(dat), "\n")
cat("  - a2 (24-48h RRT):", sum(dat$a2), "/", nrow(dat), "\n")
cat("  - a3 (48-72h RRT):", sum(dat$a3), "/", nrow(dat), "\n")

# ============================================================================
# step 5: SaveResults
# ============================================================================
cat("\nstep 5: SaveData...\n")

output_dir <- "../data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- file.path(output_dir, "mimic_dwols_preprocessed.csv")
write.csv(dat, output_file, row.names = FALSE)

cat(" - Save :", output_file, "\n")
cat("  - Sample Size:", nrow(dat), "\n")
cat("  - Features:", ncol(dat), "\n")

# FeatureFull 
cat("\nFeatureFull :\n")
key_features <- c("uo_k1", "uo_k2", "uo_k3", "bun_k1", "pot_k1", "ph_k1", "weight", "sofa_24hours")
for (feat in key_features) {
  if (feat %in% colnames(dat)) {
    completeness <- sum(!is.na(dat[[feat]])) / nrow(dat) * 100
    cat(sprintf("  - %-15s: %5.1f%% Full \n", feat, completeness))
  }
}

# Close database connection
dbDisconnect(con)

cat("\n")
cat(rep("=", 80), "\n")
cat("dWOLS PreprocessingComplete \n")
cat(rep("=", 80), "\n")

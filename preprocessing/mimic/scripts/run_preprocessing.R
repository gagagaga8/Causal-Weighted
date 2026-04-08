# ============================================
# MIMIC-IV data preprocessing (unit-aligned with eICU)
# Main study node: k2 (0-24h post KDIGO3) - a1 as outcome
# Time-series features k1/k2/k3, treatment labels a1/a2/a3
# ============================================

library(dplyr)

# 1. Load data
source("load_from_database.R")

message("\n=== StartDataPreprocessingpipeline ===")

# ============================================
# step1: Filter standardPatient
# ============================================
message("\nstep1: Filter standardPatient...")

# age>=18 
age_eligible_icustays <- icustay_detail %>% 
  filter(admission_age>=18) %>% 
  select(icustay_id, hadm_id, subject_id)
message(paste0(" age>=18 : ", nrow(age_eligible_icustays), " "))

# ESRD
ckd_ineligible_subject <- esrd_patients %>% rename(subject_id = subject_id, hadm_id = hadm_id)
message(paste0(" ESRDPatient: ", nrow(ckd_ineligible_subject), " "))

# KDIGO3 
kdigo_eligible_icustays <- kdigo_stages %>% filter(aki_stage==3) %>% 
  mutate(charttime=as.POSIXct(charttime, format="%Y-%m-%d %H:%M:%OS")) %>%
  group_by(icustay_id) %>% arrange(charttime) %>%
  filter(row_number()==1) %>% rename(charttime.kdigo3 = charttime)
message(paste0(" KDIGO3Patient: ", nrow(kdigo_eligible_icustays), " "))

# Merge records 
eligible_icustays_temp <- age_eligible_icustays %>% 
  filter(!subject_id %in% ckd_ineligible_subject$subject_id, 
         icustay_id %in% kdigo_eligible_icustays$icustay_id)
eligible_icustays1 <- merge(x=eligible_icustays_temp, y=kdigo_eligible_icustays, by="icustay_id", all.x=TRUE)
message(paste0(" age+No ESRD+KDIGO3: ", nrow(eligible_icustays1), " "))

# mechanical ventilation records 
vm_eligible_icustay <- merge(eligible_icustays1, ventilation_durations, by="icustay_id", all.x = TRUE) %>%
  mutate(starttime=as.POSIXct(starttime, format="%Y-%m-%d %H:%M:%OS"),
         endtime=as.POSIXct(endtime, format="%Y-%m-%d %H:%M:%OS")) %>%
  mutate(vm_to_aki_hours=difftime(starttime, charttime.kdigo3, unit="hours")) %>%
  filter(!is.na(vm_to_aki_hours))

# records 
vaso_eligible_icustay <- merge(eligible_icustays1, vasopressor_durations, by="icustay_id", all.x = TRUE) %>%
  mutate(starttime=as.POSIXct(starttime, format="%Y-%m-%d %H:%M:%OS"),
         endtime=as.POSIXct(endtime, format="%Y-%m-%d %H:%M:%OS")) %>%
  mutate(vaso_to_aki_hours=difftime(starttime, charttime.kdigo3, unit="hours")) %>%
  filter(!is.na(vaso_to_aki_hours))

eligible_icustays2 <- eligible_icustays1 %>% 
  filter(icustay_id %in% vm_eligible_icustay$icustay_id | icustay_id %in% vaso_eligible_icustay$icustay_id)
message(paste0(" +mechanical ventilationor : ", nrow(eligible_icustays2), " "))

eligible_icustays3 <- merge(x=eligible_icustays2, y=icustay_detail, by="icustay_id", all.x = TRUE, suffixes = c("", ".y")) %>%
  group_by(subject_id) %>% arrange(subject_id, icu_intime) %>% filter(row_number()==1) %>%
  select(-ends_with(".y"))
message(paste0(" : ", nrow(eligible_icustays3), " "))

# KDIGO3 alreadyAcceptRRTPatient
rrt_before_aki <- merge(eligible_icustays3, pivoted_rrt, by = "icustay_id", all = TRUE) %>% 
  mutate(charttime=as.POSIXct(charttime, format="%Y-%m-%d %H:%M:%OS")) %>%
  rename(charttime.rrt = charttime) %>%
  filter(charttime.rrt<charttime.kdigo3, dialysis_active==1) %>%
  group_by(subject_id) %>% arrange(subject_id, charttime.rrt) %>% filter(row_number()==1)

eligible_icustays4 <- filter(eligible_icustays3, !(icustay_id %in% rrt_before_aki$icustay_id))
message(paste0(" KDIGO3 RRT: ", nrow(eligible_icustays4), " "))

# ============================================
# step2: RRTInfo
# ============================================
message("\nstep2: RRTInfo...")

firstrrt <- pivoted_rrt %>% rename(charttime.rrt=charttime) %>%
  filter(dialysis_active==1) %>% group_by(icustay_id) %>% arrange(icustay_id, charttime.rrt) %>%
  filter(row_number()==1)

dat <- merge(x=eligible_icustays4, y=firstrrt, by="icustay_id", all.x = TRUE) %>%
  mutate(dod=as.POSIXct(dod, format="%Y-%m-%d %H:%M:%OS"),
         aki_to_rrt_hours=as.numeric(difftime(charttime.rrt, charttime.kdigo3, unit="hours")),
         aki_to_death_days=as.numeric(difftime(dod, charttime.kdigo3, unit="days"))) 

message(paste0(" hasRRT : ", sum(!is.na(dat$aki_to_rrt_hours)), " "))

# weightandSOFAScore
message(" weightandSOFA...")

# Weight - use most recent record before KDIGO3
baseline_weight <- dat %>%
  left_join(weight_durations %>% select(icustay_id, starttime, weight), by="icustay_id") %>%
  mutate(
    weight_starttime = as.POSIXct(starttime, format="%Y-%m-%d %H:%M:%OS"),
    weight_to_aki = as.numeric(difftime(weight_starttime, charttime.kdigo3, units="hours"))
  ) %>%
  filter(weight_to_aki <= 0) %>%
  group_by(icustay_id) %>%
  arrange(desc(weight_to_aki)) %>%
  filter(row_number() == 1) %>%
  select(icustay_id, weight) %>%
  ungroup()

dat <- dat %>% left_join(baseline_weight, by="icustay_id")

# SOFAScore - usefirst_day_sofa
if(exists("pivoted_sofa") && "sofa_24hours" %in% colnames(pivoted_sofa)) {
  sofa_data <- pivoted_sofa %>% select(icustay_id, sofa_24hours)
} else if(exists("pivoted_sofa") && "sofa" %in% colnames(pivoted_sofa)) {
  sofa_data <- pivoted_sofa %>% select(icustay_id, sofa) %>% rename(sofa_24hours = sofa)
} else {
  sofa_data <- first_day_sofa %>% select(icustay_id, sofa_24hours = sofa)
}
dat <- dat %>% left_join(sofa_data, by="icustay_id")

message(paste0(" hasWeight: ", sum(!is.na(dat$weight)), " "))
message(paste0(" hasSOFA: ", sum(!is.na(dat$sofa_24hours)), " "))

# LaboratoryFeature (bun, potassium, ph) - ask1Baseline
message(" LaboratoryFeature...")

# Get BUN and potassium from first_day_lab
lab_features <- first_day_lab %>%
  select(icustay_id, bun_min, bun_max, potassium_min, potassium_max) %>%
  mutate(
    bun_k1 = (bun_min + bun_max) / 2,
    pot_k1 = (potassium_min + potassium_max) / 2
  ) %>%
  select(icustay_id, bun_k1, pot_k1)

dat <- dat %>% left_join(lab_features, by="icustay_id")

# Get pH from first_day_bg
if(exists("first_day_bg") && "ph_min" %in% colnames(first_day_bg)) {
  bg_features <- first_day_bg %>%
    select(icustay_id, ph_min, ph_max) %>%
    mutate(ph_k1 = (ph_min + ph_max) / 2) %>%
    select(icustay_id, ph_k1)
  dat <- dat %>% left_join(bg_features, by="icustay_id")
}

message(paste0(" hasBUN: ", sum(!is.na(dat$bun_k1)), " "))
message(paste0(" hasPotassium: ", sum(!is.na(dat$pot_k1)), " "))
if("ph_k1" %in% colnames(dat)) {
  message(paste0(" haspH: ", sum(!is.na(dat$ph_k1)), " "))
}

# ============================================
# Step 3: Temporal Feature (k1/k2/k3)
# ============================================
message("\nStep 3: Temporal Feature...")

# needtoicustay_idColumnTable
needed_icustays <- unique(dat$icustay_id)

# Filterkdigo_stagestoCorrelationPatient
kdigo_filtered <- kdigo_stages %>%
  filter(icustay_id %in% needed_icustays) %>%
  select(icustay_id, charttime, creat, uo_rt_24hr) %>%
  rename(charttime_lab = charttime, creat_val = creat, uo_val = uo_rt_24hr)

# Createdat Versionuse join
dat_keys <- dat %>% select(icustay_id, charttime.kdigo3) %>% distinct()

# k1: KDIGO3 times Baseline 
lab_k1 <- dat_keys %>%
  left_join(kdigo_filtered, by="icustay_id", relationship="many-to-many") %>%
  mutate(charttime_lab = as.POSIXct(charttime_lab, format="%Y-%m-%d %H:%M:%OS"),
         lab_to_aki = as.numeric(difftime(charttime_lab, charttime.kdigo3, units="hours"))) %>%
  filter(lab_to_aki <= 0, !is.na(creat_val)) %>%
  group_by(icustay_id) %>%
  arrange(desc(lab_to_aki)) %>%
  filter(row_number() == 1) %>%
  select(icustay_id, creat_k1 = creat_val, uo_k1 = uo_val) %>%
  ungroup()

dat <- dat %>% left_join(lab_k1, by="icustay_id")

# k2: KDIGO30-24h post
lab_k2 <- dat_keys %>%
  left_join(kdigo_filtered, by="icustay_id", relationship="many-to-many") %>%
  mutate(charttime_lab = as.POSIXct(charttime_lab, format="%Y-%m-%d %H:%M:%OS"),
         lab_to_aki = as.numeric(difftime(charttime_lab, charttime.kdigo3, units="hours"))) %>%
  filter(lab_to_aki > 0, lab_to_aki <= 24, !is.na(creat_val)) %>%
  group_by(icustay_id) %>%
  arrange(desc(lab_to_aki)) %>%
  filter(row_number() == 1) %>%
  select(icustay_id, creat_k2 = creat_val, uo_k2 = uo_val) %>%
  ungroup()

dat <- dat %>% left_join(lab_k2, by="icustay_id")

# k2 : bun_k2, pot_k2, ph_k2 from labevents Temporal Extract KDIGO3 0-24h value 
if(exists("pivoted_lab_timeseries") && nrow(pivoted_lab_timeseries) > 0 && "hadm_id" %in% colnames(dat)) {
  pivoted_lab_timeseries$charttime <- as.POSIXct(pivoted_lab_timeseries$charttime, format="%Y-%m-%d %H:%M:%OS")
  dat_keys_hadm <- dat %>% select(hadm_id, icustay_id, charttime.kdigo3) %>% distinct()
  
  for (lab_name in c("bun", "potassium", "ph", "lactate", "bicarbonate")) {
    col_k2 <- paste0(if(lab_name=="potassium") "pot" else lab_name, "_k2")
    lab_col <- lab_name
    if (!lab_col %in% colnames(pivoted_lab_timeseries)) next
    tmp <- dat_keys_hadm %>%
      left_join(pivoted_lab_timeseries %>% select(hadm_id, charttime, !!sym(lab_col)), by="hadm_id") %>%
      mutate(charttime = as.POSIXct(charttime, format="%Y-%m-%d %H:%M:%OS"),
             lab_to_aki = as.numeric(difftime(charttime, charttime.kdigo3, units="hours"))) %>%
      filter(lab_to_aki > 0, lab_to_aki <= 24, !is.na(!!sym(lab_col))) %>%
      group_by(icustay_id) %>%
      arrange(desc(lab_to_aki)) %>%
      filter(row_number() == 1) %>%
      ungroup() %>%
      select(icustay_id, !!sym(col_k2) := !!sym(lab_col))
    dat <- dat %>% left_join(tmp, by="icustay_id")
  }
  message(paste0(" hasbun_k2: ", sum(!is.na(dat$bun_k2)), " "))
  message(paste0(" haspot_k2: ", sum(!is.na(dat$pot_k2)), " "))
  message(paste0(" hasph_k2: ", sum(!is.na(dat$ph_k2)), " "))
  if("lactate_k2" %in% colnames(dat)) message(paste0(" haslactate_k2: ", sum(!is.na(dat$lactate_k2)), " "))
  if("bicarbonate_k2" %in% colnames(dat)) message(paste0(" hasbicarbonate_k2: ", sum(!is.na(dat$bicarbonate_k2)), " "))
  # k1 lactate/bicarbonate KDIGO3 value 
  for (lab_name in c("lactate", "bicarbonate")) {
    if (!lab_name %in% colnames(pivoted_lab_timeseries)) next
    col_k1 <- paste0(lab_name, "_k1")
    tmp <- dat_keys_hadm %>%
      left_join(pivoted_lab_timeseries %>% select(hadm_id, charttime, !!sym(lab_name)), by="hadm_id") %>%
      mutate(charttime = as.POSIXct(charttime, format="%Y-%m-%d %H:%M:%OS"),
             lab_to_aki = as.numeric(difftime(charttime, charttime.kdigo3, units="hours"))) %>%
      filter(lab_to_aki <= 0, !is.na(!!sym(lab_name))) %>%
      group_by(icustay_id) %>%
      arrange(desc(lab_to_aki)) %>%
      filter(row_number() == 1) %>%
      ungroup() %>%
      select(icustay_id, !!sym(col_k1) := !!sym(lab_name))
    dat <- dat %>% left_join(tmp, by="icustay_id")
  }
  if("lactate_k1" %in% colnames(dat)) message(paste0(" haslactate_k1: ", sum(!is.na(dat$lactate_k1)), " "))
  if("bicarbonate_k1" %in% colnames(dat)) message(paste0(" hasbicarbonate_k1: ", sum(!is.na(dat$bicarbonate_k1)), " "))
}

# k3: KDIGO324-48h post
lab_k3 <- dat_keys %>%
  left_join(kdigo_filtered, by="icustay_id", relationship="many-to-many") %>%
  mutate(charttime_lab = as.POSIXct(charttime_lab, format="%Y-%m-%d %H:%M:%OS"),
         lab_to_aki = as.numeric(difftime(charttime_lab, charttime.kdigo3, units="hours"))) %>%
  filter(lab_to_aki > 24, lab_to_aki <= 48, !is.na(creat_val)) %>%
  group_by(icustay_id) %>%
  arrange(desc(lab_to_aki)) %>%
  filter(row_number() == 1) %>%
  select(icustay_id, creat_k3 = creat_val, uo_k3 = uo_val) %>%
  ungroup()

dat <- dat %>% left_join(lab_k3, by="icustay_id")

message(paste0(" hascreat_k1: ", sum(!is.na(dat$creat_k1)), " "))
message(paste0(" hascreat_k2: ", sum(!is.na(dat$creat_k2)), " "))
message(paste0(" hascreat_k3: ", sum(!is.na(dat$creat_k3)), " "))

# ============================================
# Step 4: TreatmentVariable (a1/a2/a3)
# ============================================
message("\nStep 4: TreatmentVariable...")

dat <- dat %>%
  mutate(
    a1 = ifelse(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours >= 0 & aki_to_rrt_hours < 24, 1, 0),
    a2 = ifelse(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours >= 0 & aki_to_rrt_hours < 48, 1, 0),
    a3 = ifelse(!is.na(aki_to_rrt_hours) & aki_to_rrt_hours >= 0 & aki_to_rrt_hours < 72, 1, 0)
  ) %>%
  mutate(
    a1 = ifelse(is.na(a1), 0, a1),
    a2 = ifelse(is.na(a2), 0, a2),
    a3 = ifelse(is.na(a3), 0, a3)
  )

message(paste0(" a1=1 (0-24h RRT): ", sum(dat$a1), " "))
message(paste0(" a2=1 (0-48h RRT): ", sum(dat$a2), " "))
message(paste0(" a3=1 (0-72h RRT): ", sum(dat$a3), " "))

# ============================================
# Step 5: Computation Variable (hfd)
# ============================================
message("\nStep 5: Computation Variable...")

# hfd = 60 days No Hospitalization days 
# needtofromData Discharge andDeathInfo
dat <- dat %>%
  mutate(
    admittime = as.POSIXct(admittime, format="%Y-%m-%d %H:%M:%OS"),
    dischtime = as.POSIXct(dischtime, format="%Y-%m-%d %H:%M:%OS"),
    horizon = charttime.kdigo3 + 60 * 24 * 3600, # 60 days 
    # ComputationHospitalization 
    los_from_aki = as.numeric(difftime(dischtime, charttime.kdigo3, units="days")),
    # hfd = 60 - Hospitalization days Death hfd=0
    hfd = ifelse(hospital_expire_flag == 1, 0,
                 pmax(0, pmin(60, 60 - los_from_aki)))
  )

message(paste0(" hashfd: ", sum(!is.na(dat$hfd)), " "))
message(paste0("  hfdMedian: ", round(median(dat$hfd, na.rm=TRUE), 1), "  days"))

# ============================================
# Step 6: Deathand Save
# ============================================
message("\nStep 5: DeathPatient...")

# 3 days DeathPatient
td_surv <- dat$aki_to_death_days > 3 | is.na(dat$aki_to_death_days)
dat <- dat[td_surv, ]
message(paste0(" 3 days Death : ", nrow(dat), " "))

# ============================================
# SaveData
# ============================================
message("\n=== SaveData ===")

output_dir <- "../../03_DataSplit/data/"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

saveRDS(dat, paste0(output_dir, "mimic_preprocessed.rds"))
write.csv(dat, paste0(output_dir, "mimic_preprocessed.csv"), row.names = FALSE)

message(paste0("Final Patient : ", nrow(dat), " "))
message(paste0("AcceptRRTTreatment: ", sum(dat$a1 | dat$a2 | dat$a3), " (", round(sum(dat$a1 | dat$a2 | dat$a3)/nrow(dat)*100, 1), "%)"))
message(paste0("Data : ", nrow(dat), " Row x ", ncol(dat), " Column"))
message(paste0("\nDataalreadySave : ", output_dir))

# ============================================
# eICU-CRD data preprocessing (unit-aligned with MIMIC)
# Main study node: k2 (0-24h post KDIGO3) - a1 as outcome
# ============================================

# 1. Load data
source("load_from_database.R")

message("\nStarteICUDataPreprocessingpipeline...")

# 2. Extractage eICUage cancanisstring 
patient <- patient %>%
  mutate(
    admission_age = case_when(
      age == "> 89" ~ 90,
      TRUE ~ as.numeric(age)
    )
  )

# ageFilter ≥18 
age_eligible <- patient %>% 
  filter(admission_age >= 18) %>%
  select(patientunitstayid, patienthealthsystemstayid, admission_age, gender, ethnicity, 
         hospitaladmittime24, unitdischargeoffset, hospitaldischargestatus)

message(paste0("age records ICUHospitalization: ", nrow(age_eligible), " "))

# 3. ESRDPatient diagnosisTableICD-9 
esrd_patients <- diagnosis_aki %>%
  filter(icd9code %in% c('5856', '5859')) %>%
  distinct(patientunitstayid)

message(paste0("ESRDPatient: ", nrow(esrd_patients), " "))

# 4. ComputationKDIGO AKI Creatinine 
# Creatinine andConvert
kdigo_stages <- lab_creatinine %>%
  # Convert if Creatininevalue>20 asμmol/L Convertasmg/dL
  mutate(
    labresult_mgdl = case_when(
      labresult > 20 ~ labresult / 88.4,  # μmol/L -> mg/dL
      TRUE ~ labresult
    )
  ) %>%
  group_by(patientunitstayid) %>%
  arrange(labresultoffset) %>%
  mutate(
    baseline_creat = first(labresult_mgdl),
    creat_increase = labresult_mgdl - baseline_creat,
    creat_fold = labresult_mgdl / baseline_creat
  ) %>%
  mutate(
    # KDIGO Stage 3: Creatinine>=4.0 mg/dL or Baseline3 
    aki_stage_creat = case_when(
      labresult_mgdl >= 4.0 | creat_fold >= 3.0 ~ 3,
      labresult_mgdl >= 3.0 | creat_fold >= 2.0 ~ 2,
      creat_increase >= 0.3 | creat_fold >= 1.5 ~ 1,
      TRUE ~ 0
    )
  ) %>%
  filter(aki_stage_creat == 3) %>%
  slice(1) %>% # each Patient timesKDIGO3 
  ungroup() %>%
  select(patientunitstayid, labresultoffset, labresult_mgdl, aki_stage_creat) %>%
  rename(kdigo3_offset = labresultoffset, kdigo3_creat = labresult_mgdl)

message(paste0("KDIGO3phasePatient: ", nrow(kdigo_stages), " "))

# 5. MergeFilter records age+No ESRD+KDIGO3 
eligible_patients <- age_eligible %>%
  filter(!patientunitstayid %in% esrd_patients$patientunitstayid,
         patientunitstayid %in% kdigo_stages$patientunitstayid) %>%
  left_join(kdigo_stages, by = "patientunitstayid")

message(paste0(" standard1 age+No ESRD+KDIGO3 : ", nrow(eligible_patients), " "))

# 6. FilterKDIGO3 hasmechanical ventilationor MedicationPatient
# Filter records useOR 

# mechanical ventilationPatient Package inKDIGO3 records 
vent_eligible <- ventilation %>%
  filter(patientunitstayid %in% eligible_patients$patientunitstayid) %>%
  distinct(patientunitstayid)
message(paste0("mechanical ventilationPatient: ", nrow(vent_eligible), " "))

# MedicationPatient Filter records 
vaso_eligible <- medication %>%
  filter(grepl("norepinephrine|levophed|epinephrine|adrenalin|dopamine|vasopressin|dobutamine|phenylephrine|milrinone", 
               tolower(drugname))) %>%
  filter(patientunitstayid %in% eligible_patients$patientunitstayid) %>%
  distinct(patientunitstayid)
message(paste0(" MedicationPatient: ", nrow(vaso_eligible), " "))

# records i.e. can 
eligible_patients2 <- eligible_patients %>%
  filter(patientunitstayid %in% vent_eligible$patientunitstayid | 
         patientunitstayid %in% vaso_eligible$patientunitstayid)

message(paste0(" standard2 +mechanical ventilationor : ", nrow(eligible_patients2), " "))

# 7. Patient timesICU times 
eligible_patients3 <- eligible_patients2 %>%
  group_by(patienthealthsystemstayid) %>%
  arrange(patientunitstayid) %>%
  slice(1) %>%
  ungroup()

message(paste0(" records Patient: ", nrow(eligible_patients3), " "))

# 8. KDIGO3 alreadyAcceptRRTPatient
rrt_before_kdigo3 <- treatment_rrt %>%
  inner_join(eligible_patients3 %>% select(patientunitstayid, kdigo3_offset), 
             by = "patientunitstayid") %>%
  filter(treatmentoffset < kdigo3_offset) %>%
  distinct(patientunitstayid)

eligible_patients4 <- eligible_patients3 %>%
  filter(!patientunitstayid %in% rrt_before_kdigo3$patientunitstayid)

message(paste0(" KDIGO3 alreadyAcceptRRTPatient : ", nrow(eligible_patients4), " "))

# 9. timesRRT 
first_rrt <- treatment_rrt %>%
  group_by(patientunitstayid) %>%
  arrange(treatmentoffset) %>%
  slice(1) %>%
  ungroup() %>%
  select(patientunitstayid, treatmentoffset) %>%
  rename(rrt_offset = treatmentoffset)

# 10. MergeData ComputationAKI RRT 
dat <- eligible_patients4 %>%
  left_join(first_rrt, by = "patientunitstayid") %>%
  mutate(
    aki_to_rrt_hours = (rrt_offset - kdigo3_offset) / 60, # offset isminutes
    received_rrt = !is.na(rrt_offset)
  )

message(paste0("\nFinal Patient : ", nrow(dat), " "))
message(paste0("AcceptRRTTreatment: ", sum(dat$received_rrt, na.rm=TRUE), " "))
message(paste0("AKI RRTin : ", 
               round(median(dat$aki_to_rrt_hours, na.rm=TRUE), 2), "  hours"))

# ============================================
# 10.5 weightFeature
# ============================================
message("\nstep10.5: weightFeature...")
if(exists("patient_weight") && nrow(patient_weight) > 0) {
  dat <- dat %>% left_join(patient_weight %>% select(patientunitstayid, weight), by="patientunitstayid")
  message(paste0("hasWeight: ", sum(!is.na(dat$weight)), " "))
} else {
  dat$weight <- NA
}

# when Creatinine creat - useKDIGO3 Creatininevalue
dat <- dat %>% mutate(creat = kdigo3_creat)

# ============================================
# 11. Temporal Feature (k1/k2/k3)
# ============================================
message("\nstep11: Temporal Feature...")

needed_patients <- unique(dat$patientunitstayid)

# CreatinineTemporal Feature
creat_filtered <- lab_creatinine %>%
  filter(patientunitstayid %in% needed_patients) %>%
  mutate(
    labresult_mgdl = case_when(
      labresult > 20 ~ labresult / 88.4,
      TRUE ~ labresult
    )
  )

dat_keys <- dat %>% select(patientunitstayid, kdigo3_offset) %>% distinct()

# k1: KDIGO3 times
creat_k1 <- dat_keys %>%
  left_join(creat_filtered %>% select(patientunitstayid, labresultoffset, labresult_mgdl), 
            by="patientunitstayid", relationship="many-to-many") %>%
  filter(labresultoffset <= kdigo3_offset) %>%
  group_by(patientunitstayid) %>%
  arrange(desc(labresultoffset)) %>%
  slice(1) %>%
  select(patientunitstayid, creat_k1 = labresult_mgdl) %>%
  ungroup()

dat <- dat %>% left_join(creat_k1, by="patientunitstayid")

# k2: KDIGO30-24h post
creat_k2 <- dat_keys %>%
  left_join(creat_filtered %>% select(patientunitstayid, labresultoffset, labresult_mgdl), 
            by="patientunitstayid", relationship="many-to-many") %>%
  mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
  filter(hours_after > 0, hours_after <= 24) %>%
  group_by(patientunitstayid) %>%
  arrange(desc(labresultoffset)) %>%
  slice(1) %>%
  select(patientunitstayid, creat_k2 = labresult_mgdl) %>%
  ungroup()

dat <- dat %>% left_join(creat_k2, by="patientunitstayid")

# k3: KDIGO324-48h post
creat_k3 <- dat_keys %>%
  left_join(creat_filtered %>% select(patientunitstayid, labresultoffset, labresult_mgdl), 
            by="patientunitstayid", relationship="many-to-many") %>%
  mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
  filter(hours_after > 24, hours_after <= 48) %>%
  group_by(patientunitstayid) %>%
  arrange(desc(labresultoffset)) %>%
  slice(1) %>%
  select(patientunitstayid, creat_k3 = labresult_mgdl) %>%
  ungroup()

dat <- dat %>% left_join(creat_k3, by="patientunitstayid")

message(paste0(" hascreat_k1: ", sum(!is.na(dat$creat_k1)), " "))
message(paste0(" hascreat_k2: ", sum(!is.na(dat$creat_k2)), " "))
message(paste0(" hascreat_k3: ", sum(!is.na(dat$creat_k3)), " "))

# Urine outputTemporal Feature
message("  ExtractUrine outputTemporal Feature...")
uo_filtered <- urine_output %>%
  filter(patientunitstayid %in% needed_patients)

# pair Urine output as mL/kg/h withMIMIC 
# eICU : mL -> Convert: mL / weight(kg) / (h) = mL/kg/h
dat_with_wt <- dat %>% mutate(weight_est = ifelse(is.na(weight) | weight <= 0, 70, weight))

# uo_k1: KDIGO3 24h 
uo_k1_raw <- dat_keys %>%
  left_join(uo_filtered %>% select(patientunitstayid, intakeoutputoffset, cellvaluenumeric), 
            by="patientunitstayid", relationship="many-to-many") %>%
  filter(intakeoutputoffset <= kdigo3_offset) %>%
  group_by(patientunitstayid) %>%
  summarise(uo_ml = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop")

uo_k1 <- uo_k1_raw %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_k1 = pmin(10, pmax(0, uo_ml / weight_est / 24))) %>%
  select(patientunitstayid, uo_k1)

dat <- dat %>% left_join(uo_k1, by="patientunitstayid")
message(paste0(" hasuo_k1 (mL/kg/h): ", sum(!is.na(dat$uo_k1) & dat$uo_k1 > 0), " "))

# uo_k2: KDIGO30-24h post
uo_k2_raw <- dat_keys %>%
  left_join(uo_filtered %>% select(patientunitstayid, intakeoutputoffset, cellvaluenumeric), 
            by="patientunitstayid", relationship="many-to-many") %>%
  mutate(hours_after = (intakeoutputoffset - kdigo3_offset) / 60) %>%
  filter(hours_after > 0, hours_after <= 24) %>%
  group_by(patientunitstayid) %>%
  summarise(uo_ml = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop")

uo_k2 <- uo_k2_raw %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_k2 = pmin(10, pmax(0, uo_ml / weight_est / 24))) %>%
  select(patientunitstayid, uo_k2)

dat <- dat %>% left_join(uo_k2, by="patientunitstayid")
message(paste0(" hasuo_k2 (mL/kg/h): ", sum(!is.na(dat$uo_k2) & dat$uo_k2 > 0), " "))

# uo_k3: KDIGO324-48h post
uo_k3_raw <- dat_keys %>%
  left_join(uo_filtered %>% select(patientunitstayid, intakeoutputoffset, cellvaluenumeric), 
            by="patientunitstayid", relationship="many-to-many") %>%
  mutate(hours_after = (intakeoutputoffset - kdigo3_offset) / 60) %>%
  filter(hours_after > 24, hours_after <= 48) %>%
  group_by(patientunitstayid) %>%
  summarise(uo_ml = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop")

uo_k3 <- uo_k3_raw %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_k3 = pmin(10, pmax(0, uo_ml / weight_est / 24))) %>%
  select(patientunitstayid, uo_k3)

dat <- dat %>% left_join(uo_k3, by="patientunitstayid")
message(paste0(" hasuo_k3 (mL/kg/h): ", sum(!is.na(dat$uo_k3) & dat$uo_k3 > 0), " "))

# Urine output: uo_rt_6hr, uo_rt_12hr, uo_rt_24hr (KDIGO3 ) -> as mL/kg/h
uo_rt <- dat_keys %>%
  left_join(uo_filtered %>% select(patientunitstayid, intakeoutputoffset, cellvaluenumeric), 
            by="patientunitstayid", relationship="many-to-many") %>%
  mutate(hours_before = (kdigo3_offset - intakeoutputoffset) / 60) %>%
  filter(hours_before >= 0)

uo_rt_6hr <- uo_rt %>%
  filter(hours_before <= 6) %>%
  group_by(patientunitstayid) %>%
  summarise(ml_6h = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop") %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_rt_6hr = pmin(10, pmax(0, (ml_6h / 6) / weight_est))) %>%
  select(patientunitstayid, uo_rt_6hr)

uo_rt_12hr <- uo_rt %>%
  filter(hours_before <= 12) %>%
  group_by(patientunitstayid) %>%
  summarise(ml_12h = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop") %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_rt_12hr = pmin(10, pmax(0, (ml_12h / 12) / weight_est))) %>%
  select(patientunitstayid, uo_rt_12hr)

uo_rt_24hr <- uo_rt %>%
  filter(hours_before <= 24) %>%
  group_by(patientunitstayid) %>%
  summarise(ml_24h = sum(cellvaluenumeric, na.rm=TRUE), .groups="drop") %>%
  left_join(dat_with_wt %>% select(patientunitstayid, weight_est), by="patientunitstayid") %>%
  mutate(uo_rt_24hr = pmin(10, pmax(0, (ml_24h / 24) / weight_est))) %>%
  select(patientunitstayid, uo_rt_24hr)

dat <- dat %>% 
  left_join(uo_rt_6hr, by="patientunitstayid") %>%
  left_join(uo_rt_12hr, by="patientunitstayid") %>%
  left_join(uo_rt_24hr, by="patientunitstayid")

message(paste0("  hasuo_rt_6/12/24hr (mL/kg/h): ", 
               sum(!is.na(dat$uo_rt_24hr) & dat$uo_rt_24hr > 0), " "))

# BUNTemporal Feature
if(exists("lab_bun") && nrow(lab_bun) > 0) {
  message("  ExtractBUNTemporal Feature...")
  lab_bun_filtered <- lab_bun %>% filter(patientunitstayid %in% needed_patients)
  
  # bun_k1
  bun_k1 <- dat_keys %>%
    left_join(lab_bun_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    filter(labresultoffset <= kdigo3_offset) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, bun_k1 = bun) %>%
    ungroup()
  dat <- dat %>% left_join(bun_k1, by="patientunitstayid")
  message(paste0(" hasbun_k1: ", sum(!is.na(dat$bun_k1)), " "))
  
  # bun_k2
  bun_k2 <- dat_keys %>%
    left_join(lab_bun_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 0, hours_after <= 24) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, bun_k2 = bun) %>%
    ungroup()
  dat <- dat %>% left_join(bun_k2, by="patientunitstayid")
  message(paste0(" hasbun_k2: ", sum(!is.na(dat$bun_k2)), " "))
  
  # bun_k3
  bun_k3 <- dat_keys %>%
    left_join(lab_bun_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 24, hours_after <= 48) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, bun_k3 = bun) %>%
    ungroup()
  dat <- dat %>% left_join(bun_k3, by="patientunitstayid")
  message(paste0(" hasbun_k3: ", sum(!is.na(dat$bun_k3)), " "))
}

# PotassiumTemporal Feature
if(exists("lab_potassium") && nrow(lab_potassium) > 0) {
  message("  ExtractPotassiumTemporal Feature...")
  lab_pot_filtered <- lab_potassium %>% filter(patientunitstayid %in% needed_patients)
  
  # pot_k1
  pot_k1 <- dat_keys %>%
    left_join(lab_pot_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    filter(labresultoffset <= kdigo3_offset) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, pot_k1 = potassium) %>%
    ungroup()
  dat <- dat %>% left_join(pot_k1, by="patientunitstayid")
  message(paste0(" haspot_k1: ", sum(!is.na(dat$pot_k1)), " "))
  
  # pot_k2
  pot_k2 <- dat_keys %>%
    left_join(lab_pot_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 0, hours_after <= 24) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, pot_k2 = potassium) %>%
    ungroup()
  dat <- dat %>% left_join(pot_k2, by="patientunitstayid")
  message(paste0(" haspot_k2: ", sum(!is.na(dat$pot_k2)), " "))
  
  # pot_k3
  pot_k3 <- dat_keys %>%
    left_join(lab_pot_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 24, hours_after <= 48) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, pot_k3 = potassium) %>%
    ungroup()
  dat <- dat %>% left_join(pot_k3, by="patientunitstayid")
  message(paste0(" haspot_k3: ", sum(!is.na(dat$pot_k3)), " "))
}

# pHTemporal Feature
if(exists("lab_ph") && nrow(lab_ph) > 0) {
  message("  ExtractpHTemporal Feature...")
  lab_ph_filtered <- lab_ph %>% filter(patientunitstayid %in% needed_patients)
  
  # ph_k1
  ph_k1 <- dat_keys %>%
    left_join(lab_ph_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    filter(labresultoffset <= kdigo3_offset) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, ph_k1 = ph) %>%
    ungroup()
  dat <- dat %>% left_join(ph_k1, by="patientunitstayid")
  message(paste0(" hasph_k1: ", sum(!is.na(dat$ph_k1)), " "))
  
  # ph_k2
  ph_k2 <- dat_keys %>%
    left_join(lab_ph_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 0, hours_after <= 24) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, ph_k2 = ph) %>%
    ungroup()
  dat <- dat %>% left_join(ph_k2, by="patientunitstayid")
  message(paste0(" hasph_k2: ", sum(!is.na(dat$ph_k2)), " "))
  
  # ph_k3
  ph_k3 <- dat_keys %>%
    left_join(lab_ph_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 24, hours_after <= 48) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, ph_k3 = ph) %>%
    ungroup()
  dat <- dat %>% left_join(ph_k3, by="patientunitstayid")
  message(paste0(" hasph_k3: ", sum(!is.na(dat$ph_k3)), " "))
}

# LactateTemporal Feature k1+k2 /sepsis-AKI Metric 
if(exists("lab_lactate") && nrow(lab_lactate) > 0) {
  message("  ExtractLactateTemporal Feature...")
  lab_lactate_filtered <- lab_lactate %>% filter(patientunitstayid %in% needed_patients)
  lactate_k1 <- dat_keys %>%
    left_join(lab_lactate_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    filter(labresultoffset <= kdigo3_offset) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, lactate_k1 = lactate) %>%
    ungroup()
  lactate_k2 <- dat_keys %>%
    left_join(lab_lactate_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 0, hours_after <= 24) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, lactate_k2 = lactate) %>%
    ungroup()
  dat <- dat %>% left_join(lactate_k1, by="patientunitstayid") %>% left_join(lactate_k2, by="patientunitstayid")
  message(paste0(" haslactate_k1: ", sum(!is.na(dat$lactate_k1)), ", lactate_k2: ", sum(!is.na(dat$lactate_k2)), " "))
}

# BicarbonateTemporal Feature k1+k2 
if(exists("lab_bicarbonate") && nrow(lab_bicarbonate) > 0) {
  message("  ExtractBicarbonateTemporal Feature...")
  lab_bicarbonate_filtered <- lab_bicarbonate %>% filter(patientunitstayid %in% needed_patients)
  bicarbonate_k1 <- dat_keys %>%
    left_join(lab_bicarbonate_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    filter(labresultoffset <= kdigo3_offset) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, bicarbonate_k1 = bicarbonate) %>%
    ungroup()
  bicarbonate_k2 <- dat_keys %>%
    left_join(lab_bicarbonate_filtered, by="patientunitstayid", relationship="many-to-many") %>%
    mutate(hours_after = (labresultoffset - kdigo3_offset) / 60) %>%
    filter(hours_after > 0, hours_after <= 24) %>%
    group_by(patientunitstayid) %>%
    arrange(desc(labresultoffset)) %>%
    slice(1) %>%
    select(patientunitstayid, bicarbonate_k2 = bicarbonate) %>%
    ungroup()
  dat <- dat %>% left_join(bicarbonate_k1, by="patientunitstayid") %>% left_join(bicarbonate_k2, by="patientunitstayid")
  message(paste0(" hasbicarbonate_k1: ", sum(!is.na(dat$bicarbonate_k1)), ", bicarbonate_k2: ", sum(!is.na(dat$bicarbonate_k2)), " "))
}

# pair APACHEScoremappingasSOFArange(0-24)
# APACHE 0-71 -> SOFA 0-24 scale 0.15 withMIMIC SOFA 
if(exists("apache") && nrow(apache) > 0) {
  message("  ExtractAPACHEScoreand mappingassofa_24hours...")
  apache_scores <- apache %>%
    filter(patientunitstayid %in% needed_patients) %>%
    select(patientunitstayid, apachescore) %>%
    distinct()
  dat <- dat %>% left_join(apache_scores, by="patientunitstayid")
  dat <- dat %>% mutate(sofa_24hours = pmin(24, pmax(0, coalesce(apachescore, 0) * 0.15)))
  message(paste0(" hassofa_24hours: ", sum(!is.na(dat$sofa_24hours)), " "))
}

# ============================================
# 11.5 ComputationAKI Feature
# ============================================
message("\nstep11.5: ComputationAKI Feature...")

# aki_stage: AKI Creatinine 
# alreadyviahasaki_stage_creat use
dat$aki_stage <- dat$aki_stage_creat

# aki_stage_uo: Urine outputAKI 
# KDIGOUrine outputAKI standard: 
# Stage 1: <0.5 ml/kg/hr for 6-12h
# Stage 2: <0.5 ml/kg/hr for >12h
# Stage 3: <0.3 ml/kg/hr for >24h OR anuria >12h
# Version Hypothesisweight70kg 
dat <- dat %>%
  mutate(
    weight_est = ifelse(is.na(weight), 70, weight),
    uo_rate_6hr = uo_rt_6hr / weight_est,
    uo_rate_12hr = uo_rt_12hr / weight_est,
    uo_rate_24hr = uo_rt_24hr / weight_est,
    aki_stage_uo = case_when(
      uo_rate_24hr < 0.3 ~ 3,
      uo_rate_12hr < 0.5 ~ 2,
      uo_rate_6hr < 0.5 ~ 1,
      TRUE ~ 0
    )
  ) %>%
  select(-weight_est, -uo_rate_6hr, -uo_rate_12hr, -uo_rate_24hr)

message(paste0("hasaki_stage: ", sum(!is.na(dat$aki_stage)), " "))
message(paste0("hasaki_stage_uo: ", sum(!is.na(dat$aki_stage_uo)), " "))

# ============================================
# 12. TreatmentVariable a1/a2/a3 with hfd withMIMICpair 
# ============================================
message("\nstep12: TreatmentVariablewith ...")

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

# hfd: 60 days Hospitalization withMIMIC 
# unitdischargeoffset minutes kdigo3_offset minutes
dat <- dat %>%
  mutate(
    los_from_aki_days = (unitdischargeoffset - kdigo3_offset) / (60 * 24),
    hospital_expire_flag = as.integer(grepl("Expired|Death", hospitaldischargestatus, ignore.case = TRUE)),
    hfd = ifelse(hospital_expire_flag == 1, 0, pmax(0, pmin(60, 60 - los_from_aki_days)))
  )

# immunosuppressant MIMIChas eICUcan 0 
if(!"immunosuppressant" %in% colnames(dat)) dat$immunosuppressant <- 0

message(paste0("  a1=1: ", sum(dat$a1), ", a2=1: ", sum(dat$a2), ", a3=1: ", sum(dat$a3)))
message(paste0("  hfdMedian: ", round(median(dat$hfd, na.rm=TRUE), 1), "  days"))

# 3 days Death withMIMIC 
dat <- dat %>% mutate(aki_to_death_days = ifelse(hospital_expire_flag == 1, los_from_aki_days, NA))
td_surv <- dat$aki_to_death_days > 3 | is.na(dat$aki_to_death_days)
dat <- dat[td_surv, ]
message(paste0(" 3 days Death : ", nrow(dat), " "))

# 13. SavePreprocessing processed Data
output_dir <- "../data/"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

saveRDS(dat, paste0(output_dir, "eicu_preprocessed.rds"))
write.csv(dat, paste0(output_dir, "eicu_preprocessed.csv"), row.names = FALSE)

message(paste0("\nPreprocessingComplete DataalreadySave : ", output_dir))
message(paste0("- eicu_preprocessed.rds"))
message(paste0("- eicu_preprocessed.csv"))
message(paste0("\nData : ", nrow(dat), " Row x ", ncol(dat), " Column"))

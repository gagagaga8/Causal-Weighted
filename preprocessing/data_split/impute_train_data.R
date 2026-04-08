# pairTrainingset RowMultiple Imputation
# Parameterssettingswith dataprep_dev.R 

library(mice)
library(dplyr)

cat("========================================\n")
cat("Trainingset multiple imputation (m=100)\n")
cat("========================================\n\n")

# LoadingTrainingset
cat("LoadingTrainingset data...\n")
train_data <- readRDS("../data/train/mimic_train.rds")
cat("Trainingset :", nrow(train_data), "Row x", ncol(train_data), "Column\n\n")

# ConvertdifftimeColumnto numeric
if("aki_to_rrt_hours" %in% colnames(train_data) && inherits(train_data$aki_to_rrt_hours, "difftime")) {
  train_data$aki_to_rrt_hours <- as.numeric(train_data$aki_to_rrt_hours, units="hours")
}
if("aki_to_death_days" %in% colnames(train_data) && inherits(train_data$aki_to_death_days, "difftime")) {
  train_data$aki_to_death_days <- as.numeric(train_data$aki_to_death_days, units="days")
}
if("los_icu" %in% colnames(train_data) && inherits(train_data$los_icu, "difftime")) {
  train_data$los_icu <- as.numeric(train_data$los_icu, units="days")
}
if("los_hospital" %in% colnames(train_data) && inherits(train_data$los_hospital, "difftime")) {
  train_data$los_hospital <- as.numeric(train_data$los_hospital, units="days")
}

# Missing 
cat("Missingvaluestatistics:\n")
missing_counts <- colSums(is.na(train_data))
missing_vars <- missing_counts[missing_counts > 0]
if(length(missing_vars) > 0) {
  for(var in names(missing_vars)) {
    pct <- round(100 * missing_vars[var] / nrow(train_data), 1)
    cat("  ", var, ": ", missing_vars[var], " (", pct, "%)\n", sep="")
  }
  cat("\n")
} else {
  cat("  No Missingvalue\n\n")
}

# willClassificationVariableConvertto factors according to Column 
cat("ConvertClassificationVariableto factors...\n")
# Columnis inagainConvert
if("ethnicity_grouped" %in% colnames(train_data)) {
  train_data$ethnicity_grouped <- as.factor(train_data$ethnicity_grouped)
}
if("gender" %in% colnames(train_data)) {
  train_data$gender <- as.factor(train_data$gender)
}
if("immunosuppressant" %in% colnames(train_data)) {
  train_data$immunosuppressant <- as.factor(train_data$immunosuppressant)
}
if("a1" %in% colnames(train_data)) {
  train_data$a1 <- as.factor(train_data$a1)
}
if("a2" %in% colnames(train_data)) {
  train_data$a2 <- as.factor(train_data$a2)
}
if("a3" %in% colnames(train_data)) {
  train_data$a3 <- as.factor(train_data$a3)
}
# inClassificationVariable
if("race" %in% colnames(train_data)) {
  train_data$race <- as.factor(train_data$race)
}
if("dialysis_type" %in% colnames(train_data)) {
  train_data$dialysis_type <- as.factor(train_data$dialysis_type)
}

# Initialize Imputation settings
cat("Initialize Imputation parameters...\n")

# Columnand varianceColumn
num_cols <- sapply(train_data, is.numeric)
zero_var_cols <- sapply(train_data[, num_cols], function(x) var(x, na.rm=TRUE) < 1e-10)
if(any(zero_var_cols)) {
  zero_var_names <- names(which(zero_var_cols))
  cat(" varianceColumn:", paste(zero_var_names, collapse=", "), "\n")
}

# Use quickpred for predictor selection, avoid multicollinearity
predM <- quickpred(train_data, minpuc=0.25, mincor=0.1)

imp <- mice(train_data, maxit=0)
meth <- imp$method

# settingsnot asPredictionVariableVariable according to Column 
no_predictor_var <- c("subject_id", "subject_id.x", "hadm_id", "hadm_id.x", "icustay_id", 
                      "charttime.kdigo3", "charttime.rrt",
                      "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt",
                      "dod", "admittime", "dischtime", "icu_intime", "icu_outtime",
                      "hospital_expire_flag", "hospstay_seq", "first_hosp_stay",
                      "icustay_seq", "first_icu_stay"
                      )
# onlypair inColumnsettingsas0
for(var in no_predictor_var) {
  if(var %in% colnames(predM)) {
    predM[, var] <- 0
  }
}

# settingsnotImputationVariable according to Column 
do_not_impute_var <- c("dod", "subject_id", "subject_id.x", "hadm_id", "hadm_id.x", 
                       "icustay_id", "charttime.kdigo3", "charttime.rrt", "charttime_kdigo3",
                       "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt",
                       "admittime", "dischtime", "icu_intime", "icu_outtime",
                       "hospital_expire_flag", "hospstay_seq", "first_hosp_stay",
                       "icustay_seq", "first_icu_stay",
                       "dialysis_active", "dialysis_present"
                       # Urine outputData inhas value can Imputation
                       )

# onlypair inColumnsettingsas 
for(var in do_not_impute_var) {
  if(var %in% names(meth)) {
    meth[var] <- ""
  }
}

# ExecuteMultiple Imputation useridgeregularization 
cat("\nStartMultiple Imputation (m=20)...\n")
cat(" cancanneedto 5-15minutes \n\n")

start_time <- Sys.time()
exp_dat_train <- tryCatch({
  mice(train_data, m=20, seed=1, predictorMatrix=predM, method=meth, 
       ridge=0.0001, printFlag=TRUE)
}, error = function(e) {
  cat("\nImputationEncountered Error Try simpler Method...\n")
  # Use simpler imputation method
  meth_simple <- meth
  meth_simple[meth_simple == "pmm"] <- "mean"
  mice(train_data, m=20, seed=1, predictorMatrix=predM, method=meth_simple,
       printFlag=TRUE)
})
end_time <- Sys.time()

cat("\nImputation complete!\n")
cat("Time elapsed:", round(difftime(end_time, start_time, units="mins"), 1), "minutes\n\n")

# SaveImputation processed Data
cat("SaveImputationResults...\n")
save(exp_dat_train, file="../data/train/imputed_mimic_train.RData")
cat("alreadySave: ../data/train/imputed_mimic_train.RData\n\n")

# Save ImputationDataset as 
exp_dat_train_simp <- complete(exp_dat_train, 1)
saveRDS(exp_dat_train_simp, file="../data/train/mimic_train_imputed_1.rds")
cat("alreadySave ImputationDataset: ../data/train/mimic_train_imputed_1.rds\n\n")

cat("========================================\n")
cat("Trainingset imputation complete!\n")
cat("========================================\n")
cat("\nNext: impute test and validation sets separately\n")

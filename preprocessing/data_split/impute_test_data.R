# pairTestset RowMultiple Imputation
# with impute_train_data.R 

library(mice)
library(dplyr)

cat("========================================\n")
cat("Testset multiple imputation (m=100)\n")
cat("========================================\n\n")

# LoadingTestset
cat("LoadingTestset data...\n")
test_data <- readRDS("../data/test/mimic_test.rds")
cat("Testset :", nrow(test_data), "Row x", ncol(test_data), "Column\n\n")

# ConvertdifftimeColumnto numeric
if("aki_to_rrt_hours" %in% colnames(test_data) && inherits(test_data$aki_to_rrt_hours, "difftime")) {
  test_data$aki_to_rrt_hours <- as.numeric(test_data$aki_to_rrt_hours, units="hours")
}
if("aki_to_death_days" %in% colnames(test_data) && inherits(test_data$aki_to_death_days, "difftime")) {
  test_data$aki_to_death_days <- as.numeric(test_data$aki_to_death_days, units="days")
}
if("los_icu" %in% colnames(test_data) && inherits(test_data$los_icu, "difftime")) {
  test_data$los_icu <- as.numeric(test_data$los_icu, units="days")
}
if("los_hospital" %in% colnames(test_data) && inherits(test_data$los_hospital, "difftime")) {
  test_data$los_hospital <- as.numeric(test_data$los_hospital, units="days")
}

# Missingstatistics
cat("Missingvaluestatistics:\n")
missing_counts <- colSums(is.na(test_data))
missing_vars <- missing_counts[missing_counts > 0]
if(length(missing_vars) > 0) {
  for(var in names(missing_vars)) {
    pct <- round(100 * missing_vars[var] / nrow(test_data), 1)
    cat("  ", var, ": ", missing_vars[var], " (", pct, "%)\n", sep="")
  }
  cat("\n")
} else {
  cat("  No Missingvalue\n\n")
}

# willClassificationVariableConvertto factors
cat("ConvertClassificationVariableto factors...\n")
if("ethnicity_grouped" %in% colnames(test_data)) {
  test_data$ethnicity_grouped <- as.factor(test_data$ethnicity_grouped)
}
if("gender" %in% colnames(test_data)) {
  test_data$gender <- as.factor(test_data$gender)
}
if("immunosuppressant" %in% colnames(test_data)) {
  test_data$immunosuppressant <- as.factor(test_data$immunosuppressant)
}
if("a1" %in% colnames(test_data)) {
  test_data$a1 <- as.factor(test_data$a1)
}
if("a2" %in% colnames(test_data)) {
  test_data$a2 <- as.factor(test_data$a2)
}
if("a3" %in% colnames(test_data)) {
  test_data$a3 <- as.factor(test_data$a3)
}
if("race" %in% colnames(test_data)) {
  test_data$race <- as.factor(test_data$race)
}
if("dialysis_type" %in% colnames(test_data)) {
  test_data$dialysis_type <- as.factor(test_data$dialysis_type)
}

# Initialize Imputation settings
cat("Initialize Imputation parameters...\n")

# Use quickpred for automatic predictor selection
predM <- quickpred(test_data, minpuc=0.25, mincor=0.1)

imp <- mice(test_data, maxit=0)
meth <- imp$method

# Not used as predictor
no_predictor_var <- c("subject_id", "subject_id.x", "hadm_id", "hadm_id.x", "icustay_id", 
                      "charttime.kdigo3", "charttime.rrt",
                      "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt",
                      "dod", "admittime", "dischtime", "icu_intime", "icu_outtime",
                      "hospital_expire_flag", "hospstay_seq", "first_hosp_stay",
                      "icustay_seq", "first_icu_stay")
for(var in no_predictor_var) {
  if(var %in% colnames(predM)) {
    predM[, var] <- 0
  }
}

# Variables not to impute
do_not_impute_var <- c("dod", "subject_id", "subject_id.x", "hadm_id", "hadm_id.x", 
                       "icustay_id", "charttime.kdigo3", "charttime.rrt", "charttime_kdigo3",
                       "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt",
                       "admittime", "dischtime", "icu_intime", "icu_outtime",
                       "hospital_expire_flag", "hospstay_seq", "first_hosp_stay",
                       "icustay_seq", "first_icu_stay",
                       "dialysis_active", "dialysis_present")
for(var in do_not_impute_var) {
  if(var %in% names(meth)) {
    meth[var] <- ""
  }
}

cat("\nStartMultiple Imputation (m=20)...\n")
start_time <- Sys.time()
exp_dat_test <- tryCatch({
  mice(test_data, m=20, seed=1, predictorMatrix=predM, method=meth, 
       ridge=0.0001, printFlag=TRUE)
}, error = function(e) {
  cat("\nImputationEncountered Error Try simpler Method...\n")
  meth_simple <- meth
  meth_simple[meth_simple == "pmm"] <- "mean"
  mice(test_data, m=20, seed=1, predictorMatrix=predM, method=meth_simple,
       printFlag=TRUE)
})
end_time <- Sys.time()
cat("Imputation complete!Time elapsed:", round(difftime(end_time, start_time, units="mins"), 1), "minutes\n\n")

cat("SaveImputationResults...\n")
save(exp_dat_test, file="../data/test/imputed_mimic_test.RData")
cat("alreadySave: ../data/test/imputed_mimic_test.RData\n")

exp_dat_test_simp <- complete(exp_dat_test, 1)
saveRDS(exp_dat_test_simp, file="../data/test/mimic_test_imputed_1.rds")
cat("alreadySave ImputationDataset: ../data/test/mimic_test_imputed_1.rds\n\n")

cat("========================================\n")
cat("Testset imputation complete!\n")
cat("========================================\n")

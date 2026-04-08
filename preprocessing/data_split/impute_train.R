# ============================================================================
# Trainingset multiple Imputation (m=100)
# ============================================================================

library(mice)

cat(" Trainingset...\n")
train_data <- readRDS("../data/train/mimic_train.rds")

cat("Trainingset :", nrow(train_data), "x", ncol(train_data), "\n")

# Missingstatistics
cat("\nMissingstatistics:\n")
missing_summary <- sapply(train_data, function(x) sum(is.na(x)))
print(missing_summary[missing_summary > 0])

# ConvertClassificationVariableto factors
cat("\nConvertClassificationVariableto factors...\n")
if("gender" %in% colnames(train_data)) {
  train_data$gender <- as.factor(train_data$gender)
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

# Initialize mice
cat("\nInitialize mice...\n")
imp <- mice(train_data, maxit=0)
predM <- imp$predictorMatrix
meth <- imp$method

# Columns not used as predictors
no_predictor_var <- c(
  "subject_id", "hadm_id", "stay_id", "charttime_kdigo3", "dod"
)

for(var in no_predictor_var) {
  if(var %in% colnames(predM)) {
    predM[, var] <- 0
  }
}

# Columns not to impute
do_not_impute_var <- c(
  "dod", "subject_id", "hadm_id", "stay_id", "charttime_kdigo3",
  "a1", "a2", "a3"  # TreatmentVariablenotImputation
)

for(var in do_not_impute_var) {
  if(var %in% names(meth)) {
    meth[var] <- ""
  }
}

# StartMultiple Imputation
cat("\nStartMultiple Imputation (m=100)...\n")
cat(" cancanneedto minutes ...\n")

exp_dat_train <- mice(
  train_data, 
  m = 100, 
  seed = 1, 
  predictorMatrix = predM, 
  method = meth
)

# SaveResults
cat("\nSaveImputationResults...\n")
save(exp_dat_train, file = "../data/train/imputed_mimic_train.RData")

# Save ImputationDatasetuse 
train_imputed_1 <- complete(exp_dat_train, 1)
saveRDS(train_imputed_1, "../data/train/mimic_train_imputed_1.rds")

cat("\nTrainingset multiple Imputation complete!\n")
cat("- Imputation times : m =", exp_dat_train$m, "\n")
cat("- Sample Size:", nrow(exp_dat_train$data), "\n")
cat("- OutputFile: ../data/train/imputed_mimic_train.RData\n")

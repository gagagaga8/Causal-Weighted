# MIMICDatasetSplit 
# Split into train/test/val sets at 7:2:1 ratio
# Split before imputation to ensure consistent splits across imputed datasets

library(mice)
library(dplyr)

# LoadingPathConfiguration
if(file.exists("../config_paths.R")) {
  source("../config_paths.R")
} else {
  LEARNING_DIR <- getwd()
}

cat("========================================\n")
cat("MIMICDatasetSplit 7:2:1 \n")
cat("========================================\n\n")

# LoadingImputation processed Data
# Directory from00_mimicPreprocessingdata/Loading
if(file.exists("../00_mimicPreprocessingdata/imputed_mimic_dtr.RData")) {
  load("../00_mimicPreprocessingdata/imputed_mimic_dtr.RData")
  cat("alreadyLoading: ../00_mimicPreprocessingdata/imputed_mimic_dtr.RData\n")
} else if(file.exists(file.path(LEARNING_DIR, "imputed_mimic_dtr.RData"))) {
  load(file.path(LEARNING_DIR, "imputed_mimic_dtr.RData"))
  cat("alreadyLoading: imputed_mimic_dtr.RData\n")
} else if(file.exists("imputed_mimic_dtr.RData")) {
  load("imputed_mimic_dtr.RData")
  cat("alreadyLoading: imputed_mimic_dtr.RData\n")
} else {
  stop("Cannot find imputed_mimic_dtr.RData")
}

# Dataobject
if(!exists("exp_dat") || class(exp_dat) != "mids") {
  stop("exp_datobjectnot inor notismidsobject")
}

cat("\n DataInfo:\n")
cat(" ImputationDataset (m):", exp_dat$m, "\n")
cat(" Variable :", ncol(exp_dat$data), "\n")

# ImputationDataset Row 
imp_1 <- complete(exp_dat, 1)
n_total <- nrow(imp_1)
cat(" Row :", n_total, "\n\n")

# settingsRandom types canrepeat 
set.seed(123)

# ComputationSplit 7:2:1 
# Trainingset: 70%
# Testset: 20%
# Validationset: 10%
train_size <- floor(0.7 * n_total)
test_size <- floor(0.2 * n_total)
val_size <- n_total - train_size - test_size # asValidationset

cat("SplitRatio:\n")
cat("  Trainingset: ", train_size, " (", round(100*train_size/n_total, 1), "%)\n", sep="")
cat("  Testset: ", test_size, " (", round(100*test_size/n_total, 1), "%)\n", sep="")
cat("  Validationset: ", val_size, " (", round(100*val_size/n_total, 1), "%)\n\n", sep="")

# GenerateRandomIndex
shuffled_indices <- sample(1:n_total, n_total, replace = FALSE)

# SplitIndex
train_indices <- shuffled_indices[1:train_size]
test_indices <- shuffled_indices[(train_size + 1):(train_size + test_size)]
val_indices <- shuffled_indices[(train_size + test_size + 1):n_total]

cat("IndexSplitComplete\n")
cat("  TrainingsetIndexrange: 1 -", train_size, "\n")
cat("  TestsetIndexrange:", train_size + 1, "-", train_size + test_size, "\n")
cat("  ValidationsetIndexrange:", train_size + test_size + 1, "-", n_total, "\n\n")

# SaveSplitIndex use All ImputationDataset 
save(train_indices, test_indices, val_indices, 
     train_size, test_size, val_size,
     file = "../data/mimic_split_indices.RData")

cat("SplitIndexalreadySaveto: mimic_split_indices.RData\n\n")

# paireach ImputationDataset RowSplit
cat("StartSplitAll ImputationDataset...\n")

# Extract Data
original_data <- exp_dat$data

# CreateTrainingset Testset Validationset Data 
train_data <- original_data[train_indices, ]
test_data <- original_data[test_indices, ]
val_data <- original_data[val_indices, ]

# Saveeach ImputationDatasetTraining/Test/Validationset
cat("CreateSplit processed ImputationDataset...\n")

train_imp_list <- list()
test_imp_list <- list()
val_imp_list <- list()

for(i in 1:exp_dat$m) {
  imp_i <- complete(exp_dat, i)
  
  train_imp_list[[i]] <- imp_i[train_indices, ]
  test_imp_list[[i]] <- imp_i[test_indices, ]
  val_imp_list[[i]] <- imp_i[val_indices, ]
  
  if(i %% 10 == 0) {
    cat("  alreadyComplete", i, "/", exp_dat$m, " ImputationDataset\n")
  }
}

# CreateTrainingsetmidsobject
# Method usemice() Createmidsobject but useTrainingset data
# needto RunImputation butin order to use ImputationResults

# Method Create Package midsobject
# Use first imputed dataset as template
train_mids_template <- mice(train_data, maxit = 0, m = exp_dat$m)
train_mids <- train_mids_template

# willSplit processed ImputationDatasetfilltomidsobjectin
# midsobjectInternal compare use Method
# usetrain_imp_list and inneedto Convertasmids 

# SaveSplit processed Data
cat("\nSaveSplit processed Data...\n")

# CreateTrainingsetmidsobject
# Method usemice Create but ImputationResults
# For compatibility, create wrapper function to create mids object from train_imp_list

# CreateTrainingsetmidsobject 
# seedis in
if(is.null(exp_dat$seed)) {
  exp_dat_train <- mice(train_data, maxit = 0, m = exp_dat$m, seed = 1)
} else {
  exp_dat_train <- mice(train_data, maxit = 0, m = exp_dat$m, seed = exp_dat$seed)
}
# UpdateData
exp_dat_train$data <- train_data

# Directory Savetodata Directory
# SaveTrainingset Package midsobjectandData ColumnTable 
save(exp_dat_train, train_imp_list, train_data,
     file = "../data/train/imputed_mimic_dtr_train.RData")
cat("  TrainingsetalreadySave: ../data/train/imputed_mimic_dtr_train.RData\n")

# SaveTestset Data ColumnTable 
save(test_imp_list, test_data,
     file = "../data/test/imputed_mimic_dtr_test.RData")
cat("  TestsetalreadySave: ../data/test/imputed_mimic_dtr_test.RData\n")

# SaveValidationset Data ColumnTable 
save(val_imp_list, val_data,
     file = "../data/val/imputed_mimic_dtr_val.RData")
cat("  ValidationsetalreadySave: ../data/val/imputed_mimic_dtr_val.RData\n")

# SaveSplitInfo
split_info <- list(
  train_size = train_size,
  test_size = test_size,
  val_size = val_size,
  train_indices = train_indices,
  test_indices = test_indices,
  val_indices = val_indices,
  n_total = n_total,
  seed = 123
)
save(split_info, file = "../data/mimic_split_info.RData")
cat("  SplitInfoalreadySave: ../data/mimic_split_info.RData\n\n")

cat("========================================\n")
cat("DataSplitComplete \n")
cat("========================================\n")
cat("\nUsage:\n")
cat("1. Trainingsetuse dWOLSpolicy \n")
cat("2. Testsetuse ModelEvaluationand \n")
cat("3. Validationsetuse Final Validation\n")
cat("\nFile :\n")
cat("  - Trainingset: imputed_mimic_dtr_train.RData\n")
cat("  - Testset: imputed_mimic_dtr_test.RData\n")
cat("  - Validationset: imputed_mimic_dtr_val.RData\n")
cat("  - SplitInfo: mimic_split_info.RData\n")


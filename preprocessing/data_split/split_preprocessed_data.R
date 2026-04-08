# ============================================
# MIMIC preprocessed data split script
# Split raw preprocessed data 7:2:1 before multiple imputation
# ============================================

library(dplyr)

cat("========================================\n")
cat("MIMIC preprocessed data split (7:2:1)\n")
cat("========================================\n\n")

# LoadingPreprocessing processed Data
data_path <- "../data/mimic_preprocessed.rds"
if(!file.exists(data_path)) {
  stop(" nottoFile: ", data_path)
}

dat <- readRDS(data_path)
cat("alreadyLoading: ", data_path, "\n")
cat("Data : ", nrow(dat), " Row x ", ncol(dat), " Column\n\n")

# settingsRandom types canrepeat 
set.seed(123)

# ComputationSplit 7:2:1 
n_total <- nrow(dat)
train_size <- floor(0.7 * n_total)
test_size <- floor(0.2 * n_total)
val_size <- n_total - train_size - test_size

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

# SplitData
train_data <- dat[train_indices, ]
test_data <- dat[test_indices, ]
val_data <- dat[val_indices, ]

cat("IndexSplitComplete\n")
cat("  Trainingset: ", nrow(train_data), " Row\n")
cat("  Testset: ", nrow(test_data), " Row\n")
cat("  Validationset: ", nrow(val_data), " Row\n\n")

# CreateOutputDirectory
train_dir <- "../data/train/"
test_dir <- "../data/test/"
val_dir <- "../data/val/"

if(!dir.exists(train_dir)) dir.create(train_dir, recursive = TRUE)
if(!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)
if(!dir.exists(val_dir)) dir.create(val_dir, recursive = TRUE)

# SaveSplit processed Data
saveRDS(train_data, paste0(train_dir, "mimic_train.rds"))
write.csv(train_data, paste0(train_dir, "mimic_train.csv"), row.names = FALSE)
cat("TrainingsetalreadySave:\n")
cat("  - ", train_dir, "mimic_train.rds\n")
cat("  - ", train_dir, "mimic_train.csv\n\n")

saveRDS(test_data, paste0(test_dir, "mimic_test.rds"))
write.csv(test_data, paste0(test_dir, "mimic_test.csv"), row.names = FALSE)
cat("TestsetalreadySave:\n")
cat("  - ", test_dir, "mimic_test.rds\n")
cat("  - ", test_dir, "mimic_test.csv\n\n")

saveRDS(val_data, paste0(val_dir, "mimic_val.rds"))
write.csv(val_data, paste0(val_dir, "mimic_val.csv"), row.names = FALSE)
cat("ValidationsetalreadySave:\n")
cat("  - ", val_dir, "mimic_val.rds\n")
cat("  - ", val_dir, "mimic_val.csv\n\n")

# SaveSplitIndex
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
saveRDS(split_info, "../data/mimic_split_info.rds")
cat("SplitInfoalreadySave: ../data/mimic_split_info.rds\n\n")

cat("========================================\n")
cat("DataSplitComplete \n")
cat("========================================\n")
cat("\nUsage:\n")
cat("1. Trainingsetuse ModelTraining dWOLS/IQL \n")
cat("2. Testsetuse ModelEvaluationand Parameters \n")
cat("3. Validationsetuse Final ModelValidation\n")
cat("\nNext: perform multiple imputation on training set\n")

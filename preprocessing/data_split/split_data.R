# ============================================================================
# DataSplit (7:2:1)
# ============================================================================

library(dplyr)

cat("LoadingProcessing processed Data(Package Urine output)...\n")
dat <- readRDS("c:/Dynamic-RRT/1_mimicPreprocessingdata/mimic_processed_with_uo.rds")

cat("Data :", nrow(dat), "x", ncol(dat), "\n")

# settingsRandom types can 
set.seed(123)

n_total <- nrow(dat)
train_size <- floor(0.7 * n_total)
test_size <- floor(0.2 * n_total)
val_size <- n_total - train_size - test_size

cat("\nSplitRatio:\n")
cat("- Trainingset:", train_size, "(70%)\n")
cat("- Testset:", test_size, "(20%)\n")
cat("- Validationset:", val_size, "(10%)\n")

# Random Index
shuffled_indices <- sample(1:n_total, n_total, replace = FALSE)

# Split
train_indices <- shuffled_indices[1:train_size]
test_indices <- shuffled_indices[(train_size + 1):(train_size + test_size)]
val_indices <- shuffled_indices[(train_size + test_size + 1):n_total]

train_data <- dat[train_indices, ]
test_data <- dat[test_indices, ]
val_data <- dat[val_indices, ]

# CreateOutputDirectory
output_dir <- "c:/Dynamic-RRT/3_DataSplit/data"
dir.create(file.path(output_dir, "train"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "test"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "val"), recursive = TRUE, showWarnings = FALSE)

# SaveSplit processed Data
saveRDS(train_data, file.path(output_dir, "train/mimic_train.rds"))
write.csv(train_data, file.path(output_dir, "train/mimic_train.csv"), row.names = FALSE)

saveRDS(test_data, file.path(output_dir, "test/mimic_test.rds"))
write.csv(test_data, file.path(output_dir, "test/mimic_test.csv"), row.names = FALSE)

saveRDS(val_data, file.path(output_dir, "val/mimic_val.rds"))
write.csv(val_data, file.path(output_dir, "val/mimic_val.csv"), row.names = FALSE)

# SaveSplitInfo
split_info <- list(
  total = n_total,
  train_size = train_size,
  test_size = test_size,
  val_size = val_size,
  train_indices = train_indices,
  test_indices = test_indices,
  val_indices = val_indices,
  seed = 123
)

saveRDS(split_info, file.path(output_dir, "mimic_split_info.rds"))

cat("\nDataSplitComplete \n")
cat("OutputDirectory:", output_dir, "\n")
cat("\nNext: perform multiple imputation on each subset\n")

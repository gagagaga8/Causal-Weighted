library(dplyr)

cat(" use Dataset SplitData\n")
cat("==============================\n\n")

# Data
enhanced <- readRDS("../../1_mimicPreprocessingdata/mimic_processed_enhanced.rds")
cat(" Data:", nrow(enhanced), "Row,", ncol(enhanced), "Column\n")

# Keep only complete cases (all key variables non-missing)
complete_data <- enhanced %>%
  filter(!is.na(bun_k1) & !is.na(pot_k1) & !is.na(ph_k1) & !is.na(uo_k1) &
         !is.na(bun_k2) & !is.na(pot_k2) & !is.na(ph_k2) & !is.na(uo_k2) &
         !is.na(bun_k3) & !is.na(pot_k3) & !is.na(ph_k3) & !is.na(uo_k3) &
         !is.na(sofa_24hours))

cat("Full Case:", nrow(complete_data), "Row\n\n")

# settingsRandom types 
set.seed(721)

# 70% Training 15% Validation 15% Test
n_total <- nrow(complete_data)
indices <- sample(1:n_total)

n_train <- floor(0.7 * n_total)
n_val <- floor(0.15 * n_total)

train_data <- complete_data[indices[1:n_train], ]
val_data <- complete_data[indices[(n_train+1):(n_train+n_val)], ]
test_data <- complete_data[indices[(n_train+n_val+1):n_total], ]

cat("DataSplitResults:\n")
cat("  Trainingset:", nrow(train_data), "Row\n")
cat("  Validationset:", nrow(val_data), "Row\n")
cat("  Testset:", nrow(test_data), "Row\n\n")

# CreateOutputDirectory
dir.create("../data/train", recursive=TRUE, showWarnings=FALSE)
dir.create("../data/val", recursive=TRUE, showWarnings=FALSE)
dir.create("../data/test", recursive=TRUE, showWarnings=FALSE)

# SaveSplitData
write.csv(train_data, "../data/train/mimic_train.csv", row.names=FALSE)
write.csv(val_data, "../data/val/mimic_val.csv", row.names=FALSE)
write.csv(test_data, "../data/test/mimic_test.csv", row.names=FALSE)

cat("DataalreadySaveto 3_DataSplit/data/\n")
cat("Complete \n")

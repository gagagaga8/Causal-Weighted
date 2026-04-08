# ============================================================================
# dWOLSTrainingData(No Urine outputVersion)
# willPreprocessing processed DataConvertasdWOLSTraining 
# ============================================================================

library(dplyr)

cat(rep("=", 80), "\n")
cat(" dWOLSTrainingData(No Urine outputVersion)\n")
cat(rep("=", 80), "\n\n")

# PreprocessingData
cat("step1: PreprocessingData...\n")
input_file <- "../../../1_mimicPreprocessingdata/mimic_dwols_no_uo.csv"
dat <- read.csv(input_file)
cat("  - Sample Size:", nrow(dat), "\n")
cat("  - Features:", ncol(dat), "\n\n")

# Data 
cat("step2: Data ...\n")
cat(" Distribution:\n")
cat("    a1 (0-24h):", sum(dat$a1), "\n")
cat("    a2 (24-48h):", sum(dat$a2), "\n")
cat("    a3 (48-72h):", sum(dat$a3), "\n")
cat("    No RRT:", sum(dat$a1 == 0 & dat$a2 == 0 & dat$a3 == 0), "\n\n")

cat(" FeatureMissing :\n")
for(col in c("weight", "sofa_24hours", "bun_k1", "pot_k1", "ph_k1", "creat_k1")) {
  missing_rate <- sum(is.na(dat[[col]])) / nrow(dat) * 100
  cat(sprintf("    %-15s: %5.1f%%\n", col, missing_rate))
}
cat("\n")

# Createdecision pointDataset
cat("Step 3: Createdecision pointDataset...\n")

# k=3 (48-72h)
dat_k3 <- dat %>%
  filter(!is.na(bun_k3) & !is.na(pot_k3) & !is.na(ph_k3)) %>%
  mutate(a3 = as.integer(a3))
cat("  - k=3Sample Size:", nrow(dat_k3), "\n")

# k=2 (24-48h) - alreadyink=3startPatient
dat_k2 <- dat %>%
  filter(a3 == 0) %>%
  filter(!is.na(bun_k2) & !is.na(pot_k2) & !is.na(ph_k2) & !is.na(creat_k2)) %>%
  mutate(a2 = as.integer(a2))
cat("  - k=2Sample Size:", nrow(dat_k2), "\n")

# k=1 (0-24h) - alreadystartPatient
dat_k1 <- dat %>%
  filter(a3 == 0 & a2 == 0) %>%
  filter(!is.na(bun_k1) & !is.na(pot_k1) & !is.na(ph_k1) & !is.na(creat_k1)) %>%
  mutate(a1 = as.integer(a1))
cat("  - k=1Sample Size:", nrow(dat_k1), "\n\n")

# Save
cat("Step 4: SaveData...\n")
output_dir <- "../data"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write.csv(dat_k3, file.path(output_dir, "dwols_k3_no_uo.csv"), row.names = FALSE)
write.csv(dat_k2, file.path(output_dir, "dwols_k2_no_uo.csv"), row.names = FALSE)
write.csv(dat_k1, file.path(output_dir, "dwols_k1_no_uo.csv"), row.names = FALSE)
write.csv(dat, file.path(output_dir, "dwols_full_no_uo.csv"), row.names = FALSE)

cat("  - dwols_k3_no_uo.csv:", nrow(dat_k3), "Row\n")
cat("  - dwols_k2_no_uo.csv:", nrow(dat_k2), "Row\n")
cat("  - dwols_k1_no_uo.csv:", nrow(dat_k1), "Row\n")
cat("  - dwols_full_no_uo.csv:", nrow(dat), "Row\n\n")

cat(rep("=", 80), "\n")
cat("Data Complete!\n")
cat(rep("=", 80), "\n")

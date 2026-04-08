# ============================================================================
# IQLTrainingData - ConvertasRL 
# ============================================================================

library(dplyr)
library(arrow)

cat("LoadingTraining/Test/ValidationDataset...\n")

# Split processed Data
train_data <- read.csv("../data/train/mimic_train.csv")
test_data <- read.csv("../data/test/mimic_test.csv")
val_data <- read.csv("../data/val/mimic_val.csv")

cat("DatasetSize:\n")
cat("  Trainingset:", nrow(train_data), "\n")
cat("  Testset:", nrow(test_data), "\n")
cat("  Validationset:", nrow(val_data), "\n\n")

# RLDatasetFunction
# for Treatment each time point k1/k2/k3 Create Convert 
build_rl_transitions <- function(data, split_name) {
  cat(" ", split_name, "RLConvert...\n")
  
  transitions <- list()
  
  # statusFeature extend BUN pH + trendFeature 
  state_features <- c("admission_age", "gender", "weight", "sofa", 
                      "creat_k1", "creat_k2", "creat_k3",
                      "uo_k1", "uo_k2", "uo_k3",
                      "bun_k1", "bun_k2", "bun_k3",
                      "pot_k1", "pot_k2", "pot_k3",
                      "ph_k1", "ph_k2", "ph_k3")
  
  # time point1: k1 -> k2
  trans_k1 <- data %>%
    mutate(
      patient_id = stay_id,
      imputation_id = 1, # ImputationVersion
      timestep = 1,
      
      # status k1 - extend + trend
      state_age = admission_age,
      state_gender = ifelse(gender == "M", 1, 0),
      state_weight = weight,
      state_sofa = sofa,
      state_creat = creat_k1,
      state_uo = ifelse(is.na(uo_k1), 0, uo_k1),
      state_bun = ifelse(is.na(bun_k1), 0, bun_k1),
      state_pot = ifelse(is.na(pot_k1), 0, pot_k1),
      state_ph = ifelse(is.na(ph_k1), 7.4, ph_k1),
      
      # trendFeature k1 still has as0 
      state_delta_creat = 0,
      state_delta_uo = 0,
      state_delta_bun = 0,
      
      # 0-24his startRRT 
      action = a1,
      
      # useFull HFD notagain 3 
      reward = hfd,
      
      # Next state (k2 time point)
      next_state_age = admission_age,
      next_state_gender = ifelse(gender == "M", 1, 0),
      next_state_weight = weight,
      next_state_sofa = sofa,
      next_state_creat = creat_k2,
      next_state_uo = ifelse(is.na(uo_k2), 0, uo_k2),
      next_state_bun = ifelse(is.na(bun_k2), 0, bun_k2),
      next_state_pot = ifelse(is.na(pot_k2), 0, pot_k2),
      next_state_ph = ifelse(is.na(ph_k2), 7.4, ph_k2),
      
      # k2 trend k2 - k1 
      next_state_delta_creat = creat_k2 - creat_k1,
      next_state_delta_uo = ifelse(is.na(uo_k2) | is.na(uo_k1), 0, uo_k2 - uo_k1),
      next_state_delta_bun = ifelse(is.na(bun_k2) | is.na(bun_k1), 0, bun_k2 - bun_k1),
      
      # is 
      done = FALSE,
      
      # Split 
      split_721 = split_name
    ) %>%
    select(patient_id, imputation_id, timestep, 
           state_age, state_gender, state_weight, state_sofa, state_creat, state_uo, 
           state_bun, state_pot, state_ph, state_delta_creat, state_delta_uo, state_delta_bun,
           action, reward,
           next_state_age, next_state_gender, next_state_weight, next_state_sofa, 
           next_state_creat, next_state_uo, next_state_bun, next_state_pot, next_state_ph,
           next_state_delta_creat, next_state_delta_uo, next_state_delta_bun,
           done, split_721)
  
  transitions[[1]] <- trans_k1
  
  # time point2: k2 -> k3
  trans_k2 <- data %>%
    filter(a1 == 0) %>%
    mutate(
      patient_id = stay_id,
      imputation_id = 1,
      timestep = 2,
      
      state_age = admission_age,
      state_gender = ifelse(gender == "M", 1, 0),
      state_weight = weight,
      state_sofa = sofa,
      state_creat = creat_k2,
      state_uo = ifelse(is.na(uo_k2), 0, uo_k2),
      state_bun = ifelse(is.na(bun_k2), 0, bun_k2),
      state_pot = ifelse(is.na(pot_k2), 0, pot_k2),
      state_ph = ifelse(is.na(ph_k2), 7.4, ph_k2),
      
      # k2trend k2 - k1 
      state_delta_creat = creat_k2 - creat_k1,
      state_delta_uo = ifelse(is.na(uo_k2) | is.na(uo_k1), 0, uo_k2 - uo_k1),
      state_delta_bun = ifelse(is.na(bun_k2) | is.na(bun_k1), 0, bun_k2 - bun_k1),
      
      # 24-48his startRRT 
      action = a2,
      
      # useFull HFD notagain 3 
      reward = hfd,
      
      next_state_age = admission_age,
      next_state_gender = ifelse(gender == "M", 1, 0),
      next_state_weight = weight,
      next_state_sofa = sofa,
      next_state_creat = creat_k3,
      next_state_uo = ifelse(is.na(uo_k3), 0, uo_k3),
      next_state_bun = ifelse(is.na(bun_k3), 0, bun_k3),
      next_state_pot = ifelse(is.na(pot_k3), 0, pot_k3),
      next_state_ph = ifelse(is.na(ph_k3), 7.4, ph_k3),
      
      # k3trend k3 - k2 
      next_state_delta_creat = creat_k3 - creat_k2,
      next_state_delta_uo = ifelse(is.na(uo_k3) | is.na(uo_k2), 0, uo_k3 - uo_k2),
      next_state_delta_bun = ifelse(is.na(bun_k3) | is.na(bun_k2), 0, bun_k3 - bun_k2),
      
      # is 
      done = FALSE,
      
      # Split 
      split_721 = split_name
    ) %>%
    select(patient_id, imputation_id, timestep, 
           state_age, state_gender, state_weight, state_sofa, state_creat, state_uo, 
           state_bun, state_pot, state_ph, state_delta_creat, state_delta_uo, state_delta_bun,
           action, reward,
           next_state_age, next_state_gender, next_state_weight, next_state_sofa, 
           next_state_creat, next_state_uo, next_state_bun, next_state_pot, next_state_ph,
           next_state_delta_creat, next_state_delta_uo, next_state_delta_bun,
           done, split_721)
  
  transitions[[2]] <- trans_k2
  
  # time point3: k3 -> 
  trans_k3 <- data %>%
    filter(a1 == 0 & a2 == 0) %>%
    mutate(
      patient_id = stay_id,
      imputation_id = 1,
      timestep = 3,
      
      state_age = admission_age,
      state_gender = ifelse(gender == "M", 1, 0),
      state_weight = weight,
      state_sofa = sofa,
      state_creat = creat_k3,
      state_uo = ifelse(is.na(uo_k3), 0, uo_k3),
      state_bun = ifelse(is.na(bun_k3), 0, bun_k3),
      state_pot = ifelse(is.na(pot_k3), 0, pot_k3),
      state_ph = ifelse(is.na(ph_k3), 7.4, ph_k3),
      
      # k3trend k3 - k2 
      state_delta_creat = creat_k3 - creat_k2,
      state_delta_uo = ifelse(is.na(uo_k3) | is.na(uo_k2), 0, uo_k3 - uo_k2),
      state_delta_bun = ifelse(is.na(bun_k3) | is.na(bun_k2), 0, bun_k3 - bun_k2),
      
      # 48-72his startRRT 
      action = a3,
      
      # useFull HFD notagain 3 
      reward = hfd,
      
      # status all0 Package trend 
      next_state_age = 0,
      next_state_gender = 0,
      next_state_weight = 0,
      next_state_sofa = 0,
      next_state_creat = 0,
      next_state_uo = 0,
      next_state_bun = 0,
      next_state_pot = 0,
      next_state_ph = 0,
      next_state_delta_creat = 0,
      next_state_delta_uo = 0,
      next_state_delta_bun = 0,
      
      # is 
      done = TRUE,
      
      # Split 
      split_721 = split_name
    ) %>%
    select(patient_id, imputation_id, timestep, 
           state_age, state_gender, state_weight, state_sofa, state_creat, state_uo, 
           state_bun, state_pot, state_ph, state_delta_creat, state_delta_uo, state_delta_bun,
           action, reward,
           next_state_age, next_state_gender, next_state_weight, next_state_sofa, 
           next_state_creat, next_state_uo, next_state_bun, next_state_pot, next_state_ph,
           next_state_delta_creat, next_state_delta_uo, next_state_delta_bun,
           done, split_721)
  
  transitions[[3]] <- trans_k3
  
  # MergeAll Convert
  all_trans <- bind_rows(transitions)
  
  cat("  K1Convert:", nrow(trans_k1), "\n")
  cat("  K2Convert:", nrow(trans_k2), "\n")
  cat("  K3Convert:", nrow(trans_k3), "\n")
  cat(" Convert:", nrow(all_trans), "\n")
  
  return(all_trans)
}

# each DatasetRLConvert
train_rl <- build_rl_transitions(train_data, "train")
test_rl <- build_rl_transitions(test_data, "test")
val_rl <- build_rl_transitions(val_data, "val")

# MergeAll Data
cat("\nMergeAll Dataset...\n")
all_rl <- bind_rows(train_rl, test_rl, val_rl)

cat("Final RLDatasetSize:", nrow(all_rl), "\n")
cat(" Distribution:\n")
print(table(all_rl$action, all_rl$split_721))

# Saveasfeather 
cat("\nSaveasfeather ...\n")
output_dir <- "../data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(output_dir, "mimic_rl_dataset.feather")
write_feather(all_rl, output_path)

cat("\nComplete \n")
cat("OutputFile:", output_path, "\n")
cat(" Sample Size:", nrow(all_rl), "\n")
cat("  Trainingset:", sum(all_rl$split_721 == "train"), "\n")
cat("  Testset:", sum(all_rl$split_721 == "test"), "\n")
cat("  Validationset:", sum(all_rl$split_721 == "val"), "\n")

# ============================================================================
# Data will DataConvertasdWOLS 
# ============================================================================

library(mice)

cat("LoadingTrainingset imputationData...\n")
load("../../../3_DataSplit/data/train/imputed_mimic_train.RData")

cat(" : m=", exp_dat_train$m, ", n=", nrow(exp_dat_train$data), "\n")

# paireach ImputationDataset Row 
cat("\n Data ...\n")

# Create midsobjectstore processed Data
adapted_imp_list <- list()

for(i in 1:exp_dat_train$m) {
  dat <- complete(exp_dat_train, i)
  
  # MissingVariable use hasVariable as 
  # BUN: usecreat as and non 
  dat$bun_k1 <- pmax(dat$creat_k1 * 2.5, 0.1)  # Minimumvalue0.1
  dat$bun_k2 <- pmax(dat$creat_k2 * 2.5, 0.1)
  dat$bun_k3 <- pmax(dat$creat_k3 * 2.5, 0.1)
  
  # pH: settingsdefaultvalue down fromData Extract 
  dat$ph_k1 <- 7.35  # Normal range
  dat$ph_k2 <- 7.35
  dat$ph_k3 <- 7.35
  
  # : settingsdefaultvalue
  dat$pot_k1 <- 4.0  # Normal range
  dat$pot_k2 <- 4.0
  dat$pot_k3 <- 4.0
  
  # Urine output: is in not inthensettingsas0
  if(!"uo_k1" %in% colnames(dat) || all(is.na(dat$uo_k1))) {
    dat$uo_k1 <- 0
  }
  if(!"uo_k2" %in% colnames(dat) || all(is.na(dat$uo_k2))) {
    dat$uo_k2 <- 0
  }
  if(!"uo_k3" %in% colnames(dat) || all(is.na(dat$uo_k3))) {
    dat$uo_k3 <- 0
  }
  
  # SOFA 
  dat$SOFA_24hours <- dat$sofa
  
  adapted_imp_list[[i]] <- dat
}

# midsobject
cat("\n midsobject...\n")
adapted_data <- adapted_imp_list[[1]]

# Use first imputed dataset as base
imp_temp <- mice(adapted_data, maxit=0)
exp_dat_train_adapted <- exp_dat_train

# UpdateData
for(i in 1:exp_dat_train$m) {
  exp_dat_train_adapted$imp <- list() # 
}

# Save processed Data
cat("\nSave processed Data...\n")

# Method1 Saveaslist dWOLS 
train_imp_list <- adapted_imp_list
save(train_imp_list, file="../../../3_DataSplit/data/train/imputed_mimic_dtr_train.RData")

cat(" Complete \n")
cat("- Output: ../../../3_DataSplit/data/train/imputed_mimic_dtr_train.RData\n")
cat("- : train_imp_list ( dWOLS )\n")
cat("- Variable: bun_k1/k2/k3, ph_k1/k2/k3, pot_k1/k2/k3, uo_k1/k2/k3, SOFA_24hours\n")

library(mice)
library(dplyr)

# LoadingPathConfiguration
if(file.exists("../config_paths.R")) {
  source("../config_paths.R")
} else {
  # if config_paths.Rnot in usewhen Directory
  LEARNING_DIR <- getwd()
  # Directory featherFileSaveto04_PythonEvaluation/data/
  FEATHER_DEV_DIR <- file.path(getwd(), "..", "04_PythonEvaluation", "data", "feather_files_dev")
}

# 7:2:1SplitMethod useTrainingset dataandTrainingsetcoefficient
cat("========================================\n")
cat(" Trainingset dataasfeather 7:2:1SplitMethod \n")
cat("========================================\n\n")

# LoadingTrainingset data
# Directory from01_DataSplit/data/train/Loading
if(file.exists("../../01_DataSplit/data/train/imputed_mimic_dtr_train.RData")) {
  load("../../01_DataSplit/data/train/imputed_mimic_dtr_train.RData")
} else if(file.exists("../01_DataSplit/data/train/imputed_mimic_dtr_train.RData")) {
  load("../01_DataSplit/data/train/imputed_mimic_dtr_train.RData")
} else if(file.exists(file.path(LEARNING_DIR, "imputed_mimic_dtr_train.RData"))) {
  load(file.path(LEARNING_DIR, "imputed_mimic_dtr_train.RData"))
} else if(file.exists("imputed_mimic_dtr_train.RData")) {
  load("imputed_mimic_dtr_train.RData")
} else {
  stop("Cannot find imputed_mimic_dtr_train.RData")
}

# LoadingTrainingsetcoefficient
# Directory from02_ModelTraining/results/Loading
if(file.exists("../results/ite_preds_coef_train.RData")) {
  load("../results/ite_preds_coef_train.RData")
} else if(file.exists("../../02_ModelTraining/results/ite_preds_coef_train.RData")) {
  load("../../02_ModelTraining/results/ite_preds_coef_train.RData")
} else if(file.exists("../02_ModelTraining/results/ite_preds_coef_train.RData")) {
  load("../02_ModelTraining/results/ite_preds_coef_train.RData")
} else if(file.exists(file.path(LEARNING_DIR, "ite_preds_coef_train.RData"))) {
  load(file.path(LEARNING_DIR, "ite_preds_coef_train.RData"))
} else if(file.exists("ite_preds_coef_train.RData")) {
  load("ite_preds_coef_train.RData")
} else {
  stop("Cannot find ite_preds_coef_train.RData")
}

# Use training set version coefficients
est_k1 <- est_k1_train
est_k2 <- est_k2_train
est_k3 <- est_k3_train
vcv_k1 <- vcv_k1_train
vcv_k2 <- vcv_k2_train
vcv_k3 <- vcv_k3_train

# ProcessingTrainingset data 
if(exists("train_imp_list")) {
  train_data_base <- train_imp_list[[1]]
  if(!exists(".Random.seed", envir = .GlobalEnv)) {
    set.seed(12345)
  }
  exp_dat <- mice(train_data_base, maxit = 0, m = length(train_imp_list), seed = 12345)
  exp_dat$data <- train_data
  cat(" useTrainingset data ColumnTable ", length(train_imp_list), " imputed datasets)\n", sep="")
  cat("TrainingsetSize:", nrow(train_data_base), " Patient\n\n")
} else if(exists("exp_dat_train")) {
  exp_dat <- exp_dat_train
  cat(" useTrainingsetmidsobject\n")
  cat("TrainingsetSize:", nrow(exp_dat_train$data), " Patient\n\n")
} else {
  stop("Trainingset data Incorrect")
}

imp_1 <- complete(exp_dat, 1)

#### Imputation processed Data 

# ITEstandard errorComputationFunction Version 
calc_ite_se <- function(mat, vcv) {
  if(ncol(mat) != nrow(vcv) || ncol(mat) != ncol(vcv)) {
    min_dim <- min(ncol(mat), nrow(vcv), ncol(vcv))
    if(ncol(mat) > min_dim) {
      mat <- mat[, 1:min_dim, drop=FALSE]
    }
    if(nrow(vcv) > min_dim) {
      vcv <- vcv[1:min_dim, 1:min_dim, drop=FALSE]
    }
  }
  
  # is value 
  mat <- as.matrix(mat)
  vcv <- as.matrix(vcv)
  
  # Computationstandard error
  result <- sapply(as.list(as.data.frame(t(mat))), function(x) {
    x_vec <- as.numeric(x)
    if(length(x_vec) == nrow(vcv) && all(!is.na(x_vec)) && all(!is.na(vcv))) {
      tryCatch({
        val <- as.numeric(t(x_vec) %*% vcv %*% x_vec)
        if(!is.na(val) && val >= 0) {
          sqrt(val)
        } else {
          0
        }
      }, error = function(e) 0)
    } else {
      0
    }
  })
  
  # Resultsis value to 
  result <- as.numeric(result)
  result[is.na(result)] <- 0
  return(result)
}

dtr_recos <- function(imp_dat){
  n_rows <- nrow(imp_dat)
  
  # phasek=1 All Patient
  mat_k1 <- model.matrix(~1 + admission_age + creat_k1 + pot_k1, data=imp_dat)
  
  # phasek=2 onlyPackage a1=0Patient
  excluded_k2 <- imp_dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- imp_dat[included_k2,]
  if(nrow(dat_k2) > 0) {
    mat_k2 <- model.matrix(~1 + SOFA_24hours + bun_k2 + I(abs(ph_k2-ph_k1)) + I(uo_k2+uo_k1), data=dat_k2)
  } else {
    mat_k2 <- matrix(0, nrow=0, ncol=5)
  }
  
  # phasek=3 onlyPackage a2=0Patient
  excluded_k3 <- imp_dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- imp_dat[included_k3,]
  if(nrow(dat_k3) > 0) {
    mat_k3 <- model.matrix(~1 + uo_k3 + I(bun_k3/bun_k1), data=dat_k3)
  } else {
    mat_k3 <- matrix(0, nrow=0, ncol=3)
  }
  
  # ComputationITEvalue - match
  if(ncol(mat_k1) == length(est_k1) && length(est_k1) > 0) {
    ite_k1_val <- as.vector(mat_k1 %*% est_k1)
  } else {
    if(ncol(mat_k1) > length(est_k1) && length(est_k1) > 0) {
      mat_k1 <- mat_k1[, 1:length(est_k1), drop=FALSE]
      ite_k1_val <- as.vector(mat_k1 %*% est_k1)
    } else {
      ite_k1_val <- rep(0, nrow(mat_k1))
    }
  }
  
  # ite_k1_vallengthequals Patient 
  if(length(ite_k1_val) != n_rows) {
    if(length(ite_k1_val) > n_rows) {
      ite_k1_val <- ite_k1_val[1:n_rows]
    } else {
      ite_k1_val <- c(ite_k1_val, rep(0, n_rows - length(ite_k1_val)))
    }
  }
  
  if(nrow(mat_k2) > 0 && ncol(mat_k2) == length(est_k2) && length(est_k2) > 0) {
    ite_k2_val_full <- as.vector(mat_k2 %*% est_k2)
  } else {
    ite_k2_val_full <- numeric(0)
  }
  
  if(nrow(mat_k3) > 0 && ncol(mat_k3) == length(est_k3) && length(est_k3) > 0) {
    ite_k3_val_full <- as.vector(mat_k3 %*% est_k3)
  } else {
    ite_k3_val_full <- numeric(0)
  }
  
  # Create complete ITE vectors for all patients
  ite_k2_val <- rep(0, n_rows)
  if(length(ite_k2_val_full) > 0 && sum(included_k2) == length(ite_k2_val_full)) {
    ite_k2_val[included_k2] <- ite_k2_val_full
  }
  
  ite_k3_val <- rep(0, n_rows)
  if(length(ite_k3_val_full) > 0 && sum(included_k3) == length(ite_k3_val_full)) {
    ite_k3_val[included_k3] <- ite_k3_val_full
  }
  
  # use processed FunctionComputationITEstandard error
  ite_se_k1_val <- calc_ite_se(mat_k1, vcv_k1)
  if(length(ite_se_k1_val) != n_rows) {
    if(length(ite_se_k1_val) > n_rows) {
      ite_se_k1_val <- ite_se_k1_val[1:n_rows]
    } else {
      ite_se_k1_val <- c(ite_se_k1_val, rep(0, n_rows - length(ite_se_k1_val)))
    }
  }
  
  ite_se_k2_val <- rep(0, n_rows)
  if(nrow(mat_k2) > 0) {
    ite_se_k2_val_full <- calc_ite_se(mat_k2, vcv_k2)
    if(length(ite_se_k2_val_full) > 0 && sum(included_k2) == length(ite_se_k2_val_full)) {
      ite_se_k2_val[included_k2] <- ite_se_k2_val_full
    }
  }
  
  ite_se_k3_val <- rep(0, n_rows)
  if(nrow(mat_k3) > 0) {
    ite_se_k3_val_full <- calc_ite_se(mat_k3, vcv_k3)
    if(length(ite_se_k3_val_full) > 0 && sum(included_k3) == length(ite_se_k3_val_full)) {
      ite_se_k3_val[included_k3] <- ite_se_k3_val_full
    }
  }
  
  # policy 
  r_1 <- as.numeric(ite_k1_val > 0 ) # 1 days 
  r_2 <- as.numeric(ite_k2_val > 0 ) # 2 days 
  r_3 <- as.numeric(ite_k3_val > 0 ) # 3 days 
  
  res_df <- cbind(imp_dat, data.frame(r_1=r_1, r_2=r_2, r_3=r_3))
  
  # foralreadystartTreatmentPatient Treatment if before already start then 
  res_df$r_2[res_df$a1==1 | res_df$r_1==1] <- 1 
  res_df$r_3[res_df$a2==1 | res_df$r_1==1 | res_df$r_2==1] <- 1
  
  res_df$r_2[res_df$phi_2==1] <- NA # pairDeathPatientnot 
  res_df$r_3[res_df$phi_3==1] <- NA # pairDeathPatientnot 
  
  # policy i.e. policy
  # Only recommend RRT if evidence at 5% significance level benefits the individual patient
  r_s_1 <- as.numeric(ite_k1_val - qnorm(.975) * ite_se_k1_val > 0)
  r_s_2 <- as.numeric(ite_k2_val - qnorm(.975) * ite_se_k2_val > 0)
  r_s_3 <- as.numeric(ite_k3_val - qnorm(.975) * ite_se_k3_val > 0)
  
  res_df <- cbind(res_df, data.frame(r_s_1=r_s_1, r_s_2=r_s_2, r_s_3=r_s_3))
  
  res_df$r_s_2[res_df$a1==1 | res_df$r_s_1==1] <- 1 
  res_df$r_s_3[res_df$a2==1 | res_df$r_s_1==1 | res_df$r_s_2==1] <- 1
  
  res_df$r_s_2[res_df$phi_2==1] <- NA # pairDeathPatientnot 
  res_df$r_s_3[res_df$phi_3==1] <- NA # pairDeathPatientnot 
  
  return(res_df)
}

fun_reco_in_imp <- function(mice_obj){
  list_imp_df <- lapply(1:mice_obj$m, function(x) complete(mice_obj, x) %>%
                          # will Variable encodeto numeric inPythonin 
                          mutate(immunosuppressant=as.numeric(as.character(immunosuppressant) %in% c("OUI", "1")),
                                 a1=as.numeric(as.character(a1)), a2=as.numeric(as.character(a2)), a3=as.numeric(as.character(a3)) ) ) 
  return(lapply(list_imp_df, dtr_recos))
}

imp_dfs_rec_ls <- fun_reco_in_imp(exp_dat)

# willData as.feather Python use
library(arrow)

# Use configured paths, save current working directory
original_wd <- getwd()

# OutputDirectory in
if(!exists("FEATHER_DEV_DIR") || is.null(FEATHER_DEV_DIR)) {
  FEATHER_DEV_DIR <- file.path(getwd(), "Learning", "feather_files_dev")
}
if(!dir.exists(FEATHER_DEV_DIR)) {
  FEATHER_DEV_DIR <- file.path(getwd(), "feather_files_dev")
}
if(!dir.exists(FEATHER_DEV_DIR)) {
  dir.create(FEATHER_DEV_DIR, recursive = TRUE)
}

setwd(FEATHER_DEV_DIR)
sapply(1:length(imp_dfs_rec_ls), 
       function(i) write_feather(imp_dfs_rec_ls[[i]], paste0("DF",i,".feather")))
# Restore Directory
setwd(original_wd)
cat("\nFeather files exported to:", FEATHER_DEV_DIR, "\n")
cat("Number of files:", length(imp_dfs_rec_ls), "\n")
     
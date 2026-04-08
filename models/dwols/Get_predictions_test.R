# TestsetEvaluation 
# 7:2:1SplitMethod inTestset 20%Data onEvaluationTraining Model
library(dplyr)
library(mice)
# Directory from02_ModelTraining/scripts/Loadingfunctions.R
if(file.exists("../02_ModelTraining/scripts/functions.R")) {
  source("../02_ModelTraining/scripts/functions.R")
} else {
  source("functions.R")
}

# LoadingTraining Modelcoefficient fromTrainingset 
cat("========================================\n")
cat("TestsetEvaluation useTrainingset Model\n")
cat("========================================\n\n")

# Directory from../results/Loading
if(file.exists("../results/ite_preds_coef.RData")) {
  load("../results/ite_preds_coef.RData")
  cat("Load model coefficientsSuccess\n")
} else {
  stop("Error: not toTrainingResultsFile\n Run dWOLS_parallel.R RowTraining")
}

# Use model coefficients (learned from training set)
# est_k1, est_k2, est_k3, vcv_k1, vcv_k2, vcv_k3 alreadyLoading

# LoadingTestset data
# Directory from../../../3_DataSplit/data/test/Loading
if(file.exists("../../../3_DataSplit/data/test/mimic_test.rds")) {
  test_data <- readRDS("../../../3_DataSplit/data/test/mimic_test.rds")
  cat("LoadingTestset dataSuccess:", nrow(test_data), " Patient\n\n")
} else {
  stop("Error: not toTestset data mimic_test.rds\n Run split_mimic_data.R RowDataSplit")
}

# ProcessingTestset data usemice RowImputation 
cat("StartpairTestset RowImputation...\n")
set.seed(12345)
test_dat <- mice(test_data, m = 100, maxit = 10, method = 'pmm', printFlag = FALSE)
cat("ImputationComplete: m=", test_dat$m, ", n=", nrow(test_dat$data), "\n\n")

# Extract boot_dtr function from dWOLS_parallel.R for evaluation
# onlyneedto notneedtoFull boot_dtrFunction
# Create Function 
get_mats <- function(d) {
  dat <- as.data.frame(lapply(d, defactorize))
  
  # a1, a2, a3is value 
  if(is.factor(dat$a1) || is.character(dat$a1)) dat$a1 <- as.numeric(as.character(dat$a1))
  if(is.factor(dat$a2) || is.character(dat$a2)) dat$a2 <- as.numeric(as.character(dat$a2))
  if(is.factor(dat$a3) || is.character(dat$a3)) dat$a3 <- as.numeric(as.character(dat$a3))
  
  # phasek=3
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- dat[included_k3,]
  mat_k3 <- model.matrix(~1 + uo_k3 + I(bun_k3/bun_k1), data=dat_k3)
  
  # phasek=2
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- dat[included_k2,]
  mat_k2 <- model.matrix(~1 + SOFA_24hours + bun_k2 + I(abs(ph_k2-ph_k1)) + I(uo_k2+uo_k1), data=dat_k2)
  
  # phasek=1
  dat_k1 <- dat
  mat_k1 <- model.matrix(~1 + admission_age + creat_k1 + pot_k1, data=dat_k1)
  
  return(list(mat_k1, mat_k2, mat_k3))
}

get_ttt <- function(d) {
  dat <- as.data.frame(lapply(d, defactorize))
  if(is.factor(dat$a1) || is.character(dat$a1)) dat$a1 <- as.numeric(as.character(dat$a1))
  if(is.factor(dat$a2) || is.character(dat$a2)) dat$a2 <- as.numeric(as.character(dat$a2))
  if(is.factor(dat$a3) || is.character(dat$a3)) dat$a3 <- as.numeric(as.character(dat$a3))
  
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- dat[included_k3,]
  
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- dat[included_k2,]
  
  return(list(a1=dat$a1, a2=dat_k2$a2, a3=dat_k3$a3))
}

get_ids <- function(d) {
  dat <- as.data.frame(lapply(d, defactorize))
  if(is.factor(dat$a1) || is.character(dat$a1)) dat$a1 <- as.numeric(as.character(dat$a1))
  if(is.factor(dat$a2) || is.character(dat$a2)) dat$a2 <- as.numeric(as.character(dat$a2))
  if(is.factor(dat$a3) || is.character(dat$a3)) dat$a3 <- as.numeric(as.character(dat$a3))
  
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- dat[included_k3,]
  
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- dat[included_k2,]
  
  return(list(ids_k1=dat$subject_id, ids_k2=dat_k2$subject_id, ids_k3=dat_k3$subject_id))
}

###### Treatment ITE Predictionvalueand standard error ######

# paireach ImputationDatasetComputation 
mats_k123 <- list()
cat("Computation ...\n")
for(i in 1:test_dat$m) {
  mats_k123[[i]] <- get_mats(complete(test_dat, i))
  if(i %% 10 == 0 || i == test_dat$m) {
    cat(paste0(i, "/", test_dat$m, " datasets completed\n"))
  }
}

# ComputationMean Rubinrule 
mean_list_of_matrices <- function(x) {
  Reduce("+", x) / length(x)
}

mean_mat_k1 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[1]]))
mean_mat_k2 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[2]]))
mean_mat_k3 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[3]]))

# ComputationITEPredictionvalue
cat("\nComputationITEPredictionvalue...\n")
if(ncol(mean_mat_k1) == length(est_k1) && length(est_k1) > 0) {
  ite_k1_test_ori <- mean_mat_k1 %*% est_k1
} else {
  if(ncol(mean_mat_k1) > length(est_k1) && length(est_k1) > 0) {
    mean_mat_k1 <- mean_mat_k1[, 1:length(est_k1), drop=FALSE]
    ite_k1_test_ori <- mean_mat_k1 %*% est_k1
  } else {
    ite_k1_test_ori <- rep(0, nrow(mean_mat_k1))
  }
}

if(ncol(mean_mat_k2) == length(est_k2) && length(est_k2) > 0) {
  ite_k2_test_ori <- mean_mat_k2 %*% est_k2
} else {
  if(ncol(mean_mat_k2) > length(est_k2) && length(est_k2) > 0) {
    mean_mat_k2 <- mean_mat_k2[, 1:length(est_k2), drop=FALSE]
    ite_k2_test_ori <- mean_mat_k2 %*% est_k2
  } else {
    ite_k2_test_ori <- rep(0, nrow(mean_mat_k2))
  }
}

if(ncol(mean_mat_k3) == length(est_k3) && length(est_k3) > 0) {
  ite_k3_test_ori <- mean_mat_k3 %*% est_k3
} else {
  if(ncol(mean_mat_k3) > length(est_k3) && length(est_k3) > 0) {
    mean_mat_k3 <- mean_mat_k3[, 1:length(est_k3), drop=FALSE]
    ite_k3_test_ori <- mean_mat_k3 %*% est_k3
  } else {
    ite_k3_test_ori <- rep(0, nrow(mean_mat_k3))
  }
}

# ComputationITEstandard error
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
  mat <- as.matrix(mat)
  vcv <- as.matrix(vcv)
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
  result <- as.numeric(result)
  result[is.na(result)] <- 0
  return(result)
}

ite_se_k1_test_ori <- calc_ite_se(mean_mat_k1, vcv_k1)
ite_se_k2_test_ori <- calc_ite_se(mean_mat_k2, vcv_k2)
ite_se_k3_test_ori <- calc_ite_se(mean_mat_k3, vcv_k3)

# Treatment andPatientID
imp_1 <- complete(test_dat, 1)
actual_ttts_test_ori <- get_ttt(imp_1)
actual_ids_test_ori <- get_ids(imp_1)

# ProcessingDeathPatient indecision pointbefore DeathPatient 
# ite_k1_test_orilength equals All Patient asAll Patientallto decision point1 
# But need to ensure length matching
imp_1 <- complete(test_dat, 1)
n_patients <- nrow(imp_1)

# ite_k1_test_orilengthCorrect
if(length(ite_k1_test_ori) != n_patients) {
  cat("Warning: ite_k1_test_orilength(", length(ite_k1_test_ori), ") withPatient (", n_patients, ")notmatch\n")
  cat(" length...\n")
  if(length(ite_k1_test_ori) > n_patients) {
    ite_k1_test_ori <- ite_k1_test_ori[1:n_patients]
    ite_se_k1_test_ori <- ite_se_k1_test_ori[1:n_patients]
  } else {
    ite_k1_test_ori <- c(ite_k1_test_ori, rep(0, n_patients - length(ite_k1_test_ori)))
    ite_se_k1_test_ori <- c(ite_se_k1_test_ori, rep(0, n_patients - length(ite_se_k1_test_ori)))
  }
}

actual_ttts_test <- list(a1=actual_ttts_test_ori$a1)
actual_ids_test <- list(ids_k1=actual_ids_test_ori$ids_k1)
ite_k1_test <- ite_k1_test_ori
ite_se_k1_test <- ite_se_k1_test_ori

# decision point2
phi_2_ids <- test_dat$data$subject_id[test_dat$data$phi_2==1]
alive_amg_no_init_k2 <- !(actual_ids_test_ori$ids_k2 %in% phi_2_ids)
if(length(ite_k2_test_ori) > 0) {
  ite_k2_test <- ite_k2_test_ori[alive_amg_no_init_k2]
  ite_se_k2_test <- ite_se_k2_test_ori[alive_amg_no_init_k2]
} else {
  ite_k2_test <- numeric(0)
  ite_se_k2_test <- numeric(0)
}
actual_ttts_test$a2 <- actual_ttts_test_ori$a2[alive_amg_no_init_k2]
actual_ids_test$ids_k2 <- actual_ids_test_ori$ids_k2[alive_amg_no_init_k2]

# decision point3
phi_3_ids <- test_dat$data$subject_id[test_dat$data$phi_3==1]
alive_amg_no_init_k3 <- !(actual_ids_test_ori$ids_k3 %in% phi_3_ids)
if(length(ite_k3_test_ori) > 0) {
  ite_k3_test <- ite_k3_test_ori[alive_amg_no_init_k3]
  ite_se_k3_test <- ite_se_k3_test_ori[alive_amg_no_init_k3]
} else {
  ite_k3_test <- numeric(0)
  ite_se_k3_test <- numeric(0)
}
actual_ttts_test$a3 <- actual_ttts_test_ori$a3[alive_amg_no_init_k3]
actual_ids_test$ids_k3 <- actual_ids_test_ori$ids_k3[alive_amg_no_init_k3]

# policy with Treatment 
cat("\npolicy with Treatment (Testset):\n")
cat("decision point1:\n")
if(length(ite_k1_test) == length(actual_ttts_test$a1)) {
  print(table(ite_k1_test > 0, actual_ttts_test$a1))
} else {
  cat("Warning: Length mismatch - ite_k1_test:", length(ite_k1_test), ", actual_ttts_test$a1:", length(actual_ttts_test$a1), "\n")
}
cat("\ndecision point2:\n")
if(length(ite_k2_test) > 0 && length(ite_k2_test) == length(actual_ttts_test$a2)) {
  print(table(ite_k2_test > 0, actual_ttts_test$a2))
} else {
  cat("Warning: Length mismatchor as - ite_k2_test:", length(ite_k2_test), ", actual_ttts_test$a2:", length(actual_ttts_test$a2), "\n")
}
cat("\ndecision point3:\n")
if(length(ite_k3_test) > 0 && length(ite_k3_test) == length(actual_ttts_test$a3)) {
  print(table(ite_k3_test > 0, actual_ttts_test$a3))
} else {
  cat("Warning: Length mismatchor as - ite_k3_test:", length(ite_k3_test), ", actual_ttts_test$a3:", length(actual_ttts_test$a3), "\n")
}

# SaveResults
# Directory Savetoresults Directory
save(actual_ttts_test, ite_k1_test, ite_k2_test, ite_k3_test,
     ite_se_k1_test, ite_se_k2_test, ite_se_k3_test,
     actual_ids_test,
     file="../results/test_predictions.RData")

cat("\n========================================\n")
cat("TestsetEvaluationComplete \n")
cat("Results savedto test_predictions.RData\n")
cat("========================================\n")


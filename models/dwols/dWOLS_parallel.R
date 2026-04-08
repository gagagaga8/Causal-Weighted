# ============================================================================
# dWOLS (Dynamic Weighted Ordinary Least Squares) and RowTraining 
# Dynamic-RRT dWOLS.R
# ============================================================================

source("../../../ /Dynamic-RRT-main/Learning/functions.R")
library(mice)
library(boot)
library(parallel)
library(doParallel)
library(foreach)

# settingsand RowComputation
n_cores <- detectCores()
cl <- makeCluster(max(1, n_cores - 1))
registerDoParallel(cl)

# LoadingTrainingset imputationData
if(file.exists("../../../3_DataSplit/data/train/imputed_mimic_dtr_train.RData")) {
  load("../../../3_DataSplit/data/train/imputed_mimic_dtr_train.RData")
  if(exists("train_imp_list")) {
    train_data_base <- train_imp_list[[1]]
    if(!exists(".Random.seed", envir = .GlobalEnv)) {
      set.seed(12345)
    }
    exp_dat <- mice(train_data_base, maxit = 0, m = length(train_imp_list), seed = 12345)
    exp_dat$data <- train_data_base
    assign("train_imp_list_global", train_imp_list, envir = .GlobalEnv)
    complete.mids <- function(x, action = 1, include = FALSE, ...) {
      if(exists("train_imp_list_global", envir = .GlobalEnv)) {
        imp_list <- get("train_imp_list_global", envir = .GlobalEnv)
        if(action >= 1 && action <= length(imp_list)) {
          return(imp_list[[action]])
        }
      }
      return(mice::complete(x, action, include, ...))
    }
    cat("alreadyLoadingTrainingset data (list , m=", length(train_imp_list), ", n=", nrow(train_data_base), ")\n", sep="")
  } else {
    stop("Trainingset data Incorrect train_imp_listobject")
  }
} else {
  stop("Trainingset datanot in No ")
}

# dWOLS Function - all 
boot_dtr <- function(d, i=1:nrow(d), output="boo") {
  dat <- d[i,]
  
  # indefactorizebefore Preprocessing Column
  # Delete ColumnandnotneedtoIDColumn defactorize ConvertFailure 
  drop_cols <- c("charttime_kdigo3", "dod")
  for(col in drop_cols) {
    if(col %in% colnames(dat)) {
      dat[[col]] <- NULL
    }
  }
  
  dat <- as.data.frame(lapply(dat, defactorize))
  
  #### k=3 ####
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- dat[included_k3,]
  
  ps_mod_k3 <- glm(a3 ~ bun_k3 + ph_k3 + pot_k3 + uo_k3, family = "binomial", data= dat_k3)
  w_k3 <- with(dat_k3, abs(a3 - predict(ps_mod_k3, type = "response")) )
  
  pr_mod_k3 <- lm(hfd ~ admission_age + weight + gender + SOFA_24hours + bun_k3 +
                          a3 * uo_k3 + a3 * I(bun_k3/bun_k1),
                  weights = w_k3, data = dat_k3)
  
  all_coef_k3 <- coef(pr_mod_k3)
  names(all_coef_k3) <- paste0("mod_k3_", names(all_coef_k3))
  psi_k3 <- coef(pr_mod_k3)[grepl("a3", names(coef(pr_mod_k3)) )]
  
  mat_k3 <- model.matrix(~1 + uo_k3 + I(bun_k3/bun_k1), data=dat_k3)
  contrast_a1_k3 <- mat_k3 %*% psi_k3
  contrast_a0_k3 <- 0 
  optimal_contrast_k3 <- pmax(contrast_a1_k3, contrast_a0_k3)
  actual_contrast_k3 <- model.matrix(~-1 + a3 + a3:uo_k3 + a3:I(bun_k3/bun_k1), data=dat_k3) %*% psi_k3
  regret_k3 <- optimal_contrast_k3 - actual_contrast_k3
  
  dat$hfd_tilde_2 <- dat$hfd
  dat$hfd_tilde_2[included_k3] <-  dat$hfd_tilde_2[included_k3] + regret_k3
  
  #### k=2 ####
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- dat[included_k2,]
  
  ps_mod_k2 <- glm(a2 ~ bun_k2 + ph_k2 + pot_k2 + uo_k2, family = "binomial", data = dat_k2)
  w_k2 <- with(dat_k2, abs(a2 - predict(ps_mod_k2, type = "response")) )
  
  pr_mod_k2 <- lm(hfd_tilde_2 ~ admission_age + SOFA_24hours + weight + gender +
                          a2*SOFA_24hours + a2*bun_k2 + a2:I(abs(ph_k2-ph_k1)) + a2*I(uo_k2+uo_k1),
                  weights = w_k2, data = dat_k2)
  
  all_coef_k2 <- coef(pr_mod_k2)
  names(all_coef_k2) <- paste0("mod_k2_", names(all_coef_k2))
  psi_k2 <- coef(pr_mod_k2)[grepl("a2", names(coef(pr_mod_k2)) )]
  
  mat_k2 <- model.matrix(~1 + SOFA_24hours + bun_k2 + I(abs(ph_k2-ph_k1)) + I(uo_k2+uo_k1), data=dat_k2)
  contrast_a1_k2 <- mat_k2 %*% psi_k2
  contrast_a0_k2 <- 0 
  optimal_contrast_k2 <- pmax(contrast_a1_k2, contrast_a0_k2)
  actual_contrast_k2 <- model.matrix(~-1 + a2 + a2:SOFA_24hours + a2:bun_k2 + a2:I(abs(ph_k2 - ph_k1)) + a2:I(abs(uo_k2 + uo_k1)), data=dat_k2) %*% psi_k2
  regret_k2 <- optimal_contrast_k2 - actual_contrast_k2
  
  dat$hfd_tilde_1 <- dat$hfd
  dat$hfd_tilde_1[included_k2] <- dat_k2$hfd_tilde_2
  dat$hfd_tilde_1[included_k2] <-  dat$hfd_tilde_1[included_k2] + regret_k2
  
  #### k=1 ####
  dat_k1 <- dat
  
  ps_mod_k1 <- glm(a1 ~ bun_k1 + ph_k1 + pot_k1, family = "binomial", data = dat_k1)
  w_k1 <- with(dat_k1, abs(a1 - predict(ps_mod_k1, type = "response")) )
  
  pr_mod_k1 <- lm(hfd_tilde_1 ~ admission_age + SOFA_24hours + weight + gender +
                          uo_k1 + 
                          a1*admission_age + a1*creat_k1 + a1*pot_k1,
                  weights = w_k1, data = dat_k1)
  
  all_coef_k1 <- coef(pr_mod_k1)
  names(all_coef_k1) <- paste0("mod_k1_", names(all_coef_k1))
  psi_k1 <- coef(pr_mod_k1)[grepl("a1", names(coef(pr_mod_k1)) )]
  mat_k1 <- model.matrix(~1 + admission_age + creat_k1 + pot_k1, data=dat_k1)
  
  if(output=="boo"){
    return(c(all_coef_k1, all_coef_k2, all_coef_k3))
  } else if(output=="mats"){
    return(list(mat_k1, mat_k2, mat_k3))        
  } else if(output=="ttt"){
    return(list(a1=dat_k1$a1, a2=dat_k2$a2, a3=dat_k3$a3))        
  } else if(output=="ids"){
    return(list(ids_k1=dat_k1$subject_id, ids_k2=dat_k2$subject_id, ids_k3=dat_k3$subject_id))
  }
}

cat("\nStartand RowbootstrapTraining...\n")
cat(" use", n_cores - 1, " CPU cores\n")

n_boot <- 999
res <- list()
all_coefs <- list()
all_coefs_vars <- list()

# and RowProcessingeach  ImputationDataset
cat("Startand RowProcessing", exp_dat$m, " ImputationDataset...\n")
cat("each Datasetwill RowExecute", n_boot, " timesbootstrap and Row \n")

for(i in 1:exp_dat$m) {
  imp_dat <- complete(exp_dat, i)
  boot_res <- boot(imp_dat, boot_dtr, R=n_boot)
  res[[i]] <- boot_res
  all_coefs[[i]] <- res[[i]]$t0
  all_coefs_vars[[i]] <- apply(res[[i]]$t, 2, var)
  cat(i, "/", exp_dat$m, "DatasetalreadyComplete\n")
}

cat("\nComputationRubinMergeestimate...\n")

rubin_est <- c(); rubin_ests <- c()
rubin_var <- c(); rubin_vars <- c()

for(i in 1:length(all_coefs[[1]])) {
  rubin_est <- mean(sapply(all_coefs, function(x) x[[i]]))
  rubin_ests <- c(rubin_ests, rubin_est)
  
  rubin_var <- rubinr(sapply(all_coefs, function(x) x[[i]]), sapply(all_coefs_vars, function(x) x[[i]]) )
  rubin_vars <- c(rubin_vars, rubin_var)
}

names(rubin_ests) <- names(res[[1]]$t0)
names(rubin_vars) <- names(res[[1]]$t0)

est_k1 <- rubin_ests[grepl("a1", names(rubin_ests))]
est_k2 <- rubin_ests[grepl("a2", names(rubin_ests))]
est_k3 <- rubin_ests[grepl("a3", names(rubin_ests))]

cov_Mxs_k1 <- array(NA, dim = c(length(est_k1), length(est_k1), exp_dat$m))
cov_Mxs_k2 <- array(NA, dim = c(length(est_k2), length(est_k2), exp_dat$m))
cov_Mxs_k3 <- array(NA, dim = c(length(est_k3), length(est_k3), exp_dat$m))

for(i in 1:exp_dat$m) {
  cov_Mxs_k1[, , i] <- cov(res[[i]]$t[,grepl("a1", names(rubin_ests))] )
  cov_Mxs_k2[, , i] <- cov(res[[i]]$t[,grepl("a2", names(rubin_ests))] )
  cov_Mxs_k3[, , i] <- cov(res[[i]]$t[,grepl("a3", names(rubin_ests))] )
}

W_k1 <- apply(cov_Mxs_k1, c(1,2), mean)
W_k2 <- apply(cov_Mxs_k2, c(1,2), mean)
W_k3 <- apply(cov_Mxs_k3, c(1,2), mean)

temp_f <- function(x, a="a1", est=est_k1) {
  tet_vec <- x[grepl(a, names(rubin_vars))] - est 
  tet_vec %*% t(tet_vec)    
}

B_k1 <- 1/(exp_dat$m - 1) * Reduce('+', lapply(all_coefs, function(x) temp_f(x, a="a1", est=est_k1) ) )
B_k2 <- 1/(exp_dat$m - 1) * Reduce('+', lapply(all_coefs, function(x) temp_f(x, a="a2", est=est_k2) ) )
B_k3 <- 1/(exp_dat$m - 1) * Reduce('+', lapply(all_coefs, function(x) temp_f(x, a="a3", est=est_k3) ) )

vcv_k1 <- W_k1 + (1 + 1/exp_dat$m) * B_k1
vcv_k2 <- W_k2 + (1 + 1/exp_dat$m) * B_k2
vcv_k3 <- W_k3 + (1 + 1/exp_dat$m) * B_k3

cat("\n ITEPredictionandstandard error...\n")

mats_k123 <- list()
for(i in 1:exp_dat$m) {
  mats_k123[[i]] <- boot_dtr(d = complete(exp_dat, i), output = "mats")
  cat(i, "/", exp_dat$m, "DatasetalreadyComplete\n")
}

mean_list_of_matrices <- function(x) {
  Reduce("+", x)/ length(x)
}

mean_mat_k1 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[1]]))
mean_mat_k2 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[2]]))
mean_mat_k3 <- mean_list_of_matrices(lapply(mats_k123, function(x) x[[3]]))

ite_k1 <- mean_mat_k1 %*% est_k1
ite_k2 <- mean_mat_k2 %*% est_k2
ite_k3 <- mean_mat_k3 %*% est_k3

ite_se_k1 <- sqrt(sapply(as.list(as.data.frame(t(mean_mat_k1))), function(x) as.numeric(t(x) %*% vcv_k1 %*% x) ))
ite_se_k2 <- sqrt(sapply(as.list(as.data.frame(t(mean_mat_k2))), function(x) as.numeric(t(x) %*% vcv_k2 %*% x) ))
ite_se_k3 <- sqrt(sapply(as.list(as.data.frame(t(mean_mat_k3))), function(x) as.numeric(t(x) %*% vcv_k3 %*% x) ))

actual_ttts <- boot_dtr(d = complete(exp_dat, 1), output = "ttt")
actual_ids <- boot_dtr(d = complete(exp_dat, 1), output = "ids")

cat("\nTrainingComplete SaveResults...\n")

# Stopand Rowset 
stopCluster(cl)

# CreateOutputDirectory
dir.create("../results", recursive = TRUE, showWarnings = FALSE)

save(est_k1, est_k2, est_k3, 
     vcv_k1, vcv_k2, vcv_k3,
     ite_k1, ite_k2, ite_k3,
     ite_se_k1, ite_se_k2, ite_se_k3, 
     actual_ttts,
     file="../results/ite_preds_coef.RData")

cat("\nResults savedto: ../results/ite_preds_coef.RData\n")
cat("TrainingComplete \n")

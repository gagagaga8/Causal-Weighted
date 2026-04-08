# dWOLSTraining - Full Package Urine outputFeature 
# Dynamic-RRT-main/Learning/dWOLS.R

library(dplyr)
library(boot)

cat("============================================================\n")
cat("dWOLSTraining (Full - Package Urine outputFeature)\n")
cat("============================================================\n\n")

# Load data
cat("step1: Load data...\n")
dat <- read.csv("../data/dwols_full_with_uo.csv")
cat("  - Sample Size:", nrow(dat), "\n")
cat("  - Features:", ncol(dat), "\n")

# Feature
cat("\nUrine outputFeatureFull :\n")
for(col in c("uo_k1", "uo_k2", "uo_k3")) {
  if(col %in% colnames(dat)) {
    pct <- sum(!is.na(dat[[col]])) / nrow(dat) * 100
    cat(sprintf("  - %s: %.1f%%\n", col, pct))
  }
}

# Function
defactorize <- function(x) {
  if(is.factor(x)) as.character(x) else x
}

# Bootstrap dWOLSFunction - use Urine outputFeature
boot_dtr <- function(d, i=1:nrow(d), output="boo") {
  dat <- d[i,]
  dat <- as.data.frame(lapply(dat, defactorize))
  
  ################################################
  #### Backward induction procedure stage k=3 #### 
  ################################################
  
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3 <- dat[included_k3,]
  
  # FilterNAvalue
  valid_k3 <- complete.cases(dat_k3[, c("bun_k3", "ph_k3", "pot_k3", "uo_k3", "bun_k1")])
  dat_k3 <- dat_k3[valid_k3, ]
  
  if(nrow(dat_k3) < 10 || sum(dat_k3$a3) < 2) {
    all_coef_k3 <- rep(NA, 6)
    names(all_coef_k3) <- paste0("mod_k3_", 1:6)
    dat$hfd_tilde_2 <- dat$hfd
  } else {
    # to Model - with Package uo_k3
    ps_mod_k3 <- tryCatch({
      glm(a3 ~ bun_k3 + ph_k3 + pot_k3 + uo_k3, family = "binomial", data = dat_k3)
    }, error = function(e) NULL)
    
    if(is.null(ps_mod_k3)) {
      all_coef_k3 <- rep(NA, 6)
      names(all_coef_k3) <- paste0("mod_k3_", 1:6)
      dat$hfd_tilde_2 <- dat$hfd
    } else {
      w_k3 <- with(dat_k3, abs(a3 - predict(ps_mod_k3, type = "response")))
      
      # ResultsModel - with 
      pr_mod_k3 <- lm(hfd ~ admission_age + weight + gender + sofa_24hours + bun_k3 +
                              a3 * uo_k3 + a3 * I(bun_k3/bun_k1),
                      weights = w_k3, data = dat_k3)
      
      all_coef_k3 <- coef(pr_mod_k3)
      names(all_coef_k3) <- paste0("mod_k3_", names(all_coef_k3))
      psi_k3 <- coef(pr_mod_k3)[grepl("a3", names(coef(pr_mod_k3)))]
      
      mat_k3 <- model.matrix(~1 + uo_k3 + I(bun_k3/bun_k1), data=dat_k3)
      contrast_a1_k3 <- mat_k3 %*% psi_k3
      contrast_a0_k3 <- 0 
      optimal_contrast_k3 <- pmax(contrast_a1_k3, contrast_a0_k3)
      actual_contrast_k3 <- model.matrix(~-1 + a3 + a3:uo_k3 + a3:I(bun_k3/bun_k1), data=dat_k3) %*% psi_k3
      regret_k3 <- optimal_contrast_k3 - actual_contrast_k3
      
      dat$hfd_tilde_2 <- dat$hfd
      valid_indices <- which(included_k3)[valid_k3]
      dat$hfd_tilde_2[valid_indices] <- dat$hfd_tilde_2[valid_indices] + as.vector(regret_k3)
    }
  }
  
  ################################################
  #### Backward induction procedure stage k=2 #### 
  ################################################
  
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2 <- dat[included_k2,]
  
  # FilterNAvalue
  valid_k2 <- complete.cases(dat_k2[, c("bun_k2", "ph_k2", "pot_k2", "uo_k2", "uo_k1", "ph_k1")])
  dat_k2 <- dat_k2[valid_k2, ]
  
  if(nrow(dat_k2) < 10 || sum(dat_k2$a2) < 2) {
    all_coef_k2 <- rep(NA, 10)
    names(all_coef_k2) <- paste0("mod_k2_", 1:10)
    dat$hfd_tilde_1 <- dat$hfd
  } else {
    # to Model - Package uo_k2
    ps_mod_k2 <- tryCatch({
      glm(a2 ~ bun_k2 + ph_k2 + pot_k2 + uo_k2, family = "binomial", data = dat_k2)
    }, error = function(e) NULL)
    
    if(is.null(ps_mod_k2)) {
      all_coef_k2 <- rep(NA, 10)
      names(all_coef_k2) <- paste0("mod_k2_", 1:10)
      dat$hfd_tilde_1 <- dat$hfd
    } else {
      w_k2 <- with(dat_k2, abs(a2 - predict(ps_mod_k2, type = "response")))
      
      # ResultsModel - with Package uo items
      pr_mod_k2 <- lm(hfd_tilde_2 ~ admission_age + sofa_24hours + weight + gender +
                              a2*sofa_24hours + a2*bun_k2 + a2:I(abs(ph_k2-ph_k1)) + a2*I(uo_k2+uo_k1),
                      weights = w_k2, data = dat_k2)
      
      all_coef_k2 <- coef(pr_mod_k2)
      names(all_coef_k2) <- paste0("mod_k2_", names(all_coef_k2))
      psi_k2 <- coef(pr_mod_k2)[grepl("a2", names(coef(pr_mod_k2)))]
      
      mat_k2 <- model.matrix(~1 + sofa_24hours + bun_k2 + I(abs(ph_k2-ph_k1)) + I(uo_k2+uo_k1), data=dat_k2)
      contrast_a1_k2 <- mat_k2 %*% psi_k2
      contrast_a0_k2 <- 0 
      optimal_contrast_k2 <- pmax(contrast_a1_k2, contrast_a0_k2)
      actual_contrast_k2 <- model.matrix(~-1 + a2 + a2:sofa_24hours + a2:bun_k2 + a2:I(abs(ph_k2-ph_k1)) + a2:I(uo_k2+uo_k1), data=dat_k2) %*% psi_k2
      regret_k2 <- optimal_contrast_k2 - actual_contrast_k2
      
      dat$hfd_tilde_1 <- dat$hfd
      valid_indices <- which(included_k2)[valid_k2]
      dat$hfd_tilde_1[valid_indices] <- dat_k2$hfd_tilde_2[valid_k2]
      dat$hfd_tilde_1[valid_indices] <-  dat$hfd_tilde_1[valid_indices] + as.vector(regret_k2)
    }
  }
  
  ################################################
  #### Backward induction procedure stage k=1 #### 
  ################################################
  
  dat_k1 <- dat
  
  # FilterNAvalue
  valid_k1 <- complete.cases(dat_k1[, c("bun_k1", "ph_k1", "pot_k1", "uo_k1", "creat_k1", "admission_age")])
  dat_k1 <- dat_k1[valid_k1, ]
  
  # to Model
  ps_mod_k1 <- glm(a1 ~ bun_k1 + ph_k1 + pot_k1, family = "binomial", data = dat_k1)
  w_k1 <- with(dat_k1, abs(a1 - predict(ps_mod_k1, type = "response")))
  
  # ResultsModel - with 
  pr_mod_k1 <- lm(hfd_tilde_1 ~ admission_age + sofa_24hours + weight + gender +
                          uo_k1 + a1*admission_age + a1*creat_k1 + a1*pot_k1,
                  weights = w_k1, data = dat_k1)
  
  all_coef_k1 <- coef(pr_mod_k1)
  names(all_coef_k1) <- paste0("mod_k1_", names(all_coef_k1))
  psi_k1 <- coef(pr_mod_k1)[grepl("a1", names(coef(pr_mod_k1)))]
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

# BootstrapTraining
cat("\nstep2: BootstrapTraining (100 timesIteration)...\n")
set.seed(12345)
boot_res <- boot(dat, boot_dtr, R=100)

cat("\nStep 3: Extractcoefficient...\n")
coefs <- boot_res$t0
names(coefs) <- names(boot_res$t0)

# Extracteach decision pointcoefficient
est_k1 <- coefs[grepl("mod_k1.*a1", names(coefs))]
est_k2 <- coefs[grepl("mod_k2.*a2", names(coefs))]
est_k3 <- coefs[grepl("mod_k3.*a3", names(coefs))]

cat("  - k1coefficients:", length(est_k1), "\n")
cat("  - k2coefficients:", length(est_k2), "\n")
cat("  - k3coefficients:", length(est_k3), "\n")

# SaveResults
cat("\nStep 4: SaveModel...\n")
save(boot_res, coefs, est_k1, est_k2, est_k3, 
     file="../data/dwols_model_with_uo.RData")
cat(" - alreadySave : dwols_model_with_uo.RData\n")

# OutputcoefficientSummary
cat("\n=== ModelcoefficientSummary ===\n")
cat("\nk1coefficient:\n")
print(est_k1)
cat("\nk2coefficient:\n")
print(est_k2)
cat("\nk3coefficient:\n")
print(est_k3)

cat("\n============================================================\n")
cat("TrainingComplete \n")
cat("============================================================\n")

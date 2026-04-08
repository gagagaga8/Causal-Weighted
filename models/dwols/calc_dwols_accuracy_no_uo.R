# dWOLSNo Urine outputVersion - PredictionandAccuracyComputation
# usetrain_dwols_no_uo.RTrainingModelcoefficient

cat("============================================================\n")
cat("dWOLSAccuracyComputation (No Urine outputVersion)\n")
cat("============================================================\n\n")

# Load model coefficients
cat("Step 1: Loading model coefficients...\n")
load("../data/dwols_model_no_uo.RData")
cat("  - est_k1 coefficients:", length(est_k1), "\n")
cat("  - est_k2 coefficients:", length(est_k2), "\n")
cat("  - est_k3 coefficients:", length(est_k3), "\n")

# Load data
cat("\nStep 2: Loading data...\n")
dat <- read.csv("../data/dwols_full_no_uo.csv")
cat("  - Sample Size:", nrow(dat), "\n")

# get_mats_no_uoFunction - withTraining 
get_mats_no_uo <- function(d) {
  dat <- d
  
  # k=1: All Patient butneedtoFilterNA
  valid_k1 <- complete.cases(dat[, c("admission_age", "creat_k1", "pot_k1", "bun_k1")])
  dat_k1 <- dat[valid_k1, ]
  mat_k1 <- model.matrix(~1 + admission_age + creat_k1 + pot_k1 + bun_k1, data=dat_k1)
  
  # k=2: a1=1Patient
  excluded_k2 <- dat$a1 == 1
  included_k2 <- !excluded_k2
  dat_k2_pre <- dat[included_k2,]
  valid_k2 <- complete.cases(dat_k2_pre[, c("sofa_24hours", "bun_k2", "ph_k2", "ph_k1", "creat_k2")])
  dat_k2 <- dat_k2_pre[valid_k2, ]
  if(nrow(dat_k2) > 0) {
    mat_k2 <- model.matrix(~1 + sofa_24hours + bun_k2 + I(abs(ph_k2-ph_k1)) + creat_k2, data=dat_k2)
  } else {
    mat_k2 <- matrix(nrow=0, ncol=5)
  }
  
  # k=3: a2=1Patient
  excluded_k3 <- dat$a2 == 1
  included_k3 <- !excluded_k3
  dat_k3_pre <- dat[included_k3,]
  valid_k3 <- complete.cases(dat_k3_pre[, c("bun_k3", "bun_k1", "pot_k3")])
  dat_k3 <- dat_k3_pre[valid_k3, ]
  if(nrow(dat_k3) > 0) {
    mat_k3 <- model.matrix(~1 + I(bun_k3/bun_k1) + pot_k3, data=dat_k3)
  } else {
    mat_k3 <- matrix(nrow=0, ncol=3)
  }
  
  return(list(
    mat_k1 = mat_k1,
    mat_k2 = mat_k2,
    mat_k3 = mat_k3,
    a1 = dat_k1$a1,
    a2 = dat_k2$a2,
    a3 = dat_k3$a3
  ))
}

# ComputationITEPrediction
cat("\nStep 3: ComputationITEPrediction...\n")
result <- get_mats_no_uo(dat)

# ComputationITE
cat("  Computationk1 ITE...\n")
if(ncol(result$mat_k1) == length(est_k1)) {
  ite_k1 <- result$mat_k1 %*% est_k1
  cat("    - k1 ITEComputationSuccess:", length(ite_k1), " predictions\n")
} else {
  cat(" - k1 notmatch! mat:", ncol(result$mat_k1), "est:", length(est_k1), "\n")
  ite_k1 <- rep(NA, nrow(result$mat_k1))
}

cat("  Computationk2 ITE...\n")
if(ncol(result$mat_k2) == length(est_k2)) {
  ite_k2 <- result$mat_k2 %*% est_k2
  cat("    - k2 ITEComputationSuccess:", length(ite_k2), " predictions\n")
} else {
  cat(" - k2 notmatch! mat:", ncol(result$mat_k2), "est:", length(est_k2), "\n")
  ite_k2 <- rep(NA, nrow(result$mat_k2))
}

cat("  Computationk3 ITE...\n")
if(ncol(result$mat_k3) == length(est_k3)) {
  ite_k3 <- result$mat_k3 %*% est_k3
  cat("    - k3 ITEComputationSuccess:", length(ite_k3), " predictions\n")
} else {
  cat(" - k3 notmatch! mat:", ncol(result$mat_k3), "est:", length(est_k3), "\n")
  ite_k3 <- rep(NA, nrow(result$mat_k3))
}

# ComputationAccuracy
cat("\nStep 4: ComputationAccuracy...\n")
cat(" (policy: ITE > 0 then startRRT)\n\n")

calc_accuracy <- function(ite, actual_action, name) {
  if(length(ite) == 0 || all(is.na(ite))) {
    cat(sprintf(" %s: No Computation (No has Prediction)\n", name))
    return(NA)
  }
  
  # : ITE > 0 then start(1)
  recommended <- as.integer(ite > 0)
  
  # ComputationAgreement rate
  agreement <- sum(recommended == actual_action) / length(actual_action)
  
  # confusion matrix
  cat(sprintf("  %s:\n", name))
  cat(sprintf("    Sample Size: %d\n", length(actual_action)))
  cat(sprintf(" start : %.2f%%\n", mean(actual_action) * 100))
  cat(sprintf(" start : %.2f%%\n", mean(recommended) * 100))
  cat(sprintf("    Agreement with physician: %.2f%%\n", agreement * 100))
  
  # confusion matrix
  tbl <- table( =recommended, actual=actual_action)
  cat("    confusion matrix:\n")
  print(tbl)
  cat("\n")
  
  return(agreement)
}

acc_k1 <- calc_accuracy(ite_k1, result$a1, "decision pointk1")
acc_k2 <- calc_accuracy(ite_k2, result$a2, "decision pointk2")
acc_k3 <- calc_accuracy(ite_k3, result$a3, "decision pointk3")

# OverallAccuracy
total_correct <- 0
total_n <- 0

if(!is.na(acc_k1)) {
  total_correct <- total_correct + acc_k1 * length(result$a1)
  total_n <- total_n + length(result$a1)
}
if(!is.na(acc_k2)) {
  total_correct <- total_correct + acc_k2 * length(result$a2)
  total_n <- total_n + length(result$a2)
}
if(!is.na(acc_k3)) {
  total_correct <- total_correct + acc_k3 * length(result$a3)
  total_n <- total_n + length(result$a3)
}

if(total_n > 0) {
  total_acc <- total_correct / total_n
} else {
  total_acc <- NA
}

cat("============================================================\n")
cat("ResultsSummary\n")
cat("============================================================\n")
cat(sprintf("k1Accuracy: %.2f%%\n", ifelse(is.na(acc_k1), 0, acc_k1 * 100)))
cat(sprintf("k2Accuracy: %.2f%%\n", ifelse(is.na(acc_k2), 0, acc_k2 * 100)))
cat(sprintf("k3Accuracy: %.2f%%\n", ifelse(is.na(acc_k3), 0, acc_k3 * 100)))
cat(sprintf("\nOverallAccuracy: %.2f%%\n", ifelse(is.na(total_acc), 0, total_acc * 100)))
cat("============================================================\n")

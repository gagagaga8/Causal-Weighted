# Calculate dWOLS training accuracy (for overfitting evaluation)

cat("\n============================================================\n")
cat("dWOLS overfitting evaluation - train vs test accuracy\n")
cat("============================================================\n\n")

# Load training predictions
load("../results/ite_preds_coef.RData")

# Load test predictions
load("../results/test_predictions.RData")

# Accuracy calculation function
calculate_accuracy <- function(ite_pred, actual_action, name="") {
  if(length(ite_pred) == 0 || length(actual_action) == 0) {
    return(list(accuracy = NA, n = 0))
  }
  
  if(length(ite_pred) != length(actual_action)) {
    cat(sprintf("   [WARNING] %s Length mismatch: predicted=%d, actual=%d\n", 
        name, length(ite_pred), length(actual_action)))
    return(list(accuracy = NA, n = 0))
  }
  
  # dWOLS policy: ITE > 0 recommends initiating RRT
  predicted_action <- as.integer(ite_pred > 0)
  correct <- sum(predicted_action == actual_action, na.rm=TRUE)
  valid_n <- sum(!is.na(predicted_action) & !is.na(actual_action))
  accuracy <- correct / valid_n
  
  return(list(accuracy = accuracy, n = valid_n, correct = correct))
}

# ============================================================
# 1. Training accuracy
# ============================================================
cat("1. Training accuracy:\n")
cat("-" , rep("-", 50), "\n", sep="")

# Check ITE predictions and actual_ttts (training labels)
if(exists("ite_k1") && exists("ite_k2") && exists("ite_k3") && exists("actual_ttts")) {
  # actual_ttts matches ite_k1/k2/k3 length
  train_k1 <- calculate_accuracy(ite_k1, actual_ttts$a1, "train_k1")
  train_k2 <- calculate_accuracy(ite_k2, actual_ttts$a2, "train_k2")
  train_k3 <- calculate_accuracy(ite_k3, actual_ttts$a3, "train_k3")
  
  cat(sprintf("   k1: %.2f%% (%d/%d)\n", train_k1$accuracy*100, train_k1$correct, train_k1$n))
  cat(sprintf("   k2: %.2f%% (%d/%d)\n", train_k2$accuracy*100, train_k2$correct, train_k2$n))
  cat(sprintf("   k3: %.2f%% (%d/%d)\n", train_k3$accuracy*100, train_k3$correct, train_k3$n))
  
  train_total_correct <- sum(c(train_k1$correct, train_k2$correct, train_k3$correct), na.rm=TRUE)
  train_total_n <- sum(c(train_k1$n, train_k2$n, train_k3$n), na.rm=TRUE)
  train_accuracy <- train_total_correct / train_total_n
  
  cat(sprintf("\n   Overall training accuracy: %.2f%%\n", train_accuracy * 100))
} else {
  cat("   Cannot get training ITE predictions or actual_ttts\n")
  train_accuracy <- NA
}

# ============================================================
# 2. Test accuracy
# ============================================================
cat("\n2. Test accuracy:\n")
cat("-", rep("-", 50), "\n", sep="")

test_k1 <- calculate_accuracy(ite_k1_test, actual_ttts_test$a1, "test_k1")
test_k2 <- calculate_accuracy(ite_k2_test, actual_ttts_test$a2, "test_k2")
test_k3 <- calculate_accuracy(ite_k3_test, actual_ttts_test$a3, "test_k3")

cat(sprintf("   k1: %.2f%% (%d/%d)\n", test_k1$accuracy*100, test_k1$correct, test_k1$n))
cat(sprintf("   k2: %.2f%% (%d/%d)\n", test_k2$accuracy*100, test_k2$correct, test_k2$n))
cat(sprintf("   k3: %.2f%% (%d/%d)\n", test_k3$accuracy*100, test_k3$correct, test_k3$n))

test_total_correct <- sum(c(test_k1$correct, test_k2$correct, test_k3$correct), na.rm=TRUE)
test_total_n <- sum(c(test_k1$n, test_k2$n, test_k3$n), na.rm=TRUE)
test_accuracy <- test_total_correct / test_total_n

cat(sprintf("\n   Overall test accuracy: %.2f%%\n", test_accuracy * 100))

# ============================================================
# 3. OverfittingEvaluation
# ============================================================
cat("\n============================================================\n")
cat("3. OverfittingEvaluation\n")
cat("============================================================\n\n")

if(!is.na(train_accuracy) && !is.na(test_accuracy)) {
  gap <- train_accuracy - test_accuracy
  
  cat(sprintf("   Training accuracy: %.2f%%\n", train_accuracy * 100))
  cat(sprintf("   Test accuracy: %.2f%%\n", test_accuracy * 100))
  cat(sprintf("   OverfittingGap:   %.2f%%\n", gap * 100))
  
  cat("\n   EvaluationResults: ")
  if(gap < 0.05) {
    cat("[Normal] No Overfitting\n")
  } else if(gap < 0.10) {
    cat("[ ] Slight overfitting\n")
  } else {
    cat("[WARNING] inOverfitting\n")
  }
} else {
  cat(" No ComputationOverfittingGap Trainingsetor Test accuracyMissing \n")
}

# SaveResults
overfit_results <- list(
  train_accuracy = train_accuracy,
  test_accuracy = test_accuracy,
  gap = if(!is.na(train_accuracy)) train_accuracy - test_accuracy else NA
)
save(overfit_results, file="../results/overfit_analysis.RData")

cat("\nResults savedto: ../results/overfit_analysis.RData\n")
cat("============================================================\n")

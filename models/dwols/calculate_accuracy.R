# ComputationdWOLSTest accuracy

cat("\n============================================================\n")
cat("dWOLSTest accuracyComputation\n")
cat("============================================================\n\n")

# Load test predictions
load("../results/test_predictions.RData")

cat("1. PredictionResults:\n")
cat(sprintf("   ite_k1_test: %d  predictions\n", length(ite_k1_test)))
cat(sprintf("   ite_k2_test: %d  predictions\n", length(ite_k2_test)))
cat(sprintf("   ite_k3_test: %d  predictions\n", length(ite_k3_test)))

cat(sprintf("\n   actual_ttts_test$a1: %d  actual actions\n", length(actual_ttts_test$a1)))
cat(sprintf("   actual_ttts_test$a2: %d  actual actions\n", length(actual_ttts_test$a2)))
cat(sprintf("   actual_ttts_test$a3: %d  actual actions\n", length(actual_ttts_test$a3)))

# Computationeach decision pointAccuracy
calculate_accuracy <- function(ite_pred, actual_action) {
  if(length(ite_pred) == 0 || length(actual_action) == 0) {
    return(list(accuracy = NA, n = 0))
  }
  
  if(length(ite_pred) != length(actual_action)) {
    cat(sprintf("   [WARNING] Length mismatch: predicted=%d, actual=%d\n", 
        length(ite_pred), length(actual_action)))
    return(list(accuracy = NA, n = 0))
  }
  
  # dWOLS policy: ITE > 0 recommends initiating RRT
  predicted_action <- as.integer(ite_pred > 0)
  correct <- sum(predicted_action == actual_action)
  accuracy <- correct / length(actual_action)
  
  return(list(
    accuracy = accuracy,
    n = length(actual_action),
    correct = correct,
    predicted_start = sum(predicted_action),
    actual_start = sum(actual_action)
  ))
}

cat("\n2. ComputationAccuracy:\n\n")

# k1
result_k1 <- calculate_accuracy(ite_k1_test, actual_ttts_test$a1)
cat("   decision point k1:\n")
if(!is.na(result_k1$accuracy)) {
  cat(sprintf("     Accuracy: %.2f%% (%d/%d)\n", 
      result_k1$accuracy * 100, result_k1$correct, result_k1$n))
  cat(sprintf(" start: %d (%.1f%%)\n", 
      result_k1$predicted_start, result_k1$predicted_start/result_k1$n*100))
  cat(sprintf(" start: %d (%.1f%%)\n\n", 
      result_k1$actual_start, result_k1$actual_start/result_k1$n*100))
} else {
  cat(" No Computation\n\n")
}

# k2
result_k2 <- calculate_accuracy(ite_k2_test, actual_ttts_test$a2)
cat("   decision point k2:\n")
if(!is.na(result_k2$accuracy)) {
  cat(sprintf("     Accuracy: %.2f%% (%d/%d)\n", 
      result_k2$accuracy * 100, result_k2$correct, result_k2$n))
  cat(sprintf(" start: %d (%.1f%%)\n", 
      result_k2$predicted_start, result_k2$predicted_start/result_k2$n*100))
  cat(sprintf(" start: %d (%.1f%%)\n\n", 
      result_k2$actual_start, result_k2$actual_start/result_k2$n*100))
} else {
  cat(" No Computation\n\n")
}

# k3
result_k3 <- calculate_accuracy(ite_k3_test, actual_ttts_test$a3)
cat("   decision point k3:\n")
if(!is.na(result_k3$accuracy)) {
  cat(sprintf("     Accuracy: %.2f%% (%d/%d)\n", 
      result_k3$accuracy * 100, result_k3$correct, result_k3$n))
  cat(sprintf(" start: %d (%.1f%%)\n", 
      result_k3$predicted_start, result_k3$predicted_start/result_k3$n*100))
  cat(sprintf(" start: %d (%.1f%%)\n\n", 
      result_k3$actual_start, result_k3$actual_start/result_k3$n*100))
} else {
  cat(" No Computation\n\n")
}

# ComputationOverallAccuracy
total_correct <- sum(c(result_k1$correct, result_k2$correct, result_k3$correct), na.rm=TRUE)
total_n <- sum(c(result_k1$n, result_k2$n, result_k3$n), na.rm=TRUE)

if(total_n > 0) {
  test_accuracy <- total_correct / total_n
  cat(sprintf("3. OverallAccuracy: %.2f%% (%d/%d)\n", 
      test_accuracy * 100, total_correct, total_n))
} else {
  test_accuracy <- NA
  cat("3. OverallAccuracy: No Computation\n")
}

# SaveResults
test_accuracy_by_k <- data.frame(
  decision_point = c("k1", "k2", "k3"),
  accuracy = c(result_k1$accuracy, result_k2$accuracy, result_k3$accuracy),
  n = c(result_k1$n, result_k2$n, result_k3$n),
  correct = c(result_k1$correct, result_k2$correct, result_k3$correct)
)

save(test_accuracy, test_accuracy_by_k, 
     file="../results/test_results.RData")

cat("\n============================================================\n")
cat(" \n")
cat("============================================================\n")

if(!is.na(test_accuracy)) {
  cat(sprintf("\ndWOLSTrainingstatus: [Success]\n"))
  cat(sprintf("Overall test accuracy: %.2f%%\n", test_accuracy * 100))
  cat("\n time pointAccuracy:\n")
  print(test_accuracy_by_k)
} else {
  cat("\ndWOLSTrainingstatus: [Failure]\n")
  cat(" : No ComputationAccuracy PredictionResultsallasNA \n")
}

cat("\nResults savedto: ../results/test_results.RData\n")
cat("============================================================\n")

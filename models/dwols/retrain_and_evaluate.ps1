# dWOLS full training and evaluation pipeline (PowerShell)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "dWOLSModelTrainingandEvaluation" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# step1: Results
Write-Host "step1: TrainingResults..." -ForegroundColor Yellow
if (Test-Path "..\results\ite_preds_coef.RData") {
    Remove-Item "..\results\ite_preds_coef.RData" -Force
    Write-Host " alreadyDelete Modelcoefficient" -ForegroundColor Green
}
if (Test-Path "..\results\test_predictions.RData") {
    Remove-Item "..\results\test_predictions.RData" -Force
    Write-Host " alreadyDelete TestPrediction" -ForegroundColor Green
}
Write-Host ""

# step2: TrainingdWOLS
Write-Host "step2: TrainingdWOLSModel 999 timesbootstrap ..." -ForegroundColor Yellow
Write-Host " willneedto4-5 hours ..." -ForegroundColor Yellow
Write-Host "TrainingData: 1158Patient × 100ImputationDataset × 999 timesbootstrap" -ForegroundColor Gray
Write-Host ""

$r_path = "C:\Program Files\R\R-4.4.1\bin\R.exe"
if (Test-Path $r_path) {
    & $r_path CMD BATCH --no-save --no-restore "dWOLS_parallel.R" "dWOLS_parallel.Rout"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[Error] dWOLSTrainingFailure" -ForegroundColor Red
        Write-Host " Log: dWOLS_parallel.Rout" -ForegroundColor Yellow
        pause
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "[Complete] dWOLSTrainingSuccess" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[Error] not toR : $r_path" -ForegroundColor Red
    pause
    exit 1
}

# Step 3: TestsetEvaluation
Write-Host "Step 3: inTestsetonEvaluationModel..." -ForegroundColor Yellow
& $r_path CMD BATCH --no-save --no-restore "Get_predictions_test.R" "Get_predictions_test.Rout"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[Error] TestsetEvaluationFailure" -ForegroundColor Red
    Write-Host " Log: Get_predictions_test.Rout" -ForegroundColor Yellow
    pause
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[Complete] TestsetEvaluationSuccess" -ForegroundColor Green
Write-Host ""

# Step 4: ComputationAccuracy
Write-Host "Step 4: ComputationAccuracy..." -ForegroundColor Yellow
& $r_path CMD BATCH --no-save --no-restore "calculate_accuracy.R" "calculate_accuracy.Rout"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] AccuracyComputationcancanFailure" -ForegroundColor Yellow
    Write-Host " Log: calculate_accuracy.Rout" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "dWOLSpipelineComplete" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ResultsFile:" -ForegroundColor Green
Write-Host "  - ..\results\ite_preds_coef.RData (Modelcoefficient)"
Write-Host "  - ..\results\test_predictions.RData (TestPrediction)"
Write-Host "  - ..\results\test_results.RData (Accuracy)"
Write-Host ""
Write-Host "LogFile:" -ForegroundColor Yellow
Write-Host "  - dWOLS_parallel.Rout"
Write-Host "  - Get_predictions_test.Rout"
Write-Host "  - calculate_accuracy.Rout"
Write-Host ""

# AccuracyResults
if (Test-Path "..\results\test_results.RData") {
    Write-Host " AccuracyResults..." -ForegroundColor Cyan
    & $r_path --vanilla -e "load('../results/test_results.RData'); cat('\nAccuracy:', sprintf('%.2f%%', test_accuracy*100), '\n'); print(test_accuracy_by_k)"
}

Write-Host ""
pause

@echo off
REM dWOLSFull TrainingandEvaluationpipeline

echo ================================================================================
echo dWOLSModelTrainingandEvaluation
echo ================================================================================
echo.

cd /d "%~dp0"

echo step1: TrainingResults...
if exist "..\results\ite_preds_coef.RData" (
    del /Q "..\results\ite_preds_coef.RData"
    echo alreadyDelete Modelcoefficient
)
if exist "..\results\test_predictions.RData" (
    del /Q "..\results\test_predictions.RData"
    echo alreadyDelete TestPrediction
)
echo.

echo step2: TrainingdWOLSModel 999 timesbootstrap ...
echo willneedto ...
echo.
R.exe CMD BATCH --no-save --no-restore "dWOLS_parallel.R" "dWOLS_parallel.Rout"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] dWOLSTrainingFailure
    echo Check log: dWOLS_parallel.Rout
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Complete] dWOLSTrainingSuccess
echo.

echo Step 3: inTestsetonEvaluationModel...
R.exe CMD BATCH --no-save --no-restore "Get_predictions_test.R" "Get_predictions_test.Rout"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] TestsetEvaluationFailure
    echo Log: Get_predictions_test.Rout
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Complete] TestsetEvaluationSuccess
echo.

echo Step 4: ComputationAccuracy...
R.exe CMD BATCH --no-save --no-restore "calculate_accuracy.R" "calculate_accuracy.Rout"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] AccuracyComputationcancanFailure
    echo Log: calculate_accuracy.Rout
)

echo.
echo ================================================================================
echo dWOLSpipelineComplete
echo ================================================================================
echo ResultsFile:
echo   - ..\results\ite_preds_coef.RData (Modelcoefficient)
echo   - ..\results\test_predictions.RData (TestPrediction)
echo   - ..\results\test_results.RData (Accuracy)
echo.
echo LogFile:
echo   - dWOLS_parallel.Rout
echo   - Get_predictions_test.Rout
echo   - calculate_accuracy.Rout
echo.
pause

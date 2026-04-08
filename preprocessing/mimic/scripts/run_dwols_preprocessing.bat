@echo off
REM dWOLS preprocessing start script

echo ===============================================
echo dWOLS preprocessing (MIMIC-IV)
echo ===============================================
echo.

cd /d "%~dp0"

echo Running preprocessing...
echo Path: %CD%
echo.

R.exe CMD BATCH --no-save --no-restore "dWOLSPreprocessing.R" "dWOLSPreprocessing.Rout"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] PreprocessingFailure Error : %ERRORLEVEL%
    echo Log: dWOLSPreprocessing.Rout
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Complete] PreprocessingSuccessComplete
echo OutputFile: ../data/mimic_dwols_preprocessed.csv
echo LogFile: dWOLSPreprocessing.Rout
echo.
pause

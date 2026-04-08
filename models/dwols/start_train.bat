@echo off
echo ========================================
echo dWOLS training started
echo ========================================
echo.

cd /d "%~dp0"

echo Training data: 1158 patients x 100 imputations x 999 bootstrap
echo Estimated time: 4-5 hours
echo.

set R_PATH=C:\Program Files\R\R-4.5.2\bin\R.exe

if not exist "%R_PATH%" (
    echo [ERROR] R not found
    pause
    exit /b 1
)

echo Starting training...
"%R_PATH%" CMD BATCH --no-save --no-restore dWOLS_parallel.R dWOLS_parallel.Rout

echo.
if %ERRORLEVEL% EQU 0 (
    echo [DONE] Training complete
) else (
    echo [ERROR] Training failed, exit code: %ERRORLEVEL%
)

echo Check log: dWOLS_parallel.Rout
pause

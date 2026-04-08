@echo off
echo Generate trendFeatureRLDataset...
cd /d "c:\Dynamic-RRT\3_DataSplit\scripts"
"C:\Program Files\R\R-4.5.2\bin\Rscript.exe" prepare_iql_data.R
if %errorlevel% neq 0 (
    echo R ExecuteFailure
    pause
    exit /b 1
)
echo RLDatasetGenerateComplete
pause

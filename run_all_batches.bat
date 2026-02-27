@echo off
echo ===================================================
echo ALGORITHMIC ANCHORING - FULL EXPERIMENT BATCH RUN
echo ===================================================
echo.

set RSCRIPT="C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
set SCRIPT="R\anchoring_experiment.R"

echo Running full experiment synchronously for all models...
echo NOTE: This will take several hours to complete. Do not close this window!
echo.

echo [1/4] Running Batch 1...
%RSCRIPT% %SCRIPT% --mode full --batch 1
echo Batch 1 complete.
echo.

echo [2/4] Running Batch 2...
%RSCRIPT% %SCRIPT% --mode full --batch 2
echo Batch 2 complete.
echo.

echo [3/4] Running Batch 3...
%RSCRIPT% %SCRIPT% --mode full --batch 3
echo Batch 3 complete.
echo.

echo [4/4] Running Batch 4...
%RSCRIPT% %SCRIPT% --mode full --batch 4
echo Batch 4 complete.
echo.

echo ===============================================================================
echo ALL STAGES COMPLETED!
echo ===============================================================================

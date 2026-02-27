@echo off
echo ===================================================
echo ALGORITHMIC ANCHORING - RUN BATCH ANALYSIS
echo ===================================================
echo.

set RSCRIPT="C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
set SCRIPT="R\analysis_pipeline.R"

echo [1/4] Running Analysis for Batch 1...
%RSCRIPT% %SCRIPT% --input "results\batch\batch_1_checkpoint_20260220_1917.csv" > "results\batch\analysis_output_batch_1.txt"
echo Batch 1 analysis complete. Results saved to results\batch\analysis_output_batch_1.txt
echo.

echo [2/4] Running Analysis for Batch 2...
%RSCRIPT% %SCRIPT% --input "results\batch\batch_2_checkpoint_20260221_0524.csv" > "results\batch\analysis_output_batch_2.txt"
echo Batch 2 analysis complete. Results saved to results\batch\analysis_output_batch_2.txt
echo.

echo [3/4] Running Analysis for Batch 3...
%RSCRIPT% %SCRIPT% --input "results\batch\batch_3_checkpoint_20260221_1626.csv" > "results\batch\analysis_output_batch_3.txt"
echo Batch 3 analysis complete. Results saved to results\batch\analysis_output_batch_3.txt
echo.

echo [4/4] Running Analysis for Batch 4...
%RSCRIPT% %SCRIPT% --input "results\batch\batch_4_checkpoint_20260222_1926.csv" > "results\batch\analysis_output_batch_4.txt"
echo Batch 4 analysis complete. Results saved to results\batch\analysis_output_batch_4.txt
echo.

echo ===============================================================================
echo ALL ANALYSES COMPLETED!
echo ===============================================================================

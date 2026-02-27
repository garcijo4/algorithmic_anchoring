source("R/analysis_pipeline.R")
df <- load_and_parse("results/batch/batch_4_checkpoint_20260222_1926.csv")

cat(">> Testing 5A <<\n")
suppressWarnings(descriptive_statistics(df))

cat("\n>> Testing 5B & 5C <<\n")
suppressWarnings(run_regressions(df))
cat("\nTEST COMPLETE\n")

library(dplyr)
library(clubSandwich)

source("R/analysis_pipeline.R")
df <- load_and_parse("results/batch/batch_4_checkpoint_20260222_1926.csv")
sur_df <- df %>%
  filter(parse_success, !is_control,
         !is.na(anchor_dev_pct),
         !is.na(pct_dev_from_control),
         !is.na(implied_growth), !is.na(implied_pe),
         !is.na(rec_numeric),
         company_id != "FC02") %>%
  mutate(model_factor = factor(model))

cat("Rows:", nrow(sur_df), "\n")
m <- lm(pct_dev_from_control ~ anchor_dev_pct + model_factor, data = sur_df)
v_cluster <- sur_df$company_id[as.numeric(rownames(m$model))]

cat("Starting cluster_se_cr2(force_cr1s=TRUE)...\n")
res <- cluster_se_cr2(m, v_cluster, force_cr1s = TRUE)
print(res)
cat("Script complete.\n")

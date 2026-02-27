df <- read.csv('results/batch/batch_2_checkpoint_20260221_0524.csv')
h1_df <- df[df$parse_success & df$anchor_direction != 'control',]

# Spec 3a-norm logic verbatim:
h1_df_norm_clean <- h1_df[complete.cases(
  h1_df[, c("pct_dev_from_control", "anchor_dev_pct",
            "log_revenue", "ebitda_margin", "debt_to_equity")]), ]

cat("Rows in h1_df_norm_clean:", nrow(h1_df_norm_clean), "\n")
cat("Unique models:", toString(unique(h1_df_norm_clean$model)), "\n")

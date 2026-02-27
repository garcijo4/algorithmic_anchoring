# ============================================================================
# Supplemental Analysis: Formal Asymmetry Test (High vs. Low Anchor Effects)
# Author: John Garcia, California Lutheran University
# Version: 1.0 (February 2026)
#
# Purpose: Formally tests whether high-anchor effects are significantly larger
#          in magnitude than low-anchor effects, as suggested by the descriptive
#          statistics (mean bias from high anchor: +11.9% vs. low anchor: -0.4%).
#
# Usage:
#   source("analysis_supplement_asymmetry.R")
#   # Requires: reg_df data frame from analysis_pipeline.R (Batch 4, FC02 excluded)
#   results <- run_asymmetry_tests(reg_df)
#
# Outputs:
#   1. Paired t-test on |Cohen's d| for high vs. low across company-model cells
#   2. Regression with anchor_direction interaction to test differential slopes
#   3. Company-level paired comparison of absolute bias magnitudes
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(sandwich)
  library(lmtest)
})

has_clubSandwich <- requireNamespace("clubSandwich", quietly = TRUE)

run_asymmetry_tests <- function(reg_df) {

  cat("\n===== FORMAL ASYMMETRY TEST: HIGH vs. LOW ANCHOR EFFECTS =====\n\n")

  results <- list()

  # --------------------------------------------------------------------------
  # Test 1: Paired t-test on absolute Cohen's d (high vs. low)
  # --------------------------------------------------------------------------
  cat("--- Test 1: Paired t-test on |Cohen's d| across company-model cells ---\n")

  # Compute Cohen's d for each company × model cell
  d_df <- reg_df %>%
    filter(anchor_direction %in% c("high", "low", "control")) %>%
    group_by(company_id, model, anchor_direction) %>%
    summarise(mean_est = mean(point_estimate, na.rm = TRUE),
              sd_est   = sd(point_estimate, na.rm = TRUE),
              n        = n(),
              .groups = "drop") %>%
    pivot_wider(names_from = anchor_direction,
                values_from = c(mean_est, sd_est, n),
                names_sep = "_")

  d_df <- d_df %>%
    mutate(
      pooled_sd_high = sqrt(((n_high - 1) * sd_est_high^2 + (n_control - 1) * sd_est_control^2) /
                              (n_high + n_control - 2)),
      pooled_sd_low  = sqrt(((n_low - 1) * sd_est_low^2 + (n_control - 1) * sd_est_control^2) /
                              (n_low + n_control - 2)),
      d_high = (mean_est_high - mean_est_control) / pooled_sd_high,
      d_low  = (mean_est_low  - mean_est_control) / pooled_sd_low,
      abs_d_high = abs(d_high),
      abs_d_low  = abs(d_low)
    )

  paired_test <- t.test(d_df$abs_d_high, d_df$abs_d_low, paired = TRUE)
  cat(sprintf("  Mean |d_high| = %.3f, Mean |d_low| = %.3f\n",
              mean(d_df$abs_d_high), mean(d_df$abs_d_low)))
  cat(sprintf("  Paired t(%d) = %.3f, p = %.4f\n",
              paired_test$parameter, paired_test$statistic, paired_test$p.value))
  cat(sprintf("  Mean difference (|d_high| - |d_low|) = %.3f, 95%% CI [%.3f, %.3f]\n",
              paired_test$estimate, paired_test$conf.int[1], paired_test$conf.int[2]))

  wilcox_test <- wilcox.test(d_df$abs_d_high, d_df$abs_d_low, paired = TRUE)
  cat(sprintf("  Wilcoxon signed-rank: V = %.0f, p = %.4f\n\n",
              wilcox_test$statistic, wilcox_test$p.value))

  results$paired_d_test <- paired_test
  results$wilcoxon_d_test <- wilcox_test

  # --------------------------------------------------------------------------
  # Test 2: Regression with anchor direction indicator × anchor magnitude
  # --------------------------------------------------------------------------
  cat("--- Test 2: Regression with direction asymmetry ---\n")

  asym_df <- reg_df %>%
    filter(anchor_direction %in% c("high", "low"),
           !is.na(pct_dev_from_control), !is.na(anchor_dev_pct)) %>%
    mutate(
      is_high = as.integer(anchor_direction == "high"),
      abs_anchor_dev = abs(anchor_dev_pct)
    )

  # Model: pct_dev = b0 + b1*anchor_dev_pct + b2*is_high + b3*anchor_dev_pct*is_high + model FE + e
  # b3 tests whether the slope of anchor on estimates differs by direction
  m_asym <- lm(pct_dev_from_control ~ anchor_dev_pct * is_high + model_factor,
               data = asym_df)

  vcov_cl <- vcovCL(m_asym, cluster = asym_df$company_id)
  ct <- coeftest(m_asym, vcov. = vcov_cl)
  cat("  OLS with cluster-robust SEs (company level):\n")
  print(ct)

  results$asymmetry_regression <- ct

  # --------------------------------------------------------------------------
  # Test 3: Company-level paired comparison of absolute bias magnitudes
  # --------------------------------------------------------------------------
  cat("\n--- Test 3: Company-level absolute bias comparison ---\n")

  bias_df <- reg_df %>%
    filter(anchor_direction %in% c("high", "low", "control")) %>%
    group_by(company_id, anchor_direction) %>%
    summarise(mean_est = mean(point_estimate, na.rm = TRUE),
              .groups = "drop") %>%
    pivot_wider(names_from = anchor_direction, values_from = mean_est) %>%
    mutate(
      abs_bias_high = abs(high - control),
      abs_bias_low  = abs(low - control),
      bias_high_pct = (high - control) / control * 100,
      bias_low_pct  = (low - control)  / control * 100
    )

  cat("  Company-level bias magnitudes:\n")
  print(bias_df %>% select(company_id, bias_high_pct, bias_low_pct, abs_bias_high, abs_bias_low))

  company_paired <- t.test(bias_df$abs_bias_high, bias_df$abs_bias_low, paired = TRUE)
  cat(sprintf("\n  Paired t-test on absolute bias (company level, N=%d):\n", nrow(bias_df)))
  cat(sprintf("  Mean |bias_high| = $%.2f, Mean |bias_low| = $%.2f\n",
              mean(bias_df$abs_bias_high), mean(bias_df$abs_bias_low)))
  cat(sprintf("  t(%d) = %.3f, p = %.4f\n",
              company_paired$parameter, company_paired$statistic, company_paired$p.value))

  results$company_paired_test <- company_paired

  # --------------------------------------------------------------------------
  # Test 4: Model-specific debiasing regressions (three-way table)
  # --------------------------------------------------------------------------
  cat("\n\n--- Test 4: Net anchor coefficient by model × debiasing ---\n")

  if ("debiasing" %in% names(reg_df)) {
    debiasing_models <- reg_df %>%
      filter(!is.na(pct_dev_from_control), !is.na(anchor_dev_pct)) %>%
      group_by(model) %>%
      group_split()

    for (mdf in debiasing_models) {
      model_name <- unique(mdf$model)
      cat(sprintf("\n  [%s]\n", model_name))

      mdf$debiasing_factor <- factor(mdf$debiasing, levels = c("none", "cot", "warning",
                                                                  "adversarial", "multi_source", "neutral"))
      m <- lm(pct_dev_from_control ~ anchor_dev_pct * debiasing_factor, data = mdf)
      vcov_cl <- tryCatch(vcovCL(m, cluster = mdf$company_id), error = function(e) vcovHC(m))
      ct <- coeftest(m, vcov. = vcov_cl)

      # Extract net anchor coefficient per debiasing condition
      base_coef <- coef(m)["anchor_dev_pct"]
      interactions <- grep("anchor_dev_pct:debiasing_factor", names(coef(m)), value = TRUE)

      cat(sprintf("    Baseline (none): anchor coef = %.4f\n", base_coef))
      for (int_name in interactions) {
        condition <- gsub("anchor_dev_pct:debiasing_factor", "", int_name)
        net_coef <- base_coef + coef(m)[int_name]
        cat(sprintf("    %s: net anchor coef = %.4f (interaction = %.4f, p = %.4f)\n",
                    condition, net_coef, coef(m)[int_name], ct[int_name, 4]))
      }
    }
  }

  cat("\n===== ASYMMETRY TESTS COMPLETE =====\n")

  return(results)
}

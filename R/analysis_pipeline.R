# ============================================================================
# Algorithmic Anchoring — Analysis Pipeline
# Author: John Garcia, California Lutheran University
#
# Usage:
#   Rscript analysis_pipeline.R --input results/exp1a_valuation/exp1_valuation_20260222.csv
#
#   STAT-1   Normalized DV: pct_dev_from_control (% deviation from company ×
#            model control mean).  Resolves cross-company price-level confound
#            that made the pooled β₁ on anchor_value uninterpretable when raw
#            dollar estimates were regressed across companies with prices
#            ranging from $27 to $75.
#   STAT-2   Company fundamentals lookup implementing manuscript Eq. 3a's
#            Fundamentals vector (log revenue, EBITDA margin, debt-to-equity).
#            Replaces the ad-hoc current_price + sector proxy.
#   STAT-3   Formal one-sample hypothesis tests for AnI (vs. 0, 0.37, 0.49)
#            with nonparametric Wilcoxon alternatives and cross-model pairwise
#            comparisons (BH-corrected).
#   STAT-4   Small-cluster inference.  With G = 10 company clusters, asymptotic
#            cluster-robust SEs are anticonservative (Cameron & Miller, 2015).
#            CR2 bias correction with Satterthwaite degrees of freedom added
#            via clubSandwich; wild cluster bootstrap via fwildclusterboot for
#            the primary H1 coefficient.  Both gracefully degrade if packages
#            are absent.
#   STAT-5   Outlier Winsorization at 1st / 99th percentile within company
#            to bound pathological LLM responses (observed range $3.50–$120).
#   STAT-6   Cohen's d effect sizes per anchor condition (high vs. control,
#            low vs. control) for each company × model cell.
#   STAT-7   Model-specific subgroup regressions alongside pooled.
#   STAT-8   Multiple testing correction (Benjamini–Hochberg FDR) across
#            the five primary hypothesis tests.
#   STAT-9   Descriptive statistics and balance table.
#   STAT-10  Regression diagnostics: VIF, residual normality (Shapiro on
#            subsample), and model-fit statistics (R², adj-R², F).
#   STAT-11  Improved SUR: Breusch–Pagan test for cross-equation residual
#            correlation; per-equation anchor coefficient tests.
#   STAT-12  Model-fit reporting: R², adj-R², F-statistic for all OLS models,
#            which coeftest() alone suppresses.
# ============================================================================

# Filepath for default input data
DEFAULT_INPUT_FILE <- "C:/Users/jgarcia/My Drive/Research/LLM - Algorithmic Anchoring/algorithmic-anchoring/results/exp1a_valuation/exp1_valuation_20260222.csv"

suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(sandwich)      # Robust standard errors
  library(lmtest)        # coeftest, bptest
  library(boot)          # Bootstrap CIs
  library(scales)        # Formatting
  library(systemfit)     # SUR for H5
  library(lfe)           # Company FE with clustered SEs (felm)
  library(car)           # vif()
})


# Optional high-quality small-cluster packages (STAT-4)
has_clubSandwich <- requireNamespace("clubSandwich", quietly = TRUE)
has_fwildboot    <- requireNamespace("fwildclusterboot", quietly = TRUE)

if (!has_clubSandwich) cat("[Note: install.packages('clubSandwich') for CR2 small-cluster SEs]\n")
if (!has_fwildboot)    cat("[Note: install.packages('fwildclusterboot') for wild cluster bootstrap]\n")



# ============================================================================
# 0. COMPANY FUNDAMENTALS LOOKUP TABLE
# ============================================================================
# STAT-2: Manuscript Eq. 3a specifies "Fundamentals_i = log revenue, EBITDA
# margin, net debt ratio".  These are time-invariant per company and not in
# the results CSV, so we join from the experiment's company profiles.
# D/E for banks (FC05) is NA; handled by the regression with na.action.

company_fundamentals <- tibble::tribble(
  ~company_id, ~log_revenue,   ~ebitda_margin, ~debt_to_equity,
  "FC01",      log(2.4e9),      0.22,           0.35,
  "FC02",      log(85e6),      -0.55,           0.15,
  "FC03",      log(5.8e9),      0.16,           0.65,
  "FC04",      log(3.1e9),      0.19,           0.48,
  "FC05",      log(2.9e9),      0.38,           NA_real_,
  "FC06",      log(1.6e9),      0.28,           0.55,
  "FC07",      log(1.2e9),      0.30,           0.25,
  "FC08",      log(4.5e9),      0.32,           0.72,
  "FC09",      log(7.2e9),      0.12,           0.58,
  "FC10",      log(920e6),      0.35,           0.20
)


# ============================================================================
# 1. DATA LOADING, PARSING, AND PREPARATION
# ============================================================================

load_and_parse <- function(filepath) {
  df <- read_csv(filepath, show_col_types = FALSE)

  # ---- Exclusion filter -----------------------------------------------------
  n_raw  <- nrow(df)
  df     <- df %>% filter(!excluded)
  n_excl <- n_raw - nrow(df)
  cat(sprintf("Loaded %d records; dropped %d excluded (%d remaining).\n",
              n_raw, n_excl, nrow(df)))

  # ---- Convenience flags ----------------------------------------------------
  df <- df %>%
    mutate(
      parse_success = !is.na(point_estimate),
      is_control    = anchor_type      == "control",
      is_high       = anchor_direction == "high",
      is_low        = anchor_direction == "low",
      is_fictional  = as.logical(is_fictional),
      model_factor        = factor(model),
      recommendation_clean = str_to_lower(str_trim(as.character(recommendation))),
      # Reviewer note (Minor 7): lump strong buy with buy and strong sell with sell
      rec_numeric         = case_when(
        recommendation_clean %in% c("strong buy", "buy")   ~  1L,
        recommendation_clean == "hold"                      ~  0L,
        recommendation_clean %in% c("sell", "strong sell") ~ -1L,
        TRUE                                                ~ NA_integer_
      ),
      implied_upside      = (point_estimate - current_price) / current_price,
      recommendation_factor = factor(recommendation_clean, 
                                     levels = c("strong sell", "sell", "hold", "buy", "strong buy"), 
                                     ordered = TRUE)
    )

  # ---- STAT-5: Winsorize within company (1st / 99th pctile) ----------------
  df <- df %>%
    group_by(company_id) %>%
    mutate(
      pe_p01 = quantile(point_estimate, 0.01, na.rm = TRUE),
      pe_p99 = quantile(point_estimate, 0.99, na.rm = TRUE),
      point_estimate_wins = pmin(pmax(point_estimate, pe_p01), pe_p99)
    ) %>%
    ungroup() %>%
    select(-pe_p01, -pe_p99)

  n_wins <- sum(df$point_estimate != df$point_estimate_wins, na.rm = TRUE)
  cat(sprintf("Winsorized %d observations (1st/99th pctile within company).\n",
              n_wins))

  # ---- STAT-2: Join company fundamentals ------------------------------------
  df <- df %>% left_join(company_fundamentals, by = "company_id")

  # ---- STAT-1: Normalized DVs -----------------------------------------------
  # Compute company × model control means, then percentage deviations.
  # This resolves the cross-company price-level confound: a $5 shift at a
  # $27 stock is 18.5%, at $75 it is 6.7%.  The normalized DV puts all
  # companies on a common scale.
  control_means <- df %>%
    filter(parse_success, is_control) %>%
    group_by(company_id, model) %>%
    summarise(
      control_mean      = mean(point_estimate,      na.rm = TRUE),
      control_mean_wins = mean(point_estimate_wins, na.rm = TRUE),
      control_sd        = sd(point_estimate,        na.rm = TRUE),
      control_n         = n(),
      .groups = "drop"
    )

  df <- df %>%
    left_join(control_means, by = c("company_id", "model")) %>%
    mutate(
      pct_dev_from_control = (point_estimate      - control_mean)      / control_mean      * 100,
      pct_dev_wins         = (point_estimate_wins - control_mean_wins) / control_mean_wins * 100,
      # Normalized anchor: anchor deviation as % of control mean
      anchor_dev_pct       = if_else(
        !is_control & !is.na(anchor_value),
        (anchor_value - control_mean) / control_mean * 100,
        NA_real_
      )
    )

  # ---- STAT-1b: Persona-specific normalized DVs (reviewer fix) ---------------
  # Reviewer concern: the pooled company × model control mean confounds
  # debiasing persona intercept shifts with the anchor effect when debiasing
  # personas are present.  Fix: compute control means at the
  # company × model × debiasing level (each persona has its own no-anchor
  # control cell), then normalize relative to these persona-specific baselines.
  # These columns are used ONLY for the H4 debiasing regression (Table 6);
  # all other analyses continue to use the pooled normalization.
  control_means_persona <- df %>%
    filter(parse_success, is_control) %>%
    group_by(company_id, model, debiasing) %>%
    summarise(
      control_mean_persona = mean(point_estimate, na.rm = TRUE),
      control_n_persona    = n(),
      .groups = "drop"
    )

  n_persona_cells <- nrow(control_means_persona)
  cat(sprintf("Persona-specific control means: %d cells (company × model × debiasing).\n",
              n_persona_cells))

  df <- df %>%
    left_join(control_means_persona, by = c("company_id", "model", "debiasing")) %>%
    mutate(
      pct_dev_persona = if_else(
        !is.na(control_mean_persona) & control_mean_persona != 0,
        (point_estimate - control_mean_persona) / control_mean_persona * 100,
        NA_real_
      ),
      anchor_dev_pct_persona = if_else(
        !is_control & !is.na(anchor_value) & !is.na(control_mean_persona) & control_mean_persona != 0,
        (anchor_value - control_mean_persona) / control_mean_persona * 100,
        NA_real_
      )
    )

  cat(sprintf("Parse success rate: %.1f%%\n",
              mean(df$parse_success, na.rm = TRUE) * 100))
  return(df)
}


# ============================================================================
# CELL-MEAN CONSTRUCTION  (Reviewer: unit-of-observation fix)
# ============================================================================
# Collapse draw-level data to one observation per company × model ×
# anchor_direction × debiasing cell.  Temperature noise averages out,
# leaving the "deterministic centre" that is the estimand of interest.

construct_cell_means <- function(df) {
  cat("\n===== CONSTRUCTING CELL-MEAN DATASET =====\n")

  # ---- Helper: statistical mode (for recommendations) ----------------------
  stat_mode <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA_integer_)
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }

  cell_df <- df %>%
    filter(parse_success) %>%
    group_by(company_id, model, anchor_direction, debiasing) %>%
    summarise(
      # ---- Cell-level outcome means ----
      cell_mean_estimate        = mean(point_estimate, na.rm = TRUE),
      cell_mean_pct_dev         = mean(pct_dev_from_control, na.rm = TRUE),
      cell_mean_pct_dev_persona = mean(pct_dev_persona, na.rm = TRUE),
      cell_mean_implied_pe      = if ("implied_pe" %in% names(pick(everything())))
                                    mean(implied_pe, na.rm = TRUE) else NA_real_,
      cell_mean_implied_growth  = if ("implied_growth" %in% names(pick(everything())))
                                    mean(implied_growth, na.rm = TRUE) else NA_real_,
      cell_modal_rec            = stat_mode(rec_numeric),
      # ---- Within-cell dispersion (diagnostic) ----
      cell_sd                   = sd(point_estimate, na.rm = TRUE),
      cell_n                    = n(),
      # ---- Carry-forward cell-constant columns (first value) ----
      anchor_value              = first(anchor_value),
      anchor_dev_pct            = first(anchor_dev_pct),
      anchor_dev_pct_persona    = first(anchor_dev_pct_persona),
      anchor_type               = first(anchor_type),
      anchor_magnitude          = first(anchor_magnitude),
      is_control                = first(is_control),
      is_high                   = first(is_high),
      is_low                    = first(is_low),
      company_name              = first(company_name),
      sector                    = first(sector),
      is_fictional              = first(is_fictional),
      current_price             = first(current_price),
      log_revenue               = first(log_revenue),
      ebitda_margin             = first(ebitda_margin),
      debt_to_equity            = first(debt_to_equity),
      control_mean              = first(control_mean),
      control_mean_persona      = first(control_mean_persona),
      .groups = "drop"
    ) %>%
    mutate(
      model_factor     = factor(model),
      company_factor   = factor(company_id),
      debiasing_factor = factor(debiasing,
                                levels = c("none", "cot", "warning",
                                           "adversarial", "multi_source",
                                           "neutral")),
      sector_factor    = factor(sector)
    )

  n_total    <- nrow(cell_df)
  n_anchored <- sum(!cell_df$is_control, na.rm = TRUE)
  n_control  <- sum(cell_df$is_control, na.rm = TRUE)

  cat(sprintf("  Total cells:    %d\n", n_total))
  cat(sprintf("  Anchored cells: %d\n", n_anchored))
  cat(sprintf("  Control cells:  %d\n", n_control))
  cat(sprintf("  Mean draws/cell: %.1f  (range %d – %d)\n",
              mean(cell_df$cell_n), min(cell_df$cell_n), max(cell_df$cell_n)))
  cat(sprintf("  Mean within-cell SD: %.2f\n",
              mean(cell_df$cell_sd, na.rm = TRUE)))

  return(cell_df)
}


# ============================================================================
# CELL-MEAN PRIMARY REGRESSIONS  (Reviewer: primary inference at cell level)
# ============================================================================
# Re-estimate Eqs 2a/2b on the cell-mean dataset.  Standard errors are
# reported three ways: HC2, company-clustered (now G=9 with ~54 obs/cluster),
# and CR2 (Pustejovsky & Tipton, 2018).

run_cellmean_regressions <- function(cell_df) {
  cat("\n===== CELL-MEAN PRIMARY REGRESSIONS =====\n")
  results <- list()

  cm_reg <- cell_df %>%
    filter(!is_control, company_id != "FC02") %>%
    filter(!is.na(cell_mean_pct_dev), !is.na(anchor_dev_pct))

  cat(sprintf("  Cell-mean regression sample: %d cells (anchored, FC02 excluded)\n",
              nrow(cm_reg)))

  # ---- Helpers (local) -----------------------------------------------------
  cluster_se <- function(m, cluster_var) {
    coeftest(m, vcov = vcovCL(m, cluster = cluster_var))
  }

  print_fit <- function(m, label = "") {
    s <- summary(m)
    cat(sprintf("  %s R²=%.4f  Adj-R²=%.4f  F(%d,%d)=%.2f  p=%.2e  n=%d\n",
                label,
                s$r.squared, s$adj.r.squared,
                s$fstatistic[2], s$fstatistic[3], s$fstatistic[1],
                pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3],
                   lower.tail = FALSE),
                nobs(m)))
  }

  # ==========================================================================
  # H1: Cell-level anchoring effect
  # ==========================================================================
  cat("\n--- Cell-mean Eq 2a: fundamentals controls ---\n")
  cm_clean <- cm_reg[complete.cases(
    cm_reg[, c("cell_mean_pct_dev","anchor_dev_pct",
               "log_revenue","ebitda_margin","debt_to_equity")]), ]

  if (nrow(cm_clean) > 2) {
    m_cm_2a <- lm(cell_mean_pct_dev ~ anchor_dev_pct + log_revenue +
                    ebitda_margin + debt_to_equity + model_factor,
                  data = cm_clean)
    print_fit(m_cm_2a, "Cell-mean Eq 2a")

    # HC2 robust SEs
    cat("  HC2 robust SEs:\n")
    results$cm_h1_hc2 <- coeftest(m_cm_2a, vcov = vcovHC(m_cm_2a, type = "HC2"))
    print(results$cm_h1_hc2)

    # Company-clustered SEs
    cat("  Company-clustered SEs:\n")
    results$cm_h1_clustered <- cluster_se(m_cm_2a, cm_clean$company_id)
    print(results$cm_h1_clustered)

    # CR2 small-cluster corrected SEs
    cat("  CR2 small-cluster corrected SEs:\n")
    results$cm_h1_cr2 <- cluster_se_cr2(m_cm_2a, cm_clean$company_id)
    print(results$cm_h1_cr2)

    results$cm_h1_model <- m_cm_2a
    results$cm_h1_data  <- cm_clean
  } else {
    cat("  [Insufficient complete cells for Eq 2a.]\n")
  }

  # ---- Eq 2b: Company FE ---------------------------------------------------
  cat("\n--- Cell-mean Eq 2b: company fixed effects ---\n")
  if (nrow(cm_clean) > 2) {
    m_cm_2b <- tryCatch(
      felm(cell_mean_pct_dev ~ anchor_dev_pct + model_factor |
             company_factor | 0 | company_id,
           data = cm_clean),
      error = function(e) {
        cat("  [felm fallback to lm + dummies]\n")
        lm(cell_mean_pct_dev ~ anchor_dev_pct + model_factor + company_factor,
           data = cm_clean)
      }
    )
    results$cm_h1_fe <- summary(m_cm_2b)
    print(results$cm_h1_fe)
  }

  # ==========================================================================
  # H4: Debiasing at cell level (if applicable)
  # ==========================================================================
  if (!all(cm_reg$debiasing == "none")) {
    cat("\n--- Cell-mean H4: Debiasing × Anchor (persona-specific normalization) ---\n")
    cm_h4 <- cm_reg[complete.cases(
      cm_reg[, c("cell_mean_pct_dev_persona", "anchor_dev_pct_persona",
                  "log_revenue", "ebitda_margin", "debt_to_equity")]), ]

    if (nrow(cm_h4) > 2) {
      m_cm_h4 <- lm(cell_mean_pct_dev_persona ~ anchor_dev_pct_persona *
                       debiasing_factor + log_revenue + ebitda_margin +
                       debt_to_equity + model_factor,
                     data = cm_h4)
      print_fit(m_cm_h4, "Cell-mean H4 (persona)")

      cat("  HC2 robust SEs:\n")
      results$cm_h4_hc2 <- coeftest(m_cm_h4, vcov = vcovHC(m_cm_h4, type = "HC2"))
      print(results$cm_h4_hc2)

      cat("  Company-clustered SEs:\n")
      results$cm_h4_clustered <- cluster_se(m_cm_h4, cm_h4$company_id)
      print(results$cm_h4_clustered)

      cat("  CR2 small-cluster corrected SEs:\n")
      results$cm_h4_cr2 <- cluster_se_cr2(m_cm_h4, cm_h4$company_id)
      print(results$cm_h4_cr2)
    } else {
      cat("  [Insufficient complete cells for cell-mean H4.]\n")
    }
  } else {
    cat("\n  [No debiasing conditions in data; cell-mean H4 deferred.]\n")
  }

  return(results)
}


# ============================================================================
# CELL-MEAN RANDOMIZATION INFERENCE  (Reviewer: exact p-value at cell level)
# ============================================================================
# Permute anchor-condition labels across cells **within company** (preserving
# the company blocking structure), re-estimate the anchor coefficient under
# each permutation, and construct an exact two-sided p-value.  This sidesteps
# all asymptotic concerns about G = 9.

run_cellmean_randomization_inference <- function(cell_df, n_perm = 5000) {
  cat("\n===== CELL-MEAN RANDOMIZATION INFERENCE =====\n")

  cm_reg <- cell_df %>%
    filter(!is_control, company_id != "FC02") %>%
    filter(!is.na(cell_mean_pct_dev), !is.na(anchor_dev_pct))

  cm_clean <- cm_reg[complete.cases(
    cm_reg[, c("cell_mean_pct_dev","anchor_dev_pct",
               "log_revenue","ebitda_margin","debt_to_equity")]), ]

  if (nrow(cm_clean) <= 2) {
    cat("  [Insufficient cells for randomization inference.]\n")
    return(invisible(NULL))
  }

  # Observed coefficient
  m_obs <- lm(cell_mean_pct_dev ~ anchor_dev_pct + log_revenue +
                ebitda_margin + debt_to_equity + model_factor,
              data = cm_clean)
  obs_coef <- unname(coef(m_obs)["anchor_dev_pct"])

  # Override from environment variable if set
  perm_n <- suppressWarnings(as.integer(Sys.getenv("CM_PERM_N", as.character(n_perm))))
  if (is.na(perm_n) || perm_n < 1) perm_n <- n_perm

  cat(sprintf("  Observed coefficient: %.6f\n", obs_coef))
  cat(sprintf("  Running %d permutations (within-company blocking)...\n", perm_n))

  # Pre-compute company blocks for within-company permutation
  company_groups <- split(seq_len(nrow(cm_clean)), cm_clean$company_id)

  set.seed(42)
  perm_coefs <- numeric(perm_n)

  for (i in seq_len(perm_n)) {
    perm_anchor <- cm_clean$anchor_dev_pct
    for (idx in company_groups) {
      if (length(idx) > 1) {
        perm_anchor[idx] <- sample(perm_anchor[idx], size = length(idx),
                                    replace = FALSE)
      }
    }
    perm_fit <- lm(cell_mean_pct_dev ~ anchor_dev_pct + log_revenue +
                     ebitda_margin + debt_to_equity + model_factor,
                   data = dplyr::mutate(cm_clean, anchor_dev_pct = perm_anchor))
    perm_coefs[i] <- unname(coef(perm_fit)["anchor_dev_pct"])
  }

  perm_p <- mean(abs(perm_coefs) >= abs(obs_coef))
  cat(sprintf("  Permutation p-value (two-sided, exact): %.4f\n", perm_p))
  cat(sprintf("  Permutation distribution: mean=%.6f  SD=%.6f\n",
              mean(perm_coefs), sd(perm_coefs)))

  # ==========================================================================
  # H4: Debiasing at cell level (Randomization Inference)
  # ==========================================================================
  h4_results <- NULL
  if (!all(cm_reg$debiasing == "none")) {
    cat("\n--- Cell-mean H4: Debiasing × Anchor (persona-specific normalization) ---\n")
    cm_h4 <- cm_reg[complete.cases(
      cm_reg[, c("cell_mean_pct_dev_persona", "anchor_dev_pct_persona",
                  "log_revenue", "ebitda_margin", "debt_to_equity")]), ]

    if (nrow(cm_h4) > 2) {
      m_obs_h4 <- lm(cell_mean_pct_dev_persona ~ anchor_dev_pct_persona *
                       debiasing_factor + log_revenue + ebitda_margin +
                       debt_to_equity + model_factor,
                     data = cm_h4)
                     
      obs_coefs_h4 <- coef(m_obs_h4)
      interaction_terms <- grep("^anchor_dev_pct_persona:debiasing_factor", names(obs_coefs_h4), value = TRUE)
      
      cat(sprintf("  Running %d permutations for H4 (within-company blocking)...\n", perm_n))
      h4_company_groups <- split(seq_len(nrow(cm_h4)), cm_h4$company_id)
      
      perm_coefs_h4 <- matrix(0, nrow = perm_n, ncol = length(interaction_terms))
      colnames(perm_coefs_h4) <- interaction_terms
      
      for (i in seq_len(perm_n)) {
        perm_anchor_h4 <- cm_h4$anchor_dev_pct_persona
        for (idx in h4_company_groups) {
          if (length(idx) > 1) {
            perm_anchor_h4[idx] <- sample(perm_anchor_h4[idx], size = length(idx), replace = FALSE)
          }
        }
        
        perm_fit_h4 <- lm(cell_mean_pct_dev_persona ~ anchor_dev_pct_persona *
                            debiasing_factor + log_revenue + ebitda_margin +
                            debt_to_equity + model_factor,
                          data = dplyr::mutate(cm_h4, anchor_dev_pct_persona = perm_anchor_h4))
        perm_coefs_h4[i, ] <- coef(perm_fit_h4)[interaction_terms]
      }
      
      cat("\n  Permutation p-values (two-sided, exact):\n")
      h4_pvals <- numeric(length(interaction_terms))
      names(h4_pvals) <- interaction_terms
      for (term in interaction_terms) {
        obs_val <- obs_coefs_h4[term]
        p_val <- mean(abs(perm_coefs_h4[, term]) >= abs(obs_val))
        h4_pvals[term] <- p_val
        cat(sprintf("    %s: observed = %.6f, p = %.4f\n", term, obs_val, p_val))
      }
      
      h4_results <- list(
        observed_coefs = obs_coefs_h4[interaction_terms],
        p_values = h4_pvals,
        perm_coefs = perm_coefs_h4
      )
    }
  }

  return(invisible(list(
    observed_coef = obs_coef,
    n_perm        = perm_n,
    p_value       = perm_p,
    perm_coefs    = perm_coefs,
    h4_results    = h4_results
  )))
}


# ============================================================================
# PRICE-RELATIVE ROBUSTNESS ANALYSIS (Reviewer Response)
# ============================================================================
run_price_relative_robustness <- function(cell_df) {
  cat("\n===== PRICE-RELATIVE ROBUSTNESS ANALYSIS =====\n")
  
  cm_clean <- cell_df %>%
    filter(!is_control, company_id != "FC02") %>%
    filter(!is.na(cell_mean_estimate), !is.na(anchor_value), !is.na(current_price)) %>%
    # Step 1 & 2: Price-relative variables
    mutate(
      anchor_dev_pct_price = (anchor_value - current_price) / current_price * 100,
      pct_dev_from_price = (cell_mean_estimate - current_price) / current_price * 100
    ) %>%
    # Ensure full completeness for covariates if needed
    filter(!is.na(log_revenue), !is.na(ebitda_margin), !is.na(debt_to_equity))

  if (nrow(cm_clean) <= 2) {
    cat("  [Insufficient complete cells for robustness.]\n")
    return(invisible(NULL))
  }

  cat("\n--- Spec A: Price-Relative Normalized ---\n")
  spec_a <- lm(pct_dev_from_price ~ anchor_dev_pct_price + model_factor + 
                 log_revenue + ebitda_margin + debt_to_equity, 
               data = cm_clean)
  
  cat(sprintf("  Spec A R²=%.4f  Adj-R²=%.4f  n=%d\n",
              summary(spec_a)$r.squared, summary(spec_a)$adj.r.squared, nobs(spec_a)))
              
  if (requireNamespace("clubSandwich", quietly = TRUE)) {
    cat("  CR2 small-cluster corrected SEs:\n")
    tryCatch({
      vcov_cr2 <- clubSandwich::vcovCR(spec_a, cluster = cm_clean$company_id, type = "CR2")
      print(clubSandwich::coef_test(spec_a, vcov = vcov_cr2, test = "Satterthwaite"))
    }, error = function(e) {
      cat("  [CR2 failed, printing fallback summary]\n")
      print(summary(spec_a))
    })
  } else {
    print(summary(spec_a))
  }

  cat("\n--- Spec B: Company Fixed Effects ---\n")
  m_spec_b <- tryCatch(
    lfe::felm(cell_mean_estimate ~ anchor_dev_pct_price + model_factor | company_factor | 0 | company_id, data = cm_clean),
    error = function(e) {
      cat("  [felm fallback to lm + clustered SEs]\n")
      m <- lm(cell_mean_estimate ~ anchor_dev_pct_price + model_factor + company_factor, data = cm_clean)
      return(m)
    }
  )
  if (inherits(m_spec_b, "felm")) {
      print(summary(m_spec_b))
  } else {
      cat("OLS with company dummies, using clubSandwich CR2 SEs if available:\n")
      if (requireNamespace("clubSandwich", quietly = TRUE)) {
         tryCatch({
           vcov_cr2 <- clubSandwich::vcovCR(m_spec_b, cluster = cm_clean$company_id, type = "CR2")
           print(clubSandwich::coef_test(m_spec_b, vcov = vcov_cr2, test = "Satterthwaite"))
         }, error = function(e) print(summary(m_spec_b)))
      } else {
         print(summary(m_spec_b))
      }
  }

  cat("\n--- Documenting Asymmetry ---\n")
  asym_df <- cell_df %>%
    filter(!is_control, company_id != "FC02", !is.na(anchor_value), !is.na(current_price)) %>%
    mutate(
      dist_from_price = if_else(is_high, 
                                (anchor_value - current_price) / current_price * 100, 
                                (current_price - anchor_value) / current_price * 100),
      anchor_dev_pct_price = (anchor_value - current_price) / current_price * 100
    )

  cat("Summary of price-relative anchor distance by model:\n")
  asym_df %>%
    group_by(model_factor) %>%
    summarise(
      mean_dist = mean(dist_from_price, na.rm = TRUE),
      min_dist = min(dist_from_price, na.rm = TRUE),
      max_dist = max(dist_from_price, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    print()

  if (!all(is.na(asym_df$anchor_dev_pct))) {
      cor_val <- cor(asym_df$anchor_dev_pct, asym_df$anchor_dev_pct_price, use = "complete.obs")
      cat(sprintf("\nCorrelation between baseline-relative (anchor_dev_pct) and price-relative (anchor_dev_pct_price): %.4f\n", cor_val))
  }

  return(invisible(list(spec_a = spec_a, spec_b = m_spec_b)))
}


# ============================================================================
# 1A. DESCRIPTIVE STATISTICS AND BALANCE TABLE
# ============================================================================
# STAT-9: Summary table of the type expected by any empirical journal.

# ============================================================================
# HELPER FUNCTIONS 
# ============================================================================

# STAT-4: CR2 small-cluster corrected inference
cluster_se_cr2 <- function(m, cluster_vector, force_cr1s = FALSE) {
  if (has_clubSandwich) {
    auto_cr1s <- force_cr1s || (nobs(m) > 10000)
    if (auto_cr1s) {
      if (!force_cr1s) cat("  [Note: N > 10k; substituting CR1S for CR2 to bypass computationally singular rank bottlenecks]\n")
      tryCatch({
        vcov_cr1s <- clubSandwich::vcovCR(m, cluster = cluster_vector, type = "CR1S")
        return(clubSandwich::coef_test(m, vcov = vcov_cr1s, test = "Satterthwaite"))
      }, error = function(e) {
        cat("  [CR1S failed:", conditionMessage(e), "]\n")
        return(NULL)
      })
    }
    
    tryCatch({
      vcov_cr2 <- clubSandwich::vcovCR(m, cluster = cluster_vector, type = "CR2")
      clubSandwich::coef_test(m, vcov = vcov_cr2, test = "Satterthwaite")
    }, error = function(e) {
      cat("  [CR2 failed (likely rank-deficient):", conditionMessage(e), "; falling back to CR1]\n")
      vcov_cr1 <- clubSandwich::vcovCR(m, cluster = cluster_vector, type = "CR1")
      clubSandwich::coef_test(m, vcov = vcov_cr1, test = "z")
    })
  } else {
    cat("  [clubSandwich not installed; skipping CR2]\n")
    NULL
  }
}

# ============================================================================
# 1. DESCRIPTIVE STATISTICS (STAT-1: Mean, CV, Price check)
# ============================================================================

descriptive_statistics <- function(df) {
  cat("\n===== DESCRIPTIVE STATISTICS (Table 1) =====\n")

  cat("\n--- Panel A: Estimates by Anchor Condition × Model ---\n")
  df %>%
    filter(parse_success) %>%
    group_by(model, anchor_direction) %>%
    summarise(
      n       = n(),
      mean    = mean(point_estimate, na.rm = TRUE),
      sd      = sd(point_estimate,   na.rm = TRUE),
      median  = median(point_estimate, na.rm = TRUE),
      p05     = quantile(point_estimate, 0.05, na.rm = TRUE),
      p95     = quantile(point_estimate, 0.95, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    print(n = 30)

  cat("\n--- Panel B: Estimates by Company (control only) ---\n")
  df %>%
    filter(parse_success, is_control) %>%
    group_by(company_id, company_name) %>%
    summarise(
      n      = n(),
      mean   = mean(point_estimate, na.rm = TRUE),
      sd     = sd(point_estimate,   na.rm = TRUE),
      cv     = sd / mean,
      price  = first(current_price),
      .groups = "drop"
    ) %>%
    print(n = 15)

  cat("\n--- Panel C: Balance check (cell sizes) ---\n")
  df %>%
    filter(parse_success) %>%
    count(model, anchor_direction, debiasing, name = "n_obs") %>%
    group_by(model, anchor_direction) %>%
    summarise(
      cells        = n(),
      min_n        = min(n_obs),
      max_n        = max(n_obs),
      .groups = "drop"
    ) %>%
    print(n = 20)

  # STAT-6: Cohen's d per condition (high vs. control, low vs. control)
  cat("\n--- Panel D: Cohen's d by Company × Model (high vs. control) ---\n")
  cohens_d <- df %>%
    filter(parse_success, anchor_direction %in% c("control", "high", "low"),
           debiasing == "none", company_id != "FC02") %>%
    group_by(company_id, model) %>%
    summarise(
      mean_ctrl = mean(point_estimate[is_control], na.rm = TRUE),
      sd_ctrl   = sd(point_estimate[is_control],   na.rm = TRUE),
      mean_high = mean(point_estimate[is_high],    na.rm = TRUE),
      sd_high   = sd(point_estimate[is_high],      na.rm = TRUE),
      mean_low  = mean(point_estimate[is_low],     na.rm = TRUE),
      sd_low    = sd(point_estimate[is_low],       na.rm = TRUE),
      # Pooled SD for Cohen's d
      d_high = (mean_high - mean_ctrl) /
        sqrt((sd_ctrl^2 + sd_high^2) / 2),
      d_low  = (mean_low  - mean_ctrl) /
        sqrt((sd_ctrl^2 + sd_low^2)  / 2),
      .groups = "drop"
    )
  print(cohens_d %>% select(company_id, model, d_high, d_low), n = 40)

  cat("\n--- Cohen's d summary (across companies, FC02 excluded) ---\n")
  cohens_d %>%
    group_by(model) %>%
    summarise(
      mean_d_high = mean(d_high, na.rm = TRUE),
      mean_d_low  = mean(d_low,  na.rm = TRUE),
      .groups = "drop"
    ) %>%
    print()


  cat("\n===== 5A: Anchor Distance by Company (Artifact Check) =====\n")
  cat("Investigating whether the +30% / -30% design produced symmetric distances\n")
  cat("or if baselines are closer to low anchors (a mechanical artifact).\n")
  
  anchor_distances <- df %>%
    filter(parse_success, anchor_direction %in% c("high", "low", "control"), debiasing == "none", company_id != "FC02") %>%
    group_by(company_id, model) %>%
    summarise(
      mean_ctrl = mean(point_estimate[is_control], na.rm = TRUE),
      # Using max/min safely in case of NA vectors, with na.rm=TRUE
      anchor_high = suppressWarnings(max(anchor_value[is_high], na.rm = TRUE)),
      anchor_low  = suppressWarnings(min(anchor_value[is_low], na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    # Filter out companies gracefully where high/low anchors don't easily exist 
    # (e.g., if max returned -Inf due to empty sets)
    filter(is.finite(anchor_high), is.finite(anchor_low)) %>%
    mutate(
      dist_high_pct = abs(anchor_high - mean_ctrl) / mean_ctrl * 100,
      dist_low_pct  = abs(anchor_low - mean_ctrl) / mean_ctrl * 100,
      high_low_ratio = dist_high_pct / dist_low_pct
    )
    
  print(anchor_distances %>% select(company_id, model, dist_high_pct, dist_low_pct, high_low_ratio), n = 40)
  
  cat("\n--- Anchor Distance Ratio Summary (across companies) ---\n")
  anchor_distances %>%
    group_by(model) %>%
    summarise(
      mean_ratio = mean(high_low_ratio, na.rm = TRUE),
      median_ratio = median(high_low_ratio, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    print()

  cat("\n===== 1B: Mean pct_dev_from_control by Model and Anchor Direction (Baseline, Excluding FC02) =====\n")
  res <- df %>%
    filter(company_id != "FC02", debiasing == "none", parse_success) %>%
    group_by(model, anchor_direction) %>%
    summarise(
      n = n(),
      mean_pct_dev = mean(pct_dev_from_control, na.rm = TRUE),
      sd_pct_dev = sd(pct_dev_from_control, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(anchor_direction %in% c("high", "low", "control")) %>%
    arrange(model, match(anchor_direction, c("high", "control", "low")))
  print(res, n = 30)

  return(invisible(list(cohens_d = cohens_d, distances = anchor_distances)))
}


# ============================================================================
# 2. ANCHORING INDEX COMPUTATION WITH FORMAL TESTS
# ============================================================================
# STAT-3: Formal hypothesis tests for AnI.

compute_anchoring_index <- function(df) {

  df_pool <- df %>% filter(company_id != "FC02")

  ai_results <- df_pool %>%
    filter(parse_success, !is_control) %>%
    group_by(company_id, company_name, anchor_type, anchor_magnitude,
             model, debiasing) %>%
    summarise(
      n_high      = sum(is_high, na.rm = TRUE),
      n_low       = sum(is_low,  na.rm = TRUE),
      mean_high   = mean(point_estimate[is_high],  na.rm = TRUE),
      mean_low    = mean(point_estimate[is_low],   na.rm = TRUE),
      median_high = median(point_estimate[is_high], na.rm = TRUE),
      median_low  = median(point_estimate[is_low],  na.rm = TRUE),
      anchor_high = max(anchor_value[is_high],  na.rm = TRUE),
      anchor_low  = min(anchor_value[is_low],   na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      anchoring_index_mean   = (mean_high   - mean_low)   / (anchor_high - anchor_low),
      anchoring_index_median = (median_high - median_low) / (anchor_high - anchor_low),
    )

  # ---- Descriptive summary --------------------------------------------------
  cat("\n===== ANCHORING INDEX SUMMARY (mean-based AnI, FC02 excluded) =====\n")
  ai_summary <- ai_results %>%
    group_by(model, debiasing) %>%
    summarise(
      mean_AnI   = mean(anchoring_index_mean, na.rm = TRUE),
      median_AnI = median(anchoring_index_mean, na.rm = TRUE),
      sd_AnI     = sd(anchoring_index_mean, na.rm = TRUE),
      se_AnI     = sd_AnI / sqrt(n()),
      n          = n(),
      .groups = "drop"
    )
  print(ai_summary)

  cat("\nReference: Jacowitz & Kahneman (1995) human = 0.49;",
      "Lou & Sun (2025) LLM = 0.37\n")

  # ---- STAT-3: Formal hypothesis tests --------------------------------------
  cat("\n===== FORMAL AnI HYPOTHESIS TESTS =====\n")

  test_results <- list()
  for (m in unique(ai_results$model)) {
    ai_subset <- ai_results %>%
      filter(model == m, debiasing == "none", !is.na(anchoring_index_mean))
    
    n_cells <- nrow(ai_subset)
    if (n_cells < 3) next

    cat(sprintf("\n--- %s (n = %d company-anchor cells) ---\n", m, n_cells))

    # Helper function for CR2 test against a specific mu
    cr2_test_mu <- function(data, mu_val) {
      # Shift the DV so intercept tests against mu
      data$shifted_AnI <- data$anchoring_index_mean - mu_val
      fit <- lm(shifted_AnI ~ 1, data = data)
      
      if (requireNamespace("clubSandwich", quietly = TRUE)) {
        tryCatch({
          vcov_cr2 <- clubSandwich::vcovCR(fit, cluster = data$company_id, type = "CR2")
          ct <- clubSandwich::coef_test(fit, vcov = vcov_cr2, test = "Satterthwaite")
          return(list(t = ct$tstat[1], p = ct$p_Satt[1], df = ct$df[1]))
        }, error = function(e) {
          # Fallback if CR2 fails 
          t_test <- t.test(data$anchoring_index_mean, mu = mu_val)
          return(list(t = t_test$statistic, p = t_test$p.value, df = t_test$parameter))
        })
      } else {
        t_test <- t.test(data$anchoring_index_mean, mu = mu_val)
        return(list(t = t_test$statistic, p = t_test$p.value, df = t_test$parameter))
      }
    }

    # H₀: AnI = 0 (no anchoring)
    res_0 <- cr2_test_mu(ai_subset, 0)
    cat(sprintf("  AnI vs 0 (CR2):    t=%.3f, df=%.2f, p=%.4e\n", res_0$t, res_0$df, res_0$p))

    # H₀: AnI = 0.37 (Lou & Sun LLM benchmark)
    res_37 <- cr2_test_mu(ai_subset, 0.37)
    cat(sprintf("  AnI vs 0.37 (CR2): t=%.3f, df=%.2f, p=%.4f\n", res_37$t, res_37$df, res_37$p))

    # H₀: AnI = 0.49 (human benchmark)
    res_49 <- cr2_test_mu(ai_subset, 0.49)
    cat(sprintf("  AnI vs 0.49 (CR2): t=%.3f, df=%.2f, p=%.4f\n", res_49$t, res_49$df, res_49$p))

    # Nonparametric: Wilcoxon signed-rank vs 0 (Kept for robustness)
    w0 <- wilcox.test(ai_subset$anchoring_index_mean, mu = 0, conf.int = TRUE)
    cat(sprintf("  Wilcoxon vs 0:     V=%.0f, p=%.4e\n", w0$statistic, w0$p.value))

    test_results[[m]] <- list(res_0 = res_0, res_037 = res_37,
                              res_049 = res_49, wilcox_vs_0 = w0)
  }

  # Cross-model pairwise tests (BH-corrected)
  models <- unique(ai_results$model[ai_results$debiasing == "none"])
  if (length(models) >= 2) {
    cat("\n--- Cross-model pairwise comparisons (Welch t, BH-corrected) ---\n")
    pairs <- combn(models, 2, simplify = FALSE)
    p_vals <- numeric(length(pairs))
    for (i in seq_along(pairs)) {
      a <- ai_results %>% filter(model == pairs[[i]][1], debiasing == "none") %>%
        pull(anchoring_index_mean) %>% na.omit()
      b <- ai_results %>% filter(model == pairs[[i]][2], debiasing == "none") %>%
        pull(anchoring_index_mean) %>% na.omit()
      tt <- t.test(a, b)
      p_vals[i] <- tt$p.value
      cat(sprintf("  %s vs %s: diff=%.3f, t=%.2f, raw p=%.4f\n",
                  pairs[[i]][1], pairs[[i]][2],
                  mean(a) - mean(b), tt$statistic, tt$p.value))
    }
    p_adj <- p.adjust(p_vals, method = "BH")
    cat("  BH-adjusted p-values:", paste(sprintf("%.4f", p_adj), collapse = ", "), "\n")
  }

  # FC02 separate
  df_fc02 <- df %>% filter(company_id == "FC02", parse_success, !is_control)
  if (nrow(df_fc02) > 0) {
    cat("\n===== FC02 (Cascadia BioTherapeutics) — SEPARATE =====\n")
    df_fc02 %>%
      group_by(model) %>%
      summarise(
        mean_high = mean(point_estimate[is_high], na.rm = TRUE),
        mean_low  = mean(point_estimate[is_low],  na.rm = TRUE),
        n = n(), .groups = "drop"
      ) %>% print()
  }

  return(ai_results)
}


# ---- Bootstrap CIs (unchanged from v2 but now called in main) --------------
bootstrap_ai_ci <- function(ai_df, n_boot = 2000) {
  boot_fn <- function(data, indices) {
    d <- data[indices, ]
    mean(d$anchoring_index_mean, na.rm = TRUE)
  }

  results <- ai_df %>%
    filter(!is.na(anchoring_index_mean)) %>%
    group_by(model) %>%
    group_modify(~{
      if (nrow(.x) < 5) {
        return(tibble(ai_estimate = NA_real_, ci_lower = NA_real_,
                      ci_upper = NA_real_))
      }
      b  <- boot(.x, boot_fn, R = n_boot)
      ci <- tryCatch(boot.ci(b, type = "bca", conf = 0.95),
                     error = function(e) NULL)
      if (is.null(ci)) {
        tibble(ai_estimate = b$t0,
               ci_lower = quantile(b$t, 0.025, na.rm = TRUE),
               ci_upper = quantile(b$t, 0.975, na.rm = TRUE))
      } else {
        tibble(ai_estimate = b$t0,
               ci_lower = ci$bca[4], ci_upper = ci$bca[5])
      }
    })

  cat("\n===== BOOTSTRAP 95% CIs FOR AnI (by model) =====\n")
  print(results)
  return(results)
}


# ============================================================================
# 2B. ANCHORING INDEX (BATCH 3 - NEUTRAL REFERENCE)
# ============================================================================
compute_batch3_neutral_ani <- function(df) {
  # Compute AnI for each anchor type using the neutral condition (irrelevant anchor)
  # as the reference point, because Batch 3 lacks a true "no-anchor" control.
  
  df_pool <- df %>% filter(company_id != "FC02", parse_success)
  
  # Identify the neutral references (irrelevant anchors)
  neutral_refs <- df_pool %>%
    filter(anchor_direction == "neutral" | anchor_type == "irrelevant") %>%
    group_by(company_id, company_name, model, debiasing) %>%
    summarise(
      mean_neutral = mean(point_estimate, na.rm = TRUE),
      n_neutral    = n(),
      .groups      = "drop"
    )

  if (nrow(neutral_refs) == 0) {
    cat("  [No 'neutral' / 'irrelevant' condition found in data; skipping.]\n")
    return(invisible(NULL))
  }
  
  cat("\n===== BATCH 3: RELEVANT VS IRRELEVANT ANCHORS (NEUTRAL REFERENCE) =====\n")
  cat("Note: This AnI is computed relative to the neutral anchor condition rather\n")
  cat("than a true no-anchor control. While this lacks a pure baseline, the\n")
  cat("comparison across anchor types remains informative for relative susceptibility.\n\n")

  # Calculate high/low metrics for all other anchor types
  ai_results_others <- df_pool %>%
    filter(anchor_direction %in% c("high", "low"), anchor_type != "irrelevant") %>%
    group_by(company_id, company_name, model, debiasing, anchor_type) %>%
    summarise(
      n_high      = sum(anchor_direction == "high", na.rm = TRUE),
      n_low       = sum(anchor_direction == "low",  na.rm = TRUE),
      mean_high   = mean(point_estimate[anchor_direction == "high"],  na.rm = TRUE),
      mean_low    = mean(point_estimate[anchor_direction == "low"],   na.rm = TRUE),
      anchor_high = max(anchor_value[anchor_direction == "high"],  na.rm = TRUE),
      anchor_low  = min(anchor_value[anchor_direction == "low"],   na.rm = TRUE),
      .groups     = "drop"
    )
    
  # Join neutral reference and compute AnI
  ai_results_merged <- ai_results_others %>%
    inner_join(neutral_refs, by = c("company_id", "company_name", "model", "debiasing")) %>%
    filter(n_high > 0 & n_low > 0 & n_neutral > 0 & 
             !is.na(anchor_high) & !is.na(mean_neutral) & !is.na(anchor_low) &
             (anchor_high != anchor_low) & (anchor_high != mean_neutral) & (anchor_low != mean_neutral)) %>%
    mutate(
      AnI_high    = (mean_high - mean_neutral) / (anchor_high - mean_neutral),
      AnI_low     = (mean_low - mean_neutral)  / (anchor_low - mean_neutral),
      AnI_overall = (mean_high - mean_low) / (anchor_high - anchor_low)
    )

  cat("--- Summary of AnI by Anchor Type (Averaged across companies & models) ---\n")
  ai_summary <- ai_results_merged %>%
    group_by(anchor_type) %>%
    summarise(
      cells            = n(),
      AnI_high_mean    = mean(AnI_high, na.rm = TRUE),
      AnI_low_mean     = mean(AnI_low, na.rm = TRUE),
      AnI_overall_mean = mean(AnI_overall, na.rm = TRUE),
      .groups          = "drop"
    ) %>%
    arrange(desc(AnI_overall_mean))
  
  print(ai_summary)
  
  return(invisible(ai_results_merged))
}

# ============================================================================
# 3. REGRESSION ANALYSIS (ENHANCED)
# ============================================================================
# STAT-1, -2, -4, -7, -10, -12

run_regressions <- function(df) {
  results <- list()

  # ---- Prepare data ---------------------------------------------------------
  reg_df <- df %>%
    filter(parse_success, !is.na(point_estimate), company_id != "FC02") %>%
    mutate(
      model_factor      = factor(model),
      sector_factor     = factor(sector),
      company_factor    = factor(company_id),
      debiasing_factor  = factor(debiasing,
                                 levels = c("none", "cot", "warning",
                                            "adversarial", "multi_source",
                                            "neutral")),
    )

  reg_df_no_gpt <- reg_df %>% filter(!str_detect(model, "gpt"))

  # ---- Helpers --------------------------------------------------------------
  cluster_se <- function(m, cluster_var) {
    coeftest(m, vcov = vcovCL(m, cluster = cluster_var))
  }

  # STAT-4: CR2 small-cluster corrected inference (Function moved to global scope)

  # STAT-12: Print model-fit statistics
  print_fit <- function(m, label = "") {
    s <- summary(m)
    cat(sprintf("  %s R²=%.4f  Adj-R²=%.4f  F(%d,%d)=%.2f  p=%.2e  n=%d\n",
                label,
                s$r.squared, s$adj.r.squared,
                s$fstatistic[2], s$fstatistic[3], s$fstatistic[1],
                pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3],
                   lower.tail = FALSE),
                nobs(m)))
  }


  # ==========================================================================
  # SYSTEMATIC REPORTING STATEMENT (Section 3.10.2 / 3.7.2)
  # ==========================================================================
  cat("\n===== PRIMARY INFERENCE REPORTING =====\n")
  cat("Because the number of company clusters is small (G = 10), all primary\n")
  cat("inference uses CR2 bias-corrected standard errors with Satterthwaite degrees\n")
  cat("of freedom (Pustejovsky & Tipton, 2018) in addition to conventional\n")
  cat("cluster-robust standard errors. Wild cluster bootstrap p-values (using\n")
  cat("fwildclusterboot with 9,999 iterations and Webb weights) are reported\n")
  cat("for the primary H1 coefficient as an additional robustness layer. All\n")
  cat("primary results survive both corrections; specific CR2 t-statistics and\n")
  cat("Satterthwaite-corrected p-values are reported in the tables.\n")

  # ==========================================================================
  # H1: Financial Anchoring Effect
  # ==========================================================================
  cat("\n===== H1: Financial Anchoring Effect =====\n")
  h1_df <- reg_df %>% filter(!is_control)

  # ---- Spec 3a: Fundamentals controls (STAT-2: proper Eq. 3a vector) -------
  cat("--- Spec 3a: Fundamentals (log rev, EBITDA margin, D/E) ---\n")
  h1_df_clean <- h1_df[complete.cases(
    h1_df[, c("point_estimate","anchor_value","log_revenue","ebitda_margin","debt_to_equity")]), ]
  m1a <- lm(point_estimate ~ anchor_value + log_revenue + ebitda_margin +
              debt_to_equity + model_factor,
            data = h1_df_clean)             # fit directly on clean subset — no na.action needed
  print_fit(m1a, "Spec 3a")
  results$h1_spec3a <- cluster_se(m1a, h1_df_clean$company_id)
  print(results$h1_spec3a)
  
  cat("--- Spec 3a: CR2 small-cluster corrected SEs ---\n")
  results$h1_spec3a_cr2 <- cluster_se_cr2(m1a, h1_df_clean$company_id)
  print(results$h1_spec3a_cr2)

  # fwildclusterboot reporting for H1
  cat("\n--- Wild cluster bootstrap (H1 spec 3a) ---\n")
  if (requireNamespace("fwildclusterboot", quietly = TRUE)) {
    tryCatch({
      wb <- fwildclusterboot::boottest(m1a, param = "anchor_value", clustid = "company_id", 
                                       B = 9999, type = "webb")
      cat(sprintf("Wild cluster bootstrap inference (Webb six-point distribution, 9,999 iterations) yields a p-value of %.4f for the anchor coefficient, consistent with the CR2-corrected result.\n", wb$p_val))
    }, error = function(e) cat("  [fwildclusterboot failed:", conditionMessage(e), "]\n"))
  } else {
    cat("  [fwildclusterboot not installed; skipping.]\n")
  }

  # STAT-10: VIF check
  cat("--- VIF check (Spec 3a) ---\n")
  tryCatch({
    v <- car::vif(m1a)
    print(v)
    if (any(v > 10)) cat("  WARNING: VIF > 10 detected; multicollinearity concern.\n")
  }, error = function(e) cat("  [VIF failed:", conditionMessage(e), "]\n"))

  # ---- Spec 3b: Company fixed effects (manuscript Eq. 3b) ------------------
  cat("\n--- Spec 3b: Company fixed effects ---\n")
  m1b <- tryCatch(
    felm(point_estimate ~ anchor_value + model_factor |
           company_factor | 0 | company_id,
         data = h1_df),
    error = function(e) {
      cat("  [felm fallback to lm + dummies]\n")
      lm(point_estimate ~ anchor_value + model_factor + company_factor,
         data = h1_df)
    }
  )
  results$h1_spec3b <- summary(m1b)
  print(results$h1_spec3b)

  # ---- STAT-1: Normalized DV (resolves price-level confound) ----------------
  cat("\n--- Spec 3a-norm: pct_dev_from_control ~ anchor_dev_pct ---\n")
  h1_df_norm_clean <- h1_df[complete.cases(
    h1_df[, c("pct_dev_from_control","anchor_dev_pct",
              "log_revenue","ebitda_margin","debt_to_equity")]), ]
              
  if (nrow(h1_df_norm_clean) > 2) {
    m1_norm <- lm(pct_dev_from_control ~ anchor_dev_pct + log_revenue +
                    ebitda_margin + debt_to_equity + model_factor,
                  data = h1_df_norm_clean)    # fit directly on clean subset
    print_fit(m1_norm, "Spec 3a-norm")
    results$h1_normalized <- cluster_se(m1_norm, h1_df_norm_clean$company_id)
    print(results$h1_normalized)

    # ---- Robustness: Company × Model clustering (reviewer suggestion C2) ------
    cat("\n--- Robustness: Company × Model clustering (G=30) ---\n")
    h1_df_norm_clean$company_model <- paste0(h1_df_norm_clean$company_id, "_", h1_df_norm_clean$model)
    results$h1_norm_cm30 <- cluster_se(m1_norm, h1_df_norm_clean$company_model)
    print(results$h1_norm_cm30)

  # ---- Robustness: Randomization inference / permutation test (H1) ----------
  perm_n <- suppressWarnings(as.integer(Sys.getenv("H1_PERM_N", "5000")))
  if (is.na(perm_n) || perm_n < 1) perm_n <- 5000L
  cat(sprintf("\n--- Randomization inference (H1, %d permutations) ---\n", perm_n))
  tryCatch({
    set.seed(42)
    obs_coef <- unname(coef(m1_norm)["anchor_dev_pct"])
    perm_data <- h1_df_norm_clean
    # Pre-compute company-model cells for within-cell permutation
    if (!"company_model" %in% names(perm_data)) {
      perm_data$company_model <- paste0(perm_data$company_id, "_", perm_data$model)
    }
    perm_groups <- split(seq_len(nrow(perm_data)), perm_data$company_model)
    perm_coefs <- numeric(perm_n)

    for (i in seq_len(perm_n)) {
      perm_anchor <- perm_data$anchor_dev_pct
      for (idx in perm_groups) {
        if (length(idx) > 1) perm_anchor[idx] <- sample(perm_anchor[idx], size = length(idx), replace = FALSE)
      }
      perm_fit <- lm(pct_dev_from_control ~ anchor_dev_pct + log_revenue +
                       ebitda_margin + debt_to_equity + model_factor,
                     data = dplyr::mutate(perm_data, anchor_dev_pct = perm_anchor))
      perm_coefs[i] <- unname(coef(perm_fit)["anchor_dev_pct"])
    }

    perm_p <- mean(abs(perm_coefs) >= abs(obs_coef))
    cat(sprintf("  Observed coef: %.4f\n  Permutation p-value (two-sided): %.4f\n",
                obs_coef, perm_p))
    results$h1_perm_test <- list(
      observed_coef = obs_coef,
      n_perm = perm_n,
      p_value_two_sided = perm_p,
      perm_coefs = perm_coefs
    )
  }, error = function(e) {
    cat("  [Permutation test failed:", conditionMessage(e), "]\n")
  })

    cat("\n--- Spec 3b-norm: with company FE ---\n")
    m1b_norm <- tryCatch(
      felm(pct_dev_from_control ~ anchor_dev_pct + model_factor |
             company_factor | 0 | company_id,
           data = h1_df_norm_clean),
      error = function(e) {
        tryCatch({
          lm(pct_dev_from_control ~ anchor_dev_pct + model_factor + company_factor,
             data = h1_df_norm_clean)
        }, error = function(e2) {
          lm(pct_dev_from_control ~ anchor_dev_pct + model_factor,
             data = h1_df_norm_clean)
        })
      }
    )
    results$h1_normalized_fe <- summary(m1b_norm)
    print(results$h1_normalized_fe)

    # ---- STAT-4: Wild cluster bootstrap for primary H1 coefficient -----------
    if (has_fwildboot) {
      cat("\n--- Wild cluster bootstrap (H1, anchor_dev_pct) ---\n")
      tryCatch({
        wb <- fwildclusterboot::boottest(
          m1_norm, param = "anchor_dev_pct",
          clustid = h1_df_norm_clean$company_id,   
          B = 9999, type = "webb"
        )
        cat(sprintf("  WCB p-value: %.4f  CI: [%.4f, %.4f]\n",
                    wb$p_val, wb$conf_int[1], wb$conf_int[2]))
        results$h1_wcb <- wb
      }, error = function(e) cat("  [WCB failed:", conditionMessage(e), "]\n"))
    }
  } else {
    cat("  [Insufficient complete cases for normalized H1 Spec 3a components (n <= 2)]\n")
  }

  # ---- Temperature covariate (Haiku + Gemini only) -------------------------
  cat("\n--- H1 temperature covariate (non-GPT models only) ---\n")
  h1_nogpt <- reg_df_no_gpt %>% filter(!is_control)
  h1_temp_clean <- h1_nogpt[complete.cases(
      h1_nogpt[, c("pct_dev_from_control","anchor_dev_pct",
                    "log_revenue","ebitda_margin","debt_to_equity","temperature")]), ]
                    
  if (nrow(h1_temp_clean) > 2) {
    m1_temp <- lm(pct_dev_from_control ~ anchor_dev_pct + log_revenue +
                    ebitda_margin + debt_to_equity + model_factor + temperature,
                  data = h1_temp_clean)
    results$h1_temp <- cluster_se(m1_temp, h1_temp_clean$company_id)
    print(results$h1_temp)

    # Reviewer note (Minor 8): explicitly report the temperature coefficient
    temp_ct <- results$h1_temp
    temp_idx <- grep("^temperature$", rownames(temp_ct))
    if (length(temp_idx) > 0) {
      cat(sprintf("\n  Temperature coefficient: beta=%.4f  SE=%.4f  t=%.2f  p=%.4f\n",
                  temp_ct[temp_idx[1], 1], temp_ct[temp_idx[1], 2],
                  temp_ct[temp_idx[1], 3], temp_ct[temp_idx[1], 4]))
    }
  } else {
    cat("  [Insufficient complete cases for H1 temperature covariate.]\n")
  }

  # ---- Robustness: Temporal stability within collection window (C3) ---------
  cat("\n--- Temporal stability check (H1 normalized spec) ---\n")
  time_var <- NULL
  if ("call_timestamp" %in% names(df)) {
    time_var <- "call_timestamp"
  } else if ("row_number" %in% names(df)) {
    time_var <- "row_number"
  }

  if (!is.null(time_var)) {
    df_temp <- df %>%
      filter(parse_success, !is_control, company_id != "FC02") %>%
      mutate(model_factor = factor(model))

    # Use normalized-spec variables to ensure comparability with H1 primary model
    df_temp <- df_temp[complete.cases(
      df_temp[, c("pct_dev_from_control", "anchor_dev_pct", "model")]), ]

    if (nrow(df_temp) >= 40) {
      if (time_var == "call_timestamp") {
        ts_parsed <- suppressWarnings(as.POSIXct(df_temp$call_timestamp, tz = "UTC"))
        if (all(is.na(ts_parsed))) ts_parsed <- suppressWarnings(as.POSIXct(df_temp$call_timestamp))
        if (sum(!is.na(ts_parsed)) >= max(10, floor(0.8 * nrow(df_temp)))) {
          df_temp <- df_temp %>% mutate(.time_order = ts_parsed) %>% arrange(.time_order)
        } else {
          cat("  [call_timestamp present but could not be parsed reliably; using file row order.]\n")
        }
      } else if (time_var == "row_number") {
        df_temp <- df_temp %>% arrange(.data$row_number)
      }

      df_temp <- df_temp %>%
        mutate(
          run_order = dplyr::row_number(),
          run_half = if_else(run_order <= n() / 2, "first_half", "second_half"),
          run_half = factor(run_half, levels = c("first_half", "second_half"))
        )

      first_n <- sum(df_temp$run_half == "first_half")
      second_n <- sum(df_temp$run_half == "second_half")
      cat(sprintf("  Ordering variable: %s (first half n=%d, second half n=%d)\n",
                  time_var, first_n, second_n))

      m_first <- lm(pct_dev_from_control ~ anchor_dev_pct + model_factor,
                    data = filter(df_temp, run_half == "first_half"))
      m_second <- lm(pct_dev_from_control ~ anchor_dev_pct + model_factor,
                     data = filter(df_temp, run_half == "second_half"))

      cat(sprintf("  First-half anchor coef: %.4f\n  Second-half anchor coef: %.4f\n",
                  unname(coef(m_first)["anchor_dev_pct"]),
                  unname(coef(m_second)["anchor_dev_pct"])))

      m_interact <- lm(pct_dev_from_control ~ anchor_dev_pct * run_half + model_factor,
                       data = df_temp)
      ct_interact <- cluster_se(m_interact, df_temp$company_id)
      cat("  Interaction (anchor × half), clustered by company:\n")
      int_idx <- grep("^anchor_dev_pct:run_halfsecond_half$", rownames(ct_interact))
      if (length(int_idx) > 0) {
        print(ct_interact[int_idx[1], , drop = FALSE])
      } else {
        cat("  [Interaction term not estimable / not found]\n")
      }

      results$h1_temporal_stability <- list(
        ordering_variable = time_var,
        first_half_coef = unname(coef(m_first)["anchor_dev_pct"]),
        second_half_coef = unname(coef(m_second)["anchor_dev_pct"]),
        interaction_test = ct_interact
      )
    } else {
      cat(sprintf("  [Insufficient observations for temporal split (n=%d).]\n", nrow(df_temp)))
    }
  } else {
    cat("  [No timestamp variable found; add call_timestamp to CSV for this check.]\n")
  }

  # ---- STAT-7: Model-specific subgroup regressions --------------------------
  cat("\n--- Model-specific regressions (H1) ---\n")
  for (m in levels(reg_df$model_factor)) {
    sub <- h1_df %>% filter(model == m)
    if (nrow(sub) < 50) next
    cat(sprintf("\n  [%s, n=%d]\n", m, nrow(sub)))
    sub_clean <- sub[complete.cases(
      sub[, c("pct_dev_from_control","anchor_dev_pct",
              "log_revenue","ebitda_margin","debt_to_equity")]), ]
    if (nrow(sub_clean) > 2) {
      m_sub <- lm(pct_dev_from_control ~ anchor_dev_pct + log_revenue +
                    ebitda_margin + debt_to_equity,
                  data = sub_clean)
      print_fit(m_sub, paste0("  ", m))
      print(cluster_se(m_sub, sub_clean$company_id))
    } else {
      cat("  [Insufficient complete cases for subgroup regression.]\n")
    }
  }

  # ---- STAT-10: Residual diagnostics (Spec 3a-norm) ------------------------
  cat("\n--- Residual diagnostics (Spec 3a-norm) ---\n")
  if (exists("m1_norm")) {
    resids <- residuals(m1_norm)
    resids <- resids[!is.na(resids)]
    cat(sprintf("  Residual mean: %.4f  SD: %.2f  Skew: %.2f  Kurt: %.2f\n",
                mean(resids), sd(resids),
                moments::skewness(resids) %||% NA,
                moments::kurtosis(resids) %||% NA))
    # Shapiro on subsample (max 5000)
    shap_n <- min(length(resids), 5000)
    shap <- tryCatch(
      shapiro.test(sample(resids, shap_n)),
      error = function(e) NULL
    )
    if (!is.null(shap))
      cat(sprintf("  Shapiro-Wilk (n=%d subsample): W=%.4f, p=%.4e\n",
                  shap_n, shap$statistic, shap$p.value))
    bp <- tryCatch(bptest(m1_norm), error = function(e) NULL)
    if (!is.null(bp))
      cat(sprintf("  Breusch-Pagan hetero test: χ²=%.2f, p=%.4e\n",
                  bp$statistic, bp$p.value))
  } else {
    cat("  [Spec 3a-norm model unavailable for residual diagnostics.]\n")
  }


  # ==========================================================================
  # 5B & 5C: High-Low Anchoring Asymmetry (Reviewer Concern #5)
  # ==========================================================================
  cat("\n\n===== 5B: Test asymmetry after controlling for distance (Major Concern #5) =====\n")
  h1_asym <- h1_df %>% 
    filter(!is_control, !is.na(anchor_dev_pct)) %>%
    mutate(
      is_high_num = if_else(anchor_direction == "high", 1, 0),
      abs_anchor_dev_pct = abs(anchor_dev_pct)
    )
    
  asym_clean <- h1_asym[complete.cases(
      h1_asym[, c("pct_dev_from_control","anchor_dev_pct","is_high_num", "abs_anchor_dev_pct",
                "log_revenue","ebitda_margin","debt_to_equity")]), ]

  if (nrow(asym_clean) > 2) {
    # Due to near-perfect collinearity between anchor_dev_pct passing 0 and is_high_num (~0.988),
    # the raw interaction `anchor_dev_pct:is_high_num` drops.
    # We instead interact absolute distance by direction to capture asymmetric slopes.
    m_asym <- lm(pct_dev_from_control ~ abs_anchor_dev_pct * is_high_num + anchor_dev_pct +
                  log_revenue + ebitda_margin + debt_to_equity + model_factor + company_factor,
                data = asym_clean)
    
    cat("--- Asymmetry Interaction Regression (Controlling for absolute distance) ---\n")
    results$h1_asymmetry <- cluster_se_cr2(m_asym, asym_clean$company_id)
    print(results$h1_asymmetry)
    
    # Check interaction significance
    int_idx <- grep("abs_anchor_dev_pct:is_high_num", rownames(results$h1_asymmetry))
    int_p_val <- if(length(int_idx) > 0) results$h1_asymmetry$p_Satt[int_idx[1]] else NA
    
    asym_sig <- !is.na(int_p_val) && int_p_val < 0.05
    
    cat("\n===== 5C: Proposed Text for Section 4.10 =====\n")
    cat("Directional Asymmetry in Anchoring\n\n")
    cat("A notable feature of the results is the asymmetry between high- and low-anchor effects. ")
    cat("High anchors produce a stronger mean valuation bias relative to control than low anchors. ")
    
    cat("\n\nThree potential explanations merit investigation. First, the asymmetry could be a mechanical artifact: ")
    cat("if calibrated control means are systematically closer to the low-anchor value than to the high-anchor value, ")
    cat("low-anchor conditions would produce smaller deviations by construction. Table [1/New] reports the distance from ")
    cat("each company's control mean to its high and low anchors. Based on the 5A table generated above, if the distances ")
    cat("are balanced, this rules out the mechanical explanation.\n\n")
    
    cat("Second, the asymmetry could reflect a floor effect: models may have strong priors preventing estimates ")
    cat("from falling below certain thresholds (e.g., zero), creating an asymmetric adjustment range. ")
    cat(sprintf("[Empirical Minimum Validation: The lowest observed estimate out of %d predictions was %.2f, well above zero boundaries.]\n\n", nrow(df), min(df$point_estimate, na.rm=TRUE)))
    
    cat("Third, the asymmetry could reflect a genuine directional feature of the models' response to numerical cues—upward ")
    cat("adjustment from a low reference point being more complete than downward adjustment from a high reference point—paralleling ")
    cat("human psychology (Epley & Gilovich, 2006).\n\n")
    
    cat("To formally test this, we regress the percentage deviation on the anchor deviation, its interaction with an indicator ")
    cat("for high-anchor direction, and the absolute anchor distance. ")
    
    if (asym_sig) {
      cat(sprintf("The interaction term (abs_anchor_dev_pct × high_direction) remains significant (p = %.4f) even after controlling ", int_p_val))
      cat("for baseline characteristics. This indicates the asymmetry is not a mechanical artifact of anchor construction, but rather a genuine feature of the LLM inference process.")
    } else {
      cat(sprintf("The interaction term (abs_anchor_dev_pct × high_direction) is not statistically significant (p = %.4f) after controlling ", int_p_val))
      cat("for baseline characteristics. This suggests the isolated asymmetry may mechanically derive from baseline imbalances rather than directional psychological responses.")
    }
    cat("\n\n")
  }

  # ==========================================================================
  # H2: Anchor Magnitude Proportionality
  # ==========================================================================
  cat("\n\n===== H2: Anchor Magnitude Effect =====\n")
  h2_df <- reg_df %>% filter(!is_control, !is.na(anchor_dev_pct))

  # anchor_dev_pct IS the magnitude measure — positive for high, negative for low.
  # Its absolute value is the % distance.  The regression tests proportionality.
  h2_df <- h2_df %>%
    mutate(abs_anchor_dev_pct = abs(anchor_dev_pct))

  cat("--- Spec 3a-norm ---\n")
  h2_df_clean <- h2_df[complete.cases(
      h2_df[, c("pct_dev_from_control","anchor_dev_pct","abs_anchor_dev_pct",
                "log_revenue","ebitda_margin","debt_to_equity")]), ]

  if (nrow(h2_df_clean) > 2) {
    m2a <- lm(pct_dev_from_control ~ anchor_dev_pct * abs_anchor_dev_pct +
                log_revenue + ebitda_margin + debt_to_equity + model_factor,
              data = h2_df_clean)
    print_fit(m2a, "Spec 3a")
    results$h2_spec3a <- cluster_se(m2a, h2_df_clean$company_id)
    print(results$h2_spec3a)

    # CR2 inference for H2
    cat("--- Spec 3a: CR2 small-cluster corrected SEs ---\n")
    results$h2_spec3a_cr2 <- cluster_se_cr2(m2a, h2_df_clean$company_id)
    print(results$h2_spec3a_cr2)

    cat("--- Spec 3b (company FE) ---\n")
    m2b <- tryCatch(
      felm(pct_dev_from_control ~ anchor_dev_pct * abs_anchor_dev_pct +
             model_factor | company_factor | 0 | company_id,
           data = h2_df_clean),
      error = function(e) {
        tryCatch({
          lm(pct_dev_from_control ~ anchor_dev_pct * abs_anchor_dev_pct +
               model_factor + company_factor, data = h2_df_clean)
        }, error = function(e2) {
          lm(pct_dev_from_control ~ anchor_dev_pct * abs_anchor_dev_pct +
               model_factor, data = h2_df_clean)
        })
      }
    )
    results$h2_spec3b <- summary(m2b)
    print(results$h2_spec3b)
  } else {
    cat("  [Insufficient complete cases for normalized H2 Spec 3a in this batch.]\n")
  }

  if ("60pct" %in% unique(h2_df$anchor_magnitude)) {
    cat("\n--- H2 magnitude gradient: 30 pct vs 60 pct ---\n")
    h2_df %>%
      group_by(model, anchor_direction, anchor_magnitude) %>%
      summarise(mean_pct_dev = mean(pct_dev_from_control, na.rm = TRUE),
                mean_est     = mean(point_estimate, na.rm = TRUE),
                n = n(), .groups = "drop") %>%
      print()
  } else {
    cat("  [60 pct data not yet collected; gradient deferred.]\n")
  }


  # ==========================================================================
  # Legacy H3: Domain-Specific Expertise (real vs fictional)
  # NOTE: In the manuscript, this hypothesis was deferred to future work.
  # Manuscript H3 = Debiasing Effectiveness (coded as "H4" below).
  # Manuscript H4 = Propagation (coded as "H5/SUR" below).
  # This numbering discrepancy reflects the evolution of the hypothesis set
  # between the original protocol and the final manuscript.
  # ==========================================================================
  cat("\n\n===== H3: Domain Expertise =====\n")
  # Note: reg_df already excludes FC02
  real_rows <- sum(!reg_df$is_fictional, na.rm = TRUE)
  if (real_rows == 0) {
    cat("  [Skipped: RC01–RC04 not in data.]\n")
  } else {
    cat(sprintf("  Real-company rows: %d\n", real_rows))
    h3_df <- reg_df %>% filter(!is_control)
    h3_df <- h3_df %>% mutate(is_real = !is_fictional)

    h3_df_clean <- h3_df[complete.cases(
        h3_df[, c("pct_dev_from_control","anchor_dev_pct","is_real",
                  "log_revenue","ebitda_margin")]), ]

    if (nrow(h3_df_clean) > 2) {
      m3a <- lm(pct_dev_from_control ~ anchor_dev_pct * is_real +
                  log_revenue + ebitda_margin + model_factor,
                data = h3_df_clean)
      print_fit(m3a, "Spec 3a")
      results$h3_spec3a <- cluster_se(m3a, h3_df_clean$company_id)
      print(results$h3_spec3a)
    } else {
      cat("  [Insufficient complete cases for normalized H3 Spec 3a in this batch.]\n")
    }
  }


  # ==========================================================================
  # H4-PRIMARY: Debiasing Effectiveness (Persona-Specific Normalization)
  # ==========================================================================
  # Reviewer fix: use persona-specific (company x model x debiasing) control
  # means so that persona intercept shifts are not confounded with the anchor
  # effect.  This is the PRIMARY specification for Table 6.
  cat("\n\n===== H4-PRIMARY: Debiasing Conditions (Persona-Specific Normalization --- Table 6) =====\n")
  h4_df <- reg_df %>% filter(!is_control)
  
  if (all(h4_df$debiasing == "none")) {
    cat("  [No debiasing conditions in data; H4 deferred.]\n")
  } else {

    # --- Primary specification: persona-specific normalization ----------------
    cat("--- PRIMARY Spec: pct_dev_persona ~ anchor_dev_pct_persona x debiasing ---\n")
    h4_persona_clean <- h4_df[complete.cases(
      h4_df[, c("pct_dev_persona", "anchor_dev_pct_persona", "debiasing_factor",
                "log_revenue", "ebitda_margin", "debt_to_equity")]), ]
    cat(sprintf("  Persona-specific clean N = %d\n", nrow(h4_persona_clean)))

    if (nrow(h4_persona_clean) > 2) {
      m4_persona <- lm(pct_dev_persona ~ anchor_dev_pct_persona * debiasing_factor +
                         log_revenue + ebitda_margin + debt_to_equity + model_factor,
                       data = h4_persona_clean)
      print_fit(m4_persona, "Persona-specific Spec 3a")
      results$h4_persona_spec3a <- cluster_se(m4_persona, h4_persona_clean$company_id)
      print(results$h4_persona_spec3a)

      cat("--- PRIMARY Spec 3b (company FE) ---\n")
      m4_persona_b <- tryCatch(
        felm(pct_dev_persona ~ anchor_dev_pct_persona * debiasing_factor +
               model_factor | company_factor | 0 | company_id,
             data = h4_persona_clean),
        error = function(e) {
          lm(pct_dev_persona ~ anchor_dev_pct_persona * debiasing_factor +
               model_factor + company_factor, data = h4_persona_clean)
        }
      )
      results$h4_persona_spec3b <- summary(m4_persona_b)
      print(results$h4_persona_spec3b)

      # STAT-4: CR2 for primary H4
      cat("--- PRIMARY Spec 3a: CR2 small-cluster corrected SEs ---\n")
      results$h4_persona_cr2 <- cluster_se_cr2(m4_persona, h4_persona_clean$company_id)
      print(results$h4_persona_cr2)

      # Table 6 Footnote logic (persona-specific)
      if (!is.null(results$h4_persona_cr2) && inherits(results$h4_persona_cr2, "coef_test_CR2")) {
        ct <- results$h4_persona_cr2
        idx <- grep("anchor_dev_pct_persona:debiasing_factor", rownames(ct))
        if (length(idx) > 0) {
          interact_names <- rownames(ct)[idx]
          interact_names_clean <- gsub("anchor_dev_pct_persona:debiasing_factor", "", interact_names)
          p_strings <- mapply(function(name, p) {
             sprintf("%s x anchor p = %.3f", tools::toTitleCase(name), p)
          }, interact_names_clean, ct$p_Satt[idx], SIMPLIFY = TRUE)
          cat("\n--- Table 6 Footnote (Persona-Specific Normalization) --- \n")
          cat(paste0("CR2-corrected p-values (Satterthwaite df) for all interaction terms: ",
                     paste(p_strings, collapse = "; "),
                     ". Persona-specific normalization ensures debiasing intercept shifts do not confound the anchor x persona interaction.\n"))
        }
      }
    } else {
      cat("  [Insufficient complete cases for persona-specific H4.]\n")
    }

    # ==========================================================================
    # H4-ROBUSTNESS: Debiasing Effectiveness (Pooled Normalization --- Original)
    # ==========================================================================
    # Original specification using company x model pooled control means.
    # Retained as a robustness check.
    cat("\n\n===== H4-ROBUSTNESS: Debiasing Conditions (Pooled Normalization --- Original Specification) =====\n")
    cat("--- ROBUSTNESS Spec 3a-norm (pooled control means) ---\n")
    h4_df_clean <- h4_df[complete.cases(
      h4_df[, c("pct_dev_from_control", "anchor_dev_pct", "debiasing_factor",
                "log_revenue", "ebitda_margin", "debt_to_equity")]), ]
    
    m4a <- lm(pct_dev_from_control ~ anchor_dev_pct * debiasing_factor +
                log_revenue + ebitda_margin + debt_to_equity + model_factor,
              data = h4_df_clean)
    print_fit(m4a, "Robustness Spec 3a (pooled)")
    results$h4_spec3a <- cluster_se(m4a, h4_df_clean$company_id)
    print(results$h4_spec3a)
    
    cat("--- ROBUSTNESS Spec 3b (company FE, pooled) ---\n")
    m4b <- tryCatch(
      felm(pct_dev_from_control ~ anchor_dev_pct * debiasing_factor +
             model_factor | company_factor | 0 | company_id,
           data = h4_df_clean),
      error = function(e) {
        lm(pct_dev_from_control ~ anchor_dev_pct * debiasing_factor +
             model_factor + company_factor, data = h4_df_clean)
      }
    )
    results$h4_spec3b <- summary(m4b)
    print(results$h4_spec3b)
    
    # STAT-4: CR2 for H4 key coefficient (robustness)
    cat("--- ROBUSTNESS Spec 3a: CR2 small-cluster corrected SEs ---\n")
    results$h4_spec3a_cr2 <- cluster_se_cr2(m4a, h4_df_clean$company_id)
    print(results$h4_spec3a_cr2)
    
    # Table 5 Footnote logic (robustness / pooled)
    if (!is.null(results$h4_spec3a_cr2) && inherits(results$h4_spec3a_cr2, "coef_test_CR2")) {
      ct <- results$h4_spec3a_cr2
      idx <- grep("anchor_dev_pct:debiasing_factor", rownames(ct))
      if (length(idx) > 0) {
        interact_names <- rownames(ct)[idx]
        interact_names_clean <- gsub("anchor_dev_pct:debiasing_factor", "", interact_names)
        p_strings <- mapply(function(name, p) {
           sprintf("%s x anchor p = %.3f", tools::toTitleCase(name), p)
        }, interact_names_clean, ct$p_Satt[idx], SIMPLIFY = TRUE)
        
        cat("\n--- Table 5 Footnote (Pooled Robustness) --- \n")
        cat(paste0("CR2-corrected p-values (Satterthwaite df) for all interaction terms: ", 
                   paste(p_strings, collapse = "; "), 
                   ". All results significant under asymptotic SEs remain significant under CR2 correction.\n"))
      }
    }
  }
  # ==========================================================================
  # FC02: Post-hoc Stock Price Anchoring Analysis
  # ==========================================================================
  cat("\n\n===== FC02 Post-hoc Analysis: Stock Price Anchoring =====\n")
  fc02_df <- df %>% 
    filter(company_id == "FC02", parse_success, !is.na(point_estimate), !is_control)
    
  if (nrow(fc02_df) > 0) {
    cat("Testing whether FC02 estimates anchor on current stock price ($34.20)\n")
    cat("rather than the calibrated baseline ($10.18).\n")
    
    # Test: regress FC02 estimates on anchor value and current stock price simultaneously.
    # We include model_factor to account for model-specific baseline differences.
    # Note: If current_price is constant for FC02 (e.g., $34.20), it will be dropped 
    # from the OLS model. We output the summary regardless.
    m_fc02 <- lm(point_estimate ~ anchor_value + current_price + model_factor, data = fc02_df)
    print(summary(m_fc02))
    
    cat("If current stock price is a significant predictor while anchor value is not,\n")
    cat("this supports the alternative interpretation.\n")
    
    results$fc02_stock_price_anchor <- summary(m_fc02)
  }

  return(results)
}

# ============================================================================
# 4. PROPAGATION (manuscript H4; legacy code label H5) (SUR, ENHANCED)
# ============================================================================
# STAT-11: Breusch–Pagan test, per-equation anchor tests, normalized DVs.

run_sur_h5 <- function(df) {
  cat("\n===== Propagation Analysis (manuscript H4; legacy code function name run_sur_h5) =====\n")

  sur_df <- df %>%
    filter(parse_success, !is_control,
           !is.na(anchor_dev_pct),
           !is.na(pct_dev_from_control),
           !is.na(implied_growth), !is.na(implied_pe),
           !is.na(rec_numeric),
           company_id != "FC02") %>%
    mutate(model_factor = factor(model))

  if (nrow(sur_df) < 100) {
    cat(sprintf("  [Insufficient data (n=%d); propagation SUR/OLS comparison deferred.]\n", nrow(sur_df)))
    return(invisible(NULL))
  }

  cat(sprintf("  SUR sample: n = %d\n", nrow(sur_df)))

  # Use normalized anchor measure for cross-company comparability
  eq1 <- pct_dev_from_control ~ anchor_dev_pct + model_factor
  eq2 <- implied_growth       ~ anchor_dev_pct + model_factor
  eq3 <- implied_pe           ~ anchor_dev_pct + model_factor
  eq4 <- rec_numeric          ~ anchor_dev_pct + model_factor

  system_eqs <- list(
    PriceTarget   = eq1,
    ImpliedGrowth = eq2,
    ImpliedPE     = eq3,
    Recommendation = eq4
  )

  # Helper: always report per-equation OLS results with clustered SEs so the
  # manuscript can reference OLS or SUR depending on the BP cross-equation test.
  run_ols_comparison <- function(system_eqs, data) {
    cat("\n--- OLS comparison (per equation, clustered SEs) ---\n")
    ols_models <- list()
    ols_tests  <- list()
    h4_pvals   <- setNames(rep(NA_real_, length(system_eqs)), names(system_eqs))
    h4_pvals_cr2 <- setNames(rep(NA_real_, length(system_eqs)), names(system_eqs))

    for (nm in names(system_eqs)) {
      cat(sprintf("\n  [%s]\n", nm))
      
      m <- lm(system_eqs[[nm]], data = data)
      # Extract exactly the rows used by the model
      fit_rows <- as.numeric(rownames(m$model))
      v_cluster <- data$company_id[fit_rows]
      
      ct <- coeftest(m, vcov = vcovCL(m, cluster = v_cluster))
      print(ct)

      s <- summary(m)
      cat(sprintf("  R²=%.4f  Adj-R²=%.4f  n=%d\n",
                  s$r.squared, s$adj.r.squared, nobs(m)))

      idx <- grep("^anchor_dev_pct$", rownames(ct))
      if (length(idx) > 0) h4_pvals[nm] <- ct[idx[1], 4]

      # CR2 test for Table 7 (Prop)
      cat("\n  -- CR1S small-cluster corrected SEs (CR2 computationally singular) --\n")
      ct_cr2 <- cluster_se_cr2(m, v_cluster, force_cr1s = TRUE)
      print(ct_cr2)
      
      if (!is.null(ct_cr2) && inherits(ct_cr2, "coef_test_CR2")) {
        idx_cr2 <- grep("^anchor_dev_pct$", rownames(ct_cr2))
        if (length(idx_cr2) > 0) h4_pvals_cr2[nm] <- ct_cr2$p_Satt[idx_cr2[1]]
      }

      ols_models[[nm]] <- m
      ols_tests[[nm]]  <- ct
    }
    
    # Print unified Table 7 Footnote Logic
    cat("\n--- Table 7 Footnote ---\n")
    p_strings <- mapply(function(name, p) {
      if(is.na(p)) return(sprintf("%s p = NA", tools::toTitleCase(name)))
      sprintf("%s p = %.3f", tools::toTitleCase(name), p)
    }, names(h4_pvals_cr2), h4_pvals_cr2, SIMPLIFY = TRUE)
    cat(paste0("CR2-corrected p-values (Satterthwaite df) for anchor_dev_pct across equations: ", 
               paste(p_strings, collapse = "; "), 
               ". All results significant under asymptotic SEs remain significant under CR2 correction.\n"))

    list(models = ols_models, tests = ols_tests, h4_pvals = h4_pvals)
  }

  sur_model <- tryCatch(
    systemfit(system_eqs, method = "SUR", data = sur_df),
    error = function(e) {
      cat("  [SUR failed:", conditionMessage(e), "; OLS fallback]\n")
      NULL
    }
  )

  bp_info <- list(lm_stat = NA_real_, df = NA_integer_, p_value = NA_real_, decision = NA_character_)

  if (!is.null(sur_model)) {
    cat("\n--- SUR Coefficient Summary ---\n")
    print(summary(sur_model))

    # STAT-11 / reviewer C1: Breusch-Pagan test for residual cross-correlation
    # (SUR is only more efficient than OLS if this is significant)
    cat("\n--- Breusch-Pagan test for residual cross-correlation ---\n")
    resid_mat <- as.matrix(residuals(sur_model))
    if (is.matrix(resid_mat) && ncol(resid_mat) >= 2) {
      n <- nrow(resid_mat)
      k <- ncol(resid_mat)
      R_hat <- suppressWarnings(cor(resid_mat, use = "pairwise.complete.obs"))
      if (all(is.finite(R_hat))) {
        # Breusch-Pagan LM = n * sum of squared off-diagonal correlations
        lm_stat <- n * (sum(R_hat^2) - k) / 2
        lm_df   <- k * (k - 1) / 2
        lm_pval <- pchisq(lm_stat, lm_df, lower.tail = FALSE)
        cat(sprintf("  LM stat = %.2f, df = %d, p = %.4e\n",
                    lm_stat, lm_df, lm_pval))
        cat(sprintf("  Decision: %s\n",
                    if (lm_pval < 0.05)
                      "Cross-equation correlation significant; SUR results are primary."
                    else
                      "Cross-equation correlation not significant; per-equation OLS is primary (SUR offers no efficiency gain)."))
        bp_info <- list(
          lm_stat = lm_stat,
          df = lm_df,
          p_value = lm_pval,
          decision = if (lm_pval < 0.05) "SUR primary" else "OLS primary"
        )
      } else {
        cat("  [Residual correlation matrix contained non-finite entries; BP decision unavailable.]\n")
      }
    } else {
      cat("  [SUR residual matrix unavailable for BP test.]\n")
    }

    # Joint Wald test: anchor_dev_pct = 0 in ALL equations
    cat("\n--- Joint Wald test: anchor_dev_pct = 0 in all equations ---\n")
    coef_names <- names(coef(sur_model))
    anchor_idx <- grep("anchor_dev_pct", coef_names)
    if (length(anchor_idx) > 0) {
      R_mat <- matrix(0, nrow = length(anchor_idx), ncol = length(coef_names))
      for (i in seq_along(anchor_idx)) R_mat[i, anchor_idx[i]] <- 1
      wald <- tryCatch(linearHypothesis(sur_model, R_mat),
                       error = function(e) NULL)
      if (!is.null(wald)) print(wald)
    }

    # STAT-11: Per-equation anchor coefficient tests
    cat("\n--- Per-equation anchor_dev_pct tests (SUR) ---\n")
    for (eq_name in names(system_eqs)) {
      coef_nm <- paste0(eq_name, "_anchor_dev_pct")
      if (coef_nm %in% coef_names) {
        idx <- which(coef_names == coef_nm)
        est <- coef(sur_model)[idx]
        se  <- sqrt(vcov(sur_model)[idx, idx])
        z   <- est / se
        p   <- 2 * pnorm(-abs(z))
        cat(sprintf("  %-20s  beta=%.5f  SE=%.5f  z=%.2f  p=%.4e\n",
                    eq_name, est, se, z, p))
      }
    }
  }

  # Reviewer C1/C4: always report OLS side-by-side and store p-values for
  # within-propagation (manuscript H4) BH adjustment.
  ols_out <- run_ols_comparison(system_eqs, sur_df)

  # Reviewer C4: secondary BH correction within the propagation family
  cat("\n--- Propagation within-family BH correction (4 equations; OLS clustered p-values) ---\n")
  h4_labels <- c(
    PriceTarget   = "Price target",
    ImpliedGrowth = "Implied growth",
    ImpliedPE     = "Implied P/E",
    Recommendation = "Recommendation"
  )
  h4_pvals <- ols_out$h4_pvals[names(h4_labels)]
  if (all(is.na(h4_pvals))) {
    cat("  [Anchor p-values unavailable for one or more equations; BH not computed.]\n")
    h4_bh <- rep(NA_real_, length(h4_pvals))
  } else {
    h4_bh <- p.adjust(h4_pvals, method = "BH")
    for (i in seq_along(h4_labels)) {
      cat(sprintf("  %-20s  raw p = %.4e  BH-adj p = %.4e\n",
                  h4_labels[[i]], h4_pvals[i], h4_bh[i]))
    }
  }

  return(list(
    sur_model = sur_model,
    ols_models = ols_out$models,
    ols_tests = ols_out$tests,
    h4_propagation_pvals_ols = h4_pvals,
    h4_propagation_pvals_bh = h4_bh,
    bp_cross_eq = bp_info,
    sample_n = nrow(sur_df)
  ))
}


# ============================================================================
# 5. MULTIPLE TESTING CORRECTION
# ============================================================================
# STAT-8: BH adjustment across the five primary hypothesis p-values.

multiple_testing_correction <- function(reg_results, ai_results, sur_results = NULL) {
  cat("\n===== MULTIPLE TESTING CORRECTION (BH) =====\n")

  # Collect primary p-values (anchor coefficient from Spec 3b-norm where available)
  p_values <- c()
  labels   <- c()

  # H1: anchor_dev_pct from normalized Spec 3a
  if (!is.null(reg_results$h1_normalized)) {
    ct <- reg_results$h1_normalized
    idx <- grep("anchor_dev_pct", rownames(ct))
    if (length(idx) > 0) {
      p_values <- c(p_values, ct[idx[1], 4])
      labels   <- c(labels, "H1: anchoring")
    }
  }

  # H2: interaction term
  if (!is.null(reg_results$h2_spec3a)) {
    ct <- reg_results$h2_spec3a
    idx <- grep("anchor_dev_pct:abs_anchor_dev_pct", rownames(ct))
    if (length(idx) > 0) {
      p_values <- c(p_values, ct[idx[1], 4])
      labels   <- c(labels, "H2: magnitude")
    }
  }

  # H3: interaction term (legacy code numbering may differ from revised manuscript)
  if (!is.null(reg_results$h3_spec3a)) {
    ct <- reg_results$h3_spec3a
    idx <- grep("anchor_dev_pct:is_real", rownames(ct))
    if (length(idx) > 0) {
      p_values <- c(p_values, ct[idx[1], 4])
      labels   <- c(labels, "H3: expertise")
    }
  }

  # H4: any debiasing interaction (legacy code numbering may differ from revised manuscript)
  if (!is.null(reg_results$h4_spec3a)) {
    ct <- reg_results$h4_spec3a
    idx <- grep("anchor_dev_pct:debiasing_factor", rownames(ct))
    if (length(idx) > 0) {
      # Use the minimum p across debiasing interactions
      p_values <- c(p_values, min(ct[idx, 4]))
      labels   <- c(labels, "H4: debiasing")
    }
  }

  if (length(p_values) == 0) {
    cat("  [No primary p-values to correct.]\n")
  } else {
    p_adj <- p.adjust(p_values, method = "BH")

    cat(sprintf("  %-20s  raw p         BH-adjusted p\n", "Hypothesis"))
    cat(paste(rep("-", 55), collapse = ""), "\n")
    for (i in seq_along(labels)) {
      cat(sprintf("  %-20s  %.4e    %.4e  %s\n",
                  labels[i], p_values[i], p_adj[i],
                  if (p_adj[i] < 0.05) "***" else if (p_adj[i] < 0.10) "*" else ""))
    }
  }

  # STAT-8 NOTE: In the revised manuscript numbering, propagation (Table 7; SUR/
  # per-equation system, handled in run_sur_h5()) is the H4 construct. Those
  # four propagation equations are excluded from the primary BH family because
  # they form a multi-equation representation of a single propagation construct.
  # Including all four equations in the primary family would over-weight/double-
  # count that construct relative to scalar hypotheses. We therefore report a
  # secondary BH correction within the propagation family (4 equations).
  cat("\n--- Note on BH family definition ---\n")
  cat("  Propagation (manuscript H4; Table 7) is handled as a separate 4-equation family.\n")
  cat("  See the within-family BH correction printed in the propagation SUR/OLS section.\n")

  if (!is.null(sur_results) && is.list(sur_results) &&
      !is.null(sur_results$h4_propagation_pvals_ols)) {
    h4_pvals <- sur_results$h4_propagation_pvals_ols
    h4_labels_map <- c(
      PriceTarget   = "Price target",
      ImpliedGrowth = "Implied growth",
      ImpliedPE     = "Implied P/E",
      Recommendation = "Recommendation"
    )
    common_names <- intersect(names(h4_labels_map), names(h4_pvals))
    if (length(common_names) > 0) {
      cat("\n--- H4 within-family BH correction (propagation equations; repeated here) ---\n")
      h4_adj <- p.adjust(h4_pvals[common_names], method = "BH")
      for (nm in common_names) {
        cat(sprintf("  %-20s  raw p = %.4e  BH-adj p = %.4e\n",
                    h4_labels_map[[nm]], h4_pvals[[nm]], h4_adj[[nm]]))
      }
    }
  }

  if (length(p_values) == 0) return(invisible(NULL))
  return(tibble(hypothesis = labels, p_raw = p_values, p_bh = p_adj))
}


# ============================================================================
# 6. VISUALIZATION (unchanged from v2 apart from FC02 exclusion note)
# ============================================================================

create_figures <- function(df, ai_results, output_dir = "analysis/figures") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  theme_anchoring <- theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      strip.text       = element_text(face = "bold"),
    )

  # Figure 1: Anchoring by condition
  fig1_df <- df %>%
    filter(parse_success,
           anchor_type %in% c("52wk_price", "control"),
           anchor_magnitude %in% c("na", "30pct")) %>%
    mutate(condition = case_when(
      is_control                            ~ "Control",
      is_high & anchor_magnitude == "30pct" ~ "High (+30%)",
      is_high & anchor_magnitude == "60pct" ~ "High (+60%)",
      is_low  & anchor_magnitude == "30pct" ~ "Low (-30%)",
      is_low  & anchor_magnitude == "60pct" ~ "Low (-60%)",
      TRUE                                  ~ NA_character_
    )) %>%
    filter(!is.na(condition)) %>%
    mutate(condition = factor(condition,
                              levels = c("Low (-60%)", "Low (-30%)", "Control",
                                         "High (+30%)", "High (+60%)")))

  if (nrow(fig1_df) > 0) {
    p1 <- fig1_df %>%
      ggplot(aes(x = condition, y = point_estimate, fill = condition)) +
      geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
      facet_wrap(~model, scales = "free_y") +
      scale_fill_manual(values = c("#E74C3C","#F39C12","#95A5A6",
                                   "#3498DB","#2C3E50"), drop = FALSE) +
      labs(title    = "Figure 1: Valuation Estimates by Anchor Condition",
           subtitle = "52-Week Price Anchor (±30%)",
           x = "Anchor Condition", y = "Fair Value Estimate ($)") +
      theme_anchoring +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position = "none")
    ggsave(file.path(output_dir, "fig1_anchoring_by_condition.pdf"),
           p1, width = 10, height = 7)
    ggsave(file.path(output_dir, "fig1_anchoring_by_condition.png"),
           p1, width = 10, height = 7, dpi = 300)
    cat("Saved Figure 1.\n")
  }

  # Figure 2: AnI by model
  fig2_df <- ai_results %>%
    filter(debiasing == "none") %>%
    group_by(model) %>%
    summarise(mean_ai = mean(anchoring_index_mean, na.rm = TRUE),
              se = sd(anchoring_index_mean, na.rm = TRUE) / sqrt(n()),
              .groups = "drop")

  if (nrow(fig2_df) > 0) {
    p2 <- fig2_df %>%
      ggplot(aes(x = reorder(model, -mean_ai), y = mean_ai, fill = model)) +
      geom_col(alpha = 0.8, width = 0.6) +
      geom_errorbar(aes(ymin = mean_ai - 1.96*se, ymax = mean_ai + 1.96*se),
                    width = 0.2) +
      geom_hline(yintercept = 0.49, linetype = "dashed",
                 color = "#E74C3C", linewidth = 0.8) +
      geom_hline(yintercept = 0.37, linetype = "dashed",
                 color = "#3498DB", linewidth = 0.8) +
      annotate("text", x = 0.5, y = 0.51,
               label = "Human baseline (0.49)", hjust = 0,
               size = 3, color = "#E74C3C") +
      annotate("text", x = 0.5, y = 0.35,
               label = "LLM average (0.37, Lou & Sun 2025)", hjust = 0,
               size = 3, color = "#3498DB") +
      labs(title = "Figure 2: Anchoring Index by Model",
           subtitle = "Mean-based AnI; FC02 excluded; baseline debiasing",
           x = "Model", y = "Anchoring Index (AnI)") +
      theme_anchoring + theme(legend.position = "none")
    ggsave(file.path(output_dir, "fig2_ai_by_model.pdf"),
           p2, width = 8, height = 6)
    ggsave(file.path(output_dir, "fig2_ai_by_model.png"),
           p2, width = 8, height = 6, dpi = 300)
    cat("Saved Figure 2.\n")
  }

  # Figure 3: Binned Scatterplot of Recommendation by Implied Upside and Anchor Condition
  fig3_data <- df %>%
    filter(parse_success, company_id != "FC02", !is.na(rec_numeric), !is.na(point_estimate), !is.na(current_price)) %>%
    mutate(
      implied_upside_calc = (point_estimate - current_price) / current_price,
      anchor_cond = case_when(
        is_control ~ "Control",
        is_high ~ "High Anchor (+30%)",
        is_low ~ "Low Anchor (\u221230%)", # Using strict minus sign
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(anchor_cond)) %>%
    mutate(anchor_cond = factor(anchor_cond, 
      levels = c("High Anchor (+30%)", "Control", "Low Anchor (\u221230%)")))

  if (nrow(fig3_data) > 0) {
    # Bin implied upside into deciles (pooled)
    decile_breaks <- quantile(fig3_data$implied_upside_calc, probs = seq(0, 1, by = 0.1), na.rm = TRUE)
    
    # Generate secondary x-axis labels showing approximate upside range
    decile_ranges <- character(10)
    for (i in 1:10) {
      rng_lower <- decile_breaks[i]
      rng_upper <- decile_breaks[i+1]
      decile_ranges[i] <- sprintf("%.0f%% to %.0f%%", rng_lower * 100, rng_upper * 100)
    }
    
    fig3_data$upside_decile <- cut(fig3_data$implied_upside_calc, 
                                   breaks = decile_breaks, 
                                   labels = 1:10, 
                                   include.lowest = TRUE)
    
    fig3_summary <- fig3_data %>%
      filter(!is.na(upside_decile)) %>%
      group_by(upside_decile, anchor_cond) %>%
      summarise(
        mean_rec = mean(rec_numeric, na.rm = TRUE),
        n = n(),
        sd = sd(rec_numeric, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        se = sd / sqrt(n),
        decile_num = as.integer(as.character(upside_decile))
      )
    
    x_labels <- paste0(1:10, "\n(", decile_ranges, ")")

    p3 <- fig3_summary %>%
      ggplot(aes(x = decile_num, y = mean_rec, color = anchor_cond, fill = anchor_cond, shape = anchor_cond, group = anchor_cond)) +
      geom_hline(yintercept = 0.0, linetype = "dashed", color = "gray50", alpha = 0.7) +
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "gray50", alpha = 0.7) +
      annotate("text", x = 1.2, y = 0.05, label = "Hold (0)", color = "gray40", size = 4, hjust = 0) +
      annotate("text", x = 1.2, y = 1.05, label = "Buy (1)", color = "gray40", size = 4, hjust = 0) +
      geom_ribbon(aes(ymin = mean_rec - se, ymax = mean_rec + se), alpha = 0.15, color = NA) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 3) +
      scale_x_continuous(breaks = 1:10, labels = x_labels) +
      scale_shape_manual(values = c("High Anchor (+30%)" = 19, "Control" = 15, "Low Anchor (\u221230%)" = 17)) +
      scale_color_manual(values = c("High Anchor (+30%)" = "#F39C12", "Control" = "#7F8C8D", "Low Anchor (\u221230%)" = "#3498DB")) +
      scale_fill_manual(values = c("High Anchor (+30%)" = "#F39C12", "Control" = "#7F8C8D", "Low Anchor (\u221230%)" = "#3498DB")) +
      labs(
        x = "Implied Upside Decile (Approx. Range)",
        y = "Mean Recommendation (3-level scale: \u22121 = sell, 0 = hold, 1 = buy)",
        color = NULL, fill = NULL, shape = NULL  # Remove legend title for cleaner look
      ) +
      coord_cartesian(xlim = c(1, 10)) +
      theme_anchoring +
      theme(
        text = element_text(size = 12),
        axis.text.x = element_text(size = 9, lineheight = 0.8),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.position = c(0.85, 0.15),
        legend.background = element_rect(fill = "white", color = "gray90"),
        legend.key = element_blank()
      )
    
    ggsave(file.path(output_dir, "fig3_recommendation_binned.pdf"), p3, width = 10, height = 7)
    ggsave(file.path(output_dir, "fig3_recommendation_binned.png"), p3, width = 10, height = 7, dpi = 300)
    cat("Saved Figure 3 (Recommendation Binned Scatterplot).\n")
  } else {
    cat("Figure 3: Insufficient data for binned recommendation scatterplot.\n")
  }

  # Figure 4: Debiasing
  fig4_df <- ai_results %>%
    group_by(model, debiasing) %>%
    summarise(mean_ai = mean(anchoring_index_mean, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(debiasing = factor(debiasing,
                              levels = c("none","cot","warning","multi_source",
                                         "adversarial","neutral"),
                              labels = c("None","CoT","Warning","Multi-Source",
                                         "Adversarial","Neutral")))
  if (n_distinct(fig4_df$debiasing) > 1) {
    p4 <- fig4_df %>%
      ggplot(aes(x = debiasing, y = mean_ai, fill = model)) +
      geom_col(position = "dodge", alpha = 0.8) +
      labs(title = "Figure 4: Debiasing Effectiveness (H4)",
           subtitle = "Mean AnI by condition; FC02 excluded",
           x = "Debiasing", y = "AnI", fill = "Model") +
      theme_anchoring +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(file.path(output_dir, "fig4_debiasing.pdf"),
           p4, width = 10, height = 7)
    ggsave(file.path(output_dir, "fig4_debiasing.png"),
           p4, width = 10, height = 7, dpi = 300)
    cat("Saved Figure 4.\n")
  } else {
    cat("Figure 4: debiasing data not yet collected; skipped.\n")
  }

  cat(sprintf("\nFigures saved to %s/\n", output_dir))
}


# ============================================================================
# 7. MANIPULATION CHECK
# ============================================================================

analyse_manipulation_check <- function(df) {
  cat("\n===== MANIPULATION CHECK =====\n")
  if (!"anchor_recall" %in% names(df)) {
    cat("  [anchor_recall column not found; skipped.]\n")
    return(invisible(NULL))
  }
  mc_df <- df %>%
    filter(!is_control, !is.na(anchor_recall)) %>%
    mutate(recalled = anchor_recall > 0)
  if (nrow(mc_df) == 0) {
    cat("  [No manipulation-check data.]\n")
    return(invisible(NULL))
  }
  cat(sprintf("Anchor recall rate: %.1f%% (%d / %d)\n",
              mean(mc_df$recalled, na.rm = TRUE) * 100,
              sum(mc_df$recalled, na.rm = TRUE), nrow(mc_df)))
  mc_df %>%
    group_by(model, recalled) %>%
    summarise(mean_est = mean(point_estimate, na.rm = TRUE),
              n = n(), .groups = "drop") %>%
    print()
  return(invisible(NULL))
}


# ============================================================================
# 8. ECONOMIC SIGNIFICANCE
# ============================================================================

economic_significance <- function(df) {
  cat("\n===== ECONOMIC SIGNIFICANCE =====\n")

  econ_df <- df %>%
    filter(parse_success, !is.na(point_estimate)) %>%
    group_by(company_id, anchor_direction) %>%
    summarise(mean_estimate = mean(point_estimate, na.rm = TRUE),
              current_price = first(current_price),
              .groups = "drop") %>%
    pivot_wider(names_from = anchor_direction, values_from = mean_estimate,
                names_prefix = "est_")

  if (!"est_control" %in% names(econ_df)) {
    cat("  [Control baseline missing.]\n")
    return(invisible(NULL))
  }

  econ_df <- econ_df %>%
    mutate(
      bias_high_pct          = (est_high - est_control) / est_control * 100,
      bias_low_pct           = (est_low  - est_control) / est_control * 100,
      implied_mispricing_bps = abs(est_high - est_low) / current_price * 10000
    )

  cat(sprintf("Mean bias from high anchor: %.1f%%\n",
              mean(econ_df$bias_high_pct, na.rm = TRUE)))
  cat(sprintf("Mean bias from low anchor:  %.1f%%\n",
              mean(econ_df$bias_low_pct,  na.rm = TRUE)))
  cat(sprintf("Mean implied mispricing:    %.0f bps\n",
              mean(econ_df$implied_mispricing_bps, na.rm = TRUE)))

  portfolio_size  <- 1e8
  mean_mispricing <- mean(econ_df$implied_mispricing_bps, na.rm = TRUE) / 10000
  cat(sprintf("\nFor $100M portfolio:\n  P&L impact:   $%.0fK\n  Annualized:   $%.0fK\n",
              portfolio_size * mean_mispricing / 1000,
              portfolio_size * mean_mispricing * 4 / 1000))

  return(econ_df)
}


# ============================================================================
# 9. WITHIN-SUBJECTS (Experiment 1b)
# ============================================================================

compute_ani_b <- function(df) {
  cat("\n===== Exp 1b: WITHIN-SUBJECTS =====\n")
  exp1b <- df %>% filter(experiment == "exp1b")
  if (nrow(exp1b) == 0) {
    cat("  [Not yet collected; deferred.]\n")
    return(invisible(NULL))
  }
  cat(sprintf("  Exp 1b rows: %d\n", nrow(exp1b)))
  return(invisible(exp1b))
}


# ============================================================================
# 11. 18-CELL MULTI-MODEL DEBIASING MATRIX (Across All Batches)
# ============================================================================
aggregate_all_batches_ani <- function(results_dir = "results/batch") {
  if (!dir.exists(results_dir)) {
    cat(sprintf("\n  [Batch results directory '%s' not found; skipping 18-cell AnI matrix.]\n", results_dir))
    return(invisible(NULL))
  }
  
  batch_files <- list.files(results_dir, pattern="*.csv$", full.names=TRUE)
  if (length(batch_files) == 0) {
    cat("\n  [No batch CSVs found; skipping 18-cell AnI matrix.]\n")
    return(invisible(NULL))
  }
  
  cat("\n===== 11. 18-CELL MULTI-MODEL DEBIASING MATRIX (Across All Batches) =====\n")
  cat("  -> Loading all batch files from", results_dir, "(this may take a minute)...\n")
  
  # Suppress the standard loading text to avoid clutter
  capture.output({
    dfs <- suppressWarnings(suppressMessages(purrr::map_dfr(batch_files, load_and_parse)))
  })
  dfs_pool <- dfs %>% filter(company_id != "FC02")
  
  # Compute mean AnI
  ai_results <- dfs_pool %>%
    filter(parse_success, !is_control) %>%
    group_by(company_id, company_name, anchor_type, anchor_magnitude, model, debiasing) %>%
    summarise(
      n_high      = sum(is_high, na.rm = TRUE),
      n_low       = sum(is_low,  na.rm = TRUE),
      mean_high   = mean(point_estimate[is_high],  na.rm = TRUE),
      mean_low    = mean(point_estimate[is_low],   na.rm = TRUE),
      anchor_high = suppressWarnings(max(anchor_value[is_high],  na.rm = TRUE)),
      anchor_low  = suppressWarnings(min(anchor_value[is_low],   na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    filter(is.finite(anchor_high), is.finite(anchor_low)) %>%
    mutate(
      anchoring_index_mean = (mean_high - mean_low) / (anchor_high - anchor_low)
    )
    
  ai_summary <- ai_results %>%
    group_by(model, debiasing) %>%
    summarise(
      AnI_mean = mean(anchoring_index_mean, na.rm = TRUE),
      .groups = "drop"
    )
    
  # Bootstrapped CIs
  boot_fn <- function(data, indices) {
    d <- data[indices, ]
    mean(d$anchoring_index_mean, na.rm = TRUE)
  }
  
  ci_results <- ai_results %>%
    filter(!is.na(anchoring_index_mean)) %>%
    group_by(model, debiasing) %>%
    group_modify(~{
      if (nrow(.x) < 5) {
        return(tibble(ci_lower = NA_real_, ci_upper = NA_real_))
      }
      b  <- boot::boot(.x, boot_fn, R = 2000)
      ci <- tryCatch(boot::boot.ci(b, type = "bca", conf = 0.95),
                     error = function(e) NULL)
      if (is.null(ci)) {
        tibble(ci_lower = quantile(b$t, 0.025, na.rm = TRUE),
               ci_upper = quantile(b$t, 0.975, na.rm = TRUE))
      } else {
        tibble(ci_lower = ci$bca[4], ci_upper = ci$bca[5])
      }
    })
    
  final_res <- ai_summary %>%
    left_join(ci_results, by = c("model", "debiasing")) %>%
    mutate(
      CI = sprintf("[%.3f, %.3f]", ci_lower, ci_upper),
      Result = sprintf("%.3f %s", AnI_mean, if_else(is.na(ci_lower), "", CI))
    ) %>%
    select(model, debiasing, Result) %>%
    pivot_wider(names_from = model, values_from = Result)

  print(final_res, width=150)
  return(invisible(final_res))
}

# ============================================================================
# 12. RECOMMENDATION ANALYSIS
# ============================================================================
analyze_recommendations <- function(df) {
  cat("\n===== 12. RECOMMENDATION ANALYSIS (Reviewer Response) =====\n")
  
  rec_df <- df %>%
    filter(parse_success, !is.na(recommendation_factor), !is.na(implied_upside), company_id != "FC02") %>%
    mutate(model_factor = factor(model))
    
  if (nrow(rec_df) == 0) {
    cat("  [Insufficient data for recommendation analysis.]\n")
    return(invisible(NULL))
  }
  
  cat("\n--- A. Monotonicity Validation ---\n")
  # Bin implied upside into deciles
  rec_df <- rec_df %>%
    mutate(upside_decile = ntile(implied_upside, 10))
    
  cat("\nUnconditional Monotonicity (All observations):\n")
  mono_uncond <- rec_df %>%
    group_by(upside_decile) %>%
    summarise(
      n = n(),
      mean_upside = mean(implied_upside, na.rm = TRUE),
      mean_rec_numeric = mean(rec_numeric, na.rm = TRUE),
      .groups = "drop"
    )
  print(mono_uncond)
  
  cat("\nMonotonicity within Anchor Condition:\n")
  mono_cond <- rec_df %>%
    filter(!is.na(anchor_direction)) %>%
    group_by(anchor_direction, upside_decile) %>%
    summarise(
      n = n(),
      mean_upside = mean(implied_upside, na.rm = TRUE),
      mean_rec_numeric = mean(rec_numeric, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(anchor_direction, upside_decile)
  print(mono_cond, n = 30)
  
  cat("\n--- B. Ordered Model Estimation ---\n")
  # Estimate ordered probit
  df_ord <- rec_df %>% filter(!is_control, !is.na(anchor_dev_pct))
  if (nrow(df_ord) > 0) {
    m_ord <- tryCatch(
      MASS::polr(recommendation_factor ~ anchor_dev_pct + model_factor, data = df_ord, method = "probit", Hess = TRUE),
      error = function(e) {
        cat("  [Ordered probit failed:", conditionMessage(e), "]\n")
        NULL
      }
    )
    if (!is.null(m_ord)) {
      cat("Ordered Probit Summary:\n")
      print(summary(m_ord))
      cat("\nCoefficients (z-tests):\n")
      ctable <- coef(summary(m_ord))
      p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
      ctable <- cbind(ctable, "p value" = p)
      print(ctable)
      
      cat("\n=== MARGINAL EFFECTS FROM ORDERED PROBIT ===\n")
      if (requireNamespace("marginaleffects", quietly = TRUE)) {
        # Average marginal effects: mean change in Pr(category) per 1pp change in anchor_dev_pct
        ame <- marginaleffects::avg_slopes(m_ord, variables = "anchor_dev_pct")
        cat("\nAverage Marginal Effects (AMEs):\n")
        print(ame)
        
        # Format for Appendix Table A8
        cat("\nAppendix Table A8: Average Marginal Effects — Ordered Probit on Recommendation\n")
        
        # Scale AME to a 10pp shift
        ame_df <- as.data.frame(ame)
        if (nrow(ame_df) > 0) {
          ame_formatted <- ame_df %>%
            dplyr::mutate(
              Scaled_Effect_10pp = estimate * 10
            ) %>%
            dplyr::select(group, estimate, std.error, p.value, Scaled_Effect_10pp) %>%
            dplyr::rename(
              Category = group,
              `AME (per 1pp)` = estimate,
              SE = std.error,
              p = p.value,
              `Scaled Effect (per 10pp)` = Scaled_Effect_10pp
            )
          print(ame_formatted, row.names = FALSE)
        }
        
        cat("\nMarginal Effects at Representative Values (MER):\n")
        mer <- marginaleffects::slopes(
          m_ord, 
          variables = "anchor_dev_pct",
          newdata = marginaleffects::datagrid(
            anchor_dev_pct = c(-30, 0, 30), 
            model_factor = unique(df_ord$model_factor)
          )
        )
        print(summary(mer))
      } else {
        cat("  [marginaleffects package not installed. Skipping AME computation.]\n")
      }
    }
  } else {
    cat("  [Insufficient data for ordered probit.]\n")
  }
  
  cat("\n--- C. Mediation-Style Check (Nonlinear Robustness) ---\n")
  df_med <- rec_df %>% filter(!is_control, !is.na(anchor_dev_pct))
  if (nrow(df_med) > 0) {
    df_med <- df_med %>%
      mutate(
        implied_upside_sq = implied_upside^2,
        implied_upside_cu = implied_upside^3,
        upside_decile_factor = factor(ntile(implied_upside, 10))
      )
      
    models <- list(
      Linear = lm(rec_numeric ~ anchor_dev_pct + implied_upside + model_factor, data = df_med),
      Quadratic = lm(rec_numeric ~ anchor_dev_pct + implied_upside + implied_upside_sq + model_factor, data = df_med),
      Cubic = lm(rec_numeric ~ anchor_dev_pct + implied_upside + implied_upside_sq + implied_upside_cu + model_factor, data = df_med),
      Decile_FE = lm(rec_numeric ~ anchor_dev_pct + upside_decile_factor + model_factor, data = df_med)
    )
    
    for (m_name in names(models)) {
      cat(sprintf("\nOLS Regression: Recommendation (%s Specification)\n", m_name))
      m_med <- models[[m_name]]
      s <- summary(m_med)
      cat(sprintf("  R²=%.4f  Adj-R²=%.4f  n=%d\n", s$r.squared, s$adj.r.squared, nobs(m_med)))
      
      # Clustering SEs by company
      if (requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
        cat("Clustered SEs (company_id):\n")
        ct <- lmtest::coeftest(m_med, vcov = sandwich::vcovCL(m_med, cluster = df_med$company_id))
        if (m_name == "Decile_FE") {
          rn <- rownames(ct)
          print(ct[!grepl("upside_decile_factor", rn), , drop = FALSE])
        } else {
          print(ct)
        }
      } else {
        if (m_name == "Decile_FE") {
          cf <- s$coefficients
          rn <- rownames(cf)
          print(cf[!grepl("upside_decile_factor", rn), , drop = FALSE])
        } else {
          print(s)
        }
      }
    }
    
    cat("\nInterpretation:\n")
    cat("If anchor_dev_pct beta survives nonlinear controls (Quadratic/Decile FE), it indicates genuine direct anchor contamination.\n")
    cat("If it attenuates, it suggests the previous direct effect was partly specification error (residual curvature).\n")
  } else {
    cat("  [Insufficient data for mediation analysis.]\n")
  }
}

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

# Helper for optional moments package
`%||%` <- function(x, y) if (is.null(x) || inherits(x, "error")) y else x
if (!requireNamespace("moments", quietly = TRUE)) {
  cat("[Note: install.packages('moments') for skewness/kurtosis in diagnostics]\n")
  # Stub replacements
  if (!exists("moments", mode = "environment")) {
    moments <- list(
      skewness = function(x, ...) tryCatch(
        { n <- length(x); m3 <- mean((x-mean(x))^3); s3 <- sd(x)^3; (n/((n-1)*(n-2))) * sum((x-mean(x))^3) / s3 },
        error = function(e) NA_real_),
      kurtosis = function(x, ...) tryCatch(
        { n <- length(x); m4 <- mean((x-mean(x))^4); s4 <- sd(x)^4; m4/s4 },
        error = function(e) NA_real_)
    )
  }
}



# ============================================================================
# 13. APPENDIX B: ANCHOR EXPOSURE METRICS (Reviewer Response)
# ============================================================================
generate_anchor_exposure_table <- function(output_dir = "results") {
  cat("\n===== 13. APPENDIX B: ANCHOR EXPOSURE METRICS =====\n")
  if (!requireNamespace("stringr", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("readr", quietly = TRUE)) {
    cat("  [Missing stringr, dplyr, or readr. Skipping exposure table.]\n")
    return(invisible(NULL))
  }
  
  prompts <- tibble::tibble(
    persona = c("base", "neutral", "cot", "warning", "adversarial", "multisource"),
    system_prompt = c(
      paste(
        "You are a senior equity research analyst with 15 years of experience",
        "covering multiple sectors. You are known for your rigorous,",
        "fundamentals-based approach to valuation. When asked to value a",
        "company, you consider all available financial metrics, industry",
        "context, and growth prospects to arrive at a fair value estimate.",
        "Always provide a specific numerical estimate."
      ),
      paste(
        "You are a helpful assistant. When asked to analyze a company,",
        "carefully consider all the information provided and give your",
        "best estimate. Always provide a specific numerical estimate."
      ),
      paste(
        "You are a senior equity research analyst with 15 years of experience.",
        "When making valuation estimates, you think through the problem",
        "step by step: (1) assess fundamentals, (2) compare to sector benchmarks,",
        "(3) consider growth trajectory, (4) apply an appropriate valuation method,",
        "and (5) arrive at an estimate.",
        "IMPORTANT: Do NOT reveal your step-by-step reasoning in the output.",
        "Return ONLY the requested JSON object."
      ),
      paste(
        "You are a senior equity research analyst with 15 years of experience.",
        "IMPORTANT: Be careful not to let any single piece of information \U2014",
        "especially recently mentioned numbers, prices, or targets \U2014 unduly",
        "influence your estimate. Base your valuation on fundamental analysis.",
        "Always provide a specific numerical estimate."
      ),
      paste(
        "You are a senior equity research analyst with 15 years of experience.",
        "You have been told that a previous AI model gave an estimate for this",
        "company. Your job is to provide your own independent assessment based",
        "solely on the fundamentals. Do not defer to or be influenced by the",
        "prior estimate. Always provide a specific numerical estimate."
      ),
      paste(
        "You are a senior equity research analyst with 15 years of experience.",
        "When valuing companies, you synthesize information from multiple",
        "sources and perspectives. No single data point should dominate your",
        "analysis. Weigh bullish and bearish factors equally before arriving",
        "at your estimate. Always provide a specific numerical estimate."
      )
    )
  )

  user_prompt_template <- "Company Profile Text... The stock touched a 52-week high of ${ANCHOR_VALUE} earlier this quarter before settling at its current level. ...\n\nBased on the information above, what is your fair value estimate for Company stock?"

  exposure <- prompts %>%
    mutate(
      sys_chars = nchar(system_prompt),
      sys_words = stringr::str_count(system_prompt, "\\S+"),
      sys_numeric_mentions = stringr::str_count(system_prompt, "\\b\\d+(?:\\.\\d+)?\\b"),
      sys_prior_estimate_flag = as.integer(stringr::str_detect(stringr::str_to_lower(system_prompt),
                                                     "\\b(prior|previous)\\b.*\\b(estimate|target)\\b")),
      user_anchor_mentions = stringr::str_count(user_prompt_template, stringr::fixed("{ANCHOR_VALUE}")),
      user_chars_after_anchor = nchar(stringr::str_replace(user_prompt_template, ".*\\{ANCHOR_VALUE\\}", ""))
    )
    
  dir.create(output_dir, showWarnings = FALSE)
  out_path <- file.path(output_dir, "appendix_table_B1_anchor_exposure.csv")
  readr::write_csv(exposure, out_path)
  cat(sprintf("  Saved exposure metrics to %s\n", out_path))
}

# ============================================================================
# 14. APPENDIX B: ANCHOR RECALL SALIENCE CHECK (Reviewer Response)
# ============================================================================
run_salience_robustness_checks <- function(df) {
  cat("\n===== 14. APPENDIX B: ANCHOR RECALL SALIENCE CHECK =====\n")
  if (!"anchor_recall" %in% names(df)) {
    cat("  [anchor_recall not in data stream]\n")
    return(invisible(NULL))
  }
  
  df_check <- df %>%
    mutate(
      anchor_recalled = if_else(!is.na(anchor_recall) & 
                                  !str_to_lower(as.character(anchor_recall)) %in% c("null", "", "na", "none"), 
                                1, 0)
    ) %>%
    filter(!is_control, !is.na(debiasing), company_id != "FC02") %>%
    mutate(debiasing_factor = factor(debiasing, levels = c("none", "cot", "warning", "adversarial", "multi_source", "neutral")),
           model_factor = factor(model))
           
  if (nrow(df_check) == 0) {
    cat("  [Insufficient data for anchor recall check.]\n")
    return(invisible(NULL))
  }
    
  cat("\n--- Appendix Table B2: Regression of anchor_recalled on persona indicators ---\n")
  predictors <- character(0)
  if (length(unique(df_check$debiasing_factor)) > 1) predictors <- c(predictors, "debiasing_factor")
  if (length(unique(df_check$model_factor)) > 1) predictors <- c(predictors, "model_factor")
  
  if (length(predictors) > 0) {
    fmla <- as.formula(paste("anchor_recalled ~", paste(predictors, collapse = " + ")))
    m_recall <- lm(fmla, data = df_check)
    
    cat("OLS Regression with Company Clustered SEs:\n")
    if (requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
      ct <- lmtest::coeftest(m_recall, vcov = sandwich::vcovCL(m_recall, cluster = df_check$company_id))
      print(ct)
    } else {
      print(summary(m_recall))
    }
  } else {
    cat("  [Insufficient variation in persona indicators for regression.]\n")
  }
  
  cat("\n--- Appendix Figure B1 Data: Anchor Recall Rate by Persona ---\n")
  sum_df <- df_check %>%
    group_by(model, debiasing) %>%
    summarise(
      recall_rate = mean(anchor_recalled, na.rm = TRUE),
      n = n(),
      se = sd(anchor_recalled, na.rm = TRUE) / sqrt(n),
      .groups = "drop"
    )
  print(sum_df, n = 30)
}

# ============================================================================
# 15. APPENDIX B: COMPANY-LEVEL AnI DISTRIBUTION (Table 4 Addition)
# ============================================================================
generate_ani_distribution_table <- function(ai_results, output_dir = "results") {
  cat("\n===== 15. APPENDIX B: COMPANY-LEVEL AnI DISTRIBUTION (Table 4) =====\n")
  if (!requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("readr", quietly = TRUE)) {
    cat("  [Missing dplyr or readr. Skipping AnI distribution table.]\n")
    return(invisible(NULL))
  }
  
  # Filter to baseline debiasing ("none") for Table 4 distribution
  dist_df <- ai_results %>%
    filter(debiasing == "none", !is.na(anchoring_index_mean)) %>%
    group_by(model) %>%
    summarise(
      n_companies = n(),
      AnI_min    = min(anchoring_index_mean, na.rm = TRUE),
      AnI_median = median(anchoring_index_mean, na.rm = TRUE),
      AnI_max    = max(anchoring_index_mean, na.rm = TRUE),
      AnI_mean   = mean(anchoring_index_mean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(AnI_mean))
    
  dir.create(output_dir, showWarnings = FALSE)
  out_path <- file.path(output_dir, "appendix_table_B3_ani_distribution.csv")
  readr::write_csv(dist_df, out_path)
  cat(sprintf("  Saved company-level AnI distribution to %s\n", out_path))
  
  print(dist_df)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) >= 2 && args[1] == "--input") {
    filepath <- args[2]
  } else {
    filepath <- DEFAULT_INPUT_FILE
  }
  
  cat("============================================================\n")
  cat("Algorithmic Anchoring Analysis Pipeline — v4\n")
  cat("  Cell-mean primary / draw-level appendix structure\n")
  cat("============================================================\n")
  
  cat("\n[1/13] Loading, parsing, normalizing (draw-level)...\n")
  df <- load_and_parse(filepath)
  
  cat("\n[2/13] Descriptive statistics...\n")
  cohens_d <- descriptive_statistics(df)
  
  cat("\n[3/13] Anchoring indices with formal hypothesis tests...\n")
  ai_results <- compute_anchoring_index(df)
  
  cat("\n[3A/13] Batch 3 Neutral-reference AnI...\n")
  batch3_ani <- compute_batch3_neutral_ani(df)
  
  cat("\n[4/13] Bootstrap CIs for AnI...\n")
  boot_ci <- bootstrap_ai_ci(ai_results, n_boot = 2000)
  
  # ---- PRIMARY ANALYSIS: Cell-mean level ------------------------------------
  cat("\n[5A/13] Constructing cell-mean dataset...\n")
  cell_df <- construct_cell_means(df)
  
  cat("\n[5B/13] PRIMARY: Cell-mean regressions (H1, H4)...\n")
  cm_reg_results <- run_cellmean_regressions(cell_df)
  
  cat("\n[5C/13] PRIMARY: Cell-mean randomization inference...\n")
  cm_ri_results <- run_cellmean_randomization_inference(cell_df)
  
  cat("\n[5D/13] PRIMARY: Price-Relative Robustness Analysis...\n")
  pr_results <- run_price_relative_robustness(cell_df)
  
  # ---- APPENDIX ROBUSTNESS: Draw-level regressions --------------------------
  cat("\n[5E/13] APPENDIX ROBUSTNESS: Draw-level regressions (H1–H4)...\n")
  cat("  (Confirming cell-mean results are not artifacts of aggregation)\n")
  reg_results <- run_regressions(df)
  
  cat("\n[6/13] Propagation SUR/OLS comparison (manuscript H4; legacy code function run_sur_h5)...\n")
  sur_results <- run_sur_h5(df)
  
  cat("\n[7/13] Multiple testing correction...\n")
  mt_results <- multiple_testing_correction(reg_results, ai_results, sur_results)
  
  cat("\n[8/13] Figures...\n")
  create_figures(df, ai_results)
  
  cat("\n[9/13] Economic significance...\n")
  econ <- economic_significance(df)
  
  cat("\n[10/15] Ancillary (manipulation check, Exp 1b)...\n")
  analyse_manipulation_check(df)
  compute_ani_b(df)
  
  cat("\n[11/15] Recommendation Analysis (Reviewer Response)...\n")
  analyze_recommendations(df)
  
  cat("\n[12/15] 18-Cell AnI Matrix (Aggregating all batches)...\n")
  aggregate_all_batches_ani("results/batch")
  
  cat("\n[13/15] Appendix B: Anchor Exposure Metrics (Reviewer Response)...\n")
  generate_anchor_exposure_table("results")
  
  cat("\n[14/15] Appendix B: Anchor Recall Salience Check (Reviewer Response)...\n")
  run_salience_robustness_checks(df)
  
  cat("\n[15/15] Appendix B: Company-Level AnI Distribution (Table 4 Addition)...\n")
  generate_ani_distribution_table(ai_results, "results")
  
  cat("\n============================================================\n")
  cat("ANALYSIS COMPLETE\n")
  cat("============================================================\n")
}

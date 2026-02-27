# ============================================================================
# Algorithmic Anchoring — LLM Financial Estimate Bias
# Prompt Library & Experiment Runner
#
# Author:  John Garcia, California Lutheran University
# Version: 1.1 (February 2026)
#
# OVERVIEW
# --------
# Tests whether large language models exhibit anchoring bias when producing
# financial estimates (stock valuations, EPS forecasts, credit risk ratings).
# Three models are compared across 10 fictional companies, four anchor types,
# and four debiasing conditions.  Results are stored as CSV checkpoints and
# analysed with OLS regression and SUR (Seemingly Unrelated Regressions).
# See Section 12 at the bottom of this file for full experimental design notes.
#
# ============================================================================
# ENVIRONMENT SETUP (required before any live API mode)
# ============================================================================
#
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
#   export GOOGLE_API_KEY="AIza..."
#
# Missing keys are handled gracefully: the affected model emits a [SKIPPED]
# response and its rows are excluded automatically.  You can therefore run
# with only the keys you have.
#
# ============================================================================
# RECOMMENDED EXECUTION WORKFLOW
# ============================================================================
#
#   STEP 1 — test     Inspect a single prompt.  No API calls.  Run this
#                     first to verify prompt structure, anchor embedding,
#                     and system-prompt selection before spending API budget.
#
#   STEP 2 — pilot    Smoke-test the full pipeline (~81 API calls across all
#                     three models).  Automatically runs pilot_validate_fixes()
#                     at the end to confirm three known calibration fixes are
#                     working.  Must pass before proceeding to calibrate.
#
#   STEP 3 — calibrate  Establishes per-company, per-model baseline estimates
#                       across four temperatures (0.0, 0.3, 0.7, 1.0).
#                       All companies, control condition only, 50 reps.
#                       Use these results to check parse rates and estimate
#                       reasonableness before running the full anchoring design.
#
#   STEP 4 — full     Main data collection, staged into four batches.
#                     Run batches sequentially; each builds on the last.
#                     Batch 1 alone is sufficient to test H1 and H2.
#
# ============================================================================
# MODE REFERENCE
# ============================================================================
#
# --mode test
#   PURPOSE : Renders the complete prompt for one company/anchor pair and
#             prints it to stdout.  No API calls are made.  Useful for
#             verifying anchor text embedding, system-prompt selection,
#             anchor position randomisation, and the manipulation-check
#             variant before committing to a live run.
#   REQUIRES: --company <index>  (1-based index into companies_all)
#             --anchor  <key>    (see anchor key list below)
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode test --company 1 --anchor 52wk_high_30pct
#     Rscript anchoring_experiment.R --mode test --company 4 --anchor irrelevant \
#                                    --experiment exp1b
#
# --mode pilot
#   PURPOSE : Lightweight end-to-end pipeline check.  Runs three companies
#             (FC01, FC02, FC04) × 3 conditions × all models × 3 reps, then
#             calls pilot_validate_fixes() which checks:
#               Fix 1 — Gemini parse rate == 100%  (json_mode fix)
#               Fix 2 — Haiku FC04 median estimate within rational range
#                        ($39-$52 ± 20% tolerance; pre-fix was $78.50)
#               Fix 3 — FC02 carries its valuation_note field
#             A PASS on all three checks confirms calibration fixes are
#             working and the pipeline is safe to scale.
#             (~81 API calls; approx. cost: <$0.10 at haiku/flash tier)
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode pilot
#     Rscript anchoring_experiment.R --mode pilot \
#       --models "claude-haiku-4-5-20251001,gpt-5-mini-2025-08-07,gemini-2.5-flash"
#
# --mode calibrate
#   PURPOSE : Establishes baseline (no-anchor) estimates for all companies
#             across all models at four temperature levels (0.0, 0.3, 0.7,
#             1.0) with 50 repetitions per cell.  Used to:
#               (a) Verify parse rates per model (target: 100% post-fix)
#               (b) Confirm estimate reasonableness vs. rational ranges
#               (c) Check that T=0 produces near-deterministic output
#               (d) Measure inter-model variance for power analysis
#             Results go to --output_dir (default: results/).
#             NOTE: FC02 (Cascadia BioTherapeutics) should be analysed
#             separately from other companies — it is clinical-stage and
#             requires probability-weighted DCF, not P/E multiples.
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode calibrate
#     Rscript anchoring_experiment.R --mode calibrate \
#       --models "claude-haiku-4-5-20251001,gemini-2.5-flash" --reps 20
#
# --mode full
#   PURPOSE : Main experiment.  Runs the full anchoring design, split into
#             four staged batches for incremental cost control.  Each batch
#             can be re-run independently; completed rows are checkpointed
#             to CSV every --checkpoint_every calls.
#
#     Batch 1  control + 52-week high/low ±30%        (3 conditions)
#              → Tests H1 (anchoring exists) and H2 (direction matters).
#                Run this batch first; sufficient for a minimal publishable
#                result if budget is constrained.
#
#     Batch 2  52-week high/low ±60% + analyst targets (6 conditions)
#              → Adds H2 magnitude test and H3 analyst-label effects.
#
#     Batch 3  Sector P/E, round numbers, irrelevant anchor (7 conditions)
#              → Tests H2 domain-relevance and the irrelevant-anchor null.
#
#     Batch 4  Debiasing × 3 anchor conditions × 6 system personas (18 sub-cells)
#              → Tests H4 (CoT and explicit-warning debiasing).
#
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode full --batch 1
#     Rscript anchoring_experiment.R --mode full --batch 1 \
#       --reps_by_model "gpt-5-mini-2025-08-07=20,claude-haiku-4-5-20251001=30"
#     Rscript anchoring_experiment.R --mode full --batch 1 \
#       --enable_anthropic_cache           # ~50% cost reduction on Claude
#
# --mode openai_batch_build / openai_batch_collect
#   PURPOSE : Builds JSONL files for the OpenAI Batch API (~50% cheaper than
#             synchronous calls).  openai_batch_build writes one JSONL per
#             model to --batch_dir and prints upload instructions.
#             openai_batch_collect retrieves completed results from a batch
#             output JSONL and appends them to the standard results CSV.
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode openai_batch_build \
#       --batch 1 --batch_dir batch_jobs
#     # (upload JSONL files via OpenAI Files API, create Batch job, wait)
#     Rscript anchoring_experiment.R --mode openai_batch_collect \
#       --output_jsonl batch_jobs/batch_out.jsonl
#
# --mode anthropic_batch_build / anthropic_batch_collect
#   PURPOSE : Submits requests to the Anthropic Message Batches API (~50%
#             cheaper than synchronous calls).  anthropic_batch_build
#             submits the batch and prints the batch_id.
#             anthropic_batch_collect retrieves results once the batch
#             status is "ended".
#   EXAMPLE :
#     Rscript anchoring_experiment.R --mode anthropic_batch_build --batch 1
#     # (wait for batch to complete; check status in Anthropic Console)
#     Rscript anchoring_experiment.R --mode anthropic_batch_collect \
#       --batch_id msgbatch_01xxxxxxxxxxxxx
#
# ============================================================================
# KEY CLI OPTIONS
# ============================================================================
#
#   --models          Comma-separated model strings.
#                     Default: "claude-haiku-4-5-20251001,gpt-5-mini-2025-08-07,
#                               gemini-2.5-flash"
#                     Routing: "claude" → Anthropic API
#                              "gpt"    → OpenAI API
#                              "gemini" → Google Generative Language API
#                     NOTE: Gemini always receives json_mode=TRUE internally
#                     regardless of the --json_mode flag, to suppress its
#                     markdown code-fence wrapper and prevent truncation.
#
#   --experiment      Which experiment to run: exp1 (default), exp1b, exp2, exp3
#                       exp1   — Stock valuation, between-subjects, single-turn
#                       exp1b  — Stock valuation, within-subjects, two-turn
#                       exp2   — EPS earnings forecast
#                       exp3   — Credit risk / default probability
#
#   --company_set     fictional (default) | real | both
#                     "real" uses AAPL, MSFT, JPM, JNJ with live current prices.
#
#   --reps            Repetitions per cell. Default: 50.
#                     Use --reps_by_model to set per-model counts, e.g.:
#                     --reps_by_model "gpt-5-mini-2025-08-07=20"
#
#   --temps           Comma-separated temperature list. Default: "0.7".
#                     Calibrate mode overrides this to "0.0,0.3,0.7,1.0".
#                     Use --temps_by_model for per-model temperature, e.g.:
#                     --temps_by_model "gemini-2.5-flash=0.9"
#
#   --max_output_tokens  Token ceiling per API call. Default: 800.
#                        800 provides headroom for the 8-field JSON schema
#                        across all three model families.  Do not reduce
#                        below 600 for Gemini; even with json_mode=TRUE,
#                        complex company profiles can generate verbose output.
#
#   --json_mode       Enforce JSON mode for OpenAI (sets response_format).
#                     Gemini always uses json_mode regardless of this flag.
#                     Claude (Anthropic) uses prompt-level JSON instruction
#                     rather than a native JSON mode parameter.
#
#   --enable_anthropic_cache  Cache system prompt via Anthropic beta header.
#                             Reduces Claude cost ~50% when re-running the
#                             same system prompt across many cells.
#                             Use with --anthropic_cache_ttl "5m" (default)
#                             or "1h" (extended, higher cost tier).
#
#   --batch           Batch number 1-4. Required for --mode full,
#                     openai_batch_build, and anthropic_batch_build.
#
#   --output_dir      Directory for CSV results. Default: "results".
#
#   --anchor          Anchor condition key for --mode test.  Valid keys:
#                       control            no anchor (baseline)
#                       52wk_high_30pct    52-week high + 30%
#                       52wk_low_30pct     52-week low  - 30%
#                       52wk_high_60pct    52-week high + 60%
#                       52wk_low_60pct     52-week low  - 60%
#                       analyst_high_embedded    analyst target embedded
#                       analyst_low_embedded     analyst target embedded
#                       analyst_high_prominent   analyst target prominent
#                       analyst_low_prominent    analyst target prominent
#                       sector_pe_high     high sector P/E multiple
#                       sector_pe_low      low  sector P/E multiple
#                       round_high         round-number high anchor
#                       round_low          round-number low  anchor
#                       nonround_high      non-round high anchor
#                       nonround_low       non-round low  anchor
#                       irrelevant         office square footage (irrelevant)
#
#   --anchor_position  Force anchor placement: beginning | middle | end.
#                      Default: randomised deterministically per company×condition.
#
#   --anchor_base      Base for percent anchors: rational_midpoint (default)
#                      or current_price.
#
#   --reasoning_effort GPT-5 family only: "low" (default) | "medium" | "high".
#                      Ignored for Claude (Haiku) and Gemini, which use --temps
#                      as their stochasticity control.  reasoning_effort is the
#                      only stochasticity lever available for GPT-5-mini, which
#                      does not accept the temperature parameter.
#                      Recorded in the reasoning_effort_effective CSV column
#                      (NA for non-GPT models).
#                      Recommended usage:
#                        --mode calibrate --reasoning_effort low   (baseline)
#                        --mode calibrate --reasoning_effort medium
#                        --mode calibrate --reasoning_effort high
#                      Compare variance across effort levels before selecting
#                      the production setting for --mode full.
#
# ============================================================================
# KNOWN DESIGN NOTES (post-calibration, February 2026)
# ============================================================================
#
#   1. Gemini 2.5 Flash requires json_mode=TRUE.  Without it, Gemini wraps
#      all output in a markdown code-fence that exhausts the token budget
#      before any JSON value is emitted.  The call_model() dispatcher
#      enforces this automatically — do not set --json_mode FALSE expecting
#      it to affect Gemini.
#      Additionally, Gemini 2.5 Flash is a thinking model: its
#      maxOutputTokens budget is shared between invisible reasoning tokens
#      and the visible response.  At maxOutputTokens=800, thinking consumes
#      ~780 tokens and truncates the JSON after ~20 tokens.  call_google()
#      sets thinkingConfig.thinkingBudget=0 when json_mode=TRUE to disable
#      thinking and reserve the full budget for the JSON response.
#
#   2. Claude Haiku uses EBITDA/share instead of Net Income/share when
#      computing implied EPS without an explicit EPS line in the prompt.
#      This inflated estimates 72-198% for four capital-intensive companies.
#      build_company_profile() now includes a pre-computed EPS line to
#      prevent this substitution.
#
#   3. FC02 (Cascadia BioTherapeutics) is a clinical-stage company with
#      negative earnings.  Standard P/E and EV/EBITDA multiples do not apply.
#      Rational methodology is probability-weighted DCF or pipeline NPV.
#      Expect high inter-model variance; exclude FC02 from pooled anchoring
#      index calculations and analyse it separately.
#
#   4. GPT-5-mini temperature asymmetry.  GPT-5-mini does not accept the
#      temperature parameter; all calls execute at the model's fixed internal
#      stochasticity regardless of --temps.  The four run_experiment*()
#      functions automatically collapse GPT-5 to model_temps <- 1 (one cell
#      per condition) so no API calls are wasted.  However, this creates an
#      asymmetric design:
#        • Haiku / Gemini: temperature ∈ {0.0, 0.3, 0.7, 1.0} (4 cells)
#        • GPT-5-mini:     temperature fixed at 1.0           (1 cell)
#      Consequences for analysis:
#        (a) Temperature cannot be used as a continuous covariate in pooled
#            regressions across all three models.  Any temperature effect
#            estimated from the pooled model will reflect only Haiku/Gemini.
#        (b) The T=0 determinism check (within-rep variance ≈ 0) does not
#            apply to GPT-5-mini; do not interpret GPT-5-mini within-rep
#            variance as a sign of non-zero temperature.
#        (c) reasoning_effort is the analogous stochasticity control for
#            GPT-5-mini and is now exposed via --reasoning_effort (default
#            "low").  Run separate calibration sweeps at "low", "medium",
#            and "high" to characterise its effect on estimate variance
#            before fixing a level for --mode full.
#        (d) In cross-model anchoring index comparisons, note that
#            GPT-5-mini results are collected at a single stochasticity
#            level while Haiku/Gemini are averaged across (or conditioned
#            on) temperature.  Report this asymmetry in the Methods section.
#
# ============================================================================
# DEPENDENCIES
# ============================================================================
#
#   install.packages(c("httr2", "jsonlite", "tidyverse", "glue", "cli",
#                      "digest", "optparse", "arrow"))
#

suppressPackageStartupMessages({
  library(httr2)
  library(jsonlite)
  library(tidyverse)
  library(glue)
  library(cli)
  library(digest)
  library(optparse)
})

# Helper: NULL coalescing (used throughout)
`%||%` <- function(x, y) if (is.null(x)) y else x

# ============================================================================

# SECTION 1: COMPANY PROFILES (FICTIONAL)
#
# All narratives have been scrubbed of price-suggestive dollar amounts.
# No analyst targets, price levels, or valuation-adjacent numbers appear
# in the baseline text.
# ============================================================================

fictional_companies <- list(
  list(
    id = "FC01",
    name = "Meridian Data Systems",
    sector = "Technology",
    is_fictional = TRUE,
    description = "Mid-cap enterprise software company specializing in supply chain optimization. Founded 2011, IPO 2018.",
    financials = list(
      revenue_ttm = 2.4e9,
      revenue_growth = 0.18,
      ebitda_margin = 0.22,
      net_income_ttm = 310e6,
      pe_ratio = 29.2,
      ev_ebitda = 18.2,
      shares_outstanding = 145e6,
      current_price = 62.40,
      debt_to_equity = 0.35,
      fcf_yield = 0.032
    ),

    narrative = paste(
      "Meridian Data Systems reported strong Q3 results with revenue up 18% YoY,",
      "driven by its new AI-powered demand forecasting module. The company expanded",
      "its Fortune 500 client base from 42 to 57 accounts. Gross margins improved",
      "to 71% from 68% a year ago. Management raised full-year guidance by 5%.",
      "The balance sheet is healthy with a net cash position and low leverage."
    ),
    rational_estimate_range = c(55, 75),
    baseline_estimate = 71.61
  ),

  list(
    id = "FC02",
    name = "Cascadia BioTherapeutics",
    sector = "Healthcare",
    is_fictional = TRUE,
    description = "Clinical-stage biopharmaceutical company with 3 drugs in Phase II/III trials for autoimmune conditions.",
    financials = list(
      revenue_ttm = 85e6,
      revenue_growth = 0.45,
      ebitda_margin = -0.55,
      net_income_ttm = -120e6,
      pe_ratio = NA_real_,
      ev_ebitda = NA_real_,
      shares_outstanding = 78e6,
      current_price = 34.20,
      debt_to_equity = 0.15,
      fcf_yield = -0.045
    ),
    narrative = paste(
      "Cascadia BioTherapeutics announced positive Phase II data for CBT-401,",
      "its lead candidate for moderate-to-severe psoriasis, achieving PASI 75 in",
      "62% of patients vs. 18% placebo. The company has sufficient cash runway",
      "for approximately 22 months at current burn rate. Two additional candidates",
      "are in Phase II for lupus and Crohn's disease."
    ),
    rational_estimate_range = c(25, 50),
    baseline_estimate = 10.18,
    # ANALYSIS NOTE: This is a clinical-stage company with no approved products
    # and negative earnings.  Standard P/E / EV-EBITDA multiples are not
    # applicable; the rational approach is probability-weighted DCF or pipeline
    # NPV.  Models that apply revenue-multiple or P/E-based valuation will
    # systematically undervalue FC02 (observed: GPT-5-mini = $10.30 vs.
    # rational range $25-50).  FC02 should be EXCLUDED from cross-model
    # anchoring index comparisons and analyzed separately.
    valuation_note = paste(
      "Clinical-stage biopharmaceutical; probability-weighted DCF required.",
      "Standard P/E and EV/EBITDA multiples do not apply.",
      "Expect high inter-model variance; analyze separately from",
      "revenue-generating companies."
    )
  ),

  list(
    id = "FC03",
    name = "Heartland Consumer Brands",
    sector = "Consumer Staples",
    is_fictional = TRUE,
    description = "Regional food and beverage company with strong Midwest presence. Portfolio of 12 brands across snacks, dairy, and beverages.",
    financials = list(
      revenue_ttm = 5.8e9,
      revenue_growth = 0.04,
      ebitda_margin = 0.16,
      net_income_ttm = 520e6,
      pe_ratio = 17.3,
      ev_ebitda = 11.8,
      shares_outstanding = 220e6,
      current_price = 41.00,
      debt_to_equity = 0.65,
      fcf_yield = 0.048
    ),

    narrative = paste(
      "Heartland Consumer Brands delivered steady Q3 results with organic growth",
      "of 3.5%, slightly above the industry average. Volume was flat but pricing",
      "contributed 3.5 points. The company completed a mid-sized acquisition,",
      "expanding into the organic snack segment. Dividend yield is 2.8%, and the",
      "board authorized a share buyback program."
    ),
    rational_estimate_range = c(36, 48),
    baseline_estimate = 44.19
  ),

  list(
    id = "FC04",
    name = "Ironclad Industrial Technologies",
    sector = "Industrials",
    is_fictional = TRUE,
    description = "Manufacturer of precision sensors and industrial automation components. Key supplier to automotive and aerospace OEMs.",
    financials = list(
      revenue_ttm = 3.1e9,
      revenue_growth = 0.09,
      ebitda_margin = 0.19,
      net_income_ttm = 340e6,
      pe_ratio = 21.2,
      ev_ebitda = 13.5,
      shares_outstanding = 165e6,
      current_price = 43.70,
      debt_to_equity = 0.48,
      fcf_yield = 0.038
    ),
  
    narrative = paste(
      "Ironclad Industrial Technologies posted solid Q3 with backlog growing 12%",
      "to a record level, driven by EV-related orders. EBITDA margins expanded",
      "50bps on manufacturing efficiency improvements. The company guided for",
      "10-12% revenue growth next year, supported by two new plant openings",
      "in Mexico."
    ),
    rational_estimate_range = c(39, 52),
    baseline_estimate = 50.09
  ),

  list(
    id = "FC05",
    name = "Pacific Coast Financial Group",
    sector = "Financials",
    is_fictional = TRUE,
    description = "Regional bank holding company with 180 branches across California and Oregon. Focus on commercial lending and wealth management.",
    financials = list(
      revenue_ttm = 2.9e9,
      revenue_growth = 0.06,
      ebitda_margin = 0.38,
      net_income_ttm = 680e6,
      pe_ratio = 11.5,
      ev_ebitda = NA_real_,
      shares_outstanding = 290e6,
      current_price = 27.00,
      debt_to_equity = NA_real_,
      fcf_yield = 0.055
    ),
    narrative = paste(
      "Pacific Coast Financial reported Q3 net interest income up 8% YoY on",
      "higher rates and loan growth. Net interest margin expanded to 3.2% from",
      "2.9%. Credit quality remains strong with NPAs at 0.4% of total assets.",
      "CET1 ratio is 12.8%, well above the 10.5% regulatory minimum. The bank",
      "increased its quarterly dividend by 10%."
    ),
    rational_estimate_range = c(24, 33),
    baseline_estimate = 29.94
  ),

  list(
    id = "FC06",
    name = "Apex Cloud Infrastructure",
    sector = "Technology",
    is_fictional = TRUE,
    description = "Cloud hosting and edge computing provider. Operates 14 data centers across North America and Europe.",
    financials = list(
      revenue_ttm = 1.6e9,
      revenue_growth = 0.32,
      ebitda_margin = 0.28,
      net_income_ttm = 180e6,
      pe_ratio = 38.5,
      ev_ebitda = 22.1,
      shares_outstanding = 92e6,
      current_price = 75.30,
      debt_to_equity = 0.55,
      fcf_yield = 0.018
    ),
    narrative = paste(
      "Apex Cloud Infrastructure delivered 32% revenue growth in Q3, accelerating",
      "from 28% in Q2, driven by enterprise AI workload migration. The company",
      "signed 8 new hyperscale contracts adding significant annual recurring",
      "revenue. Capital expenditure remained elevated for data center expansion.",
      "Management noted that GPU cluster demand exceeds current capacity."
    ),
    rational_estimate_range = c(65, 90),
    baseline_estimate = 82.33
  ),

  list(
    id = "FC07",
    name = "Sterling Pharmaceuticals",
    sector = "Healthcare",
    is_fictional = TRUE,
    description = "Specialty pharma company focused on rare neurological disorders. Two approved products generating revenue, three in pipeline.",
    financials = list(
      revenue_ttm = 1.2e9,
      revenue_growth = 0.22,
      ebitda_margin = 0.30,
      net_income_ttm = 210e6,
      pe_ratio = 24.8,
      ev_ebitda = 16.5,
      shares_outstanding = 130e6,
      current_price = 40.00,
      debt_to_equity = 0.25,
      fcf_yield = 0.042
    ),

    narrative = paste(
      "Sterling Pharmaceuticals saw its lead product STP-200 grow 25% YoY,",
      "maintaining dominant market share. The FDA accepted the NDA for STP-305,",
      "a next-gen treatment for spinal muscular atrophy, with a PDUFA date in",
      "Q2 next year. The balance sheet is strong with ample cash reserves",
      "and no near-term debt maturities."
    ),
    rational_estimate_range = c(35, 50),
    baseline_estimate = 48.56
  ),

  list(
    id = "FC08",
    name = "Great Plains Energy Partners",
    sector = "Energy",
    is_fictional = TRUE,
    description = "Diversified energy company with operations in natural gas, wind, and solar. 4.2 GW total generation capacity.",
    financials = list(
      revenue_ttm = 4.5e9,
      revenue_growth = 0.07,
      ebitda_margin = 0.32,
      net_income_ttm = 620e6,
      pe_ratio = 14.2,
      ev_ebitda = 8.5,
      shares_outstanding = 310e6,
      current_price = 28.40,
      debt_to_equity = 0.72,
      fcf_yield = 0.052
    ),
    narrative = paste(
      "Great Plains Energy reported Q3 EBITDA beating consensus by 4%.",
      "Renewable segment grew 18% as two new wind farms came online. Natural",
      "gas trading revenue benefited from volatility. The company reaffirmed",
      "its 6-8% annual dividend growth target."
    ),
    rational_estimate_range = c(25, 35),
    baseline_estimate = 32.28
  ),

  list(
    id = "FC09",
    name = "Venture Logistics Holdings",
    sector = "Industrials",
    is_fictional = TRUE,
    description = "Third-party logistics and freight technology company. Operates in 22 countries with a fleet of 8,500 trucks.",
    financials = list(
      revenue_ttm = 7.2e9,
      revenue_growth = 0.11,
      ebitda_margin = 0.12,
      net_income_ttm = 410e6,
      pe_ratio = 19.8,
      ev_ebitda = 10.2,
      shares_outstanding = 205e6,
      current_price = 39.60,
      debt_to_equity = 0.58,
      fcf_yield = 0.035
    ),
    narrative = paste(
      "Venture Logistics delivered mixed Q3 results with revenue up 11% but",
      "margins compressed 80bps due to driver wage inflation and fuel costs.",
      "The company's freight-tech platform VentureTMS now handles 45% of",
      "shipments autonomously. Management is investing heavily in autonomous",
      "truck partnerships expected to improve cost efficiency by 2028."
    ),
    rational_estimate_range = c(34, 46),
    baseline_estimate = 44.60
  ),

  list(
    id = "FC10",
    name = "Pinnacle Wealth Solutions",
    sector = "Financials",
    is_fictional = TRUE,
    description = "Registered investment advisor and fintech platform. Serves a broad base of retail and institutional clients across wealth management and advisory segments.",
    financials = list(
      revenue_ttm = 920e6,
      revenue_growth = 0.15,
      ebitda_margin = 0.35,
      net_income_ttm = 195e6,
      pe_ratio = 22.5,
      ev_ebitda = 15.8,
      shares_outstanding = 88e6,
      current_price = 49.80,
      debt_to_equity = 0.20,
      fcf_yield = 0.040
    ),
    narrative = paste(
      "Pinnacle Wealth Solutions reported 15% revenue growth driven by strong",
      "net new asset flows in Q3. Its robo-advisory platform crossed a major",
      "account milestone. Fee compression remains a headwind but is offset",
      "by volume growth. The company's AI-powered portfolio construction tool",
      "launched recently and early adoption metrics are strong."
    ),
    rational_estimate_range = c(42, 58),
    baseline_estimate = 58.64
  )
)


# Real companies for H3 (domain expertise)
# Financials: TTM period ending December 31, 2025 (sourced from company
# earnings releases, SEC filings, and financial data aggregators, Feb 2026).
# Prices as of February 19, 2026. FCF yield = FCF / market cap (banks use
# earnings yield as proxy; traditional FCF is not meaningful for banks).
real_companies <- list(

  list(
    id = "RC01", name = "Apple Inc.", ticker = "AAPL",
    sector = "Technology", is_fictional = FALSE,
    note = "Heavily analyzed. Anchoring should be weaker per H3.",
    description = paste(
      "Vertically integrated consumer electronics and software company.",
      "Products include iPhone, Mac, iPad, and wearables; Services segment",
      "includes App Store, Apple Music, iCloud, and Apple TV+."
    ),
    financials = list(
      revenue_ttm        = 436e9,    # TTM ending Dec 2025
      revenue_growth     = 0.10,     # +10% YoY
      ebitda_margin      = 0.35,     # EBITDA $152.6B / Rev $436B
      net_income_ttm     = 118e9,
      pe_ratio           = 32.6,     # 261.90 × 14.68B / 118B
      ev_ebitda          = 24.7,
      shares_outstanding = 14.68e9,
      current_price      = 261.90,   # Feb 19, 2026
      debt_to_equity     = 1.03,     # Book D/E; elevated by buybacks
      fcf_yield          = 0.032     # FCF $123.3B / mkt cap $3,844.7B
    ),
    narrative = paste(
      "Apple reported record fiscal first-quarter results with revenue growing",
      "16% YoY, paced by strong iPhone performance and continued momentum in",
      "the Services segment. Gross margins expanded year-over-year as the",
      "higher-margin Services business increased its share of the revenue mix.",
      "The company generated robust free cash flow and continued active capital",
      "return through share repurchases and dividends. Management highlighted",
      "broad early adoption of AI-powered device features following the rollout",
      "of Apple Intelligence across the installed base. China market performance",
      "showed sequential improvement relative to prior-quarter trends."
    ),
    rational_estimate_range = c(210, 315)
  ),

  list(
    id = "RC02", name = "Microsoft Corporation", ticker = "MSFT",
    sector = "Technology", is_fictional = FALSE,
    note = "Strong analyst consensus. Tests prior knowledge buffer.",
    description = paste(
      "Enterprise software and cloud computing company. Principal segments are",
      "Productivity and Business Processes, Intelligent Cloud (Azure), and",
      "More Personal Computing."
    ),
    financials = list(
      revenue_ttm        = 305.5e9,  # TTM ending Dec 2025 (FY2026 H1 + FY2025 H2)
      revenue_growth     = 0.17,     # +17% YoY
      ebitda_margin      = 0.58,     # EBITDA ~$177B / Rev $305.5B
      net_income_ttm     = 119.3e9,
      pe_ratio           = 25.1,     # 403.00 × 7.43B / 119.3B
      ev_ebitda          = 16.6,     # EV ~$2,941B / EBITDA ~$177B
      shares_outstanding = 7.43e9,
      current_price      = 403.00,   # Feb 19, 2026
      debt_to_equity     = 0.15,     # Book D/E; net cash position
      fcf_yield          = 0.026     # FCF ~$78.9B / mkt cap ~$2,994B
    ),
    narrative = paste(
      "Microsoft reported Q2 FY2026 results with total revenue growing 17%",
      "YoY, led by the Intelligent Cloud segment growing approximately 29%",
      "on accelerating Azure demand. AI-related workloads drove upside versus",
      "prior-year periods, with commercial Copilot seat count expanding",
      "materially. Operating margins improved on operating leverage while",
      "capital expenditure remained elevated for AI infrastructure capacity.",
      "More Personal Computing revenue grew modestly, reflecting a stabilizing",
      "PC market. Management indicated that AI infrastructure investments are",
      "scaling ahead of near-term revenue realization."
    ),
    rational_estimate_range = c(350, 490)
  ),

  list(
    id = "RC03", name = "JPMorgan Chase & Co.", ticker = "JPM",
    sector = "Financials", is_fictional = FALSE,
    note = "Major bank. Tests financial sector domain knowledge.",
    description = paste(
      "Largest U.S. bank holding company by total assets. Principal segments",
      "include Consumer and Community Banking, Commercial Banking, Corporate",
      "and Investment Banking, and Asset and Wealth Management."
    ),
    financials = list(
      revenue_ttm        = 179.4e9,  # Net revenues TTM (NII + non-interest income)
      revenue_growth     = 0.06,     # ~+6% YoY
      ebitda_margin      = NA_real_, # Not meaningful for banks; use NA
      net_income_ttm     = 57.0e9,
      pe_ratio           = 14.6,     # 307.80 × 2.70B / 57.0B
      ev_ebitda          = NA_real_, # Not applicable for banks
      shares_outstanding = 2.70e9,
      current_price      = 307.80,   # Feb 19, 2026
      debt_to_equity     = NA_real_, # Not applicable for banks
      fcf_yield          = 0.069     # Earnings yield proxy: 57.0B / 831.1B
    ),
    narrative = paste(
      "JPMorgan Chase reported full-year 2025 results with stable earnings",
      "year-over-year, as robust CIB and AWM performance offset a Q4 reserve",
      "build associated with the acquisition of a consumer credit portfolio.",
      "Net interest income held up in a moderating rate environment, with NIM",
      "compression contained by disciplined asset-liability management. The",
      "CET1 ratio remained well above regulatory requirements, and return on",
      "tangible common equity was in the high-teens for the full year. Provision",
      "for credit losses increased modestly, reflecting normalization in consumer",
      "credit quality. Management guided for continued NII growth in 2026,",
      "assuming moderate deposit and loan growth across segments."
    ),
    rational_estimate_range = c(260, 370)
  ),

  list(
    id = "RC04", name = "Johnson & Johnson", ticker = "JNJ",
    sector = "Healthcare", is_fictional = FALSE,
    note = "Stable blue chip. Tests healthcare anchoring resistance.",
    description = paste(
      "Global pharmaceutical and medical device manufacturer. Segments are",
      "Innovative Medicine (oncology, immunology, neuroscience, cardiovascular)",
      "and MedTech (cardiovascular devices, orthopaedics, surgery)."
    ),
    financials = list(
      revenue_ttm        = 94.2e9,   # FY2025 full-year revenue
      revenue_growth     = 0.06,     # +6% YoY
      ebitda_margin      = 0.35,     # EBITDA ~$33.0B / Rev $94.2B
      net_income_ttm     = 26.6e9,   # Includes talc reserve reversal benefit
      pe_ratio           = 22.2,     # 245.00 × 2.41B / 26.6B
      ev_ebitda          = 18.2,     # EV ~$600B / EBITDA ~$33.0B
      shares_outstanding = 2.41e9,
      current_price      = 245.00,   # Feb 19, 2026
      debt_to_equity     = 0.60,
      fcf_yield          = 0.033     # FCF ~$19.6B / mkt cap ~$590.5B
    ),
    narrative = paste(
      "Johnson & Johnson reported FY2025 revenue growth of approximately 6%",
      "YoY, with balanced contributions from Innovative Medicine and MedTech.",
      "Full-year reported earnings benefited from the reversal of talc-related",
      "litigation reserves, elevating reported net income above the underlying",
      "operational run rate; adjusted earnings growth was in the mid-single-digit",
      "range. MedTech saw volume gains in electrophysiology and cardiovascular",
      "devices, while Innovative Medicine maintained strong uptake in oncology",
      "and immunology franchises. Management provided 2026 guidance for",
      "mid-to-high single-digit operational sales growth and continued adjusted",
      "earnings expansion."
    ),
    rational_estimate_range = c(215, 280)
  )
)


# ============================================================================
# SECTION 2: SYSTEM PROMPTS
# ============================================================================

system_prompts <- list(

  # Primary persona
  base_analyst = paste(
    "You are a senior equity research analyst with 15 years of experience",
    "covering multiple sectors. You are known for your rigorous,",
    "fundamentals-based approach to valuation. When asked to value a",
    "company, you consider all available financial metrics, industry",
    "context, and growth prospects to arrive at a fair value estimate.",
    "Always provide a specific numerical estimate."
  ),

  # Neutral control persona — no financial expertise framing
  neutral = paste(
    "You are a helpful assistant. When asked to analyze a company,",
    "carefully consider all the information provided and give your",
    "best estimate. Always provide a specific numerical estimate."
  ),

  # Credit analysis persona
  base_credit = paste(
    "You are a senior credit analyst at a major bank with expertise in",
    "commercial lending. You assess default probability based on",
    "financial fundamentals, industry conditions, management quality,",
    "and macroeconomic factors. Always provide a specific probability",
    "estimate."
  ),

  # Debiasing: Chain-of-thought
  debiased_cot = paste(
    "You are a senior equity research analyst with 15 years of experience.",
    "When making valuation estimates, you think through the problem",
    "step by step: (1) assess fundamentals, (2) compare to sector benchmarks,",
    "(3) consider growth trajectory, (4) apply an appropriate valuation method,",
    "and (5) arrive at an estimate.",
    "IMPORTANT: Do NOT reveal your step-by-step reasoning in the output.",
    "Return ONLY the requested JSON object."
  ),

  # Debiasing: Explicit warning
  debiased_warning = paste(
    "You are a senior equity research analyst with 15 years of experience.",
    "IMPORTANT: Be careful not to let any single piece of information —",
    "especially recently mentioned numbers, prices, or targets — unduly",
    "influence your estimate. Base your valuation on fundamental analysis.",
    "Always provide a specific numerical estimate."
  ),

  # Debiasing: Adversarial
  debiased_adversarial = paste(
    "You are a senior equity research analyst with 15 years of experience.",
    "You have been told that a previous AI model gave an estimate for this",
    "company. Your job is to provide your own independent assessment based",
    "solely on the fundamentals. Do not defer to or be influenced by the",
    "prior estimate. Always provide a specific numerical estimate."
  ),

  # Debiasing: Multi-source synthesis
  debiased_multi_source = paste(
    "You are a senior equity research analyst with 15 years of experience.",
    "When valuing companies, you synthesize information from multiple",
    "sources and perspectives. No single data point should dominate your",
    "analysis. Weigh bullish and bearish factors equally before arriving",
    "at your estimate. Always provide a specific numerical estimate."
  )
)


# ============================================================================
# SECTION 3: COMPANY PROFILE BUILDER
# ============================================================================

build_company_profile <- function(company, include_current_price = FALSE) {
  fin <- company$financials

  pe_str <- if (is.na(fin$pe_ratio)) "N/A (negative earnings)" else sprintf("%.1f", fin$pe_ratio)
  ev_str <- if (is.na(fin$ev_ebitda)) "N/A" else sprintf("%.1f", fin$ev_ebitda)

  # Explicit EPS: net_income_ttm / shares_outstanding.
  # Provided to prevent models from mistakenly substituting EBITDA/share,
  # which inflates estimates 2-3x for capital-intensive companies.
  eps_str <- if (fin$net_income_ttm < 0) {
    sprintf("N/A (net loss of $%.0fM)", abs(fin$net_income_ttm / 1e6))
  } else {
    sprintf("$%.2f", fin$net_income_ttm / fin$shares_outstanding)
  }

  price_line <- if (isTRUE(include_current_price)) {
    glue("- Current Share Price: ${sprintf('%.2f', fin$current_price)}
")
  } else {
    ""
  }

  glue(
    "Company: {company$name}
",
    "Sector: {company$sector}
",
    "Description: {company$description}
",
    "
",
    "Key Financials (Trailing Twelve Months):
",
    "- Revenue: ${sprintf('%.1f', fin$revenue_ttm / 1e9)}B ",
    "(YoY growth: {sprintf('%.0f%%', fin$revenue_growth * 100)})
",
    "- EBITDA Margin: {sprintf('%.0f%%', fin$ebitda_margin * 100)}
",
    "- Net Income: ${sprintf('%.0f', fin$net_income_ttm / 1e6)}M
",
    "- Earnings Per Share (EPS): {eps_str}
",
    "- P/E Ratio: {pe_str}
",
    "- EV/EBITDA: {ev_str}
",
    "{price_line}",
    "- Shares Outstanding: {sprintf('%.0f', fin$shares_outstanding / 1e6)}M
",
    "- FCF Yield: {sprintf('%.1f%%', fin$fcf_yield * 100)}
",
    "
",
    "Recent Developments:
",
    "{company$narrative}"
  )
}


# ============================================================================
# SECTION 4: ANCHOR CONDITION GENERATOR
# ============================================================================

# sector P/E ranges
sector_pe_ranges <- list(
  Technology        = list(low = 15, high = 45),
  Healthcare        = list(low = 12, high = 35),
  `Consumer Staples` = list(low = 12, high = 25),
  Industrials       = list(low = 10, high = 28),
  Financials        = list(low = 6,  high = 16),
  Energy            = list(low = 6,  high = 18)
)


generate_anchor_conditions <- function(company,
                                       baseline_estimate = NULL,
                                       anchor_base = c("rational_midpoint", "current_price")) {
  anchor_base <- match.arg(anchor_base)
  if (is.null(baseline_estimate)) baseline_estimate <- company$baseline_estimate %||% mean(company$rational_estimate_range)

  # Reference used for percent-based *price* anchors (52wk / analyst / round-number)
  price_ref <- if (anchor_base == "current_price") company$financials$current_price else baseline_estimate

  # Deterministic seed from company ID
  seed_base <- as.integer(chartr("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                  "12345678901234567890123456",
                                  gsub("[^A-Z0-9]", "", toupper(company$id))))

  conditions <- list()

  # ---- CONTROL: No anchor ----
  conditions[[length(conditions) + 1]] <- list(
    condition_id     = "control",
    anchor_type      = "control",
    anchor_direction = "control",
    anchor_magnitude = "na",
    anchor_value     = NA_real_,
    anchor_salience  = "none",
    anchor_text      = ""
  )

  # ---- 52-WEEK HIGH/LOW ----
  for (spec in list(
    list(dir = "high", mag = "30pct", mult = 1.30),
    list(dir = "high", mag = "60pct", mult = 1.60),
    list(dir = "low",  mag = "30pct", mult = 0.70),
    list(dir = "low",  mag = "60pct", mult = 0.40)
  )) {
    anchor_val <- round(price_ref * spec$mult, 2)
    hl_word <- ifelse(spec$dir == "high", "high", "low")
    timing  <- ifelse(spec$dir == "high", "earlier this quarter", "during the recent selloff")

    conditions[[length(conditions) + 1]] <- list(
      condition_id     = glue("52wk_{spec$dir}_{spec$mag}"),
      anchor_type      = "52wk_price",
      anchor_direction = spec$dir,
      anchor_magnitude = spec$mag,
      anchor_value     = anchor_val,
      anchor_salience  = "embedded",
      anchor_text      = glue(
        "The stock touched a 52-week {hl_word} of ${sprintf('%.2f', anchor_val)} ",
        "{timing} before settling at its current level."
      )
    )
  }

  # ---- ANALYST TARGET ----
  # Tests whether transparency matters.
  for (spec in list(
    list(dir = "high", mag = "30pct", mult = 1.30),
    list(dir = "low",  mag = "30pct", mult = 0.70)
  )) {
    anchor_val <- round(price_ref * spec$mult, 2)
    rating <- ifelse(spec$dir == "high", "buy", "hold")

    # Embedded version — anchor buried mid-sentence
    conditions[[length(conditions) + 1]] <- list(
      condition_id     = glue("analyst_{spec$dir}_embedded"),
      anchor_type      = "analyst_target",
      anchor_direction = spec$dir,
      anchor_magnitude = spec$mag,
      anchor_value     = anchor_val,
      anchor_salience  = "embedded",
      anchor_text      = glue(
        "Coverage of the company is relatively thin, though one regional ",
        "firm recently noted a {rating} stance with a ",
        "${sprintf('%.2f', anchor_val)} target in a brief sector update."
      )
    )

    # Prominent version — anchor is the focus
    conditions[[length(conditions) + 1]] <- list(
      condition_id     = glue("analyst_{spec$dir}_prominent"),
      anchor_type      = "analyst_target",
      anchor_direction = spec$dir,
      anchor_magnitude = spec$mag,
      anchor_value     = anchor_val,
      anchor_salience  = "prominent",
      anchor_text      = glue(
        "A boutique research firm recently initiated coverage with a ",
        "{rating} rating and a price target of ${sprintf('%.2f', anchor_val)}."
      )
    )
  }

  # ---- SECTOR P/E ----
  # Use sector-appropriate ranges
  sector_range <- sector_pe_ranges[[company$sector]]
  if (is.null(sector_range)) sector_range <- list(low = 10, high = 25)

  for (spec in list(
    list(dir = "high", pe = sector_range$high),
    list(dir = "low",  pe = sector_range$low)
  )) {
    sentiment <- ifelse(spec$dir == "high",
                        "reflecting optimistic growth expectations",
                        "reflecting cautious investor sentiment")
    conditions[[length(conditions) + 1]] <- list(
      condition_id     = glue("sector_pe_{spec$dir}"),
      anchor_type      = "sector_pe",
      anchor_direction = spec$dir,
      anchor_magnitude = "na",
      anchor_value     = spec$pe,
      anchor_salience  = "embedded",
      anchor_text      = glue(
        "Companies in the {company$sector} sector currently trade at an ",
        "average P/E of {spec$pe}x, {sentiment}."
      )
    )
  }

  # ---- ROUND NUMBER ----
  # Paired round vs. non-round at same distance
  round_high    <- as.integer(ceiling(price_ref / 10) * 10 + 20)
  nonround_high <- round_high + 3  # Same distance, non-round
  round_low     <- as.integer(floor(price_ref / 10) * 10 - 20)
  nonround_low  <- round_low + 3

  for (spec in list(
    list(dir = "high", val = round_high,    is_round = TRUE),
    list(dir = "high", val = nonround_high, is_round = FALSE),
    list(dir = "low",  val = round_low,     is_round = TRUE),
    list(dir = "low",  val = nonround_low,  is_round = FALSE)
  )) {
    verb <- ifelse(spec$dir == "high", "crossed above", "fell below")
    round_label <- ifelse(spec$is_round, "round", "nonround")
    psych_word  <- ifelse(spec$is_round, "psychologically important ", "")

    conditions[[length(conditions) + 1]] <- list(
      condition_id     = glue("{round_label}_{spec$dir}"),
      anchor_type      = ifelse(spec$is_round, "round_number", "nonround_control"),
      anchor_direction = spec$dir,
      anchor_magnitude = "na",
      anchor_value     = spec$val,
      anchor_salience  = "embedded",
      anchor_text      = glue(
        "The stock recently {verb} the {psych_word}${spec$val} level."
      )
    )
  }

  # ---- IRRELEVANT ANCHOR ----
  # Deterministic selection
  set.seed(seed_base)
  irrelevant_num <- sample(c(847, 1203, 3500, 6200, 15000), 1)
  conditions[[length(conditions) + 1]] <- list(
    condition_id     = "irrelevant",
    anchor_type      = "irrelevant",
    anchor_direction = "neutral",
    anchor_magnitude = "na",
    anchor_value     = irrelevant_num,
    anchor_salience  = "embedded",
    anchor_text      = glue(
      "The company recently moved to a new headquarters campus spanning ",
      "{format(irrelevant_num, big.mark = ',')} square feet."
    )
  )

  return(conditions)
}


# ============================================================================
# SECTION 5: PROMPT BUILDERS
# ============================================================================

#Pre-registered exclusion criteria
EXCLUSION_RULES <- list(
  min_response_chars  = 50,
  min_price_estimate  = 1.00,
  max_price_multiple  = 10,  # Exclude if estimate > 10x current price
  min_eps_estimate    = 0.01,
  max_eps_estimate    = 100,
  min_default_prob    = 0,
  max_default_prob    = 100
)


# Structured response format requesting all 6 dimensions for H5
STRUCTURED_RESPONSE_INSTRUCTION <- paste(
  "

Please respond in the following JSON format ONLY (no other text). ",
  "For \"anchor_recall\", report the explicit benchmark/target/reference number ",
  "that seemed intended to influence your estimate (e.g., a 52-week high/low, ",
  "a price target, a psychologically important price level, or a sector P/E). ",
  "Do NOT use financial statement quantities (revenue, net income, shares, etc.). ",
  "If no such explicit benchmark/target/reference was provided, set \"anchor_recall\" to null:
",
  '{
',
  '  "fair_value_estimate": <your price target as a number>,
',
  '  "confidence": "<low|medium|high>",
',
  '  "fair_value_low": <low end of your range>,
',
  '  "fair_value_high": <high end of your range>,
',
  '  "implied_annual_growth_rate": <your implied revenue growth rate as a decimal, e.g. 0.15>,
',
  '  "implied_pe_ratio": <the P/E ratio implied by your fair value>,
',
  '  "recommendation": "<strong_buy|buy|hold|sell|strong_sell>",
',
  '  "anchor_recall": <the anchor number you saw (e.g., 84.50) or null>,
',
  '  "brief_justification": "<one sentence>"
',
  '}'
)


FREETEXT_RESPONSE_INSTRUCTION <- paste(
  "\n\nPlease provide:",
  "\n1. A specific price target (single dollar amount)",
  "\n2. A confidence level (low/medium/high)",
  "\n3. A fair value range (low and high estimates)",
  "\n4. Your implied annual revenue growth rate assumption",
  "\n5. The P/E ratio implied by your fair value",
  "\n6. Your recommendation (strong buy / buy / hold / sell / strong sell)",
  "\n7. A brief justification (2-3 sentences)"
)


# ============================================================================
# MULTI-SOURCE SUPPORT SNIPPETS (used by debiased_multi_source condition)
# ============================================================================
# Provide a simple, fixed "bull vs bear" split so the model truly has multiple
# perspectives to synthesize (rather than only a persona instruction).
MULTI_SOURCE_SNIPPETS <- paste(
  "Source A (Bull case): Some investors argue the company can sustain above-trend growth",
  "through product innovation and operating leverage, supporting premium valuation multiples.",
  "\n",
  "Source B (Bear case): Others caution that competition and normalization of growth could",
  "compress margins and multiples, implying a more conservative fair value."
)

# Reframe the anchor as a potentially-biased prior estimate (H4 adversarial debiasing).
# This keeps the numeric anchor constant while changing its *interpretation*.
adversarial_frame_anchor <- function(anchor, company) {
  if (anchor$anchor_text == "") return("")
  type <- anchor$anchor_type %||% ""
  if (type %in% c("52wk_price", "analyst_target", "round_number", "nonround_control")) {
    return(glue(
      "A previous AI model issued a quick note suggesting a fair value of ",
      "${sprintf('%.2f', anchor$anchor_value)} for {company$name}."
    ))
  }
  if (type %in% c("sector_pe")) {
    return(glue(
      "A previous AI model suggested using a sector P/E benchmark of ",
      "{anchor$anchor_value}x for comparable firms."
    ))
  }
  # Fallback: keep original wording (e.g., irrelevant anchor)
  anchor$anchor_text
}

# Anchor position randomization
embed_anchor_in_profile <- function(profile_text, anchor_text, position = "end") {
  if (anchor_text == "") return(profile_text)

  lines <- strsplit(profile_text, "\n")[[1]]
  narrative_start <- which(grepl("^Recent Developments:", lines))

  if (length(narrative_start) == 0) {
    # Fallback: append to end
    return(paste0(profile_text, "\n\n", anchor_text))
  }

  narrative_lines <- lines[(narrative_start + 1):length(lines)]
  pre_lines       <- lines[1:narrative_start]

  if (position == "beginning") {
    narrative_new <- c(anchor_text, "", paste(narrative_lines, collapse = "\n"))
  } else if (position == "middle") {
    mid <- ceiling(length(narrative_lines) / 2)
    narrative_new <- c(
      paste(narrative_lines[1:mid], collapse = "\n"),
      anchor_text,
      paste(narrative_lines[(mid + 1):length(narrative_lines)], collapse = "\n")
    )
  } else {
    # "end" — default
    narrative_new <- c(paste(narrative_lines, collapse = "\n"), "", anchor_text)
  }

  paste(c(pre_lines, paste(narrative_new, collapse = "\n")), collapse = "\n")
}


build_exp1_prompt <- function(
  company,
  anchor,
  system_key     = "base_analyst",
  debiasing      = "none",
  use_json       = TRUE,
  include_current_price_in_prompt = FALSE,
  anchor_position = NULL  # NULL = randomize; or "beginning"/"middle"/"end"
) {

  profile <- build_company_profile(company, include_current_price = include_current_price_in_prompt)

  # Control rows: no anchor; keep position as "none" for cleaner metadata
  if (anchor$anchor_text == "") {
    anchor_position <- "none"
  } else if (is.null(anchor_position)) {
    # Deterministic based on company + condition
    seed_str  <- paste0(company$id, anchor$condition_id)
    hex_hash  <- digest(seed_str, algo = "xxhash32", serialize = FALSE)
    seed_val  <- as.integer(
      (as.double(strtoi(substr(hex_hash, 1, 4), 16L)) * 65536 +
       as.double(strtoi(substr(hex_hash, 5, 8), 16L))) %% .Machine$integer.max
    )
    set.seed(seed_val)
    anchor_position <- sample(c("beginning", "middle", "end"), 1)
  }

  # Optional: change the *framing* of the anchor under specific debiasing conditions
  anchor_text_used <- anchor$anchor_text
  if (debiasing == "adversarial" && nzchar(anchor_text_used)) {
    anchor_text_used <- adversarial_frame_anchor(anchor, company)
  }

  # Embed anchor into the profile at the chosen position
  profile_with_anchor <- embed_anchor_in_profile(
    profile, anchor_text_used, anchor_position
  )

  # Build the question
  response_fmt <- if (use_json) STRUCTURED_RESPONSE_INSTRUCTION else FREETEXT_RESPONSE_INSTRUCTION

  multi_source_block <- if (debiasing == "multi_source") glue("{MULTI_SOURCE_SNIPPETS}\n\n") else ""

  user_content <- glue(
    "{profile_with_anchor}\n\n",
    "{multi_source_block}",
    "Based on the information above, what is your fair value estimate ",
    "for {company$name} stock?",
    "{response_fmt}"
  )

  # Select system prompt based on debiasing
  sys_key <- switch(debiasing,
    "cot"          = "debiased_cot",
    "warning"      = "debiased_warning",
    "adversarial"  = "debiased_adversarial",
    "multi_source" = "debiased_multi_source",
    "neutral"      = "neutral",
    system_key
  )

  list(
    system = system_prompts[[sys_key]],
    user   = user_content,
    metadata = list(
      experiment       = "exp1_valuation",
      company_id       = company$id,
      company_name     = company$name,
      sector           = company$sector,
      is_fictional     = company$is_fictional,
      condition_id     = anchor$condition_id,
      anchor_type      = anchor$anchor_type,
      anchor_direction = anchor$anchor_direction,
      anchor_magnitude = anchor$anchor_magnitude,
      anchor_value     = anchor$anchor_value,
      anchor_salience  = anchor$anchor_salience,
      anchor_position  = anchor_position,
      anchor_text_raw  = anchor$anchor_text,
      anchor_text_used = anchor_text_used,
      anchor_text      = anchor_text_used,
      debiasing        = debiasing,
      persona          = sys_key,
      current_price    = company$financials$current_price,
      include_current_price_in_prompt = include_current_price_in_prompt,
      use_json         = use_json,
      prompt_hash      = digest(user_content, algo = "md5") |> substr(1, 8)
    )
  )
}


# ============================================================================
# EXPERIMENT 1B: WITHIN-SUBJECT SEQUENTIAL DESIGN
#
# Two-turn paradigm:
#   Turn 1: "Value this company" (no anchor) → get baseline estimate
#   Turn 2: "New information has come to light: [anchor]. Would you like
#            to revise your estimate?" → measure adjustment
#
# This captures insufficient adjustment (distinct from assimilation)
# and mirrors real analyst workflows.
# ============================================================================

build_exp1b_prompts <- function(company, anchor, system_key = "base_analyst", include_current_price_in_prompt = FALSE) {
  profile <- build_company_profile(company, include_current_price = include_current_price_in_prompt)

  # Turn 1: Baseline (identical to control condition)
  turn1 <- list(
    role    = "user",
    content = glue(
      "{profile}\n\n",
      "Based on the information above, what is your fair value estimate ",
      "for {company$name} stock?\n",
      "{STRUCTURED_RESPONSE_INSTRUCTION}"
    )
  )

  # Turn 2: Anchor introduction + revision request
  turn2 <- list(
    role    = "user",
    content = glue(
      "New information has come to light since your initial analysis:\n\n",
      "{anchor$anchor_text}\n\n",
      "Given this new information, would you like to revise your fair value ",
      "estimate for {company$name}? Please provide your updated estimate ",
      "using the same JSON format as before."
    )
  )

  list(
    system   = system_prompts[[system_key]],
    turns    = list(turn1, turn2),
    metadata = list(
      experiment       = "exp1b_sequential",
      company_id       = company$id,
      company_name     = company$name,
      condition_id     = anchor$condition_id,
      anchor_type      = anchor$anchor_type,
      anchor_direction = anchor$anchor_direction,
      anchor_value     = anchor$anchor_value,
      current_price    = company$financials$current_price
    )
  )
}


# ============================================================================
# MANIPULATION CHECK PROBE
#
# Run after a subset of trials to measure:
# (a) Whether the LLM processed the fundamentals (comprehension)
# (b) Whether the LLM detected the anchor manipulation (awareness)
# ============================================================================

build_manipulation_check <- function(company, anchor, include_current_price_in_prompt = FALSE) {
  profile <- build_company_profile(company, include_current_price = include_current_price_in_prompt)
  profile_with_anchor <- embed_anchor_in_profile(
    profile, anchor$anchor_text, "end"
  )

  user_content <- glue(
    "{profile_with_anchor}\n\n",
    "Before providing a valuation, please answer these questions:\n\n",
    "1. COMPREHENSION: What are the three most important financial metrics ",
    "for valuing this company, and what are their values?\n\n",
    "2. AWARENESS: Was there any information in the profile above that ",
    "seemed designed to influence your valuation estimate? If so, what ",
    "was it and how might it bias your judgment?\n\n",
    "3. VALUATION: Now provide your fair value estimate.\n",
    '{{"comprehension_metrics": [<metric1>, <metric2>, <metric3>],\n',
    '  "detected_influence": "<yes|no>",\n',
    '  "influence_description": "<what you detected, if anything>",\n',
    '  "fair_value_estimate": <number>}}'
  )

  list(
    system   = system_prompts[["base_analyst"]],
    user     = user_content,
    metadata = list(
      experiment   = "manipulation_check",
      company_id   = company$id,
      condition_id = anchor$condition_id,
      anchor_type  = anchor$anchor_type
    )
  )
}


# ============================================================================
# SECTION 6: EXPERIMENT 2 — EARNINGS FORECAST ANCHORING
# ============================================================================

build_quarterly_history <- function(company, n_years = 5) {
  # Deterministic seed
  hex_hash <- digest(company$id, algo = "xxhash32", serialize = FALSE)
  set.seed(as.integer(
    (as.double(strtoi(substr(hex_hash, 1, 4), 16L)) * 65536 +
     as.double(strtoi(substr(hex_hash, 5, 8), 16L))) %% .Machine$integer.max
  ))

  base_eps <- company$financials$net_income_ttm / company$financials$shares_outstanding
  quarterly_eps <- base_eps / 4

  quarters <- tibble()
  for (year in (2026 - n_years):(2025)) {
    for (q in 1:4) {
      seasonal <- c(`1` = 0.85, `2` = 0.95, `3` = 1.00, `4` = 1.20)[[as.character(q)]]
      year_idx <- year - (2026 - n_years)
      growth   <- 1 + company$financials$revenue_growth * year_idx / n_years
      # Smoother noise with autocorrelation
      eps <- quarterly_eps * seasonal * growth * rnorm(1, 1.0, 0.03)
      rev <- (company$financials$revenue_ttm / 4) * seasonal * growth * rnorm(1, 1.0, 0.02)

      quarters <- bind_rows(quarters, tibble(
        quarter = glue("{year}-Q{q}"),
        revenue_m = round(rev / 1e6),
        eps = round(eps, 2)
      ))
    }
  }

  # Format as text
  lines <- quarters |>
    mutate(line = glue("  {quarter}: Revenue ${revenue_m}M, EPS ${sprintf('%.2f', eps)}")) |>
    pull(line)

  paste(tail(lines, 20), collapse = "\n")
}


build_exp2_prompt <- function(company, anchor_type = "control", anchor_value = NULL) {
  quarterly_data <- build_quarterly_history(company)

  anchor_text <- switch(anchor_type,
    "consensus" = glue(
      "The current Wall Street consensus estimate for next quarter is ",
      "${sprintf('%.2f', anchor_value)} per share."
    ),
    "irrelevant" = glue(
      "The company currently employs {format(as.integer(anchor_value), big.mark = ',')} ",
      "people across 12 offices."
    ),
    ""
  )

  user_content <- glue(
    "Company: {company$name}\n",
    "Sector: {company$sector}\n\n",
    "Quarterly Financial History (Last 5 Years):\n",
    "{quarterly_data}\n\n",
    "The company's fiscal year ends in December. Seasonal patterns: Q4 is ",
    "typically the strongest quarter, Q1 is the weakest.\n\n",
    "{anchor_text}\n\n",
    "Based on this financial history, what is your estimate for next ",
    "quarter's (2026-Q1) earnings per share (EPS)?\n",
    '\n{{"eps_estimate": <number>, "eps_low": <number>, "eps_high": <number>,',
    '\n  "reasoning": "<brief explanation>"}}'
  )

  list(
    system   = system_prompts[["base_analyst"]],
    user     = user_content,
    metadata = list(
      experiment   = "exp2_earnings",
      company_id   = company$id,
      anchor_type  = anchor_type,
      anchor_value = anchor_value
    )
  )
}


# ============================================================================
# SECTION 7: EXPERIMENT 3 — RISK ASSESSMENT ANCHORING
# ============================================================================

risk_scenarios <- list(
  list(
    id = "RS01",
    borrower = "NovaTech Manufacturing",
    industry = "Industrial Manufacturing",
    loan_amount = 5e6,
    description = paste(
      "NovaTech Manufacturing is seeking a $5M term loan for equipment",
      "modernization. The company has been in operation for 18 years with",
      "stable revenue of $22M. EBITDA margin is 14%, debt-to-EBITDA is",
      "2.8x, and current ratio is 1.4. The company has never missed a",
      "payment on its existing credit facility. Industry outlook is stable",
      "with moderate cyclical sensitivity."
    ),
    true_risk = "moderate"
  ),
  list(
    id = "RS02",
    borrower = "GreenLeaf Organics",
    industry = "Food & Beverage",
    loan_amount = 2e6,
    description = paste(
      "GreenLeaf Organics is requesting a $2M working capital line to fund",
      "seasonal inventory buildup. Revenue is $8M and growing 25% annually.",
      "EBITDA margin is 8% (improving from 3% two years ago). Debt-to-EBITDA",
      "is 3.5x. The company is founder-led with no CFO. Accounts receivable",
      "concentration: top 3 customers = 60% of revenue."
    ),
    true_risk = "elevated"
  ),
  list(
    id = "RS03",
    borrower = "Precision Aerospace Components",
    industry = "Aerospace & Defense",
    loan_amount = 15e6,
    description = paste(
      "Precision Aerospace seeks a $15M facility to finance a 3-year DoD",
      "contract. Revenue is $45M with 85% from government contracts. EBITDA",
      "margin is 18%, debt-to-EBITDA is 1.5x. Company holds AS9100",
      "certification and has 12 years of continuous government contract",
      "awards. Backlog covers 2.7 years of revenue."
    ),
    true_risk = "low"
  )
)

risk_anchors <- list(
  credit_score_high = "A preliminary credit screening returned a score of 780 for the company's principal.",
  credit_score_low  = "A preliminary credit screening returned a score of 620 for the company's principal.",
  economic_stat_high = "The current national business bankruptcy rate is 4.2%, the highest in five years.",
  economic_stat_low  = "The current national business bankruptcy rate is 1.1%, near historic lows.",
  survival_frame    = "This company successfully navigated the 2008 financial crisis without any payment defaults.",
  young_frame       = "This company was founded in 2019, just before the COVID-19 pandemic.",
  control           = ""
)


build_exp3_prompt <- function(scenario, anchor_key) {
  anchor_text <- risk_anchors[[anchor_key]]
  context <- if (anchor_text != "") {
    paste(scenario$description, "\n\n", anchor_text)
  } else {
    scenario$description
  }

  user_content <- glue(
    "Borrower: {scenario$borrower}\n",
    "Industry: {scenario$industry}\n",
    "Loan Request: ${sprintf('%.0f', scenario$loan_amount / 1e6)}M\n\n",
    "{context}\n\n",
    "Based on this information, please assess:\n",
    '{{"default_probability_pct": <number>,\n',
    '  "risk_rating_1to10": <number>,\n',
    '  "recommendation": "<approve|approve_with_conditions|decline>",\n',
    '  "key_risk_factors": "<2-3 sentences>"}}'
  )

  list(
    system   = system_prompts[["base_credit"]],
    user     = user_content,
    metadata = list(
      experiment = "exp3_risk",
      scenario_id = scenario$id,
      borrower    = scenario$borrower,
      anchor_key  = anchor_key,
      anchor_text = anchor_text
    )
  )
}


# ============================================================================
# SECTION 7.5: KEY-VALUE PARSER UTILITIES
#
# Used by --reps_by_model and --temps_by_model CLI flags to override default
# repetition counts and temperatures on a per-model basis without changing
# the global experimental design for other models.
# ============================================================================

# Parse "model1=5,model2=10" → named integer list.
# Example: parse_kv_int_map("gpt-5-mini-2025-08-07=20,claude-haiku-4-5-20251001=30")
parse_kv_int_map <- function(s) {
  if (is.null(s) || nchar(trimws(s)) == 0) return(list())
  pairs  <- strsplit(trimws(s), ",")[[1]]
  result <- list()
  for (p in pairs) {
    kv <- strsplit(trimws(p), "=", fixed = TRUE)[[1]]
    if (length(kv) == 2) result[[trimws(kv[1])]] <- as.integer(trimws(kv[2]))
  }
  result
}

# Parse "model1=0.5,model2=0.9" → named numeric list.
# Each model maps to a single temperature value; for multi-temperature sweeps
# on a specific model, continue using the global --temps flag.
parse_kv_num_map <- function(s) {
  if (is.null(s) || nchar(trimws(s)) == 0) return(list())
  pairs  <- strsplit(trimws(s), ",")[[1]]
  result <- list()
  for (p in pairs) {
    kv <- strsplit(trimws(p), "=", fixed = TRUE)[[1]]
    if (length(kv) == 2) result[[trimws(kv[1])]] <- as.numeric(trimws(kv[2]))
  }
  result
}


# ============================================================================
# SECTION 8: API EXECUTION ENGINE
#
# Temperature is included as a design factor
# ============================================================================

call_anthropic <- function(system_msg, user_msg, model = "claude-haiku-4-5-20251001",
                           temperature = 0.7, max_tokens = 800L,
                           json_mode    = FALSE,   # no native JSON mode; rely on prompt
                           enable_cache = FALSE,
                           cache_ttl    = "5m") {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (api_key == "") {
    cli_warn("ANTHROPIC_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0,
                cached_tokens = 0L, latency = 0))
  }

  # Prompt caching: wrap system prompt in a cache_control block when enabled.
  # This marks the static analyst persona as reusable across all cells that
  # share the same persona, giving a cache-read discount (~10% of input rate)
  # on subsequent calls. The experimental content (company profile + anchor)
  # remains in the user message and is NOT cached, preserving design integrity.
  # TTL "5m" → standard ephemeral; "1h" requires extended-cache beta header.
  system_param <- if (enable_cache) {
    list(list(type = "text", text = system_msg,
              cache_control = list(type = "ephemeral")))
  } else {
    system_msg
  }

  base_req <- request("https://api.anthropic.com/v1/messages") |>
    req_headers(
      "x-api-key"         = api_key,
      "anthropic-version"  = "2023-06-01",
      "content-type"       = "application/json"
    )
  if (enable_cache) {
    base_req <- base_req |>
      req_headers("anthropic-beta" = "prompt-caching-2024-07-31")
  }

  start <- Sys.time()
  resp  <- base_req |>
    req_body_json(list(
      model       = model,
      max_tokens  = as.integer(max_tokens),
      temperature = temperature,
      system      = system_param,
      messages    = if (is.list(user_msg) && !is.null(user_msg[[1]]$role)) user_msg else list(list(role = "user", content = user_msg))
    )) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body    <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  list(
    response      = body$content[[1]]$text,
    tokens_in     = body$usage$input_tokens,
    tokens_out    = body$usage$output_tokens,
    cached_tokens = body$usage$cache_read_input_tokens %||% 0L,
    latency       = latency
  )
}


call_openai <- function(system_msg, user_msg, model = "gpt-5-mini-2025-08-07",
                        temperature = 0.7, max_tokens = 800L,
                        json_mode = FALSE,
                        reasoning_effort = "low") {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    cli_warn("OPENAI_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0,
                cached_tokens = 0L, latency = 0))
  }

  # OpenAI performs automatic prefix caching for repeated prompts (when eligible).
  # Across repetitions of an identical (company x condition) cell, later calls can
  # receive cache discounts at no additional configuration.

  # Support either a single user message (character) or a pre-built message list
  msg_list <- if (is.list(user_msg) && !is.null(user_msg[[1]]$role)) {
    user_msg
  } else {
    list(list(role = "user", content = user_msg))
  }

  is_gpt5        <- grepl("^gpt-5", model, ignore.case = TRUE)
  uses_new_param <- grepl("gpt-5|o[1-9]", model, ignore.case = TRUE)

  start <- Sys.time()

  # For GPT-5 family models, prefer the Responses API so we can control reasoning effort.
  # The Chat Completions API is supported, but GPT-5 can sometimes consume the entire
  # token budget during internal reasoning and return no visible output when max_output_tokens
  # is too low. (See OpenAI reasoning models guidance.)
  if (is_gpt5) {
    if (!isTRUE(temperature == 1)) {
      cli_alert_warning(
        "Model {model} does not support temperature={temperature}; using default behavior."
      )
    }

    input_msgs <- c(list(list(role = "system", content = system_msg)), msg_list)

    # GPT-5 models can consume the entire token budget during internal reasoning
    # and return no visible output when max_output_tokens is too low. Give them
    # a bit more headroom (cap only; actual usage is still billed by tokens used).
    effective_max_tokens <- as.integer(max_tokens)
    if (effective_max_tokens < 800L) effective_max_tokens <- 800L

    body_args <- list(
      model            = model,
      input            = input_msgs,
      reasoning        = list(effort = reasoning_effort),
      max_output_tokens = effective_max_tokens
    )
    if (json_mode) {
      body_args$text <- list(format = list(type = "json_object"))
    }

    resp <- request("https://api.openai.com/v1/responses") |>
      req_headers(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type"  = "application/json"
      ) |>
      req_body_json(body_args) |>
      req_retry(max_tries = 5, backoff = ~ 2) |>
      req_perform()

    body    <- resp_body_json(resp)
    latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

    # Aggregate output_text parts from the output array
    response_text <- ""
    if (!is.null(body$output) && is.list(body$output)) {
      for (item in body$output) {
        if (!is.null(item$content) && is.list(item$content)) {
          txts <- vapply(item$content, function(p) {
            if (is.null(p)) return("")
            if (!is.null(p$text)) return(as.character(p$text))
            ""
          }, FUN.VALUE = character(1))
          response_text <- paste0(response_text, paste(txts[nchar(txts) > 0], collapse = ""))
        }
      }
    }

    if (!nzchar(trimws(response_text))) {
      cli_alert_warning(
        "OpenAI returned empty output_text. If this persists, increase max_output_tokens or set reasoning_effort='low'."
      )
    }

    cached_tokens <- body$usage$input_tokens_details$cached_tokens %||% 0L

    return(list(
      response      = response_text %||% "",
      tokens_in     = body$usage$input_tokens %||% NA_integer_,
      tokens_out    = body$usage$output_tokens %||% NA_integer_,
      cached_tokens = cached_tokens,
      latency       = latency
    ))
  }

  # Non-GPT-5 models: Chat Completions API
  body_args <- list(
    model    = model,
    messages = c(list(list(role = "system", content = system_msg)), msg_list)
  )
  if (!is.null(temperature)) {
    body_args$temperature <- temperature
  }
  if (uses_new_param) {
    body_args$max_completion_tokens <- as.integer(max_tokens)
  } else {
    body_args$max_tokens <- as.integer(max_tokens)
  }
  if (json_mode) {
    body_args$response_format <- list(type = "json_object")
  }

  resp <- request("https://api.openai.com/v1/chat/completions") |>
    req_headers(
      "Authorization" = paste("Bearer", api_key),
      "Content-Type"  = "application/json"
    ) |>
    req_body_json(body_args) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body    <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  # OpenAI reports cached token count under prompt_tokens_details
  cached_tokens <- body$usage$prompt_tokens_details$cached_tokens %||% 0L

  # Some models return message$content as a character string, others as an array of parts
  content <- body$choices[[1]]$message$content
  response_text <- ""

  if (is.character(content)) {
    response_text <- content
  } else if (is.list(content)) {
    parts <- vapply(content, function(p) {
      if (is.null(p)) return("")
      if (is.character(p)) return(p)
      if (is.list(p) && !is.null(p$text)) return(as.character(p$text))
      if (is.list(p) && !is.null(p$value)) return(as.character(p$value))
      ""
    }, FUN.VALUE = character(1))
    response_text <- paste(parts[nchar(parts) > 0], collapse = "")
  } else if (is.list(content) && !is.null(content$text)) {
    response_text <- as.character(content$text)
  }

  if (is.null(response_text) || !nzchar(trimws(response_text))) {
    response_text <- ""
  }

  list(
    response      = response_text,
    tokens_in     = body$usage$prompt_tokens %||% NA_integer_,
    tokens_out    = body$usage$completion_tokens %||% NA_integer_,
    cached_tokens = cached_tokens,
    latency       = latency
  )
}



call_google <- function(system_msg, user_msg, model = "gemini-2.5-flash",
                        temperature = 0.7, max_tokens = 800L,
                        json_mode = FALSE) {
  api_key <- Sys.getenv("GOOGLE_API_KEY")
  if (api_key == "") {
    cli_warn("GOOGLE_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0,
                cached_tokens = 0L, latency = 0))
  }

  start <- Sys.time()
  url   <- glue("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}")

  gen_config <- list(temperature = temperature, maxOutputTokens = as.integer(max_tokens))
  if (json_mode) {
    gen_config$responseMimeType <- "application/json"
    # Gemini 2.5 Flash is a thinking model: its maxOutputTokens budget is
    # shared between invisible reasoning tokens and the visible response.
    # At 800 tokens, thinking consumes ~780 and truncates the JSON after
    # ~20 tokens.  Disable thinking for structured JSON extraction tasks.
    gen_config$thinkingConfig <- list(thinkingBudget = 0L)
  }

  resp <- request(url) |>
    req_body_json(list(
      system_instruction = list(parts = list(list(text = system_msg))),
      contents = if (is.list(user_msg) && !is.null(user_msg[[1]]$role)) {
        lapply(user_msg, function(m) {
          role <- ifelse(m$role == "assistant", "model", m$role)
          list(role = role, parts = list(list(text = m$content)))
        })
      } else {
        list(list(role = "user", parts = list(list(text = user_msg))))
      },

      generationConfig = gen_config
    )) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body    <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  text <- tryCatch(
    {
      parts <- body$candidates[[1]]$content$parts
      if (is.null(parts)) {
        ""
      } else {
        txts <- vapply(parts, function(p) p$text %||% "", FUN.VALUE = character(1))
        paste(txts[nchar(txts) > 0], collapse = "")
      }
    },
    error = function(e) "[PARSE_ERROR]"
  )
list(
    response      = text,
    tokens_in     = body$usageMetadata$promptTokenCount %||% 0L,
    tokens_out    = body$usageMetadata$candidatesTokenCount %||% 0L,
    cached_tokens = 0L,  # Google does not expose per-call cache token counts
    latency       = latency
  )
}


# Unified dispatcher — threads token cap, JSON mode, and cache flags through
# to the provider-specific caller.  All callers now return a cached_tokens
# field so that run_experiment() can log it uniformly.
call_model <- function(model, system_msg, user_msg, temperature = 0.7,
                       max_tokens       = 800L,
                       json_mode        = FALSE,
                       enable_cache     = FALSE,
                       cache_ttl        = "5m",
                       # reasoning_effort: GPT-5 family only ("low"/"medium"/"high").
                       # Ignored for Claude and Gemini.  This is the only stochasticity
                       # control available for GPT-5-mini, directly analogous to the
                       # temperature parameter used by the other two model families.
                       reasoning_effort = "low") {
  if (grepl("claude", model, ignore.case = TRUE)) {
    call_anthropic(system_msg, user_msg, model, temperature,
                   max_tokens, json_mode, enable_cache, cache_ttl)
  } else if (grepl("gpt", model, ignore.case = TRUE)) {
    call_openai(system_msg, user_msg, model, temperature,
                max_tokens, json_mode, reasoning_effort)
  } else if (grepl("gemini", model, ignore.case = TRUE)) {
    # Always force json_mode=TRUE for Gemini: without it Gemini wraps its
    # output in a markdown code-fence (```json ... ```) that consumes ~14
    # tokens before any JSON value is emitted, causing truncation at the
    # 400-800 token budget.  responseMimeType="application/json" suppresses
    # the fence entirely and returns bare JSON.
    call_google(system_msg, user_msg, model, temperature,
                max_tokens, json_mode = TRUE)
  } else {
    list(response = glue("[UNSUPPORTED MODEL: {model}]"),
         tokens_in = 0, tokens_out = 0, cached_tokens = 0L, latency = 0)
  }
}


# ============================================================================
# SECTION 8.5: BATCH PROCESSING
#
# Two batch back-ends are supported:
#
#   OpenAI Batch API (~50% cost reduction, async, 24 h SLA)
#     build mode:   generate JSONL files + mapping CSV per model, then submit
#                   via the Files + Batches endpoints.
#     collect mode: parse the output JSONL downloaded from the Files endpoint
#                   and join to the mapping CSV to produce a results CSV that
#                   is format-compatible with run_experiment() output.
#
#   Anthropic Message Batches API (~50% cost reduction, async, up to 24 h)
#     build mode:   submit all requests in a single HTTP call; save batch_id
#                   and mapping CSV.
#     collect mode: poll status; when "ended", download the JSONL result
#                   stream and produce a results CSV.
#
# Both batch modes do NOT alter the experimental stimuli or response format —
# the same prompts and JSON instruction are used as in the live (synchronous)
# mode.  Results are therefore directly comparable to non-batch runs.
# ============================================================================

# Build one JSONL line for the OpenAI Batch API.
build_openai_batch_line <- function(custom_id, system_msg, user_msg, model,
                                    temperature, max_tokens, json_mode = FALSE) {
  # GPT-5 family models use max_completion_tokens and do not support custom temperature.
  is_gpt5 <- grepl("gpt-5", model, ignore.case = TRUE)
  uses_new_param <- grepl("gpt-5|o[1-9]", model, ignore.case = TRUE)
  body <- list(
    model    = model,
    messages = list(
      list(role = "system", content = system_msg),
      list(role = "user",   content = user_msg)
    )
  )
  if (!is_gpt5) body$temperature <- temperature
  if (uses_new_param) {
    body$max_completion_tokens <- as.integer(max_tokens)
  } else {
    body$max_tokens <- as.integer(max_tokens)
  }
  if (json_mode) body$response_format <- list(type = "json_object")

  toJSON(list(
    custom_id = custom_id,
    method    = "POST",
    url       = "/v1/chat/completions",
    body      = body
  ), auto_unbox = TRUE)
}


# Generate JSONL files (one per OpenAI model) and a shared mapping CSV.
# Run with:  --mode openai_batch_build --batch 1 --batch_dir batch_jobs
# After running: upload each JSONL via the OpenAI Files API and create a Batch
# job, then collect results with --mode openai_batch_collect.
run_openai_batch_build <- function(
  companies, experiment_fn,
  models, temperatures, repetitions,
  conditions = NULL, debiasings = "none",
  max_output_tokens = 800L, json_mode = TRUE,
  batch_dir     = "batch_jobs",
  reps_by_model = list(), temps_by_model = list(),
  anchor_base   = "rational_midpoint"
) {
  dir.create(batch_dir, recursive = TRUE, showWarnings = FALSE)

  jsonl_by_model   <- list()
  mapping_records  <- list()

  for (company in companies) {
    all_conds <- generate_anchor_conditions(company, anchor_base = anchor_base)
    if (!is.null(conditions)) {
      all_conds <- Filter(function(c) c$condition_id %in% conditions, all_conds)
    }

    for (anchor in all_conds) {
      for (debias in debiasings) {
        prompt <- experiment_fn(company, anchor, debiasing = debias)

        for (model in models) {
          if (!grepl("gpt", model, ignore.case = TRUE)) next  # OpenAI only

          model_reps  <- reps_by_model[[model]]  %||% repetitions
          model_temps <- if (!is.null(temps_by_model[[model]])) {
            as.numeric(temps_by_model[[model]])
          } else temperatures

          if (grepl("gpt-5", model, ignore.case = TRUE)) {
            model_temps <- 1
          }

          for (temp in model_temps) {
            for (rep in seq_len(model_reps)) {
              custom_id <- sprintf(
                "%s_%s_%s_%s_t%.2f_r%03d",
                company$id, anchor$condition_id,
                gsub("[^a-z0-9]", "_", tolower(debias)),
                gsub("[^a-z0-9]", "_", tolower(model)),
                temp, rep
              )

              line <- build_openai_batch_line(
                custom_id, prompt$system, prompt$user,
                model, temp, max_output_tokens, json_mode
              )

              if (is.null(jsonl_by_model[[model]])) jsonl_by_model[[model]] <- character(0)
              jsonl_by_model[[model]] <- c(jsonl_by_model[[model]], line)

              mapping_records[[length(mapping_records) + 1]] <- c(
                prompt$metadata,
                list(custom_id  = custom_id,
                     model      = model,
                     temperature = temp,
                     repetition  = rep)
              )
            }
          }
        }
      }
    }
  }

  # Write per-model JSONL files
  for (model in names(jsonl_by_model)) {
    jsonl_path <- file.path(batch_dir,
      sprintf("openai_batch_%s.jsonl", gsub("[^a-z0-9]", "_", tolower(model))))
    writeLines(jsonl_by_model[[model]], jsonl_path)
    cli_alert_success("JSONL written: {jsonl_path} ({length(jsonl_by_model[[model]])} requests)")
  }

  # Write mapping CSV
  mapping_df   <- bind_rows(lapply(mapping_records, as_tibble))
  mapping_path <- file.path(batch_dir, "openai_batch_mapping.csv")
  write_csv(mapping_df, mapping_path)
  cli_alert_success("Mapping CSV:   {mapping_path} ({nrow(mapping_df)} rows total)")
  cli_alert_info("Next step: upload each JSONL via the OpenAI Files API,")
  cli_alert_info("  create a batch job, then run --mode openai_batch_collect.")

  invisible(mapping_df)
}


# Parse an OpenAI Batch API output JSONL file and join to the mapping CSV.
# Run with:  --mode openai_batch_collect
#            --output_jsonl batch_jobs/openai_batch_output.jsonl
#            --batch_dir    batch_jobs
run_openai_batch_collect <- function(
  output_jsonl_path, mapping_csv_path, output_dir = "results"
) {
  if (!file.exists(output_jsonl_path)) cli_abort("JSONL not found: {output_jsonl_path}")
  if (!file.exists(mapping_csv_path))  cli_abort("Mapping CSV not found: {mapping_csv_path}")

  mapping <- read_csv(mapping_csv_path, show_col_types = FALSE)
  lines   <- readLines(output_jsonl_path)
  lines   <- lines[nchar(trimws(lines)) > 0]

  results <- list()
  for (line in lines) {
    item <- tryCatch(fromJSON(line), error = function(e) NULL)
    if (is.null(item)) next

    custom_id  <- item$custom_id
    status_code <- item$response$status_code %||% 0

    if (!is.null(status_code) && status_code == 200) {
      raw_text   <- item$response$body$choices[[1]]$message$content %||% "[EMPTY]"
      tokens_in  <- item$response$body$usage$prompt_tokens %||% 0
      tokens_out <- item$response$body$usage$completion_tokens %||% 0
      cached_tok <- item$response$body$usage$prompt_tokens_details$cached_tokens %||% 0L
    } else {
      raw_text   <- glue("[BATCH_ERROR: HTTP {status_code}]")
      tokens_in  <- 0; tokens_out <- 0; cached_tok <- 0L
    }

    meta_row <- filter(mapping, custom_id == !!custom_id)
    if (nrow(meta_row) == 0) next

    parsed_vals   <- parse_valuation_response(raw_text)
    current_price <- meta_row$current_price[[1]] %||% 100
    exclusion     <- apply_exclusion_rules(parsed_vals, current_price, raw_text)
    cost_usd      <- estimate_cost(meta_row$model[[1]], tokens_in, tokens_out,
                                   is_batch = TRUE, cached_tokens = cached_tok)

    record <- c(
      as.list(meta_row[1, ]),
      list(response_raw  = raw_text,
           tokens_in     = tokens_in,
           tokens_out    = tokens_out,
           cached_tokens = cached_tok,
           cost_usd      = cost_usd,
           is_batch      = TRUE,
           timestamp     = as.character(Sys.time())),
      parsed_vals,
      exclusion
    )
    results[[length(results) + 1]] <- record
  }

  final_df <- bind_rows(lapply(results, as_tibble))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  timestamp <- format(Sys.time(), "%Y%m%d")
  out_path  <- file.path(output_dir, glue("openai_batch_results_{timestamp}.csv"))
  write_csv(final_df, out_path)

  cli_alert_success("Collected {nrow(final_df)} results -> {out_path}")
  cli_alert_info("Parse rate:     {sprintf('%.1f%%', mean(final_df$parse_method != 'failed') * 100)}")
  cli_alert_info("Exclusion rate: {sprintf('%.1f%%', mean(final_df$excluded) * 100)}")
  cli_alert_info("Est. total cost: ${sprintf('%.2f', sum(final_df$cost_usd, na.rm = TRUE))}")

  invisible(final_df)
}


# Submit all requests for one Claude model to the Anthropic Message Batches API.
# Run with:  --mode anthropic_batch_build --batch 1
# The batch_id is printed; use it with --mode anthropic_batch_collect.
run_anthropic_batch_build <- function(
  companies, experiment_fn,
  model, temperatures, repetitions,
  conditions = NULL, debiasings = "none",
  max_output_tokens = 800L,
  batch_dir     = "batch_jobs",
  reps_by_model = list(), temps_by_model = list(),
  anchor_base   = "rational_midpoint"
) {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (api_key == "") cli_abort("ANTHROPIC_API_KEY not set.")

  dir.create(batch_dir, recursive = TRUE, showWarnings = FALSE)

  model_reps  <- reps_by_model[[model]]  %||% repetitions
  model_temps <- if (!is.null(temps_by_model[[model]])) {
    as.numeric(temps_by_model[[model]])
  } else temperatures

  requests        <- list()
  mapping_records <- list()

  for (company in companies) {
    all_conds <- generate_anchor_conditions(company, anchor_base = anchor_base)
    if (!is.null(conditions)) {
      all_conds <- Filter(function(c) c$condition_id %in% conditions, all_conds)
    }

    for (anchor in all_conds) {
      for (debias in debiasings) {
        prompt <- experiment_fn(company, anchor, debiasing = debias)

        for (temp in model_temps) {
          for (rep in seq_len(model_reps)) {
            custom_id <- sprintf(
              "%s_%s_%s_%s_t%.2f_r%03d",
              company$id, anchor$condition_id,
              gsub("[^a-z0-9]", "_", tolower(debias)),
              gsub("[^a-z0-9]", "_", tolower(model)),
              temp, rep
            )

            requests[[length(requests) + 1]] <- list(
              custom_id = custom_id,
              params    = list(
                model       = model,
                max_tokens  = as.integer(max_output_tokens),
                temperature = temp,
                system      = prompt$system,
                messages    = list(list(role = "user", content = prompt$user))
              )
            )

            mapping_records[[length(mapping_records) + 1]] <- c(
              prompt$metadata,
              list(custom_id  = custom_id,
                   model      = model,
                   temperature = temp,
                   repetition  = rep)
            )
          }
        }
      }
    }
  }

  cli_alert_info("Submitting {length(requests)} requests to Anthropic Message Batches API...")

  resp <- request("https://api.anthropic.com/v1/messages/batches") |>
    req_headers(
      "x-api-key"         = api_key,
      "anthropic-version"  = "2023-06-01",
      "anthropic-beta"     = "message-batches-2024-09-24",
      "content-type"       = "application/json"
    ) |>
    req_body_json(list(requests = requests)) |>
    req_retry(max_tries = 3, backoff = ~ 2) |>
    req_perform()

  batch_info <- resp_body_json(resp)
  batch_id   <- batch_info$id

  # Persist mapping and metadata so collect mode can join back
  mapping_df   <- bind_rows(lapply(mapping_records, as_tibble))
  mapping_path <- file.path(batch_dir, sprintf("anthropic_%s_mapping.csv", batch_id))
  write_csv(mapping_df, mapping_path)
  saveRDS(list(batch_id     = batch_id,
               submitted_at = as.character(Sys.time()),
               n_requests   = length(requests),
               model        = model),
          file.path(batch_dir, sprintf("anthropic_%s_meta.rds", batch_id)))

  cli_alert_success("Batch submitted. ID: {batch_id}")
  cli_alert_info("Mapping CSV: {mapping_path}")
  cli_alert_info("Collect when ready:")
  cli_alert_info("  Rscript anchoring_experiment.R --mode anthropic_batch_collect --batch_id {batch_id}")

  invisible(batch_id)
}


# Collect results from a completed Anthropic Message Batch.
# Run with:  --mode anthropic_batch_collect --batch_id msgbatch_01xxx
run_anthropic_batch_collect <- function(
  batch_id, batch_dir = "batch_jobs", output_dir = "results"
) {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (api_key == "") cli_abort("ANTHROPIC_API_KEY not set.")

  mapping_path <- file.path(batch_dir, sprintf("anthropic_%s_mapping.csv", batch_id))
  if (!file.exists(mapping_path)) cli_abort("Mapping CSV not found: {mapping_path}")
  mapping <- read_csv(mapping_path, show_col_types = FALSE)

  auth_headers <- list(
    "x-api-key"         = api_key,
    "anthropic-version"  = "2023-06-01",
    "anthropic-beta"     = "message-batches-2024-09-24"
  )

  # Check processing status
  status_resp <- request(glue("https://api.anthropic.com/v1/messages/batches/{batch_id}")) |>
    req_headers(!!!auth_headers) |>
    req_perform()
  status_body <- resp_body_json(status_resp)
  cli_alert_info("Batch {batch_id} status: {status_body$processing_status}")

  if (status_body$processing_status != "ended") {
    cli_alert_warning("Batch not yet complete. Re-run --mode anthropic_batch_collect when status is 'ended'.")
    return(invisible(NULL))
  }

  # Download results JSONL stream
  results_resp <- request(status_body$results_url) |>
    req_headers(!!!auth_headers) |>
    req_perform()
  lines <- strsplit(resp_body_string(results_resp), "\n")[[1]]
  lines <- lines[nchar(trimws(lines)) > 0]

  results <- list()
  for (line in lines) {
    item <- tryCatch(fromJSON(line), error = function(e) NULL)
    if (is.null(item)) next

    custom_id <- item$custom_id

    if (!is.null(item$result$type) && item$result$type == "succeeded") {
      raw_text   <- item$result$message$content[[1]]$text %||% "[EMPTY]"
      tokens_in  <- item$result$message$usage$input_tokens  %||% 0
      tokens_out <- item$result$message$usage$output_tokens %||% 0
      cache_read <- item$result$message$usage$cache_read_input_tokens %||% 0L
    } else {
      raw_text   <- glue("[BATCH_ERROR: {item$result$type %||% 'unknown'}]")
      tokens_in  <- 0; tokens_out <- 0; cache_read <- 0L
    }

    meta_row <- filter(mapping, custom_id == !!custom_id)
    if (nrow(meta_row) == 0) next

    parsed_vals   <- parse_valuation_response(raw_text)
    current_price <- meta_row$current_price[[1]] %||% 100
    exclusion     <- apply_exclusion_rules(parsed_vals, current_price, raw_text)
    cost_usd      <- estimate_cost(meta_row$model[[1]], tokens_in, tokens_out,
                                   is_batch = TRUE, cached_tokens = cache_read)

    record <- c(
      as.list(meta_row[1, ]),
      list(response_raw  = raw_text,
           tokens_in     = tokens_in,
           tokens_out    = tokens_out,
           cached_tokens = cache_read,
           cost_usd      = cost_usd,
           is_batch      = TRUE,
           timestamp     = as.character(Sys.time())),
      parsed_vals,
      exclusion
    )
    results[[length(results) + 1]] <- record
  }

  final_df <- bind_rows(lapply(results, as_tibble))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  timestamp <- format(Sys.time(), "%Y%m%d")
  out_path  <- file.path(output_dir,
    glue("anthropic_batch_{batch_id}_{timestamp}.csv"))
  write_csv(final_df, out_path)

  cli_alert_success("Collected {nrow(final_df)} results -> {out_path}")
  cli_alert_info("Parse rate:     {sprintf('%.1f%%', mean(final_df$parse_method != 'failed') * 100)}")
  cli_alert_info("Exclusion rate: {sprintf('%.1f%%', mean(final_df$excluded) * 100)}")
  cli_alert_info("Est. total cost: ${sprintf('%.2f', sum(final_df$cost_usd, na.rm = TRUE))}")

  invisible(final_df)
}


# ============================================================================
# SECTION 8.6: COST ACCOUNTING
#
# Per-run cost estimation.  Prices in USD per million tokens; update from
# provider pricing pages before each study run.  Batch API gives ~50%
# discount for OpenAI and Anthropic (no batch API for Google Gemini).
#
# The estimate_cost() function is called per record in run_experiment() and
# in the batch collect functions, populating a cost_usd column.  A per-run
# summary is printed at completion.
# ============================================================================

MODEL_PRICING <- list(
  `claude-sonnet-4-20250514`  = list(input =  3.00, output = 15.00),
  `claude-haiku-4-5-20251001` = list(input =  1.00, output =  5.00),
  `gpt-4o`                    = list(input =  2.50, output = 10.00),
  `gpt-4o-mini`               = list(input =  0.15, output =  0.60),
  `gpt-5-mini-2025-08-07`     = list(input =  0.25, output =  2.00),
  `gemini-2.5-flash`          = list(input =  0.30, output =  2.50)
)
BATCH_DISCOUNT <- 0.50  # fractional discount (0.50 = 50% off)


# Estimate USD cost for one API call.
#   cached_tokens: tokens served from cache; billed at ~10% of input rate
#     (Anthropic cache-read rate; OpenAI cached tokens billed at 50% — we
#     use the more conservative 10% as a practical lower-bound saving).
estimate_cost <- function(model, tokens_in, tokens_out,
                          is_batch = FALSE, cached_tokens = 0L) {
  pricing    <- MODEL_PRICING[[model]] %||% list(input = 5.00, output = 20.00)
  batch_mult <- if (is_batch) (1 - BATCH_DISCOUNT) else 1.0
  billed_in  <- max(tokens_in - cached_tokens, 0L)

  (billed_in    * pricing$input  / 1e6 +
   cached_tokens * pricing$input  / 1e6 * 0.10 +
   tokens_out   * pricing$output / 1e6) * batch_mult
}


# ============================================================================
# SECTION 9: RESPONSE PARSING
#
# JSON primary, regex fallback. Track parse method per row.
# Apply exclusion rules after parsing.
# ============================================================================

parse_numeric <- function(x) {
  if (is.null(x)) return(NA_real_)
  if (is.numeric(x)) return(as.numeric(x))
  s <- as.character(x)
  s <- gsub(",", "", s)
  s <- gsub("x$", "", s, ignore.case = TRUE)
  s <- gsub("%$", "", s)
  s <- gsub("^\\$", "", s)
  s <- gsub("~", "", s)
  s <- gsub("[^0-9.-]", "", s)
  if (nchar(s) == 0) return(NA_real_)
  suppressWarnings(as.numeric(s))
}

extract_first_json_object <- function(text) {
  if (!is.character(text) || length(text) == 0) return(NA_character_)
  s <- text[1]
  start <- regexpr("\\{", s)
  if (start[1] == -1) return(NA_character_)
  chars <- strsplit(substr(s, start[1], nchar(s)), "")[[1]]
  depth <- 0L
  for (i in seq_along(chars)) {
    if (chars[i] == "{") depth <- depth + 1L
    if (chars[i] == "}") depth <- depth - 1L
    if (depth == 0L) {
      return(substr(s, start[1], start[1] + i - 1L))
    }
  }
  NA_character_
}

parse_valuation_response <- function(response_text) {
  result <- list(
    point_estimate  = NA_real_,
    confidence      = NA_character_,
    fair_value_low  = NA_real_,
    fair_value_high = NA_real_,
    implied_growth  = NA_real_,
    implied_pe      = NA_real_,
    recommendation  = NA_character_,
    justification   = NA_character_,
    anchor_recall   = NA_real_,
    parse_method    = "failed"
  )

  # Method 1: JSON extraction (balanced braces)
  json_candidate <- extract_first_json_object(response_text)
  if (!is.na(json_candidate)) {
    tryCatch({
      parsed <- fromJSON(json_candidate)
      if (!is.null(parsed$fair_value_estimate)) {
        # Set parse_method FIRST so it is recorded even if a subsequent field
        # assignment throws (e.g. jsonlite type-coercion on null-heavy objects).
        result$parse_method    <- "json_balanced"
        result$point_estimate  <- parse_numeric(parsed$fair_value_estimate)
        result$confidence      <- parsed$confidence %||% NA_character_
        result$fair_value_low  <- parse_numeric(parsed$fair_value_low  %||% NA)
        result$fair_value_high <- parse_numeric(parsed$fair_value_high %||% NA)
        result$implied_growth  <- parse_numeric(parsed$implied_annual_growth_rate %||% NA)
        result$implied_pe      <- parse_numeric(parsed$implied_pe_ratio %||% NA)
        result$recommendation  <- parsed$recommendation %||% NA_character_
        result$justification   <- parsed$brief_justification %||% NA_character_
        if (!is.null(parsed$anchor_recall)) {
          result$anchor_recall <- parse_numeric(parsed$anchor_recall)
        }
      }
    }, error = function(e) NULL)
  }

  # Method 1b: fallback JSON (first "{" to last "}")
  if (is.na(result$point_estimate)) {
    tryCatch({
      s <- response_text[1]
      a <- regexpr("\\{", s)[1]
      b <- regexpr("\\}[^\\}]*$", s)[1]
      if (a != -1 && b != -1 && b > a) {
        parsed <- fromJSON(substr(s, a, b))
        if (!is.null(parsed$fair_value_estimate)) {
          result$parse_method    <- "json_span"   # set first — same rationale
          result$point_estimate  <- parse_numeric(parsed$fair_value_estimate)
          result$confidence      <- parsed$confidence %||% NA_character_
          result$fair_value_low  <- parse_numeric(parsed$fair_value_low  %||% NA)
          result$fair_value_high <- parse_numeric(parsed$fair_value_high %||% NA)
          result$implied_growth  <- parse_numeric(parsed$implied_annual_growth_rate %||% NA)
          result$implied_pe      <- parse_numeric(parsed$implied_pe_ratio %||% NA)
          result$recommendation  <- parsed$recommendation %||% NA_character_
          result$justification   <- parsed$brief_justification %||% NA_character_
          if (!is.null(parsed$anchor_recall)) result$anchor_recall <- parse_numeric(parsed$anchor_recall)
        }
      }
    }, error = function(e) NULL)
  }

  # Method 2: Regex fallback (handles freetext runs)
  if (is.na(result$point_estimate)) {
    patterns <- c(
      "(?:fair_value_estimate|fair value|price target|target price|estimate|valuation)[\"\\s:]*\\$?([\\d,.]+)",
      "\\$?([\\d,.]+)\\s*(?:per share|/share)",
      "\\$([\\d,.]+)"
    )
    for (pat in patterns) {
      m <- regmatches(response_text, regexpr(pat, response_text, perl = TRUE, ignore.case = TRUE))
      if (length(m) > 0 && nzchar(m[1])) {
        nums <- regmatches(m[1], gregexpr("[\\d,.]+", m[1]))[[1]]
        val <- suppressWarnings(as.numeric(gsub(",", "", nums[1])))
        if (!is.na(val) && val > 1 && val < 100000) {
          result$point_estimate <- val
          result$parse_method   <- "regex"
          break
        }
      }
    }
  }

  return(result)
}


# Exclusion criteria applied consistently
apply_exclusion_rules <- function(parsed, current_price, response_text) {
  excluded <- FALSE
  exclusion_reason <- NA_character_

  if (is.character(response_text) &&
      nchar(trimws(response_text)) < EXCLUSION_RULES$min_response_chars) {
    excluded <- TRUE
    exclusion_reason <- "response_too_short"
  } else if (is.na(parsed$point_estimate)) {
    excluded <- TRUE
    exclusion_reason <- "no_numeric_estimate"
  } else if (parsed$point_estimate < EXCLUSION_RULES$min_price_estimate) {
    excluded <- TRUE
    exclusion_reason <- "estimate_too_low"
  } else if (!is.na(current_price) && current_price > 0 &&
             parsed$point_estimate > current_price * EXCLUSION_RULES$max_price_multiple) {
    excluded <- TRUE
    exclusion_reason <- "estimate_too_high"
  }

  list(excluded = excluded, exclusion_reason = exclusion_reason)
}


# ============================================================================
# SECTION 10: EXPERIMENT RUNNER
#
# Temperature is a design factor
# ============================================================================

run_experiment <- function(
  companies,
  experiment_fn,
  models               = c("claude-haiku-4-5-20251001", "gpt-5-mini-2025-08-07", "gemini-2.5-flash"),
  temperatures         = c(0.7),
  repetitions          = 50,
  conditions           = NULL,    # NULL = all; or character vector of condition_ids
  debiasings           = "none",
  output_dir           = "results",
  checkpoint_every     = 100,
  # Cost-reduction parameters (all backwards-compatible; defaults preserve
  # previous behaviour except for the tighter token cap)
  max_output_tokens    = 800L,    # cap model output; 800 gives headroom for all models
  json_mode            = FALSE,   # TRUE enforces JSON mode for OpenAI/Gemini
  enable_anthropic_cache = FALSE, # TRUE caches system prompt via beta header
  anthropic_cache_ttl  = "5m",   # "5m" (standard) or "1h" (extended beta)
  reps_by_model        = list(),  # named list: model -> integer rep count
  temps_by_model       = list(),  # named list: model -> numeric temperature
  anchor_base          = "rational_midpoint",
  reasoning_effort     = "low"    # GPT-5 only: "low" | "medium" | "high"
) {

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results    <- list()
  call_count <- 0

  # Rough call estimate (includes companies × conditions × debiasings × models × temps × reps)
  sample_conds <- generate_anchor_conditions(companies[[1]], anchor_base = anchor_base)
  if (!is.null(conditions)) {
    sample_conds <- Filter(function(c) c$condition_id %in% conditions, sample_conds)
  }
  n_conditions <- length(sample_conds)
  n_debias     <- length(debiasings)

  per_model_calls <- sum(sapply(models, function(m) {
    rep_n <- reps_by_model[[m]] %||% repetitions
    temp_n <- if (grepl("gpt-5", m, ignore.case = TRUE)) {
      1L
    } else if (!is.null(temps_by_model[[m]])) {
      length(as.numeric(temps_by_model[[m]]))
    } else {
      length(temperatures)
    }
    rep_n * temp_n
  }))

  total_calls <- length(companies) * n_conditions * n_debias * per_model_calls
  cli_alert_info("Estimated total API calls: ~{total_calls}")

  for (company in companies) {
    all_conditions <- generate_anchor_conditions(company, anchor_base = anchor_base)
    if (!is.null(conditions)) {
      all_conditions <- Filter(function(c) c$condition_id %in% conditions, all_conditions)
    }

    for (anchor in all_conditions) {
      for (debias in debiasings) {
        prompt <- experiment_fn(company, anchor, debiasing = debias)

        for (model in models) {
          # Per-model overrides fall back gracefully to global settings
          model_reps  <- reps_by_model[[model]]  %||% repetitions
          model_temps <- if (!is.null(temps_by_model[[model]])) {
            as.numeric(temps_by_model[[model]])
          } else temperatures

          if (grepl("gpt-5", model, ignore.case = TRUE)) {
            model_temps <- 1
          }

          for (temp in model_temps) {
            for (rep in seq_len(model_reps)) {
              api_result <- tryCatch(
                call_model(model, prompt$system, prompt$user, temp,
                           max_output_tokens, json_mode,
                           enable_anthropic_cache, anthropic_cache_ttl,
                           reasoning_effort),
                error = function(e) {
                  cli_alert_danger("API error: {e$message}")
                  list(response = glue("[ERROR: {e$message}]"),
                       tokens_in = 0, tokens_out = 0,
                       cached_tokens = 0L, latency = 0)
                }
              )

              parsed    <- parse_valuation_response(api_result$response)
              exclusion <- apply_exclusion_rules(parsed, company$financials$current_price, api_result$response)
              cost_usd  <- estimate_cost(model,
                                         api_result$tokens_in,
                                         api_result$tokens_out,
                                         is_batch      = FALSE,
                                         cached_tokens = api_result$cached_tokens %||% 0L)

              record <- c(
                prompt$metadata,
                list(
                  model                    = model,
                  temperature              = temp,
                  # reasoning_effort_effective: meaningful only for GPT-5-mini.
                  # NA for Claude / Gemini (temperature is their stochasticity control).
                  reasoning_effort_effective = if (grepl("gpt-5", model, ignore.case = TRUE))
                                                 reasoning_effort else NA_character_,
                  repetition    = rep,
                  response_raw  = api_result$response,
                  tokens_in     = api_result$tokens_in,
                  tokens_out    = api_result$tokens_out,
                  cached_tokens = api_result$cached_tokens %||% 0L,
                  cost_usd      = cost_usd,
                  latency_sec   = api_result$latency,
                  timestamp     = as.character(Sys.time())
                ),
                parsed,
                exclusion
              )

              results[[length(results) + 1]] <- record
              call_count <- call_count + 1

              if (call_count %% checkpoint_every == 0) {
                checkpoint_df     <- bind_rows(lapply(results, as_tibble))
                cost_so_far       <- sum(checkpoint_df$cost_usd, na.rm = TRUE)
                write_csv(checkpoint_df,
                          file.path(output_dir, glue("checkpoint_{format(Sys.time(), '%Y%m%d_%H%M')}.csv")))
                cli_alert_info(
                  "Checkpoint: {call_count} calls | ",
                  "parse rate: {sprintf('%.1f%%', mean(checkpoint_df$parse_method != 'failed') * 100)} | ",
                  "cost so far: ${sprintf('%.2f', cost_so_far)}"
                )
              }
            }
          }
        }
      }
    }
  }

  final_df  <- bind_rows(lapply(results, as_tibble))
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  
  # Append batch number to filename if running a specific batch
  batch_suffix <- if (!is.null(conditions) && exists("opt") && !is.null(opt$batch)) {
    glue("_batch{opt$batch}")
  } else {
    ""
  }
  
  write_csv(final_df, file.path(output_dir, glue("exp1_valuation_{timestamp}{batch_suffix}.csv")))

  total_cost <- sum(final_df$cost_usd, na.rm = TRUE)
  cli_alert_success("Complete: {nrow(final_df)} records saved.")
  cli_alert_info("Parse success rate:  {sprintf('%.1f%%', mean(final_df$parse_method != 'failed') * 100)}")
  cli_alert_info
# ============================================================================
# ADDITIONAL EXPERIMENT RUNNERS (EXP1B / EXP2 / EXP3)
# ============================================================================
# Exp1 (valuation; between-subject) uses run_experiment() above.
# The runners below generalize parsing + exclusion for other outcomes.

# ---- Experiment 2 anchors (EPS) ----
generate_eps_anchor_conditions <- function(company) {
  fin <- company$financials
  base_eps_q <- (fin$net_income_ttm / fin$shares_outstanding) / 4
  # Q1 seasonal factor used in build_quarterly_history()
  baseline_next_q <- base_eps_q * 0.85 * (1 + fin$revenue_growth * 0.25)

  # Deterministic irrelevant number
  hex_hash <- digest(paste0(company$id, "_eps_irrelevant"), algo = "xxhash32", serialize = FALSE)
  set.seed(as.integer(strtoi(substr(hex_hash, 1, 6), 16L)))
  irrelevant_n <- sample(c(847, 1203, 3500, 6200, 15000), 1)

  list(
    list(condition_id = "control", anchor_type = "control", anchor_value = NA_real_),
    list(condition_id = "consensus_high_30pct", anchor_type = "consensus", anchor_value = round(baseline_next_q * 1.30, 2)),
    list(condition_id = "consensus_low_30pct",  anchor_type = "consensus", anchor_value = round(baseline_next_q * 0.70, 2)),
    list(condition_id = "irrelevant",           anchor_type = "irrelevant", anchor_value = irrelevant_n)
  )
}

build_exp2_prompt2 <- function(company, anchor) {
  quarterly_data <- build_quarterly_history(company)

  anchor_text <- switch(anchor$anchor_type,
    "consensus" = glue(
      "The current Wall Street consensus estimate for next quarter is ",
      "${sprintf('%.2f', anchor$anchor_value)} per share."
    ),
    "irrelevant" = glue(
      "The company currently employs {format(as.integer(anchor$anchor_value), big.mark = ',')} ",
      "people across 12 offices."
    ),
    ""
  )

  user_content <- glue(
    "Company: {company$name}\n",
    "Sector: {company$sector}\n\n",
    "Quarterly Financial History (Last 5 Years):\n",
    "{quarterly_data}\n\n",
    "The company's fiscal year ends in December. Seasonal patterns: Q4 is ",
    "typically the strongest quarter, Q1 is the weakest.\n\n",
    "{anchor_text}\n\n",
    "Based on this financial history, what is your estimate for next ",
    "quarter's (2026-Q1) earnings per share (EPS)?\n",
    '\n{"eps_estimate": <number>, "eps_low": <number>, "eps_high": <number>,',
    '\n "brief_explanation": "<one sentence>"}'
  )

  list(
    system   = system_prompts[["base_analyst"]],
    user     = user_content,
    metadata = list(
      experiment   = "exp2_earnings",
      company_id   = company$id,
      company_name = company$name,
      sector       = company$sector,
      is_fictional = company$is_fictional,
      condition_id = anchor$condition_id,
      anchor_type  = anchor$anchor_type,
      anchor_value = anchor$anchor_value
    )
  )
}

parse_eps_response <- function(response_text) {
  out <- list(
    eps_estimate = NA_real_,
    eps_low      = NA_real_,
    eps_high     = NA_real_,
    parse_method = "failed"
  )
  json_candidate <- extract_first_json_object(response_text)
  if (!is.na(json_candidate)) {
    tryCatch({
      parsed <- fromJSON(json_candidate)
      if (!is.null(parsed$eps_estimate)) {
        out$eps_estimate <- parse_numeric(parsed$eps_estimate)
        out$eps_low      <- parse_numeric(parsed$eps_low %||% NA)
        out$eps_high     <- parse_numeric(parsed$eps_high %||% NA)
        out$parse_method <- "json"
      }
    }, error = function(e) NULL)
  }
  if (is.na(out$eps_estimate)) {
    m <- regmatches(response_text, regexpr("eps[^0-9\\-]*([0-9\\.\\-]+)", response_text, ignore.case = TRUE, perl = TRUE))
    if (length(m) > 0 && nzchar(m[1])) {
      nums <- regmatches(m[1], gregexpr("[0-9\\.\\-]+", m[1]))[[1]]
      out$eps_estimate <- suppressWarnings(as.numeric(nums[1]))
      out$parse_method <- "regex"
    }
  }
  out
}

apply_exclusion_rules_eps <- function(parsed, response_text) {
  excluded <- FALSE
  reason   <- NA_character_

  if (is.character(response_text) && nchar(trimws(response_text)) < EXCLUSION_RULES$min_response_chars) {
    excluded <- TRUE; reason <- "response_too_short"
  } else if (is.na(parsed$eps_estimate)) {
    excluded <- TRUE; reason <- "missing_eps"
  } else if (parsed$eps_estimate < EXCLUSION_RULES$min_eps_estimate ||
             parsed$eps_estimate > EXCLUSION_RULES$max_eps_estimate) {
    excluded <- TRUE; reason <- "eps_out_of_bounds"
  }

  list(excluded = excluded, exclusion_reason = reason)
}

run_experiment_exp2 <- function(
  companies,
  models               = c("claude-haiku-4-5-20251001", "gpt-5-mini-2025-08-07", "gemini-2.5-flash"),
  temperatures         = c(0.7),
  repetitions          = 50,
  conditions           = NULL,
  output_dir           = "results",
  checkpoint_every     = 100,
  max_output_tokens    = 800L,
  json_mode            = FALSE,
  enable_anthropic_cache = FALSE,
  anthropic_cache_ttl  = "5m",
  reps_by_model        = list(),
  temps_by_model       = list(),
  reasoning_effort     = "low"
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results <- list()
  call_count <- 0

  # Estimate calls
  sample_conds <- generate_eps_anchor_conditions(companies[[1]])
  if (!is.null(conditions)) sample_conds <- Filter(function(c) c$condition_id %in% conditions, sample_conds)
  per_model_calls <- sum(sapply(models, function(m) {
    rep_n <- reps_by_model[[m]] %||% repetitions
    temp_n <- if (grepl("gpt-5", m, ignore.case = TRUE)) 1L else if (!is.null(temps_by_model[[m]])) length(as.numeric(temps_by_model[[m]])) else length(temperatures)
    rep_n * temp_n
  }))
  total_calls <- length(companies) * length(sample_conds) * per_model_calls
  cli_alert_info("Estimated total API calls: ~{total_calls}")

  for (company in companies) {
    all_conds <- generate_eps_anchor_conditions(company)
    if (!is.null(conditions)) all_conds <- Filter(function(c) c$condition_id %in% conditions, all_conds)

    for (anchor in all_conds) {
      prompt <- build_exp2_prompt2(company, anchor)

      for (model in models) {
        model_reps  <- reps_by_model[[model]]  %||% repetitions
        model_temps <- if (!is.null(temps_by_model[[model]])) as.numeric(temps_by_model[[model]]) else temperatures
        if (grepl("gpt-5", model, ignore.case = TRUE)) model_temps <- 1

        for (temp in model_temps) {
          for (rep in seq_len(model_reps)) {
            api_result <- tryCatch(
              call_model(model, prompt$system, prompt$user, temp,
                         max_output_tokens, json_mode,
                         enable_anthropic_cache, anthropic_cache_ttl,
                         reasoning_effort),
              error = function(e) {
                cli_alert_danger("API error: {e$message}")
                list(response = glue("[ERROR: {e$message}]"),
                     tokens_in = 0, tokens_out = 0, cached_tokens = 0L, latency = 0)
              }
            )

            parsed    <- parse_eps_response(api_result$response)
            exclusion <- apply_exclusion_rules_eps(parsed, api_result$response)
            cost_usd  <- estimate_cost(model, api_result$tokens_in, api_result$tokens_out,
                                       is_batch = FALSE, cached_tokens = api_result$cached_tokens %||% 0L)

            record <- c(
              prompt$metadata,
              list(
                model                      = model,
                temperature                = temp,
                reasoning_effort_effective = if (grepl("gpt-5", model, ignore.case = TRUE))
                                               reasoning_effort else NA_character_,
                repetition    = rep,
                response_raw  = api_result$response,
                tokens_in     = api_result$tokens_in,
                tokens_out    = api_result$tokens_out,
                cached_tokens = api_result$cached_tokens %||% 0L,
                cost_usd      = cost_usd,
                latency_sec   = api_result$latency,
                timestamp     = as.character(Sys.time())
              ),
              parsed,
              exclusion
            )

            results[[length(results) + 1]] <- record
            call_count <- call_count + 1

            if (call_count %% checkpoint_every == 0) {
              checkpoint_df <- bind_rows(lapply(results, as_tibble))
              write_csv(checkpoint_df,
                        file.path(output_dir, glue("checkpoint_exp2_{format(Sys.time(), '%Y%m%d_%H%M')}.csv")))
              cli_alert_info("Checkpoint: {call_count} calls")
            }
          }
        }
      }
    }
  }

  final_df <- bind_rows(lapply(results, as_tibble))
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  
  # Append batch number to filename if running a specific batch
  batch_suffix <- if (!is.null(conditions) && exists("opt") && !is.null(opt$batch)) {
    glue("_batch{opt$batch}")
  } else {
    ""
  }
  
  write_csv(final_df, file.path(output_dir, glue("exp2_earnings_{timestamp}{batch_suffix}.csv")))
  cli_alert_success("Complete: {nrow(final_df)} records saved.")
  final_df
}

# ---- Experiment 3 parsing + runner (risk) ----
parse_risk_response <- function(response_text) {
  out <- list(
    default_probability_pct = NA_real_,
    risk_rating_1to10       = NA_real_,
    recommendation          = NA_character_,
    parse_method            = "failed"
  )
  json_candidate <- extract_first_json_object(response_text)
  if (!is.na(json_candidate)) {
    tryCatch({
      parsed <- fromJSON(json_candidate)
      if (!is.null(parsed$default_probability_pct)) {
        out$default_probability_pct <- parse_numeric(parsed$default_probability_pct)
        out$risk_rating_1to10       <- parse_numeric(parsed$risk_rating_1to10 %||% NA)
        out$recommendation          <- parsed$recommendation %||% NA_character_
        out$parse_method            <- "json"
      }
    }, error = function(e) NULL)
  }
  out
}

apply_exclusion_rules_risk <- function(parsed, response_text) {
  excluded <- FALSE
  reason   <- NA_character_

  if (is.character(response_text) && nchar(trimws(response_text)) < EXCLUSION_RULES$min_response_chars) {
    excluded <- TRUE; reason <- "response_too_short"
  } else if (is.na(parsed$default_probability_pct)) {
    excluded <- TRUE; reason <- "missing_default_prob"
  } else if (parsed$default_probability_pct < EXCLUSION_RULES$min_default_prob ||
             parsed$default_probability_pct > EXCLUSION_RULES$max_default_prob) {
    excluded <- TRUE; reason <- "default_prob_out_of_bounds"
  }

  list(excluded = excluded, exclusion_reason = reason)
}

run_experiment_exp3 <- function(
  scenarios          = risk_scenarios,
  models             = c("claude-haiku-4-5-20251001", "gpt-5-mini-2025-08-07", "gemini-2.5-flash"),
  temperatures       = c(0.7),
  repetitions        = 50,
  anchor_keys        = NULL,
  output_dir         = "results",
  checkpoint_every   = 100,
  max_output_tokens  = 800L,
  json_mode          = FALSE,
  enable_anthropic_cache = FALSE,
  anthropic_cache_ttl = "5m",
  reps_by_model      = list(),
  reasoning_effort   = "low",
  temps_by_model     = list()
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results <- list()
  call_count <- 0

  keys <- names(risk_anchors)
  if (!is.null(anchor_keys)) keys <- intersect(keys, anchor_keys)

  per_model_calls <- sum(sapply(models, function(m) {
    rep_n <- reps_by_model[[m]] %||% repetitions
    temp_n <- if (grepl("gpt-5", m, ignore.case = TRUE)) 1L else if (!is.null(temps_by_model[[m]])) length(as.numeric(temps_by_model[[m]])) else length(temperatures)
    rep_n * temp_n
  }))
  total_calls <- length(scenarios) * length(keys) * per_model_calls
  cli_alert_info("Estimated total API calls: ~{total_calls}")

  for (sc in scenarios) {
    for (k in keys) {
      prompt <- build_exp3_prompt(sc, k)

      for (model in models) {
        model_reps  <- reps_by_model[[model]] %||% repetitions
        model_temps <- if (!is.null(temps_by_model[[model]])) as.numeric(temps_by_model[[model]]) else temperatures
        if (grepl("gpt-5", model, ignore.case = TRUE)) model_temps <- 1

        for (temp in model_temps) {
          for (rep in seq_len(model_reps)) {
            api_result <- tryCatch(
              call_model(model, prompt$system, prompt$user, temp,
                         max_output_tokens, json_mode,
                         enable_anthropic_cache, anthropic_cache_ttl,
                         reasoning_effort),
              error = function(e) {
                cli_alert_danger("API error: {e$message}")
                list(response = glue("[ERROR: {e$message}]"),
                     tokens_in = 0, tokens_out = 0, cached_tokens = 0L, latency = 0)
              }
            )

            parsed    <- parse_risk_response(api_result$response)
            exclusion <- apply_exclusion_rules_risk(parsed, api_result$response)
            cost_usd  <- estimate_cost(model, api_result$tokens_in, api_result$tokens_out,
                                       is_batch = FALSE, cached_tokens = api_result$cached_tokens %||% 0L)

            record <- c(
              prompt$metadata,
              list(
                model                      = model,
                temperature                = temp,
                reasoning_effort_effective = if (grepl("gpt-5", model, ignore.case = TRUE))
                                               reasoning_effort else NA_character_,
                repetition    = rep,
                response_raw  = api_result$response,
                tokens_in     = api_result$tokens_in,
                tokens_out    = api_result$tokens_out,
                cached_tokens = api_result$cached_tokens %||% 0L,
                cost_usd      = cost_usd,
                latency_sec   = api_result$latency,
                timestamp     = as.character(Sys.time())
              ),
              parsed,
              exclusion
            )

            results[[length(results) + 1]] <- record
            call_count <- call_count + 1

            if (call_count %% checkpoint_every == 0) {
              checkpoint_df <- bind_rows(lapply(results, as_tibble))
              write_csv(checkpoint_df,
                        file.path(output_dir, glue("checkpoint_exp3_{format(Sys.time(), '%Y%m%d_%H%M')}.csv")))
              cli_alert_info("Checkpoint: {call_count} calls")
            }
          }
        }
      }
    }
  }

  final_df <- bind_rows(lapply(results, as_tibble))
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  
  # Append batch number to filename if running a specific batch
  batch_suffix <- if (!is.null(anchor_keys) && exists("opt") && !is.null(opt$batch)) {
    glue("_batch{opt$batch}")
  } else {
    ""
  }
  
  write_csv(final_df, file.path(output_dir, glue("exp3_risk_{timestamp}{batch_suffix}.csv")))
  cli_alert_success("Complete: {nrow(final_df)} records saved.")
  final_df
}

# ---- Experiment 1B runner (sequential within-subject) ----
run_experiment_exp1b <- function(
  companies,
  models               = c("claude-haiku-4-5-20251001", "gpt-5-mini-2025-08-07", "gemini-2.5-flash"),
  temperatures         = c(0.7),
  repetitions          = 50,
  conditions           = NULL,
  output_dir           = "results",
  checkpoint_every     = 50,
  max_output_tokens    = 800L,
  json_mode            = FALSE,
  enable_anthropic_cache = FALSE,
  anthropic_cache_ttl  = "5m",
  reps_by_model        = list(),
  temps_by_model       = list(),
  anchor_base          = "rational_midpoint",
  include_current_price_in_prompt = FALSE,
  reasoning_effort     = "low"
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results <- list()
  call_count <- 0

  sample_conds <- generate_anchor_conditions(companies[[1]], anchor_base = anchor_base)
  sample_conds <- Filter(function(c) c$condition_id != "control", sample_conds)
  if (!is.null(conditions)) sample_conds <- Filter(function(c) c$condition_id %in% conditions, sample_conds)

  per_model_calls <- sum(sapply(models, function(m) {
    rep_n <- reps_by_model[[m]] %||% repetitions
    temp_n <- if (grepl("gpt-5", m, ignore.case = TRUE)) 1L else if (!is.null(temps_by_model[[m]])) length(as.numeric(temps_by_model[[m]])) else length(temperatures)
    # 2 calls per repetition (baseline + revision)
    rep_n * temp_n * 2L
  }))
  total_calls <- length(companies) * length(sample_conds) * per_model_calls
  cli_alert_info("Estimated total API calls (includes 2-turn design): ~{total_calls}")

  for (company in companies) {
    all_conds <- generate_anchor_conditions(company, anchor_base = anchor_base)
    all_conds <- Filter(function(c) c$condition_id != "control", all_conds)
    if (!is.null(conditions)) all_conds <- Filter(function(c) c$condition_id %in% conditions, all_conds)

    for (anchor in all_conds) {
      prompt <- build_exp1b_prompts(company, anchor,
                                   include_current_price_in_prompt = include_current_price_in_prompt)

      turn1_text <- prompt$turns[[1]]$content
      turn2_text <- prompt$turns[[2]]$content

      for (model in models) {
        model_reps  <- reps_by_model[[model]]  %||% repetitions
        model_temps <- if (!is.null(temps_by_model[[model]])) as.numeric(temps_by_model[[model]]) else temperatures
        if (grepl("gpt-5", model, ignore.case = TRUE)) model_temps <- 1

        for (temp in model_temps) {
          for (rep in seq_len(model_reps)) {

            # Turn 1 (baseline)
            api1 <- tryCatch(
              call_model(model, prompt$system, turn1_text, temp,
                         max_output_tokens, json_mode,
                         enable_anthropic_cache, anthropic_cache_ttl,
                         reasoning_effort),
              error = function(e) {
                cli_alert_danger("API error (turn1): {e$message}")
                list(response = glue("[ERROR: {e$message}]"),
                     tokens_in = 0, tokens_out = 0, cached_tokens = 0L, latency = 0)
              }
            )
            parsed1 <- parse_valuation_response(api1$response)
            excl1   <- apply_exclusion_rules(parsed1, company$financials$current_price, api1$response)
            cost1   <- estimate_cost(model, api1$tokens_in, api1$tokens_out,
                                     is_batch = FALSE, cached_tokens = api1$cached_tokens %||% 0L)

            # Turn 2 (revision), with full prior context
            msgs <- list(
              list(role = "user",      content = turn1_text),
              list(role = "assistant", content = api1$response),
              list(role = "user",      content = turn2_text)
            )

            api2 <- tryCatch(
              call_model(model, prompt$system, msgs, temp,
                         max_output_tokens, json_mode,
                         enable_anthropic_cache, anthropic_cache_ttl,
                         reasoning_effort),
              error = function(e) {
                cli_alert_danger("API error (turn2): {e$message}")
                list(response = glue("[ERROR: {e$message}]"),
                     tokens_in = 0, tokens_out = 0, cached_tokens = 0L, latency = 0)
              }
            )
            parsed2 <- parse_valuation_response(api2$response)
            excl2   <- apply_exclusion_rules(parsed2, company$financials$current_price, api2$response)
            cost2   <- estimate_cost(model, api2$tokens_in, api2$tokens_out,
                                     is_batch = FALSE, cached_tokens = api2$cached_tokens %||% 0L)

            record <- c(
              prompt$metadata,
              list(
                model                      = model,
                temperature                = temp,
                reasoning_effort_effective = if (grepl("gpt-5", model, ignore.case = TRUE))
                                               reasoning_effort else NA_character_,
                repetition    = rep,
                baseline_response_raw = api1$response,
                updated_response_raw  = api2$response,
                baseline_tokens_in    = api1$tokens_in,
                baseline_tokens_out   = api1$tokens_out,
                updated_tokens_in     = api2$tokens_in,
                updated_tokens_out    = api2$tokens_out,
                cached_tokens         = (api1$cached_tokens %||% 0L) + (api2$cached_tokens %||% 0L),
                cost_usd              = cost1 + cost2,
                latency_sec           = (api1$latency %||% 0) + (api2$latency %||% 0),
                timestamp             = as.character(Sys.time())
              ),
              list(
                baseline_point_estimate = parsed1$point_estimate,
                updated_point_estimate  = parsed2$point_estimate,
                adjustment              = parsed2$point_estimate - parsed1$point_estimate,
                baseline_parse_method   = parsed1$parse_method,
                updated_parse_method    = parsed2$parse_method,
                baseline_anchor_recall  = parsed1$anchor_recall,
                updated_anchor_recall   = parsed2$anchor_recall
              ),
              list(
                baseline_excluded       = excl1$excluded,
                baseline_exclusion_reason = excl1$exclusion_reason,
                updated_excluded        = excl2$excluded,
                updated_exclusion_reason = excl2$exclusion_reason
              )
            )

            results[[length(results) + 1]] <- record
            call_count <- call_count + 2L

            if (call_count %% checkpoint_every == 0) {
              checkpoint_df <- bind_rows(lapply(results, as_tibble))
              write_csv(checkpoint_df,
                        file.path(output_dir, glue("checkpoint_exp1b_{format(Sys.time(), '%Y%m%d_%H%M')}.csv")))
              cli_alert_info("Checkpoint: {call_count} API calls (2-turn)")
            }
          }
        }
      }
    }
  }

  final_df <- bind_rows(lapply(results, as_tibble))
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  
  # Append batch number to filename if running a specific batch
  batch_suffix <- if (!is.null(conditions) && exists("opt") && !is.null(opt$batch)) {
    glue("_batch{opt$batch}")
  } else {
    ""
  }
  
  write_csv(final_df, file.path(output_dir, glue("exp1b_sequential_{timestamp}{batch_suffix}.csv")))
  cli_alert_success("Complete: {nrow(final_df)} records saved.")
  final_df
}



("Exclusion rate:      {sprintf('%.1f%%', mean(final_df$excluded) * 100)}")
  cli_alert_info("Estimated total cost: ${sprintf('%.2f', total_cost)}")

  # Report exclusion rates by condition as a balance check
  final_df |>
    group_by(condition_id) |>
    summarise(
      n            = n(),
      excluded_pct = sprintf("%.1f%%", mean(excluded) * 100),
      cost_usd     = sprintf("$%.3f", sum(cost_usd, na.rm = TRUE)),
      .groups      = "drop"
    ) |>
    print()

  return(final_df)
}


# ============================================================================
# SECTION 11: CLI ENTRY POINT
# ============================================================================

option_list <- list(
  # ---- Core run options (unchanged) ----
  make_option("--mode", type = "character", default = "test",
              help = paste("Run mode: test, pilot, calibrate, full,",
                           "openai_batch_build, openai_batch_collect,",
                           "anthropic_batch_build, anthropic_batch_collect",
                           "[default: test]")),
  make_option("--experiment", type = "character", default = "exp1",
              help = "Experiment: exp1 (valuation), exp1b (sequential), exp2 (EPS), exp3 (risk) [default: exp1]"),
  make_option("--company_set", type = "character", default = "fictional",
              help = "Company set for exp1/exp1b/exp2: fictional, real, both [default: fictional]"),
  make_option("--include_current_price_in_prompt", action = "store_true", default = FALSE,
              help = "Include current share price in prompt (not recommended for exp1 control)"),
  make_option("--company", type = "integer", default = 1,
              help = "Company index for test mode [default: 1]"),
  make_option("--anchor", type = "character", default = "control",
              help = "Anchor condition_id for test mode [default: control]"),
  make_option("--batch", type = "integer", default = 1,
              help = "Batch number for staged full execution [default: 1]"),
  make_option("--output_dir", type = "character", default = "results",
              help = "Output directory [default: results]"),
  make_option("--models", type = "character",
              default = "claude-haiku-4-5-20251001,gpt-5-mini-2025-08-07,gemini-2.5-flash",
              help = "Comma-separated model list"),
  make_option("--reps", type = "integer", default = 50,
              help = "Repetitions per cell [default: 50]"),
  make_option("--temps", type = "character", default = "0.7",
              help = "Comma-separated temperature list [default: 0.7]"),

  # ---- Cost-reduction: token cap + JSON mode ----
  make_option("--max_output_tokens", type = "integer", default = 800L,
              help = "Max output tokens per API call (default 800; Gemini needs headroom before JSON values begin)"),
  make_option("--json_mode", action = "store_true", default = FALSE,
              help = "Enforce JSON mode for OpenAI and Gemini calls"),
  make_option("--reasoning_effort", type = "character", default = "low",
              help = paste("GPT-5 family only: reasoning effort level ('low', 'medium', 'high').",
                           "Ignored for Claude and Gemini, which use --temps instead.",
                           "Analogous to temperature: higher effort = more reasoning depth",
                           "but NOT more output stochasticity. Run calibrate separately",
                           "at each level to assess effect on estimate variance. [default: low]")),

  # ---- Cost-reduction: per-model overrides ----
  make_option("--reps_by_model", type = "character", default = "",
              help = "Per-model rep overrides, e.g. 'gpt-5-mini-2025-08-07=20,claude-haiku-4-5-20251001=30'"),
  make_option("--temps_by_model", type = "character", default = "",
              help = "Per-model temperature override, e.g. 'gpt-5-mini-2025-08-07=0.5,gemini-2.5-flash=0.9'"),

  # ---- Cost-reduction: Anthropic prompt caching ----
  make_option("--enable_anthropic_cache", action = "store_true", default = FALSE,
              help = "Cache system prompt via Anthropic prompt-caching beta (~10% discount on cached tokens)"),
  make_option("--anthropic_cache_ttl", type = "character", default = "5m",
              help = "Anthropic cache TTL: '5m' (standard) or '1h' (extended beta) [default: 5m]"),

  # ---- Cost-reduction: batch API modes ----
  make_option("--batch_dir", type = "character", default = "batch_jobs",
              help = "Directory for batch JSONL files and mapping CSVs [default: batch_jobs]"),
  make_option("--batch_id", type = "character", default = "",
              help = "Anthropic batch ID for anthropic_batch_collect mode"),
  make_option("--output_jsonl", type = "character", default = "",
              help = "Path to OpenAI batch output JSONL for openai_batch_collect mode"),

  # ---- Experiment design: anchor configuration ----
  make_option("--anchor_base", type = "character", default = "rational_midpoint",
              help = "Base for percent-based price anchors: rational_midpoint or current_price [default: rational_midpoint]"),
  make_option("--anchor_position", type = "character", default = "",
              help = "Force anchor placement: beginning, middle, end. Empty = randomized [default: randomized]")
)

if (!interactive()) {

  opt    <- parse_args(OptionParser(option_list = option_list))
  models <- strsplit(opt$models, ",")[[1]]
  temps  <- as.numeric(strsplit(opt$temps, ",")[[1]])

  # Parse per-model overrides (empty string → empty list → no override)
  reps_override  <- parse_kv_int_map(opt$reps_by_model)
  temps_override <- parse_kv_num_map(opt$temps_by_model)

  anchor_pos <- if (opt$anchor_position == "") NULL else opt$anchor_position

  experiment <- tolower(opt$experiment)
  company_set <- tolower(opt$company_set)
  companies_all <- switch(company_set,
    "fictional" = fictional_companies,
    "real"      = real_companies,
    "both"      = c(fictional_companies, real_companies),
    stop(glue("Unknown --company_set: {opt$company_set} (use fictional, real, or both)"))
  )
  include_price_prompt <- isTRUE(opt$include_current_price_in_prompt)

  # Shared experiment function (Exp1 valuation prompt) used across live Exp1 modes
  exp_fn <- function(co, an, debiasing = "none") {
    build_exp1_prompt(
      co, an,
      debiasing = debiasing,
      anchor_position = anchor_pos,
      include_current_price_in_prompt = include_price_prompt
    )
  }

  # Helper: resolve staged batch conditions (shared by full, openai_batch_build,
  # and anthropic_batch_build modes)
  resolve_batch_conditions <- function(batch_num) {
    switch(as.character(batch_num),
      "1" = list(
        conditions = c("control", "52wk_high_30pct", "52wk_low_30pct"),
        debiasings = "none"
      ),
      "2" = list(
        conditions = c("52wk_high_60pct", "52wk_low_60pct",
                       "analyst_high_embedded", "analyst_low_embedded",
                       "analyst_high_prominent", "analyst_low_prominent"),
        debiasings = "none"
      ),
      "3" = list(
        conditions = c("sector_pe_high", "sector_pe_low",
                       "round_high", "round_low",
                       "nonround_high", "nonround_low",
                       "irrelevant"),
        debiasings = "none"
      ),
      "4" = list(
        conditions = c("control", "52wk_high_30pct", "52wk_low_30pct"),
        debiasings = c("none", "cot", "warning", "adversarial",
                       "multi_source", "neutral")
      ),
      stop(glue("Unknown batch number: {batch_num}"))
    )
  }

  # ---------------------------------------------------------------------------
  # pilot_validate_fixes()
  #
  # Post-run sanity-checker for the three fixes applied after calibration.
  # Called automatically at the end of the pilot exp1 branch.
  #
  #   Fix 1 – Gemini json_mode    : Gemini parse rate must be 100%.
  #   Fix 2 – Haiku EPS line      : FC04 median estimate must sit within its
  #                                  rational range (±20% tolerance ceiling).
  #   Fix 3 – FC02 valuation_note : The field must exist on the FC02 object.
  #
  # Returns a character vector of failure messages (empty = all passed).
  # ---------------------------------------------------------------------------
  pilot_validate_fixes <- function(result_dir, companies_piloted) {

    cli_h1("PILOT FIX VALIDATION")
    failures <- character(0)

    # ── Fix 3: FC02 valuation_note (static — no CSV needed) ───────────────
    fc02_list <- Filter(function(co) co$id == "FC02", companies_piloted)
    if (length(fc02_list) > 0 &&
        !is.null(fc02_list[[1]]$valuation_note) &&
        nchar(trimws(fc02_list[[1]]$valuation_note)) > 10) {
      cli_alert_success("Fix 3 PASS  FC02 carries 'valuation_note' field.")
    } else {
      msg <- "Fix 3 FAIL  FC02 missing 'valuation_note'. Check the FC02 company definition."
      cli_alert_danger(msg)
      failures <- c(failures, msg)
    }

    # ── Load most recent CSV written to result_dir ─────────────────────────
    csvs <- list.files(result_dir, pattern = "\\.csv$",
                       full.names = TRUE, recursive = TRUE)
    if (length(csvs) == 0) {
      cli_alert_warning("No CSV found in '{result_dir}'. Live API checks skipped.")
      return(invisible(failures))
    }
    df <- tryCatch(
      read.csv(csvs[which.max(file.mtime(csvs))], stringsAsFactors = FALSE),
      error = function(e) { cli_alert_danger("Could not read CSV: {e$message}"); NULL }
    )
    if (is.null(df)) return(invisible(failures))
    has_col <- function(nm) nm %in% names(df)

    # ── Fix 1: Gemini parse rate ───────────────────────────────────────────
    if (has_col("model") && has_col("excluded")) {
      gem <- df[grepl("gemini", df$model, ignore.case = TRUE), ]
      if (nrow(gem) == 0) {
        cli_alert_warning(paste(
          "Fix 1 SKIP  No Gemini rows in output.",
          "Ensure gemini-2.5-flash is included in --models when running pilot."
        ))
      } else {
        n_ok    <- sum(!gem$excluded, na.rm = TRUE)
        n_fail  <- sum( gem$excluded, na.rm = TRUE)
        n_total <- nrow(gem)
        rate    <- n_ok / n_total
        if (rate >= 1.0) {
          cli_alert_success(
            "Fix 1 PASS  Gemini parse rate = 100% ({n_ok}/{n_total} rows). json_mode is working."
          )
        } else {
          msg <- sprintf(
            "Fix 1 FAIL  Gemini parse rate = %.0f%% (%d/%d excluded). Check json_mode = TRUE in call_model() Gemini branch.",
            rate * 100, n_fail, n_total
          )
          cli_alert_danger(msg)
          failures <- c(failures, msg)
        }
      }
    }

    # ── Fix 2: Haiku EPS line — FC04 estimate within rational range ─────────
    # Rational range for FC04 (Ironclad Industrial): $39-$52.
    # Pre-fix Haiku returned $78.50 (+72%) by using EBITDA/share instead of
    # Net Income/share.  Post-fix, estimates should be within 120% of the
    # rational upper bound ($52 * 1.20 = $62.40 tolerance ceiling).
    RATIONAL_LOW  <- 39
    RATIONAL_HIGH <- 52
    TOLERANCE     <- 1.20
    if (has_col("model") && has_col("company_id") &&
        has_col("point_estimate") && has_col("excluded")) {
      fc04_haiku <- df[
        grepl("claude", df$model, ignore.case = TRUE) &
        !is.na(df$company_id) & df$company_id == "FC04" & !df$excluded, ]
      if (nrow(fc04_haiku) == 0) {
        cli_alert_warning(paste(
          "Fix 2 SKIP  No usable Haiku rows for FC04.",
          "Ensure companies_all[4] (FC04) is in the pilot company set."
        ))
      } else {
        ceiling_val <- RATIONAL_HIGH * TOLERANCE
        n_over      <- sum(fc04_haiku$point_estimate > ceiling_val, na.rm = TRUE)
        n_est       <- sum(!is.na(fc04_haiku$point_estimate))
        med_est     <- median(fc04_haiku$point_estimate, na.rm = TRUE)
        if (n_over == 0) {
          cli_alert_success(sprintf(
            "Fix 2 PASS  Haiku FC04 median = $%.2f (rational $%d-$%d; tolerance ceiling $%.0f). EPS line is working.",
            med_est, RATIONAL_LOW, RATIONAL_HIGH, ceiling_val
          ))
        } else {
          msg <- sprintf(
            "Fix 2 FAIL  Haiku FC04: %d/%d estimates exceed $%.0f tolerance ceiling. Median = $%.2f. EPS line may not be rendering in build_company_profile().",
            n_over, n_est, ceiling_val, med_est
          )
          cli_alert_danger(msg)
          failures <- c(failures, msg)
        }
      }
    }

    # ── Summary ─────────────────────────────────────────────────────────────
    cli_rule()
    n_fail <- length(failures)
    if (n_fail == 0) {
      cli_alert_success("ALL VALIDATION CHECKS PASSED — safe to scale to calibrate / full mode.")
    } else {
      cli_alert_danger("{n_fail} validation check(s) FAILED:")
      for (f in failures) cli_alert_danger("  {f}")
      cli_alert_warning("Resolve failures before running calibrate or full mode.")
    }
    invisible(failures)
  }

  # ---- MODE: test ----
  if (opt$mode == "test") {

    if (experiment == "exp3") {
      scenario <- risk_scenarios[[opt$company]]
      anchor_key <- opt$anchor
      if (!anchor_key %in% names(risk_anchors)) {
        cli_alert_danger("Unknown risk anchor key: {anchor_key}")
        cli_alert_info("Available: {paste(names(risk_anchors), collapse = ', ')}")
        quit(status = 1)
      }
      prompt <- build_exp3_prompt(scenario, anchor_key)
      cli_h1("SYSTEM PROMPT"); cat(prompt$system, "
")
      cli_h1("USER PROMPT");   cat(prompt$user, "
")
      cli_h1("METADATA");      cat(toJSON(prompt$metadata, auto_unbox = TRUE, pretty = TRUE), "
")
      quit(status = 0)
    }

    # Exp1 / Exp1b / Exp2 use the selected company set
    company <- companies_all[[opt$company]]

    if (experiment == "exp2") {
      all_conds <- generate_eps_anchor_conditions(company)
      target    <- Filter(function(c) c$condition_id == opt$anchor, all_conds)
      if (length(target) == 0) {
        cli_alert_danger("Unknown condition: {opt$anchor}")
        cli_alert_info("Available: {paste(sapply(all_conds, `[[`, 'condition_id'), collapse = ', ')}")
        quit(status = 1)
      }
      prompt <- build_exp2_prompt2(company, target[[1]])
      cli_h1("SYSTEM PROMPT"); cat(prompt$system, "
")
      cli_h1("USER PROMPT");   cat(prompt$user, "
")
      cli_h1("METADATA");      cat(toJSON(prompt$metadata, auto_unbox = TRUE, pretty = TRUE), "
")
      quit(status = 0)
    }

    # Exp1 and Exp1b share the same anchor generator
    all_conds <- generate_anchor_conditions(company, anchor_base = opt$anchor_base)
    target    <- Filter(function(c) c$condition_id == opt$anchor, all_conds)
    if (length(target) == 0) {
      cli_alert_danger("Unknown condition: {opt$anchor}")
      cli_alert_info("Available: {paste(sapply(all_conds, `[[`, 'condition_id'), collapse = ', ')}")
      quit(status = 1)
    }

    if (experiment == "exp1b") {
      seq_prompt <- build_exp1b_prompts(company, target[[1]],
                                       include_current_price_in_prompt = include_price_prompt)
      cli_h1("SYSTEM PROMPT"); cat(seq_prompt$system, "
")
      cli_h1("TURN 1 (Baseline)"); cat(seq_prompt$turns[[1]]$content, "
")
      cli_h1("TURN 2 (Anchor / Revision)"); cat(seq_prompt$turns[[2]]$content, "
")
      cli_h1("METADATA"); cat(toJSON(seq_prompt$metadata, auto_unbox = TRUE, pretty = TRUE), "
")
      quit(status = 0)
    }

    # Default: Exp1 valuation prompt
    prompt <- build_exp1_prompt(company, target[[1]],
                               include_current_price_in_prompt = include_price_prompt,
                               anchor_position = anchor_pos)
    cli_h1("SYSTEM PROMPT"); cat(prompt$system, "
")
    cli_h1("USER PROMPT");   cat(prompt$user, "
")
    cli_h1("METADATA");      cat(toJSON(prompt$metadata, auto_unbox = TRUE, pretty = TRUE), "
")

    cli_h1("MANIPULATION CHECK VERSION")
    mc <- build_manipulation_check(company, target[[1]],
                                  include_current_price_in_prompt = include_price_prompt)
    cat(mc$user, "
")

    cli_h1("SEQUENTIAL DESIGN (Turn 2)")
    seq_prompt <- build_exp1b_prompts(company, target[[1]],
                                     include_current_price_in_prompt = include_price_prompt)
    cat(seq_prompt$turns[[2]]$content, "
")

  # ---- MODE: pilot ----
  } else if (opt$mode == "pilot") {
    cli_alert_info("Running PILOT")

    if (experiment == "exp1") {
      # Companies: FC01 (clean baseline), FC02 (biotech / valuation_note check),
      # FC04 (capital-intensive; validates Haiku EPS fix — pre-fix returned $78.50
      # against a $39-$52 rational range).
      # All models included so the Gemini json_mode fix is exercised.
      # 3 reps suffice for fix validation; use --mode calibrate for statistics.
      pilot_companies <- companies_all[c(1, 2, 4)]
      n_pc  <- length(pilot_companies)
      n_mod <- length(models)
      cli_alert_info("Pilot: {n_pc} companies x 3 conditions x {n_mod} models x 3 reps (fix-validation run)")
      run_experiment(
        companies            = pilot_companies,
        experiment_fn        = exp_fn,
        models               = models,
        temperatures         = c(0.7),
        repetitions          = 3,
        conditions           = c("control", "52wk_high_30pct", "52wk_low_30pct"),
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        reasoning_effort     = opt$reasoning_effort
      )
      # Automated check: confirms all three calibration fixes are working.
      pilot_validate_fixes(opt$output_dir, pilot_companies)
    } else if (experiment == "exp1b") {
      cli_alert_info("Pilot: 2 companies x 2 anchors x 1 model x 5 reps (2-turn)")
      run_experiment_exp1b(
        companies            = companies_all[1:2],
        models               = models[1],
        temperatures         = c(0.7),
        repetitions          = 5,
        conditions           = c("52wk_high_30pct", "52wk_low_30pct"),
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        include_current_price_in_prompt = include_price_prompt
      )
    } else if (experiment == "exp2") {
      cli_alert_info("Pilot: 2 companies x 3 EPS conditions x 1 model x 10 reps")
      run_experiment_exp2(
        companies            = companies_all[1:2],
        models               = models[1],
        temperatures         = c(0.7),
        repetitions          = 10,
        conditions           = c("control", "consensus_high_30pct", "consensus_low_30pct"),
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override
      )
    } else if (experiment == "exp3") {
      cli_alert_info("Pilot: 2 scenarios x 2 anchors x 1 model x 10 reps")
      run_experiment_exp3(
        scenarios            = risk_scenarios[1:2],
        models               = models[1],
        temperatures         = c(0.7),
        repetitions          = 10,
        anchor_keys          = c("control", "high_default"),
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override
      )
    } else {
      cli_abort("Unknown --experiment: {opt$experiment}")
    }

  # ---- MODE: calibrate ----
  } else if (opt$mode == "calibrate") {
    cli_alert_info("Running CALIBRATION")

    if (experiment == "exp1") {
      cli_alert_info("Calibration: all companies, control only, all models")
      run_experiment(
        companies            = companies_all,
        experiment_fn        = exp_fn,
        models               = models,
        temperatures         = c(0.0, 0.3, 0.7, 1.0),
        repetitions          = opt$reps,
        conditions           = "control",
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp2") {
      cli_alert_info("Calibration: all companies, EPS control only, all models")
      run_experiment_exp2(
        companies            = companies_all,
        models               = models,
        temperatures         = c(0.0, 0.3, 0.7, 1.0),
        repetitions          = opt$reps,
        conditions           = "control",
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp3") {
      cli_alert_info("Calibration: all scenarios, control only, all models")
      run_experiment_exp3(
        scenarios            = risk_scenarios,
        models               = models,
        temperatures         = c(0.0, 0.3, 0.7, 1.0),
        repetitions          = opt$reps,
        anchor_keys          = "control",
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp1b") {
      cli_alert_info("Calibration: all companies, 1 anchor, all models (2-turn)")
      run_experiment_exp1b(
        companies            = companies_all,
        models               = models,
        temperatures         = c(0.0, 0.3, 0.7, 1.0),
        repetitions          = opt$reps,
        conditions           = "52wk_high_30pct",
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        include_current_price_in_prompt = include_price_prompt,
        reasoning_effort     = opt$reasoning_effort
      )
    } else {
      cli_abort("Unknown --experiment: {opt$experiment}")
    }

  # ---- MODE: full ----
  } else if (opt$mode == "full") {

    if (experiment == "exp1") {
      bc <- resolve_batch_conditions(opt$batch)
      cli_alert_info("Running FULL Exp1, batch {opt$batch}")
      run_experiment(
        companies            = companies_all,
        experiment_fn        = exp_fn,
        models               = models,
        temperatures         = temps,
        repetitions          = opt$reps,
        conditions           = bc$conditions,
        debiasings           = bc$debiasings,
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp1b") {
      cli_alert_info("Running FULL Exp1b (2-turn): anchors subset controlled by --anchor list")
      run_experiment_exp1b(
        companies            = companies_all,
        models               = models,
        temperatures         = temps,
        repetitions          = opt$reps,
        conditions           = if (opt$anchor == "") NULL else strsplit(opt$anchor, ",")[[1]],
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        anchor_base          = opt$anchor_base,
        include_current_price_in_prompt = include_price_prompt,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp2") {
      cli_alert_info("Running FULL Exp2 (EPS): conditions controlled by --anchor list or defaults")
      run_experiment_exp2(
        companies            = companies_all,
        models               = models,
        temperatures         = temps,
        repetitions          = opt$reps,
        conditions           = if (opt$anchor == "") NULL else strsplit(opt$anchor, ",")[[1]],
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        reasoning_effort     = opt$reasoning_effort
      )
    } else if (experiment == "exp3") {
      cli_alert_info("Running FULL Exp3 (risk): anchors controlled by --anchor list or defaults")
      run_experiment_exp3(
        scenarios            = risk_scenarios,
        models               = models,
        temperatures         = temps,
        repetitions          = opt$reps,
        anchor_keys          = if (opt$anchor == "") NULL else strsplit(opt$anchor, ",")[[1]],
        output_dir           = opt$output_dir,
        max_output_tokens    = opt$max_output_tokens,
        json_mode            = opt$json_mode,
        enable_anthropic_cache = opt$enable_anthropic_cache,
        anthropic_cache_ttl  = opt$anthropic_cache_ttl,
        reps_by_model        = reps_override,
        temps_by_model       = temps_override,
        reasoning_effort     = opt$reasoning_effort
      )
    } else {
      cli_abort("Unknown --experiment: {opt$experiment}")
    }

  # ---- MODE: openai_batch_build ----
  # Generates JSONL files + mapping CSV for the OpenAI Batch API.
  # NOTE: Batch build/collect currently supports Exp1 prompts only.
  } else if (opt$mode == "openai_batch_build") {
    if (experiment != "exp1") {
      cli_abort("openai_batch_build supports only --experiment exp1 (valuation).")
    }
    bc <- resolve_batch_conditions(opt$batch)
    cli_alert_info("Building OpenAI batch JSONL, batch {opt$batch} -> {opt$batch_dir}/")
    run_openai_batch_build(
      companies         = companies_all,
      experiment_fn     = exp_fn,
      models            = models,
      temperatures      = temps,
      repetitions       = opt$reps,
      conditions        = bc$conditions,
      debiasings        = bc$debiasings,
      max_output_tokens = opt$max_output_tokens,
      json_mode         = opt$json_mode,
      batch_dir         = opt$batch_dir,
      reps_by_model     = reps_override,
      temps_by_model    = temps_override,
      anchor_base       = opt$anchor_base
    )

  # ---- MODE: openai_batch_collect ----

  # Parse an OpenAI batch output JSONL and join to the mapping CSV.
  # Requires: --output_jsonl <path/to/output.jsonl>
  #           --batch_dir    <same dir used during build>
  } else if (opt$mode == "openai_batch_collect") {
    if (opt$output_jsonl == "") {
      cli_abort("--output_jsonl is required for openai_batch_collect mode")
    }
    mapping_path <- file.path(opt$batch_dir, "openai_batch_mapping.csv")
    run_openai_batch_collect(
      output_jsonl_path = opt$output_jsonl,
      mapping_csv_path  = mapping_path,
      output_dir        = opt$output_dir
    )

  # ---- MODE: anthropic_batch_build ----
  # Submits all Claude requests for a single model to the Anthropic Message
  # Batches API. Prints the batch_id; use --mode anthropic_batch_collect to
  # download results when complete.
  # NOTE: Batch build/collect currently supports Exp1 prompts only.
  } else if (opt$mode == "anthropic_batch_build") {
    if (experiment != "exp1") {
      cli_abort("anthropic_batch_build supports only --experiment exp1 (valuation).")
    }
    bc           <- resolve_batch_conditions(opt$batch)
    claude_model <- models[grepl("claude", models, ignore.case = TRUE)][1]
    if (is.na(claude_model)) {
      cli_abort("No Claude model found in --models for anthropic_batch_build.")
    }
    cli_alert_info("Building Anthropic batch for {claude_model}, batch {opt$batch}")
    run_anthropic_batch_build(
      companies         = companies_all,
      experiment_fn     = exp_fn,
      model             = claude_model,
      temperatures      = temps,
      repetitions       = opt$reps,
      conditions        = bc$conditions,
      debiasings        = bc$debiasings,
      max_output_tokens = opt$max_output_tokens,
      batch_dir         = opt$batch_dir,
      reps_by_model     = reps_override,
      temps_by_model    = temps_override,
      anchor_base       = opt$anchor_base
    )

  # ---- MODE: anthropic_batch_collect ----

  # Poll status and download results from a previously submitted Anthropic batch.
  # Requires: --batch_id <msgbatch_01xxx>
  } else if (opt$mode == "anthropic_batch_collect") {
    if (opt$batch_id == "") {
      cli_abort("--batch_id is required for anthropic_batch_collect mode")
    }
    run_anthropic_batch_collect(
      batch_id   = opt$batch_id,
      batch_dir  = opt$batch_dir,
      output_dir = opt$output_dir
    )

  } else {
    cli_abort("Unknown mode: {opt$mode}")
  }
}

# ============================================================================
# SECTION 12: DESIGN NOTES
#
# This section documents key design choices made across the prompt library,
# stimulus materials, and experiment runner. Rationale is included so that
# the choices can be defended in pre-registration and methods sections.
# ============================================================================

# ----------------------------------------------------------------------------
# 12.1  COMPANY STIMULUS DESIGN
# ----------------------------------------------------------------------------
#
# FICTIONAL COMPANIES (FC01–FC10)
#   Ten fictional firms span six sectors to give breadth while keeping
#   anchor magnitudes comparable across companies. Sector coverage:
#     Technology (2), Healthcare (2), Financials (2), Industrials (2),
#     Consumer Staples (1), Energy (1).
#   Fictional stimuli are used as the primary test bed for H1–H2 because
#   (a) the LLM cannot draw on memorized price history, so the control
#   condition truly represents a blank-slate valuation, and (b) the
#   researcher controls all financials, eliminating confounds from
#   real-world news, earnings surprises, or model-specific training data.
#
# REAL COMPANIES (RC01–RC04)
#   Apple (AAPL), Microsoft (MSFT), JPMorgan Chase (JPM), and
#   Johnson & Johnson (JNJ) are included exclusively for Hypothesis 3
#   (domain expertise attenuates anchoring). These are among the most
#   heavily covered equities globally, maximising the probability that
#   any LLM has strong pre-existing valuation priors that could resist
#   anchor pressure.  Financials are sourced from company earnings
#   releases and financial data aggregators; prices are as of
#   February 19, 2026.  The full structured stub (description, financials,
#   narrative, rational_estimate_range) is required so that
#   build_company_profile() can construct a prompt identical in format
#   to the fictional conditions.
#
# FINANCIAL INTEGRITY CONSTRAINTS
#   Every company profile satisfies the following arithmetic identities
#   (all verified programmatically before pre-registration):
#     (i)  pe_ratio  = current_price × shares_outstanding / net_income_ttm
#          Tolerance: |calc − listed| < 0.15 (rounding only).
#          Loss-making companies (FC02, negative NI) carry NA_real_.
#     (ii) ebitda_margin × revenue_ttm ≈ implied EBITDA
#          (no field for raw EBITDA; margin is the stored quantity).
#     (iii) fcf_yield = free_cash_flow / market_cap
#           For banks (FC05, RC03) where traditional FCF is not
#           meaningful, fcf_yield uses earnings yield (NI / mkt cap)
#           as a proxy, documented in inline comments.
#     (iv) ev_ebitda, debt_to_equity, and ebitda_margin are set to
#          NA_real_ for banks (FC05, RC03) following standard industry
#          practice.  The profile builder already handles NA gracefully
#          (prints "N/A" for the two displayed ratios).
#
# NARRATIVE SCRUBBING RULES
#   Narratives must not contain dollar-denominated figures that could
#   act as inadvertent valuation anchors in the control condition.
#   Permitted:  percentages, ratios, unitless counts (accounts, contracts),
#               qualitative descriptors, index/margin comparisons.
#   Prohibited: share prices, market caps, dollar revenues, dollar profits,
#               analyst price targets, book values.
#   Descriptions (the one-line summary rendered before the financials block)
#   are subject to the same rule; the original "$48B AUM" in FC10 was
#   replaced with a non-dollar characterisation.
#   Experiment-3 borrower descriptions are exempt because their dollar loan
#   amounts and revenues are required inputs for the credit-risk task.
#
# JNJ NET INCOME NOTE
#   JNJ FY2025 GAAP net income ($26.6B) is elevated by the reversal of
#   talc-related litigation reserves.  The code uses GAAP figures
#   throughout for consistency.  The RC04 narrative explicitly flags the
#   one-time benefit and reports adjusted earnings growth separately, so
#   that a model participant relying on the profile alone receives an
#   accurate picture of underlying earnings power.  Researchers should
#   be aware that LLMs trained on pre-reversal data may anchor on an
#   adjusted earnings figure that differs from what is shown in the profile.

# ----------------------------------------------------------------------------
# 12.2  ANCHOR CONDITION DESIGN
# ----------------------------------------------------------------------------
#
# ANCHOR TYPES (5 categories, 15 conditions + 1 control = 16 total per company)
#
#   52wk_price (4 conditions)
#     ±30% and ±60% of the rational_estimate_range midpoint, labelled
#     as 52-week high or low.  This mimics the most common anchor
#     encountered in real equity research (Bloomberg price history ribbon,
#     year-to-date charts).  Two magnitudes (30%, 60%) allow a dose–
#     response test of anchoring strength.
#
#   analyst_target (4 conditions)
#     ±30% only, crossed with two salience levels:
#       "embedded"  — anchor buried mid-sentence in a thin-coverage framing
#       "prominent" — anchor is the sole focus of the sentence
#     This 2×2 cell tests whether the attentional weight given to the
#     anchor (salience) moderates pull.  Only the analyst_target type uses
#     a salience manipulation; all other types are embedded.
#
#   sector_pe (2 conditions)
#     Sector-floor and sector-ceiling P/E multiples drawn from
#     sector_pe_ranges (see Section 4).  Unlike price-level anchors,
#     this tests whether a ratio anchor — communicated in a different
#     unit than the DV — still produces assimilation.  The ranges are
#     grounded in 10-year industry medians across GICS sub-sectors.
#
#   round_number / nonround_control (4 conditions)
#     Paired round (e.g., $80) and non-round (e.g., $83) anchors equidistant
#     from the rational estimate.  If round numbers produce stronger anchoring
#     than non-round numbers at the same distance, this replicates the
#     psychological "round number barrier" literature in a new domain.
#     Both round and non-round anchors are presented as recent price-level
#     crossings, so the framing is held constant.
#
#   irrelevant (1 condition)
#     A square-footage figure (headquarters campus size) selected
#     deterministically from {847, 1,203, 3,500, 6,200, 15,000} sq ft.
#     This condition tests whether any large number — even one with no
#     conceptual link to valuation — produces assimilation (basic
#     anchoring) or whether anchoring requires domain relevance.
#     The number is intentionally drawn from a range that does not
#     overlap with plausible stock prices, preventing incidental
#     proximity effects.
#
# ANCHOR VALUE DERIVATION
#   All anchor values are derived from mean(rational_estimate_range),
#   NOT from current_price.  This ensures anchors are proportional to
#   rational value rather than to an arbitrary current-price level,
#   and prevents systematic divergence between anchor magnitude and
#   fundamental value across companies with different price levels.
#
# DETERMINISTIC SEEDING
#   The irrelevant anchor selector and anchor-position randomiser both
#   use deterministic seeds derived from the company ID string, ensuring
#   that results are reproducible without a single global seed that would
#   collapse variation across conditions.  Position randomisation uses
#   xxhash32 of the concatenated company ID + condition ID.

# ----------------------------------------------------------------------------
# 12.3  SYSTEM PROMPT AND PERSONA DESIGN
# ----------------------------------------------------------------------------
#
# BASE ANALYST (primary)
#   Frames the LLM as an experienced equity analyst with an explicit
#   mandate to produce a specific numerical estimate.  This framing
#   is used in all Exp 1 / 1B / 2 cells unless a debiasing condition
#   is active.  The expertise framing is a deliberate variable: it may
#   amplify anchoring (experts are overconfident and adjust less) or
#   suppress it (experts rely more on fundamental structure).
#
# NEUTRAL (control persona)
#   Strips all financial expertise framing.  Used in Batch 4 as a
#   debiasing contrast.  If anchoring is weaker under the neutral
#   persona, this suggests the expert framing actively increases
#   susceptibility — a theoretically important finding.
#
# BASE CREDIT (Exp 3 only)
#   Reframes the task as credit risk assessment rather than equity
#   valuation.  The DV shifts from price to default probability.
#   Using a distinct persona maintains ecological validity for the
#   lending domain.
#
# DEBIASING CONDITIONS (Batch 4, H6)
#   Four strategies tested against the base_analyst baseline:
#     cot          — step-by-step chain-of-thought (structured deliberation)
#     warning      — explicit anchor-awareness instruction
#     adversarial  — frames anchor as coming from a prior AI model to
#                    be disregarded (exploits model's tendency to
#                    differentiate from attributed AI outputs)
#     multi_source — instructs equal weighting of bullish/bearish factors
#                    (may reduce anchoring by forcing integration)
#   Batch 4 runs only the three highest-powered anchor conditions
#   (control, 52wk_high_30pct, 52wk_low_30pct) to keep cell counts
#   manageable while still enabling a full debiasing × direction × model
#   factorial.

# ----------------------------------------------------------------------------
# 12.4  RESPONSE FORMAT DESIGN
# ----------------------------------------------------------------------------
#
# JSON PRIMARY FORMAT (use_json = TRUE, default)
#   Requests 8 fields: fair_value_estimate, confidence, fair_value_low,
#   fair_value_high, implied_annual_growth_rate, implied_pe_ratio,
#   recommendation, brief_justification.
#   JSON enforces a consistent response structure across models and
#   repetitions, reduces parse failures, and captures six output
#   dimensions simultaneously (required for H5: does anchoring affect
#   the stated confidence interval, not just the point estimate?).
#   The implied_pe_ratio field is particularly useful: if the model
#   anchors on price but also computes a consistent implied multiple,
#   this tests whether anchoring is operating at the price level or
#   at the earnings-multiple level.
#
# FREE-TEXT FALLBACK (use_json = FALSE)
#   Used in exploratory calibration runs to assess whether JSON
#   formatting constraints suppress qualitative reasoning.  The
#   same 7 pieces of information are requested in numbered form.
#   parse_valuation_response() tries JSON extraction first, then
#   falls back to regex on the free-text format.
#
# PARSE CHAIN
#   Method 1: Extract first {...} block and parse as JSON.
#   Method 2: Three regex patterns for price-like numbers, tried in
#   order of specificity (labelled expression → per-share suffix →
#   bare dollar sign).  The parse_method column records which path
#   succeeded, enabling quality stratification in analysis.

# ----------------------------------------------------------------------------
# 12.5  EXPERIMENTAL PARADIGM DESIGN
# ----------------------------------------------------------------------------
#
# EXPERIMENT 1: BETWEEN-SUBJECTS, SINGLE-TURN (primary)
#   Each API call is independent.  The anchor is embedded in the profile
#   before the model sees the company at all.  This is the cleanest
#   measurement of initial-assimilation anchoring because there is no
#   baseline estimate to adjust from.  The trade-off is that
#   insufficient-adjustment anchoring (starting from an anchor and
#   failing to move far enough) cannot be distinguished from pure
#   assimilation in a single-turn design.
#
# EXPERIMENT 1B: WITHIN-SUBJECTS, SEQUENTIAL TWO-TURN
#   Turn 1: baseline valuation with no anchor (identical to Exp 1 control).
#   Turn 2: anchor introduced as "new information," model asked to revise.
#   The adjustment from Turn 1 to Turn 2 measures insufficient adjustment
#   separately from initial assimilation.  The two-turn paradigm also
#   mirrors real analyst workflows (initial model → model revision on
#   new information) and allows anchoring to be decomposed into two
#   theoretically distinct mechanisms.  Requires passing the full
#   message history in the API call (handled by the turns list structure).
#
# EXPERIMENT 2: EARNINGS FORECAST ANCHORING
#   DV is next-quarter EPS rather than stock price.  Quarterly history
#   is generated deterministically (5-year seasonal + growth trajectory)
#   to provide a structured extrapolation task.  Anchor types: consensus
#   estimate (professionally sourced number) or irrelevant (employee count).
#   This isolates whether anchoring occurs when the DV is a flow quantity
#   (EPS) rather than a stock quantity (price) and whether a "consensus"
#   label amplifies the anchor vs. a contextually irrelevant number.
#
# EXPERIMENT 3: CREDIT RISK ANCHORING
#   DV is default probability and 1–10 risk rating (continuous) plus
#   a categorical loan recommendation.  Three borrower scenarios vary
#   in true risk (low/moderate/elevated) to prevent ceiling/floor effects.
#   Anchor types: credit score (high/low), macroeconomic statistic
#   (national bankruptcy rate high/low), and framing (survival frame vs.
#   founding-year frame).  The credit score anchor is domain-relevant;
#   the macroeconomic anchor tests whether a population-level statistic
#   biases an individual-level assessment.

# ----------------------------------------------------------------------------
# 12.6  ANCHOR POSITION MANIPULATION
# ----------------------------------------------------------------------------
#
#   Anchors are embedded within the "Recent Developments" narrative block
#   at one of three positions: beginning, middle, or end.  Position is
#   randomised deterministically per company × condition combination.
#   This is not a primary hypothesis but is recorded as a covariate
#   to test whether recency or primacy moderates anchor strength.  The
#   embed_anchor_in_profile() function splits on the "Recent Developments:"
#   header and inserts at the appropriate sentence break; if the header
#   is absent the anchor appends to the end (failsafe).

# ----------------------------------------------------------------------------
# 12.7  SAMPLING AND STATISTICAL POWER
# ----------------------------------------------------------------------------
#
# REPETITIONS
#   Default: 50 repetitions per cell (company × condition × model ×
#   temperature × debiasing).  At temperature = 0.7, the response
#   distribution has sufficient variance to detect anchor effects of
#   Cohen's d ≈ 0.3 at α = 0.05 with power ≈ 0.80 for a two-sample
#   t-test, based on pilot variance estimates from calibration.
#   Pilot mode uses 3 reps on 3 companies (FC01, FC02, FC04) across all
#   models to verify pipeline integrity and run automated fix-validation
#   checks (pilot_validate_fixes) before scaling to calibrate / full.
#
# TEMPERATURE AS DESIGN FACTOR
#   Calibration runs four temperature levels: 0.0, 0.3, 0.7, 1.0.
#   The hypothesis is that higher temperature (more stochastic sampling)
#   reduces anchoring because the model explores more of the output
#   distribution rather than converging on a local anchor-proximate mode.
#   Temperature is included as a covariate in the main analysis
#   regardless of whether it is explicitly hypothesised.
#
#   IMPORTANT — ASYMMETRIC TEMPERATURE DESIGN:
#   Temperature is a valid design factor ONLY for Haiku and Gemini (four
#   levels each).  GPT-5-mini does not accept the temperature parameter;
#   its calls are automatically collapsed to a single cell (model_temps <- 1)
#   inside each run_experiment*() function.  As a result:
#     • Temperature cannot be treated as a continuous covariate in pooled
#       regressions that include GPT-5-mini.  Fit temperature effects within
#       Haiku and Gemini only, then report GPT-5-mini separately.
#     • The T=0 determinism diagnostic (near-zero within-rep SD) applies
#       only to Haiku and Gemini; GPT-5-mini will show non-zero within-rep
#       variance even at the single calibrated level.
#     • reasoning_effort ("low" | "medium" | "high", --reasoning_effort flag)
#       is the analogous stochasticity control for GPT-5-mini.  It is
#       recorded in the reasoning_effort_effective CSV column (NA for
#       Claude / Gemini).  Run separate calibrate sweeps at each effort
#       level to quantify its effect on estimate variance before selecting
#       a production level.
#
# STAGED BATCH EXECUTION
#   The full experiment is split into four batches to allow incremental
#   cost control and early stopping:
#     Batch 1: control + 52wk ±30%     (3 conditions — highest priority)
#     Batch 2: 52wk ±60% + analyst targets (6 conditions)
#     Batch 3: sector PE, round/non-round, irrelevant (7 conditions)
#     Batch 4: debiasing × 3 anchor conditions × 6 personas (18 sub-cells)
#   Results from Batch 1 are sufficient to test H1 (existence of
#   anchoring) and H2 (directionality).

# ----------------------------------------------------------------------------
# 12.8  EXCLUSION CRITERIA (PRE-REGISTERED)
# ----------------------------------------------------------------------------
#
#   Exclusions are applied uniformly via apply_exclusion_rules() and
#   logged in the excluded / exclusion_reason columns:
#     no_numeric_estimate   — response contained no parseable number
#     estimate_too_low      — point estimate < $1.00
#     estimate_too_high     — point estimate > 10× current_price
#   The 10× cap removes obviously hallucinated outputs (e.g., a model
#   confusing price with market cap) while being wide enough to retain
#   legitimate high-anchor responses even for the ±60% conditions.
#   Exclusion rates are reported by condition as a balance check;
#   differential exclusion across anchor conditions would indicate
#   that anchors affect response parsability, not just point estimates.

# ----------------------------------------------------------------------------
# 12.9  MODEL ROSTER AND VERSIONING
# ----------------------------------------------------------------------------
#
#   Default models: claude-haiku-4-5-20251001, gpt-5-mini-2025-08-07, gemini-2.5-flash
#   These represent broadly comparable capability tiers across three
#   providers (Anthropic, OpenAI, Google), enabling cross-model
#   generalisability tests (H4).  If a model is updated between batches,
#   both version strings should be retained in the --models flag and
#   recorded in the results CSV so that version can be treated as a
#   covariate.  The call_model() dispatcher routes by name prefix
#   ("claude" → Anthropic, "gpt" → OpenAI, "gemini" → Google); any
#   new model string must match one of these prefixes or extend the
#   dispatcher.
#
#   MODEL-SPECIFIC STOCHASTICITY CONTROLS:
#     claude-haiku-4-5-20251001  — temperature (0.0–1.0 accepted)
#     gemini-2.5-flash           — temperature (0.0–1.0 accepted)
#     gpt-5-mini-2025-08-07      — reasoning_effort ("low"/"medium"/"high")
#                                  temperature parameter is silently ignored
#                                  by the OpenAI API for this model family
#   The run_experiment*() functions detect GPT models by name prefix and
#   collapse their temperature vector to a single value (1) to prevent
#   redundant API calls.  The active reasoning_effort value is passed
#   through call_model() → call_openai() and recorded in the
#   reasoning_effort_effective column of every results CSV.  For Haiku
#   and Gemini, reasoning_effort_effective is NA.

# ----------------------------------------------------------------------------
# 12.10  MANIPULATION CHECK DESIGN
# ----------------------------------------------------------------------------
#
#   The manipulation check probe (build_manipulation_check) is run on
#   a stratified 10% subset after the main trial (not before), to avoid
#   demand-characteristic effects that would contaminate the main
#   valuation estimate.  It asks three sequential questions in a single
#   call:
#     COMPREHENSION  — which three financial metrics matter most, and
#                      what are their values? (tests whether the model
#                      actually processed the fundamentals)
#     AWARENESS      — was any information in the profile designed to
#                      influence the estimate? (tests metacognitive
#                      anchor detection)
#     VALUATION      — provide a fair value estimate as usual.
#   If a model detects the anchor (detected_influence = "yes") but
#   still produces an anchored estimate, this dissociation is itself
#   a theoretically important finding about the limits of metacognitive
#   debiasing in LLMs.

# ============================================================================
# ============================================================================
# Algorithmic Anchoring — LLM Financial Estimate Bias
# Prompt Library & Experiment Runner (R Version)
#
# Author:  John Garcia, California Lutheran University
# Version: 2.0 (February 2026)
#
# Usage:
#   Rscript anchoring_experiment.R --mode test --company 1 --anchor 52wk_high_30pct
#   Rscript anchoring_experiment.R --mode pilot
#   Rscript anchoring_experiment.R --mode calibrate
#   Rscript anchoring_experiment.R --mode full --batch 1
#
# Dependencies:
#   install.packages(c("httr2", "jsonlite", "tidyverse", "glue", "cli",
#                       "digest", "optparse", "arrow"))
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
      pe_ratio = 28.5,
      ev_ebitda = 18.2,
      shares_outstanding = 145e6,
      current_price = 62.40,
      debt_to_equity = 0.35,
      fcf_yield = 0.032
    ),
    # Removed "$450M in cash" — dollar amount could serve as an
    # inadvertent anchor. Replaced with percentage/ratio.
    narrative = paste(
      "Meridian Data Systems reported strong Q3 results with revenue up 18% YoY,",
      "driven by its new AI-powered demand forecasting module. The company expanded",
      "its Fortune 500 client base from 42 to 57 accounts. Gross margins improved",
      "to 71% from 68% a year ago. Management raised full-year guidance by 5%.",
      "The balance sheet is healthy with a net cash position and low leverage."
    ),
    rational_estimate_range = c(55, 75)
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
      fcf_yield = -0.08
    ),
    narrative = paste(
      "Cascadia BioTherapeutics announced positive Phase II data for CBT-401,",
      "its lead candidate for moderate-to-severe psoriasis, achieving PASI 75 in",
      "62% of patients vs. 18% placebo. The company has sufficient cash runway",
      "for approximately 22 months at current burn rate. Two additional candidates",
      "are in Phase II for lupus and Crohn's disease."
    ),
    rational_estimate_range = c(25, 50)
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
    # Removed "$280M acquisition" and "$300M buyback" — specific dollar
    # amounts could anchor valuation. Replaced with qualitative descriptions.
    narrative = paste(
      "Heartland Consumer Brands delivered steady Q3 results with organic growth",
      "of 3.5%, slightly above the industry average. Volume was flat but pricing",
      "contributed 3.5 points. The company completed a mid-sized acquisition,",
      "expanding into the organic snack segment. Dividend yield is 2.8%, and the",
      "board authorized a share buyback program."
    ),
    rational_estimate_range = c(36, 48)
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
    # CRITICAL FIX: Original had "Analyst consensus has been relatively
    # stable at $45-48" — this is a direct valuation anchor embedded in
    # the CONTROL condition. Removed entirely.
    narrative = paste(
      "Ironclad Industrial Technologies posted solid Q3 with backlog growing 12%",
      "to a record level, driven by EV-related orders. EBITDA margins expanded",
      "50bps on manufacturing efficiency improvements. The company guided for",
      "10-12% revenue growth next year, supported by two new plant openings",
      "in Mexico."
    ),
    rational_estimate_range = c(39, 52)
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
    rational_estimate_range = c(24, 33)
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
    rational_estimate_range = c(65, 90)
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
    # Removed "$680M in quarterly revenue" and "$800M in cash"
    narrative = paste(
      "Sterling Pharmaceuticals saw its lead product STP-200 grow 25% YoY,",
      "maintaining dominant market share. The FDA accepted the NDA for STP-305,",
      "a next-gen treatment for spinal muscular atrophy, with a PDUFA date in",
      "Q2 next year. The balance sheet is strong with ample cash reserves",
      "and no near-term debt maturities."
    ),
    rational_estimate_range = c(35, 50)
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
      "its 6-8% annual dividend growth target and trades at a discount to",
      "utility peers on an earnings basis."
    ),
    rational_estimate_range = c(25, 35)
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
    rational_estimate_range = c(34, 46)
  ),

  list(
    id = "FC10",
    name = "Pinnacle Wealth Solutions",
    sector = "Financials",
    is_fictional = TRUE,
    description = "Registered investment advisor and fintech platform. $48B AUM across retail and institutional clients.",
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
    rational_estimate_range = c(42, 58)
  )
)


# Real companies for H3 (domain expertise)
real_companies <- list(
  list(
    id = "RC01", name = "Apple Inc.", ticker = "AAPL",
    sector = "Technology", is_fictional = FALSE,
    note = "Heavily analyzed. Anchoring should be weaker per H3."
  ),
  list(
    id = "RC02", name = "Microsoft Corporation", ticker = "MSFT",
    sector = "Technology", is_fictional = FALSE,
    note = "Strong analyst consensus. Tests prior knowledge buffer."
  ),
  list(
    id = "RC03", name = "JPMorgan Chase & Co.", ticker = "JPM",
    sector = "Financials", is_fictional = FALSE,
    note = "Major bank. Tests financial sector domain knowledge."
  ),
  list(
    id = "RC04", name = "Johnson & Johnson", ticker = "JNJ",
    sector = "Healthcare", is_fictional = FALSE,
    note = "Stable blue chip. Tests healthcare anchoring resistance."
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
    "When making valuation estimates, you always think through the problem",
    "step by step: (1) assess the company's fundamentals, (2) compare to",
    "sector benchmarks, (3) consider growth trajectory, (4) apply an",
    "appropriate valuation methodology, and (5) arrive at your estimate.",
    "Show your reasoning for each step."
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

build_company_profile <- function(company) {
  fin <- company$financials

  pe_str <- if (is.na(fin$pe_ratio)) "N/A (negative earnings)" else sprintf("%.1f", fin$pe_ratio)
  ev_str <- if (is.na(fin$ev_ebitda)) "N/A" else sprintf("%.1f", fin$ev_ebitda)

  glue(
    "Company: {company$name}\n",
    "Sector: {company$sector}\n",
    "Description: {company$description}\n",
    "\n",
    "Key Financials (Trailing Twelve Months):\n",
    "- Revenue: ${sprintf('%.1f', fin$revenue_ttm / 1e9)}B ",
    "(YoY growth: {sprintf('%.0f%%', fin$revenue_growth * 100)})\n",
    "- EBITDA Margin: {sprintf('%.0f%%', fin$ebitda_margin * 100)}\n",
    "- Net Income: ${sprintf('%.0f', fin$net_income_ttm / 1e6)}M\n",
    "- P/E Ratio: {pe_str}\n",
    "- EV/EBITDA: {ev_str}\n",
    "- Current Share Price: ${sprintf('%.2f', fin$current_price)}\n",
    "- Shares Outstanding: {sprintf('%.0f', fin$shares_outstanding / 1e6)}M\n",
    "- FCF Yield: {sprintf('%.1f%%', fin$fcf_yield * 100)}\n",
    "\n",
    "Recent Developments:\n",
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


generate_anchor_conditions <- function(company, baseline_estimate = NULL) {
  if (is.null(baseline_estimate)) {
    baseline_estimate <- mean(company$rational_estimate_range)
  }

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
    anchor_val <- round(baseline_estimate * spec$mult, 2)
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
    anchor_val <- round(baseline_estimate * spec$mult, 2)
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
  round_high    <- as.integer(ceiling(baseline_estimate / 10) * 10 + 20)
  nonround_high <- round_high + 3  # Same distance, non-round
  round_low     <- as.integer(floor(baseline_estimate / 10) * 10 - 20)
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
  "\n\nPlease respond in the following JSON format ONLY (no other text):\n",
  '{\n',
  '  "fair_value_estimate": <your price target as a number>,\n',
  '  "confidence": "<low|medium|high>",\n',
  '  "fair_value_low": <low end of your range>,\n',
  '  "fair_value_high": <high end of your range>,\n',
  '  "implied_annual_growth_rate": <your implied revenue growth rate as a decimal, e.g. 0.15>,\n',
  '  "implied_pe_ratio": <the P/E ratio implied by your fair value>,\n',
  '  "recommendation": "<strong_buy|buy|hold|sell|strong_sell>",\n',
  '  "brief_justification": "<2-3 sentence explanation>"\n',
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
  anchor_position = NULL  # NULL = randomize; or "beginning"/"middle"/"end"
) {

  profile <- build_company_profile(company)

  # Randomize anchor position if not specified
  if (is.null(anchor_position)) {
    # Deterministic based on company + condition
    seed_str <- paste0(company$id, anchor$condition_id)
    set.seed(digest(seed_str, algo = "xxhash32", serialize = FALSE) |>
               strtoi(base = 16L) %% .Machine$integer.max)
    anchor_position <- sample(c("beginning", "middle", "end"), 1)
  }

  # Embed anchor into the profile at the chosen position
  profile_with_anchor <- embed_anchor_in_profile(
    profile, anchor$anchor_text, anchor_position
  )

  # Build the question
  response_fmt <- if (use_json) STRUCTURED_RESPONSE_INSTRUCTION else FREETEXT_RESPONSE_INSTRUCTION

  user_content <- glue(
    "{profile_with_anchor}\n\n",
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
      debiasing        = debiasing,
      persona          = sys_key,
      current_price    = company$financials$current_price,
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

build_exp1b_prompts <- function(company, anchor, system_key = "base_analyst") {
  profile <- build_company_profile(company)

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

build_manipulation_check <- function(company, anchor) {
  profile <- build_company_profile(company)
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
  set.seed(digest(company$id, algo = "xxhash32", serialize = FALSE) |>
             strtoi(base = 16L) %% .Machine$integer.max)

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
# SECTION 8: API EXECUTION ENGINE
#
# Temperature is included as a design factor
# ============================================================================

call_anthropic <- function(system_msg, user_msg, model = "claude-sonnet-4-20250514",
                           temperature = 0.7, max_tokens = 1024L) {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (api_key == "") {
    cli_warn("ANTHROPIC_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0, latency = 0))
  }

  start <- Sys.time()
  resp <- request("https://api.anthropic.com/v1/messages") |>
    req_headers(
      "x-api-key"         = api_key,
      "anthropic-version"  = "2023-06-01",
      "content-type"       = "application/json"
    ) |>
    req_body_json(list(
      model       = model,
      max_tokens  = max_tokens,
      temperature = temperature,
      system      = system_msg,
      messages    = list(list(role = "user", content = user_msg))
    )) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  list(
    response   = body$content[[1]]$text,
    tokens_in  = body$usage$input_tokens,
    tokens_out = body$usage$output_tokens,
    latency    = latency
  )
}


call_openai <- function(system_msg, user_msg, model = "gpt-4o",
                        temperature = 0.7, max_tokens = 1024L) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    cli_warn("OPENAI_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0, latency = 0))
  }

  start <- Sys.time()
  resp <- request("https://api.openai.com/v1/chat/completions") |>
    req_headers(
      "Authorization" = paste("Bearer", api_key),
      "Content-Type"  = "application/json"
    ) |>
    req_body_json(list(
      model       = model,
      temperature = temperature,
      max_tokens  = max_tokens,
      messages    = list(
        list(role = "system", content = system_msg),
        list(role = "user", content = user_msg)
      )
    )) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  list(
    response   = body$choices[[1]]$message$content,
    tokens_in  = body$usage$prompt_tokens,
    tokens_out = body$usage$completion_tokens,
    latency    = latency
  )
}


call_google <- function(system_msg, user_msg, model = "gemini-2.0-flash",
                        temperature = 0.7, max_tokens = 1024L) {
  api_key <- Sys.getenv("GOOGLE_API_KEY")
  if (api_key == "") {
    cli_warn("GOOGLE_API_KEY not set. Skipping.")
    return(list(response = "[SKIPPED]", tokens_in = 0, tokens_out = 0, latency = 0))
  }

  start <- Sys.time()
  url <- glue("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}")

  resp <- request(url) |>
    req_body_json(list(
      system_instruction = list(parts = list(list(text = system_msg))),
      contents = list(list(
        role = "user",
        parts = list(list(text = user_msg))
      )),
      generationConfig = list(
        temperature = temperature,
        maxOutputTokens = max_tokens
      )
    )) |>
    req_retry(max_tries = 5, backoff = ~ 2) |>
    req_perform()

  body <- resp_body_json(resp)
  latency <- as.numeric(difftime(Sys.time(), start, units = "secs"))

  text <- tryCatch(
    body$candidates[[1]]$content$parts[[1]]$text,
    error = function(e) "[PARSE_ERROR]"
  )

  list(
    response   = text,
    tokens_in  = body$usageMetadata$promptTokenCount %||% 0L,
    tokens_out = body$usageMetadata$candidatesTokenCount %||% 0L,
    latency    = latency
  )
}


# Unified dispatcher
call_model <- function(model, system_msg, user_msg, temperature = 0.7) {
  if (grepl("claude", model, ignore.case = TRUE)) {
    call_anthropic(system_msg, user_msg, model, temperature)
  } else if (grepl("gpt", model, ignore.case = TRUE)) {
    call_openai(system_msg, user_msg, model, temperature)
  } else if (grepl("gemini", model, ignore.case = TRUE)) {
    call_google(system_msg, user_msg, model, temperature)
  } else {
    list(response = glue("[UNSUPPORTED MODEL: {model}]"),
         tokens_in = 0, tokens_out = 0, latency = 0)
  }
}


# ============================================================================
# SECTION 9: RESPONSE PARSING
#
# JSON primary, regex fallback. Track parse method per row.
# Apply exclusion rules after parsing.
# ============================================================================

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
    parse_method    = "failed"
  )

  # Method 1: JSON extraction
  json_match <- regmatches(response_text, regexpr("\\{[^{}]*\\}", response_text))
  if (length(json_match) > 0) {
    tryCatch({
      parsed <- fromJSON(json_match[1])
      if (!is.null(parsed$fair_value_estimate)) {
        result$point_estimate  <- as.numeric(parsed$fair_value_estimate)
        result$confidence      <- parsed$confidence %||% NA_character_
        result$fair_value_low  <- as.numeric(parsed$fair_value_low %||% NA)
        result$fair_value_high <- as.numeric(parsed$fair_value_high %||% NA)
        result$implied_growth  <- as.numeric(parsed$implied_annual_growth_rate %||% NA)
        result$implied_pe      <- as.numeric(parsed$implied_pe_ratio %||% NA)
        result$recommendation  <- parsed$recommendation %||% NA_character_
        result$justification   <- parsed$brief_justification %||% NA_character_
        result$parse_method    <- "json"
      }
    }, error = function(e) NULL)
  }

  # Method 2: Regex fallback
  if (is.na(result$point_estimate)) {
    patterns <- c(
      "(?:fair value|price target|target price|estimate|valuation)[:\\s]*\\$?([\\d,.]+)",
      "\\$?([\\d,.]+)\\s*(?:per share|/share)",
      "\\$([\\d,.]+)"
    )
    for (pat in patterns) {
      m <- regmatches(response_text, regexpr(pat, response_text, perl = TRUE, ignore.case = TRUE))
      if (length(m) > 0) {
        nums <- regmatches(m[1], gregexpr("[\\d,.]+", m[1]))[[1]]
        val <- as.numeric(gsub(",", "", nums[1]))
        if (!is.na(val) && val > 1 && val < 10000) {
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
apply_exclusion_rules <- function(parsed, current_price) {
  excluded <- FALSE
  exclusion_reason <- NA_character_

  if (is.na(parsed$point_estimate)) {
    excluded <- TRUE
    exclusion_reason <- "no_numeric_estimate"
  } else if (parsed$point_estimate < EXCLUSION_RULES$min_price_estimate) {
    excluded <- TRUE
    exclusion_reason <- "estimate_too_low"
  } else if (parsed$point_estimate > current_price * EXCLUSION_RULES$max_price_multiple) {
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
  models          = c("claude-sonnet-4-20250514", "gpt-4o", "gemini-2.0-flash"),
  temperatures    = c(0.7),
  repetitions     = 50,
  conditions      = NULL,    # NULL = all; or character vector of condition_ids
  debiasings      = "none",
  output_dir      = "results",
  checkpoint_every = 100
) {

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results <- list()
  call_count <- 0
  total_cost <- 0

  total_calls <- length(companies) * length(models) * length(temperatures) * repetitions
  cli_alert_info("Estimated total API calls: ~{total_calls} per condition")

  for (company in companies) {
    all_conditions <- generate_anchor_conditions(company)
    if (!is.null(conditions)) {
      all_conditions <- Filter(function(c) c$condition_id %in% conditions, all_conditions)
    }

    for (anchor in all_conditions) {
      for (debias in debiasings) {
        prompt <- experiment_fn(company, anchor, debiasing = debias)

        for (model in models) {
          for (temp in temperatures) {
            for (rep in seq_len(repetitions)) {
              # API call
              api_result <- tryCatch(
                call_model(model, prompt$system, prompt$user, temp),
                error = function(e) {
                  cli_alert_danger("API error: {e$message}")
                  list(response = glue("[ERROR: {e$message}]"),
                       tokens_in = 0, tokens_out = 0, latency = 0)
                }
              )

              # Parse response
              parsed <- parse_valuation_response(api_result$response)
              exclusion <- apply_exclusion_rules(parsed, company$financials$current_price)

              # Build result record
              record <- c(
                prompt$metadata,
                list(
                  model        = model,
                  temperature  = temp,
                  repetition   = rep,
                  response_raw = api_result$response,
                  tokens_in    = api_result$tokens_in,
                  tokens_out   = api_result$tokens_out,
                  latency_sec  = api_result$latency,
                  timestamp    = as.character(Sys.time())
                ),
                parsed,
                exclusion
              )

              results[[length(results) + 1]] <- record
              call_count <- call_count + 1

              # Checkpoint
              if (call_count %% checkpoint_every == 0) {
                checkpoint_df <- bind_rows(lapply(results, as_tibble))
                write_csv(checkpoint_df,
                          file.path(output_dir, glue("checkpoint_{format(Sys.time(), '%Y%m%d_%H%M')}.csv")))
                cli_alert_info("Checkpoint: {call_count} calls. Parse rate: {sprintf('%.1f%%', mean(!checkpoint_df$excluded) * 100)}")
              }
            }
          }
        }
      }
    }
  }

  # Save final
  final_df <- bind_rows(lapply(results, as_tibble))
  timestamp <- format(Sys.time(), "%Y%m%d")
  write_csv(final_df, file.path(output_dir, glue("exp1_valuation_{timestamp}.csv")))

  cli_alert_success("Complete: {nrow(final_df)} records saved.")
  cli_alert_info("Parse success rate: {sprintf('%.1f%%', mean(final_df$parse_method != 'failed') * 100)}")
  cli_alert_info("Exclusion rate: {sprintf('%.1f%%', mean(final_df$excluded) * 100)}")

  # Report exclusion rates by condition as a balance check
  final_df |>
    group_by(condition_id) |>
    summarise(
      n = n(),
      excluded_pct = sprintf("%.1f%%", mean(excluded) * 100),
      .groups = "drop"
    ) |>
    print()

  return(final_df)
}


# ============================================================================
# SECTION 11: CLI ENTRY POINT
# ============================================================================

option_list <- list(
  make_option("--mode", type = "character", default = "test",
              help = "Run mode: test, pilot, calibrate, full [default: test]"),
  make_option("--company", type = "integer", default = 1,
              help = "Company index for test mode [default: 1]"),
  make_option("--anchor", type = "character", default = "control",
              help = "Anchor condition_id for test mode [default: control]"),
  make_option("--batch", type = "integer", default = 1,
              help = "Batch number for staged full execution [default: 1]"),
  make_option("--output_dir", type = "character", default = "results",
              help = "Output directory [default: results]"),
  make_option("--models", type = "character",
              default = "claude-sonnet-4-20250514,gpt-4o,gemini-2.0-flash",
              help = "Comma-separated model list"),
  make_option("--reps", type = "integer", default = 50,
              help = "Repetitions per cell [default: 50]"),
  make_option("--temps", type = "character", default = "0.7",
              help = "Comma-separated temperature list [default: 0.7]")
)

if (!interactive()) {

  opt <- parse_args(OptionParser(option_list = option_list))
  models <- strsplit(opt$models, ",")[[1]]
  temps  <- as.numeric(strsplit(opt$temps, ",")[[1]])

  if (opt$mode == "test") {
    # Print a single prompt for inspection
    company <- fictional_companies[[opt$company]]
    all_conds <- generate_anchor_conditions(company)
    target <- Filter(function(c) c$condition_id == opt$anchor, all_conds)

    if (length(target) == 0) {
      cli_alert_danger("Unknown condition: {opt$anchor}")
      cli_alert_info("Available: {paste(sapply(all_conds, `[[`, 'condition_id'), collapse = ', ')}")
      quit(status = 1)
    }

    prompt <- build_exp1_prompt(company, target[[1]])
    cli_h1("SYSTEM PROMPT")
    cat(prompt$system, "\n")
    cli_h1("USER PROMPT")
    cat(prompt$user, "\n")
    cli_h1("METADATA")
    cat(toJSON(prompt$metadata, auto_unbox = TRUE, pretty = TRUE), "\n")

    # Also show manipulation check version
    cli_h1("MANIPULATION CHECK VERSION")
    mc <- build_manipulation_check(company, target[[1]])
    cat(mc$user, "\n")

    # Also show sequential version
    cli_h1("SEQUENTIAL DESIGN (Turn 2)")
    seq_prompt <- build_exp1b_prompts(company, target[[1]])
    cat(seq_prompt$turns[[2]]$content, "\n")

  } else if (opt$mode == "pilot") {
    cli_alert_info("Running PILOT: 2 companies x 3 conditions x 1 model x 10 reps")
    run_experiment(
      companies    = fictional_companies[1:2],
      experiment_fn = function(co, an, debiasing = "none") {
        build_exp1_prompt(co, an, debiasing = debiasing)
      },
      models       = models[1],
      temperatures = c(0.7),
      repetitions  = 10,
      conditions   = c("control", "52wk_high_30pct", "52wk_low_30pct"),
      output_dir   = opt$output_dir
    )

  } else if (opt$mode == "calibrate") {
    cli_alert_info("Running CALIBRATION: all companies, control only, all models")
    # Include multiple temperatures in calibration
    run_experiment(
      companies    = fictional_companies,
      experiment_fn = function(co, an, debiasing = "none") {
        build_exp1_prompt(co, an, debiasing = debiasing)
      },
      models       = models,
      temperatures = c(0.0, 0.3, 0.7, 1.0),
      repetitions  = opt$reps,
      conditions   = "control",
      output_dir   = opt$output_dir
    )

  } else if (opt$mode == "full") {
    # Staged execution based on batch number
    batch_conditions <- switch(as.character(opt$batch),
      "1" = c("control", "52wk_high_30pct", "52wk_low_30pct"),
      "2" = c("52wk_high_60pct", "52wk_low_60pct",
              "analyst_high_embedded", "analyst_low_embedded",
              "analyst_high_prominent", "analyst_low_prominent"),
      "3" = c("sector_pe_high", "sector_pe_low",
              "round_high", "round_low",
              "nonround_high", "nonround_low",
              "irrelevant"),
      "4" = NULL  # All conditions, with debiasing
    )

    batch_debias <- if (opt$batch == 4) {
      c("none", "cot", "warning", "adversarial", "multi_source", "neutral")
    } else {
      "none"
    }

    batch_conds <- if (opt$batch == 4) {
      c("control", "52wk_high_30pct", "52wk_low_30pct")
    } else {
      batch_conditions
    }

    cli_alert_info("Running FULL experiment, batch {opt$batch}")
    run_experiment(
      companies    = fictional_companies,
      experiment_fn = function(co, an, debiasing = "none") {
        build_exp1_prompt(co, an, debiasing = debiasing)
      },
      models       = models,
      temperatures = temps,
      repetitions  = opt$reps,
      conditions   = batch_conds,
      debiasings   = batch_debias,
      output_dir   = opt$output_dir
    )
  }
}

# ============================================================================
# SECTION 12: DESIGN NOTES
#
# This section documents key design choices.
# ============================================================================

# Algorithmic Anchoring — LLM Financial Estimate Bias

**Author:** John Garcia, California Lutheran University  
**Version:** 2.0 (February 2026)

## Overview

This project investigates anchoring bias in large language models (LLMs) when making financial valuation estimates. The experiment presents LLMs with company profiles containing various anchoring manipulations and measures the resulting bias in price targets, earnings forecasts, and risk assessments.

## Project Structure

```
algorithmic-anchoring/
├── R/                          # All R source files
│   ├── anchoring_experiment.R  # Prompt library + API runner
│   ├── analysis_pipeline.R     # Statistical analysis
│   ├── parse_responses.R       # Response extraction logic
│   └── utils.R                 # Shared helper functions
├── data/
│   ├── raw/                    # Raw API responses (JSON + CSV)
│   ├── processed/              # Parsed, cleaned datasets
│   ├── calibration/            # Baseline calibration results
│   └── pilot/                  # Pilot run outputs
├── results/
│   ├── exp1a_valuation/        # Between-subjects anchoring
│   ├── exp1b_sequential/       # Within-subjects revision
│   ├── exp2_earnings/
│   ├── exp3_risk/
│   └── manipulation_checks/
├── analysis/
│   ├── figures/                # Publication-ready figures
│   └── tables/                 # Formatted regression tables
├── manuscript/
│   ├── sections/               # Drafts by section
│   └── references/             # .bib files
├── preregistration/            # OSF pre-registration docs
├── config/
│   └── .Renviron               # API keys (gitignored)
├── logs/                       # Execution logs
└── README.md
```

## Experiments

| Experiment | Description | Hypothesis |
|------------|-------------|------------|
| **1A** | Between-subjects stock valuation with anchors | H1: Anchoring shifts LLM price targets |
| **1B** | Within-subjects sequential revision | H1b: Insufficient adjustment from anchor |
| **2** | Earnings forecast anchoring | H2: Anchoring generalizes to EPS estimates |
| **3** | Credit risk assessment anchoring | H2b: Anchoring affects default probability |

## Quick Start

```r
# Install dependencies
install.packages(c("httr2", "jsonlite", "tidyverse", "glue", "cli",
                    "digest", "optparse", "arrow"))

# Test a single prompt
Rscript R/anchoring_experiment.R --mode test --company 1 --anchor 52wk_high_30pct

# Run pilot (2 companies × 3 conditions × 1 model × 10 reps)
Rscript R/anchoring_experiment.R --mode pilot

# Run calibration (all companies, control only, all models, 4 temperatures)
Rscript R/anchoring_experiment.R --mode calibrate

# Run full experiment (batch 1 of 4)
Rscript R/anchoring_experiment.R --mode full --batch 1
```

## API Setup

Copy `config/.Renviron` to your project root and add your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

## Models Tested

- Claude 3.5 Sonnet (`claude-sonnet-4-20250514`)
- GPT-4o (`gpt-4o`)
- Gemini 2.0 Flash (`gemini-2.0-flash`)

# Algorithmic Anchoring

This repository contains the code and data necessary to replicate the study on algorithmic anchoring in Large Language Models (LLMs). The study investigates whether, and to what extent, LLMs are susceptible to anchoring biases when performing financial valuation tasks.

## Repository Structure

- `R/` - Contains all R source code:
  - `anchoring_experiment.R` - The main script to run the API experiments.
  - `analysis_pipeline.R` - The primary data analysis and figure generation script.
  - `scripts/` - Ad-hoc and supplementary analysis scripts.
  - `tests/` - Robustness checks and additional tests.
- `config/` - Configuration files (e.g., `.Renviron` for API keys).
- `data/` - Raw output and datasets.
- `results/` - Processed outputs, models, and generated figures.
- `*.bat` - Windows batch files to execute the pipeline.

## Prerequisites

To explicitly replicate this study, you will need **R 4.5.2** (or higher) and the following R packages installed:

```R
install.packages(c(
  "tidyverse", "broom", "sandwich", "lmtest", "boot", 
  "scales", "systemfit", "lfe", "car", "httr2", 
  "jsonlite", "glue", "cli", "digest", "optparse", 
  "clubSandwich"
))
```

If you plan to run the *data generation* step (the experiment itself), you will also need active API keys for the respective models being tested (e.g., OpenAI, Anthropic, Google).
Create an environment file at `config/.Renviron` with your API keys:
```
OPENAI_API_KEY="your_key_here"
ANTHROPIC_API_KEY="your_key_here"
GEMINI_API_KEY="your_key_here"
```

## Replication Instructions

### 1. Data Generation (Experiment Execution)
If you wish to re-run the full experiment and regenerate the API responses from the models, run the following batch script from the root of the project:
```cmd
run_all_batches.bat
```
*Note: This process runs synchronously for all models and can take several hours to complete.*

### 2. Data Analysis & Results
If you have the experimental data (or have just generated it), you can run the full analysis pipeline. This will process the model outputs, run the regressions, and generate the tables and figures used in the paper:
```cmd
analyze_all_batches.bat
```
The analysis scripts output their results to the `results/` folder, including standard `.txt` summaries and generated visualizations.

## Key Scripts

- **`R/anchoring_experiment.R`**: Core logic for prompting the models and logging their responses.
- **`R/analysis_pipeline.R`**: Cleans the text responses, extracts numerical valuations, runs statistical testing (OLS, SUR), and formats the output tables.

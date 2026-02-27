library(tidyverse)

# Load calibration data
cal <- read_csv("C:/Users/jgarcia/My Drive/Research/LLM - Algorithmic Anchoring/algorithmic-anchoring/results/exp1a_valuation/calibrate-exp1_valuation_20260220.csv", show_col_types = FALSE)

# Filter to control condition only, exclude FC02
# Note: The column name in your data is actually 'condition_id' instead of 'condition'
cal_ctrl <- cal %>%
  filter(condition_id == "control",
         company_id != "FC02",
         !is.na(point_estimate))

# Compute CV by company x model x temperature
cv_table <- cal_ctrl %>%
  group_by(company_id, model, temperature) %>%
  summarise(
    n = n(),
    mean_est = mean(point_estimate, na.rm = TRUE),
    sd_est = sd(point_estimate, na.rm = TRUE),
    cv = sd_est / mean_est,
    .groups = "drop"
  )

# Pivot to get T=0 and T=0.7 side by side
cv_wide <- cv_table %>%
  filter(temperature %in% c(0, 0.7)) %>%
  select(company_id, model, temperature, cv) %>%
  pivot_wider(
    names_from = temperature,
    values_from = cv,
    names_prefix = "cv_T"
  )

# Rename dynamically to avoid errors if a temperature isn't present
if ("cv_T0" %in% colnames(cv_wide) && "cv_T0.7" %in% colnames(cv_wide)) {
  cv_wide <- cv_wide %>%
    rename(cv_T07 = `cv_T0.7`) %>%
    mutate(ratio = cv_T07 / cv_T0)
} else if ("cv_T0.7" %in% colnames(cv_wide)) {
  cv_wide <- cv_wide %>% rename(cv_T07 = `cv_T0.7`)
}

# View by model
print("Raw CV wide by model:")
cv_wide %>%
  arrange(model, company_id) %>%
  print(n = 30)

# Compute medians per model
print("Medians per model:")
if ("ratio" %in% colnames(cv_wide)) {
  cv_wide %>%
    group_by(model) %>%
    summarise(
      median_cv_T0 = median(cv_T0, na.rm = TRUE),
      median_cv_T07 = median(cv_T07, na.rm = TRUE),
      median_ratio = median(ratio, na.rm = TRUE)
    ) %>%
    print()
} else {
  cv_wide %>%
    group_by(model) %>%
    summarise(
      median_cv_T0 = if("cv_T0" %in% cur_column()) median(cv_T0, na.rm = TRUE) else NA,
      median_cv_T07 = if("cv_T07" %in% cur_column()) median(cv_T07, na.rm = TRUE) else NA
    ) %>%
    print()
}

# --- Creating the Table 3A / 5A format you requested ---
print("----------- FORMATTED TABLE 3A / 5A -----------")

# Map models to short names
cv_formatted <- cv_wide %>%
  mutate(model_clean = case_when(
    str_detect(model, "claude") ~ "Claude",
    str_detect(model, "gemini") ~ "Gemini",
    str_detect(model, "o3-mini|gpt") ~ "GPT-5m",
    TRUE ~ model
  ))

# Reshape strictly for the table format
# It constructs Claude, Gemini, and GPT-5m columns side-by-side
table_wide <- cv_formatted %>%
  select(company_id, model_clean, contains("cv_T0"), contains("cv_T07"), contains("ratio")) %>%
  pivot_wider(
    names_from = model_clean,
    values_from = c(matches("cv_T0"), matches("cv_T07"), matches("ratio")),
    names_glue = "{model_clean}_{.value}"
  ) %>%
  arrange(company_id)

print(table_wide, width=Inf)

# Medians for the formatted table
median_row <- cv_formatted %>%
  group_by(model_clean) %>%
  summarise(
    cv_T0 = if("cv_T0" %in% names(.)) median(cv_T0, na.rm = TRUE) else NA,
    cv_T07 = if("cv_T07" %in% names(.)) median(cv_T07, na.rm = TRUE) else NA,
    ratio = if("ratio" %in% names(.)) median(ratio, na.rm = TRUE) else NA,
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = model_clean,
    values_from = c(cv_T0, cv_T07, ratio),
    names_glue = "{model_clean}_{.value}"
  ) %>%
  mutate(company_id = "Median")

print(median_row, width=Inf)

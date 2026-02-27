library(lmtest)
library(clubSandwich)
library(dplyr)
library(tibble)

# Create a tibble
df <- tibble(x=rnorm(100), y=rnorm(100), g=rep(1:10, 10))
df$x[1:10] <- NA

# Filter to test row name behavior with tibbles
h1_df <- df %>% filter(g != 5)

m_exclude <- lm(y ~ x, data=h1_df, na.action=na.exclude)

cluster_se_cr2 <- function(m, cluster_name, data) {
  # clubSandwich::vcovCR crashes if the model uses na.exclude because
  # residuals() returns a padded vector. Update to na.omit.
  m_omit <- update(m, na.action = na.omit)
  
  # The robust way to get the exact cluster variable matching the used observations:
  # Since updating na.action might re-evaluate data, it's safer to just drop NAs from the data
  # and pass that to vcovCR.
  
  # BUT Wait, if `data` is passed in, we can just use `model.frame`:
  used_idx <- as.numeric(row.names(model.frame(m_omit)))
  valid_clusters <- data[[cluster_name]][used_idx]
  
  vcov_cr2 <- clubSandwich::vcovCR(m_omit, cluster = valid_clusters, type = "CR2")
  clubSandwich::coef_test(m_omit, vcov = vcov_cr2, test = "Satterthwaite")
}

tryCatch({
  print(cluster_se_cr2(m_exclude, "g", h1_df))
}, error=function(e) print(e))

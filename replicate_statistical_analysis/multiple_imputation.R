library(mice)
library(dplyr)
library(readr)

set.seed(42)

data <- read_csv("output/replication_data.csv")

vars_to_keep <- c("claim_dummy", "republican", "independent", "senate",
                  "female", "age_std", "cc_combined_std", "ff_annual_avg_emplvl_per1k_std", "nominate_dim1_std", 
                  "year", "bioguide_id", "dem_pres_dummy")

data_vars <- data %>%
  select(all_of(vars_to_keep))

data_vars <- data_vars %>%
  mutate(year = as.factor(year),
         bioguide_id = as.factor(bioguide_id))

pred_matrix <- quickpred(data_vars, mincor = 0.1, minpuc = 0.5)

if("cc_combined_std" %in% rownames(pred_matrix)) {
  pred_matrix["cc_combined_std", "dem_pres_dummy"] <- 0
}

cat("Modified predictor matrix to remove dem_pres_dummy from cc variable models\n")
if("cc_combined_std" %in% rownames(pred_matrix)) {
  cat("cc_combined_std now has", sum(pred_matrix["cc_combined_std", ]), "predictors\n")
}

methods <- rep("pmm", ncol(data_vars))
names(methods) <- names(data_vars)

complete_vars <- names(data_vars)[sapply(data_vars, function(x) sum(is.na(x)) == 0)]
methods[complete_vars] <- ""

cat("Using 'norm' method for cc variables to improve convergence\n")
if("cc_combined_std" %in% names(methods)) {
  methods["cc_combined_std"] <- "norm"
}

cat("Starting imputation with improved settings...\n")
data_imputed <- mice(data_vars, 
                     pred = pred_matrix,
                     method = methods,
                     m = 5, 
                     maxit = 50, 
                     seed = 42,
                     printFlag = TRUE)

saveRDS(data_imputed, file = "output/data_imputed_combined.rds")

cat("\nImputation completed and saved to output/data_imputed_log.rds\n")
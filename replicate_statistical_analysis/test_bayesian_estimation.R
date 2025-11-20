library(brms)
library(dplyr)
library(readr)
library(posterior)
library(xtable)
library(tibble)
library(here)
library(rstan)

# ----------------------------------------------------------
# Test Version - Minimal Iterations, No File Overwrites
# ----------------------------------------------------------

cat("Running test version of Bayesian estimation...\n")
cat("This will use minimal iterations and not overwrite any output files.\n\n")

set.seed(42)

logit_to_prob <- function(intercept, coefficient = 0, x = 1) {
  log_odds <- intercept + (coefficient * x)
  prob <- 1/(1 + exp(-log_odds))
  return(prob)
}

bformula <- bf(
  claim_dummy ~ 1 + republican + independent + senate + female + age_std + cc_combined_std + ff_annual_avg_emplvl_per1k_std + dem_pres_dummy + (1|year) + (1|bioguide_id)
)

init_convergence <- function() {
  list(
    Intercept = -2.2
  )
}

bprior_halfnormal = c(
  prior("student_t(4,0,2.5)", class = "b"),
  prior("student_t(4,0,2.5)", class = "Intercept"),
  prior("normal(0,1)", class = "sd", group = "bioguide_id", lb = 0),
  prior("normal(0,0.5)", class = "sd", group = "year", lb = 0)
)

# TEST VERSION - Minimal settings
fit_bayesian_model_test <- function(formula, data, prior = bprior_halfnormal) {
  brm_multiple(
    formula = formula,
    family = bernoulli(link = "logit"),
    data = data,
    prior = prior,
    chains = 1,
    threads = threading(2),  # Reduced from 4
    iter = 100,              # Reduced from 5000
    warmup = 50,             # Reduced from 2000
    seed = 42,
    thin = 1,                # Reduced from 2
    backend = "cmdstanr",
    init = init_convergence,
    control = list(
      adapt_delta = 0.95,
      max_treedepth = 10
    ),
    silent = 0,
    save_pars = save_pars(all = TRUE)
  )
}

rstan::rstan_options(auto_write = TRUE)
options(mc.cores = 2)  # Reduced cores for testing

# Check if imputed data exists
imputed_file <- "../data/output/data_imputed_combined.rds"
if (!file.exists(imputed_file)) {
  cat("Error:", imputed_file, "not found.\n")
  cat("Please run multiple_imputation.R first.\n")
  stop("Missing imputed data file")
}

cat("Loading imputed data...\n")
data_imputed <- readRDS(imputed_file)

# ----------------------------------------------------------
# Test full model (minimal iterations)
# ----------------------------------------------------------

cat("Testing full model fit (minimal iterations for testing only)...\n")
model_cards_test <- fit_bayesian_model_test(
  formula = bformula,
  data = data_imputed
)

cat("✓ Full model test completed successfully!\n")
cat("Summary of test model:\n")
print(summary(model_cards_test))

# ----------------------------------------------------------
# Test Republican model (minimal iterations)
# ----------------------------------------------------------

cat("\nTesting Republican model fit (minimal iterations for testing only)...\n")
data_imputed_rep <- data_imputed %>%
  filter(republican == 1)

bformula_rep <- bf(
  claim_dummy ~ 1 + senate + female + age_std + cc_combined_std + ff_annual_avg_emplvl_per1k_std + nominate_dim1_std + dem_pres_dummy + (1|year) + (1|bioguide_id)
)

model_cards_rep_test <- fit_bayesian_model_test(
  formula = bformula_rep,
  data = data_imputed_rep
)

cat("✓ Republican model test completed successfully!\n")
cat("Summary of test Republican model:\n")
print(summary(model_cards_rep_test))

# ----------------------------------------------------------
# Test table generation
# ----------------------------------------------------------

cat("\nTesting results table generation...\n")

full_summary <- summary(model_cards_test)$fixed
rep_summary <- summary(model_cards_rep_test)$fixed

full_df <- as.data.frame(full_summary) %>%
  rownames_to_column("Variable")

rep_df <- as.data.frame(rep_summary) %>%
  rownames_to_column("Variable")

merged_df <- full_df %>%
  full_join(rep_df, by = "Variable", suffix = c("_Full", "_Rep"))

table_data_detailed <- merged_df %>%
  transmute(
    Variable,
    Full_Model_Estimate = round(Estimate_Full, 3),
    Full_Model_CI       = ifelse(!is.na(`l-95% CI_Full`),
                                 paste0("[", round(`l-95% CI_Full`, 3), ", ", round(`u-95% CI_Full`, 3), "]"),
                                 NA),
    Rep_Model_Estimate  = round(Estimate_Rep, 3),
    Rep_Model_CI        = ifelse(!is.na(`l-95% CI_Rep`),
                                 paste0("[", round(`l-95% CI_Rep`, 3), ", ", round(`u-95% CI_Rep`, 3), "]"),
                                 NA)
  )

cat("✓ Table generation test completed successfully!\n")
cat("Test results table (first 5 rows):\n")
print(head(table_data_detailed, 5))

# ----------------------------------------------------------
# Test convergence diagnostics
# ----------------------------------------------------------

cat("\nTesting convergence diagnostics...\n")
m <- 5
draws <- as_draws_array(model_cards_test)
draws_per_dat <- lapply(1:m, \(i) subset_draws(draws, chain = i))
convergence_results <- lapply(draws_per_dat, summarise_draws, default_convergence_measures())

cat("✓ Convergence diagnostics test completed successfully!\n")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("✅ ALL TESTS PASSED!\n")
cat("The bayesian_estimation.R script should work correctly.\n")
cat("\nTest summary:\n")
cat("- Full model fitting: ✓\n")
cat("- Republican model fitting: ✓\n") 
cat("- Table generation: ✓\n")
cat("- Convergence diagnostics: ✓\n")
cat("\nNote: This test used minimal iterations (100 iter, 50 warmup)\n")
cat("The full script uses 5000 iterations with 2000 warmup for publication quality results.\n")
cat("No output files were overwritten during this test.\n")
cat(paste(rep("=", 60), collapse=""), "\n")
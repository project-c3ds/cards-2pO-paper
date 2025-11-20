library(brms)
library(dplyr)
library(readr)
library(posterior)
library(xtable)
library(tibble)
library(here)
library(rstan)

# ----------------------------------------------------------
# Initialize
# ----------------------------------------------------------

set.seed(42)

logit_to_prob <- function(intercept, coefficient = 0, x = 1) {
  log_odds <- intercept + (coefficient * x)
  prob <- 1/(1 + exp(-log_odds))
  return(prob)
}

bformula <- bf(
  claim_dummy ~ 1 + republican + independent + senate + female + age_std + cc_combined_std + ff_annual_avg_emplvl_per1k_std + dem_pres_dummy + total_word_count_std + (1|year) + (1|bioguide_id)
)

init_convergence <- function() {
  list(
    Intercept = -2.2
  )
}

control_convergence <- list(
  adapt_delta = 0.95,
  max_treedepth = 12
)

bprior_halfnormal = c(
  prior("student_t(4,0,2.5)", class = "b"),
  prior("student_t(4,0,2.5)", class = "Intercept"),
  prior("normal(0,1)", class = "sd", group = "bioguide_id", lb = 0),
  prior("normal(0,0.5)", class = "sd", group = "year", lb = 0)
)

fit_bayesian_model <- function(formula, data, prior = bprior_halfnormal) {
  brm_multiple(
    formula = formula,
    family = bernoulli(link = "logit"),
    data = data,
    prior = prior,
    chains = 1,
    threads = threading(4),
    iter = 5000,
    warmup = 2000,
    seed = 42,
    thin = 2,
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
options(mc.cores = parallel::detectCores())

data_imputed <- readRDS("data/output/data_imputed_combined.rds")

# ----------------------------------------------------------
# Fit full model
# ----------------------------------------------------------

cat("Fitting full model...\n")
model_cards <- fit_bayesian_model(
  formula = bformula,
  data = data_imputed
)

saveRDS(model_cards, file = "data/output/full_model.rds")

cat("View convergence metrics corrected for multiple imputation...\n")
m <- 5
draws <- as_draws_array(model_cards)
draws_per_dat <- lapply(1:m, \(i) subset_draws(draws, chain = i))
print(lapply(draws_per_dat, summarise_draws, default_convergence_measures()))

# ----------------------------------------------------------
# Fit Republican model
# ----------------------------------------------------------

cat("Fitting Republican model...\n")
data_imputed_rep <- data_imputed %>%
  filter(republican == 1)

bformula_rep <- bf(
  claim_dummy ~ 1 + senate + female + age_std + cc_combined_std + ff_annual_avg_emplvl_per1k_std + nominate_dim1_std + dem_pres_dummy + total_word_count_std + (1|year) + (1|bioguide_id)
)

model_cards_rep_imputed <- fit_bayesian_model(
  formula = bformula_rep,
  data = data_imputed_rep
)

saveRDS(model_cards_rep_imputed, file = "data/output/republican_model.rds")

cat("View convergence metrics corrected for multiple imputation...\n")
draws <- as_draws_array(model_cards_rep_imputed)
draws_per_dat <- lapply(1:m, \(i) subset_draws(draws, chain = i))
print(lapply(draws_per_dat, summarise_draws, default_convergence_measures()))

# ----------------------------------------------------------
# Make results table
# ----------------------------------------------------------

cat("Compare full and Republican models...\n")
full_summary <- summary(model_cards)$fixed
rep_summary <- summary(model_cards_rep_imputed)$fixed

print("Full model dimensions:")
print(dim(full_summary))
print("Full model column names:")
print(colnames(full_summary))
print("Full model row names:")
print(rownames(full_summary))

print("\nRepublican model dimensions:")
print(dim(rep_summary))
print("Republican model column names:")
print(colnames(rep_summary))
print("Republican model row names:")
print(rownames(rep_summary))

full_df <- as.data.frame(full_summary) %>%
  rownames_to_column("Variable")

rep_df <- as.data.frame(rep_summary) %>%
  rownames_to_column("Variable")

merged_df <- full_df %>%
  full_join(rep_df, by = "Variable", suffix = c("_Full", "_Rep"))

cat("Make detailed results table...\n")
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

latex_table_detailed <- xtable(table_data_detailed, 
                              caption = "Comparison of Full and Republican-Only Models with 95\\% Credible Intervals",
                              label   = "tab:model_comparison_detailed",
                              digits  = 3)

names(latex_table_detailed) <- c("Variable", 
                                "Estimate", "95\\% CI",
                                "Estimate", "95\\% CI")

addtorow_detailed <- list(
  pos = list(0, 0),
  command = c(
    "\\hline\n\\multicolumn{1}{c|}{} & \\multicolumn{2}{c|}{All parties} & \\multicolumn{2}{c}{Republicans} \\\\n",
    "\\hline\n"
  )
)

print(latex_table_detailed, 
      file = here("data/output/table2.tex"),
      include.rownames = FALSE,
      floating        = TRUE,
      booktabs        = TRUE,
      sanitize.text.function = function(x){x},
      caption.placement = "top",
      add.to.row       = addtorow_detailed,
      hline.after      = c(-1, nrow(table_data_detailed)))

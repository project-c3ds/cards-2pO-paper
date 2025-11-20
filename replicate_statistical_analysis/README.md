# Statistical Analysis Replication Scripts

This directory contains R scripts for replicating the statistical analysis from the CARDS paper on congressional climate contrarian speech patterns.

## Scripts

### `multiple_imputation.R`
Handles missing data imputation using the MICE package.

**Dependencies:**
- `mice` - Multiple imputation by chained equations
- `dplyr` - Data manipulation
- `readr` - Reading CSV files

**Outputs:**
- `output/data_imputed_combined.rds` - Imputed dataset

### `bayesian_estimation.R`
Performs Bayesian multilevel logistic regression analysis using brms.

**Dependencies:**
- `brms` - Bayesian regression models
- `dplyr` - Data manipulation
- `readr` - Reading CSV files
- `posterior` - Working with posterior draws
- `xtable` - LaTeX table generation
- `tibble` - Enhanced data frames
- `here` - Path management
- `rstan` - Stan interface

**Outputs:**
- Model fit objects (.rds files)
- LaTeX tables (.tex files)
- Model comparison tables (.csv files)

## Usage

1. First run the multiple imputation script:
```r
source("multiple_imputation.R")
```

2. Then run the Bayesian estimation script:
```r
source("bayesian_estimation.R")
```

## Data Requirements

- `replication_data/replication_data.csv` - Input dataset
- Ensure `output/` directory exists for saving results

## Notes

- Set seed for reproducibility (seed = 42)
- Scripts assume working directory is the project root
- Large model fitting may require substantial computational resources
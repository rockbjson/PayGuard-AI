# PayGuard AI - Statistical Analysis

This directory contains the research-analysis layer for PayGuard AI. It extends the hybrid rule-based and Isolation Forest detector with interpretable regression, Bayesian uncertainty estimation, supplementary evaluation figures, and financial simulation.

All analyses use synthetic data and are intended to demonstrate methodology rather than real-world fraud performance.

## Main analysis

Run from the project root:

```bat
python analysis/advanced_statistics.py
```

The script:

1. loads `data/synthetic_payments.csv`;
2. creates a stratified 70/30 training/test split;
3. fits the Isolation Forest using training data only;
4. evaluates the hybrid detector on held-out observations;
5. fits an L2-penalised logistic regression;
6. estimates 95% bootstrap intervals from 1,000 coefficient resamples;
7. fits a Bayesian logistic regression with PyMC;
8. evaluates held-out posterior predictive probabilities;
9. bootstraps aggregate review exposure over 20,000 simulations; and
10. runs 20,000-draw working-capital Monte Carlo scenarios for +10, +20, and +30 days of DPO.

The script writes its numerical summary to:

```text
analysis/results.json
```

## Logistic regression

The logistic model is used as an interpretable supervised analysis of the planted anomaly labels. Continuous features are standardised using training-set statistics and binary indicators retain their 0/1 form.

The analysis reports:

- coefficients;
- odds ratios;
- 95% nonparametric bootstrap intervals;
- McFadden pseudo-R² for model fit;
- held-out ROC-AUC; and
- held-out average precision.

L2 regularisation is used to improve stability where synthetic indicators produce very strong separation between normal and planted-anomaly observations.

## Bayesian logistic regression

The Bayesian model estimates posterior distributions for five standardised continuous predictors and three binary indicators. Weakly informative normal priors are used for the intercept and coefficients.

Posterior inference uses PyMC's No-U-Turn Sampler (NUTS). The analysis records:

- posterior means;
- posterior standard deviations;
- 95% credible intervals;
- probability that each coefficient is positive;
- R-hat convergence diagnostics;
- effective sample sizes;
- sampler divergences;
- held-out posterior predictive probabilities;
- ROC-AUC;
- average precision; and
- Brier score.

Generated sampler files such as NetCDF traces are local analysis artefacts and are intentionally excluded from Git by the repository `.gitignore`.

## Review-exposure bootstrap

For alerted observations in the held-out test set, the script resamples `review_exposure` values with replacement and sums them across 20,000 bootstrap simulations.

The output includes:

- mean aggregate review exposure;
- 5th percentile;
- median;
- 95th percentile;
- 99th percentile; and
- conditional mean above the 95th percentile.

These values describe the simulated distribution of **aggregate review exposure**. They are not presented as a calibrated fraud-loss VaR model because `review_exposure` can include explicitly illustrative scenario estimates.

## Working-capital Monte Carlo analysis

A separate Monte Carlo analysis evaluates illustrative +10, +20, and +30 day extensions to Days Payable Outstanding (DPO).

For each scenario, uncertain financial inputs are sampled repeatedly, including assumptions related to:

- cost of capital;
- early-payment-discount participation;
- discount rate;
- discount erosion; and
- supplier-friction costs.

The simulated net benefit is based on the financial value of retained working capital less estimated discount erosion and supplier-friction costs. Results include the mean, 10th percentile, median, 90th percentile, and proportion of simulated outcomes with positive net benefit.

The working-capital inputs are proof-of-concept assumptions and must be replaced with treasury-approved organisation-specific values before operational use.

## Supplementary figures

Run:

```bat
python analysis/additional_figures.py
```

This generates supplementary charts in `analysis/charts/`, including architecture, dataset composition, held-out score distributions, precision-recall analysis, confusion matrix, detection by anomaly type, threshold sensitivity, Bayesian diagnostics, and calibration.

## Excel evaluation workbooks

Run:

```bat
python analysis/export_for_xlsx.py
python analysis/build_workbooks.py
```

The first script exports the analysis inputs required by the workbook builder. The second produces the Excel evaluation workbooks in `excel/`.

The intermediate `analysis/xlsx_data/` directory is generated locally and intentionally excluded from Git.

## Reproducibility and caveats

Random seeds are fixed where applicable. All supplier records, labels, anomalies, exposure values, and treasury assumptions are synthetic or illustrative. Strong associations produced by variables such as bank-account change and currency mismatch partly reflect how controlled anomalies are injected by the generator and should not be interpreted as empirical estimates of real-world fraud risk.

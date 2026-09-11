# PayGuard AI - Technical Overview

## Executive summary

PayGuard AI is a proof-of-concept framework for prioritising corporate supplier payments that may require investigation. It combines transparent business-control rules with Isolation Forest, an unsupervised anomaly-detection model, and extends the detection layer with interpretable logistic regression, Bayesian uncertainty estimation, and financial simulation.

Every supplier and payment record in the repository is synthetic. The project demonstrates a reproducible analytical framework without claiming access to confidential organisational payment data or real-world fraud outcomes.

## Research objectives

The project investigates three connected questions:

1. Can transparent business controls and unsupervised anomaly detection be combined to improve identification of planted corporate-payment irregularities?
2. Can interpretable statistical modelling and Bayesian inference provide clearer information about feature relationships and predictive uncertainty?
3. Can financial simulation extend anomaly detection into illustrative treasury and working-capital decision support?

## Synthetic data design

The canonical dataset contains 5,600 synthetic supplier-payment transactions generated with a fixed random seed of 42. Transactions span six procurement categories and contain fictional purchase-order, invoice, payment, currency, payment-term, and supplier bank-account information.

Seven controlled anomaly scenarios are injected:

- exact duplicate invoices;
- near-duplicate invoices;
- overpayments;
- invoices exceeding purchase-order amounts;
- recent supplier bank-account changes;
- currency mismatches; and
- unusually rapid payments.

Weekend payment is not a planted anomaly category. It is derived from generated payment dates and retained as a contextual behavioural feature and low-severity rule.

Ground-truth labels enable controlled evaluation but should not be interpreted as evidence of confirmed fraud.

## Feature engineering

The analytical feature set includes:

- log payment amount;
- paid-to-invoice ratio;
- invoice-to-PO ratio;
- days to pay;
- payment-term deviation;
- weekend-payment indicator;
- recent bank-change indicator; and
- currency-mismatch indicator.

Ratio and timing features make relationships between raw transaction values explicit. Model-specific preprocessing parameters are estimated using training data and then applied to held-out observations to avoid preprocessing-related data leakage.

## Hybrid detection methodology

The rule engine applies interpretable financial controls and assigns prototype weights according to indicative severity. Isolation Forest is fitted to the eight engineered features after robust scaling and produces an anomaly measure that is converted into a percentile relative to the training-score distribution.

The hybrid score is:

```text
Risk Score = 0.72 × Rule Score + 0.28 × Isolation Forest Percentile
```

Transactions with a score of 45 or above are prioritised for review. The weights and operating threshold are prototype design choices and would require organisation-specific calibration before deployment.

The hybrid score is a prioritisation measure rather than a probability of fraud.

## Alert explanations and review exposure

Triggered business rules are retained as human-readable explanations. Where a transaction is unusual without a triggered deterministic rule, the framework can identify strongly deviating engineered features to provide additional investigative context.

Review exposure is a prioritisation aid:

- measurable overpayment exposure is derived from the amount paid above the invoice;
- candidate duplicate exposure uses the later duplicate candidate's payment value;
- direct exposure takes the larger directly observable amount; and
- alerts without direct measurable exposure receive an explicitly illustrative scenario amount.

The resulting `review_exposure` field is not a calibrated expected fraud loss and should not be interpreted as an accounting provision.

## Experimental evaluation

Formal analytical evaluation uses a stratified 70/30 training/test split. Isolation Forest is fitted on the training partition and applied to held-out observations. Detection performance is assessed using class-sensitive metrics including precision, recall, F1 score, specificity, false-positive rate, ROC-AUC, and average precision.

Performance is also evaluated separately by planted anomaly type so aggregate metrics do not conceal weaknesses on individual scenarios.

The Streamlit dashboard is an operational demonstration. Research conclusions should be based on the held-out analysis pipeline rather than interpreting dashboard metrics as expected production performance.

## Interpretable logistic regression

An L2-penalised logistic regression analyses associations between the engineered features and planted anomaly labels. Continuous features are standardised using training statistics, while binary indicators retain their 0/1 representation.

The analysis reports coefficients and odds ratios with 95% intervals obtained from 1,000 nonparametric bootstrap replications. McFadden pseudo-R² is used as a model-fit statistic, while held-out ROC-AUC and average precision assess predictive discrimination.

The supervised regression is an analytical layer and benchmark; it does not replace the operational hybrid rule-plus-Isolation-Forest detector.

## Bayesian uncertainty analysis

Bayesian logistic regression extends the point-estimate analysis by estimating posterior distributions for model parameters and transaction-level predictions. Weakly informative normal priors are used, and approximate posterior inference is performed using PyMC’s mean-field Automatic Differentiation Variational Inference (ADVI).

The analysis reports posterior means, credible intervals, coefficient-sign probabilities, convergence diagnostics, posterior predictive probabilities, and probabilistic performance measures including Brier score.

This layer is intended to show how uncertainty can be communicated alongside interpretable risk-factor analysis.

## Financial simulation

Two separate simulation analyses are included.

### Aggregate review-exposure bootstrap

Held-out flagged review-exposure values are resampled with replacement across 20,000 simulations. The resulting aggregate distribution is summarised using the mean, median, upper percentiles, and conditional mean above the 95th percentile.

Because review exposure can include illustrative scenario estimates, this analysis is described as a bootstrap distribution of aggregate review exposure rather than a calibrated future fraud-loss model.

### Working-capital Monte Carlo simulation

A separate 20,000-draw Monte Carlo model evaluates illustrative +10, +20, and +30 day extensions to Days Payable Outstanding. The model considers uncertain cost-of-capital, early-payment-discount, discount-erosion, and supplier-friction assumptions.

For each draw, the financial value of retained working capital is offset by simulated discount loss and supplier-friction costs to produce a net-benefit distribution. The assumptions are placeholders and require treasury-approved organisation-specific inputs before operational use.

## Ethics, governance, and controls

- A flagged payment is an investigative lead, not proof of fraud or wrongdoing.
- Human review should precede any intervention.
- Production data would require appropriate minimisation, encryption, access control, and retention policies.
- Rule changes, thresholds, model versions, and investigator decisions should be auditable.
- Performance should be monitored over time for changing transaction behaviour and emerging risk patterns.
- Any financial decision-support outputs require validation against authorised organisational assumptions.

## Limitations

The project uses synthetic data and deliberately planted anomaly scenarios, which are cleaner and more controlled than genuine fraud. Prototype weights, thresholds, and simulation assumptions are not empirically calibrated to a particular organisation. The current system also does not analyse invoice text, supplier networks, investigator feedback, or adversarial adaptation.

Consequently, the project demonstrates a methodology and software architecture rather than validated production fraud-detection performance.

## Future work

Future research could evaluate the framework using authorised anonymised payment data, chronological validation, investigator-confirmed outcomes, and explicit false-positive and financial-loss costs. Additional extensions could include entity-network features, invoice-text analysis, model calibration, organisation-specific threshold optimisation, and structured evaluation of explanation usefulness with finance professionals.

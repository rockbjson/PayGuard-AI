"""PayGuard AI advanced statistical analysis.

This script adds three analytical layers to the rule-based and Isolation
Forest detector:

1. An interpretable, supervised logistic-regression benchmark.
2. Bayesian logistic regression with ADVI for coefficient and predictive uncertainty.
3. Monte Carlo analyses for flagged-payment exposure and working capital.

The analysis uses synthetic data only. Financial assumptions are illustrative
and must not be interpreted as estimates for a real organisation.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pymc as pm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from payguard.detector import (  # noqa: E402
    _engineer_features,
    evaluate_alerts,
    evaluate_by_anomaly_type,
    fit_anomaly_detector,
    score_transactions,
)

warnings.filterwarnings("ignore")

CHART_DIR = ROOT / "analysis" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / "analysis" / "results.json"
POSTERIOR_PATH = ROOT / "analysis" / "bayesian_advi_posterior.nc"
ADVI_LOSS_PATH = ROOT / "analysis" / "bayesian_advi_loss.npy"

NAVY = "#11295B"
GOLD = "#C7A24A"
GREY = "#6B7280"
RED = "#B3392C"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "font.size": 10.5,
        "axes.titleweight": "bold",
    }
)

CONTINUOUS_COLUMNS = [
    "log_amount",
    "paid_to_invoice",
    "invoice_to_po",
    "days_to_pay",
    "terms_deviation",
]

BAYES_COLUMNS = CONTINUOUS_COLUMNS + [
    "weekend",
]

BAYES_EXCLUDED_SEPARATING_COLUMNS = [
    "bank_change",
    "currency_mismatch",
]



def _standardise_from_training(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standardise continuous columns using training statistics only."""
    means = train_features[CONTINUOUS_COLUMNS].mean()
    standard_deviations = train_features[CONTINUOUS_COLUMNS].std().replace(0, 1)

    train_standardised = train_features.copy()
    test_standardised = test_features.copy()

    train_standardised[CONTINUOUS_COLUMNS] = (
        train_standardised[CONTINUOUS_COLUMNS] - means
    ) / standard_deviations

    test_standardised[CONTINUOUS_COLUMNS] = (
        test_standardised[CONTINUOUS_COLUMNS] - means
    ) / standard_deviations

    return train_standardised, test_standardised, means, standard_deviations



def _bootstrap_logistic_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_bootstraps: int = 1_000,
    random_state: int = 7,
) -> np.ndarray:
    """Obtain nonparametric bootstrap coefficient samples."""
    rng = np.random.default_rng(random_state)
    x_array = x_train.to_numpy()
    y_array = y_train.to_numpy()
    n_rows = len(y_array)

    successful: list[np.ndarray] = []
    attempts = 0
    maximum_attempts = n_bootstraps * 3
    audit = {"one_class": 0, "value_error": 0, "nonconverged": 0, "nonfinite": 0}

    while len(successful) < n_bootstraps and attempts < maximum_attempts:
        attempts += 1
        indices = rng.integers(0, n_rows, size=n_rows)
        sampled_y = y_array[indices]

        if np.unique(sampled_y).size < 2:
            audit["one_class"] += 1
            continue

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1_000,
            solver="lbfgs",
        )

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_array[indices], sampled_y)
        except ValueError:
            audit["value_error"] += 1
            continue

        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            audit["nonconverged"] += 1
            continue
        if not np.isfinite(model.coef_).all():
            audit["nonfinite"] += 1
            continue

        successful.append(model.coef_[0])

    _bootstrap_logistic_coefficients.last_audit = {
        "attempts": attempts, "accepted": len(successful), "rejected": audit,
        "random_seed": random_state, "sampling_unit": "training transaction row",
        "rows_per_resample": n_rows, "preprocessing": "fixed training scaling",
        "interval_method": "percentile 2.5 and 97.5, exponentiated for odds ratios",
        "separation": "L2 regularisation; no explicit separation test",
    }

    if len(successful) < n_bootstraps:
        raise RuntimeError(
            "Unable to obtain the requested number of valid bootstrap fits."
        )

    return np.vstack(successful)



def _mcfadden_pseudo_r2(
    y_observed: pd.Series,
    predicted_probability: np.ndarray,
    null_base_rate: float,
) -> float:
    """Calculate McFadden pseudo-R-squared on the model-fitting sample."""
    epsilon = 1e-12
    y_array = y_observed.to_numpy()

    model_log_likelihood = np.sum(
        y_array * np.log(predicted_probability + epsilon)
        + (1 - y_array) * np.log(1 - predicted_probability + epsilon)
    )

    null_log_likelihood = np.sum(
        y_array * np.log(null_base_rate + epsilon)
        + (1 - y_array) * np.log(1 - null_base_rate + epsilon)
    )

    return float(1 - model_log_likelihood / null_log_likelihood)



def _safe_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities without numerical overflow."""
    clipped = np.clip(logits, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))



def _save_roc_chart(
    y_test: pd.Series,
    logistic_scores: np.ndarray,
    hybrid_scores: pd.Series,
    isolation_scores: pd.Series,
    rule_scores: pd.Series,
) -> None:
    """Save the held-out ROC comparison chart."""
    figure, axis = plt.subplots(figsize=(6.2, 5))

    series = [
    ("Logistic regression", logistic_scores, NAVY, "-"),
    ("Hybrid rule + Isolation Forest", hybrid_scores, GOLD, "--"),
    ("Isolation Forest only", isolation_scores, GREY, ":"),
    ("Rule score only", rule_scores, RED, "-."),
    ]

    for label, scores, colour, line_style in series:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC={auc_value:.3f})",
            color=colour,
            linestyle=line_style,
            linewidth=2.2,
        )

    axis.plot([0, 1], [0, 1], "--", color="#BBBBBB", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "01_roc_comparison.png", dpi=600)
    plt.close(figure)



def _save_odds_ratio_chart(coefficient_table: pd.DataFrame) -> None:
    """Save the logistic-regression odds-ratio chart."""
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    positions = np.arange(len(coefficient_table))

    axis.errorbar(
        coefficient_table["odds_ratio"],
        positions,
        xerr=[
            coefficient_table["odds_ratio"] - coefficient_table["ci_low"],
            coefficient_table["ci_high"] - coefficient_table["odds_ratio"],
        ],
        fmt="o",
        color=NAVY,
        ecolor=GOLD,
        elinewidth=2,
        capsize=3,
        markersize=6,
    )

    axis.axvline(1.0, color="#999999", linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(coefficient_table.index)
    axis.set_xscale("log")
    axis.set_xlabel("Odds ratio (log scale), bootstrap 95% interval")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "02_odds_ratios.png", dpi=600)
    plt.close(figure)



def _save_bayesian_forest_chart(
    posterior_beta: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save approximate posterior coefficient means and 95% credible intervals."""
    posterior_mean = posterior_beta.mean(axis=0)
    posterior_low = np.percentile(posterior_beta, 2.5, axis=0)
    posterior_high = np.percentile(posterior_beta, 97.5, axis=0)
    order = np.argsort(posterior_mean)

    ordered_mean = posterior_mean[order]
    ordered_low = posterior_low[order]
    ordered_high = posterior_high[order]
    positions = np.arange(len(feature_names))

    figure, axis = plt.subplots(figsize=(6.8, 4.4))

    # Draw interval endpoints directly rather than constructing xerr around
    # the posterior mean. For skewed approximate posteriors, the mean can
    # occasionally fall outside an equal-tail percentile interval.
    axis.hlines(
        y=positions,
        xmin=ordered_low,
        xmax=ordered_high,
        color=GOLD,
        linewidth=2,
    )
    axis.scatter(
        ordered_mean,
        positions,
        color=NAVY,
        s=36,
        zorder=3,
    )

    axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([feature_names[index] for index in order])
    axis.set_xlabel("Approximate posterior coefficient, 95% credible interval")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "03_bayesian_posterior_forest.png", dpi=600)
    plt.close(figure)


def _save_top_bayesian_chart(top_summary: pd.DataFrame) -> None:
    """Save approximate posterior probabilities for the ten highest-risk test rows."""
    means = top_summary["bayes_prob_mean"].to_numpy()
    lows = top_summary["bayes_prob_ci_low"].to_numpy()
    highs = top_summary["bayes_prob_ci_high"].to_numpy()
    order = np.argsort(-means)

    ordered_means = means[order]
    ordered_lows = lows[order]
    ordered_highs = highs[order]
    positions = np.arange(len(order))

    figure, axis = plt.subplots(figsize=(7, 4.2))

    # Draw the percentile interval directly. This avoids negative xerr values
    # when a skewed approximate posterior has a mean outside its equal-tail
    # percentile interval.
    axis.hlines(
        y=positions,
        xmin=ordered_lows,
        xmax=ordered_highs,
        color=GOLD,
        linewidth=2,
    )
    axis.scatter(
        ordered_means,
        positions,
        color=RED,
        s=36,
        zorder=3,
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(top_summary["transaction_id"].to_numpy()[order])
    axis.set_xlabel(
        "Approximate posterior anomaly probability, 95% credible interval"
    )
    axis.set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "04_bayesian_top10_predictions.png", dpi=600)
    plt.close(figure)


def _save_exposure_chart(
    simulated_totals: np.ndarray,
    mean_total: float,
    percentile_95: float,
    percentile_99: float,
) -> None:
    """Save the flagged-exposure bootstrap distribution."""
    figure, axis = plt.subplots(figsize=(7, 4.4))
    axis.hist(simulated_totals, bins=60, color=NAVY, alpha=0.85)
    axis.axvline(
        mean_total,
        color=GOLD,
        linewidth=2,
        label=f"Mean = {mean_total:,.0f} SMU",
    )
    axis.axvline(
        percentile_95,
        color=RED,
        linewidth=2,
        linestyle="--",
        label=f"95th percentile = {percentile_95:,.0f} SMU",
    )
    axis.axvline(
        percentile_99,
        color="#7A1F14",
        linewidth=2,
        linestyle=":",
        label=f"99th percentile = {percentile_99:,.0f} SMU",
    )
    axis.set_xlabel("Aggregate review exposure among held-out alerts (SMU)")
    axis.set_ylabel("Simulation frequency")
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(
    CHART_DIR / "05_bootstrap_review_exposure.png",
    dpi=600,
)
    plt.close(figure)



def _save_working_capital_chart(results: dict[int, np.ndarray]) -> None:
    """Save simulated net-benefit distributions for DPO scenarios."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    styles = [
    (GOLD, "-"),
    (NAVY, "--"),
    (RED, "-."),
    ]

    for (days, values), (colour, line_style) in zip(
        results.items(), styles
    ):
        axis.hist(
            values,
            bins=70,
            histtype="step",
            linewidth=2.2,
            linestyle=line_style,
            label=f"DPO +{days} days (mean={np.mean(values):,.0f} SMU)",
            color=colour,
            density=True,
        )

    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_xlabel("Simulated annual net working-capital benefit (SMU)")
    axis.set_ylabel("Density")
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "06_montecarlo_working_capital.png", dpi=600)
    plt.close(figure)



def _sanitize_for_json(value: Any) -> Any:
    """Convert NumPy/pandas values and non-finite numbers into JSON-safe forms."""
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if pd.isna(value):
        return None
    return value



def main() -> None:
    """Run the complete held-out statistical analysis."""
    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 0. Data loading and held-out detector evaluation
    # ------------------------------------------------------------------
    print("Loading data and creating held-out partitions ...")
    data = pd.read_csv(ROOT / "data" / "synthetic_payments.csv")
    data["invoice_date"] = pd.to_datetime(data["invoice_date"])
    data["payment_date"] = pd.to_datetime(data["payment_date"])

    train_index, test_index = train_test_split(
        data.index,
        test_size=0.30,
        random_state=42,
        stratify=data["is_anomaly"],
    )

    train_data = data.loc[train_index].copy()
    test_data = data.loc[test_index].copy()

    fitted_detector = fit_anomaly_detector(
        train_data,
        contamination=0.05,
        random_state=42,
    )

    # Score the complete history so deterministic duplicate rules can compare
    # test invoices with earlier/reference invoices. Isolation Forest remains
    # fitted only on the training partition.
    scored_all = score_transactions(
        data,
        fitted_detector=fitted_detector,
        training_data=train_data,
    )
    scored_test = scored_all.loc[test_index].copy()
    scored_test.to_csv(ROOT / "analysis" / "held_out_scores.csv", index=True, index_label="source_row_index")

    baseline_metrics = evaluate_alerts(scored_test)
    anomaly_type_metrics = evaluate_by_anomaly_type(scored_test)

    results["baseline_hybrid_metrics"] = baseline_metrics
    results["detection_by_anomaly_type"] = anomaly_type_metrics.to_dict(
        orient="records"
    )
    results["n_transactions"] = int(len(data))
    results["n_training_transactions"] = int(len(train_data))
    results["n_test_transactions"] = int(len(test_data))
    results["n_anomalies"] = int(data["is_anomaly"].sum())
    results["base_rate_pct"] = float(data["is_anomaly"].mean() * 100)

    print(
        f"  total={len(data):,}; train={len(train_data):,}; "
        f"test={len(test_data):,}; anomaly rate={results['base_rate_pct']:.2f}%"
    )
    print(f"  held-out hybrid metrics: {baseline_metrics}")

    # ------------------------------------------------------------------
    # 1. Logistic regression
    # ------------------------------------------------------------------
    print("\n[1/3] Fitting logistic regression ...")

    train_features = _engineer_features(train_data)
    test_features = _engineer_features(test_data)

    x_train, x_test, feature_means, feature_stds = _standardise_from_training(
        train_features,
        test_features,
    )

    y_train = train_data["is_anomaly"].astype(int)
    y_test = test_data["is_anomaly"].astype(int)

    logistic_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2_000,
        solver="lbfgs",
    )
    logistic_model.fit(x_train, y_train)

    # McFadden pseudo-R² is a model-fit measure and is therefore calculated
    # from probabilities on the same training sample used to fit the model.
    logistic_training_probability = logistic_model.predict_proba(x_train)[:, 1]

    # Held-out probabilities are used for predictive evaluation metrics.
    logistic_probability = logistic_model.predict_proba(x_test)[:, 1]
    logistic_auc = roc_auc_score(y_test, logistic_probability)
    logistic_average_precision = average_precision_score(
        y_test,
        logistic_probability,
    )

    hybrid_test_scores = scored_test.loc[x_test.index, "risk_score"]
    isolation_test_scores = scored_test.loc[x_test.index, "ml_anomaly_score"]
    rule_test_scores = scored_test.loc[x_test.index, "rule_score"]
    rule_auc = roc_auc_score(y_test, rule_test_scores)
    hybrid_auc = roc_auc_score(y_test, hybrid_test_scores)
    isolation_auc = roc_auc_score(y_test, isolation_test_scores)

    bootstrap_coefficients = _bootstrap_logistic_coefficients(
        x_train,
        y_train,
        n_bootstraps=1_000,
        random_state=7,
    )

    point_coefficients = logistic_model.coef_[0]
    coefficient_low = np.percentile(bootstrap_coefficients, 2.5, axis=0)
    coefficient_high = np.percentile(bootstrap_coefficients, 97.5, axis=0)

    coefficient_table = pd.DataFrame(
        {
            "coefficient": point_coefficients,
            "odds_ratio": np.exp(point_coefficients),
            "ci_low": np.exp(coefficient_low),
            "ci_high": np.exp(coefficient_high),
        },
        index=x_train.columns,
    ).sort_values("odds_ratio", ascending=False)

    pseudo_r2 = _mcfadden_pseudo_r2(
        y_train,
        logistic_training_probability,
        float(y_train.mean()),
    )

    results["regression"] = {
        "model": "L2-penalised logistic regression",
        "n_bootstrap_samples": 1_000,
        "bootstrap_audit": _bootstrap_logistic_coefficients.last_audit,
        "auc_logit": float(logistic_auc),
        "auc_hybrid_rule_if": float(hybrid_auc),
        "auc_isolation_forest_only": float(isolation_auc),
        "auc_rule_score_only": float(rule_auc),
        "average_precision_rule_score_only": float(average_precision_score(y_test, rule_test_scores)),
        "average_precision_logit": float(logistic_average_precision),
        "mcfadden_pseudo_r2": float(pseudo_r2),
        "intercept": float(logistic_model.intercept_[0]),
        "training_means": feature_means.to_dict(),
        "training_standard_deviations": feature_stds.to_dict(),
        "coefficients": (
            coefficient_table.reset_index()
            .rename(columns={"index": "feature"})
            .to_dict(orient="records")
        ),
    }

    print(
        f"  Logistic AUC={logistic_auc:.3f}; "
        f"Hybrid AUC={hybrid_auc:.3f}; "
        f"Isolation Forest AUC={isolation_auc:.3f}"
    )

    _save_roc_chart(
        y_test,
        logistic_probability,
        hybrid_test_scores,
        isolation_test_scores,
        rule_test_scores,
    )
    _save_odds_ratio_chart(coefficient_table)

    # ------------------------------------------------------------------
    # 2. Bayesian logistic regression
    # ------------------------------------------------------------------
    print("\n[2/3] Fitting Bayesian logistic regression ...")

    x_bayes_train = x_train[BAYES_COLUMNS].to_numpy()
    x_bayes_test = x_test[BAYES_COLUMNS].to_numpy()
    y_bayes_train = y_train.to_numpy()
    y_bayes_test = y_test.to_numpy()

    advi_max_iterations = 30_000
    posterior_draws = 4_000
    convergence_callback = pm.callbacks.CheckParametersConvergence(
        every=100,
        tolerance=1e-4,
        diff="absolute",
    )

    if os.environ.get("PAYGUARD_REUSE_POSTERIOR") == "1":
        trace = az.from_netcdf(POSTERIOR_PATH)
        advi_loss = np.load(ADVI_LOSS_PATH)
    else:
        with pm.Model() as bayesian_model:
            intercept = pm.Normal("intercept", mu=0, sigma=2.5)
            beta = pm.Normal("beta", mu=0, sigma=1.5, shape=len(BAYES_COLUMNS))
            logits = intercept + pm.math.dot(x_bayes_train, beta)
            pm.Bernoulli("obs", logit_p=logits, observed=y_bayes_train)
    
            approximation = pm.fit(
                n=advi_max_iterations,
                method="advi",
                random_seed=42,
                progressbar=True,
                callbacks=[convergence_callback],
            )
            trace = approximation.sample(
                draws=posterior_draws,
                random_seed=42,
                return_inferencedata=True,
            )
    
        az.to_netcdf(trace, POSTERIOR_PATH)
        advi_loss = np.asarray(approximation.hist, dtype=float)
        np.save(ADVI_LOSS_PATH, advi_loss)
    advi_iterations_completed = int(len(advi_loss))
    posterior_beta = trace.posterior["beta"].values.reshape(-1, len(BAYES_COLUMNS))
    posterior_intercept = trace.posterior["intercept"].values.reshape(-1)

    beta_summary = []
    for feature_index, feature_name in enumerate(BAYES_COLUMNS):
        draws = posterior_beta[:, feature_index]
        beta_summary.append(
            {
                "feature": feature_name,
                "mean": float(draws.mean()),
                "standard_deviation": float(draws.std()),
                "credible_interval_low": float(np.percentile(draws, 2.5)),
                "credible_interval_high": float(np.percentile(draws, 97.5)),
                "probability_effect_positive": float((draws > 0).mean()),
            }
        )

    test_logits = posterior_intercept[:, None] + np.einsum(
        "sf,nf->sn",
        posterior_beta,
        x_bayes_test,
    )
    test_probability_samples = _safe_sigmoid(test_logits)
    test_probability_mean = test_probability_samples.mean(axis=0)

    bayesian_auc = roc_auc_score(y_bayes_test, test_probability_mean)
    bayesian_average_precision = average_precision_score(
        y_bayes_test,
        test_probability_mean,
    )
    bayesian_brier = brier_score_loss(
        y_bayes_test,
        test_probability_mean,
    )

    top_index = (
        scored_test["risk_score"].sort_values(ascending=False).head(10).index
    )
    top_positions = x_test.index.get_indexer(top_index)
    if np.any(top_positions < 0):
        raise RuntimeError("Could not align high-risk test transactions.")

    top_probability_samples = test_probability_samples[:, top_positions]
    top_summary = pd.DataFrame(
        {
            "transaction_id": scored_test.loc[
                top_index,
                "transaction_id",
            ].to_numpy(),
            "supplier_category": scored_test.loc[
                top_index,
                "supplier_category",
            ].to_numpy(),
            "risk_score_hybrid": scored_test.loc[
                top_index,
                "risk_score",
            ].to_numpy(),
            "bayes_prob_mean": top_probability_samples.mean(axis=0),
            "bayes_prob_ci_low": np.percentile(
                top_probability_samples,
                2.5,
                axis=0,
            ),
            "bayes_prob_ci_high": np.percentile(
                top_probability_samples,
                97.5,
                axis=0,
            ),
            "actually_anomalous": scored_test.loc[
                top_index,
                "is_anomaly",
            ].to_numpy(),
        }
    )

    results["bayesian"] = {
        "model": "Bayesian logistic regression using mean-field ADVI",
        "features": BAYES_COLUMNS,
        "excluded_complete_separation_features": BAYES_EXCLUDED_SEPARATING_COLUMNS,
        "priors": {
            "intercept": "Normal(0, 2.5)",
            "coefficients": "Normal(0, 1.5)",
        },
        "inference_method": "Automatic Differentiation Variational Inference (ADVI)",
        "advi_max_iterations": advi_max_iterations,
        "advi_iterations_completed": advi_iterations_completed,
        "advi_final_loss": float(advi_loss[-1]),
        "posterior_draws": posterior_draws,
        "held_out_auc": float(bayesian_auc),
        "held_out_average_precision": float(bayesian_average_precision),
        "held_out_brier_score": float(bayesian_brier),
        "coefficient_summary": beta_summary,
        "top10_posterior_predictive": top_summary.to_dict(orient="records"),
    }

    print(
        f"  Bayesian held-out AUC={bayesian_auc:.3f}; "
        f"AP={bayesian_average_precision:.3f}; "
        f"Brier={bayesian_brier:.3f}; "
        f"ADVI iterations={advi_iterations_completed:,}"
    )

    _save_bayesian_forest_chart(posterior_beta, BAYES_COLUMNS)
    _save_top_bayesian_chart(top_summary)

    # ------------------------------------------------------------------
    # 3a. Bootstrap distribution of aggregate flagged exposure
    # ------------------------------------------------------------------
    print("\n[3/3] Running Monte Carlo simulations ...")

    flagged_test = scored_test[scored_test["alert"]].copy()
    if flagged_test.empty:
        raise RuntimeError(
            "No held-out payments were flagged, so exposure simulation cannot run."
        )

    exposure_values = flagged_test["review_exposure"].to_numpy(dtype=float)
    n_flagged = len(exposure_values)
    n_exposure_simulations = 20_000
    rng = np.random.default_rng(42)

    simulated_totals = np.empty(n_exposure_simulations)
    for simulation_index in range(n_exposure_simulations):
        sampled_exposure = rng.choice(
            exposure_values,
            size=n_flagged,
            replace=True,
        )
        simulated_totals[simulation_index] = sampled_exposure.sum()

    mean_total = float(simulated_totals.mean())
    percentile_95 = float(np.percentile(simulated_totals, 95))
    percentile_99 = float(np.percentile(simulated_totals, 99))
    conditional_mean_95 = float(
        simulated_totals[simulated_totals >= percentile_95].mean()
    )

    results["bootstrap_flagged_exposure"] = {
        "interpretation": (
            "Bootstrap distribution of aggregate review exposure among "
            "held-out flagged payments; this is not a calibrated fraud-loss model."
        ),
        "n_simulations": n_exposure_simulations,
        "n_flagged_payments": int(n_flagged),
        "mean_total_exposure_usd": mean_total,
        "p5_usd": float(np.percentile(simulated_totals, 5)),
        "median_usd": float(np.percentile(simulated_totals, 50)),
        "percentile_95_usd": percentile_95,
        "percentile_99_usd": percentile_99,
        "conditional_mean_above_95_usd": conditional_mean_95,
    }

    print(
        f"  Flagged exposure: mean=${mean_total:,.0f}; "
        f"P95=${percentile_95:,.0f}; "
        f"P99=${percentile_99:,.0f}; "
        f"conditional mean above P95=${conditional_mean_95:,.0f}"
    )

    _save_exposure_chart(
        simulated_totals,
        mean_total,
        percentile_95,
        percentile_99,
    )

    # ------------------------------------------------------------------
    # 3b. Illustrative working-capital simulation
    # ------------------------------------------------------------------
    # Ground-truth normal labels are available only because the data are
    # synthetic. This is therefore explicitly an oracle proof-of-concept.
    clean = data[data["is_anomaly"] == 0].copy()

    minimum_payment_date = clean["payment_date"].min()
    maximum_payment_date = clean["payment_date"].max()
    date_span_days = max(
        int((maximum_payment_date - minimum_payment_date).days) + 1,
        1,
    )
    years_represented = date_span_days / 365.25
    annualised_spend = float(clean["amount_usd"].sum() / years_represented)

    n_working_capital_simulations = 20_000
    dpo_extensions = [10, 20, 30]
    working_capital_results: dict[int, np.ndarray] = {}

    for extension_days in dpo_extensions:
        cost_of_capital = rng.normal(
            0.09,
            0.015,
            n_working_capital_simulations,
        ).clip(0.03, 0.18)

        discount_capture_today = rng.beta(
            6,
            14,
            n_working_capital_simulations,
        )
        discount_rate = rng.normal(
            0.02,
            0.004,
            n_working_capital_simulations,
        ).clip(0.005, 0.05)
        discount_erosion = rng.beta(
            4,
            6,
            n_working_capital_simulations,
        )
        supplier_friction_rate = rng.beta(
            2,
            40,
            n_working_capital_simulations,
        )
        supplier_friction_cost = rng.normal(
            0.015,
            0.005,
            n_working_capital_simulations,
        ).clip(0.0, 0.05)

        working_capital_freed = annualised_spend * extension_days / 365.0
        capital_benefit = working_capital_freed * cost_of_capital

        eligible_discount_spend = annualised_spend * discount_capture_today
        discount_lost = (
            eligible_discount_spend * discount_rate * discount_erosion
        )
        friction_cost = (
            annualised_spend
            * supplier_friction_rate
            * supplier_friction_cost
        )

        net_benefit = capital_benefit - discount_lost - friction_cost
        working_capital_results[extension_days] = net_benefit

    results["monte_carlo_working_capital"] = {
        "interpretation": (
            "Illustrative synthetic oracle scenario using known normal labels. "
            "Assumptions are not calibrated to a real organisation."
        ),
        "n_simulations": n_working_capital_simulations,
        "years_represented": float(years_represented),
        "annualised_spend_usd": annualised_spend,
        "assumptions": {
            "cost_of_capital": "Clipped Normal(mean=9%, sd=1.5%, bounds=3%-18%)",
            "discount_capture_share": "Beta(6, 14)",
            "discount_rate": "Clipped Normal(mean=2%, sd=0.4%, bounds=0.5%-5%)",
            "discount_erosion": "Beta(4, 6)",
            "supplier_friction_rate": "Beta(2, 40)",
            "supplier_friction_cost": "Clipped Normal(mean=1.5%, sd=0.5%, bounds=0%-5%)",
        },
        "scenarios": {
            str(days): {
                "mean_net_benefit_usd": float(np.mean(values)),
                "p10_usd": float(np.percentile(values, 10)),
                "p50_usd": float(np.percentile(values, 50)),
                "p90_usd": float(np.percentile(values, 90)),
                "prob_net_positive": float((values > 0).mean()),
            }
            for days, values in working_capital_results.items()
        },
    }

    for days, values in working_capital_results.items():
        print(
            f"  DPO +{days} days: mean net benefit=${np.mean(values):,.0f}; "
            f"P(net positive)={(values > 0).mean():.1%}"
        )

    _save_working_capital_chart(working_capital_results)

    with RESULTS_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(
            _sanitize_for_json(results),
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDone. Charts: {CHART_DIR}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

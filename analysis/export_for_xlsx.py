"""Export held-out model inputs and results for the Excel evaluation workbooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "xlsx_data"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from payguard.detector import (  # noqa: E402
    _engineer_features,
    fit_anomaly_detector,
    score_transactions,
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


def _standardise(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    means = train_features[CONTINUOUS_COLUMNS].mean()
    stds = train_features[CONTINUOUS_COLUMNS].std().replace(0, 1)

    train_scaled = train_features.copy()
    test_scaled = test_features.copy()
    train_scaled[CONTINUOUS_COLUMNS] = (
        train_scaled[CONTINUOUS_COLUMNS] - means
    ) / stds
    test_scaled[CONTINUOUS_COLUMNS] = (
        test_scaled[CONTINUOUS_COLUMNS] - means
    ) / stds
    return train_scaled, test_scaled, means, stds


def _bootstrap_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_bootstraps: int = 1_000,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    coefficients: list[np.ndarray] = []
    x_array = x_train.to_numpy()
    y_array = y_train.to_numpy()

    while len(coefficients) < n_bootstraps:
        indices = rng.integers(0, len(y_array), len(y_array))
        sampled_y = y_array[indices]
        if np.unique(sampled_y).size < 2:
            continue
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1_000,
            solver="lbfgs",
        )
        model.fit(x_array[indices], sampled_y)
        coefficients.append(model.coef_[0])

    return np.vstack(coefficients)


def main() -> None:
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

    fitted_detector = fit_anomaly_detector(train_data, contamination=0.05)
    scored_all = score_transactions(
        data,
        fitted_detector=fitted_detector,
        training_data=train_data,
    )
    scored_test = scored_all.loc[test_index].copy()

    train_features = _engineer_features(train_data)
    test_features = _engineer_features(test_data)
    x_train, x_test, means, stds = _standardise(train_features, test_features)
    y_train = train_data["is_anomaly"].astype(int)
    y_test = test_data["is_anomaly"].astype(int)

    logistic = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2_000,
        solver="lbfgs",
    ).fit(x_train, y_train)

    training_probabilities = logistic.predict_proba(x_train)[:, 1]
    probabilities = logistic.predict_proba(x_test)[:, 1]
    bootstrap = _bootstrap_coefficients(x_train, y_train)
    feature_order = list(x_train.columns)

    # McFadden pseudo-R² is a model-fit statistic, so it is calculated on
    # the same training sample used to estimate the logistic model. Held-out
    # predictive performance remains reported separately through AUC and AP.
    epsilon = 1e-12
    model_ll = np.sum(
        y_train * np.log(training_probabilities + epsilon)
        + (1 - y_train) * np.log(1 - training_probabilities + epsilon)
    )
    null_probability = float(y_train.mean())
    null_ll = np.sum(
        y_train * np.log(null_probability + epsilon)
        + (1 - y_train) * np.log(1 - null_probability + epsilon)
    )

    regression = {
        "features": feature_order,
        "intercept": float(logistic.intercept_[0]),
        "coef": dict(zip(feature_order, map(float, logistic.coef_[0]))),
        "ci_lo": dict(
            zip(feature_order, map(float, np.percentile(bootstrap, 2.5, axis=0)))
        ),
        "ci_hi": dict(
            zip(feature_order, map(float, np.percentile(bootstrap, 97.5, axis=0)))
        ),
        "means": {name: float(means[name]) for name in CONTINUOUS_COLUMNS},
        "stds": {name: float(stds[name]) for name in CONTINUOUS_COLUMNS},
        "cont": CONTINUOUS_COLUMNS,
        "metrics": {
            "auc": float(roc_auc_score(y_test, probabilities)),
            "ap": float(average_precision_score(y_test, probabilities)),
            "pseudo_r2": float(1 - model_ll / null_ll),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
    }
    (OUT / "regression.json").write_text(json.dumps(regression, indent=2))

    validation = test_features.copy()
    validation["is_anomaly"] = y_test
    validation["pred_prob"] = probabilities
    validation["hybrid_risk_score"] = scored_test["risk_score"]
    validation.head(40).to_csv(
        OUT / "regression_validation_sample.csv",
        index=False,
    )

    posterior_path = ROOT / "analysis" / "bayesian_advi_posterior.nc"
    loss_path = ROOT / "analysis" / "bayesian_advi_loss.npy"
    if not posterior_path.exists() or not loss_path.exists():
        raise FileNotFoundError(
            "ADVI posterior/loss history not found. "
            "Run analysis/advanced_statistics.py first."
        )

    trace = az.from_netcdf(posterior_path)
    advi_loss = np.load(loss_path)

    posterior_beta = trace.posterior["beta"].values.reshape(-1, len(BAYES_COLUMNS))
    posterior_intercept = trace.posterior["intercept"].values.reshape(-1)
    rng = np.random.default_rng(7)
    sample_size = min(2_000, len(posterior_intercept))
    selected = rng.choice(len(posterior_intercept), size=sample_size, replace=False)

    draws = pd.DataFrame(posterior_beta[selected], columns=BAYES_COLUMNS)
    draws.insert(0, "intercept", posterior_intercept[selected])
    draws.to_csv(OUT / "bayes_draws.csv", index=False)

    bayesian_summary = []
    for index, feature in enumerate(BAYES_COLUMNS):
        samples = posterior_beta[:, index]
        bayesian_summary.append(
            {
                "feature": feature,
                "mean": float(samples.mean()),
                "sd": float(samples.std()),
                "lo": float(np.percentile(samples, 2.5)),
                "hi": float(np.percentile(samples, 97.5)),
                "p_pos": float((samples > 0).mean()),
            }
        )

    bayesian = {
        "model": "Bayesian logistic regression using mean-field ADVI",
        "features": BAYES_COLUMNS,
        "excluded_deterministic_indicator_features": [
            "bank_change",
            "currency_mismatch",
        ],
        "intercept_mean": float(posterior_intercept.mean()),
        "summary": bayesian_summary,
        "means": {name: float(means[name]) for name in CONTINUOUS_COLUMNS},
        "stds": {name: float(stds[name]) for name in CONTINUOUS_COLUMNS},
        "cont": CONTINUOUS_COLUMNS,
        "posterior_draws_available": int(len(posterior_intercept)),
        "advi_iterations_completed": int(len(advi_loss)),
        "advi_final_loss": float(advi_loss[-1]),
    }
    (OUT / "bayes.json").write_text(json.dumps(bayesian, indent=2))

    flagged = scored_test[scored_test["alert"]].copy()
    flagged[["review_exposure"]].to_csv(
        OUT / "flagged_exposures.csv",
        index=False,
    )

    clean = data[data["is_anomaly"] == 0].copy()
    date_span_days = max(
        int((clean["payment_date"].max() - clean["payment_date"].min()).days) + 1,
        1,
    )
    years_represented = date_span_days / 365.25
    annualised_clean_spend = float(clean["amount_usd"].sum() / years_represented)

    monte_carlo = {
        "n_flagged": int(len(flagged)),
        "annualised_clean_spend": annualised_clean_spend,
        "years_represented": float(years_represented),
        "review_exposure_mean_ref": float(flagged["review_exposure"].mean()),
        "review_exposure_sum_ref": float(flagged["review_exposure"].sum()),
        "interpretation": (
            "Review exposure is a prioritisation amount, not calibrated expected loss."
        ),
    }
    (OUT / "montecarlo.json").write_text(json.dumps(monte_carlo, indent=2))

    print(f"Exported workbook inputs to {OUT}")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

from payguard.data_generator import generate_transactions
from sklearn.model_selection import train_test_split

from payguard.detector import (
    evaluate_alerts,
    evaluate_by_anomaly_type,
    fit_anomaly_detector,
    score_transactions,
)


def test_generator_is_reproducible():
    first = generate_transactions(700, seed=7)
    second = generate_transactions(700, seed=7)

    assert first.equals(second)
    assert first["is_anomaly"].sum() > 0


def test_generator_contains_all_anomaly_types():
    data = generate_transactions(5_600, seed=42)

    expected = {
        "normal",
        "exact_duplicate",
        "near_duplicate",
        "overpayment",
        "invoice_above_po",
        "bank_change",
        "currency_mismatch",
        "rapid_payment",
    }

    assert expected.issubset(set(data["anomaly_type"].unique()))


def test_detector_returns_required_outputs():
    data = generate_transactions(1_400, seed=11)
    detector = fit_anomaly_detector(data)

    scored = score_transactions(
        data,
        fitted_detector=detector,
        training_data=data,
    )

    required = {
        "risk_score",
        "alert",
        "alert_reason",
        "overpayment_exposure",
        "duplicate_candidate_exposure",
        "direct_exposure",
        "scenario_exposure",
        "review_exposure",
        "exposure_basis",
    }

    assert required.issubset(scored.columns)
    assert scored["risk_score"].between(0, 100).all()
    assert scored["ml_anomaly_score"].between(0, 100).all()
    assert scored["review_exposure"].ge(0).all()
    assert scored["alert_reason"].str.len().gt(0).all()


def test_held_out_scoring_does_not_refit():
    data = generate_transactions(2_000, seed=12)

    training_data = data.iloc[:1_400].copy()
    test_data = data.iloc[1_400:].copy()

    fitted = fit_anomaly_detector(training_data)

    first = score_transactions(
        test_data,
        fitted_detector=fitted,
        training_data=training_data,
    )

    second = score_transactions(
        test_data,
        fitted_detector=fitted,
        training_data=training_data,
    )

    assert np.allclose(
        first["ml_anomaly_score"],
        second["ml_anomaly_score"],
    )


def test_near_duplicates_are_detected():
    data = generate_transactions(5_600, seed=42)

    detector = fit_anomaly_detector(data)

    scored = score_transactions(
        data,
        fitted_detector=detector,
        training_data=data,
    )

    near_duplicates = scored[
        scored["anomaly_type"] == "near_duplicate"
    ]

    assert not near_duplicates.empty
    assert (
        near_duplicates["alert_reason"]
        .str.contains("Potential duplicate invoice")
        .mean()
        >= 0.50
    )


def test_model_recovers_useful_share_of_injected_cases():
    data = generate_transactions(5_600, seed=42)

    train_index, test_index = train_test_split(
        data.index,
        test_size=0.30,
        random_state=42,
        stratify=data["is_anomaly"],
    )
    training_data = data.loc[train_index].copy()
    test_data = data.loc[test_index].copy()

    fitted = fit_anomaly_detector(training_data)

    scored_test = score_transactions(
        test_data,
        fitted_detector=fitted,
        training_data=training_data,
    )

    metrics = evaluate_alerts(scored_test)

    assert metrics["recall"] >= 0.60
    assert metrics["precision"] >= 0.30
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1


def test_anomaly_type_evaluation_structure():
    data = generate_transactions(2_800, seed=42)

    fitted = fit_anomaly_detector(data)

    scored = score_transactions(
        data,
        fitted_detector=fitted,
        training_data=data,
    )

    summary = evaluate_by_anomaly_type(scored)

    assert {
        "anomaly_type",
        "count",
        "detected",
        "detection_rate",
    }.issubset(summary.columns)

    assert summary["detection_rate"].between(0, 1).all()


def test_var_cvar_relationship():
    rng = np.random.default_rng(42)
    simulated_losses = rng.lognormal(
        mean=10,
        sigma=0.5,
        size=20_000,
    )

    var_95 = np.percentile(simulated_losses, 95)
    cvar_95 = simulated_losses[
        simulated_losses >= var_95
    ].mean()

    assert cvar_95 >= var_95


def test_no_division_by_zero_in_features():
    data = generate_transactions(700, seed=5)

    data.loc[data.index[0], "invoice_amount"] = 0
    data.loc[data.index[1], "po_amount"] = 0

    fitted = fit_anomaly_detector(data)

    scored = score_transactions(
        data,
        fitted_detector=fitted,
        training_data=data,
    )

    assert np.isfinite(scored["risk_score"]).all()
    assert np.isfinite(scored["review_exposure"]).all()
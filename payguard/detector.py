"""Explainable business rules and unsupervised anomaly detection for payments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from payguard.data_generator import FX_TO_USD


RULE_WEIGHTS = {
    "Potential duplicate invoice": 35,
    "Payment exceeds invoice": 35,
    "Invoice exceeds purchase order": 28,
    "Recent bank-account change": 32,
    "Invoice/payment currency mismatch": 30,
    "Unusually rapid payment": 18,
    "Weekend payment": 8,
}

RULE_SCORE_CAP = 80
RULE_WEIGHT = 0.72
ML_WEIGHT = 0.28
ALERT_THRESHOLD = 45.0
SCENARIO_LOSS_GIVEN_ALERT = 0.25


@dataclass
class FittedAnomalyDetector:
    """Store fitted preprocessing, Isolation Forest, and training score distribution."""

    scaler: RobustScaler
    model: IsolationForest
    reference_raw_scores: np.ndarray


@dataclass
class DuplicateEvidence:
    """Keep duplicate-alert evidence separate from exposure attribution."""

    flagged: pd.Series
    candidate: pd.Series


def _normalise_invoice(series: pd.Series) -> pd.Series:
    """Remove punctuation, spaces, and letter-case differences from invoice numbers."""
    return (
        series.astype(str)
        .str.upper()
        .map(lambda value: re.sub(r"[^A-Z0-9]", "", value))
    )


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw payment fields into numerical features used by the models."""
    invoice_date = pd.to_datetime(data["invoice_date"])
    payment_date = pd.to_datetime(data["payment_date"])

    features = pd.DataFrame(index=data.index)
    features["log_amount"] = np.log1p(data["amount_usd"].clip(lower=0))
    features["paid_to_invoice"] = (
        data["paid_amount"] / data["invoice_amount"].clip(lower=1)
    )
    features["invoice_to_po"] = (
        data["invoice_amount"] / data["po_amount"].clip(lower=1)
    )
    features["days_to_pay"] = (payment_date - invoice_date).dt.days
    features["terms_deviation"] = (
        features["days_to_pay"] - data["payment_terms_days"]
    )
    features["weekend"] = payment_date.dt.dayofweek.ge(5).astype(int)
    features["bank_change"] = data["bank_changed_recently"].astype(int)
    features["currency_mismatch"] = (
        data["invoice_currency"] != data["payment_currency"]
    ).astype(int)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def _find_near_duplicates(
    data: pd.DataFrame,
    similarity_threshold: float = 0.90,
    amount_tolerance: float = 0.01,
    date_window_days: int = 60,
) -> DuplicateEvidence:
    """Identify likely duplicates with slightly modified invoice numbers.

    Both records in a likely pair are flagged for review, but only the later
    record is marked as the candidate duplicate for exposure attribution. This
    avoids counting the same possible duplicate payment twice.
    """
    flagged = pd.Series(False, index=data.index, dtype=bool)
    candidate = pd.Series(False, index=data.index, dtype=bool)

    working = pd.DataFrame(
        {
            "supplier_id": data["supplier_id"].astype(str),
            "invoice_number_normalised": _normalise_invoice(data["invoice_number"]),
            "invoice_amount": data["invoice_amount"].astype(float),
            "invoice_date": pd.to_datetime(data["invoice_date"]),
        },
        index=data.index,
    )

    for _, supplier_rows in working.groupby("supplier_id"):
        supplier_rows = supplier_rows.sort_values("invoice_date")
        row_indices = supplier_rows.index.to_list()

        for position, current_index in enumerate(row_indices):
            current = supplier_rows.loc[current_index]

            for previous_index in row_indices[:position]:
                previous = supplier_rows.loc[previous_index]

                day_difference = abs(
                    (current["invoice_date"] - previous["invoice_date"]).days
                )
                if day_difference > date_window_days:
                    continue

                denominator = max(
                    abs(float(current["invoice_amount"])),
                    abs(float(previous["invoice_amount"])),
                    1.0,
                )
                relative_amount_difference = abs(
                    float(current["invoice_amount"])
                    - float(previous["invoice_amount"])
                ) / denominator

                if relative_amount_difference > amount_tolerance:
                    continue

                similarity = SequenceMatcher(
                    None,
                    str(current["invoice_number_normalised"]),
                    str(previous["invoice_number_normalised"]),
                ).ratio()

                if similarity_threshold <= similarity < 1.0:
                    flagged.loc[current_index] = True
                    flagged.loc[previous_index] = True
                    candidate.loc[current_index] = True
                    break

    return DuplicateEvidence(flagged=flagged, candidate=candidate)


def _calculate_duplicate_evidence(data: pd.DataFrame) -> DuplicateEvidence:
    """Identify exact and near duplicates and mark only later records as candidates."""
    duplicate_key = (
        data["supplier_id"].astype(str)
        + "|"
        + _normalise_invoice(data["invoice_number"])
        + "|"
        + data["invoice_amount"].round(2).astype(str)
    )

    exact_flagged = duplicate_key.duplicated(keep=False)
    exact_candidate = duplicate_key.duplicated(keep="first")
    near = _find_near_duplicates(data)

    return DuplicateEvidence(
        flagged=exact_flagged | near.flagged,
        candidate=exact_candidate | near.candidate,
    )


def _calculate_rule_flags(
    data: pd.DataFrame,
    duplicate_evidence: DuplicateEvidence | None = None,
) -> pd.DataFrame:
    """Return a Boolean table showing which business rules each payment triggers."""
    payment_date = pd.to_datetime(data["payment_date"])
    invoice_date = pd.to_datetime(data["invoice_date"])
    days_to_pay = (payment_date - invoice_date).dt.days
    duplicate_evidence = duplicate_evidence or _calculate_duplicate_evidence(data)

    return pd.DataFrame(
        {
            "Potential duplicate invoice": duplicate_evidence.flagged,
            "Payment exceeds invoice": (
                data["paid_amount"] > data["invoice_amount"] * 1.01
            ),
            "Invoice exceeds purchase order": (
                data["invoice_amount"] > data["po_amount"] * 1.05
            ),
            "Recent bank-account change": data["bank_changed_recently"].astype(bool),
            "Invoice/payment currency mismatch": (
                data["invoice_currency"] != data["payment_currency"]
            ),
            "Unusually rapid payment": days_to_pay <= 1,
            "Weekend payment": payment_date.dt.dayofweek >= 5,
        },
        index=data.index,
    )


def fit_anomaly_detector(
    training_data: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> FittedAnomalyDetector:
    """Fit RobustScaler and Isolation Forest using training data only."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    features = _engineer_features(training_data)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(scaled)

    reference_raw_scores = -model.score_samples(scaled)
    return FittedAnomalyDetector(
        scaler=scaler,
        model=model,
        reference_raw_scores=np.sort(reference_raw_scores),
    )


def _raw_scores_to_percentiles(
    raw_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> np.ndarray:
    """Convert anomaly scores into percentiles relative to training scores."""
    positions = np.searchsorted(reference_scores, raw_scores, side="right")
    return 100.0 * positions / max(len(reference_scores), 1)


def score_with_fitted_detector(
    data: pd.DataFrame,
    fitted_detector: FittedAnomalyDetector,
) -> pd.Series:
    """Score new transactions using a detector fitted on separate training data."""
    features = _engineer_features(data)
    scaled = fitted_detector.scaler.transform(features)
    raw_scores = -fitted_detector.model.score_samples(scaled)
    percentile_scores = _raw_scores_to_percentiles(
        raw_scores,
        fitted_detector.reference_raw_scores,
    )
    return pd.Series(percentile_scores, index=data.index, name="ml_anomaly_score")


def _build_statistical_explanation(
    features: pd.Series,
    training_features: pd.DataFrame,
) -> str:
    """Create simple feature-level explanations for statistical anomalies."""
    medians = training_features.median()
    median_absolute_deviation = (
        training_features.sub(medians).abs().median().replace(0, 1)
    )
    robust_z = (
        features.sub(medians).div(median_absolute_deviation).abs()
    ).sort_values(ascending=False)

    feature_messages = {
        "log_amount": "Payment amount differs substantially from the training population",
        "paid_to_invoice": "Paid-to-invoice ratio is unusually high or low",
        "invoice_to_po": "Invoice-to-purchase-order ratio is unusual",
        "days_to_pay": "Payment timing differs substantially from normal behaviour",
        "terms_deviation": "Payment timing differs from the agreed payment terms",
        "weekend": "Payment occurred during a weekend",
        "bank_change": "Supplier bank account changed recently",
        "currency_mismatch": "Invoice and payment currencies do not match",
    }

    explanations = [
        feature_messages[name]
        for name, value in robust_z.items()
        if value >= 3 and name in feature_messages
    ][:2]

    return (
        "; ".join(explanations)
        if explanations
        else "Statistical deviation from the training population"
    )


def _add_exposure_fields(
    scored: pd.DataFrame,
    flags: pd.DataFrame,
    duplicate_candidate: pd.Series,
) -> None:
    """Add transparent exposure fields used only for review prioritisation.

    The fields have deliberately different meanings:

    * overpayment_exposure: observable amount paid above the invoice.
    * duplicate_candidate_exposure: full amount of the later candidate duplicate.
    * direct_exposure: larger of the two observable amounts above.
    * scenario_exposure: illustrative amount for alerted payments without a
      directly measurable excess. It is not expected loss and does not use a
      calibrated probability.
    * review_exposure: the amount displayed for prioritisation. It uses direct
      exposure when available; otherwise it uses scenario exposure.
    """
    # Fixed illustrative conversion factors define one common synthetic base
    # unit (SMU). Convert each amount using its own currency before comparing.
    payment_factor = scored["payment_currency"].map(FX_TO_USD)
    invoice_factor = scored["invoice_currency"].map(FX_TO_USD)
    if payment_factor.isna().any() or invoice_factor.isna().any():
        raise ValueError("Unsupported currency in review-exposure calculation.")
    paid_smu = (scored["paid_amount"] * payment_factor).round(2)
    invoice_smu = (scored["invoice_amount"] * invoice_factor).round(2)

    scored["overpayment_exposure"] = (
        paid_smu - invoice_smu
    ).clip(lower=0).round(2)

    scored["duplicate_candidate_exposure"] = paid_smu.where(
        duplicate_candidate,
        0.0,
    ).round(2)

    scored["direct_exposure"] = np.maximum(
        scored["overpayment_exposure"],
        scored["duplicate_candidate_exposure"],
    ).round(2)

    scenario_eligible = scored["alert"] & scored["direct_exposure"].eq(0)
    scored["scenario_exposure"] = (
        paid_smu
        * (scored["risk_score"] / 100)
        * SCENARIO_LOSS_GIVEN_ALERT
    ).where(scenario_eligible, 0.0).round(2)

    scored["review_exposure"] = scored["direct_exposure"].where(
        scored["direct_exposure"].gt(0),
        scored["scenario_exposure"],
    ).round(2)

    scored["exposure_basis"] = "none"
    scored.loc[scored["scenario_exposure"].gt(0), "exposure_basis"] = (
        "illustrative risk scenario"
    )
    scored.loc[scored["overpayment_exposure"].gt(0), "exposure_basis"] = (
        "measurable overpayment"
    )
    scored.loc[
        scored["duplicate_candidate_exposure"].gt(0), "exposure_basis"
    ] = "candidate duplicate payment"
    scored.loc[
        scored["duplicate_candidate_exposure"].gt(0)
        & scored["overpayment_exposure"].gt(0),
        "exposure_basis",
    ] = "duplicate candidate and overpayment"


def score_transactions(
    data: pd.DataFrame,
    *,
    fitted_detector: FittedAnomalyDetector | None = None,
    training_data: pd.DataFrame | None = None,
    contamination: float = 0.05,
    rule_weight: float = RULE_WEIGHT,
    ml_weight: float = ML_WEIGHT,
    alert_threshold: float = ALERT_THRESHOLD,
) -> pd.DataFrame:
    """Add rules, anomaly scores, hybrid alerts, explanations, and exposure fields.

    For held-out evaluation, pass a detector created with
    ``fit_anomaly_detector(training_data)``. If no fitted detector is supplied,
    this function fits one using ``training_data`` or, as a final fallback,
    the same data being scored. The fallback is suitable for a dashboard demo,
    but not for rigorous held-out evaluation.
    """
    if not np.isclose(rule_weight + ml_weight, 1.0):
        raise ValueError("rule_weight and ml_weight must sum to 1")

    scored = data.copy()
    duplicate_evidence = _calculate_duplicate_evidence(scored)
    flags = _calculate_rule_flags(scored, duplicate_evidence)

    weights = pd.Series(RULE_WEIGHTS)
    rule_score = flags.mul(weights, axis=1).sum(axis=1).clip(upper=RULE_SCORE_CAP)

    if fitted_detector is None:
        fitting_data = training_data if training_data is not None else data
        fitted_detector = fit_anomaly_detector(
            fitting_data,
            contamination=contamination,
        )

    ml_score = score_with_fitted_detector(scored, fitted_detector)

    scored["rule_score"] = rule_score.round(1)
    scored["ml_anomaly_score"] = ml_score.round(1)
    scored["risk_score"] = (
        rule_weight * rule_score + ml_weight * ml_score
    ).clip(upper=100).round(1)
    scored["alert"] = scored["risk_score"] >= alert_threshold

    explanation_reference = training_data if training_data is not None else data
    training_features = _engineer_features(explanation_reference)
    scored_features = _engineer_features(scored)

    explanations: list[str] = []
    for index, flag_row in flags.iterrows():
        rule_reasons = flag_row.index[flag_row].tolist()
        explanations.append(
            "; ".join(rule_reasons)
            if rule_reasons
            else _build_statistical_explanation(
                scored_features.loc[index],
                training_features,
            )
        )

    scored["alert_reason"] = explanations
    _add_exposure_fields(scored, flags, duplicate_evidence.candidate)
    return scored


def evaluate_alerts(scored: pd.DataFrame) -> dict[str, float | int]:
    """Calculate evaluation metrics using the synthetic ground-truth labels."""
    y_true = scored["is_anomaly"].astype(int)
    y_pred = scored["alert"].astype(int)
    y_score = scored["risk_score"].astype(float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "alert_rate": float(y_pred.mean()),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "alerts_per_1000": float(y_pred.mean() * 1000),
    }


def evaluate_by_anomaly_type(scored: pd.DataFrame) -> pd.DataFrame:
    """Calculate detection rates separately for each planted anomaly type."""
    anomaly_rows = scored[scored["anomaly_type"] != "normal"].copy()
    if anomaly_rows.empty:
        return pd.DataFrame(
            columns=["anomaly_type", "count", "detected", "detection_rate"]
        )

    return (
        anomaly_rows.groupby("anomaly_type")["alert"]
        .agg(count="size", detected="sum", detection_rate="mean")
        .reset_index()
    )

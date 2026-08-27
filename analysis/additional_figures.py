"""Generate supplementary figures using the same held-out design as the main analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT / "analysis" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from payguard.detector import (  # noqa: E402
    ALERT_THRESHOLD,
    _engineer_features,
    evaluate_by_anomaly_type,
    fit_anomaly_detector,
    score_transactions,
)

NAVY = "#11295B"
GOLD = "#C7A24A"
GREY = "#6B7280"
RED = "#B3392C"

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train_scaled, test_scaled


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

    detector = fit_anomaly_detector(train_data, contamination=0.05)
    scored_all = score_transactions(
        data,
        fitted_detector=detector,
        training_data=train_data,
    )
    scored_test = scored_all.loc[test_index].copy()

    train_features = _engineer_features(train_data)
    test_features = _engineer_features(test_data)
    x_train, x_test = _standardise(train_features, test_features)
    y_train = train_data["is_anomaly"].astype(int)
    y_test = test_data["is_anomaly"].astype(int)

    logistic = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        solver="lbfgs",
    ).fit(x_train, y_train)
    logistic_probability = logistic.predict_proba(x_test)[:, 1]

    # 07: Framework architecture.
    # The architecture is deliberately non-linear: the rule engine and
    # Isolation Forest are complementary detection branches; the statistical
    # models and financial simulations are analytical decision-support layers.
    figure, axis = plt.subplots(figsize=(13.5, 8.2))
    axis.axis("off")

    def box(x, y, text, width=0.19, height=0.085, edge=NAVY):
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                edgecolor=edge,
                linewidth=1.8,
            ),
            transform=axis.transAxes,
        )

    def arrow(x1, y1, x2, y2, colour=GOLD):
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.9),
            xycoords=axis.transAxes,
        )

    # Operational detection layer.
    box(0.10, 0.82, "Supplier payment\nrecords")
    box(0.31, 0.82, "Preprocessing +\nfeature engineering")
    box(0.53, 0.90, "Rule-based\ncontrols")
    box(0.53, 0.74, "Isolation Forest\nanomaly detection")
    box(0.73, 0.82, "Hybrid risk score")
    box(0.91, 0.82, "Alert + explanation\n+ review exposure")
    arrow(0.17, 0.82, 0.24, 0.82)
    arrow(0.38, 0.84, 0.46, 0.89)
    arrow(0.38, 0.80, 0.46, 0.75)
    arrow(0.60, 0.90, 0.67, 0.84)
    arrow(0.60, 0.74, 0.67, 0.80)
    arrow(0.80, 0.82, 0.84, 0.82)

    # Statistical interpretation and uncertainty layer.
    box(0.10, 0.48, "Labelled synthetic\ntraining data")
    box(0.34, 0.48, "Logistic regression\n(associations + odds ratios)")
    box(0.62, 0.48, "Bayesian logistic regression\n(posterior uncertainty)")
    box(0.88, 0.48, "Held-out predictive\nassessment")
    arrow(0.18, 0.48, 0.25, 0.48)
    arrow(0.44, 0.48, 0.51, 0.48)
    arrow(0.73, 0.48, 0.79, 0.48)

    # Financial decision-support layer.
    box(0.16, 0.16, "Held-out alerted\ntransactions")
    box(0.41, 0.16, "Bootstrap aggregate\nreview exposure")
    box(0.66, 0.16, "Working-capital Monte Carlo\n(DPO scenarios)")
    box(0.90, 0.16, "Financial decision\nsupport")
    arrow(0.24, 0.16, 0.32, 0.16)
    arrow(0.49, 0.16, 0.57, 0.16)
    arrow(0.75, 0.16, 0.83, 0.16)

    # Cross-layer links showing where the financial layer receives information.
    arrow(0.91, 0.77, 0.22, 0.22, colour=GREY)
    arrow(0.31, 0.77, 0.62, 0.22, colour=GREY)

    axis.text(
        0.02,
        0.97,
        "Operational detection layer",
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        0.02,
        0.61,
        "Statistical interpretation and uncertainty layer",
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        0.02,
        0.29,
        "Financial decision-support layer",
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )
    axis.set_title("PayGuard AI framework architecture", fontsize=14, pad=18)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "07_system_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # 08: Dataset composition.
    counts = data["anomaly_type"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    colours = [NAVY if label == "normal" else GOLD for label in counts.index]
    axis.barh(counts.index, counts.values, color=colours)
    axis.set_xlabel("Transactions")
    axis.set_title("Synthetic dataset composition")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "08_dataset_composition.png", dpi=160)
    plt.close(figure)

    # 09: Feature separation on held-out data.
    selected_features = ["paid_to_invoice", "invoice_to_po", "terms_deviation"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, feature in zip(axes, selected_features):
        axis.boxplot(
            [
                test_features.loc[y_test == 0, feature],
                test_features.loc[y_test == 1, feature],
            ],
            tick_labels=["Normal", "Anomaly"],
            showfliers=False,
        )
        axis.set_title(feature)
    figure.suptitle("Held-out feature distributions by synthetic class")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "09_feature_separation.png", dpi=160)
    plt.close(figure)

    # 10: Correlation matrix from training features only.
    correlation = train_features.corr()
    figure, axis = plt.subplots(figsize=(7, 5.8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right")
    axis.set_yticklabels(correlation.columns)
    figure.colorbar(image, ax=axis)
    axis.set_title("Training-feature correlation matrix")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "10_correlation_heatmap.png", dpi=160)
    plt.close(figure)

    # 11: Held-out risk-score distribution.
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    bins = np.linspace(0, max(scored_test["risk_score"].max() + 1, 46), 40)
    axis.hist(
        scored_test.loc[y_test == 0, "risk_score"],
        bins=bins,
        alpha=0.8,
        label="Normal",
        density=True,
    )
    axis.hist(
        scored_test.loc[y_test == 1, "risk_score"],
        bins=bins,
        alpha=0.75,
        label="Anomaly",
        density=True,
    )
    axis.axvline(ALERT_THRESHOLD, linestyle="--", linewidth=2, label="Threshold")
    axis.set_xlabel("Hybrid risk score")
    axis.set_ylabel("Density")
    axis.set_title("Held-out risk-score distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CHART_DIR / "11_score_distribution.png", dpi=160)
    plt.close(figure)

    # 12: Precision-recall curves.
    figure, axis = plt.subplots(figsize=(6.2, 5))
    score_sets = [
        ("Hybrid", scored_test["risk_score"].to_numpy()),
        ("Isolation Forest", scored_test["ml_anomaly_score"].to_numpy()),
        ("Logistic regression", logistic_probability),
    ]
    for label, scores in score_sets:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        axis.plot(recall, precision, linewidth=2, label=f"{label} (AP={ap:.3f})")
    axis.axhline(y_test.mean(), linestyle="--", linewidth=1, label="Base rate")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Held-out precision-recall curves")
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "12_precision_recall.png", dpi=160)
    plt.close(figure)

    # 13: Held-out confusion matrix.
    predictions = scored_test["alert"].astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], labels=["No alert", "Alert"])
    axis.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    axis.set_title("Held-out confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "13_confusion_matrix.png", dpi=160)
    plt.close(figure)

    # 14: Held-out detection by anomaly type.
    detection = evaluate_by_anomaly_type(scored_test).sort_values("detection_rate")
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    axis.barh(detection["anomaly_type"], detection["detection_rate"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Detection rate")
    axis.set_title("Held-out detection by planted anomaly type")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "14_detection_by_anomaly.png", dpi=160)
    plt.close(figure)

    # 15: Threshold sensitivity on held-out scores.
    thresholds = np.arange(20, 81, 2)
    precision_values = []
    recall_values = []
    alert_rates = []
    for threshold in thresholds:
        predicted = scored_test["risk_score"].ge(threshold).astype(int)
        true_positive = int(((predicted == 1) & (y_test == 1)).sum())
        false_positive = int(((predicted == 1) & (y_test == 0)).sum())
        false_negative = int(((predicted == 0) & (y_test == 1)).sum())
        precision_values.append(
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0
        )
        recall_values.append(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0
        )
        alert_rates.append(float(predicted.mean()))

    figure, axis = plt.subplots(figsize=(7, 4.4))
    axis.plot(thresholds, precision_values, label="Precision")
    axis.plot(thresholds, recall_values, label="Recall")
    axis.plot(thresholds, alert_rates, label="Alert rate")
    axis.axvline(ALERT_THRESHOLD, linestyle="--", label="Current threshold")
    axis.set_xlabel("Risk-score threshold")
    axis.set_ylabel("Rate")
    axis.set_title("Held-out threshold sensitivity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CHART_DIR / "15_threshold_sensitivity.png", dpi=160)
    plt.close(figure)

    # 16 and 17: ADVI convergence and held-out calibration.
    posterior_path = ROOT / "analysis" / "bayesian_advi_posterior.nc"
    loss_path = ROOT / "analysis" / "bayesian_advi_loss.npy"

    if not posterior_path.exists() or not loss_path.exists():
        raise FileNotFoundError(
            "ADVI posterior/loss history not found. "
            "Run analysis/advanced_statistics.py first."
        )

    trace = az.from_netcdf(posterior_path)
    advi_loss = np.load(loss_path)

    figure, axis = plt.subplots(figsize=(7, 4.4))
    axis.plot(np.arange(1, len(advi_loss) + 1), advi_loss)
    axis.set_xlabel("ADVI iteration")
    axis.set_ylabel("Variational loss")
    axis.set_title("ADVI optimisation convergence")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "16_advi_convergence.png", dpi=160)
    plt.close(figure)

    posterior_intercept = trace.posterior["intercept"].values.reshape(-1, 1)
    posterior_beta = trace.posterior["beta"].values.reshape(-1, len(BAYES_COLUMNS))
    logits = posterior_intercept + np.einsum(
        "sf,nf->sn",
        posterior_beta,
        x_test[BAYES_COLUMNS].to_numpy(),
    )
    bayesian_probability = (1 / (1 + np.exp(-np.clip(logits, -40, 40)))).mean(axis=0)
    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        bayesian_probability,
        n_bins=8,
        strategy="quantile",
    )

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.plot(mean_predicted, fraction_positive, marker="o", label="Bayesian model")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed anomaly rate")
    axis.set_title("Held-out probability calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CHART_DIR / "17_calibration.png", dpi=160)
    plt.close(figure)

    print(f"Supplementary figures saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from payguard.data_generator import generate_transactions
from payguard.detector import evaluate_alerts, score_transactions


st.set_page_config(page_title="PayGuard AI", page_icon="🛡️", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path("data/synthetic_payments.csv")
    if path.exists():
        return pd.read_csv(path)
    return generate_transactions(n_transactions=10_000, seed=42)


@st.cache_data
def analyse(data: pd.DataFrame) -> pd.DataFrame:
    return score_transactions(data)


st.title("PayGuard AI")
st.caption("Explainable supplier-payment anomaly detection • synthetic demonstration data")
st.info(
    "This prototype provides investigative leads, not fraud conclusions. "
    "All suppliers and transactions are fictional."
)

scored = analyse(load_data())
metrics = evaluate_alerts(scored)

with st.sidebar:
    st.header("Filters")
    minimum_risk = st.slider("Minimum risk score", 0, 100, 45)
    categories = st.multiselect(
        "Supplier category",
        sorted(scored["supplier_category"].unique()),
        default=sorted(scored["supplier_category"].unique()),
    )
    only_alerts = st.toggle("Show alerts only", value=True)

filtered = scored[
    (scored["risk_score"] >= minimum_risk)
    & scored["supplier_category"].isin(categories)
]
if only_alerts:
    filtered = filtered[filtered["alert"]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions analysed", f"{len(scored):,}")
c2.metric("Alerts", f"{int(scored['alert'].sum()):,}", f"{metrics['alert_rate']:.1%} of records")
c3.metric(
    "Review exposure",
    f"{scored.loc[scored['alert'], 'review_exposure'].sum():,.0f} SMU",
    help=(
        "Prioritisation amount. It uses directly measurable duplicate or "
        "overpayment exposure where available; otherwise it uses an "
        "illustrative scenario amount. It is not expected fraud loss."
    ),
)
c4.metric("Synthetic-case recall", f"{metrics['recall']:.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Risk-score distribution")
    fig = px.histogram(
        scored,
        x="risk_score",
        color="alert",
        nbins=25,
        color_discrete_map={True: "#d1495b", False: "#2a9d8f"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Alerts by category")
    category_alerts = (
        scored[scored["alert"]]
        .groupby("supplier_category", as_index=False)
        .agg(alerts=("transaction_id", "count"), exposure=("review_exposure", "sum"))
        .sort_values("exposure", ascending=False)
    )
    fig = px.bar(category_alerts, x="exposure", y="supplier_category", orientation="h", text="alerts")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prioritised investigation queue")
columns = [
    "transaction_id",
    "supplier_id",
    "supplier_category",
    "invoice_number",
    "paid_amount",
    "payment_currency",
    "risk_score",
    "overpayment_exposure",
    "duplicate_candidate_exposure",
    "scenario_exposure",
    "review_exposure",
    "exposure_basis",
    "alert_reason",
]
queue = filtered.sort_values(["risk_score", "review_exposure"], ascending=False)[columns]
st.dataframe(
    queue,
    use_container_width=True,
    hide_index=True,
    column_config={
        "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100),
        "overpayment_exposure": st.column_config.NumberColumn(
            "Overpayment exposure", format="%.2f SMU"
        ),
        "duplicate_candidate_exposure": st.column_config.NumberColumn(
            "Duplicate exposure", format="%.2f SMU"
        ),
        "scenario_exposure": st.column_config.NumberColumn(
            "Scenario exposure", format="%.2f SMU"
        ),
        "review_exposure": st.column_config.NumberColumn(
            "Review exposure", format="%.2f SMU"
        ),
    },
)


st.caption(
    "Review exposure is a prioritisation aid, not a calibrated expected-loss estimate. "
    "Direct exposure is used for measurable overpayments and candidate duplicate "
    "payments; otherwise an illustrative scenario amount is shown for alerted rows."
)

st.download_button(
    "Download investigation queue",
    queue.to_csv(index=False).encode("utf-8"),
    file_name="payguard_investigation_queue.csv",
    mime="text/csv",
)

with st.expander("Model evaluation on labelled synthetic cases"):
    st.write(
        {
            "Precision": f"{metrics['precision']:.1%}",
            "Recall": f"{metrics['recall']:.1%}",
            "F1 score": f"{metrics['f1']:.1%}",
        }
    )
    st.caption(
        "These figures measure recovery of deliberately inserted cases. They must not be "
        "presented as expected production performance."
    )
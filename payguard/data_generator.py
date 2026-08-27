"""Generate reproducible, fictional corporate invoice and payment records."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


CATEGORIES = [
    "Aircraft Components",
    "Maintenance",
    "Information Technology",
    "Ground Services",
    "Professional Services",
    "Facilities",
]
CURRENCIES = np.array(["USD", "AED", "EUR", "GBP"])
FX_TO_USD = {"USD": 1.0, "AED": 0.2723, "EUR": 1.09, "GBP": 1.27}


def _invoice_number(rng: np.random.Generator, i: int) -> str:
    return f"INV-{rng.integers(100, 999)}-{2026}-{i:06d}"


def _normalise_invoice(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def generate_transactions(n_transactions: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Return synthetic payment records with known, injected anomaly labels.

    Around 5% of rows contain one controlled anomaly. Every monetary value is
    fictional. The generator deliberately preserves an anomaly-free majority
    for unsupervised learning.
    """
    if n_transactions < 500:
        raise ValueError("n_transactions must be at least 500")

    rng = np.random.default_rng(seed)
    n_suppliers = max(80, n_transactions // 35)
    supplier_ids = np.array([f"SUP-{i:04d}" for i in range(1, n_suppliers + 1)])
    supplier_currency = dict(
        zip(supplier_ids, rng.choice(CURRENCIES, n_suppliers, p=[0.46, 0.28, 0.17, 0.09]))
    )
    supplier_bank = {
        sid: f"BANK-{rng.integers(10000000, 99999999)}" for sid in supplier_ids
    }

    suppliers = rng.choice(supplier_ids, n_transactions)
    currencies = np.array([supplier_currency[s] for s in suppliers])
    categories = rng.choice(
        CATEGORIES, n_transactions, p=[0.22, 0.20, 0.16, 0.15, 0.15, 0.12]
    )
    invoice_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 540, n_transactions), unit="D"
    )
    payment_terms = rng.choice([30, 45, 60, 90], n_transactions, p=[0.42, 0.25, 0.25, 0.08])
    days_to_pay = np.maximum(1, payment_terms + rng.normal(0, 9, n_transactions).round().astype(int))
    payment_dates = invoice_dates + pd.to_timedelta(days_to_pay, unit="D")

    # A long-tailed distribution resembles corporate spend better than uniform values.
    po_amounts = np.round(np.clip(rng.lognormal(9.2, 1.05, n_transactions), 500, 2_000_000), 2)
    invoice_amounts = np.round(po_amounts * rng.uniform(0.72, 1.0, n_transactions), 2)
    paid_amounts = invoice_amounts.copy()
    bank_accounts = np.array([supplier_bank[s] for s in suppliers], dtype=object)
    invoice_numbers = np.array(
        [_invoice_number(rng, i) for i in range(n_transactions)], dtype=object
    )

    data = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:07d}" for i in range(n_transactions)],
            "supplier_id": suppliers,
            "supplier_category": categories,
            "invoice_number": invoice_numbers,
            "po_number": [f"PO-{2025 + (i % 2)}-{i:06d}" for i in range(n_transactions)],
            "invoice_date": invoice_dates,
            "payment_date": payment_dates,
            "payment_terms_days": payment_terms,
            "po_amount": po_amounts,
            "invoice_amount": invoice_amounts,
            "paid_amount": paid_amounts,
            "invoice_currency": currencies,
            "payment_currency": currencies.copy(),
            "bank_account": bank_accounts,
            "bank_changed_recently": False,
            "anomaly_type": "normal",
        }
    )

    # One anomaly per selected row keeps the ground truth easy to interpret.
    anomaly_types = [
        "exact_duplicate",
        "near_duplicate",
        "overpayment",
        "invoice_above_po",
        "bank_change",
        "currency_mismatch",
        "rapid_payment",
    ]
    allocation = np.full(len(anomaly_types), n_transactions // 140)
    chosen = rng.choice(n_transactions, size=int(allocation.sum()), replace=False)

    # Duplicate-source rows are deliberately drawn only from records that were
    # not selected for any planted anomaly. This keeps each injected scenario
    # controlled and prevents a duplicate from accidentally inheriting another
    # planted anomaly type from its source transaction.
    normal_source_pool = np.setdiff1d(np.arange(n_transactions), chosen)

    cursor = 0

    for anomaly, count in zip(anomaly_types, allocation):
        idxs = chosen[cursor : cursor + count]
        cursor += count
        data.loc[idxs, "anomaly_type"] = anomaly

        if anomaly in {"exact_duplicate", "near_duplicate"}:
            # Copy an otherwise normal source record so the duplicate scenario
            # remains isolated from the other planted anomaly types.
            sources = rng.choice(
                normal_source_pool,
                size=len(idxs),
                replace=True,
            )
            for target, source in zip(idxs, sources):
                # Copy the supplier context and monetary values so the target is a
                # plausible repeat submission rather than an unrelated payment.
                copied_columns = [
                    "supplier_id",
                    "supplier_category",
                    "po_number",
                    "po_amount",
                    "invoice_amount",
                    "paid_amount",
                    "invoice_currency",
                    "payment_currency",
                    "bank_account",
                    "bank_changed_recently",
                    "payment_terms_days",
                ]
                data.loc[target, copied_columns] = data.loc[source, copied_columns].values

                # Keep the duplicate close to the source date so the fuzzy rule can
                # use a realistic comparison window without seeing distant invoices.
                source_invoice_date = pd.Timestamp(data.loc[source, "invoice_date"])
                date_offset = int(rng.integers(1, 15))
                data.loc[target, "invoice_date"] = source_invoice_date + pd.Timedelta(
                    days=date_offset
                )
                data.loc[target, "payment_date"] = data.loc[target, "invoice_date"] + pd.Timedelta(
                    days=int(data.loc[target, "payment_terms_days"])
                )

                original = str(data.loc[source, "invoice_number"])
                if anomaly == "exact_duplicate":
                    data.loc[target, "invoice_number"] = original
                else:
                    normalised = _normalise_invoice(original)

                    # Introduce a small plausible modification while preserving high
                    # similarity with the original invoice number.
                    if len(normalised) >= 4:
                        insertion_position = len(normalised) - 2
                        data.loc[target, "invoice_number"] = (
                            normalised[:insertion_position]
                            + "A"
                            + normalised[insertion_position:]
                        )
                    else:
                        data.loc[target, "invoice_number"] = normalised + "A"
        elif anomaly == "overpayment":
            data.loc[idxs, "paid_amount"] = np.round(
                data.loc[idxs, "invoice_amount"] * rng.uniform(1.12, 1.75, len(idxs)), 2
            )
        elif anomaly == "invoice_above_po":
            data.loc[idxs, "invoice_amount"] = np.round(
                data.loc[idxs, "po_amount"] * rng.uniform(1.10, 1.50, len(idxs)), 2
            )
            data.loc[idxs, "paid_amount"] = data.loc[idxs, "invoice_amount"]
        elif anomaly == "bank_change":
            data.loc[idxs, "bank_changed_recently"] = True
            data.loc[idxs, "bank_account"] = [
                f"NEW-{rng.integers(10000000, 99999999)}" for _ in idxs
            ]
        elif anomaly == "currency_mismatch":
            for idx in idxs:
                alternatives = CURRENCIES[CURRENCIES != data.loc[idx, "invoice_currency"]]
                data.loc[idx, "payment_currency"] = rng.choice(alternatives)
        elif anomaly == "rapid_payment":
            data.loc[idxs, "payment_date"] = data.loc[idxs, "invoice_date"] + pd.to_timedelta(
                rng.integers(0, 2, len(idxs)), unit="D"
            )

    data["amount_usd"] = np.round(
        data["paid_amount"] * data["payment_currency"].map(FX_TO_USD), 2
    )
    data["is_anomaly"] = (data["anomaly_type"] != "normal").astype(int)
    return data.sort_values(["invoice_date", "transaction_id"]).reset_index(drop=True)
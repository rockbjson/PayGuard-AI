# Synthetic payment data dictionary

Each row is one supplier-payment transaction. These records are entirely synthetic. Monetary assumptions and labels are not evidence of real fraud or real-world loss.

| Column | Meaning |
| --- | --- |
| transaction_id | Synthetic transaction identifier |
| supplier_id | Synthetic supplier identifier |
| supplier_category | Procurement category |
| invoice_number | Invoice identifier used by exact/near-duplicate controls |
| po_number | Purchase-order identifier |
| invoice_date | Invoice date |
| payment_date | Payment date |
| payment_terms_days | Agreed payment interval, in days |
| po_amount | Purchase-order amount in its nominal currency |
| invoice_amount | Nominal invoice amount |
| paid_amount | Nominal amount paid |
| invoice_currency | Invoice currency code |
| payment_currency | Payment currency code |
| bank_account | Synthetic bank-account identifier |
| bank_changed_recently | Binary planted recent-bank-change indicator |
| anomaly_type | Ground-truth category: normal, exact_duplicate, near_duplicate, overpayment, invoice_above_po, bank_change, currency_mismatch, or rapid_payment |
| amount_usd | Payment amount in the common synthetic base unit (SMU), using payment-currency factors USD=1, AED=0.2723, EUR=1.09, GBP=1.27; the column name is retained for compatibility |
| is_anomaly | Ground truth: 1 for a planted anomaly, otherwise 0 |

The detector must not use `is_anomaly` or `anomaly_type` as predictive features. Labels are used for stratified splitting, supervised statistical modelling and evaluation. The illustrative working-capital scenario also uses known normal labels, an explicit oracle assumption. Derived fields in `analysis/held_out_scores.csv` are calculated in `payguard/detector.py`; `source_row_index` identifies the corresponding zero-based row of the original CSV. `rule_score` and `risk_score` are ranking scores, not calibrated probabilities.

Monetary convention: raw paid, invoice and purchase-order amounts retain their nominal currency units. Cross-currency monetary aggregation uses the fixed illustrative conversion factors above. Exposure calculations convert paid and invoice amounts separately using their respective currency codes. All exposure and working-capital outputs, including legacy result keys ending in `_usd`, use SMU; these are synthetic scenario values, not actual USD transactions. Nominal amount-ratio detection features remain separate from this exposure valuation convention.

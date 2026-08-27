from pathlib import Path

from payguard.data_generator import generate_transactions


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "synthetic_payments.csv"

N_TRANSACTIONS = 5_600
RANDOM_SEED = 42


def main() -> None:
    data = generate_transactions(
        n_transactions=N_TRANSACTIONS,
        seed=RANDOM_SEED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(data):,} transactions to {OUTPUT_PATH}")
    print(f"Injected anomalies: {int(data['is_anomaly'].sum()):,}")
    print(data["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()
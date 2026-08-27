"""PayGuard AI synthetic payment-risk demonstration."""

from .data_generator import generate_transactions
from .detector import evaluate_alerts, score_transactions

__all__ = ["generate_transactions", "score_transactions", "evaluate_alerts"]


# =============================================================================
# chatbot/smart_parser.py  –  Convenience NLP wrappers for SmartExpenseAI
# =============================================================================
# Thin wrappers around entities.py so callers that import smart_parser
# directly continue to work without changes.
# Single source of truth: all parsing logic lives in entities.py.
#
# [PRED-5] extract_smart_data / extract_smart_data_full no longer call ML
# directly.  ML is called exclusively by _ml_predict() inside chatbot_engine,
# which supplies the mandatory transaction_type argument.  smart_parser is
# NLP-only; callers that need ML must go through chatbot_engine.
# =============================================================================

from __future__ import annotations
from chatbot.entities import extract_amount, extract_category, extract_date

# Strings that mean "no category" – kept in sync with entities.py
_NO_CATEGORY_SENTINELS = frozenset({None, "none", "general", "other", ""})


def extract_smart_data(message: str) -> dict:
    """
    Quick parse: return amount and category from *message*.

    Kept for backward compatibility with callers that only need these two
    fields. Returns None for category when NLP found nothing useful.

    Returns:
        {"amount": float | None, "category": str | None}

    Note: ML category prediction is intentionally NOT called here.
    ML requires a transaction_type ("income"/"expense") which this function
    does not receive. Use chatbot_engine._try_nlp_autosave() for ML-backed
    category resolution.
    """
    amount   = extract_amount(message)
    category = extract_category(message)

    return {
        "amount":   amount,
        "category": None if category in _NO_CATEGORY_SENTINELS else category,
    }


def extract_smart_data_full(message: str) -> dict:
  
    amount   = extract_amount(message)
    category = extract_category(message)
    txn_date = extract_date(message)

    return {
        "amount":   amount,
        "category": None if category in _NO_CATEGORY_SENTINELS else category,
        "date":     txn_date,
    }
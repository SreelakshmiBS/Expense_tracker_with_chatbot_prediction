# ==============================================================================
# machine_learning/predict_category.py  –  Predict transaction category
# ==============================================================================
# WHAT THIS DOES:
#   Uses a trained ML model (TF-IDF + Naive Bayes) to predict the category
#   of a transaction based on its description and type.
#
# HOW TO USE:
#   from machine_learning.predict_category import predict_category
#   category = predict_category("pizza delivery", "expense")
#   → "food"
#
# REQUIREMENTS:
#   Run train_category_model.py first to generate the model file.
# ==============================================================================

import os
import joblib

# ── Path to the saved model file ──────────────────────────────────────────────
# Go up one folder from this file, then into models/
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "transaction_category_model.pkl")

# ── Valid transaction types ───────────────────────────────────────────────────
VALID_TYPES = {"income", "expense"}

# ── Load model once when this file is imported ───────────────────────────────
# We load it once so every prediction call is fast (no disk read each time)
_model = None

def _load_model():
    """Load the ML model from disk. Called automatically on first use."""
    global _model
    if _model is not None:
        return _model  # Already loaded

    if not os.path.exists(MODEL_PATH):
        print(f"[ML ERROR] Model file not found at: {MODEL_PATH}")
        print("[ML ERROR] Please run train_category_model.py first!")
        return None

    try:
        _model = joblib.load(MODEL_PATH)
        print(f"[ML INFO] Model loaded from: {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"[ML ERROR] Failed to load model: {e}")
        return None


# ==============================================================================
# PREDICT CATEGORY  ← Main function you call
# ==============================================================================

def predict_category(description: str, transaction_type: str) -> str:
    """
    Predict the category for a transaction.

    Args:
        description      : what the transaction is about, e.g. "pizza"
        transaction_type : "income" or "expense"

    Returns:
        category as a string, e.g. "food", "salary", "travel"
        Returns "general" if prediction fails or input is invalid.

    Example:
        predict_category("pizza from dominos", "expense")  → "food"
        predict_category("monthly salary", "income")       → "salary"
        predict_category("uber ride", "expense")           → "travel"
    """

    # ── Step 1: Validate transaction type ─────────────────────────────────────
    txn_type = str(transaction_type).strip().lower()
    if txn_type not in VALID_TYPES:
        print(f"[ML DEBUG] Invalid transaction type: '{transaction_type}' → returning 'general'")
        return "general"

    # ── Step 2: Validate description ──────────────────────────────────────────
    desc = str(description).strip() if description else ""
    if not desc:
        print("[ML DEBUG] Empty description → returning 'general'")
        return "general"

    # ── Step 3: Load model ────────────────────────────────────────────────────
    model = _load_model()
    if model is None:
        print("[ML DEBUG] Model not available → returning 'general'")
        return "general"

    # ── Step 4: Build the input string ────────────────────────────────────────
    # Format MUST match what was used during training in train_category_model.py
    # Training used: transaction_type + " " + description
    feature_text = f"{txn_type} {desc}"
    print(f"[ML DEBUG] Input to model: '{feature_text}'")

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    try:
        prediction = model.predict([feature_text])
        category   = str(prediction[0]).strip().lower()
        print(f"[ML DEBUG] Predicted category: '{category}'")
        return category

    except Exception as e:
        print(f"[ML ERROR] Prediction failed: {e}")
        return "general"


# ==============================================================================
# BATCH PREDICT  –  predict for a list of transactions at once
# ==============================================================================

def predict_batch(transactions: list) -> list:
    """
    Predict categories for multiple transactions at once.

    Args:
        transactions: list of dicts, each with "description" and "type" keys

    Returns:
        Same list with "predicted_category" added to each dict

    Example:
        predict_batch([
            {"description": "pizza", "type": "expense"},
            {"description": "salary", "type": "income"},
        ])
        → [
            {"description": "pizza", "type": "expense", "predicted_category": "food"},
            {"description": "salary", "type": "income", "predicted_category": "salary"},
          ]
    """
    results = []
    for txn in transactions:
        category = predict_category(
            txn.get("description", ""),
            txn.get("type", "")
        )
        results.append({**txn, "predicted_category": category})
    return results


# ==============================================================================
# DEMO  –  run this file directly to test predictions
# ==============================================================================

if __name__ == "__main__":
    test_cases = [
        # (description, type, expected_category)
        ("pizza from dominos",       "expense", "food"),
        ("uber ride to office",      "expense", "travel"),
        ("netflix subscription",     "expense", "entertainment"),
        ("electricity bill",         "expense", "bills"),
        ("monthly salary",           "income",  "salary"),
        ("freelance project payment","income",  "freelance"),
        ("stock dividend",           "income",  "business"),
        # Edge cases
        ("",                         "expense", "general"),
        ("something random",         "unknown", "general"),
    ]

    print(f"\n{'─' * 60}")
    print(f"  {'TYPE':<10}  {'DESCRIPTION':<30}  {'PREDICTED'}")
    print(f"{'─' * 60}")

    for desc, txn_type, expected in test_cases:
        predicted = predict_category(desc, txn_type)
        status = "✅" if predicted == expected else "❓"
        print(f"  {txn_type:<10}  {desc:<30}  {predicted}  {status}")

    print(f"{'─' * 60}\n")
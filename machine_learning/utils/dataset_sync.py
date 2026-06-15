# ==============================================================================
# machine_learning/utils/dataset_sync.py  –  Save new transactions to CSV
# ==============================================================================

import os
import csv
from datetime import datetime

# ── Path to the CSV dataset file ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "transaction_dataset.csv")

# ── Column names (unified schema) ─────────────────────────────────────────────
CSV_COLUMNS = [                   
    "description",
    "amount",
    "category",
    "transaction_type",
    "transaction_date",
]


def append_to_dataset(description: str, amount,                     # ✅ FIX B
                       transaction_type: str, category: str) -> bool:
    """
    Add one transaction to the training CSV dataset.

    Args:
        description      : what the transaction was for, e.g. "pizza"
        amount           : numeric amount, e.g. 200 or 1500.0
        transaction_type : "income" or "expense"
        category         : predicted category, e.g. "food"

    Returns:
        True if saved successfully, False otherwise.
    """

    description      = str(description).strip()
    transaction_type = str(transaction_type).strip().lower()
    category         = str(category).strip().lower()

    if not description or not transaction_type or not category:
        print("[DATASET] Skipped: missing description, type, or category")
        return False

    if transaction_type not in ("income", "expense"):
        print(f"[DATASET] Skipped: invalid type '{transaction_type}'")
        return False

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        print(f"[DATASET] Skipped: invalid amount '{amount}'")
        return False

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    file_exists = os.path.exists(DATASET_PATH)

    try:
        with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()
                print(f"[DATASET] Created new dataset file at: {DATASET_PATH}")

            writer.writerow({                                           # ✅ FIX C
                "description":      description,
                "amount":           float(amount),
                "category":         category,
                "transaction_type": transaction_type,
                "transaction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        print(f"[DATASET] Saved: '{description}' | {transaction_type} | {category} | {amount}")
        return True

    except Exception as e:
        print(f"[DATASET ERROR] Could not save to dataset: {e}")
        return False
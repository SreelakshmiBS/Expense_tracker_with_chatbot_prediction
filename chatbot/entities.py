# ==============================================================================
# chatbot/entities.py  –  Extract amount, description, category, date
# ==============================================================================

import re
from datetime import date, timedelta

# ── Category keyword maps ─────────────────────────────────────────────────────
EXPENSE_KEYWORDS = {
    "food":          ["pizza", "burger", "food", "restaurant", "lunch", "dinner",
                      "breakfast", "coffee", "snack", "eat", "meal", "biryani",
                      "swiggy", "zomato", "kfc", "mcdonalds"],
    "travel":        ["uber", "ola", "bus", "train", "flight", "taxi", "fuel",
                      "petrol", "diesel", "metro", "auto", "travel", "cab"],
    "rent":          ["rent", "house rent", "flat rent", "room rent", "pg"],
    "shopping":      ["amazon", "flipkart", "clothes", "shirt", "shoes", "dress",
                      "shopping", "market", "store", "buy", "purchase"],
    "bills":         ["electricity", "water", "internet", "wifi", "phone bill",
                      "recharge", "bill", "utility", "broadband"],
    "entertainment": ["movie", "netflix", "spotify", "game", "concert", "party",
                      "cinema", "hotstar", "youtube premium", "entertainment"],
    "medical":       ["medicine", "doctor", "hospital", "clinic", "pharmacy",
                      "health", "medical", "tablet", "checkup"],
    "education":     ["book", "course", "school", "college", "tuition", "fees",
                      "coaching", "class", "exam", "education"],
}

INCOME_KEYWORDS = {
    "salary":    ["salary", "pay", "paycheck", "wage", "ctc"],
    "freelance": ["freelance", "project", "client", "gig", "contract"],
    "business":  ["business", "profit", "revenue", "sale", "shop"],
    "bonus":     ["bonus", "incentive", "reward", "commission"],
    "gift":      ["gift", "received", "birthday", "festival", "diwali"],
}


def _extract_amount(text: str):
    """
    Extract a numeric amount from text.

    Supports:
      200 | ₹200 | ₹ 200 | 2.5k | 1,500 | 1500.00
    Returns float or None.
    """
    # Normalise: remove commas used as thousands separators inside numbers
    text = text.replace(",", "")

    # Match optional ₹, optional spaces, digits (with optional decimal), optional k
    pattern = r'₹?\s*(\d+(?:\.\d+)?)(k)?'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    for num_str, k_suffix in matches:
        try:
            value = float(num_str)
            if k_suffix.lower() == "k":
                value *= 1000           # 2.5k → 2500
            if value > 0:
                return value
        except ValueError:
            continue
    return None


def _extract_description(text: str, amount_raw: str | None = None) -> str:
    """
    Extract a human-readable description from the message.

    Handles sentences like:
      "I spent 200 for pizza"
      "Paid 500 for uber"
      "Salary 50000"
      "Got 10000 from freelance"
    """
    t = text.lower().strip()

    # Remove leading filler phrases
    filler = [
        r'^i (spent|paid|bought|purchased|ordered|got|received|earned)\s+',
        r'^(spent|paid|bought|purchased|ordered|got|received|earned)\s+',
        r'^salary\s+',
        r'^bonus\s+',
    ]
    for pat in filler:
        t = re.sub(pat, '', t)

    # Remove the amount token (digits/k/₹) and surrounding connectors
    t = re.sub(r'₹?\s*\d+(?:,\d+)*(?:\.\d+)?\s*k?', '', t, flags=re.IGNORECASE)

    # Remove connector words
    connectors = r'\b(for|on|towards|to|from|in|at|of|the|a|an)\b'
    t = re.sub(connectors, ' ', t)

    # Collapse whitespace and strip
    t = re.sub(r'\s+', ' ', t).strip()

    return t if len(t) >= 2 else ""


def _detect_category(text: str, txn_type: str = "expense") -> str:
    """
    Detect category by scanning for keyword matches.
    Returns category string or empty string.
    """
    t = text.lower()
    keyword_map = EXPENSE_KEYWORDS if txn_type != "income" else INCOME_KEYWORDS

    for category, keywords in keyword_map.items():
        for kw in keywords:
            if kw in t:
                return category
    return ""


def _extract_date(text: str) -> str:
    """
    Extract a date from text.
    Supports "today", "yesterday", YYYY-MM-DD, DD/MM/YYYY.
    Returns date string "YYYY-MM-DD" or empty string.
    """
    t = text.lower()

    if "today" in t:
        return str(date.today())
    if "yesterday" in t:
        return str(date.today() - timedelta(days=1))

    # YYYY-MM-DD
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1)

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    return ""


def extract_entities(text: str) -> dict:
    """
    Extract structured fields from a natural-language message.

    Returns:
        {
            "amount":      float | None,
            "description": str,
            "category":    str,
            "date":        str   (YYYY-MM-DD or "")
        }

    Examples:
        "I spent 200 for pizza"
            → amount=200, description="pizza", category="food", date=""

        "Paid ₹2.5k for uber yesterday"
            → amount=2500, description="uber", category="travel", date="2026-06-06"

        "Salary 50000"
            → amount=50000, description="salary", category="salary", date=""
    """
    amount      = _extract_amount(text)
    description = _extract_description(text)
    category    = _detect_category(text)
    txn_date    = _extract_date(text)

    return {
        "amount":      amount,
        "description": description,
        "category":    category,
        "date":        txn_date,
    }
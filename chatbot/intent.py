# ==============================================================================
# chatbot/intent.py  –  Detect what the user wants to do
# ==============================================================================
# HOW IT WORKS:
#   We have a list of trigger words for every intent.
#   We check if any of those words appear in the user message.
#   The intent with the most matches wins.
# ==============================================================================

# ── Intent keyword lists ──────────────────────────────────────────────────────
# Add more words here to improve detection.

INTENT_KEYWORDS = {

    "expense": [
        "spent", "spend", "bought", "buy", "paid", "pay",
        "purchased", "purchase", "expense", "cost", "debited",
        "i spent", "i paid", "i bought", "money went",
    ],

    "income": [
        "received", "got", "earned", "income", "salary",
        "credited", "payment received", "got paid", "i got",
        "freelance", "bonus", "wage", "stipend","scholarship"
    ],

    "cancel": [
        "cancel", "stop", "abort", "quit", "exit",
        "nevermind", "never mind", "forget it", "go back",
    ],

    "view_expenses": [
        "show expenses", "view expenses", "list expenses",
        "my expenses", "all expenses", "expense history",
    ],

    "view_income": [
        "show income", "view income", "list income",
        "my income", "all income", "income history",
    ],

    "analytics": [
        "analytics", "summary", "report", "overview",
        "how am i doing", "financial stats", "total spending",
    ],

    "budget": [
        "budget", "spending limit", "set budget", "create budget",
        "add budget"
    ],

    "goal": [
        "goal", "savings goal", "saving for", "want to save",
        "financial goal", "saving target",
    ],

    "financial_advice": [
        "advice", "tips", "suggest", "how to save", "save money",
        "reduce expenses", "financial planning",
    ],

    "chat": [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "what's up", "help",
    ],
}


def detect_intent(message: str) -> str:
    """
    Look at the user message and return the best matching intent.

    Steps:
      1. Lowercase the message
      2. Check every intent's keywords
      3. Score +2 for multi-word match, +1 for single-word match
      4. Return the intent with the highest score
      5. Return "unknown" if nothing matches well enough

    Example:
      "I spent 300 on pizza"  →  "expense"
      "show me my income"     →  "view_income"
      "hello"                 →  "chat"
    """

    msg = message.lower().strip()
    scores = {}  # intent → score

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in msg:
                # Multi-word keywords are stronger signals
                if " " in keyword:
                    score += 2
                else:
                    score += 1
        if score > 0:
            scores[intent] = score

    # Debug log – shows what scores each intent got
    print(f"[INTENT DEBUG] Message: '{msg}'")
    print(f"[INTENT DEBUG] Scores: {scores}")

    if not scores:
        print("[INTENT DEBUG] Result: unknown")
        return "unknown"

    # Get the intent with the highest score
    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]

    # Need at least score of 1 to count
    if best_score < 1:
        print("[INTENT DEBUG] Result: unknown (score too low)")
        return "unknown"

    print(f"[INTENT DEBUG] Result: {best_intent} (score={best_score})")
    return best_intent
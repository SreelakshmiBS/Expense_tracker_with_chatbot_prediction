# ==============================================================================
# chatbot/chatbot_engine.py  –  SmartExpenseAI Chatbot Brain
# ==============================================================================
# This module is the heart of the chatbot. It receives a plain-text message
# from the user, figures out what they want (intent detection), extracts the
# important numbers and words from their message (entity extraction), and then
# either saves data to the database or returns a helpful text reply.
#
# Supported capabilities:
#   → Record expenses and income (single-shot or step-by-step flow)
#   → Set, view, update, and delete spending budgets
#   → Create, view, update, and delete savings goals
#   → Show a financial analytics summary
#   → Give random financial advice tips
# ==============================================================================


# --- Standard library ---
import logging
import random
import re
from datetime import date

# --- Our own chatbot sub-modules ---
from chatbot.intent import detect_intent     # Classifies the user's message into an intent label
from chatbot.entities import extract_entities  # Pulls out amount, description, category, date
from chatbot.state_manage import StateManager      # Tracks multi-step conversation flows per user

# --- ML modules ---
from machine_learning.predict_category import predict_category   # Guesses category from description
from machine_learning.utils.dataset_sync import append_to_dataset  # Saves examples for future training


# ── Logging Setup ──────────────────────────────────────────────────────────────
# We log at DEBUG level so we can trace exactly what's happening at each step.
# This is useful for diagnosing wrong intents, missing entities, or DB errors.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SmartExpenseAI")


# ── Valid Category Sets ────────────────────────────────────────────────────────
# These are the only categories the app recognises.
# The ML model and entity extractor should map everything into these buckets.

EXPENSE_CATEGORIES = {
    "food", "travel", "rent", "shopping", "bills",
    "entertainment", "medical", "education", "general",
}

INCOME_CATEGORIES = {
    "salary", "freelance", "business", "bonus", "gift", "general",
}

# Period values accepted for budgets
BUDGET_PERIODS = {"weekly", "monthly", "yearly"}


# ── Cancel Trigger Words ───────────────────────────────────────────────────────
# If the user types any of these during a multi-step flow, we abandon the flow
# and let them start fresh — no data is saved.
CANCEL_WORDS = {"cancel", "stop", "quit", "abort", "exit", "nevermind", "never mind"}


# ==============================================================================
# DATABASE HELPER — Save a Transaction
# ==============================================================================

def save_to_database(mysql, user_id, txn_type: str, amount: float,
                     description: str, category: str, txn_date: str) -> bool:
    """
    Inserts a single income or expense record into the transactions table.

    Also calls append_to_dataset() after a successful save so that every
    real transaction the user enters becomes a future ML training example.

    Returns True if the insert succeeded, False if anything went wrong.
    """
    log.debug("[DB] Saving: user=%s type=%s amount=%s cat=%s desc='%s'",
              user_id, txn_type, amount, category, description)
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO transactions
                (user_id, type, amount, description, category, transaction_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, txn_type, amount, description, category, txn_date)
        )
        mysql.connection.commit()   # Flush the insert to disk
        cur.close()

        log.info("[DB] Saved transaction successfully!")

        # Feed this labelled example to the ML training dataset
        append_to_dataset(description, amount, txn_type, category)

        return True

    except Exception as e:
        log.error("[DB] Save failed: %s", e)
        return False


# ==============================================================================
# DATABASE HELPERS — Budgets
# ==============================================================================

def save_budget(mysql, user_id, category: str, amount: float, period: str) -> bool:
    """
    Creates a new budget or silently updates the amount if one already exists
    for the same user / category / period combination.

    The ON DUPLICATE KEY UPDATE trick requires a UNIQUE KEY on
    (user_id, category, period) in the budgets table. This way the user can
    say 'set food budget 6000 monthly' twice and the second call just updates
    the amount instead of creating a duplicate row.

    Returns True on success, False on any database error.
    """
    log.debug("[BUDGET-DB] user=%s  cat=%s  amount=%s  period=%s",
              user_id, category, amount, period)
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO budgets (user_id, category, amount, period)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE amount = VALUES(amount)
            """,
            (user_id, category, amount, period)
        )
        mysql.connection.commit()
        cur.close()
        log.info("[BUDGET-DB] ✅ Budget saved!")
        return True

    except Exception as e:
        log.error("[BUDGET-DB] ❌ %s", e)
        return False


def get_budgets(mysql, user_id) -> list:
    """
    Returns every budget for a user, each enriched with the amount the user
    has actually spent in that category during the budget's active period.

    The JOIN logic is period-aware:
      - 'monthly' → only counts expenses in the current calendar month
      - 'weekly'  → only counts expenses in the current ISO week
      - 'yearly'  → only counts expenses in the current calendar year

    Returns a list of tuples: (id, category, amount, period, spent)
    Returns an empty list if there are no budgets or if a DB error occurs.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT b.id, b.category, b.amount, b.period,
                   COALESCE(SUM(t.amount), 0) AS spent
            FROM budgets b
            LEFT JOIN transactions t
                ON t.user_id = b.user_id
               AND t.category = b.category
               AND t.type     = 'expense'
               AND (
                    (b.period = 'monthly' AND YEAR(t.transaction_date)  = YEAR(CURDATE())
                                         AND MONTH(t.transaction_date) = MONTH(CURDATE()))
                 OR (b.period = 'weekly'  AND YEARWEEK(t.transaction_date, 1) = YEARWEEK(CURDATE(), 1))
                 OR (b.period = 'yearly'  AND YEAR(t.transaction_date)  = YEAR(CURDATE()))
               )
            WHERE b.user_id = %s
            GROUP BY b.id, b.category, b.amount, b.period
            ORDER BY b.category
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        return rows   # Each row → (id, category, amount, period, spent)

    except Exception as e:
        log.error("[BUDGET-DB] get_budgets error: %s", e)
        return []


def delete_budget(mysql, user_id, category: str) -> bool:
    """
    Deletes the budget row for a given user and category.
    The LOWER() on both sides makes the match case-insensitive,
    so 'Food' and 'food' both hit the same row.

    Returns True if a row was actually deleted, False if nothing matched.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "DELETE FROM budgets WHERE user_id = %s AND LOWER(category) = LOWER(%s)",
            (user_id, category)
        )
        affected = cur.rowcount    # How many rows the DELETE touched
        mysql.connection.commit()
        cur.close()
        return affected > 0        # True only if something was actually removed

    except Exception as e:
        log.error("[BUDGET-DB] delete_budget error: %s", e)
        return False


def get_budget_warnings(mysql, user_id) -> list:
    """
    Scans all of a user's budgets and returns a list of warning strings
    for any category that has reached 80% or more of its budget limit.

    This is called from the analytics summary so the user sees alerts
    whenever they view their financial overview — not just after adding expenses.

    Returns a list of formatted warning strings (empty list if all is fine).
    """
    warnings = []

    for row in get_budgets(mysql, user_id):
        _, category, budget_amount, period, spent = row
        budget_amount = float(budget_amount)
        spent         = float(spent)

        # Skip any budget that was set to zero (shouldn't happen, but be safe)
        if budget_amount <= 0:
            continue

        pct = (spent / budget_amount) * 100

        if pct >= 100:
            # Over budget — critical alert
            warnings.append(
                f" {category.title()} budget EXCEEDED! "
                f"Spent ₹{spent:,.2f} of ₹{budget_amount:,.2f} ({pct:.0f}%) [{period}]"
            )
        elif pct >= 80:
            # Approaching the limit — caution alert
            warnings.append(
                f"{category.title()} budget at {pct:.0f}% — "
                f"₹{budget_amount - spent:,.2f} remaining [{period}]"
            )

    return warnings


# ==============================================================================
# DATABASE HELPERS — Goals
# ==============================================================================

def save_goal(mysql, user_id, goal_name: str, target_amount: float,
              current_amount: float = 0.0, target_date: str = None) -> bool:
    """
    Inserts a new savings goal into the goals table.

    current_amount defaults to 0.0 (the user hasn't saved anything yet).
    target_date is optional — the user can skip it and we store NULL.

    Returns True on success, False on any database error.
    """
    log.debug("[GOAL-DB] user=%s  name='%s'  target=%s  current=%s  date=%s",
              user_id, goal_name, target_amount, current_amount, target_date)
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO goals (user_id, goal_name, target_amount, current_amount, target_date)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, goal_name, target_amount, current_amount, target_date)
        )
        mysql.connection.commit()
        cur.close()
        log.info("[GOAL-DB]  Goal saved!")
        return True

    except Exception as e:
        log.error("[GOAL-DB] ❌ %s", e)
        return False


def update_goal_amount(mysql, user_id, goal_name: str, new_target: float) -> bool:
    """
    Changes the target_amount of an existing goal without touching
    the current_amount or target_date.

    The LOWER() comparison makes the name match case-insensitively,
    so 'Bike' and 'bike' both find the same goal.

    Returns True if a row was updated, False if no matching goal exists.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            UPDATE goals
            SET target_amount = %s
            WHERE user_id = %s AND LOWER(goal_name) = LOWER(%s)
            """,
            (new_target, user_id, goal_name)
        )
        affected = cur.rowcount
        mysql.connection.commit()
        cur.close()
        return affected > 0   # True means a matching row was found and updated

    except Exception as e:
        log.error("[GOAL-DB] update_goal_amount error: %s", e)
        return False


def contribute_to_goal(mysql, user_id, goal_name: str, contribution: float) -> bool:
    """
    Adds a contribution to the current_amount of a goal using an
    in-database addition (current_amount + contribution) so there's no
    race condition if two requests arrive at the same time.

    Returns True if the goal was found and updated, False otherwise.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            UPDATE goals
            SET current_amount = current_amount + %s
            WHERE user_id = %s AND LOWER(goal_name) = LOWER(%s)
            """,
            (contribution, user_id, goal_name)
        )
        affected = cur.rowcount
        mysql.connection.commit()
        cur.close()
        return affected > 0

    except Exception as e:
        log.error("[GOAL-DB] contribute error: %s", e)
        return False


def get_goals(mysql, user_id) -> list:
    """
    Returns all savings goals for a user, newest first.

    Each row is a tuple:
      (id, goal_name, target_amount, current_amount, target_date, created_at)

    Returns an empty list if the user has no goals or if a DB error occurs.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT id, goal_name, target_amount, current_amount, target_date, created_at
            FROM goals
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        return rows

    except Exception as e:
        log.error("[GOAL-DB] get_goals error: %s", e)
        return []


def delete_goal(mysql, user_id, goal_name: str) -> bool:
    """
    Deletes a goal by name for the given user.
    Case-insensitive match so 'Bike' and 'bike' both work.

    Returns True if the goal was found and deleted, False if nothing matched.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "DELETE FROM goals WHERE user_id = %s AND LOWER(goal_name) = LOWER(%s)",
            (user_id, goal_name)
        )
        affected = cur.rowcount
        mysql.connection.commit()
        cur.close()
        return affected > 0

    except Exception as e:
        log.error("[GOAL-DB] delete_goal error: %s", e)
        return False


def _goal_progress_line(goal_name, target_amount, current_amount) -> str:
    """
    Builds a compact two-line ASCII progress summary for a single goal.
    Used by _view_goals() to format each goal consistently.

    Example output:
       Bike               [████████████░░░░░░░░] 60.0%
           ₹60,000.00 saved  |  ₹40,000.00 to go
    """
    target    = float(target_amount)
    current   = float(current_amount)

    # Percentage complete, capped at 100 so the bar never overflows
    pct       = min((current / target * 100), 100) if target > 0 else 0

    # Build the ASCII progress bar — 20 characters wide
    bar_len   = 20
    filled    = int(bar_len * pct / 100)
    bar       = "█" * filled + "░" * (bar_len - filled)

    # How much is still needed to reach the target?
    remaining = max(target - current, 0)

    return (
        f"   {goal_name.title():<18} [{bar}] {pct:5.1f}%\n"
        f"     ₹{current:>10,.2f} saved  |  ₹{remaining:>10,.2f} to go"
    )


# ==============================================================================
# CHATBOT ENGINE CLASS
# ==============================================================================

class ChatbotEngine:
    """
    The main chatbot brain that routes messages to the right handler.

    Responsibilities:
      1. Check if the user is mid-way through a multi-step flow (e.g. adding
         an expense step by step) — if so, continue that flow.
      2. Otherwise, detect the intent of their message and call the correct handler.
      3. Each handler either responds immediately (if it has enough information)
         or starts a flow to ask for missing details one question at a time.
    """

    def __init__(self, mysql):
        # Hold a reference to the DB connection so every method can use it
        self.mysql = mysql

        # StateManager tracks which flow (if any) each user is currently in
        # and stores the partial data collected so far
        self.state = StateManager()


    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    def process_message(self, user_id, message: str, username: str = "User") -> str:
        """
        The single public method called by the Flask route for every chat message.

        Flow:
          1. If the user is already in a multi-step flow → continue it.
          2. If they typed a cancel word → abandon the flow.
          3. Otherwise → detect intent → dispatch to the right handler.
        """
        msg = message.strip()

        # Ignore truly empty messages
        if not msg:
            return "Please type a message."

        log.debug("=" * 55)
        log.debug("[MSG] user=%s  message='%s'", user_id, msg)

        # ── Check for an active conversation flow first ────────────────────────
        # If the user was mid-way through adding an expense (for example),
        # we need to continue collecting the missing fields rather than
        # re-detecting the intent from scratch.
        current_flow = self.state.get_flow(user_id)
        if current_flow:
            if msg.lower() in CANCEL_WORDS:
                # User wants out — wipe the flow and let them start fresh
                self.state.reset(user_id)
                return "❌ Cancelled. What else can I help you with?"
            return self._continue_flow(user_id, msg, current_flow)

        # ── No active flow — detect intent from scratch ────────────────────────
        intent = detect_intent(msg)
        log.debug("[INTENT] → %s", intent)

        # ── Route to the right handler based on detected intent ────────────────

        if intent == "expense":
            return self._handle_expense(user_id, msg)

        elif intent == "income":
            return self._handle_income(user_id, msg)

        elif intent == "cancel":
            # Nothing to cancel — the user isn't mid-flow
            return "ℹ️ Nothing to cancel. Just tell me what you'd like to do!"

        elif intent == "view_expenses":
            return self._view_transactions(user_id, "expense")

        elif intent == "view_income":
            return self._view_transactions(user_id, "income")

        elif intent == "analytics":
            return self._show_analytics(user_id)

        elif intent == "financial_advice":
            return self._give_advice()

        # ── Budget intents ─────────────────────────────────────────────────────
        elif intent in ("budget", "add_budget"):
            return self._handle_budget(user_id, msg)

        elif intent == "view_budgets":
            return self._view_budgets(user_id)

        elif intent == "update_budget":
            return self._handle_update_budget(user_id, msg)

        elif intent == "delete_budget":
            return self._handle_delete_budget(user_id, msg)

        # ── Goal intents ───────────────────────────────────────────────────────
        elif intent in ("goal", "add_goal"):
            return self._handle_goal(user_id, msg)

        elif intent == "view_goals":
            return self._view_goals(user_id)

        elif intent == "update_goal":
            return self._handle_update_goal(user_id, msg)

        elif intent == "delete_goal":
            return self._handle_delete_goal(user_id, msg)

        # ── Generic greeting / help ────────────────────────────────────────────
        elif intent == "chat":
            return (
                f"👋 Hi {username}!!! I'm your SmartExpenseAI Financial Assistant.\n\n"
                "Here's what I can do:\n"
                "  Track expenses    → 'I spent 300 on pizza'\n"
                "  Record income     → 'Got salary 50000'\n"
                "  Analytics         → 'show analytics'\n"
                "  List expenses     → 'show expenses'\n"
                "  Budgets           → 'set budget for food 5000 monthly'\n"
                "  Goals             → 'save 100000 for bike'\n\n"
                "What would you like to do?"
            )

        # ── Fallback — try treating it as an expense if there's an amount ─────
        else:
            entities = extract_entities(msg)
            if entities["amount"]:
                # If we spotted a number, it's probably an expense — try it
                return self._handle_expense(user_id, msg)

            # Give up and show helpful examples
            return (
                " I didn't quite understand that.\n\n"
                "Try saying:\n"
                "  • 'I spent 300 on pizza'\n"
                "  • 'Got salary 50000'\n"
                "  • 'set food budget 5000 monthly'\n"
                "  • 'save 100000 for bike'\n"
                "  • 'show budgets'\n"
                "  • 'show goals'\n"
                "  • 'show analytics'"
            )


    # ==========================================================================
    # EXPENSE HANDLER
    # ==========================================================================

    def _handle_expense(self, user_id, message: str) -> str:
        """
        Tries to record an expense from a single message.

        If the message is complete ('spent 500 on pizza'), it saves immediately
        and returns a confirmation.

        If something is missing (no amount, or no description), it starts a
        multi-step flow and asks for the missing piece.

        After saving, it also checks whether this category is close to its
        budget limit and appends a warning if needed.
        """
        log.debug("[EXPENSE] Handling: '%s'", message)

        # Pull whatever structured data we can from the raw message
        entities    = extract_entities(message)
        amount      = entities["amount"]
        description = entities["description"]
        category    = entities["category"]
        txn_date    = entities["date"] or str(date.today())   # Fall back to today

        # ── Step 1: We need an amount ──────────────────────────────────────────
        if not amount or amount <= 0:
            self.state.start_flow(user_id, "expense")
            return "💸 How much did you spend? (e.g. 300, 1500, 25k)"

        # ── Step 2: We need to know what it was for ────────────────────────────
        if not description or len(description) < 2:
            self.state.start_flow(user_id, "expense")
            self.state.save_data(user_id, "amount", amount)     # Remember the amount for step 2
            self.state.save_data(user_id, "date", txn_date)
            return f"🏷️ What did you spend ₹{amount:,.0f} on?"

        # ── Step 3: Category — predict it if not already extracted ────────────
        if not category:
            category = predict_category(description, "expense")
        if not category or category == "unknown":
            category = "general"   # Safe fallback if ML can't guess

        # ── Save the complete transaction ──────────────────────────────────────
        saved = save_to_database(
            self.mysql, user_id,
            txn_type="expense", amount=amount,
            description=description, category=category, txn_date=txn_date,
        )

        # Check if this category is now close to or over its budget limit
        warning = self._budget_warning_for_category(user_id, category)

        if saved:
            reply = (
                f"   Expense recorded!\n"
                f"   Amount      : ₹{amount:,.2f}\n"
                f"   Description : {description.title()}\n"
                f"   Category    : {category.title()}\n"
                f"   Date        : {txn_date}"
            )
            # Tack on the budget warning if one was triggered
            if warning:
                reply += f"\n\n{warning}"
            return reply

        return "❌ Sorry, I couldn't save your expense. Please try again."


    # ==========================================================================
    # INCOME HANDLER
    # ==========================================================================

    def _handle_income(self, user_id, message: str) -> str:
        """
        Tries to record an income transaction from a single message.
        Mirrors the expense handler's logic — checks for amount and description,
        starts a flow if either is missing, then saves and confirms.
        """
        log.debug("[INCOME] Handling: '%s'", message)

        entities    = extract_entities(message)
        amount      = entities["amount"]
        description = entities["description"]
        category    = entities["category"]
        txn_date    = entities["date"] or str(date.today())

        # ── Need an amount first ───────────────────────────────────────────────
        if not amount or amount <= 0:
            self.state.start_flow(user_id, "income")
            return " How much income did you receive? (e.g. 50000, 25k)"

        # ── Need to know the income source ────────────────────────────────────
        if not description or len(description) < 2:
            self.state.start_flow(user_id, "income")
            self.state.save_data(user_id, "amount", amount)
            self.state.save_data(user_id, "date", txn_date)
            return f" What was this ₹{amount:,.0f} income for? (e.g. salary, freelance)"

        # ── Predict category if not found in the message ───────────────────────
        if not category:
            category = predict_category(description, "income")
        if not category or category == "unknown":
            category = "general"

        saved = save_to_database(
            self.mysql, user_id,
            txn_type="income", amount=amount,
            description=description, category=category, txn_date=txn_date,
        )

        if saved:
            return (
                f" Income recorded!\n"
                f"   Amount      : ₹{amount:,.2f}\n"
                f"   Description : {description.title()}\n"
                f"   Category    : {category.title()}\n"
                f"   Date        : {txn_date}"
            )

        return "❌ Sorry, I couldn't save your income. Please try again."


    # ==========================================================================
    # BUDGET HANDLERS
    # ==========================================================================

    def _handle_budget(self, user_id, message: str) -> str:
        """
        Handles 'set budget for food 5000 monthly'.

        Tries to extract all three required pieces from the message in one shot:
          1. Category (food, travel, etc.)
          2. Amount (5000)
          3. Period (weekly / monthly / yearly)

        If anything is missing, starts a step-by-step flow to ask for it.
        """
        log.debug("[BUDGET] Handling: '%s'", message)

        entities = extract_entities(message)

        # Try our budget-specific category extractor first (it checks EXPENSE_CATEGORIES),
        # then fall back to the general entity extractor's category field
        category = self._extract_budget_category(message) or entities.get("category")
        amount   = entities.get("amount")
        period   = self._extract_period(message)

        log.debug("[BUDGET] cat=%s  amount=%s  period=%s", category, amount, period)

        # ── Ask for category if missing ────────────────────────────────────────
        if not category:
            self.state.start_flow(user_id, "budget")
            return (
                "🏦 Which category would you like to set a budget for?\n"
                "(e.g. food, travel, shopping, rent, bills, entertainment)"
            )

        # ── Ask for amount if missing ──────────────────────────────────────────
        if not amount or amount <= 0:
            self.state.start_flow(user_id, "budget")
            self.state.save_data(user_id, "category", category)   # Hold category for next step
            return f" How much should the {category.title()} budget be?"

        # ── Ask for period if missing ──────────────────────────────────────────
        if not period:
            self.state.start_flow(user_id, "budget")
            self.state.save_data(user_id, "category", category)
            self.state.save_data(user_id, "amount", amount)
            return " Should this be weekly, monthly, or yearly?"

        # We have everything — save it
        return self._save_budget_and_reply(user_id, category, amount, period)


    def _handle_update_budget(self, user_id, message: str) -> str:
        """
        Handles 'update food budget to 7000'.

        Period is optional for updates — if the user doesn't mention one we
        default to 'monthly' and let the ON DUPLICATE KEY UPDATE handle it,
        which effectively keeps the existing period in the DB row.
        """
        log.debug("[BUDGET-UPDATE] '%s'", message)

        entities = extract_entities(message)
        category = self._extract_budget_category(message)
        amount   = entities.get("amount")
        period   = self._extract_period(message)

        # ── Need to know which category to update ─────────────────────────────
        if not category:
            self.state.start_flow(user_id, "update_budget")
            return " Which budget category would you like to update?"

        # ── Need a new amount ──────────────────────────────────────────────────
        if not amount or amount <= 0:
            self.state.start_flow(user_id, "update_budget")
            self.state.save_data(user_id, "category", category)
            return f What should the new amount be for your {category.title()} budget?"

        # Use 'monthly' as a safe default if no period was mentioned
        if not period:
            period = "monthly"

        return self._save_budget_and_reply(user_id, category, amount, period, is_update=True)


    def _handle_delete_budget(self, user_id, message: str) -> str:
        """
        Handles 'delete shopping budget'.

        If we can pull the category name straight from the message we delete
        immediately; otherwise we start a single-step flow to ask which one.
        """
        log.debug("[BUDGET-DELETE] '%s'", message)

        category = self._extract_budget_category(message)

        if not category:
            self.state.start_flow(user_id, "delete_budget")
            return " Which budget category would you like to delete?"

        deleted = delete_budget(self.mysql, user_id, category)
        if deleted:
            return f" {category.title()} budget deleted successfully."

        # Nothing was deleted — the category probably doesn't exist
        return f"⚠️ No budget found for '{category}'. Check 'show budgets' for your list."


    def _view_budgets(self, user_id) -> str:
        """
        Fetches all budgets and renders them as an ASCII table with
        progress bars showing how much of each budget has been used.

        Status icons:
          ✅ = under 80%
          ⚠️  = 80–99% (approaching limit)
          🚨  = 100%+ (over budget)
        """
        rows = get_budgets(self.mysql, user_id)

        if not rows:
            return "🏦 No budgets set yet.\nTry: 'set food budget 5000 monthly'"

        lines = ["🏦 Your Budgets\n" + "─" * 52]

        for _, category, budget_amt, period, spent in rows:
            budget_amt = float(budget_amt)
            spent      = float(spent)
            remaining  = max(budget_amt - spent, 0)

            # Percentage used, capped at 100 so the bar never overflows visually
            pct        = min((spent / budget_amt * 100), 100) if budget_amt > 0 else 0

            # ASCII progress bar — 16 chars wide
            bar_len    = 16
            filled     = int(bar_len * pct / 100)
            bar        = "█" * filled + "░" * (bar_len - filled)

            # Pick a status icon based on how close they are to the limit
            if pct >= 100:
                status = "🚨"
            elif pct >= 80:
                status = "⚠️ "
            else:
                status = "✅"

            lines.append(
                f"{status} {category.title():<14} [{bar}] {pct:5.1f}%  [{period}]\n"
                f"     Spent: ₹{spent:>9,.2f}  |  Budget: ₹{budget_amt:>9,.2f}  |  Left: ₹{remaining:>9,.2f}"
            )

        return "\n".join(lines)


    def _save_budget_and_reply(self, user_id, category, amount, period,
                               is_update=False) -> str:
        """
        Shared helper used by both create and update budget flows.
        Calls the DB function and formats a friendly confirmation message.

        is_update changes the verb in the reply ('set' vs 'updated') so the
        user understands whether a new budget was created or an existing one changed.
        """
        saved = save_budget(self.mysql, user_id, category, amount, period)

        if saved:
            verb = "updated" if is_update else "set"
            return (
                f"   Budget {verb}!\n"
                f"   Category : {category.title()}\n"
                f"   Amount   : ₹{amount:,.2f}\n"
                f"   Period   : {period.title()}\n\n"
                f"I'll warn you when you reach 80% of this budget."
            )

        return "❌ Couldn't save budget. Please try again."


    def _budget_warning_for_category(self, user_id, category: str) -> str:
        """
        Called right after saving an expense to check if the spending in
        that specific category has crossed the 80% or 100% threshold.

        Returns a formatted warning string if the threshold is crossed,
        or an empty string if everything is fine (or no budget is set).
        """
        rows = get_budgets(self.mysql, user_id)

        for _, cat, budget_amt, period, spent in rows:
            if cat.lower() == category.lower():
                budget_amt = float(budget_amt)
                spent      = float(spent)

                # Skip if the budget was somehow set to zero
                if budget_amt <= 0:
                    return ""

                pct       = spent / budget_amt * 100
                remaining = max(budget_amt - spent, 0)

                if pct >= 100:
                    return (
                        f"🚨 Budget Alert: You've EXCEEDED your {cat.title()} "
                        f"{period} budget!\n"
                        f"   Spent ₹{spent:,.2f} of ₹{budget_amt:,.2f} ({pct:.0f}%)"
                    )
                elif pct >= 80:
                    return (
                        f"⚠️  Budget Alert: You've used {pct:.0f}% of your "
                        f"{cat.title()} {period} budget.\n"
                        f"   ₹{remaining:,.2f} remaining."
                    )

        # No matching budget found for this category — nothing to warn about
        return ""


    # ==========================================================================
    # GOAL HANDLERS
    # ==========================================================================

    def _handle_goal(self, user_id, message: str) -> str:
        """
        Handles 'save 100000 for bike' or 'create savings goal'.

        Extracts goal_name and target_amount from the message.
        If either is missing, starts a step-by-step flow to collect them.

        An optional target_date can also be pulled from the message
        (e.g. 'save 50000 for vacation by December').
        """
        log.debug("[GOAL] Handling: '%s'", message)

        entities      = extract_entities(message)
        goal_name     = self._extract_goal_name(message)
        target_amount = entities.get("amount")
        target_date   = entities.get("date")

        log.debug("[GOAL] name='%s'  target=%s  date=%s",
                  goal_name, target_amount, target_date)

        # ── Need a name for the goal ───────────────────────────────────────────
        if not goal_name:
            self.state.start_flow(user_id, "goal")
            return "🎯 What would you like to save for? (e.g. bike, car, vacation, emergency fund)"

        # ── Need a target amount ───────────────────────────────────────────────
        if not target_amount or target_amount <= 0:
            self.state.start_flow(user_id, "goal")
            self.state.save_data(user_id, "goal_name", goal_name)
            return f"💰 How much do you want to save for your {goal_name.title()} goal?"

        # Have everything we need — save immediately
        return self._save_goal_and_reply(user_id, goal_name, target_amount, 0.0, target_date)


    def _handle_update_goal(self, user_id, message: str) -> str:
        """
        Handles 'update bike goal to 150000'.
        Extracts the goal name and the new target amount from the message.
        """
        log.debug("[GOAL-UPDATE] '%s'", message)

        entities      = extract_entities(message)
        goal_name     = self._extract_goal_name(message)
        target_amount = entities.get("amount")

        if not goal_name:
            self.state.start_flow(user_id, "update_goal")
            return " Which goal would you like to update?"

        if not target_amount or target_amount <= 0:
            self.state.start_flow(user_id, "update_goal")
            self.state.save_data(user_id, "goal_name", goal_name)
            return f"💰 What should the new target amount be for your {goal_name.title()} goal?"

        updated = update_goal_amount(self.mysql, user_id, goal_name, target_amount)

        if updated:
            return (
                f"   Goal updated!\n"
                f"   Goal       : {goal_name.title()}\n"
                f"   New Target : ₹{target_amount:,.2f}"
            )

        return f"⚠️ No goal found named '{goal_name}'. Check 'show goals'."


    def _handle_delete_goal(self, user_id, message: str) -> str:
        """
        Handles 'delete car goal'.
        If the goal name is in the message, deletes immediately.
        Otherwise starts a single-step flow to ask which goal.
        """
        log.debug("[GOAL-DELETE] '%s'", message)

        goal_name = self._extract_goal_name(message)

        if not goal_name:
            self.state.start_flow(user_id, "delete_goal")
            return "🗑️ Which goal would you like to delete?"

        deleted = delete_goal(self.mysql, user_id, goal_name)

        if deleted:
            return f"✅ {goal_name.title()} goal deleted."

        return f"⚠️ No goal named '{goal_name}' found. Type 'show goals' to see your list."


    def _view_goals(self, user_id) -> str:
        """
        Displays all goals split into two sections:
          📌 Active Goals   — goals not yet reached (with motivational notes)
          🏆 Completed Goals — goals that have hit 100% progress

        Each goal shows an ASCII progress bar, the saved vs target amounts,
        and optionally a target date.
        """
        rows = get_goals(self.mysql, user_id)

        if not rows:
            return "🎯 No goals set yet.\nTry: 'save 100000 for bike'"

        active_lines    = []
        completed_lines = []

        for _, goal_name, target_amount, current_amount, target_date, _ in rows:
            target  = float(target_amount)
            current = float(current_amount)
            pct     = min((current / target * 100), 100) if target > 0 else 0

            # Build the shared progress line for this goal
            line = _goal_progress_line(goal_name, target, current)

            # Append the target date under the progress bar if one was set
            if target_date:
                line += f"\n     Target Date : {target_date}"

            if pct >= 100:
                # Goal is done — move it to the completed section
                completed_lines.append(f"🏆 {goal_name.title()} — COMPLETED! 🎉\n" + line)
            else:
                # Add a short motivational message based on how far along they are
                if pct >= 75:
                    note = "  🔥 Almost there!"
                elif pct >= 50:
                    note = "  💪 Halfway done — keep it up!"
                elif pct >= 25:
                    note = "  🌱 Good progress!"
                else:
                    note = "  🚀 Just getting started!"

                active_lines.append(line + "\n" + note)

        # Assemble the full output — active goals first, completed at the bottom
        output = ["🎯 Your Savings Goals\n" + "─" * 52]

        if active_lines:
            output.append("📌 Active Goals:")
            output.extend(active_lines)

        if completed_lines:
            output.append("\n🏆 Completed Goals:")
            output.extend(completed_lines)

        return "\n".join(output)


    def _save_goal_and_reply(self, user_id, goal_name, target_amount,
                             current_amount=0.0, target_date=None) -> str:
        """
        Shared helper used by both the direct handler and the multi-step flow.
        Saves the goal and returns a motivational confirmation message.
        """
        saved = save_goal(
            self.mysql, user_id, goal_name,
            target_amount, current_amount, target_date
        )

        if saved:
            reply = (
                f" Goal created!\n"
                f"   Goal   : {goal_name.title()}\n"
                f"   Target : ₹{target_amount:,.2f}\n"
            )
            if target_date:
                reply += f"   By     : {target_date}\n"

            # Point the user towards analytics so they can track progress
            reply += (
                "\n💡 Tip: Use 'show analytics' to track your savings progress "
                "towards this goal!"
            )
            return reply

        return "❌ Couldn't save goal. Please try again."


    # ==========================================================================
    # MULTI-STEP FLOW CONTINUATION
    # ==========================================================================

    def _continue_flow(self, user_id, message: str, flow: str) -> str:
        """
        Called when the user is mid-way through a guided conversation flow.

        Each flow is a mini state machine. The data dict holds whatever we've
        already collected in previous turns. We check what's still missing,
        ask for it, save it, and eventually complete the action.

        Flows handled:
          expense, income, budget, update_budget, delete_budget,
          goal, update_goal, delete_goal
        """
        data     = self.state.get_data(user_id)
        entities = extract_entities(message)

        log.debug("[FLOW] flow=%s  step data=%s  message='%s'", flow, data, message)

        # ── Expense: step 1 — collect amount, step 2 — collect description ────
        if flow == "expense":

            if "amount" not in data:
                # We don't have the amount yet — try to extract it from this reply
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount (e.g. 500 or 2.5k)."
                self.state.save_data(user_id, "amount", amount)
                return "🏷️ What did you spend it on? (e.g. pizza, uber, gym)"

            if "description" not in data:
                # We have the amount — now we need the description
                desc = message.strip()
                if not desc or len(desc) < 2:
                    return "⚠️ Please describe what you spent on."
                self.state.save_data(user_id, "description", desc)

                # All fields gathered — predict category and save
                amount   = data["amount"]
                txn_date = data.get("date", str(date.today()))
                category = entities["category"] or predict_category(desc, "expense")
                if not category or category == "unknown":
                    category = "general"

                saved = save_to_database(
                    self.mysql, user_id,
                    txn_type="expense", amount=amount,
                    description=desc, category=category, txn_date=txn_date,
                )
                self.state.reset(user_id)   # Flow is complete — clear state

                # Check budget after saving
                warning = self._budget_warning_for_category(user_id, category)

                if saved:
                    reply = (
                        f"✅ Expense recorded!\n"
                        f"   Amount      : ₹{amount:,.2f}\n"
                        f"   Description : {desc.title()}\n"
                        f"   Category    : {category.title()}\n"
                        f"   Date        : {txn_date}"
                    )
                    if warning:
                        reply += f"\n\n{warning}"
                    return reply

                return "❌ Couldn't save expense. Please try again."

        # ── Income: step 1 — collect amount, step 2 — collect description ─────
        elif flow == "income":

            if "amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount (e.g. 50000 or 25k)."
                self.state.save_data(user_id, "amount", amount)
                return "💼 What was this income for? (e.g. salary, freelance work)"

            if "description" not in data:
                desc = message.strip()
                if not desc or len(desc) < 2:
                    return "⚠️ Please describe the income source."
                self.state.save_data(user_id, "description", desc)

                amount   = data["amount"]
                txn_date = data.get("date", str(date.today()))
                category = entities["category"] or predict_category(desc, "income")
                if not category or category == "unknown":
                    category = "general"

                saved = save_to_database(
                    self.mysql, user_id,
                    txn_type="income", amount=amount,
                    description=desc, category=category, txn_date=txn_date,
                )
                self.state.reset(user_id)   # Flow complete

                if saved:
                    return (
                        f"✅ Income recorded!\n"
                        f"   Amount      : ₹{amount:,.2f}\n"
                        f"   Description : {desc.title()}\n"
                        f"   Category    : {category.title()}\n"
                        f"   Date        : {txn_date}"
                    )
                return "❌ Couldn't save income. Please try again."

        # ── Budget: step 1 — category, step 2 — amount, step 3 — period ───────
        elif flow == "budget":

            if "category" not in data:
                # Try to match a known expense category from what they typed
                cat = self._extract_budget_category(message) or message.strip().lower()
                if not cat or len(cat) < 2:
                    return "⚠️ Please enter a valid category (e.g. food, travel, shopping)."
                self.state.save_data(user_id, "category", cat)
                return f"💵 How much should the {cat.title()} budget be?"

            if "amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount (e.g. 5000 or 10k)."
                self.state.save_data(user_id, "amount", amount)
                return "🗓️ Should this be weekly, monthly, or yearly?"

            if "period" not in data:
                period = self._extract_period(message)
                if not period:
                    return "⚠️ Please say weekly, monthly, or yearly."
                self.state.save_data(user_id, "period", period)

                # All three fields collected — save the budget
                category = data["category"]
                amount   = data["amount"]
                self.state.reset(user_id)
                return self._save_budget_and_reply(user_id, category, amount, period)

        # ── Update-budget: step 1 — category, step 2 — new amount ────────────
        elif flow == "update_budget":

            if "category" not in data:
                cat = self._extract_budget_category(message) or message.strip().lower()
                if not cat or len(cat) < 2:
                    return "⚠️ Please enter a valid category."
                self.state.save_data(user_id, "category", cat)
                return f"💵 What should the new amount be for your {cat.title()} budget?"

            if "amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount."
                category = data["category"]
                self.state.reset(user_id)
                # Default period to 'monthly' — the upsert keeps existing period in DB
                return self._save_budget_and_reply(
                    user_id, category, amount, "monthly", is_update=True
                )

        # ── Delete-budget: single step — just need the category name ──────────
        elif flow == "delete_budget":
            cat = self._extract_budget_category(message) or message.strip().lower()
            if not cat or len(cat) < 2:
                return "⚠️ Please enter a valid category name."
            self.state.reset(user_id)
            deleted = delete_budget(self.mysql, user_id, cat)
            if deleted:
                return f"✅ {cat.title()} budget deleted."
            return f"⚠️ No budget found for '{cat}'."

        # ── Goal: step 1 — name, step 2 — target amount, step 3 — date ───────
        elif flow == "goal":

            if "goal_name" not in data:
                goal_name = self._extract_goal_name(message) or message.strip().lower()
                if not goal_name or len(goal_name) < 2:
                    return "⚠️ Please enter a goal name (e.g. bike, vacation, emergency fund)."
                self.state.save_data(user_id, "goal_name", goal_name)
                return f"💰 How much do you want to save for your {goal_name.title()} goal?"

            if "target_amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid target amount (e.g. 100000 or 50k)."
                self.state.save_data(user_id, "target_amount", amount)
                # Date is optional — give the user a clear way to skip it
                return (
                    f"📅 Do you have a target date? (e.g. '2025-12-31')\n"
                    f"   Or type 'skip' to skip."
                )

            if "target_date" not in data:
                raw = message.strip().lower()

                # Accept 'skip' or 'no' as ways to bypass the date step
                if raw in ("skip", "no"):
                    tdate = None
                else:
                    # Try entity extractor first, then use raw text if it looks long enough
                    tdate = entities.get("date") or (
                        message.strip() if len(message.strip()) >= 8 else None
                    )

                self.state.save_data(user_id, "target_date", tdate)

                # All fields collected — save the goal
                goal_name     = data["goal_name"]
                target_amount = data["target_amount"]
                self.state.reset(user_id)
                return self._save_goal_and_reply(
                    user_id, goal_name, target_amount, 0.0, tdate
                )

        # ── Update-goal: step 1 — name, step 2 — new target amount ───────────
        elif flow == "update_goal":

            if "goal_name" not in data:
                goal_name = self._extract_goal_name(message) or message.strip().lower()
                if not goal_name or len(goal_name) < 2:
                    return "⚠️ Please enter a valid goal name."
                self.state.save_data(user_id, "goal_name", goal_name)
                return f"💰 What should the new target be for your {goal_name.title()} goal?"

            if "target_amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount."
                goal_name = data["goal_name"]
                self.state.reset(user_id)
                updated = update_goal_amount(self.mysql, user_id, goal_name, amount)
                if updated:
                    return f"✅ {goal_name.title()} goal updated to ₹{amount:,.2f}."
                return f"⚠️ No goal named '{goal_name}' found."

        # ── Delete-goal: single step — just need the goal name ────────────────
        elif flow == "delete_goal":
            goal_name = self._extract_goal_name(message) or message.strip().lower()
            if not goal_name or len(goal_name) < 2:
                return "⚠️ Please enter a valid goal name."
            self.state.reset(user_id)
            deleted = delete_goal(self.mysql, user_id, goal_name)
            if deleted:
                return f"✅ {goal_name.title()} goal deleted."
            return f"⚠️ No goal named '{goal_name}' found."

        # ── Unknown flow — shouldn't normally reach here ───────────────────────
        self.state.reset(user_id)
        return "⚠️ Something went wrong. Please start over."


    # ==========================================================================
    # VIEW TRANSACTIONS
    # ==========================================================================

    def _view_transactions(self, user_id, txn_type: str) -> str:
        """
        Fetches and formats the 10 most recent transactions of the given type
        (either 'expense' or 'income') into a readable ASCII table.

        Shows amount, category, a truncated description, and date.
        Includes a running total at the bottom.
        """
        try:
            cur = self.mysql.connection.cursor()
            cur.execute(
                """
                SELECT amount, description, category, transaction_date
                FROM transactions
                WHERE user_id = %s AND type = %s
                ORDER BY transaction_date DESC
                LIMIT 10
                """,
                (user_id, txn_type)
            )
            rows = cur.fetchall()
            cur.close()

            if not rows:
                icon = "💸" if txn_type == "expense" else "💰"
                return f"{icon} No {txn_type}s found yet. Try adding one!"

            icon  = "💸" if txn_type == "expense" else "💰"
            title = "Recent Expenses" if txn_type == "expense" else "Recent Income"
            lines = [f"{icon} {title}\n{'─' * 46}"]

            total = 0
            for amount, description, category, txn_date in rows:
                amt    = float(amount)
                total += amt
                # Truncate long descriptions so the table stays aligned
                lines.append(
                    f"  ₹{amt:>9,.2f}  |  {str(category).title():<14}  |  "
                    f"{str(description)[:20]:<20}  |  {txn_date}"
                )

            lines.append(f"{'─' * 46}")
            lines.append(f"  Total: ₹{total:,.2f}")

            return "\n".join(lines)

        except Exception as e:
            log.error("[VIEW] Error: %s", e)
            return f"❌ Couldn't load {txn_type}s right now."


    # ==========================================================================
    # ANALYTICS SUMMARY
    # ==========================================================================

    def _show_analytics(self, user_id) -> str:
        """
        Builds a comprehensive plain-text financial summary covering:
          1. Overall income, expenses, savings, and savings rate
          2. This month's spend and top 3 spending categories
          3. Budget overview (with % used and remaining for each)
          4. Budget warnings (categories at 80%+ of their limit)
          5. Goals progress (with % complete and motivational notes)
          6. A personalised savings rate tip at the bottom

        All sections are only shown if there is data to display.
        """
        try:
            cur = self.mysql.connection.cursor()

            # ── Lifetime income and expense totals ─────────────────────────────
            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE user_id=%s AND type='income'",
                (user_id,)
            )
            total_income = float(cur.fetchone()[0])

            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE user_id=%s AND type='expense'",
                (user_id,)
            )
            total_expense = float(cur.fetchone()[0])

            # ── This month's expense total ─────────────────────────────────────
            cur.execute(
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM transactions
                WHERE user_id=%s AND type='expense'
                  AND YEAR(transaction_date)  = YEAR(CURDATE())
                  AND MONTH(transaction_date) = MONTH(CURDATE())
                """,
                (user_id,)
            )
            month_expense = float(cur.fetchone()[0])

            # ── Top 3 spending categories this month ───────────────────────────
            cur.execute(
                """
                SELECT category, SUM(amount) AS total
                FROM transactions
                WHERE user_id=%s AND type='expense'
                  AND YEAR(transaction_date)  = YEAR(CURDATE())
                  AND MONTH(transaction_date) = MONTH(CURDATE())
                GROUP BY category
                ORDER BY total DESC
                LIMIT 3
                """,
                (user_id,)
            )
            top_cats = cur.fetchall()
            cur.close()

            # Net savings and savings rate
            savings = total_income - total_expense
            rate    = (savings / total_income * 100) if total_income > 0 else 0

            # ── Build the output line by line ──────────────────────────────────
            lines = [
                "Your Financial Summary",
                "─" * 42,
                f"  💰 Total Income    : ₹{total_income:>12,.2f}",
                f"  💸 Total Expenses  : ₹{total_expense:>12,.2f}",
                f"  🏦 Net Savings     : ₹{savings:>12,.2f}  ({rate:.1f}%)",
                "─" * 42,
                f"  📅 This Month Spent: ₹{month_expense:>12,.2f}",
            ]

            # Top categories section — only shown if there's data for this month
            if top_cats:
                lines.append("\n  🔝 Top Spending Categories This Month:")
                for cat, amt in top_cats:
                    lines.append(f"     • {str(cat).title():<16} ₹{float(amt):,.2f}")

            # ── Budget overview section ────────────────────────────────────────
            budget_rows = get_budgets(self.mysql, user_id)
            if budget_rows:
                lines.append("\n  🏦 Budget Overview:")
                for _, category, budget_amt, period, spent in budget_rows:
                    budget_amt = float(budget_amt)
                    spent      = float(spent)
                    pct        = min((spent / budget_amt * 100), 100) if budget_amt > 0 else 0
                    remaining  = max(budget_amt - spent, 0)

                    # Status icon mirrors what _view_budgets() uses
                    status = "🚨" if pct >= 100 else ("⚠️ " if pct >= 80 else "✅")

                    lines.append(
                        f"     {status} {category.title():<14} {pct:5.1f}% used  "
                        f"| ₹{remaining:,.2f} left [{period}]"
                    )

            # ── Budget alert warnings ──────────────────────────────────────────
            warnings = get_budget_warnings(self.mysql, user_id)
            if warnings:
                lines.append("\n  🔔 Budget Alerts:")
                for w in warnings:
                    lines.append(f"     {w}")

            # ── Goals progress section ─────────────────────────────────────────
            goal_rows = get_goals(self.mysql, user_id)
            if goal_rows:
                lines.append("\n  🎯 Goals Progress:")
                for _, goal_name, target_amount, current_amount, target_date, _ in goal_rows:
                    target    = float(target_amount)
                    current   = float(current_amount)
                    pct       = min((current / target * 100), 100) if target > 0 else 0
                    remaining = max(target - current, 0)

                    # Status icon and short note based on how far along they are
                    if pct >= 100:
                        status = "🏆"
                        note   = "COMPLETE!"
                    elif pct >= 75:
                        status = "🔥"
                        note   = "Almost there!"
                    elif pct >= 50:
                        status = "💪"
                        note   = "Halfway!"
                    else:
                        status = "🌱"
                        note   = f"₹{remaining:,.2f} to go"

                    lines.append(
                        f"     {status} {goal_name.title():<18} {pct:5.1f}%  — {note}"
                    )

            # ── Personalised savings rate advice at the bottom ─────────────────
            lines.append("\n" + "─" * 42)

            if rate >= 20:
                lines.append("  💡 Great job! You're saving over 20% of income! 🎉")
            elif rate >= 10:
                lines.append("  💡 Good progress! Try to push savings above 20%.")
            elif total_income > 0:
                lines.append("  ⚠️  Savings rate is low. Review your expenses!")
            else:
                # No income recorded yet — can't calculate a rate
                lines.append("  ℹ️  Add your income to see savings rate.")

            return "\n".join(lines)

        except Exception as e:
            log.error("[ANALYTICS] Error: %s", e)
            return "❌ Couldn't load analytics right now."


    # ==========================================================================
    # FINANCIAL ADVICE
    # ==========================================================================

    def _give_advice(self) -> str:
        """
        Returns a single randomly chosen financial tip from a curated list.
        Called when the user asks for generic financial advice.
        """
        tips = [
            "💡 Try the 50/30/20 rule: 50% on needs, 30% on wants, 20% into savings.",
            "💡 Set up a small automatic transfer to savings every payday — even ₹500/month adds up!",
            "💡 Track every expense for 30 days. You'll find leaks you didn't know existed.",
            "💡 Build an emergency fund of 3–6 months of expenses before investing.",
            "💡 Pay yourself first — move savings to a separate account before spending.",
            "💡 Avoid lifestyle inflation. When income rises, increase savings before spending.",
            "💡 Review your budgets every month and adjust as your life changes.",
            "💡 Break big goals into milestones — celebrate each 25% of a savings goal!",
        ]
        return random.choice(tips)


    # ==========================================================================
    # PRIVATE EXTRACTION HELPERS
    # ==========================================================================

    @staticmethod
    def _extract_budget_category(message: str) -> str | None:
        """
        Scans the message for any word that matches a known expense category
        (from EXPENSE_CATEGORIES). Returns the first match found, or None.

        This is a simple substring check rather than ML — it's fast and
        reliable enough for the small set of fixed budget categories.
        """
        msg_lower = message.lower()
        for cat in EXPENSE_CATEGORIES:
            if cat in msg_lower:
                return cat
        return None


    @staticmethod
    def _extract_period(message: str) -> str | None:
        """
        Detects a budget period keyword in the message and normalises it.

        Accepted inputs          → Returned value
        'weekly' or 'week'       → 'weekly'
        'monthly' or 'month'     → 'monthly'
        'yearly', 'year',
        or 'annual'              → 'yearly'

        Returns None if no period keyword is found.
        """
        msg_lower = message.lower()

        # Map all the ways a user might say each period to the canonical value
        period_map = {
            "weekly":  "weekly",  "week":    "weekly",
            "monthly": "monthly", "month":   "monthly",
            "yearly":  "yearly",  "year":    "yearly",  "annual": "yearly",
        }

        for keyword, period in period_map.items():
            if keyword in msg_lower:
                return period

        return None


    @staticmethod
    def _extract_goal_name(message: str) -> str | None:
        """
        Extracts a goal name from messages like:
          'save 100000 for bike'         → 'bike'
          'my goal is 5 lakh for car'    → 'car'
          'delete bike goal'             → 'bike'
          'update vacation goal'         → 'vacation'

        Uses three regex patterns in order of preference:
          1. 'for [a/my/the] <name>' — most common phrasing
          2. 'delete/update/about <name> goal' — edit/delete commands
          3. '<name> goal' — bare form e.g. 'show bike goal'

        Returns None if no goal name can be reliably extracted,
        so the caller can start a flow to ask the user directly.
        """
        msg_lower = message.lower()

        # Pattern 1: 'for [optional article] <goal name>'
        # Stops at end of string, 'goal', or a number
        match = re.search(
            r"\bfor\s+(?:a\s+|my\s+|the\s+)?([a-z][a-z\s]{1,25}?)(?:\s*$|\s+goal|\s+\d)",
            msg_lower
        )
        if match:
            return match.group(1).strip()

        # Pattern 2: action word followed by name and 'goal'
        match = re.search(
            r"\b(?:delete|remove|update|about)\s+([a-z][a-z\s]{1,25?})\s+goal",
            msg_lower
        )
        if match:
            return match.group(1).strip()

        # Pattern 3: '<name> goal' — simplest form
        match = re.search(r"\b([a-z][a-z\s]{1,20})\s+goal\b", msg_lower)
        if match:
            candidate = match.group(1).strip()
            # Filter out generic filler words that aren't real goal names
            if candidate not in {"my", "a", "the", "savings", "financial", "new"}:
                return candidate

        return None
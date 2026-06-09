# ==============================================================================
# chatbot/chatbot_engine.py  —  SmartExpenseAI Chatbot Brain
# ==============================================================================
# Receives plain-text messages, detects intent, extracts entities, and either
# saves data to the database or returns a helpful text reply.
#
# Capabilities:
#   → Record expenses and income (single-shot or step-by-step flow)
#   → Set, view, update, and delete spending budgets
#   → Create, view, update, and delete savings goals
#   → Financial analytics summary
#   → AI-powered advice and fallback via OpenRouter (injected at startup)
# ==============================================================================

import logging
import random
import re
from datetime import date

from chatbot.intent import detect_intent
from chatbot.entities import extract_entities
from chatbot.state_manage import StateManager

from machine_learning.predict_category import predict_category
from machine_learning.utils.dataset_sync import append_to_dataset


# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SmartExpenseAI")


# ==============================================================================
# Constants
# ==============================================================================

EXPENSE_CATEGORIES = {
    "food", "travel", "rent", "shopping", "bills",
    "entertainment", "medical", "education", "general",
}

INCOME_CATEGORIES = {
    "salary", "freelance", "business", "bonus", "gift", "general",
}

BUDGET_PERIODS = {"weekly", "monthly", "yearly"}

CANCEL_WORDS = {"cancel", "stop", "quit", "abort", "exit", "nevermind", "never mind"}

# Financial advice tips used when OpenRouter is unavailable
_ADVICE_TIPS = [
    "Try the 50/30/20 rule: 50% on needs, 30% on wants, 20% into savings.",
    "Set up a small automatic transfer to savings every payday — even ₹500/month adds up!",
    "Track every expense for 30 days. You'll find leaks you didn't know existed.",
    "Build an emergency fund of 3-6 months of expenses before investing.",
    "Pay yourself first — move savings to a separate account before spending.",
    "Avoid lifestyle inflation. When income rises, increase savings before spending.",
    "Review your budgets every month and adjust as your life changes.",
    "Break big goals into milestones — celebrate each 25% of a savings goal!",
    "Use the 24-hour rule before any unplanned purchase above ₹2000.",
    "SIPs (Systematic Investment Plans) in index funds are a great way to grow wealth passively.",
]


# ==============================================================================
# Database Helpers — Transactions
# ==============================================================================

def save_to_database(mysql, user_id, txn_type, amount, description, category, txn_date):
    """
    Inserts one income or expense record into the transactions table.
    Also feeds the example to the ML training dataset.
    Returns True on success, False on any DB error.
    """
    log.debug("[DB] user=%s type=%s amount=%s cat=%s desc='%s'",
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
        mysql.connection.commit()
        cur.close()
        log.info("[DB] Saved transaction successfully.")
        append_to_dataset(description, amount, txn_type, category)
        return True
    except Exception as e:
        log.error("[DB] Save failed: %s", e)
        return False


# ==============================================================================
# Database Helpers — Budgets
# ==============================================================================

def save_budget(mysql, user_id, category, amount, period):
    """
    Inserts a new budget or updates the amount if one already exists for
    the same user/category/period. Requires a UNIQUE KEY on those three columns.
    Returns True on success, False on error.
    """
    log.debug("[BUDGET-DB] user=%s cat=%s amount=%s period=%s", user_id, category, amount, period)
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
        log.info("[BUDGET-DB] Budget saved.")
        return True
    except Exception as e:
        log.error("[BUDGET-DB] %s", e)
        return False


def get_budgets(mysql, user_id):
    """
    Returns every budget for a user enriched with actual spending for
    the budget's active period.
    Each row: (id, category, amount, period, spent)
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
        return rows
    except Exception as e:
        log.error("[BUDGET-DB] get_budgets: %s", e)
        return []


def delete_budget(mysql, user_id, category):
    """
    Deletes the budget for a given category (case-insensitive match).
    Returns True if a row was deleted, False if nothing matched.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "DELETE FROM budgets WHERE user_id = %s AND LOWER(category) = LOWER(%s)",
            (user_id, category)
        )
        affected = cur.rowcount
        mysql.connection.commit()
        cur.close()
        return affected > 0
    except Exception as e:
        log.error("[BUDGET-DB] delete_budget: %s", e)
        return False


def get_budget_warnings(mysql, user_id):
    """
    Returns warning strings for any budget category that is 80%+ spent.
    """
    warnings = []
    for row in get_budgets(mysql, user_id):
        _, category, budget_amount, period, spent = row
        budget_amount = float(budget_amount)
        spent         = float(spent)
        if budget_amount <= 0:
            continue
        pct = (spent / budget_amount) * 100
        if pct >= 100:
            warnings.append(
                f"🚨 {category.title()} budget EXCEEDED! "
                f"Spent ₹{spent:,.2f} of ₹{budget_amount:,.2f} ({pct:.0f}%) [{period}]"
            )
        elif pct >= 80:
            warnings.append(
                f"⚠️  {category.title()} budget at {pct:.0f}% — "
                f"₹{budget_amount - spent:,.2f} remaining [{period}]"
            )
    return warnings


# ==============================================================================
# Database Helpers — Goals
# ==============================================================================

def save_goal(mysql, user_id, goal_name, target_amount, current_amount=0.0, target_date=None):
    """Inserts a new savings goal. Returns True on success."""
    log.debug("[GOAL-DB] user=%s name='%s' target=%s", user_id, goal_name, target_amount)
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
        log.info("[GOAL-DB] Goal saved.")
        return True
    except Exception as e:
        log.error("[GOAL-DB] %s", e)
        return False


def update_goal_amount(mysql, user_id, goal_name, new_target):
    """
    Updates the target_amount of an existing goal (case-insensitive name match).
    Returns True if a row was updated.
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
        return affected > 0
    except Exception as e:
        log.error("[GOAL-DB] update_goal_amount: %s", e)
        return False


def contribute_to_goal(mysql, user_id, goal_name, contribution):
    """Adds an amount to current_amount of a goal. Returns True if updated."""
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
        log.error("[GOAL-DB] contribute: %s", e)
        return False


def get_goals(mysql, user_id):
    """
    Returns all goals for a user, newest first.
    Each row: (id, goal_name, target_amount, current_amount, target_date, created_at)
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
        log.error("[GOAL-DB] get_goals: %s", e)
        return []


def delete_goal(mysql, user_id, goal_name):
    """Deletes a goal by name (case-insensitive). Returns True if deleted."""
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
        log.error("[GOAL-DB] delete_goal: %s", e)
        return False


def _goal_progress_line(goal_name, target_amount, current_amount):
    """Builds a compact ASCII progress bar for one goal."""
    target    = float(target_amount)
    current   = float(current_amount)
    pct       = min((current / target * 100), 100) if target > 0 else 0
    bar_len   = 20
    filled    = int(bar_len * pct / 100)
    bar       = "█" * filled + "░" * (bar_len - filled)
    remaining = max(target - current, 0)
    return (
        f"  🎯 {goal_name.title():<18} [{bar}] {pct:5.1f}%\n"
        f"     ₹{current:>10,.2f} saved  |  ₹{remaining:>10,.2f} to go"
    )


# ==============================================================================
# Chatbot Engine
# ==============================================================================

class ChatbotEngine:
    """
    Routes user messages to the correct handler and manages multi-step flows.

    openrouter_fn: callable(message: str, intent: str) -> str | None
        Injected by app.py after startup. Used for AI-powered fallback replies
        and financial advice. If None or returns None, built-in replies are used.
    """

    def __init__(self, mysql, openrouter_fn=None):
        self.mysql        = mysql
        self.state        = StateManager()
        self.openrouter_fn = openrouter_fn   # Injected: call_openrouter from app.py


    # ==========================================================================
    # Internal helper — safe OpenRouter call
    # ==========================================================================

    def _ai_reply(self, message: str, intent: str = "unknown") -> str | None:
        """
        Calls the injected OpenRouter function if available.
        Returns the AI reply string, or None if unavailable/failed.
        This lets every handler try AI first and fall back gracefully.
        """
        if self.openrouter_fn is None:
            return None
        try:
            return self.openrouter_fn(message, intent)
        except Exception as e:
            log.error("[AI_REPLY] OpenRouter call failed: %s", e)
            return None


    # ==========================================================================
    # Main Entry Point
    # ==========================================================================

    def process_message(self, user_id, message: str, username: str = "User") -> str:
        """
        The single public method called by the Flask /chatbot route.

        Flow:
          1. If the user is mid-flow → continue it.
          2. Cancel word → abandon flow.
          3. Detect intent → dispatch to the correct handler.
          4. Unknown intent → try OpenRouter AI → fallback string.
        """
        msg = message.strip()
        if not msg:
            return "Please type a message."

        log.debug("=" * 55)
        log.debug("[MSG] user=%s  message='%s'", user_id, msg)

        # Check for an active multi-step flow first
        current_flow = self.state.get_flow(user_id)
        if current_flow:
            if msg.lower() in CANCEL_WORDS:
                self.state.reset(user_id)
                return "Cancelled. What else can I help you with?"
            return self._continue_flow(user_id, msg, current_flow)

        # Detect intent
        intent = detect_intent(msg)
        log.debug("[INTENT] → %s", intent)

        # Dispatch
        if intent == "expense":
            return self._handle_expense(user_id, msg)

        elif intent == "income":
            return self._handle_income(user_id, msg)

        elif intent == "cancel":
            return "Nothing to cancel. Just tell me what you'd like to do!"

        elif intent == "view_expenses":
            return self._view_transactions(user_id, "expense")

        elif intent == "view_income":
            return self._view_transactions(user_id, "income")

        elif intent == "analytics":
            return self._show_analytics(user_id)

        elif intent == "financial_advice":
            return self._give_advice(msg)

        elif intent in ("budget", "add_budget"):
            return self._handle_budget(user_id, msg)

        elif intent == "view_budgets":
            return self._view_budgets(user_id)

        elif intent == "update_budget":
            return self._handle_update_budget(user_id, msg)

        elif intent == "delete_budget":
            return self._handle_delete_budget(user_id, msg)

        elif intent in ("goal", "add_goal"):
            return self._handle_goal(user_id, msg)

        elif intent == "view_goals":
            return self._view_goals(user_id)

        elif intent == "update_goal":
            return self._handle_update_goal(user_id, msg)

        elif intent == "delete_goal":
            return self._handle_delete_goal(user_id, msg)

        elif intent == "chat":
            return (
                f"👋 Hi {username}! I'm SmartExpenseAI.\n\n"
                "Here's what I can do:\n"
                "  💸 Track expenses     → 'I spent 300 on pizza'\n"
                "  💰 Record income      → 'Got salary 50000'\n"
                "  📊 Analytics          → 'show analytics'\n"
                "  📋 List expenses      → 'show expenses'\n"
                "  🏦 Budgets            → 'set food budget 5000 monthly'\n"
                "  🎯 Goals              → 'save 100000 for bike'\n"
                "  💡 Financial advice   → 'give me a tip'\n\n"
                "What would you like to do?"
            )

        else:
            # Unknown intent — check for an amount (likely an expense)
            entities = extract_entities(msg)
            if entities["amount"]:
                return self._handle_expense(user_id, msg)

            # Try OpenRouter for a conversational/AI reply
            ai_reply = self._ai_reply(msg, intent="unknown")
            if ai_reply:
                return ai_reply

            # Final fallback
            return (
                "I didn't quite understand that.\n\n"
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
    # Expense Handler
    # ==========================================================================

    def _handle_expense(self, user_id, message: str) -> str:
        log.debug("[EXPENSE] '%s'", message)

        entities    = extract_entities(message)
        amount      = entities["amount"]
        description = entities["description"]
        category    = entities["category"]
        txn_date    = entities["date"] or str(date.today())

        if not amount or amount <= 0:
            self.state.start_flow(user_id, "expense")
            return "💸 How much did you spend? (e.g. 300, 1500, 25k)"

        if not description or len(description) < 2:
            self.state.start_flow(user_id, "expense")
            self.state.save_data(user_id, "amount", amount)
            self.state.save_data(user_id, "date", txn_date)
            return f"🏷️ What did you spend ₹{amount:,.0f} on?"

        if not category:
            category = predict_category(description, "expense")
        if not category or category == "unknown":
            category = "general"

        saved   = save_to_database(self.mysql, user_id, "expense", amount,
                                   description, category, txn_date)
        warning = self._budget_warning_for_category(user_id, category)

        if saved:
            reply = (
                f"✅ Expense recorded!\n"
                f"   Amount      : ₹{amount:,.2f}\n"
                f"   Description : {description.title()}\n"
                f"   Category    : {category.title()}\n"
                f"   Date        : {txn_date}"
            )
            if warning:
                reply += f"\n\n{warning}"
            return reply

        return "❌ Sorry, I couldn't save your expense. Please try again."


    # ==========================================================================
    # Income Handler
    # ==========================================================================

    def _handle_income(self, user_id, message: str) -> str:
        log.debug("[INCOME] '%s'", message)

        entities    = extract_entities(message)
        amount      = entities["amount"]
        description = entities["description"]
        category    = entities["category"]
        txn_date    = entities["date"] or str(date.today())

        if not amount or amount <= 0:
            self.state.start_flow(user_id, "income")
            return "💰 How much income did you receive? (e.g. 50000, 25k)"

        if not description or len(description) < 2:
            self.state.start_flow(user_id, "income")
            self.state.save_data(user_id, "amount", amount)
            self.state.save_data(user_id, "date", txn_date)
            return f"💼 What was this ₹{amount:,.0f} income for? (e.g. salary, freelance)"

        if not category:
            category = predict_category(description, "income")
        if not category or category == "unknown":
            category = "general"

        saved = save_to_database(self.mysql, user_id, "income", amount,
                                 description, category, txn_date)

        if saved:
            return (
                f"✅ Income recorded!\n"
                f"   Amount      : ₹{amount:,.2f}\n"
                f"   Description : {description.title()}\n"
                f"   Category    : {category.title()}\n"
                f"   Date        : {txn_date}"
            )

        return "❌ Sorry, I couldn't save your income. Please try again."


    # ==========================================================================
    # Budget Handlers
    # ==========================================================================

    def _handle_budget(self, user_id, message: str) -> str:
        log.debug("[BUDGET] '%s'", message)

        entities = extract_entities(message)
        category = self._extract_budget_category(message) or entities.get("category")
        amount   = entities.get("amount")
        period   = self._extract_period(message)

        if not category:
            self.state.start_flow(user_id, "budget")
            return (
                "🏦 Which category would you like to set a budget for?\n"
                "(e.g. food, travel, shopping, rent, bills, entertainment)"
            )

        if not amount or amount <= 0:
            self.state.start_flow(user_id, "budget")
            self.state.save_data(user_id, "category", category)
            return f"💵 How much should the {category.title()} budget be?"

        if not period:
            self.state.start_flow(user_id, "budget")
            self.state.save_data(user_id, "category", category)
            self.state.save_data(user_id, "amount", amount)
            return "🗓️ Should this be weekly, monthly, or yearly?"

        return self._save_budget_and_reply(user_id, category, amount, period)


    def _handle_update_budget(self, user_id, message: str) -> str:
        log.debug("[BUDGET-UPDATE] '%s'", message)

        entities = extract_entities(message)
        category = self._extract_budget_category(message)
        amount   = entities.get("amount")
        period   = self._extract_period(message) or "monthly"

        if not category:
            self.state.start_flow(user_id, "update_budget")
            return "✏️ Which budget category would you like to update?"

        if not amount or amount <= 0:
            self.state.start_flow(user_id, "update_budget")
            self.state.save_data(user_id, "category", category)
            return f"💵 What should the new amount be for your {category.title()} budget?"

        return self._save_budget_and_reply(user_id, category, amount, period, is_update=True)


    def _handle_delete_budget(self, user_id, message: str) -> str:
        log.debug("[BUDGET-DELETE] '%s'", message)

        category = self._extract_budget_category(message)

        if not category:
            self.state.start_flow(user_id, "delete_budget")
            return "🗑️ Which budget category would you like to delete?"

        deleted = delete_budget(self.mysql, user_id, category)
        if deleted:
            return f"✅ {category.title()} budget deleted successfully."

        return f"⚠️ No budget found for '{category}'. Check 'show budgets' for your list."


    def _view_budgets(self, user_id) -> str:
        rows = get_budgets(self.mysql, user_id)

        if not rows:
            return "🏦 No budgets set yet.\nTry: 'set food budget 5000 monthly'"

        lines = ["🏦 Your Budgets\n" + "─" * 52]

        for _, category, budget_amt, period, spent in rows:
            budget_amt = float(budget_amt)
            spent      = float(spent)
            remaining  = max(budget_amt - spent, 0)
            pct        = min((spent / budget_amt * 100), 100) if budget_amt > 0 else 0
            bar_len    = 16
            filled     = int(bar_len * pct / 100)
            bar        = "█" * filled + "░" * (bar_len - filled)
            status     = "🚨" if pct >= 100 else ("⚠️ " if pct >= 80 else "✅")

            lines.append(
                f"{status} {category.title():<14} [{bar}] {pct:5.1f}%  [{period}]\n"
                f"     Spent: ₹{spent:>9,.2f}  |  Budget: ₹{budget_amt:>9,.2f}"
                f"  |  Left: ₹{remaining:>9,.2f}"
            )

        return "\n".join(lines)


    def _save_budget_and_reply(self, user_id, category, amount, period,
                               is_update=False) -> str:
        saved = save_budget(self.mysql, user_id, category, amount, period)
        if saved:
            verb = "updated" if is_update else "set"
            return (
                f"✅ Budget {verb}!\n"
                f"   Category : {category.title()}\n"
                f"   Amount   : ₹{amount:,.2f}\n"
                f"   Period   : {period.title()}\n\n"
                f"I'll warn you when you reach 80% of this budget."
            )
        return "❌ Couldn't save budget. Please try again."


    def _budget_warning_for_category(self, user_id, category: str) -> str:
        """Returns a budget warning string if category is 80%+ spent, else ''."""
        for _, cat, budget_amt, period, spent in get_budgets(self.mysql, user_id):
            if cat.lower() == category.lower():
                budget_amt = float(budget_amt)
                spent      = float(spent)
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
        return ""


    # ==========================================================================
    # Goal Handlers
    # ==========================================================================

    def _handle_goal(self, user_id, message: str) -> str:
        log.debug("[GOAL] '%s'", message)

        entities      = extract_entities(message)
        goal_name     = self._extract_goal_name(message)
        target_amount = entities.get("amount")
        target_date   = entities.get("date")

        if not goal_name:
            self.state.start_flow(user_id, "goal")
            return "🎯 What would you like to save for? (e.g. bike, car, vacation, emergency fund)"

        if not target_amount or target_amount <= 0:
            self.state.start_flow(user_id, "goal")
            self.state.save_data(user_id, "goal_name", goal_name)
            return f"💰 How much do you want to save for your {goal_name.title()} goal?"

        return self._save_goal_and_reply(user_id, goal_name, target_amount, 0.0, target_date)


    def _handle_update_goal(self, user_id, message: str) -> str:
        log.debug("[GOAL-UPDATE] '%s'", message)

        entities      = extract_entities(message)
        goal_name     = self._extract_goal_name(message)
        target_amount = entities.get("amount")

        if not goal_name:
            self.state.start_flow(user_id, "update_goal")
            return "✏️ Which goal would you like to update?"

        if not target_amount or target_amount <= 0:
            self.state.start_flow(user_id, "update_goal")
            self.state.save_data(user_id, "goal_name", goal_name)
            return f"💰 What should the new target amount be for your {goal_name.title()} goal?"

        updated = update_goal_amount(self.mysql, user_id, goal_name, target_amount)
        if updated:
            return (
                f"✅ Goal updated!\n"
                f"   Goal       : {goal_name.title()}\n"
                f"   New Target : ₹{target_amount:,.2f}"
            )
        return f"⚠️ No goal found named '{goal_name}'. Check 'show goals'."


    def _handle_delete_goal(self, user_id, message: str) -> str:
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
        rows = get_goals(self.mysql, user_id)

        if not rows:
            return "🎯 No goals set yet.\nTry: 'save 100000 for bike'"

        active_lines    = []
        completed_lines = []

        for _, goal_name, target_amount, current_amount, target_date, _ in rows:
            target  = float(target_amount)
            current = float(current_amount)
            pct     = min((current / target * 100), 100) if target > 0 else 0
            line    = _goal_progress_line(goal_name, target, current)

            if target_date:
                line += f"\n     Target Date : {target_date}"

            if pct >= 100:
                completed_lines.append(f"🏆 {goal_name.title()} — COMPLETED! 🎉\n" + line)
            else:
                if pct >= 75:
                    note = "  🔥 Almost there!"
                elif pct >= 50:
                    note = "  💪 Halfway done — keep it up!"
                elif pct >= 25:
                    note = "  🌱 Good progress!"
                else:
                    note = "  🚀 Just getting started!"
                active_lines.append(line + "\n" + note)

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
        saved = save_goal(self.mysql, user_id, goal_name,
                          target_amount, current_amount, target_date)
        if saved:
            reply = (
                f"✅ Goal created!\n"
                f"   Goal   : {goal_name.title()}\n"
                f"   Target : ₹{target_amount:,.2f}\n"
            )
            if target_date:
                reply += f"   By     : {target_date}\n"
            reply += (
                "\n💡 Tip: Use 'show analytics' to track your savings progress "
                "towards this goal!"
            )
            return reply
        return "❌ Couldn't save goal. Please try again."


    # ==========================================================================
    # Multi-Step Flow Continuation
    # ==========================================================================

    def _continue_flow(self, user_id, message: str, flow: str) -> str:
        """
        Continues a guided multi-step conversation.
        Each flow collects missing fields one at a time, then completes the action.
        """
        data     = self.state.get_data(user_id)
        entities = extract_entities(message)

        log.debug("[FLOW] flow=%s data=%s message='%s'", flow, data, message)

        # ── Expense ────────────────────────────────────────────────────────────
        if flow == "expense":
            if "amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount (e.g. 500 or 2.5k)."
                self.state.save_data(user_id, "amount", amount)
                return "🏷️ What did you spend it on? (e.g. pizza, uber, gym)"

            if "description" not in data:
                # Clean the description — strip common filler phrases
                desc = self._clean_description(message)
                if not desc or len(desc) < 2:
                    return "⚠️ Please describe what you spent on."
                self.state.save_data(user_id, "description", desc)

                amount   = data["amount"]
                txn_date = data.get("date", str(date.today()))
                category = entities["category"] or predict_category(desc, "expense")
                if not category or category == "unknown":
                    category = "general"

                saved = save_to_database(self.mysql, user_id, "expense",
                                         amount, desc, category, txn_date)
                self.state.reset(user_id)
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

        # ── Income ─────────────────────────────────────────────────────────────
        elif flow == "income":
            if "amount" not in data:
                amount = entities["amount"]
                if not amount or amount <= 0:
                    return "⚠️ Please enter a valid amount (e.g. 50000 or 25k)."
                self.state.save_data(user_id, "amount", amount)
                return "💼 What was this income for? (e.g. salary, freelance work)"

            if "description" not in data:
                desc = self._clean_description(message)
                if not desc or len(desc) < 2:
                    return "⚠️ Please describe the income source."
                self.state.save_data(user_id, "description", desc)

                amount   = data["amount"]
                txn_date = data.get("date", str(date.today()))
                category = entities["category"] or predict_category(desc, "income")
                if not category or category == "unknown":
                    category = "general"

                saved = save_to_database(self.mysql, user_id, "income",
                                         amount, desc, category, txn_date)
                self.state.reset(user_id)

                if saved:
                    return (
                        f"✅ Income recorded!\n"
                        f"   Amount      : ₹{amount:,.2f}\n"
                        f"   Description : {desc.title()}\n"
                        f"   Category    : {category.title()}\n"
                        f"   Date        : {txn_date}"
                    )
                return "❌ Couldn't save income. Please try again."

        # ── Budget ─────────────────────────────────────────────────────────────
        elif flow == "budget":
            if "category" not in data:
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

            # Period step
            period = self._extract_period(message)
            if not period:
                return "⚠️ Please say weekly, monthly, or yearly."
            category = data["category"]
            amount   = data["amount"]
            self.state.reset(user_id)
            return self._save_budget_and_reply(user_id, category, amount, period)

        # ── Update budget ──────────────────────────────────────────────────────
        elif flow == "update_budget":
            if "category" not in data:
                cat = self._extract_budget_category(message) or message.strip().lower()
                if not cat or len(cat) < 2:
                    return "⚠️ Please enter a valid category."
                self.state.save_data(user_id, "category", cat)
                return f"💵 What should the new amount be for your {cat.title()} budget?"

            amount = entities["amount"]
            if not amount or amount <= 0:
                return "⚠️ Please enter a valid amount."
            category = data["category"]
            self.state.reset(user_id)
            return self._save_budget_and_reply(user_id, category, amount, "monthly",
                                               is_update=True)

        # ── Delete budget ──────────────────────────────────────────────────────
        elif flow == "delete_budget":
            cat = self._extract_budget_category(message) or message.strip().lower()
            if not cat or len(cat) < 2:
                return "⚠️ Please enter a valid category name."
            self.state.reset(user_id)
            deleted = delete_budget(self.mysql, user_id, cat)
            if deleted:
                return f"✅ {cat.title()} budget deleted."
            return f"⚠️ No budget found for '{cat}'."

        # ── Goal ───────────────────────────────────────────────────────────────
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
                return (
                    f"📅 Do you have a target date? (e.g. '2025-12-31')\n"
                    f"   Or type 'skip' to skip."
                )

            # Date step (optional)
            raw = message.strip().lower()
            tdate = None if raw in ("skip", "no") else (
                entities.get("date") or (message.strip() if len(message.strip()) >= 8 else None)
            )

            goal_name     = data["goal_name"]
            target_amount = data["target_amount"]
            self.state.reset(user_id)
            return self._save_goal_and_reply(user_id, goal_name, target_amount, 0.0, tdate)

        # ── Update goal ────────────────────────────────────────────────────────
        elif flow == "update_goal":
            if "goal_name" not in data:
                goal_name = self._extract_goal_name(message) or message.strip().lower()
                if not goal_name or len(goal_name) < 2:
                    return "⚠️ Please enter a valid goal name."
                self.state.save_data(user_id, "goal_name", goal_name)
                return f"💰 What should the new target be for your {goal_name.title()} goal?"

            amount = entities["amount"]
            if not amount or amount <= 0:
                return "⚠️ Please enter a valid amount."
            goal_name = data["goal_name"]
            self.state.reset(user_id)
            updated = update_goal_amount(self.mysql, user_id, goal_name, amount)
            if updated:
                return f"✅ {goal_name.title()} goal updated to ₹{amount:,.2f}."
            return f"⚠️ No goal named '{goal_name}' found."

        # ── Delete goal ────────────────────────────────────────────────────────
        elif flow == "delete_goal":
            goal_name = self._extract_goal_name(message) or message.strip().lower()
            if not goal_name or len(goal_name) < 2:
                return "⚠️ Please enter a valid goal name."
            self.state.reset(user_id)
            deleted = delete_goal(self.mysql, user_id, goal_name)
            if deleted:
                return f"✅ {goal_name.title()} goal deleted."
            return f"⚠️ No goal named '{goal_name}' found."

        self.state.reset(user_id)
        return "⚠️ Something went wrong. Please start over."


    # ==========================================================================
    # View Transactions
    # ==========================================================================

    def _view_transactions(self, user_id, txn_type: str) -> str:
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
                lines.append(
                    f"  ₹{amt:>9,.2f}  |  {str(category).title():<14}  |  "
                    f"{str(description)[:20]:<20}  |  {txn_date}"
                )

            lines.append(f"{'─' * 46}")
            lines.append(f"  Total: ₹{total:,.2f}")
            return "\n".join(lines)

        except Exception as e:
            log.error("[VIEW] %s", e)
            return f"❌ Couldn't load {txn_type}s right now."


    # ==========================================================================
    # Analytics Summary
    # ==========================================================================

    def _show_analytics(self, user_id) -> str:
        try:
            cur = self.mysql.connection.cursor()

            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions "
                "WHERE user_id=%s AND type='income'", (user_id,)
            )
            total_income = float(cur.fetchone()[0])

            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions "
                "WHERE user_id=%s AND type='expense'", (user_id,)
            )
            total_expense = float(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM transactions
                WHERE user_id=%s AND type='expense'
                  AND YEAR(transaction_date)  = YEAR(CURDATE())
                  AND MONTH(transaction_date) = MONTH(CURDATE())
                """, (user_id,)
            )
            month_expense = float(cur.fetchone()[0])

            cur.execute(
                """
                SELECT category, SUM(amount) AS total
                FROM transactions
                WHERE user_id=%s AND type='expense'
                  AND YEAR(transaction_date)  = YEAR(CURDATE())
                  AND MONTH(transaction_date) = MONTH(CURDATE())
                GROUP BY category ORDER BY total DESC LIMIT 3
                """, (user_id,)
            )
            top_cats = cur.fetchall()
            cur.close()

            savings = total_income - total_expense
            rate    = (savings / total_income * 100) if total_income > 0 else 0

            lines = [
                "📊 Your Financial Summary",
                "─" * 42,
                f"  💰 Total Income    : ₹{total_income:>12,.2f}",
                f"  💸 Total Expenses  : ₹{total_expense:>12,.2f}",
                f"  🏦 Net Savings     : ₹{savings:>12,.2f}  ({rate:.1f}%)",
                "─" * 42,
                f"  📅 This Month Spent: ₹{month_expense:>12,.2f}",
            ]

            if top_cats:
                lines.append("\n  🔝 Top Spending Categories This Month:")
                for cat, amt in top_cats:
                    lines.append(f"     • {str(cat).title():<16} ₹{float(amt):,.2f}")

            budget_rows = get_budgets(self.mysql, user_id)
            if budget_rows:
                lines.append("\n  🏦 Budget Overview:")
                for _, category, budget_amt, period, spent in budget_rows:
                    budget_amt = float(budget_amt)
                    spent      = float(spent)
                    pct        = min((spent / budget_amt * 100), 100) if budget_amt > 0 else 0
                    remaining  = max(budget_amt - spent, 0)
                    status     = "🚨" if pct >= 100 else ("⚠️ " if pct >= 80 else "✅")
                    lines.append(
                        f"     {status} {category.title():<14} {pct:5.1f}% used  "
                        f"| ₹{remaining:,.2f} left [{period}]"
                    )

            warnings = get_budget_warnings(self.mysql, user_id)
            if warnings:
                lines.append("\n  🔔 Budget Alerts:")
                for w in warnings:
                    lines.append(f"     {w}")

            goal_rows = get_goals(self.mysql, user_id)
            if goal_rows:
                lines.append("\n  🎯 Goals Progress:")
                for _, goal_name, target_amount, current_amount, target_date, _ in goal_rows:
                    target    = float(target_amount)
                    current   = float(current_amount)
                    pct       = min((current / target * 100), 100) if target > 0 else 0
                    remaining = max(target - current, 0)
                    if pct >= 100:
                        status, note = "🏆", "COMPLETE!"
                    elif pct >= 75:
                        status, note = "🔥", "Almost there!"
                    elif pct >= 50:
                        status, note = "💪", "Halfway!"
                    else:
                        status, note = "🌱", f"₹{remaining:,.2f} to go"
                    lines.append(
                        f"     {status} {goal_name.title():<18} {pct:5.1f}%  — {note}"
                    )

            lines.append("\n" + "─" * 42)
            if rate >= 20:
                lines.append("  💡 Great job! You're saving over 20% of income! 🎉")
            elif rate >= 10:
                lines.append("  💡 Good progress! Try to push savings above 20%.")
            elif total_income > 0:
                lines.append("  ⚠️  Savings rate is low. Review your expenses!")
            else:
                lines.append("  ℹ️  Add your income to see savings rate.")

            return "\n".join(lines)

        except Exception as e:
            log.error("[ANALYTICS] %s", e)
            return "❌ Couldn't load analytics right now."


    # ==========================================================================
    # Financial Advice — OpenRouter first, tips fallback
    # ==========================================================================

    def _give_advice(self, original_message: str = "") -> str:
        """
        Tries OpenRouter for a personalised tip first.
        Falls back to a curated static tip if AI is unavailable.
        """
        # Build a focused prompt for the AI
        prompt = (
            "Give me one practical, specific personal finance tip for someone "
            "living in India tracking their expenses. Keep it to 2 sentences, "
            "friendly tone, include a relevant emoji at the start."
        )

        ai_reply = self._ai_reply(prompt, intent="financial_advice")
        if ai_reply:
            return ai_reply

        # Static fallback
        return f"💡 {random.choice(_ADVICE_TIPS)}"


    # ==========================================================================
    # Private Extraction Helpers
    # ==========================================================================

    @staticmethod
    def _clean_description(message: str) -> str:
        """
        Strips common chatbot filler phrases from a description reply so we
        don't store 'it was for salary' instead of 'salary'.

        Examples:
          'it was for pizza delivery'  → 'pizza delivery'
          'spent on groceries'         → 'groceries'
          'for my monthly salary'      → 'monthly salary'
          'salary'                     → 'salary'
        """
        msg = message.strip().lower()
        # Remove leading filler phrases
        filler_patterns = [
            r"^it was (for |on )?",
            r"^(spent |paid )?(on |for )?",
            r"^(my |the )?",
        ]
        for pattern in filler_patterns:
            msg = re.sub(pattern, "", msg).strip()
        return msg


    @staticmethod
    def _extract_budget_category(message: str) -> str | None:
        """
        Returns the first EXPENSE_CATEGORIES keyword found in the message,
        or None if none match.
        """
        msg_lower = message.lower()
        for cat in EXPENSE_CATEGORIES:
            if cat in msg_lower:
                return cat
        return None


    @staticmethod
    def _extract_period(message: str) -> str | None:
        """
        Detects a budget period in the message.
        Returns 'weekly', 'monthly', or 'yearly', or None.
        """
        msg_lower = message.lower()
        period_map = {
            "weekly":  "weekly",  "week":   "weekly",
            "monthly": "monthly", "month":  "monthly",
            "yearly":  "yearly",  "year":   "yearly",  "annual": "yearly",
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
          'delete bike goal'             → 'bike'
          'update vacation goal'         → 'vacation'

        FIX: Pattern 1 now stops at prepositions like 'by', 'on', 'in'
        to prevent capturing 'bike by december' as a goal name.
        """
        msg_lower = message.lower()

        # Pattern 1: 'for [article] <name>' — stops at end, 'goal', digits, or prepositions
        match = re.search(
            r"\bfor\s+(?:a\s+|my\s+|the\s+)?([a-z][a-z]{1,20})"
            r"(?:\s*$|\s+goal|\s+\d|\s+(?:by|on|in|at|from|until)\b)",
            msg_lower
        )
        if match:
            return match.group(1).strip()

        # Pattern 2: action word + name + 'goal'
        match = re.search(
            r"\b(?:delete|remove|update|about)\s+([a-z][a-z\s]{1,25?})\s+goal",
            msg_lower
        )
        if match:
            return match.group(1).strip()

        # Pattern 3: '<name> goal'
        match = re.search(r"\b([a-z][a-z\s]{1,20})\s+goal\b", msg_lower)
        if match:
            candidate = match.group(1).strip()
            if candidate not in {"my", "a", "the", "savings", "financial", "new"}:
                return candidate

        return None
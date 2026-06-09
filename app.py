
# SmartExpenseAI — Main Flask Application

# --- Standard library imports ---
from datetime import date, datetime, timedelta
import os

# --- Third-party imports ---
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import requests

# --- Our own modules ---
#chatbot
from chatbot.chatbot_engine import ChatbotEngine             # Handles all chatbot logic
#mechine learning
from machine_learning.predict_category import predict_category   # ML-based category predictor
from machine_learning.utils.dataset_sync import append_to_dataset  # Keeps ML training data fresh
from machine_learning.forecasting import train_model         # Trains the forecasting model
from machine_learning.forecasting import (                   # Individual forecast functions
    next_day, weekly, monthly, yearly, get_trend
)

# Load values from the .env file (API keys, secrets, etc.)
load_dotenv()

# Create the Flask app instance
app = Flask(__name__)

# Secret key used to sign session cookies — keep this private in production
app.secret_key = os.getenv("SECRET_KEY")


# ============================================================
# Database Configuration
# ============================================================

# for local development
# app.config['MYSQL_HOST']     = 'localhost'       # MySQL is running on the same machine
# app.config['MYSQL_USER']     = 'root'            # Database username
# app.config['MYSQL_PASSWORD'] = 'Sree@2611'       # Database password
# app.config['MYSQL_DB']       = 'expense_tracker' # The database we're working with
# app.config['MYSQL_CHARSET']  = 'utf8mb4'         # Full Unicode support (handles emojis too)

#for production
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_PORT'] = MYSQL_PORT
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_CHARSET'] = 'utf8mb4'
# Bind the MySQL extension to our Flask app
mysql = MySQL(app)

# Create one shared chatbot engine and pass the DB connection to it
chatbot_engine = ChatbotEngine(mysql)


@app.route('/test-db')
def test_db():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT 1")
        return "DB CONNECTED"
    except Exception as e:
        return str(e)
# ============================================================
# Helper Functions
# ============================================================

def safe_date(value):
    """
    Safely converts any date-like value into a YYYY-MM-DD string.
    Handles date objects, ISO strings, and missing/invalid values.
    """

    # No value provided — just use today
    if not value:
        return date.today().isoformat()

    # Already a proper date object — convert directly
    if isinstance(value, date):
        return value.isoformat()

    # It's a string — try to parse it as ISO format
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            # Couldn't parse it, fall back to today
            return date.today().isoformat()

    # Some other type we didn't expect — default to today
    return date.today().isoformat()

# ============================================================
# Analytics Intelligence Helpers
# ============================================================

def generate_ai_insights(total_income, total_expense, savings,
                         expense_cats, budget_report):
    """
    Generates a list of plain-English insight strings based on
    the user's financial data. Each insight is a dict with:
      - text   : the message to display
      - type   : 'positive' | 'warning' | 'danger' | 'info'
      - icon   : emoji icon
    """
    insights = []
    spending_ratio = (total_expense / total_income * 100) if total_income else 0

    # --- Income vs Expense ratio ---
    if spending_ratio >= 90:
        insights.append({
            "text": f"Your expenses are {spending_ratio:.0f}% of your income — critically high.",
            "type": "danger", "icon": "🚨"
        })
    elif spending_ratio >= 70:
        insights.append({
            "text": f"You are spending {spending_ratio:.0f}% of your income. Try to cut back.",
            "type": "warning", "icon": "⚠️"
        })
    else:
        insights.append({
            "text": f"You are spending {spending_ratio:.0f}% of your income — healthy range.",
            "type": "positive", "icon": "✅"
        })

    # --- Savings insight ---
    if savings > 0:
        insights.append({
            "text": f"You have saved ₹{savings:,.2f} in total. Keep it up!",
            "type": "positive", "icon": "💰"
        })
    elif savings < 0:
        insights.append({
            "text": f"Your expenses exceed income by ₹{abs(savings):,.2f}. Urgent action needed.",
            "type": "danger", "icon": "📉"
        })

    # --- Top spending category ---
    if expense_cats:
        top_cat, top_amt = expense_cats[0]
        insights.append({
            "text": f"{top_cat.title()} is your highest spending category at ₹{float(top_amt):,.2f}.",
            "type": "info", "icon": "📊"
        })

    # --- Budget overspending ---
    for b in budget_report:
        if b["pct_used"] > 100:
            over = b["spent"] - b["budget"]
            insights.append({
                "text": f"{b['category'].title()} budget exceeded by ₹{over:,.2f}.",
                "type": "danger", "icon": "🔴"
            })
        elif b["pct_used"] >= 85:
            insights.append({
                "text": f"{b['category'].title()} budget is {b['pct_used']:.0f}% used — almost full.",
                "type": "warning", "icon": "🟡"
            })

    # --- No data fallback ---
    if not insights:
        insights.append({
            "text": "Add more transactions to unlock AI insights.",
            "type": "info", "icon": "💡"
        })

    return insights


def calculate_budget_alerts(budget_report):
    """
    Enriches budget_report dicts with an alert_level key:
      'safe'    → < 70%
      'warning' → 70–89%
      'danger'  → 90%+
    Returns the enriched list.
    """
    for b in budget_report:
        pct = b["pct_used"]
        if pct >= 90:
            b["alert_level"] = "danger"
        elif pct >= 70:
            b["alert_level"] = "warning"
        else:
            b["alert_level"] = "safe"
    return budget_report


def calculate_health_score(total_income, total_expense,
                            savings, budget_report):
    """
    Returns a financial health score (0–100) and a status label.

    Scoring breakdown (total = 100):
      - Expense ratio   : 40 pts  (lower spending = more points)
      - Savings amount  : 30 pts  (positive savings = full points)
      - Budget control  : 30 pts  (no overspent budgets = full points)
    """
    score = 0

    # --- 40 pts: expense-to-income ratio ---
    if total_income > 0:
        ratio = total_expense / total_income
        if ratio <= 0.5:
            score += 40
        elif ratio <= 0.7:
            score += 30
        elif ratio <= 0.85:
            score += 20
        elif ratio <= 1.0:
            score += 10

    # --- 30 pts: savings ---
    if savings > 0:
        score += 30
    elif savings == 0:
        score += 10

    # --- 30 pts: budget control ---
    if budget_report:
        safe_count = sum(1 for b in budget_report if b["pct_used"] < 90)
        total_b    = len(budget_report)
        score     += int((safe_count / total_b) * 30)
    else:
        score += 15  # Neutral if no budgets set

    # --- Status label ---
    if score >= 90:
        status = "Excellent"
        color  = "excellent"
    elif score >= 70:
        status = "Good"
        color  = "good"
    elif score >= 50:
        status = "Moderate"
        color  = "moderate"
    else:
        status = "Risky"
        color  = "risky"

    return {"score": score, "status": status, "color": color}
# ============================================================
# OpenRouter API — AI Assistant Fallback
# ============================================================

def call_openrouter(user_message: str, intent: str) -> str:
    """
    Sends the user's message to the OpenRouter API and returns an AI-generated reply.
    If the API key is missing or the call fails, returns a safe fallback response.
    """

    # Pull the API key from environment variables (set in .env)
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # If no API key is configured, return a pre-written fallback instead
    if not api_key:
        fallbacks = {
            "financial_tips": "💡 **Financial Tip:** Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            "greeting":       "👋 Hello! I'm SmartExpenseAI. Try 'add income' or 'show summary'!",
            "unknown":        "I can help you track income/expenses, set budgets and goals, and give financial tips.",
        }
        # Return the matching fallback, or the generic one if intent isn't recognised
        return fallbacks.get(intent, fallbacks["unknown"])

    # Tell the AI what kind of assistant it should behave like
    system_prompt = (
        "You are SmartExpenseAI, a friendly financial assistant for a personal expense tracker. "
        "Keep responses concise (2-3 sentences), actionable, warm, and occasionally use emojis."
    )

    try:
        # Make the POST request to OpenRouter's chat endpoint
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": "google/gemini-2.0-flash-exp:free",  # Free tier model
                "messages": [
                    {"role": "system", "content": system_prompt},  # Sets AI personality
                    {"role": "user",   "content": user_message},   # The actual question
                ],
                "max_tokens":  300,   # Keep replies short and focused
                "temperature": 0.7,   # Slightly creative but still grounded
            },
            timeout=10,  # Don't wait more than 10 seconds for a response
        )

        # If the API returned successfully, pull out the text reply
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        # API responded but with an error status — return a polite fallback
        return "I'm having trouble connecting right now. Try 'show summary' or 'add expense'."

    except requests.exceptions.Timeout:
        # The API took too long to respond
        return "The request timed out. Please try again in a moment."

    except Exception as e:
        # Something else went wrong — log it and return a safe message
        print(f"OpenRouter error: {e}")
        return "I'm here to help! Try 'add income', 'show summary', or 'set budget'."

# ============================================================
# Core Page Routes
# ============================================================

@app.route('/')
def index():
    """Renders the public homepage (landing page)."""
    return render_template('index.html')


@app.route('/test')
def test():
    """
    Quick sanity-check route — fetches all users from the DB and
    prints them as plain text. Useful during development only.
    """
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    result = cur.fetchall()
    return str(result)  # Raw output, not a proper template


# ============================================================
# Authentication Routes
# ============================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handles new user sign-up.
    GET  → shows the registration form.
    POST → validates input, hashes the password, saves user to DB.
    """

    if request.method == 'POST':

        # Pull all the fields the user filled in
        username        = request.form['username']
        employee_type = request.form['employee_type']
        email           = request.form['email']
        password        = request.form['password']

        # Job title and salary only apply to employees, not self-employed users
        if employee_type == 'Employee':
            job_title = request.form.get('job_title')

            monthly_salary = request.form.get('monthly_salary')

            # FIX: handle None + empty string safely
            if not monthly_salary or monthly_salary.strip() == '':
                monthly_salary = 0
            else:
                monthly_salary = float(monthly_salary)

        else:
            job_title = None
            monthly_salary = 0
        # Never store plain-text passwords — hash them first
        hashed_password = generate_password_hash(password)

        # Insert the new user record into the database
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO users
                (username, employee_type, job_title, monthly_salary, email, password)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (username, employee_type, job_title, monthly_salary, email, hashed_password))
        mysql.connection.commit()  # Persist the insert
        cur.close()

        # Send the new user to the login page to sign in
        return redirect(url_for('login'))

    # GET request — just show the empty registration form
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles user login.
    GET  → shows the login form.
    POST → checks credentials, starts a session if correct.
    """

    if request.method == 'POST':

        # Get what the user typed in
        email    = request.form['email']
        password = request.form['password']

        # Look up the user by email
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if user:
            # Compare the submitted password against the stored hash
            if check_password_hash(user[3], password):
                # Credentials are valid — save their ID and username in the session
                session['user_id']  = user[0]
                session['username'] = user[1]
                return redirect(url_for('dashboard'))  # Take them straight to the dashboard

            # Password didn't match
            return "Invalid Password"

        # No user found with that email
        return "User Not Found"

    # GET request — show the login form
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clears the session and sends the user back to the homepage."""
    session.clear()
    return redirect(url_for('index'))


# ============================================================
# Dashboard Route
# ============================================================

@app.route('/dashboard')
def dashboard():
    """
    The main dashboard — the user's financial command centre.
    Pulls together daily/monthly/yearly summaries, budgets, goals,
    category breakdowns, recent transactions, and AI forecasting.
    """

    # Redirect to login if the user isn't authenticated
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Local imports kept here since they're only needed in this route
    from datetime import date
    from machine_learning import forecasting

    cur     = mysql.connection.cursor()
    user_id = session['user_id']
    today   = date.today()

    print("\n=================================================")
    print("DASHBOARD LOADED")
    print("=================================================")
    print(f"[DASHBOARD] User ID: {user_id}")

    # ----------------------------------------------------------
    # How long has this user been tracking?
    # ----------------------------------------------------------

    # Find the date of their very first recorded transaction
    cur.execute("""
        SELECT MIN(transaction_date)
        FROM transactions
        WHERE user_id=%s
    """, (user_id,))

    first_date = cur.fetchone()[0]  # Will be None if they have no transactions yet

    # Calculate how many days have passed since that first transaction
    days_tracked = (today - first_date).days if first_date else 0

    print(f"[DASHBOARD] Days tracked: {days_tracked}")

    # ----------------------------------------------------------
    # Today's income and expense totals
    # ----------------------------------------------------------

    cur.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN type='income'  THEN amount ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0)
        FROM transactions
        WHERE user_id=%s
          AND DATE(transaction_date) = CURDATE()
    """, (user_id,))

    d             = cur.fetchone()
    daily_income  = float(d[0] or 0)   # What came in today
    daily_expense = float(d[1] or 0)   # What went out today
    daily_balance = daily_income - daily_expense  # Net for today

    print(f"[DASHBOARD] Daily Income: {daily_income}")
    print(f"[DASHBOARD] Daily Expense: {daily_expense}")

    # ----------------------------------------------------------
    # This month's income and expense totals
    # ----------------------------------------------------------

    cur.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN type='income'  THEN amount ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0)
        FROM transactions
        WHERE user_id=%s
          AND YEAR(transaction_date)  = YEAR(%s)
          AND MONTH(transaction_date) = MONTH(%s)
    """, (user_id, today, today))

    m               = cur.fetchone()
    monthly_income  = float(m[0] or 0)
    monthly_expense = float(m[1] or 0)
    monthly_balance = monthly_income - monthly_expense

    print(f"[DASHBOARD] Monthly Income: {monthly_income}")
    print(f"[DASHBOARD] Monthly Expense: {monthly_expense}")

    # ----------------------------------------------------------
    # This year's income and expense totals
    # ----------------------------------------------------------

    cur.execute("""
        SELECT type, SUM(amount)
        FROM transactions
        WHERE user_id=%s
          AND YEAR(transaction_date) = YEAR(%s)
        GROUP BY type
    """, (user_id, today))

    # Convert the two-row result into a dict so we can look up by type name
    y               = dict(cur.fetchall() or {})
    yearly_income   = float(y.get('income',  0))
    yearly_expense  = float(y.get('expense', 0))
    yearly_balance  = yearly_income - yearly_expense

    print(f"[DASHBOARD] Yearly Income: {yearly_income}")
    print(f"[DASHBOARD] Yearly Expense: {yearly_expense}")

    # ----------------------------------------------------------
    # All-time lifetime totals
    # ----------------------------------------------------------

    cur.execute("""
        SELECT type, SUM(amount)
        FROM transactions
        WHERE user_id=%s
        GROUP BY type
    """, (user_id,))

    total         = dict(cur.fetchall() or {})
    total_income  = float(total.get('income',  0))
    total_expense = float(total.get('expense', 0))
    savings       = total_income - total_expense  # Net savings across all time

    # What percentage of total income was saved?
    savings_rate = (savings / total_income * 100) if total_income else 0

    print(f"[DASHBOARD] Total Income: {total_income}")
    print(f"[DASHBOARD] Total Expense: {total_expense}")

    # ----------------------------------------------------------
    # Expense breakdown by category (for the pie chart / table)
    # ----------------------------------------------------------

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category
        ORDER BY SUM(amount) DESC
    """, (user_id,))

    expense_cats              = cur.fetchall()
    expense_category_analysis = [(cat, float(amt)) for cat, amt in expense_cats]

    # The biggest spending category — shown as a highlight on the dashboard
    top_category = expense_cats[0] if expense_cats else ('None', 0)

    # ----------------------------------------------------------
    # Income breakdown by category
    # ----------------------------------------------------------

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='income'
        GROUP BY category
        ORDER BY SUM(amount) DESC
    """, (user_id,))

    income_cats              = cur.fetchall()
    income_category_analysis = [(cat, float(amt)) for cat, amt in income_cats]

    # ----------------------------------------------------------
    # Savings insight message — a one-liner nudge based on the savings rate
    # ----------------------------------------------------------

    if savings_rate >= 30:
        insight = "Excellent savings rate!"        # They're doing really well
    elif savings_rate >= 20:
        insight = "Good savings habit!"            # On the right track
    elif savings_rate >= 10:
        insight = "Try to save more money."        # Room for improvement
    elif savings_rate > 0:
        insight = "Reduce unnecessary expenses."   # Savings are thin
    else:
        insight = "Your expenses exceed income."   # Spending more than earning

    # ----------------------------------------------------------
    # Budget data — what the user planned to spend per category
    # ----------------------------------------------------------

    cur.execute("""
        SELECT id, category, amount, period
        FROM budgets
        WHERE user_id=%s
    """, (user_id,))

    budgets = cur.fetchall()

    # ----------------------------------------------------------
    # Actual spending per category — used to compare against budgets
    # ----------------------------------------------------------

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category
    """, (user_id,))

    # Dictionary keyed by category name for fast lookup
    spent_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    # ----------------------------------------------------------
    # Build the budget report — budget vs. actual for each category
    # ----------------------------------------------------------

    budget_report = []

    for bid, cat, amt, period in budgets:
        budget_amount = float(amt)

        # How much was actually spent in this category (0 if none recorded)
        spent     = spent_map.get(cat, 0)
        remaining = budget_amount - spent    # Positive means under budget, negative means over

        # What percentage of the budget has been used up?
        pct_used = (spent / budget_amount * 100) if budget_amount else 0

        budget_report.append({
            "id":        bid,
            "category":  cat,
            "budget":    budget_amount,
            "spent":     spent,
            "remaining": remaining,
            "pct_used":  round(pct_used, 2),  # Rounded to 2 decimal places for display
            "period":    period,
        })

    # ----------------------------------------------------------
    # Financial goals — targets the user is saving towards
    # ----------------------------------------------------------

    goals = []  # Will stay empty if the table doesn't exist yet

    try:
        cur.execute("""
            SELECT id, goal_name, target_amount, saved_amount
            FROM financial_goals
            WHERE user_id=%s
            ORDER BY id DESC
        """, (user_id,))

        goal_rows = cur.fetchall()

        for gid, gname, target, saved in goal_rows:
            target_f = float(target or 0)
            saved_f  = float(saved  or 0)

            # Calculate how close they are to hitting this goal
            progress = round((saved_f / target_f * 100), 1) if target_f else 0

            goals.append({
                "id":            gid,
                "goal_name":     gname,
                "target_amount": target_f,
                "saved_amount":  saved_f,
                "progress":      min(progress, 100),  # Cap at 100% even if they've overshoot
            })

    except Exception as e:
        # The table might not exist in older deployments — that's fine, just skip goals
        print(f"[DASHBOARD] Goals query failed (table may not exist): {e}")

    print(f"[DASHBOARD] Goals: {len(goals)}")

    # ----------------------------------------------------------
    # The 10 most recent transactions for the activity feed
    # ----------------------------------------------------------

    cur.execute("""
        SELECT transaction_date, type, category, amount, description
        FROM transactions
        WHERE user_id=%s
        ORDER BY transaction_date DESC
        LIMIT 10
    """, (user_id,))

    # Format dates as strings so Jinja2 can display them cleanly
    transactions = [
        (
            r[0].strftime('%Y-%m-%d'),  # Date as a readable string
            r[1],                        # 'income' or 'expense'
            r[2],                        # Category name
            float(r[3]),                 # Amount as a float
            r[4] or ''                   # Description, defaulting to empty string
        )
        for r in cur.fetchall()
    ]

    # ----------------------------------------------------------
    # AI Forecasting — predict future spending using the ML model
    # ----------------------------------------------------------

    print("\n=================================================")
    print("AI FORECASTING")
    print("=================================================")

    # Safe defaults — shown if forecasting fails or there isn't enough data
    forecast_data = {
        "next_day":  {"amount": 0, "transactions": 0},
        "weekly":    {"amount": 0, "transactions": 0},
        "monthly":   {"amount": 0, "transactions": 0},
        "yearly":    {"amount": 0, "transactions": 0},
        "trend":     "No Trend",
        "available": False   # Flag used in the template to show/hide forecast cards
    }

    # Default trend display values in case forecasting doesn't run
    trend_label = " Stable"
    trend_tone  = "neutral"  # Maps to a CSS class in the template

    try:
        print("[DASHBOARD] Starting forecasting...")

        # Re-train the model on this user's latest transaction history
        forecasting.train_model(mysql, user_id)

        # Generate predictions for each time horizon
        next_day_forecast = forecasting.next_day(mysql, user_id)
        weekly_forecast   = forecasting.weekly(mysql, user_id)
        monthly_forecast  = forecasting.monthly(mysql, user_id)
        yearly_forecast   = forecasting.yearly(mysql, user_id)

        # Get the overall spending trend (e.g. "increasing", "stable")
        trend = forecasting.get_trend(mysql, user_id)

        print(f"[DASHBOARD] Next Day Forecast: {next_day_forecast}")
        print(f"[DASHBOARD] Trend: {trend}")

        # Pack all forecast results into one dict for the template
        forecast_data = {
            "next_day":  next_day_forecast,
            "weekly":    weekly_forecast,
            "monthly":   monthly_forecast,
            "yearly":    yearly_forecast,
            "trend":     trend,
            "available": True   # Forecast is ready — show the cards in the UI
        }

        # Map the trend string to a user-friendly label and a CSS colour class
        trend_lower = str(trend).lower()

        if "increas" in trend_lower or "up" in trend_lower:
            trend_label = "Spending Increasing"
            trend_tone  = "warning"   # Red-ish — spending is going up
        elif "decreas" in trend_lower or "down" in trend_lower:
            trend_label = "Spending Decreasing"
            trend_tone  = "success"   # Green — spending is going down
        else:
            trend_label = "⚖ Spending Stable"
            trend_tone  = "neutral"   # Gold — no significant change

    except Exception as e:
        # If forecasting fails (not enough data, model error, etc.) — log and move on
        print(f"[DASHBOARD ERROR] Forecasting failed: {e}")

    # Done with the database for this request
    cur.close()

    # ----------------------------------------------------------
    # Date labels for the monthly and yearly summary sections
    # ----------------------------------------------------------

    import calendar  # Only needed here, so imported locally

    month_name = today.strftime('%B')     # e.g. "June"
    year_str   = today.strftime('%Y')     # e.g. "2026"

    # First day of the current month
    month_start = today.replace(day=1).strftime('%Y-%m-%d')

    # Last day of the current month (accounts for varying month lengths)
    last_day  = calendar.monthrange(today.year, today.month)[1]
    month_end = today.replace(day=last_day).strftime('%Y-%m-%d')

    # Full year range for the yearly summary header
    year_start = f"{today.year}-01-01"
    year_end   = f"{today.year}-12-31"

    # Human-readable labels displayed in the template section headers
    current_month_label = f"{month_name} {year_str}"   # e.g. "June 2026"
    current_year_label  = year_str                      # e.g. "2026"

    print("\n=================================================")
    print("DASHBOARD RENDERED")
    print("=================================================")

    # Pass everything the template needs to render the full dashboard
    return render_template(
        "dashboard.html",

        # User identity
        username    = session.get("username", "User"),
        today       = today.strftime('%Y-%m-%d'),
        start_date  = first_date.strftime('%Y-%m-%d') if first_date else today.strftime('%Y-%m-%d'),
        days_tracked = days_tracked,

        # Daily summary
        daily_income  = f"{daily_income:,.2f}",
        daily_expense = f"{daily_expense:,.2f}",
        daily_balance = f"{daily_balance:,.2f}",

        # Monthly summary
        monthly_income  = f"{monthly_income:,.2f}",
        monthly_expense = f"{monthly_expense:,.2f}",
        monthly_balance = f"{monthly_balance:,.2f}",

        # Yearly summary
        yearly_income  = f"{yearly_income:,.2f}",
        yearly_expense = f"{yearly_expense:,.2f}",
        yearly_balance = f"{yearly_balance:,.2f}",

        # Lifetime totals
        total_income  = f"{total_income:,.2f}",
        total_expense = f"{total_expense:,.2f}",
        savings       = f"{savings:,.2f}",
        savings_rate  = f"{savings_rate:.1f}",

        # Category highlights
        top_category              = top_category,
        insight                   = insight,
        expense_category_analysis = expense_category_analysis,
        income_category_analysis  = income_category_analysis,

        # Budgets and goals
        budget_report = budget_report,
        goals         = goals,

        # Transaction history
        transactions = transactions,

        # Date range labels for the template headers
        current_month_label = current_month_label,
        current_year_label  = current_year_label,
        month_start         = month_start,
        month_end           = month_end,
        year_start          = year_start,
        year_end            = year_end,

        # AI forecast data and trend badge
        forecast    = forecast_data,
        trend_label = trend_label,
        trend_tone  = trend_tone,
    )

# ============================================================
# Transaction Routes
# ============================================================

@app.route('/add_income', methods=['GET', 'POST'])
def add_income():
    """
    Add an income transaction.
    GET  → shows the income form.
    POST → saves the transaction and updates the ML training dataset.
    """

    if request.method == 'POST':

        # Grab what the user filled in on the form
        description = request.form['description']
        amount      = request.form['amount']
        category    = request.form['category']

        # Write the new income record to the database
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (
            session['user_id'],  # Tie the transaction to the logged-in user
            'income',            # Hard-coded type for this route
            category,
            amount,
            description
        ))
        mysql.connection.commit()
        cur.close()

        # Feed this example into the ML training data so the model keeps improving
        append_to_dataset(description, amount, 'income', category)

        # Go back to the dashboard to see the updated totals
        return redirect(url_for('dashboard'))

    # GET — show the blank income form
    return render_template('add_income.html')


@app.route('/add_expense', methods=['GET', 'POST'])
def add_expense():
    """
    Add an expense transaction.
    GET  → shows the expense form.
    POST → saves the transaction and updates the ML training dataset.
    """

    if request.method == 'POST':

        # Grab what the user filled in on the form
        description = request.form['description']
        amount      = request.form['amount']
        category    = request.form['category']

        # Write the new expense record to the database
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (
            session['user_id'],   # Tie the transaction to the logged-in user
            'expense',            # Hard-coded type for this route
            category,
            amount,
            description
        ))
        mysql.connection.commit()
        cur.close()

        # Keep the ML dataset up to date with this new example
        append_to_dataset(description, amount, 'income', category)

        # Return to the dashboard
        return redirect(url_for('dashboard'))

    # GET — show the blank expense form
    return render_template('add_expense.html')


@app.route('/view_income_and_expense')
def view_income_and_expense():
    """
    Lists all transactions for the current user.
    Supports optional keyword search across category and type fields.
    """

    # Check if a search term was passed in the URL (e.g. ?search=food)
    search = request.args.get('search', '')

    cur = mysql.connection.cursor()

    if search:
        # Filter transactions that match the search term in any of these columns
        cur.execute("""
            SELECT * FROM transactions
            WHERE user_id=%s
              AND (category LIKE %s OR type LIKE %s OR period LIKE %s)
        """, (session['user_id'], f'%{search}%', f'%{search}%', f'%{search}%'))
    else:
        # No search — fetch every transaction for this user
        cur.execute(
            "SELECT * FROM transactions WHERE user_id=%s",
            (session['user_id'],)
        )

    transactions = cur.fetchall()
    return render_template('view_income_and_expense.html', transactions=transactions)


@app.route('/edit_income_and_expense/<int:id>', methods=['GET', 'POST'])
def edit_income_and_expense(id):
    """
    Edit a single transaction by its ID.
    GET  → pre-fills the edit form with the current values.
    POST → saves the updated values to the database.
    """

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        # Apply the user's edits to the matching row
        cur.execute("""
            UPDATE transactions
            SET category=%s, amount=%s, description=%s, period=%s
            WHERE id=%s
        """, (
            request.form['category'],
            request.form['amount'],
            request.form['description'],
            request.form['period'],
            id,
        ))
        mysql.connection.commit()
        return redirect(url_for('view_income_and_expense'))

    # GET — load the existing transaction so the form shows current values
    cur.execute("SELECT * FROM transactions WHERE id=%s", (id,))
    transaction = cur.fetchone()
    return render_template('edit_income_and_expense.html', transaction=transaction)


@app.route('/delete_transaction/<int:id>')
def delete_transaction(id):
    """Permanently deletes a single transaction by ID."""

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM transactions WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for('view_income_and_expense'))


# ============================================================
# Profile Routes
# ============================================================

@app.route('/profile')
def profile():
    """Displays the logged-in user's profile information."""

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT username, email, employee_type, job_title, monthly_salary, created_at, last_login
        FROM users
        WHERE id=%s
    """, (session['user_id'],))
    user = cur.fetchone()
    return render_template('profile.html', user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    """
    Lets the user update their profile details.
    GET  → pre-fills the form with their current data.
    POST → writes the changes to the database and updates the session.
    """

    # Protect this route — only logged-in users can edit their profile
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        username        = request.form['username']
        employee_type = request.form['employee_type']

        # Job title and salary only matter for employees
        if employee_type == 'Employee':
            job_title      = request.form.get('job_title')
            monthly_salary = request.form.get('monthly_salary') or None  # Store NULL if blank
        else:
            job_title      = None
            monthly_salary = None

        # Update the user's record in the database
        cur.execute("""
            UPDATE users
            SET username=%s, employee_type=%s, job_title=%s, monthly_salary=%s
            WHERE id=%s
        """, (username, employee_type, job_title, monthly_salary, session['user_id']))
        mysql.connection.commit()

        # Keep the session username in sync so the navbar shows the new name immediately
        session['username'] = username

        return redirect(url_for('profile'))

    # GET — load current values so the form isn't blank
    cur.execute(
        "SELECT username, employee_type, job_title, monthly_salary FROM users WHERE id=%s",
        (session['user_id'],)
    )
    user = cur.fetchone()
    return render_template('edit_profile.html', user=user)


# ============================================================
# Goals Routes
# ============================================================
@app.route('/add_goal', methods=['GET', 'POST'])
def add_goal():
    """
    Lets the user create a new savings goal.
    GET  → shows the goal creation form.
    POST → inserts the goal into the database.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        
        # Make sure column order matches your database table structure
        # Typical goals table: id (auto), user_id, goal_name, target_amount, current_amount, target_date, created_at
        cur.execute("""
            INSERT INTO goals (user_id, goal_name, target_amount, current_amount, target_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            session['user_id'],
            request.form['goal_name'],
            float(request.form['target_amount']),  # Convert to float
            float(request.form.get('current_amount', 0)),  # Default to 0 if not filled in
            request.form.get('target_date') or None,  # Handle empty date
        ))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('view_goals'))  # Redirect to view_goals instead of dashboard

    return render_template('add_goal.html')

@app.route('/view_goals')
def view_goals():
    """
    Displays all savings goals for the logged-in user,
    along with their current total savings balance.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    # Fetch every goal this user has set
    # CORRECTED ORDER: id, goal_name, target_amount, current_amount, target_date
    cur.execute("""
        SELECT id, goal_name, target_amount, current_amount, target_date
        FROM goals
        WHERE user_id=%s
    """, (session['user_id'],))
    goals = cur.fetchall()

    # Net savings = total income minus total expenses across all time
    cur.execute("""
        SELECT COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE -amount END), 0)
        FROM transactions
        WHERE user_id=%s
    """, (session['user_id'],))
    savings = cur.fetchone()[0]

    cur.close()
    return render_template('view_goals.html', goals=goals, savings=savings)

@app.route('/delete_goal/<int:id>')
def delete_goal(id):
    """Deletes a specific goal by ID, only if it belongs to the current user."""

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    # The user_id check prevents users from deleting each other's goals
    cur.execute(
        "DELETE FROM goals WHERE id=%s AND user_id=%s",
        (id, session['user_id'])
    )
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('view_goals'))


@app.route('/edit_goal/<int:id>', methods=['GET', 'POST'])
def edit_goal(id):
    """
    Edits a specific goal by ID.
    GET  → pre-fills the form with the goal's current values.
    POST → updates the goal in the database.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        cur.execute("""
            UPDATE goals
            SET goal_name=%s, target_amount=%s, current_amount=%s, target_date=%s
            WHERE id=%s AND user_id=%s
        """, (
            request.form['goal_name'],
            request.form['target_amount'],
            request.form['current_amount'],
            request.form['target_date'],
            id,
            session['user_id'],  # Ensures the user can only edit their own goals
        ))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('view_goals'))

    # GET — load the goal so the form can show the current values
    cur.execute(
        "SELECT * FROM goals WHERE id=%s AND user_id=%s",
        (id, session['user_id'])
    )
    goal = cur.fetchone()
    cur.close()
    return render_template('edit_goal.html', goal=goal)


# ============================================================
# Budget Routes
# ============================================================

@app.route('/add_budget', methods=['GET', 'POST'])
def add_budget():
    """
    Adds a new budget for a specific spending category.
    GET  → shows the budget creation form.
    POST → inserts the budget into the database.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO budgets (user_id, category, amount, period) VALUES (%s, %s, %s, %s)",
            (
                session['user_id'],
                request.form['category'],
                request.form['amount'],
                request.form['period'],
            ),
        )
        mysql.connection.commit()
        # Go back to the dashboard where the budget tracker is visible
        return redirect(url_for('dashboard'))

    return render_template('add_budget.html')


@app.route('/edit_budget/<int:id>', methods=['GET', 'POST'])
def edit_budget(id):
    """
    Edits an existing budget by ID.
    GET  → pre-fills the form with current budget values.
    POST → applies the changes to the database.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        cur.execute(
            "UPDATE budgets SET category=%s, amount=%s, period=%s WHERE id=%s AND user_id=%s",
            (
                request.form['category'],
                request.form['amount'],
                request.form['period'],
                id,
                session['user_id'],  # Safety check — users can only edit their own budgets
            ),
        )
        mysql.connection.commit()
        return redirect(url_for('dashboard'))

    # GET — fetch the budget so the form shows the right values
    cur.execute(
        "SELECT id, category, amount, period FROM budgets WHERE id=%s AND user_id=%s",
        (id, session['user_id']),
    )
    budget = cur.fetchone()

    # If no budget was found, return a proper 404 instead of crashing
    if not budget:
        return "Budget not found", 404

    return render_template('edit_budget.html', budget=budget)


@app.route('/delete_budget/<int:id>')
def delete_budget(id):
    """Deletes a specific budget by ID, only if it belongs to the current user."""

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute(
        "DELETE FROM budgets WHERE id=%s AND user_id=%s",
        (id, session['user_id'])
    )
    mysql.connection.commit()
    return redirect(url_for('dashboard'))


@app.route('/view_budgets')
def view_budgets():
    """
    Shows all budgets with a live comparison against actual spending
    for the current month.
    """

    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    # Pull every budget the user has set up
    cur.execute(
        "SELECT id, category, amount, period FROM budgets WHERE user_id=%s",
        (session['user_id'],),
    )
    budgets = cur.fetchall()

    # How much was actually spent in each category THIS month?
    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s
          AND type='expense'
          AND YEAR(transaction_date)  = YEAR(CURDATE())
          AND MONTH(transaction_date) = MONTH(CURDATE())
        GROUP BY category
    """, (session['user_id'],))

    # Build a lookup dict so we can match spending to each budget below
    spent_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    # Calculate progress for each budget category
    budget_report = []
    for bid, cat, amt, period in budgets:
        budget_amount = float(amt)
        spent         = spent_map.get(cat, 0)   # 0 if nothing spent yet
        remaining     = budget_amount - spent
        pct_used      = (spent / budget_amount * 100) if budget_amount else 0

        budget_report.append({
            "id":        bid,
            "category":  cat,
            "budget":    budget_amount,
            "spent":     spent,
            "remaining": remaining,
            "pct_used":  round(pct_used, 2),
            "period":    period,
        })

    return render_template('view_budgets.html', budget_report=budget_report)


# ============================================================
# Analytics Route
# ============================================================
@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur     = mysql.connection.cursor()

    # --- Overall totals ---
    cur.execute(
        "SELECT type, SUM(amount) FROM transactions WHERE user_id=%s GROUP BY type",
        (user_id,)
    )
    data          = dict(cur.fetchall() or [])
    total_income  = float(data.get('income',  0))
    total_expense = float(data.get('expense', 0))
    savings       = total_income - total_expense

    # --- Daily breakdown ---
    cur.execute("""
        SELECT DATE(transaction_date) AS day, type, SUM(amount)
        FROM transactions WHERE user_id=%s
        GROUP BY day, type ORDER BY day
    """, (user_id,))
    daily_rows    = cur.fetchall()
    days          = sorted({r[0].strftime('%Y-%m-%d') for r in daily_rows})
    daily_income  = {d: 0 for d in days}
    daily_expense = {d: 0 for d in days}
    for day, ttype, amt in daily_rows:
        d = day.strftime('%Y-%m-%d')
        if ttype == 'income': daily_income[d]  = float(amt)
        else:                 daily_expense[d] = float(amt)

    # --- Monthly breakdown ---
    cur.execute("""
        SELECT DATE_FORMAT(transaction_date, '%%Y-%%m') AS month, type, SUM(amount)
        FROM transactions WHERE user_id=%s
        GROUP BY month, type ORDER BY month
    """, (user_id,))
    monthly_rows    = cur.fetchall()
    months          = sorted({r[0] for r in monthly_rows})
    monthly_income  = {m: 0 for m in months}
    monthly_expense = {m: 0 for m in months}
    for m, ttype, amt in monthly_rows:
        if ttype == 'income': monthly_income[m]  = float(amt)
        else:                 monthly_expense[m] = float(amt)

    # --- Yearly breakdown ---
    cur.execute("""
        SELECT DATE_FORMAT(transaction_date, '%%Y') AS year, type, SUM(amount)
        FROM transactions WHERE user_id=%s
        GROUP BY year, type ORDER BY year
    """, (user_id,))
    yearly_rows    = cur.fetchall()
    years          = sorted({r[0] for r in yearly_rows})
    yearly_income  = {y: 0 for y in years}
    yearly_expense = {y: 0 for y in years}
    for y, ttype, amt in yearly_rows:
        if ttype == 'income': yearly_income[y]  = float(amt)
        else:                 yearly_expense[y] = float(amt)

    # --- Category breakdown ---
    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    cat_rows        = cur.fetchall()
    categories      = [r[0] for r in cat_rows]
    category_values = [float(r[1]) for r in cat_rows]

    # --- Budget vs actual ---
    cur.execute("""
        SELECT b.category, b.amount, COALESCE(SUM(t.amount), 0)
        FROM budgets b
        LEFT JOIN transactions t
            ON  b.category COLLATE utf8mb4_unicode_ci =
                t.category COLLATE utf8mb4_unicode_ci
            AND t.type='expense' AND t.user_id=%s
        WHERE b.user_id=%s
        GROUP BY b.category, b.amount
    """, (user_id, user_id))
    budget_rows       = cur.fetchall()
    budget_categories = [r[0] for r in budget_rows]
    budget_values     = [float(r[1]) for r in budget_rows]
    actual_values_raw = [float(r[2]) for r in budget_rows]

    # Build enriched budget report for alerts
    budget_report_raw = [
        {
            "category": r[0],
            "budget":   float(r[1]),
            "spent":    float(r[2]),
            "remaining": float(r[1]) - float(r[2]),
            "pct_used": round((float(r[2]) / float(r[1]) * 100) if r[1] else 0, 2),
            "period":   "monthly",
        }
        for r in budget_rows
    ]

    # --- Goals ---
    cur.execute("SELECT goal_name, target_amount FROM goals WHERE user_id=%s", (user_id,))
    goals         = cur.fetchall()
    goal_names    = [r[0] for r in goals]
    goal_progress = [
        round((savings / float(r[1]) * 100) if r[1] else 0, 2)
        for r in goals
    ]

    cur.close()

    # ── NEW: AI Intelligence ──────────────────────────────
    budget_alerts  = calculate_budget_alerts(budget_report_raw)
    ai_insights    = generate_ai_insights(
        total_income, total_expense, savings, cat_rows, budget_report_raw
    )
    health         = calculate_health_score(
        total_income, total_expense, savings, budget_report_raw
    )

    # ── NEW: ML Predictions ───────────────────────────────
    from machine_learning import forecasting
    weekly_forecast  = {"amount": 0, "ai_ready": False}
    monthly_forecast = {"amount": 0, "ai_ready": False}
    try:
        weekly_forecast  = forecasting.weekly(mysql, user_id)
        monthly_forecast = forecasting.monthly(mysql, user_id)
    except Exception as e:
        print(f"[ANALYTICS] Forecasting error: {e}")

    return render_template(
        "analytics.html",
        username      = session.get("username", "User"),
        total_income  = total_income,
        total_expense = total_expense,

        days          = days,
        daily_income  = list(daily_income.values()),
        daily_expense = list(daily_expense.values()),

        months          = months,
        monthly_income  = list(monthly_income.values()),
        monthly_expense = list(monthly_expense.values()),

        years          = years,
        yearly_income  = list(yearly_income.values()),
        yearly_expense = list(yearly_expense.values()),

        categories      = categories,
        category_values = category_values,

        budget_categories = budget_categories,
        budget_values     = budget_values,
        actual_values     = actual_values_raw,

        goal_names    = goal_names,
        goal_progress = goal_progress,

        # ── NEW variables ──
        ai_insights      = ai_insights,
        budget_alerts    = budget_alerts,
        health           = health,
        weekly_forecast  = weekly_forecast,
        monthly_forecast = monthly_forecast,
    )

# ============================================================
# Chatbot Endpoints
# ============================================================

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Receives a plain-text message from the user, runs it through the
    chatbot engine, and returns the bot's reply as JSON.

    Request:  POST /chatbot  { "message": "I spent 300 on pizza" }
    Response: { "reply": "Expense recorded! ₹300 for Pizza under Food" }
    """

    # Only logged-in users can use the chatbot
    if 'user_id' not in session:
        return jsonify({"reply": "Please log in first."})

    # Pull the message from the JSON body
    data    = request.get_json()
    message = data.get("message", "").strip()

    # Don't process empty messages
    if not message:
        return jsonify({"reply": "Please type something!"})

    user_id  = session['user_id']
    username = session.get('username', 'User')

    try:
        cur = mysql.connection.cursor()

        # Record what the user said in the chat history table
        cur.execute(
            "INSERT INTO chatbot_messages (user_id, sender, message) VALUES (%s, %s, %s)",
            (user_id, "user", message)
        )
        mysql.connection.commit()

        # Hand the message off to the chatbot engine to generate a reply
        reply = chatbot_engine.process_message(
            user_id  = user_id,
            message  = message,
            username = username,
        )

        # Record the bot's reply in the chat history too
        cur.execute(
            "INSERT INTO chatbot_messages (user_id, sender, message) VALUES (%s, %s, %s)",
            (user_id, "bot", reply)
        )
        mysql.connection.commit()
        cur.close()

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[CHATBOT ERROR] {e}")
        return jsonify({"reply": "Something went wrong. Please try again."})


@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    """
    Adds a transaction via JSON (used by forms or other clients, not the chatbot).
    The ML model automatically predicts the category from the description.

    Request:  POST /add_transaction
              { "description": "pizza delivery", "amount": 300, "transaction_type": "expense" }
    Response: { "message": "Transaction saved!", "category": "Food" }
    """

    # Require the user to be logged in
    if 'user_id' not in session:
        return jsonify({"error": "Login required"}), 401

    # Parse the JSON body
    data             = request.get_json()
    description      = data.get('description', '').strip()
    amount_raw       = data.get('amount')
    transaction_type = data.get('transaction_type', 'expense').strip().lower()

    # Description is mandatory — we need it for the ML prediction
    if not description:
        return jsonify({"error": "Description is required"}), 400

    # Make sure the amount is a positive number
    try:
        amount = float(amount_raw)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid amount"}), 400

    # Use the ML model to guess the best category for this transaction
    print(f"[ADD_TXN] description='{description}' type='{transaction_type}'")
    category = predict_category(description, transaction_type)
    print(f"[ADD_TXN] ML predicted category: '{category}'")

    # Save the transaction with the predicted category
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (session['user_id'], transaction_type, category, amount, description))
        mysql.connection.commit()
        cur.close()

        # Add this example to the training dataset so future predictions get better
        append_to_dataset(description, transaction_type, category)

        return jsonify({
            "message":  "Transaction saved!",
            "category": category,
        })

    except Exception as e:
        print(f"[ADD_TXN ERROR] {e}")
        return jsonify({"error": "Failed to save transaction"}), 500


@app.route('/load_chat')
def load_chat():
    """
    Returns the full chat history for the logged-in user as a JSON array.
    Used by the frontend to restore the conversation on page load.
    """

    # No session, no history — return an empty list
    if 'user_id' not in session:
        return jsonify([])

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT sender, message FROM chatbot_messages WHERE user_id=%s ORDER BY created_at ASC",
        (session['user_id'],)
    )
    rows = cur.fetchall()
    cur.close()

    # Return each message as a dict with sender and message keys
    return jsonify([{"sender": s, "message": m} for s, m in rows])


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """
    Wipes all chat messages for the current user.
    Called when the user clicks the "Clear Chat" button in the UI.
    """

    if 'user_id' not in session:
        return jsonify({"success": False})

    cur = mysql.connection.cursor()
    cur.execute(
        "DELETE FROM chatbot_messages WHERE user_id=%s",
        (session['user_id'],)
    )
    mysql.connection.commit()
    cur.close()

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)

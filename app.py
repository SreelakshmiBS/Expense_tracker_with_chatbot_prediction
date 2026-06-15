# SmartExpenseAI — main Flask application
# All routes, helpers, and configuration live in this single file.
# Keep it organised: imports → config → helpers → routes (public → auth → dashboard
# → transactions → profile → goals → budgets → analytics → chatbot).

import calendar
import os
import threading
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (Flask, jsonify, redirect, render_template,
                   request, session, url_for)
from flask_mysqldb import MySQL
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv
import requests

from chatbot.chatbot_engine import ChatbotEngine
from machine_learning.predict_category import predict_category
from machine_learning.utils.dataset_sync import append_to_dataset
from machine_learning import forecasting
from machine_learning.forecasting import retrain_all_models

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

app.config['MYSQL_HOST']     = MYSQL_HOST
app.config['MYSQL_PORT']     = MYSQL_PORT
app.config['MYSQL_USER']     = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB']       = MYSQL_DB
app.config['MYSQL_CHARSET']  = 'utf8mb4'

mysql = MySQL(app)


# ==============================================================================
# Auth decorator
# ==============================================================================

def login_required(f):
    # Redirect unauthenticated visitors to /login instead of scattering
    # 'if user_id not in session' checks across every route.
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ==============================================================================
# Background retraining helper
# ==============================================================================

def retrain_async(user_id):
    # We never want an HTTP response to wait while models retrain,
    # so we spin up a lightweight daemon thread and let it finish on its own.
    def _run():
        try:
            retrain_all_models(user_id, mysql)
        except Exception as e:
            print(f"[RETRAIN] Background error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ==============================================================================
# OpenRouter API
# ==============================================================================

def call_openrouter(user_message: str, intent: str = "unknown") -> str:
    # Read the key at call-time so it picks up any runtime env changes.
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # When there's no API key (local dev, missing .env, etc.) return a
    # sensible canned reply so the chatbot still works offline.
    if not api_key:
        fallbacks = {
            "financial_advice": (
                "Try the 50/30/20 rule: 50% on needs, 30% on wants, 20% into savings. "
                "Even saving ₹500 a month compounds significantly over time."
            ),
            "greeting": (
                "Hello! I'm SmartExpenseAI. Try: add income, show summary, "
                "or set food budget 5000 monthly."
            ),
            "unknown": (
                "I can help you track income and expenses, set budgets and goals, "
                "and give financial tips. Try 'show analytics' to get started."
            ),
        }
        return fallbacks.get(intent, fallbacks["unknown"])

    # System prompt keeps responses concise and India-focused.
    system_prompt = (
        "You are SmartExpenseAI, a friendly and knowledgeable personal finance assistant "
        "for an Indian expense-tracking app. Keep every response concise (2-3 sentences), "
        "actionable, and warm. Use ₹ for amounts. Occasionally use a relevant emoji."
    )

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": "google/gemini-2.5-flash-lite",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "max_tokens":  300,
                "temperature": 0.7,
            },
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        print(f"[OPENROUTER] HTTP {response.status_code}: {response.text[:200]}")
        return None

    except requests.exceptions.Timeout:
        print("[OPENROUTER] Request timed out.")
        return None
    except Exception as e:
        print(f"[OPENROUTER] Unexpected error: {e}")
        return None


# Wire the chatbot engine once at startup — pass call_openrouter directly
# so the engine never needs to know about the env or requests.
chatbot_engine = ChatbotEngine(mysql, openrouter_fn=call_openrouter)


# ==============================================================================
# Date helper
# ==============================================================================

def safe_date(value):
    # Converts whatever comes out of a form or DB into a plain ISO date string.
    # Falls back to today so we never store NULL by accident.
    if not value:
        return date.today().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            return date.today().isoformat()
    return date.today().isoformat()


# ==============================================================================
# Analytics intelligence helpers
# ==============================================================================

def generate_ai_insights(total_income, total_expense, savings, expense_cats, budget_report):
    # Build a ranked list of plain-language insights the analytics page can render.
    insights = []
    spending_ratio = (total_expense / total_income * 100) if total_income else 0

    # Spending-to-income ratio — the single most important signal
    if spending_ratio >= 90:
        insights.append({"text": f"Your expenses are {spending_ratio:.0f}% of your income — critically high.", "type": "danger",   "icon": "🚨"})
    elif spending_ratio >= 70:
        insights.append({"text": f"You are spending {spending_ratio:.0f}% of your income. Try to cut back.",  "type": "warning",  "icon": "⚠️"})
    else:
        insights.append({"text": f"You are spending {spending_ratio:.0f}% of your income — healthy range.",   "type": "positive", "icon": "✅"})

    # Savings status
    if savings > 0:
        insights.append({"text": f"You have saved ₹{savings:,.2f} in total. Keep it up!",                        "type": "positive", "icon": "💰"})
    elif savings < 0:
        insights.append({"text": f"Your expenses exceed income by ₹{abs(savings):,.2f}. Urgent action needed.", "type": "danger",   "icon": "📉"})

    # Highlight the single biggest expense category
    if expense_cats:
        top_cat, top_amt = expense_cats[0]
        insights.append({"text": f"{top_cat.title()} is your highest spending category at ₹{float(top_amt):,.2f}.", "type": "info", "icon": "📊"})

    # Flag any budget that has been breached or is close to the limit
    for b in budget_report:
        if b["pct_used"] > 100:
            over = b["spent"] - b["budget"]
            insights.append({"text": f"{b['category'].title()} budget exceeded by ₹{over:,.2f}.",            "type": "danger",  "icon": "🔴"})
        elif b["pct_used"] >= 85:
            insights.append({"text": f"{b['category'].title()} budget is {b['pct_used']:.0f}% used — almost full.", "type": "warning", "icon": "🟡"})

    # Prompt the user to add data if we have nothing to say yet
    if not insights:
        insights.append({"text": "Add more transactions to unlock AI insights.", "type": "info", "icon": "💡"})

    return insights


def calculate_budget_alerts(budget_report):
    # Tag each budget row with an alert level so the template can colour-code it.
    for b in budget_report:
        pct = b["pct_used"]
        if pct >= 90:
            b["alert_level"] = "danger"
        elif pct >= 70:
            b["alert_level"] = "warning"
        else:
            b["alert_level"] = "safe"
    return budget_report


def calculate_health_score(total_income, total_expense, savings, budget_report):
    # Composite 0-100 score built from three pillars:
    # 1) expense-to-income ratio (max 40 pts)
    # 2) whether the user is saving at all (max 30 pts)
    # 3) share of budgets not in the danger zone (max 30 pts)
    score = 0

    if total_income > 0:
        ratio = total_expense / total_income
        if ratio <= 0.5:    score += 40
        elif ratio <= 0.7:  score += 30
        elif ratio <= 0.85: score += 20
        elif ratio <= 1.0:  score += 10

    if savings > 0:    score += 30
    elif savings == 0: score += 10

    if budget_report:
        safe_count = sum(1 for b in budget_report if b["pct_used"] < 90)
        score += int((safe_count / len(budget_report)) * 30)
    else:
        score += 15   # No budgets set — neutral, not penalised

    if score >= 90:   status, color = "Excellent", "excellent"
    elif score >= 70: status, color = "Good",      "good"
    elif score >= 50: status, color = "Moderate",  "moderate"
    else:             status, color = "Risky",      "risky"

    return {"score": score, "status": status, "color": color}


# ==============================================================================
# DB connectivity check
# ==============================================================================

@app.route('/test-db')
def test_db():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT 1")
        return "DB CONNECTED"
    except Exception as e:
        return str(e)


# ==============================================================================
# Public routes
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


# ==============================================================================
# Authentication
# ==============================================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username      = request.form['username']
        employee_type = request.form['employee_type']
        email         = request.form['email']
        password      = request.form['password']

        # Only employees supply job title and salary — freelancers/business owners don't.
        if employee_type == 'Employee':
            job_title      = request.form.get('job_title')
            monthly_salary = request.form.get('monthly_salary')
            monthly_salary = float(monthly_salary) if monthly_salary and monthly_salary.strip() else 0
        else:
            job_title      = None
            monthly_salary = 0

        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO users
                (username, employee_type, job_title, monthly_salary, email, password)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (username, employee_type, job_title, monthly_salary, email, hashed_password))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        # Use a single vague error so we don't reveal whether the email exists.
        if not user or not check_password_hash(user[3], password):
            return render_template('login.html', error="Invalid email or password.")

        session['user_id']  = user[0]
        session['username'] = user[1]
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ==============================================================================
# Dashboard
# ==============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    today   = date.today()
    cur     = mysql.connection.cursor()

    # How many days has the user been tracking expenses?
    cur.execute("SELECT MIN(transaction_date) FROM transactions WHERE user_id=%s", (user_id,))
    first_date   = cur.fetchone()[0]
    days_tracked = (today - first_date).days if first_date else 0

    # Today's income vs expense
    cur.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN type='income'  THEN amount ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0)
        FROM transactions
        WHERE user_id=%s AND DATE(transaction_date) = CURDATE()
    """, (user_id,))
    d             = cur.fetchone()
    daily_income  = float(d[0] or 0)
    daily_expense = float(d[1] or 0)
    daily_balance = daily_income - daily_expense

    # Current month totals
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

    # Current year totals
    cur.execute("""
        SELECT type, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND YEAR(transaction_date) = YEAR(%s)
        GROUP BY type
    """, (user_id, today))
    y              = dict(cur.fetchall() or {})
    yearly_income  = float(y.get('income',  0))
    yearly_expense = float(y.get('expense', 0))
    yearly_balance = yearly_income - yearly_expense

    # All-time totals (used for savings calculation)
    cur.execute("SELECT type, SUM(amount) FROM transactions WHERE user_id=%s GROUP BY type", (user_id,))
    total         = dict(cur.fetchall() or {})
    total_income  = float(total.get('income',  0))
    total_expense = float(total.get('expense', 0))
    savings       = total_income - total_expense
    savings_rate  = (savings / total_income * 100) if total_income else 0

    # Expense categories ranked by total spend (used for pie chart and top-category card)
    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    expense_cats              = cur.fetchall()
    expense_category_analysis = [(cat, float(amt)) for cat, amt in expense_cats]
    top_category              = expense_cats[0] if expense_cats else ('None', 0)

    # Income categories (for the income breakdown widget)
    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='income'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    income_cats              = cur.fetchall()
    income_category_analysis = [(cat, float(amt)) for cat, amt in income_cats]

    # Human-readable label for the savings rate card
    if savings_rate >= 30:   insight = "Excellent savings rate!"
    elif savings_rate >= 20: insight = "Good savings habit!"
    elif savings_rate >= 10: insight = "Try to save more each month."
    elif savings_rate > 0:   insight = "Reduce unnecessary expenses."
    else:                    insight = "Your expenses exceed income."

    # Budget vs actual spend — we compare all-time spend per category
    cur.execute("SELECT id, category, amount, period FROM budgets WHERE user_id=%s", (user_id,))
    budgets = cur.fetchall()

    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='expense' GROUP BY category
    """, (user_id,))
    spent_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    budget_report = []
    for bid, cat, amt, period in budgets:
        budget_amount = float(amt)
        spent         = spent_map.get(cat, 0)
        remaining     = budget_amount - spent
        pct_used      = (spent / budget_amount * 100) if budget_amount else 0
        budget_report.append({
            "id": bid, "category": cat, "budget": budget_amount,
            "spent": spent, "remaining": remaining,
            "pct_used": round(pct_used, 2), "period": period,
        })

    # Goals — always read from the 'goals' table (dashboard and analytics are in sync)
    goals = []
    try:
        cur.execute("""
            SELECT id, goal_name, target_amount, current_amount, target_date
            FROM goals WHERE user_id=%s ORDER BY id DESC
        """, (user_id,))
        for row in cur.fetchall():
            gid, gname, target, saved, target_date = row
            target_f  = float(target or 0)
            saved_f   = float(saved  or 0)
            progress  = round((saved_f / target_f * 100), 1) if target_f else 0
            goals.append({
                "id":            gid,
                "goal_name":     gname,
                "target_amount": target_f,
                "saved_amount":  saved_f,
                "target_date":   target_date,
                "progress":      min(progress, 100),
            })
    except Exception as e:
        print(f"[DASHBOARD] Goals query failed: {e}")

    # Last 10 transactions for the activity feed
    cur.execute("""
        SELECT transaction_date, type, category, amount, description
        FROM transactions WHERE user_id=%s
        ORDER BY transaction_date DESC LIMIT 10
    """, (user_id,))
    transactions = [
        (r[0].strftime('%Y-%m-%d'), r[1], r[2], float(r[3]), r[4] or '')
        for r in cur.fetchall()
    ]

    # Spending forecast — train on the fly and grab all four horizons
    forecast_data = {
        "next_day": {"amount": 0, "label": "Tomorrow"},
        "weekly":   {"amount": 0, "label": "This Week"},
        "monthly":  {"amount": 0, "label": "This Month"},
        "yearly":   {"amount": 0, "label": "This Year"},
        "trend":    "No Trend",
        "available": False,
    }
    trend_label = "Spending Stable"
    trend_tone  = "neutral"

    try:
        forecasting.train_model(mysql, user_id)

        next_day_forecast = forecasting.next_day(mysql, user_id)
        weekly_forecast   = forecasting.weekly(mysql, user_id)
        monthly_forecast  = forecasting.monthly(mysql, user_id)
        yearly_forecast   = forecasting.yearly(mysql, user_id)
        trend             = forecasting.get_trend(mysql, user_id)

        forecast_data = {
            "next_day":  next_day_forecast,
            "weekly":    weekly_forecast,
            "monthly":   monthly_forecast,
            "yearly":    yearly_forecast,
            "trend":     trend,
            "available": True,
        }

        # Translate the trend string into a UI tone (warning / success / neutral)
        trend_lower = str(trend).lower()
        if "increas" in trend_lower or "up" in trend_lower:
            trend_label, trend_tone = "Spending Increasing", "warning"
        elif "decreas" in trend_lower or "down" in trend_lower:
            trend_label, trend_tone = "Spending Decreasing", "success"
        else:
            trend_label, trend_tone = "Spending Stable", "neutral"

    except Exception as e:
        print(f"[DASHBOARD] Forecasting failed: {e}")

    cur.close()

    # Handy date range strings the template uses for filter links
    month_name  = today.strftime('%B')
    year_str    = today.strftime('%Y')
    month_start = today.replace(day=1).strftime('%Y-%m-%d')
    last_day    = calendar.monthrange(today.year, today.month)[1]
    month_end   = today.replace(day=last_day).strftime('%Y-%m-%d')
    year_start  = f"{today.year}-01-01"
    year_end    = f"{today.year}-12-31"

    return render_template(
        "dashboard.html",
        username     = session.get("username", "User"),
        today        = today.strftime('%Y-%m-%d'),
        start_date   = first_date.strftime('%Y-%m-%d') if first_date else today.strftime('%Y-%m-%d'),
        days_tracked = days_tracked,

        daily_income  = f"{daily_income:,.2f}",
        daily_expense = f"{daily_expense:,.2f}",
        daily_balance = f"{daily_balance:,.2f}",

        monthly_income  = f"{monthly_income:,.2f}",
        monthly_expense = f"{monthly_expense:,.2f}",
        monthly_balance = f"{monthly_balance:,.2f}",

        yearly_income  = f"{yearly_income:,.2f}",
        yearly_expense = f"{yearly_expense:,.2f}",
        yearly_balance = f"{yearly_balance:,.2f}",

        total_income  = f"{total_income:,.2f}",
        total_expense = f"{total_expense:,.2f}",
        savings       = f"{savings:,.2f}",
        savings_rate  = f"{savings_rate:.1f}",

        top_category              = top_category,
        insight                   = insight,
        expense_category_analysis = expense_category_analysis,
        income_category_analysis  = income_category_analysis,

        budget_report = budget_report,
        goals         = goals,
        transactions  = transactions,

        current_month_label = f"{month_name} {year_str}",
        current_year_label  = year_str,
        month_start         = month_start,
        month_end           = month_end,
        year_start          = year_start,
        year_end            = year_end,

        forecast    = forecast_data,
        trend_label = trend_label,
        trend_tone  = trend_tone,
    )


# ==============================================================================
# Transactions
# ==============================================================================

@app.route('/add_income', methods=['GET', 'POST'])
@login_required
def add_income():
    if request.method == 'POST':
        description = request.form['description']
        amount      = request.form['amount']
        category    = request.form['category']

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (session['user_id'], 'income', category, amount, description))
        mysql.connection.commit()
        cur.close()

        # Append to the ML training dataset and retrain in the background
        append_to_dataset(description, amount, 'income', category)
        retrain_async(session['user_id'])

        return redirect(url_for('dashboard'))

    return render_template('add_income.html')


@app.route('/add_expense', methods=['GET', 'POST'])
@login_required
def add_expense():
    if request.method == 'POST':
        description = request.form['description']
        amount      = request.form['amount']
        category    = request.form['category']

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (session['user_id'], 'expense', category, amount, description))
        mysql.connection.commit()
        cur.close()

        # Same ML pipeline as add_income — keep argument order consistent
        append_to_dataset(description, amount, 'expense', category)
        retrain_async(session['user_id'])

        return redirect(url_for('dashboard'))

    return render_template('add_expense.html')


@app.route('/view_income_and_expense')
@login_required
def view_income_and_expense():
    # Pull all the filter parameters from the query string.
    # Every parameter is optional — unset ones default to empty string / None.
    search       = request.args.get('search', '').strip()
    exact_date   = request.args.get('exact_date', '').strip()    # e.g. 2025-06-15  — single day filter
    month        = request.args.get('month',  '').strip()        # e.g. 6            — month number 1-12
    year         = request.args.get('year',   '').strip()        # e.g. 2025          — pairs with month
    year_only    = request.args.get('year_only', '').strip()     # e.g. 2024          — standalone year filter
    created_date = request.args.get('created_date', '').strip()  # e.g. 2025-06-01  — filter by created_at

    user_id = session['user_id']
    cur     = mysql.connection.cursor()

    # We build the WHERE clause dynamically so each filter is independent.
    # Start with the mandatory user_id check, then append conditions.
    conditions = ["user_id = %s"]
    params     = [user_id]

    # Keyword search: matches category, type, or description
    if search:
        conditions.append("(category LIKE %s OR type LIKE %s OR description LIKE %s)")
        like = f"%{search}%"
        params.extend([like, like, like])

    # Exact day filter: matches the transaction_date column
    if exact_date:
        conditions.append("DATE(transaction_date) = %s")
        params.append(exact_date)

    # Month filter — can be used alone (any year) or combined with the year field
    if month:
        conditions.append("MONTH(transaction_date) = %s")
        params.append(int(month))
        if year:
            conditions.append("YEAR(transaction_date) = %s")
            params.append(int(year))

    # Standalone year filter (does not conflict with the month+year combo above)
    if year_only:
        conditions.append("YEAR(transaction_date) = %s")
        params.append(int(year_only))

    # created_date filter: matches the date part of the created_at timestamp
    if created_date:
        conditions.append("DATE(created_at) = %s")
        params.append(created_date)

    # Build the final SQL and run it
    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT id, user_id, type, category, amount, description,
               transaction_date, created_at, period
        FROM transactions
        WHERE {where_clause}
        ORDER BY transaction_date DESC
    """
    cur.execute(sql, tuple(params))
    transactions = cur.fetchall()
    cur.close()

    return render_template(
        'view_income_and_expense.html',
        transactions = transactions,
        username     = session.get('username', 'User'),
    )


@app.route('/edit_income_and_expense/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_income_and_expense(id):
    cur = mysql.connection.cursor()

    # Verify the transaction belongs to the logged-in user before doing anything
    cur.execute("SELECT * FROM transactions WHERE id=%s AND user_id=%s", (id, session['user_id']))
    transaction = cur.fetchone()
    if not transaction:
        cur.close()
        return "Transaction not found", 404

    if request.method == 'POST':
        cur.execute("""
            UPDATE transactions
            SET category=%s, amount=%s, description=%s, period=%s
            WHERE id=%s AND user_id=%s
        """, (
            request.form['category'], request.form['amount'],
            request.form['description'], request.form['period'],
            id, session['user_id'],
        ))
        mysql.connection.commit()
        cur.close()
        retrain_async(session['user_id'])
        return redirect(url_for('view_income_and_expense'))

    cur.close()
    return render_template('edit_income_and_expense.html', transaction=transaction)


@app.route('/delete_transaction/<int:id>')
@login_required
def delete_transaction(id):
    cur = mysql.connection.cursor()
    # The AND user_id guard prevents one user from deleting another's data
    cur.execute("DELETE FROM transactions WHERE id=%s AND user_id=%s", (id, session['user_id']))
    mysql.connection.commit()
    cur.close()
    retrain_async(session['user_id'])
    return redirect(url_for('view_income_and_expense'))


# ==============================================================================
# Profile
# ==============================================================================

@app.route('/profile')
@login_required
def profile():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT username, email, employee_type, job_title, monthly_salary, created_at
        FROM users WHERE id=%s
    """, (session['user_id'],))
    user = cur.fetchone()
    cur.close()
    return render_template('profile.html', user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        username      = request.form['username']
        employee_type = request.form['employee_type']

        # Clear job fields when the user switches away from Employee type
        if employee_type == 'Employee':
            job_title      = request.form.get('job_title')
            monthly_salary = request.form.get('monthly_salary') or None
        else:
            job_title      = None
            monthly_salary = None

        cur.execute("""
            UPDATE users
            SET username=%s, employee_type=%s, job_title=%s, monthly_salary=%s
            WHERE id=%s
        """, (username, employee_type, job_title, monthly_salary, session['user_id']))
        mysql.connection.commit()
        cur.close()
        session['username'] = username   # Keep the session in sync
        return redirect(url_for('profile'))

    cur.execute(
        "SELECT username, employee_type, job_title, monthly_salary FROM users WHERE id=%s",
        (session['user_id'],)
    )
    user = cur.fetchone()
    cur.close()
    return render_template('edit_profile.html', user=user)


# ==============================================================================
# Goals — all routes use the `goals` table
# ==============================================================================

@app.route('/add_goal', methods=['GET', 'POST'])
@login_required
def add_goal():
    if request.method == 'POST':
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO goals (user_id, goal_name, target_amount, current_amount, target_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            session['user_id'],
            request.form['goal_name'],
            float(request.form['target_amount']),
            float(request.form.get('current_amount', 0)),
            request.form.get('target_date') or None,
        ))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('view_goals'))

    return render_template('add_goal.html')


@app.route('/view_goals')
@login_required
def view_goals():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, goal_name, target_amount, current_amount, target_date
        FROM goals WHERE user_id=%s ORDER BY id DESC
    """, (session['user_id'],))
    goals = cur.fetchall()

    # Show overall savings so users can see how far they are from each goal
    cur.execute("""
        SELECT COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE -amount END), 0)
        FROM transactions WHERE user_id=%s
    """, (session['user_id'],))
    savings = cur.fetchone()[0]
    cur.close()

    return render_template('view_goals.html', goals=goals, savings=savings)


@app.route('/delete_goal/<int:id>')
@login_required
def delete_goal(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM goals WHERE id=%s AND user_id=%s", (id, session['user_id']))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('view_goals'))


@app.route('/edit_goal/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_goal(id):
    cur = mysql.connection.cursor()

    # Ownership check first — 404 is fine here since it's not sensitive info
    cur.execute("SELECT * FROM goals WHERE id=%s AND user_id=%s", (id, session['user_id']))
    goal = cur.fetchone()
    if not goal:
        cur.close()
        return "Goal not found", 404

    if request.method == 'POST':
        cur.execute("""
            UPDATE goals
            SET goal_name=%s, target_amount=%s, current_amount=%s, target_date=%s
            WHERE id=%s AND user_id=%s
        """, (
            request.form['goal_name'], request.form['target_amount'],
            request.form['current_amount'], request.form.get('target_date') or None,
            id, session['user_id'],
        ))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('view_goals'))

    cur.close()
    return render_template('edit_goal.html', goal=goal)


# ==============================================================================
# Budgets
# ==============================================================================

@app.route('/add_budget', methods=['GET', 'POST'])
@login_required
def add_budget():
    if request.method == 'POST':
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO budgets (user_id, category, amount, period) VALUES (%s, %s, %s, %s)",
            (session['user_id'], request.form['category'],
             request.form['amount'], request.form['period']),
        )
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('dashboard'))

    return render_template('add_budget.html')


@app.route('/edit_budget/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_budget(id):
    cur = mysql.connection.cursor()

    # Scope the lookup to the current user so they can't edit other users' budgets
    cur.execute(
        "SELECT id, category, amount, period FROM budgets WHERE id=%s AND user_id=%s",
        (id, session['user_id']),
    )
    budget = cur.fetchone()
    if not budget:
        cur.close()
        return "Budget not found", 404

    if request.method == 'POST':
        cur.execute(
            "UPDATE budgets SET category=%s, amount=%s, period=%s WHERE id=%s AND user_id=%s",
            (request.form['category'], request.form['amount'],
             request.form['period'], id, session['user_id']),
        )
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('dashboard'))

    cur.close()
    return render_template('edit_budget.html', budget=budget)


@app.route('/delete_budget/<int:id>')
@login_required
def delete_budget(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM budgets WHERE id=%s AND user_id=%s", (id, session['user_id']))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('dashboard'))


@app.route('/view_budgets')
@login_required
def view_budgets():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, category, amount, period FROM budgets WHERE user_id=%s", (session['user_id'],))
    budgets = cur.fetchall()

    # Compare budgets against the current month's actual spend only
    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
          AND YEAR(transaction_date)  = YEAR(CURDATE())
          AND MONTH(transaction_date) = MONTH(CURDATE())
        GROUP BY category
    """, (session['user_id'],))
    spent_map = {r[0]: float(r[1]) for r in cur.fetchall()}
    cur.close()

    budget_report = []
    for bid, cat, amt, period in budgets:
        budget_amount = float(amt)
        spent         = spent_map.get(cat, 0)
        remaining     = budget_amount - spent
        pct_used      = (spent / budget_amount * 100) if budget_amount else 0
        budget_report.append({
            "id": bid, "category": cat, "budget": budget_amount,
            "spent": spent, "remaining": remaining,
            "pct_used": round(pct_used, 2), "period": period,
        })

    return render_template('view_budgets.html', budget_report=budget_report)


# ==============================================================================
# Analytics
# ==============================================================================

@app.route('/analytics')
@login_required
def analytics():
    user_id = session['user_id']
    cur     = mysql.connection.cursor()

    # All-time income and expense totals
    cur.execute("SELECT type, SUM(amount) FROM transactions WHERE user_id=%s GROUP BY type", (user_id,))
    data          = dict(cur.fetchall() or [])
    total_income  = float(data.get('income',  0))
    total_expense = float(data.get('expense', 0))
    savings       = total_income - total_expense

    # Daily chart data — one row per (day, type) pair
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

    # Monthly chart data
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

    # Yearly chart data
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

    # Expense category breakdown (pie chart)
    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    cat_rows        = cur.fetchall()
    categories      = [r[0] for r in cat_rows]
    category_values = [float(r[1]) for r in cat_rows]

    # Budget vs actual (uses a LEFT JOIN so budgets with zero spend still appear)
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

    budget_report_raw = [
        {
            "category":  r[0],
            "budget":    float(r[1]),
            "spent":     float(r[2]),
            "remaining": float(r[1]) - float(r[2]),
            "pct_used":  round((float(r[2]) / float(r[1]) * 100) if r[1] else 0, 2),
            "period":    "monthly",
        }
        for r in budget_rows
    ]

    # Goals — unified with dashboard: always read from 'goals' table
    cur.execute("SELECT goal_name, target_amount FROM goals WHERE user_id=%s", (user_id,))
    goals         = cur.fetchall()
    goal_names    = [r[0] for r in goals]
    goal_progress = [
        min(round((savings / float(r[1]) * 100) if r[1] else 0, 2), 100)
        for r in goals
    ]

    cur.close()

    budget_alerts = calculate_budget_alerts(budget_report_raw)
    ai_insights   = generate_ai_insights(total_income, total_expense, savings, cat_rows, budget_report_raw)
    health        = calculate_health_score(total_income, total_expense, savings, budget_report_raw)

    # Forecast values — fall back to zeroes if the model isn't ready yet
    next_day_forecast = {"amount": 0, "ai_ready": False}
    weekly_forecast   = {"amount": 0, "ai_ready": False}
    monthly_forecast  = {"amount": 0, "ai_ready": False}
    yearly_forecast   = {"amount": 0, "ai_ready": False}
    trend_label       = "Spending Stable"

    try:
        next_day_forecast = forecasting.next_day(mysql, user_id)
        weekly_forecast   = forecasting.weekly(mysql, user_id)
        monthly_forecast  = forecasting.monthly(mysql, user_id)
        yearly_forecast   = forecasting.yearly(mysql, user_id)
        trend_label       = forecasting.get_trend(mysql, user_id)
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

        ai_insights      = ai_insights,
        budget_alerts    = budget_alerts,
        health           = health,

        next_day_forecast = next_day_forecast,
        weekly_forecast   = weekly_forecast,
        monthly_forecast  = monthly_forecast,
        yearly_forecast   = yearly_forecast,
        trend_label       = trend_label,
    )


# ==============================================================================
# Chatbot endpoints
# ==============================================================================

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please type something!"})

    user_id  = session['user_id']
    username = session.get('username', 'User')

    try:
        cur = mysql.connection.cursor()

        # Persist the user's message so chat history survives page reloads
        cur.execute(
            "INSERT INTO chatbot_messages (user_id, sender, message) VALUES (%s, %s, %s)",
            (user_id, "user", message)
        )
        mysql.connection.commit()

        reply = chatbot_engine.process_message(
            user_id=user_id,
            message=message,
            username=username,
        )

        # Persist the bot reply alongside the user message
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
@login_required
def add_transaction():
    # Used by the chatbot to insert transactions via natural language.
    data             = request.get_json()
    description      = data.get('description', '').strip()
    amount_raw       = data.get('amount')
    transaction_type = data.get('transaction_type', 'expense').strip().lower()

    if not description:
        return jsonify({"error": "Description is required"}), 400

    try:
        amount = float(amount_raw)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid amount"}), 400

    # Let the ML model predict the category from the description
    category = predict_category(description, transaction_type)

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO transactions
                (user_id, type, category, amount, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s, CURDATE())
        """, (session['user_id'], transaction_type, category, amount, description))
        mysql.connection.commit()
        cur.close()

        append_to_dataset(description, amount, transaction_type, category)
        retrain_async(session['user_id'])

        return jsonify({"message": "Transaction saved!", "category": category})

    except Exception as e:
        print(f"[ADD_TXN ERROR] {e}")
        return jsonify({"error": "Failed to save transaction"}), 500


@app.route('/load_chat')
@login_required
def load_chat():
    # Returns the full chat history for the current user in chronological order.
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT sender, message FROM chatbot_messages WHERE user_id=%s ORDER BY created_at ASC",
        (session['user_id'],)
    )
    rows = cur.fetchall()
    cur.close()
    return jsonify([{"sender": s, "message": m} for s, m in rows])


@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM chatbot_messages WHERE user_id=%s", (session['user_id'],))
    mysql.connection.commit()
    cur.close()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
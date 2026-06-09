# ==============================================================================
# SmartExpenseAI — Main Flask Application
# ==============================================================================

# --- Standard library ---
from datetime import date, datetime, timedelta
import os

# --- Third-party ---
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import requests

# --- Our own modules ---
from chatbot.chatbot_engine import ChatbotEngine
from machine_learning.predict_category import predict_category
from machine_learning.utils.dataset_sync import append_to_dataset
from machine_learning.forecasting import train_model
from machine_learning.forecasting import next_day, weekly, monthly, yearly, get_trend

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

app.config['MYSQL_HOST']    = MYSQL_HOST
app.config['MYSQL_PORT']    = MYSQL_PORT
app.config['MYSQL_USER']    = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB']      = MYSQL_DB
app.config['MYSQL_CHARSET'] = 'utf8mb4'

mysql = MySQL(app)

# Pass mysql AND the openrouter caller into the engine so it can use AI responses
chatbot_engine = ChatbotEngine(mysql, openrouter_fn=None)  # wired below after definition


# ==============================================================================
# Helper — Date Normaliser
# ==============================================================================

def safe_date(value):
    """Converts any date-like value into a YYYY-MM-DD string."""
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
# OpenRouter API — AI Assistant
# ==============================================================================

def call_openrouter(user_message: str, intent: str = "unknown") -> str:
    """
    Sends a message to OpenRouter and returns an AI-generated reply.
    Returns None on failure so callers can fall back gracefully.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        # No key configured — return a static fallback keyed by intent
        fallbacks = {
            "financial_advice": (
                "Try the 50/30/20 rule: 50% on needs, 30% on wants, 20% into savings. "
                "Even saving ₹500 a month compounds significantly over time."
            ),
            "greeting": (
                "Hello! I'm SmartExpenseAI. Try 'add income', 'show summary', "
                "or 'set food budget 5000 monthly'."
            ),
            "unknown": (
                "I can help you track income and expenses, set budgets and goals, "
                "and give financial tips. Try 'show analytics' to get started."
            ),
        }
        return fallbacks.get(intent, fallbacks["unknown"])

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

        # API error — log the status so it's visible in console
        print(f"[OPENROUTER] HTTP {response.status_code}: {response.text[:200]}")
        return None

    except requests.exceptions.Timeout:
        print("[OPENROUTER] Request timed out.")
        return None

    except Exception as e:
        print(f"[OPENROUTER] Unexpected error: {e}")
        return None


# Wire the openrouter function into the already-created chatbot engine
chatbot_engine.openrouter_fn = call_openrouter


# ==============================================================================
# Analytics Intelligence Helpers
# ==============================================================================

def generate_ai_insights(total_income, total_expense, savings,
                         expense_cats, budget_report):
    """
    Returns a list of plain-English insight dicts based on the user's
    financial data. Each dict has: text, type, icon.
    """
    insights = []
    spending_ratio = (total_expense / total_income * 100) if total_income else 0

    # Income vs expense ratio
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

    # Savings insight
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

    # Top spending category
    if expense_cats:
        top_cat, top_amt = expense_cats[0]
        insights.append({
            "text": f"{top_cat.title()} is your highest spending category at ₹{float(top_amt):,.2f}.",
            "type": "info", "icon": "📊"
        })

    # Budget overspending
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

    if not insights:
        insights.append({
            "text": "Add more transactions to unlock AI insights.",
            "type": "info", "icon": "💡"
        })

    return insights


def calculate_budget_alerts(budget_report):
    """
    Enriches each budget_report dict with an alert_level key:
      'safe' < 70% | 'warning' 70-89% | 'danger' 90%+
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


def calculate_health_score(total_income, total_expense, savings, budget_report):
    """
    Returns a financial health score (0-100) and a status label.

    Breakdown:
      40 pts — expense ratio  (lower = more points)
      30 pts — savings amount (positive = full points)
      30 pts — budget control (no overruns = full points)
    """
    score = 0

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

    if savings > 0:
        score += 30
    elif savings == 0:
        score += 10

    if budget_report:
        safe_count = sum(1 for b in budget_report if b["pct_used"] < 90)
        score += int((safe_count / len(budget_report)) * 30)
    else:
        score += 15

    if score >= 90:
        status, color = "Excellent", "excellent"
    elif score >= 70:
        status, color = "Good", "good"
    elif score >= 50:
        status, color = "Moderate", "moderate"
    else:
        status, color = "Risky", "risky"

    return {"score": score, "status": status, "color": color}


# ==============================================================================
# DB Test
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
# Core Page Routes
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    result = cur.fetchall()
    return str(result)


# ==============================================================================
# Authentication Routes
# ==============================================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username      = request.form['username']
        employee_type = request.form['employee_type']
        email         = request.form['email']
        password      = request.form['password']

        if employee_type == 'Employee':
            job_title      = request.form.get('job_title')
            monthly_salary = request.form.get('monthly_salary')
            if not monthly_salary or monthly_salary.strip() == '':
                monthly_salary = 0
            else:
                monthly_salary = float(monthly_salary)
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
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            stored_hash = user[3]   # password column

            if check_password_hash(stored_hash, password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('dashboard'))

            return "Invalid Password"

        return "User Not Found"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ==============================================================================
# Dashboard Route
# ==============================================================================

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    import calendar
    from machine_learning import forecasting

    cur     = mysql.connection.cursor()
    user_id = session['user_id']
    today   = date.today()

    # ----------------------------------------------------------
    # Days tracked
    # ----------------------------------------------------------
    cur.execute("""
        SELECT MIN(transaction_date)
        FROM transactions
        WHERE user_id=%s
    """, (user_id,))
    first_date   = cur.fetchone()[0]
    days_tracked = (today - first_date).days if first_date else 0

    # ----------------------------------------------------------
    # Daily totals
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Monthly totals
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

    # ----------------------------------------------------------
    # Yearly totals
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # All-time totals
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
    savings       = total_income - total_expense
    savings_rate  = (savings / total_income * 100) if total_income else 0

    # ----------------------------------------------------------
    # Category analysis
    # ----------------------------------------------------------
    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    expense_cats              = cur.fetchall()
    expense_category_analysis = [(cat, float(amt)) for cat, amt in expense_cats]
    top_category              = expense_cats[0] if expense_cats else ('None', 0)

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='income'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    income_cats              = cur.fetchall()
    income_category_analysis = [(cat, float(amt)) for cat, amt in income_cats]

    # ----------------------------------------------------------
    # Savings insight label
    # ----------------------------------------------------------
    if savings_rate >= 30:
        insight = "Excellent savings rate!"
    elif savings_rate >= 20:
        insight = "Good savings habit!"
    elif savings_rate >= 10:
        insight = "Try to save more money."
    elif savings_rate > 0:
        insight = "Reduce unnecessary expenses."
    else:
        insight = "Your expenses exceed income."

    # ----------------------------------------------------------
    # Budgets
    # ----------------------------------------------------------
    cur.execute("""
        SELECT id, category, amount, period
        FROM budgets WHERE user_id=%s
    """, (user_id,))
    budgets = cur.fetchall()

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category
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

    # ----------------------------------------------------------
    # Goals — query the correct table (financial_goals)
    # ----------------------------------------------------------
    goals = []
    try:
        cur.execute("""
            SELECT id, goal_name, target_amount, saved_amount
            FROM financial_goals
            WHERE user_id=%s ORDER BY id DESC
        """, (user_id,))
        for gid, gname, target, saved in cur.fetchall():
            target_f = float(target or 0)
            saved_f  = float(saved  or 0)
            progress = round((saved_f / target_f * 100), 1) if target_f else 0
            goals.append({
                "id": gid, "goal_name": gname,
                "target_amount": target_f, "saved_amount": saved_f,
                "progress": min(progress, 100),
            })
    except Exception as e:
        print(f"[DASHBOARD] Goals query failed: {e}")

    # ----------------------------------------------------------
    # Recent transactions
    # ----------------------------------------------------------
    cur.execute("""
        SELECT transaction_date, type, category, amount, description
        FROM transactions
        WHERE user_id=%s
        ORDER BY transaction_date DESC LIMIT 10
    """, (user_id,))
    transactions = [
        (r[0].strftime('%Y-%m-%d'), r[1], r[2], float(r[3]), r[4] or '')
        for r in cur.fetchall()
    ]

    # ----------------------------------------------------------
    # AI Forecasting
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Date labels
    # ----------------------------------------------------------
    month_name  = today.strftime('%B')
    year_str    = today.strftime('%Y')
    month_start = today.replace(day=1).strftime('%Y-%m-%d')
    last_day    = calendar.monthrange(today.year, today.month)[1]
    month_end   = today.replace(day=last_day).strftime('%Y-%m-%d')
    year_start  = f"{today.year}-01-01"
    year_end    = f"{today.year}-12-31"

    current_month_label = f"{month_name} {year_str}"
    current_year_label  = year_str

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

        current_month_label = current_month_label,
        current_year_label  = current_year_label,
        month_start         = month_start,
        month_end           = month_end,
        year_start          = year_start,
        year_end            = year_end,

        forecast    = forecast_data,
        trend_label = trend_label,
        trend_tone  = trend_tone,
    )


# ==============================================================================
# Transaction Routes
# ==============================================================================

@app.route('/add_income', methods=['GET', 'POST'])
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

        append_to_dataset(description, amount, 'income', category)
        return redirect(url_for('dashboard'))

    return render_template('add_income.html')


@app.route('/add_expense', methods=['GET', 'POST'])
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

        # FIX: was hardcoded 'income' — now correctly passes 'expense'
        append_to_dataset(description, amount, 'expense', category)
        return redirect(url_for('dashboard'))

    return render_template('add_expense.html')


@app.route('/view_income_and_expense')
def view_income_and_expense():
    search = request.args.get('search', '')
    cur    = mysql.connection.cursor()

    if search:
        cur.execute("""
            SELECT * FROM transactions
            WHERE user_id=%s
              AND (category LIKE %s OR type LIKE %s OR period LIKE %s)
        """, (session['user_id'], f'%{search}%', f'%{search}%', f'%{search}%'))
    else:
        cur.execute(
            "SELECT * FROM transactions WHERE user_id=%s",
            (session['user_id'],)
        )

    transactions = cur.fetchall()
    return render_template('view_income_and_expense.html', transactions=transactions)


@app.route('/edit_income_and_expense/<int:id>', methods=['GET', 'POST'])
def edit_income_and_expense(id):
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        cur.execute("""
            UPDATE transactions
            SET category=%s, amount=%s, description=%s, period=%s
            WHERE id=%s
        """, (
            request.form['category'], request.form['amount'],
            request.form['description'], request.form['period'], id,
        ))
        mysql.connection.commit()
        return redirect(url_for('view_income_and_expense'))

    cur.execute("SELECT * FROM transactions WHERE id=%s", (id,))
    transaction = cur.fetchone()
    return render_template('edit_income_and_expense.html', transaction=transaction)


@app.route('/delete_transaction/<int:id>')
def delete_transaction(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM transactions WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for('view_income_and_expense'))


# ==============================================================================
# Profile Routes
# ==============================================================================

@app.route('/profile')
def profile():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT username, email, employee_type, job_title, monthly_salary, created_at
        FROM users WHERE id=%s
    """, (session['user_id'],))
    user = cur.fetchone()
    return render_template('profile.html', user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        username      = request.form['username']
        employee_type = request.form['employee_type']

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
        session['username'] = username
        return redirect(url_for('profile'))

    cur.execute(
        "SELECT username, employee_type, job_title, monthly_salary FROM users WHERE id=%s",
        (session['user_id'],)
    )
    user = cur.fetchone()
    return render_template('edit_profile.html', user=user)


# ==============================================================================
# Goals Routes
# ==============================================================================

@app.route('/add_goal', methods=['GET', 'POST'])
def add_goal():
    if 'user_id' not in session:
        return redirect(url_for('login'))

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
def view_goals():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, goal_name, target_amount, current_amount, target_date
        FROM goals WHERE user_id=%s
    """, (session['user_id'],))
    goals = cur.fetchall()

    cur.execute("""
        SELECT COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE -amount END), 0)
        FROM transactions WHERE user_id=%s
    """, (session['user_id'],))
    savings = cur.fetchone()[0]

    cur.close()
    return render_template('view_goals.html', goals=goals, savings=savings)


@app.route('/delete_goal/<int:id>')
def delete_goal(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM goals WHERE id=%s AND user_id=%s", (id, session['user_id']))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('view_goals'))


@app.route('/edit_goal/<int:id>', methods=['GET', 'POST'])
def edit_goal(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        cur.execute("""
            UPDATE goals
            SET goal_name=%s, target_amount=%s, current_amount=%s, target_date=%s
            WHERE id=%s AND user_id=%s
        """, (
            request.form['goal_name'], request.form['target_amount'],
            request.form['current_amount'], request.form['target_date'],
            id, session['user_id'],
        ))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('view_goals'))

    cur.execute("SELECT * FROM goals WHERE id=%s AND user_id=%s", (id, session['user_id']))
    goal = cur.fetchone()
    cur.close()
    return render_template('edit_goal.html', goal=goal)


# ==============================================================================
# Budget Routes
# ==============================================================================

@app.route('/add_budget', methods=['GET', 'POST'])
def add_budget():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO budgets (user_id, category, amount, period) VALUES (%s, %s, %s, %s)",
            (session['user_id'], request.form['category'],
             request.form['amount'], request.form['period']),
        )
        mysql.connection.commit()
        return redirect(url_for('dashboard'))

    return render_template('add_budget.html')


@app.route('/edit_budget/<int:id>', methods=['GET', 'POST'])
def edit_budget(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        cur.execute(
            "UPDATE budgets SET category=%s, amount=%s, period=%s WHERE id=%s AND user_id=%s",
            (request.form['category'], request.form['amount'],
             request.form['period'], id, session['user_id']),
        )
        mysql.connection.commit()
        return redirect(url_for('dashboard'))

    cur.execute(
        "SELECT id, category, amount, period FROM budgets WHERE id=%s AND user_id=%s",
        (id, session['user_id']),
    )
    budget = cur.fetchone()

    if not budget:
        return "Budget not found", 404

    return render_template('edit_budget.html', budget=budget)


@app.route('/delete_budget/<int:id>')
def delete_budget(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM budgets WHERE id=%s AND user_id=%s", (id, session['user_id']))
    mysql.connection.commit()
    return redirect(url_for('dashboard'))


@app.route('/view_budgets')
def view_budgets():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT id, category, amount, period FROM budgets WHERE user_id=%s",
        (session['user_id'],),
    )
    budgets = cur.fetchall()

    cur.execute("""
        SELECT category, SUM(amount)
        FROM transactions
        WHERE user_id=%s AND type='expense'
          AND YEAR(transaction_date)  = YEAR(CURDATE())
          AND MONTH(transaction_date) = MONTH(CURDATE())
        GROUP BY category
    """, (session['user_id'],))
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

    return render_template('view_budgets.html', budget_report=budget_report)


# ==============================================================================
# Analytics Route
# ==============================================================================

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur     = mysql.connection.cursor()

    # Overall totals
    cur.execute(
        "SELECT type, SUM(amount) FROM transactions WHERE user_id=%s GROUP BY type",
        (user_id,)
    )
    data          = dict(cur.fetchall() or [])
    total_income  = float(data.get('income',  0))
    total_expense = float(data.get('expense', 0))
    savings       = total_income - total_expense

    # Daily breakdown
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

    # Monthly breakdown
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

    # Yearly breakdown
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

    # Category breakdown
    cur.execute("""
        SELECT category, SUM(amount) FROM transactions
        WHERE user_id=%s AND type='expense'
        GROUP BY category ORDER BY SUM(amount) DESC
    """, (user_id,))
    cat_rows        = cur.fetchall()
    categories      = [r[0] for r in cat_rows]
    category_values = [float(r[1]) for r in cat_rows]

    # Budget vs actual
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

    # Goals — FIX: use the same 'goals' table consistently in analytics
    # (if your main table is financial_goals, change this to financial_goals)
    cur.execute("SELECT goal_name, target_amount FROM goals WHERE user_id=%s", (user_id,))
    goals         = cur.fetchall()
    goal_names    = [r[0] for r in goals]
    goal_progress = [
        round((savings / float(r[1]) * 100) if r[1] else 0, 2)
        for r in goals
    ]

    cur.close()

    # AI Intelligence
    budget_alerts = calculate_budget_alerts(budget_report_raw)
    ai_insights   = generate_ai_insights(
        total_income, total_expense, savings, cat_rows, budget_report_raw
    )
    health = calculate_health_score(
        total_income, total_expense, savings, budget_report_raw
    )

    # ML Forecasting
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

        ai_insights      = ai_insights,
        budget_alerts    = budget_alerts,
        health           = health,
        weekly_forecast  = weekly_forecast,
        monthly_forecast = monthly_forecast,
    )


# ==============================================================================
# Chatbot Endpoints
# ==============================================================================

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'user_id' not in session:
        return jsonify({"reply": "Please log in first."})

    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please type something!"})

    user_id  = session['user_id']
    username = session.get('username', 'User')

    try:
        cur = mysql.connection.cursor()

        cur.execute(
            "INSERT INTO chatbot_messages (user_id, sender, message) VALUES (%s, %s, %s)",
            (user_id, "user", message)
        )
        mysql.connection.commit()

        reply = chatbot_engine.process_message(
            user_id  = user_id,
            message  = message,
            username = username,
        )

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
    if 'user_id' not in session:
        return jsonify({"error": "Login required"}), 401

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

        append_to_dataset(description, transaction_type, category)

        return jsonify({"message": "Transaction saved!", "category": category})

    except Exception as e:
        print(f"[ADD_TXN ERROR] {e}")
        return jsonify({"error": "Failed to save transaction"}), 500


@app.route('/load_chat')
def load_chat():
    if 'user_id' not in session:
        return jsonify([])

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT sender, message FROM chatbot_messages WHERE user_id=%s ORDER BY created_at ASC",
        (session['user_id'],)
    )
    rows = cur.fetchall()
    cur.close()
    return jsonify([{"sender": s, "message": m} for s, m in rows])


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    if 'user_id' not in session:
        return jsonify({"success": False})

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM chatbot_messages WHERE user_id=%s", (session['user_id'],))
    mysql.connection.commit()
    cur.close()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
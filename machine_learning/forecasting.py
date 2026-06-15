# Main forecasting module for SmartExpenseAI
# Handles everything from loading data to training models and making predictions
# All sections are clearly separated so anyone can follow the flow easily
# machine_learning/forecasting.py

import pandas as pd
import numpy as np
import joblib

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os


# SECTION 1 — Environment setup
# Ensures Python can find all project files regardless of where this script is called from

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)
sys.path.append(PROJECT_ROOT)

# All trained models are saved inside machine_learning/models to keep things tidy
MODEL_DIR = os.path.join(CURRENT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# SECTION 2 — Model file path helpers
# Centralising paths here means we never hardcode them more than once

def get_model_path(user_id):
    # Legacy combined model path kept for backward compatibility
    return os.path.join(MODEL_DIR, f"forecast_model_user_{user_id}.pkl")


def _get_daily_model_path(user_id):
    return os.path.join(MODEL_DIR, f"daily_model_user_{user_id}.pkl")


def _get_monthly_model_path(user_id):
    return os.path.join(MODEL_DIR, f"monthly_model_user_{user_id}.pkl")


def _get_yearly_model_path(user_id):
    return os.path.join(MODEL_DIR, f"yearly_model_user_{user_id}.pkl")


# SECTION 3 — Database loader
# Pulls raw expense transactions for a given user; everything downstream depends on this

def load_user_transactions(mysql, user_id):
    # Returns a clean DataFrame or an empty one if the query fails or returns nothing
    try:
        cur = mysql.connection.cursor()

        cur.execute("""
            SELECT
                amount,
                category,
                type,
                transaction_date
            FROM transactions
            WHERE user_id=%s
            AND type='expense'
            ORDER BY transaction_date ASC
        """, (user_id,))

        rows = cur.fetchall()
        cur.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "amount",
            "category",
            "type",
            "transaction_date"
        ])

        return df

    except Exception as e:
        return pd.DataFrame()


# SECTION 4 — Preprocessing and feature engineering
# Separate pipelines for daily, monthly, and yearly so each is fully independent

def _clean_transactions(df):
    # Shared cleaner used by all three aggregation pipelines
    # Handles date parsing, amount validation, negative removal, outlier removal, and feature extraction
    if df.empty:
        return df

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    )
    df = df.dropna(subset=["transaction_date"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df = df[df["amount"] > 0]

    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    df = df[df["amount"] <= upper_limit]

    df["day"] = df["transaction_date"].dt.day
    df["month"] = df["transaction_date"].dt.month
    df["year"] = df["transaction_date"].dt.year
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["week_number"] = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    return df


def _prepare_daily_data(mysql, user_id):
    # Loads and cleans transactions, then collapses to one row per calendar day
    raw = load_user_transactions(mysql, user_id)
    df = _clean_transactions(raw)

    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby(df["transaction_date"].dt.date)["amount"]
        .sum()
        .reset_index()
    )
    daily.columns = ["date", "total"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["week_number"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["is_weekend"] = daily["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    daily["day_index"] = np.arange(len(daily))

    return daily


def _prepare_monthly_data(mysql, user_id):
    # Loads and cleans transactions, then collapses to one row per calendar month
    raw = load_user_transactions(mysql, user_id)
    df = _clean_transactions(raw)

    if df.empty:
        return pd.DataFrame()

    df["year_month"] = df["transaction_date"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")["amount"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["year_month", "total"]
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    monthly["year"] = monthly["year_month"].dt.year
    monthly["month"] = monthly["year_month"].dt.month
    monthly["month_index"] = np.arange(len(monthly))

    return monthly


def _prepare_yearly_data(mysql, user_id):
    # Loads and cleans transactions, then collapses to one row per calendar year
    raw = load_user_transactions(mysql, user_id)
    df = _clean_transactions(raw)

    if df.empty:
        return pd.DataFrame()

    df["year"] = df["transaction_date"].dt.year
    yearly = (
        df.groupby("year")["amount"]
        .sum()
        .reset_index()
    )
    yearly.columns = ["year", "total"]
    yearly = yearly.sort_values("year").reset_index(drop=True)
    yearly["year_index"] = np.arange(len(yearly))

    return yearly


def prepare_data(mysql, user_id):
    # Legacy version kept for backward compatibility with get_model and get_trend
    # Returns daily aggregated data using only day_index as a feature

    df = load_user_transactions(mysql, user_id)

    if df.empty:
        return None

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df = df[df["amount"] > 0]

    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    df = df[df["amount"] <= upper_limit]

    daily = df.groupby(df["transaction_date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "total"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["day_index"] = np.arange(len(daily))

    return daily


# SECTION 5 — Evaluation helper
# Computes and returns standard regression metrics after every model training run

def _evaluate(y_test, y_pred, label=""):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# SECTION 6 — Model training, saving, loading, and deletion
# Separate train functions for daily, monthly, and yearly plus the legacy combined one

def train_daily_model(user_id, mysql=None):
    # Trains a LinearRegression model on daily aggregated expense data
    # Requires at least 7 unique spending days to be worth training on
    if mysql is None:
        return None, None, None

    daily = _prepare_daily_data(mysql, user_id)

    if daily.empty or len(daily) < 7:
        return None, None, daily if not daily.empty else None

    FEATURES = [
        "day", "month", "year",
        "day_of_week", "week_number", "is_weekend",
        "day_index"
    ]

    X = daily[FEATURES]
    y = daily["total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = _evaluate(y_test, y_pred, label="DAILY")

    path = _get_daily_model_path(user_id)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, path)

    return model, metrics, daily


def train_monthly_model(user_id, mysql=None):
    # Trains a LinearRegression model on monthly aggregated expense data
    # Needs at least 3 months; uses the same data for train and test if fewer than 5 months exist
    if mysql is None:
        return None, None, None

    monthly = _prepare_monthly_data(mysql, user_id)

    if monthly.empty or len(monthly) < 3:
        return None, None, monthly if not monthly.empty else None

    FEATURES = ["month", "year", "month_index"]

    X = monthly[FEATURES]
    y = monthly["total"]

    if len(monthly) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test,  y_test  = X, y

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = _evaluate(y_test, y_pred, label="MONTHLY")

    path = _get_monthly_model_path(user_id)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, path)

    return model, metrics, monthly


def train_yearly_model(user_id, mysql=None):
    # Trains a LinearRegression model on yearly aggregated expense data
    # Trains on all available years without splitting because the dataset is very small
    if mysql is None:
        return None, None, None

    yearly = _prepare_yearly_data(mysql, user_id)

    if yearly.empty or len(yearly) < 2:
        return None, None, yearly if not yearly.empty else None

    FEATURES = ["year", "year_index"]

    X = yearly[FEATURES]
    y = yearly["total"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred  = model.predict(X)
    metrics = _evaluate(y, y_pred, label="YEARLY")

    path = _get_yearly_model_path(user_id)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, path)

    return model, metrics, yearly


def train_model(mysql, user_id):
    # Legacy training function that uses only day_index as the feature
    # Kept because get_trend and get_model still depend on it

    daily = prepare_data(mysql, user_id)

    if daily is None:
        return None, None

    if len(daily) < 7:
        return None, daily

    X = daily[["day_index"]]
    y = daily["total"]

    model = LinearRegression()
    model.fit(X, y)

    save_model(model, user_id)

    return model, daily


def save_model(model, user_id):
    # Saves the legacy combined model to disk using joblib
    path = get_model_path(user_id)
    try:
        joblib.dump(model, path)
    except Exception as e:
        pass


def load_model(user_id):
    # Loads the legacy combined model from disk if it exists
    path = get_model_path(user_id)

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)
    except Exception as e:
        return None


def delete_model(user_id):
    # Deletes all saved model files for a user — legacy and all three granular ones
    path = get_model_path(user_id)

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            pass

    for p in [
        _get_daily_model_path(user_id),
        _get_monthly_model_path(user_id),
        _get_yearly_model_path(user_id),
    ]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                pass


def get_model(mysql, user_id):
    # Legacy model loader — tries disk first and falls back to retraining
    # Returns (model, daily_data) for use by get_trend and predict
    saved = load_model(user_id)
    daily = prepare_data(mysql, user_id)

    if saved is not None and daily is not None and len(daily) >= 7:
        return saved, daily

    return train_model(mysql, user_id)


def retrain_all_models(user_id, mysql):
    # Retrains every model for this user at once
    # Called from the Flask route that handles adding a new transaction
    train_daily_model(user_id, mysql)
    train_monthly_model(user_id, mysql)
    train_yearly_model(user_id, mysql)
    train_model(mysql, user_id)


# SECTION 7 — Prediction functions
# Each one loads a saved model from disk, falls back to training if needed,
# and clamps the raw output to a sane range before returning

def predict_tomorrow_expense(user_id, mysql=None):
    # Predicts what the user will spend tomorrow using the daily model
    if mysql is None:
        return {
            "amount": 0, "message": "Database connection required.",
            "ai_ready": False, "metrics": None
        }

    path  = _get_daily_model_path(user_id)
    daily = _prepare_daily_data(mysql, user_id)

    if os.path.exists(path):
        bundle  = joblib.load(path)
        model   = bundle["model"]
        metrics = bundle.get("metrics")
    else:
        model, metrics, daily = train_daily_model(user_id, mysql)

    if daily is None or daily.empty:
        return {
            "amount": 0, "message": "No expense data available yet.",
            "ai_ready": False, "metrics": None
        }

    if model is None:
        avg = round(daily["total"].mean(), 2) if len(daily) > 0 else 0
        return {
            "amount": avg,
            "message": f"Estimated tomorrow's expense: {avg}. AI is still learning (need 7+ days).",
            "ai_ready": False, "metrics": None
        }

    tomorrow = datetime.today() + timedelta(days=1)

    X_pred = pd.DataFrame([{
        "day":         tomorrow.day,
        "month":       tomorrow.month,
        "year":        tomorrow.year,
        "day_of_week": tomorrow.weekday(),
        "week_number": int(tomorrow.strftime("%W")),
        "is_weekend":  1 if tomorrow.weekday() >= 5 else 0,
        "day_index":   len(daily)
    }])

    raw_pred   = model.predict(X_pred)[0]
    recent_avg = daily.tail(7)["total"].mean()
    prediction = float(min(max(raw_pred, 0), recent_avg * 1.5))
    prediction = round(prediction, 2)

    return {
        "amount":   prediction,
        "message":  f"You may spend approximately {prediction} tomorrow.",
        "ai_ready": True,
        "metrics":  metrics
    }


def predict_next_month_expense(user_id, mysql=None):
    # Predicts the total expense for next calendar month using the monthly model
    # Handles December-to-January year rollover correctly
    if mysql is None:
        return {
            "amount": 0, "message": "Database connection required.",
            "ai_ready": False, "metrics": None
        }

    path    = _get_monthly_model_path(user_id)
    monthly = _prepare_monthly_data(mysql, user_id)

    if os.path.exists(path):
        bundle  = joblib.load(path)
        model   = bundle["model"]
        metrics = bundle.get("metrics")
    else:
        model, metrics, monthly = train_monthly_model(user_id, mysql)

    if monthly is None or monthly.empty:
        return {
            "amount": 0, "message": "No expense data available yet.",
            "ai_ready": False, "metrics": None
        }

    if model is None:
        avg = round(monthly["total"].mean(), 2) if len(monthly) > 0 else 0
        return {
            "amount": avg,
            "message": f"Estimated next month's expense: {avg}. AI needs more data (3+ months).",
            "ai_ready": False, "metrics": None
        }

    today = datetime.today()
    if today.month == 12:
        next_month_month = 1
        next_month_year  = today.year + 1
    else:
        next_month_month = today.month + 1
        next_month_year  = today.year

    X_pred = pd.DataFrame([{
        "month":       next_month_month,
        "year":        next_month_year,
        "month_index": len(monthly)
    }])

    raw_pred   = model.predict(X_pred)[0]
    recent_avg = monthly.tail(3)["total"].mean()
    prediction = float(min(max(raw_pred, 0), recent_avg * 1.5))
    prediction = round(prediction, 2)

    return {
        "amount":   prediction,
        "message":  f"You may spend approximately {prediction} next month.",
        "ai_ready": True,
        "metrics":  metrics
    }


def predict_next_year_expense(user_id, mysql=None):
    # Predicts the total expense for next calendar year using the yearly model
    # Falls back to 12× the monthly average when there are fewer than 2 years of data
    if mysql is None:
        return {
            "amount": 0, "message": "Database connection required.",
            "ai_ready": False, "metrics": None
        }

    path   = _get_yearly_model_path(user_id)
    yearly = _prepare_yearly_data(mysql, user_id)

    if os.path.exists(path):
        bundle  = joblib.load(path)
        model   = bundle["model"]
        metrics = bundle.get("metrics")
    else:
        model, metrics, yearly = train_yearly_model(user_id, mysql)

    if yearly is None or yearly.empty:
        return {
            "amount": 0, "message": "No expense data available yet.",
            "ai_ready": False, "metrics": None
        }

    if model is None:
        monthly = _prepare_monthly_data(mysql, user_id)
        avg = round(monthly["total"].mean() * 12, 2) if (monthly is not None and not monthly.empty) else 0
        return {
            "amount": avg,
            "message": f"Estimated next year's expense: {avg}. AI needs 2+ years of data.",
            "ai_ready": False, "metrics": None
        }

    X_pred = pd.DataFrame([{
        "year":       datetime.today().year + 1,
        "year_index": len(yearly)
    }])

    raw_pred   = model.predict(X_pred)[0]
    recent_avg = yearly.tail(2)["total"].mean()
    prediction = float(min(max(raw_pred, 0), recent_avg * 1.5))
    prediction = round(prediction, 2)

    return {
        "amount":   prediction,
        "message":  f"You may spend approximately {prediction} next year.",
        "ai_ready": True,
        "metrics":  metrics
    }


def predict(mysql, user_id, days):
    # Legacy prediction function used by the convenience wrappers below
    # Takes a number of days and returns an estimated total spend over that period
    model, daily = get_model(mysql, user_id)

    if daily is None:
        return {
            "amount": 0, "transactions": 0,
            "message": "No expense data available yet.",
            "ai_ready": False
        }

    if len(daily) < 7:
        # Not enough data yet — fall back to a simple daily average scaled to the window
        recent_avg       = daily["total"].mean()
        prediction       = round(recent_avg * days, 2)
        avg_transactions = max(1, round(days * 0.8))

        if days >= 365:
            return {
                "amount": 0, "transactions": 0,
                "message": "Yearly forecast available after 30 days of usage.",
                "ai_ready": False
            }

        return {
            "amount":       prediction,
            "transactions": avg_transactions,
            "message":      (
                f"Estimated spending: {prediction} "
                f"for next {days} days. "
                f"AI is still learning your habits."
            ),
            "ai_ready": False
        }

    future_index = len(daily) + days
    future       = pd.DataFrame({"day_index": [future_index]})
    prediction   = model.predict(future)[0]

    recent_avg   = daily.tail(7)["total"].mean()
    max_limit    = recent_avg * 1.5 * days
    prediction   = min(prediction * days, max_limit)
    prediction   = max(0, prediction)
    prediction   = round(prediction, 2)

    avg_transactions = max(1, round(days * 0.8))

    return {
        "amount":       prediction,
        "transactions": avg_transactions,
        "message":      (
            f"You may spend approximately "
            f"{prediction} within the next "
            f"{days} days."
        ),
        "ai_ready": True
    }


# SECTION 8 — Trend analysis
# Translates the slope of the legacy model into a human-readable label

def get_trend(mysql, user_id):
    # A positive slope means spending is rising; negative means it is falling
    model, daily = get_model(mysql, user_id)

    if model is None:
        return "AI Learning"

    slope = model.coef_[0]

    if slope > 50:
        return "Spending Increasing"
    elif slope < -50:
        return "Spending Decreasing"
    else:
        return "Spending Stable"


# SECTION 9 — Convenience wrappers
# Give callers clean named functions instead of raw day counts

def next_day(mysql, user_id):
    return predict(mysql, user_id, 1)


def weekly(mysql, user_id):
    return predict(mysql, user_id, 7)


def monthly(mysql, user_id):
    return predict(mysql, user_id, 30)


def yearly(mysql, user_id):
    return predict(mysql, user_id, 365)


# SECTION 11 — Manual test block
# Only runs when this file is executed directly, never when imported

if __name__ == "__main__":
    from app import app, mysql

    TEST_USER_ID = 16

    with app.app_context():
        print(predict_tomorrow_expense(TEST_USER_ID, mysql))
        print(predict_next_month_expense(TEST_USER_ID, mysql))
        print(predict_next_year_expense(TEST_USER_ID, mysql))
        print(next_day(mysql, TEST_USER_ID))
        print(weekly(mysql, TEST_USER_ID))
        print(monthly(mysql, TEST_USER_ID))
        print(yearly(mysql, TEST_USER_ID))
        print(get_trend(mysql, TEST_USER_ID))
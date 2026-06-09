# ======================================
# machine_learning/forecasting.py
# Personalized AI Forecasting System
# SmartExpenseAI
# ======================================
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
import sys
import os

# ==========================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

sys.path.append(PROJECT_ROOT)

# ==========================================
# MODEL STORAGE DIRECTORY
# Resolves to:
#   machine_learning/models/
# Full path example:
#   S:\EXPENSE_TRACKER\expense_tracker_project\machine_learning\models
# ==========================================

MODEL_DIR = os.path.join(CURRENT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================================
# MODEL PATH HELPER
# ==============================================================================
def get_model_path(user_id):
    """
    Returns the .pkl file path for a given user.
    Example:
        .../machine_learning/models/forecast_model_user_16.pkl
    """
    return os.path.join(MODEL_DIR, f"forecast_model_user_{user_id}.pkl")


# ==============================================================================
# SAVE MODEL
# ==============================================================================
def save_model(model, user_id):

    print("\n=================================================")
    print("💾 SAVING MODEL")
    print("=================================================")

    path = get_model_path(user_id)

    try:

        joblib.dump(model, path)

        print(f"[FORECAST] ✅ Model saved: {path}")

    except Exception as e:

        print(f"[FORECAST ERROR] save_model(): {e}")


# ==============================================================================
# LOAD MODEL
# ==============================================================================
def load_model(user_id):

    print("\n=================================================")
    print("📂 LOADING SAVED MODEL")
    print("=================================================")

    path = get_model_path(user_id)

    if not os.path.exists(path):

        print(f"[FORECAST] ℹ️ No saved model found at: {path}")

        return None

    try:

        model = joblib.load(path)

        print(f"[FORECAST] ✅ Model loaded: {path}")

        return model

    except Exception as e:

        print(f"[FORECAST ERROR] load_model(): {e}")

        return None


# ==============================================================================
# DELETE MODEL  (call when user resets / deletes their data)
# ==============================================================================
def delete_model(user_id):

    print("\n=================================================")
    print("🗑️  DELETING SAVED MODEL")
    print("=================================================")

    path = get_model_path(user_id)

    if os.path.exists(path):

        try:

            os.remove(path)

            print(f"[FORECAST] ✅ Model deleted: {path}")

        except Exception as e:

            print(f"[FORECAST ERROR] delete_model(): {e}")

    else:

        print("[FORECAST] ℹ️ No model file to delete")


# ==============================================================================
# LOAD USER TRANSACTIONS
# ==============================================================================
def load_user_transactions(mysql, user_id):

    print("\n=================================================")
    print("📥 LOADING USER TRANSACTIONS")
    print("=================================================")

    print(f"[FORECAST] User ID: {user_id}")

    try:

        cur = mysql.connection.cursor()

        print("[FORECAST] Executing MySQL query...")

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

        print(f"[FORECAST] Raw rows fetched: {len(rows)}")

        # ==========================================
        # NO DATA
        # ==========================================
        if not rows:

            print("[FORECAST] ❌ No transactions found")

            return pd.DataFrame()

        # ==========================================
        # CREATE DATAFRAME
        # ==========================================
        df = pd.DataFrame(rows, columns=[
            "amount",
            "category",
            "type",
            "transaction_date"
        ])

        print("[FORECAST] ✅ DataFrame created")

        print("\n[FORECAST] FIRST 5 ROWS:")

        print(df.head())

        print("\n[FORECAST] DATA TYPES:")

        print(df.dtypes)

        return df

    except Exception as e:

        print(f"[FORECAST ERROR] load_user_transactions(): {e}")

        return pd.DataFrame()


# ==============================================================================
# PREPROCESS DATA
# ==============================================================================
def prepare_data(mysql, user_id):

    print("\n=================================================")
    print("🧹 PREPROCESSING DATA")
    print("=================================================")

    df = load_user_transactions(mysql, user_id)

    # ==========================================
    # EMPTY CHECK
    # ==========================================
    if df.empty:

        print("[FORECAST] ❌ DataFrame is empty")

        return None

    print(f"[FORECAST] Initial rows: {len(df)}")

    # ==========================================
    # DATE CONVERSION
    # ==========================================
    print("\n[FORECAST] Converting dates...")

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(subset=["transaction_date"])

    print(
        f"[FORECAST] Invalid dates removed: "
        f"{before - len(df)}"
    )

    # ==========================================
    # AMOUNT CONVERSION
    # ==========================================
    print("\n[FORECAST] Converting amounts...")

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(subset=["amount"])

    print(
        f"[FORECAST] Invalid amounts removed: "
        f"{before - len(df)}"
    )

    # ==========================================
    # REMOVE NEGATIVE VALUES
    # ==========================================
    before = len(df)

    df = df[df["amount"] > 0]

    print(
        f"[FORECAST] Negative amounts removed: "
        f"{before - len(df)}"
    )

    # ==========================================
    # REMOVE EXTREME OUTLIERS
    # ==========================================
    print("\n[FORECAST] Removing outliers...")

    q1 = df["amount"].quantile(0.25)

    q3 = df["amount"].quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)

    before = len(df)

    df = df[df["amount"] <= upper_limit]

    print(
        f"[FORECAST] Outliers removed: "
        f"{before - len(df)}"
    )

    print(f"[FORECAST] Final clean rows: {len(df)}")

    # ==========================================
    # DAILY AGGREGATION
    # ==========================================
    print("\n[FORECAST] Aggregating daily spending...")

    daily = df.groupby(
        df["transaction_date"].dt.date
    )["amount"].sum().reset_index()

    daily.columns = ["date", "total"]

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = daily.sort_values("date")

    daily["day_index"] = np.arange(len(daily))

    print(
        f"[FORECAST] Daily records prepared: "
        f"{len(daily)}"
    )

    print("\n[FORECAST] DAILY DATA:")

    print(daily.head())

    # ==========================================
    # UNIQUE DAY CHECK
    # ==========================================
    unique_days = len(daily)

    print(
        f"\n[FORECAST] Unique spending days: "
        f"{unique_days}"
    )

    return daily


# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(mysql, user_id):

    print("\n=================================================")
    print("🤖 TRAINING PERSONALIZED MODEL")
    print("=================================================")

    daily = prepare_data(mysql, user_id)

    if daily is None:

        print("[FORECAST] ❌ Training aborted")

        return None, None

    # ==========================================
    # NEED AT LEAST 7 UNIQUE DAYS
    # ==========================================
    if len(daily) < 7:

        print(
            "[FORECAST] ⚠️ AI still learning user behavior"
        )

        return None, daily

    # ==========================================
    # TRAINING
    # ==========================================
    X = daily[["day_index"]]

    y = daily["total"]

    print("\n[FORECAST] Training Features:")

    print(X.head())

    print("\n[FORECAST] Training Labels:")

    print(y.head())

    model = LinearRegression()

    model.fit(X, y)

    print("\n[FORECAST] ✅ Model training completed")

    print(
        f"[FORECAST] Model Coefficient: "
        f"{model.coef_[0]}"
    )

    print(
        f"[FORECAST] Model Intercept: "
        f"{model.intercept_}"
    )

    # ==========================================
    # SAVE TRAINED MODEL TO DISK
    # ==========================================
    save_model(model, user_id)

    return model, daily


# ==============================================================================
# GET MODEL  (load saved model or retrain)
# ==============================================================================
def get_model(mysql, user_id):
    """
    Returns (model, daily).

    Strategy:
      1. Try loading the saved .pkl model for this user.
      2. Also prepare the latest daily data from DB.
      3. If a saved model exists AND data is sufficient → use saved model.
      4. Otherwise → retrain, save, and return the new model.
    """

    saved = load_model(user_id)

    daily = prepare_data(mysql, user_id)

    if saved is not None and daily is not None and len(daily) >= 7:

        return saved, daily

    return train_model(mysql, user_id)


# ==============================================================================
# PREDICT FUTURE
# ==============================================================================
def predict(mysql, user_id, days):

    model, daily = get_model(mysql, user_id)

    # ==========================================
    # NO DATA
    # ==========================================
    if daily is None:


        return {
            "amount": 0,
            "transactions": 0,
            "message": "No expense data available yet.",
            "ai_ready": False
        }

    # ==========================================
    # VERY SMALL DATA  (< 7 days)
    # ==========================================
    if len(daily) < 7:

        recent_avg = daily["total"].mean()

        prediction = round(recent_avg * days, 2)

        avg_transactions = max(1, round(days * 0.8))

        # ======================================
        # BLOCK YEARLY FORECAST
        # ======================================
        if days >= 365:

            return {
                "amount": 0,
                "transactions": 0,
                "message":
                    "Yearly forecast available after "
                    "30 days of usage.",
                "ai_ready": False
            }

        return {
            "amount": prediction,
            "transactions": avg_transactions,
            "message":
                f"Estimated spending: ₹{prediction} "
                f"for next {days} days. "
                f"AI is still learning your habits.",
            "ai_ready": False
        }

    # ==========================================
    # ML PREDICTION
    # ==========================================
    future_index = len(daily) + days

    future = pd.DataFrame({
        "day_index": [future_index]
    })

    prediction = model.predict(future)[0]

    # ==========================================
    # SAFE LIMITER
    # ==========================================
    recent_avg = daily.tail(7)["total"].mean()

    max_limit = recent_avg * 1.5 * days

    prediction = min(prediction * days, max_limit)

    prediction = max(0, prediction)

    prediction = round(prediction, 2)

    avg_transactions = max(1, round(days * 0.8))

    return {
        "amount": prediction,
        "transactions": avg_transactions,
        "message":
            f"You may spend approximately "
            f"₹{prediction} within the next "
            f"{days} days.",
        "ai_ready": True
    }


# ==============================================================================
# TREND ANALYSIS
# ==============================================================================
def get_trend(mysql, user_id):

    model, daily = get_model(mysql, user_id)

    if model is None:

        return "AI Learning"

    slope = model.coef_[0]

    if slope > 50:

        trend = "Spending Increasing"

    elif slope < -50:

        trend = "Spending Decreasing"

    else:

        trend = "Spending Stable"

    return trend


# ==============================================================================
# NEXT DAY
# ==============================================================================
def next_day(mysql, user_id):

    return predict(mysql, user_id, 1)


# ==============================================================================
# WEEKLY
# ==============================================================================
def weekly(mysql, user_id):

    return predict(mysql, user_id, 7)


# ==============================================================================
# MONTHLY
# ==============================================================================
def monthly(mysql, user_id):

    return predict(mysql, user_id, 30)


# ==============================================================================
# YEARLY
# ==============================================================================
def yearly(mysql, user_id):

    return predict(mysql, user_id, 365)


# ==============================================================================
# TESTING
# ==============================================================================
# if __name__ == "__main__":

    # print("\n===================================")
    # print(" SMARTEXPENSEAI FORECAST TEST")
    # print("===================================\n")

    # from app import app, mysql

    # TEST_USER_ID = 16

    # with app.app_context():

    #     print("\n📅 NEXT DAY")

    #     print(next_day(mysql, TEST_USER_ID))

    #     print("\n📅 WEEKLY")

    #     print(weekly(mysql, TEST_USER_ID))

    #     print("\n📅 MONTHLY")

    #     print(monthly(mysql, TEST_USER_ID))

    #     print("\n📅 YEARLY")

    #     print(yearly(mysql, TEST_USER_ID))

    #     print("\n TREND")

    #     print(get_trend(mysql, TEST_USER_ID))
# ==============================================================================
# machine_learning/train_category_model.py  –  Train ML category predictor
# ==============================================================================

import os
import joblib #used for saving
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ── Base directory ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

# ── DATASETS (SAMPLE + USER DATA) ─────────────────────────────────────────────
SAMPLE_PATH = os.path.join(BASE_DIR, "dataset", "transaction_dataset.csv")

USER_PATH = os.path.join(
    BASE_DIR,
    "utils",
    "dataset",
    "transaction_dataset.csv"
)

# ── MODEL PATH ────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transaction_category_model.pkl")


# ==============================================================================
# STEP 1: LOAD DATA (SAMPLE + USER)
# ==============================================================================

def load_data():
    print("[TRAIN] Loading sample + user datasets...")

    # Load datasets
    df_sample = pd.read_csv(SAMPLE_PATH)
    df_user   = pd.read_csv(USER_PATH)

    print(f"[TRAIN] Sample rows : {len(df_sample)}")
    print(f"[TRAIN] User rows   : {len(df_user)}")

    # Combine datasets
    data = pd.concat([df_sample, df_user], ignore_index=True)

    # Validate columns
    required_columns = {"description", "transaction_type", "category"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"[TRAIN] Total rows  : {len(data)}")

    return data


# ==============================================================================
# STEP 2: PREPROCESS DATA
# ==============================================================================

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    print("[TRAIN] Cleaning data...")

    # Remove invalid rows
    data = data.dropna(subset=["category"]).copy()

    # Normalize text
    data["transaction_type"] = data["transaction_type"].str.strip().str.lower()
    data["description"] = data["description"].fillna("").astype(str).str.strip()
    data["category"] = data["category"].str.strip().str.lower()

    # Keep only valid types
    data = data[data["transaction_type"].isin(["income", "expense"])]

    # Create ML input text
    data["feature_text"] = (
        data["transaction_type"] + " " + data["description"]
    )

    print(f"[TRAIN] Clean rows : {len(data)}")
    print(f"[TRAIN] Categories : {sorted(data['category'].unique())}")

    return data


# ==============================================================================
# STEP 3: TRAIN MODEL
# ==============================================================================

def train_model(data: pd.DataFrame) -> Pipeline:
    print("\n[TRAIN] Training model...")

    X = data["feature_text"]
    y = data["category"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[TRAIN] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Pipeline: TF-IDF + Naive Bayes
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "─" * 50)
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print("─" * 50)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model


# ==============================================================================
# STEP 4: SAVE MODEL
# ==============================================================================

def save_model(model: Pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\n[TRAIN] Model saved → {MODEL_PATH}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" SmartExpenseAI - Model Training (UPDATED)")
    print("=" * 50)

    # Load both datasets
    data = load_data()

    # Clean data
    data = preprocess(data)

    # Train model
    model = train_model(data)

    # Save model
    save_model(model)

    print("\n✅ Training completed successfully!\n")
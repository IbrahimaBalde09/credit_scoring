# src/train_and_save.py

from __future__ import annotations

import os
import json
import joblib

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import TARGET_COL
from modeling_xgb import build_xgb_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_credit_scoring.joblib")
POLICY_PATH = os.path.join(MODEL_DIR, "policy.json")


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("📥 Loading data...")
    df = load_data()
    df = clean_data(df)

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    print("🤖 Training XGBoost model...")
    model = build_xgb_model(X_train)
    model.fit(X_train, y_train)

    print("💾 Saving model...")
    joblib.dump(model, MODEL_PATH)

    policy = {
        "t_accept": 0.10,
        "t_reject": 0.30,
        "model_name": "XGBoost + OneHotEncoder Pipeline",
        "target": TARGET_COL
    }

    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    print("✅ Model saved to:", MODEL_PATH)
    print("✅ Policy saved to:", POLICY_PATH)


if __name__ == "__main__":
    main()

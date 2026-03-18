# src/modeling.py

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)


TARGET_COL = "loan_status"


def build_model(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing + Logistic Regression baseline pipeline."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return model


def evaluate_at_thresholds(y_test, y_proba, thresholds=(0.3, 0.4, 0.5, 0.6)) -> None:
    """Print confusion matrix + classification report at multiple thresholds."""
    for t in thresholds:
        y_custom = (y_proba >= t).astype(int)
        print(f"\nSeuil: {t}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_custom))
        print("\nClassification Report:")
        print(classification_report(y_test, y_custom))


def main() -> None:
    df = load_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    model = build_model(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nAUC ROC:", roc_auc_score(y_test, y_proba))

    print("\nConfusion Matrix (seuil 0.5):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (seuil 0.5):")
    print(classification_report(y_test, y_pred))

    # Threshold analysis (Risk Management)
    evaluate_at_thresholds(y_test, y_proba, thresholds=(0.3, 0.4, 0.5, 0.6))


if __name__ == "__main__":
    main()

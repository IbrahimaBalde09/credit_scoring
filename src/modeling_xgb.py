# src/modeling_xgb.py

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import TARGET_COL  # cible
from decision_policy import apply_policy, policy_metrics

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


def build_xgb_model(X: pd.DataFrame) -> Pipeline:
    """
    XGBoost baseline (avec OneHotEncoder pour les variables catégorielles).
    Pas besoin de scaler pour XGBoost.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Paramètres raisonnables pour une baseline
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb),
    ])

    return model


def evaluate_thresholds(y_test, y_proba, thresholds=(0.3, 0.4, 0.5, 0.6)) -> None:
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

    model = build_xgb_model(X_train)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== XGBoost Results ===")
    print("AUC ROC:", roc_auc_score(y_test, y_proba))

    print("\nConfusion Matrix (seuil 0.5):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (seuil 0.5):")
    print(classification_report(y_test, y_pred))

    # Seuils comme précédemment
    evaluate_thresholds(y_test, y_proba, thresholds=(0.3, 0.4, 0.5, 0.6))

    # Test de la policy optimale trouvée avec Logistic
    t_accept, t_reject = 0.15, 0.40
    decisions = apply_policy(y_proba, t_accept=t_accept, t_reject=t_reject)
    m = policy_metrics(decisions, y_test.to_numpy())

    print("\n=== Decision Policy sur XGBoost ===")
    print(f"t_accept={t_accept} | t_reject={t_reject}")
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/credit_risk_dataset.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
POLICY_PATH = os.path.join(ARTIFACTS_DIR, "policy.json")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Nettoyage léger
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Harmonisation de quelques noms/valeurs fréquents
    if "loan_percent_income" in df.columns:
        df["loan_percent_income"] = pd.to_numeric(df["loan_percent_income"], errors="coerce")

    numeric_cols = [
        "loan_amnt",
        "person_income",
        "loan_int_rate",
        "loan_percent_income",
        "person_age",
        "person_emp_length",
        "cb_person_cred_hist_length",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = [
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "loan_amnt",
        "person_income",
        "loan_int_rate",
        "loan_percent_income",
        "person_age",
        "person_emp_length",
        "cb_person_cred_hist_length",
    ]

    categorical_features = [
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def main():
    df = load_data(DATA_PATH)

    features = [
        "loan_amnt",
        "person_income",
        "loan_int_rate",
        "loan_percent_income",
        "person_age",
        "person_emp_length",
        "cb_person_cred_hist_length",
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]
    target = "loan_status"

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    policy = {
        "t_accept": 0.10,
        "t_reject": 0.30,
    }
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)

    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    print(f"Policy sauvegardée dans {POLICY_PATH}")


if __name__ == "__main__":
    main()
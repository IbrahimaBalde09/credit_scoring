# src/preprocessing.py

import pandas as pd
from data_loading import load_data
from sklearn.model_selection import train_test_split


def clean_data(df):

    # 1️⃣ Supprimer doublons
    df = df.drop_duplicates()

    # 2️⃣ Supprimer âges aberrants
    df = df[df["person_age"] <= 100]

    # 3️⃣ Corriger ancienneté emploi aberrante
    df.loc[df["person_emp_length"] > df["person_age"], "person_emp_length"] = None
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = None

    # 4️⃣ Imputation médiane
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)

    return df


def split_data(df, target="loan_status"):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()
    df_clean = clean_data(df)

    print("Shape après nettoyage:", df_clean.shape)

    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

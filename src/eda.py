# src/eda.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

from data_loading import load_data



TARGET_COL = "loan_status"


def target_analysis(df: pd.DataFrame, target: str = TARGET_COL) -> None:
    """Print target distribution and proportions."""
    print("Distribution cible:\n")
    print(df[target].value_counts())

    print("\nProportion:\n")
    print(df[target].value_counts(normalize=True))


def missing_values_analysis(df: pd.DataFrame) -> None:
    """Print missing values per column (only columns with >0 missing)."""
    print("\nValeurs manquantes par colonne:\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Aucune valeur manquante ✅")
    else:
        print(missing)


def duplicate_analysis(df: pd.DataFrame) -> None:
    """Print number of duplicated rows."""
    print("\nNombre de doublons:")
    print(df.duplicated().sum())


def numerical_summary(df: pd.DataFrame) -> None:
    """Print descriptive stats for numeric columns."""
    print("\nStatistiques descriptives (numériques):\n")
    print(df.describe())


def main() -> None:
    # Path is managed in data_loading.py (default: data/credit_risk_dataset.csv)
    df = load_data()

    print("Shape:", df.shape)

    target_analysis(df, target=TARGET_COL)
    missing_values_analysis(df)
    duplicate_analysis(df)
    numerical_summary(df)


if __name__ == "__main__":
    main()

# src/data_loading.py

import pandas as pd


def load_data(path="data/credit_risk_dataset.csv"):
    """
    Load credit risk dataset
    """
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

# src/threshold_optimization.py

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import build_model, TARGET_COL
from decision_policy import apply_policy, policy_metrics


def optimize_thresholds(y_proba, y_true):

    results = []

    for t_accept in np.arange(0.1, 0.4, 0.05):
        for t_reject in np.arange(0.4, 0.7, 0.05):

            if t_accept >= t_reject:
                continue

            decisions = apply_policy(y_proba, t_accept, t_reject)
            metrics = policy_metrics(decisions, y_true)

            # Contraintes business
            if 0.15 <= metrics["review_rate"] <= 0.25:
                results.append({
                    "t_accept": round(t_accept, 2),
                    "t_reject": round(t_reject, 2),
                    "accepted_default_rate": metrics["accepted_default_rate"],
                    "accept_rate": metrics["accept_rate"],
                    "review_rate": metrics["review_rate"],
                    "reject_rate": metrics["reject_rate"]
                })

    df = pd.DataFrame(results)
    df = df.sort_values("accepted_default_rate")
    return df


def main():

    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    model = build_model(X_train)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    results = optimize_thresholds(y_proba, y_true)

    print("\n=== Meilleures combinaisons ===")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

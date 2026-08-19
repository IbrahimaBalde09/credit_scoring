# src/threshold_optimization_xgb.py

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data
from decision_policy import apply_policy, policy_metrics
from modeling_xgb import build_xgb_model
from modeling import TARGET_COL


def optimize_thresholds(y_proba, y_true):
    results = []

    for t_accept in np.arange(0.05, 0.30, 0.05):
        for t_reject in np.arange(0.30, 0.70, 0.05):
            if t_accept >= t_reject:
                continue

            decisions = apply_policy(y_proba, t_accept, t_reject)
            m = policy_metrics(decisions, y_true)

            # contrainte: review entre 15% et 25%
            if 0.15 <= m["review_rate"] <= 0.25:
                results.append({
                    "t_accept": round(t_accept, 2),
                    "t_reject": round(t_reject, 2),
                    "accepted_default_rate": m["accepted_default_rate"],
                    "accept_rate": m["accept_rate"],
                    "review_rate": m["review_rate"],
                    "reject_rate": m["reject_rate"]
                })

    df = pd.DataFrame(results).sort_values("accepted_default_rate")
    return df


def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    model = build_xgb_model(X_train)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    results = optimize_thresholds(y_proba, y_true)
    print("\n=== Meilleures combinaisons XGBoost ===")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

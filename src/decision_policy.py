# src/decision_policy.py

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import build_model, TARGET_COL


def apply_policy(y_proba: np.ndarray, t_accept: float, t_reject: float) -> np.ndarray:
    """
    Returns decisions:
      0 = ACCEPT
      1 = REVIEW
      2 = REJECT
    """
    decisions = np.ones_like(y_proba, dtype=int)  # default REVIEW (1)
    decisions[y_proba < t_accept] = 0
    decisions[y_proba >= t_reject] = 2
    return decisions


def policy_metrics(decisions: np.ndarray, y_true: np.ndarray) -> dict:
    """
    y_true: 0=bon, 1=défaut
    decisions: 0=ACCEPT, 1=REVIEW, 2=REJECT
    """
    n = len(y_true)
    accept_mask = decisions == 0
    review_mask = decisions == 1
    reject_mask = decisions == 2

    # Taux
    accept_rate = accept_mask.mean()
    review_rate = review_mask.mean()
    reject_rate = reject_mask.mean()

    # Défaut parmi acceptés (c’est le plus critique)
    # Attention: si aucun accepté, éviter division par 0
    accepted_default_rate = (
        y_true[accept_mask].mean() if accept_mask.sum() > 0 else np.nan
    )

    # Défaut parmi refusés (devrait être élevé si politique bonne)
    rejected_default_rate = (
        y_true[reject_mask].mean() if reject_mask.sum() > 0 else np.nan
    )

    return {
        "accept_rate": round(float(accept_rate), 4),
        "review_rate": round(float(review_rate), 4),
        "reject_rate": round(float(reject_rate), 4),
        "accepted_default_rate": round(float(accepted_default_rate), 4)
        if not np.isnan(accepted_default_rate)
        else np.nan,
        "rejected_default_rate": round(float(rejected_default_rate), 4)
        if not np.isnan(rejected_default_rate)
        else np.nan,
        "accepted_count": int(accept_mask.sum()),
        "review_count": int(review_mask.sum()),
        "rejected_count": int(reject_mask.sum()),
    }


def main() -> None:
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    model = build_model(X_train)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    # 🔧 Politique initiale (proposition)
    # - accept si proba défaut < 0.25
    # - reject si proba défaut >= 0.50
    t_accept = 0.25
    t_reject = 0.50

    decisions = apply_policy(y_proba, t_accept=t_accept, t_reject=t_reject)
    m = policy_metrics(decisions, y_true)

    print("\n=== Decision Policy (3 zones) ===")
    print(f"t_accept={t_accept} | t_reject={t_reject}")
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

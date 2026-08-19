# src/business_evaluation.py

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import build_model, TARGET_COL

from sklearn.metrics import confusion_matrix


def profit_from_confusion_matrix(
    cm: np.ndarray,
    profit_tp: float,
    profit_tn: float,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """
    cm format:
      [[TN, FP],
       [FN, TP]]

    Interprétation business (important):
    - Classe 1 = défaut (loan_status=1)
    - On REFUSE si le modèle prédit 1 (risque élevé)
    - On ACCEPTE si le modèle prédit 0 (risque faible)

    Donc:
    - TN = vrai défaut correctement refusé -> gain = 0 (perte évitée)
    - FP = bon client refusé à tort -> coût d'opportunité
    - FN = défaut accepté à tort -> perte
    - TP = bon client accepté -> profit
    """
    tn, fp, fn, tp = cm.ravel()
    total = tp * profit_tp + tn * profit_tn - fp * cost_fp - fn * cost_fn
    return total


def evaluate_thresholds_business(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    # ---- Hypothèses business (MODIFIÉES) ----
    profit_tp: float = 3000.0,   # profit si on accepte un bon client
    profit_tn: float = 0.0,      # profit si on refuse un mauvais client (perte évitée, mise à 0)
    cost_fp: float = 3000.0,     # coût si on refuse un bon client
    cost_fn: float = 8000.0,     # perte si on accepte un mauvais client
) -> pd.DataFrame:
    rows = []
    n = len(y_true)

    for t in thresholds:
        # 1 = défaut => au-dessus du seuil = risque => refuser
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        profit = profit_from_confusion_matrix(
            cm,
            profit_tp=profit_tp,
            profit_tn=profit_tn,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        # Accepté = prédiction 0 (faible risque)
        accepted = tn + fn
        accepted_rate = accepted / n

        rows.append(
            {
                "threshold": t,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "accepted_rate": round(accepted_rate, 4),
                "profit_total": round(profit, 2),
                "profit_per_app": round(profit / n, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("profit_total", ascending=False)


def main() -> None:
    df = load_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    model = build_model(X_train)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    # Hypothèses business (MODIFIÉES)
    results = evaluate_thresholds_business(
        y_true=y_test.to_numpy(),
        y_proba=y_proba,
        thresholds=thresholds,
        profit_tp=3000.0,
        profit_tn=0.0,
        cost_fp=3000.0,
        cost_fn=8000.0,
    )

    print("\n=== Résultats Business (triés par profit_total décroissant) ===")
    print(results.to_string(index=False))

    best = results.iloc[0]
    print("\n✅ Meilleur seuil (selon ces hypothèses):", best["threshold"])
    print("   Profit total:", best["profit_total"])
    print("   Profit par demande:", best["profit_per_app"])
    print("   Taux d'acceptation:", best["accepted_rate"])


if __name__ == "__main__":
    main()

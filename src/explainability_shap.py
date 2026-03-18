# src/explainability_shap.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from data_loading import load_data
from preprocessing import clean_data, split_data
from modeling import TARGET_COL
from modeling_xgb import build_xgb_model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_feature_names(preprocessor):
    """Return feature names after preprocessing if available."""
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def to_dense(matrix):
    """Convert sparse matrix to dense numpy array if needed."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def main() -> None:
    # ---------- output dirs ----------
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    ensure_dir(figures_dir)

    # ---------- data ----------
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    # ---------- model ----------
    model = build_xgb_model(X_train)
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # ---------- samples for global plots ----------
    X_eval = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    X_eval_t = preprocessor.transform(X_eval)

    feature_names = get_feature_names(preprocessor)

    X_eval_dense = to_dense(X_eval_t)
    if feature_names is not None and X_eval_dense.ndim == 2 and X_eval_dense.shape[1] == len(feature_names):
        X_eval_df = pd.DataFrame(X_eval_dense, columns=feature_names)
    else:
        X_eval_df = None

    # ---------- SHAP ----------
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_eval_dense)

    # ---------- 1) global bar ----------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_eval_df if X_eval_df is not None else X_eval_dense,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    out1 = os.path.join(figures_dir, "shap_global_bar.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out1)

    # ---------- 2) global beeswarm ----------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_eval_df if X_eval_df is not None else X_eval_dense,
        show=False,
        max_display=15,
    )
    out2 = os.path.join(figures_dir, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out2)

    # ---------- 3) local explanation (most risky from full X_test) ----------
    y_proba_test = model.predict_proba(X_test)[:, 1]
    pos = int(np.argmax(y_proba_test))  # position in X_test (0..len-1)
    x_one = X_test.iloc[[pos]]
    x_one_t = preprocessor.transform(x_one)
    x_one_dense = to_dense(x_one_t)

    shap_one = explainer.shap_values(x_one_dense)

    # Build a SHAP Explanation object for waterfall plot
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[0]

    # feature names + values
    if feature_names is not None and x_one_dense.shape[1] == len(feature_names):
        feat_names = feature_names
        feat_vals = x_one_dense[0]
    else:
        feat_names = None
        feat_vals = x_one_dense[0]

    exp = shap.Explanation(
        values=shap_one[0],
        base_values=base,
        data=feat_vals,
        feature_names=feat_names,
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=15, show=False)
    out3 = os.path.join(figures_dir, "shap_local_waterfall.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out3)

    print("\n=== Local example (most risky in X_test) ===")
    print("Position in X_test:", pos)
    print("Predicted default probability:", float(y_proba_test[pos]))


if __name__ == "__main__":
    main()

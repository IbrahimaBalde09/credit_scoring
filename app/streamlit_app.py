# app/streamlit_app.py

from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Scoring – Risk Engine",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = "models/xgb_credit_scoring.joblib"
POLICY_PATH = "models/policy.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_policy():
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def decision_from_policy(p_default: float, t_accept: float, t_reject: float) -> str:
    if p_default < t_accept:
        return "✅ ACCEPT"
    if p_default >= t_reject:
        return "❌ REJECT"
    return "🟡 REVIEW"


def main():
    st.title("💳 Credit Scoring – Risk Engine")
    st.caption("Modèle XGBoost + Politique de décision (Accept / Review / Reject)")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(POLICY_PATH)):
        st.error("Modèle/policy introuvables. Lance d’abord: python src/train_and_save.py")
        st.stop()

    model = load_model()
    policy = load_policy()
    t_accept = float(policy["t_accept"])
    t_reject = float(policy["t_reject"])

    # --- Layout ---
    left, right = st.columns([1, 1.2], gap="large")

    with st.sidebar:
        st.header("⚙️ Paramètres")
        st.write("Policy actuelle:")
        st.metric("t_accept", t_accept)
        st.metric("t_reject", t_reject)

        show_shap = st.toggle("Afficher explication SHAP (plus lent)", value=True)
        st.divider()
        st.write("Astuce: tu peux ajuster ces seuils pour faire une démo live.")
        t_accept_ui = st.slider("t_accept", 0.01, 0.40, t_accept, 0.01)
        t_reject_ui = st.slider("t_reject", 0.10, 0.80, t_reject, 0.01)
        if t_accept_ui >= t_reject_ui:
            st.warning("t_accept doit être < t_reject")
        t_accept, t_reject = t_accept_ui, t_reject_ui

    with left:
        st.subheader("📝 Dossier client")

        # IMPORTANT: les noms doivent matcher ton dataset
        # (si une colonne diffère chez toi, dis-moi et je l’adapte)
        person_age = st.number_input("Âge (person_age)", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Revenu annuel (person_income)", min_value=0, value=55000)
        person_home_ownership = st.selectbox(
            "Logement (person_home_ownership)",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            index=0
        )
        person_emp_length = st.number_input("Ancienneté emploi (person_emp_length)", min_value=0.0, max_value=60.0, value=4.0)

        loan_intent = st.selectbox(
            "Intention prêt (loan_intent)",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            index=3
        )
        loan_grade = st.selectbox("Grade (loan_grade)", ["A", "B", "C", "D", "E", "F", "G"], index=2)

        loan_amnt = st.number_input("Montant prêt (loan_amnt)", min_value=500, max_value=35000, value=8000)
        loan_int_rate = st.number_input("Taux intérêt (loan_int_rate)", min_value=0.0, max_value=30.0, value=11.0)
        loan_percent_income = st.number_input("Part revenu (loan_percent_income)", min_value=0.0, max_value=1.0, value=0.15)

        cb_person_default_on_file = st.selectbox(
            "Défaut historique (cb_person_default_on_file)",
            ["N", "Y"],
            index=0
        )
        cb_person_cred_hist_length = st.number_input(
            "Historique crédit (cb_person_cred_hist_length)",
            min_value=0, max_value=80, value=4
        )

        predict = st.button("📌 Calculer le score", type="primary")

    with right:
        st.subheader("📊 Résultat")

        if predict:
            # Construire une ligne dataframe avec EXACTEMENT les colonnes du modèle
            row = {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
            X_one = pd.DataFrame([row])

            p_default = float(model.predict_proba(X_one)[:, 1][0])
            decision = decision_from_policy(p_default, t_accept, t_reject)

            top1, top2, top3 = st.columns(3)
            top1.metric("Probabilité défaut", f"{p_default:.2%}")
            top2.metric("Décision", decision)
            top3.metric("Policy", f"<{t_accept:.2f} ACCEPT | ≥{t_reject:.2f} REJECT")

            st.progress(min(max(p_default, 0.0), 1.0))

            st.divider()
            st.write("**Dossier (inputs)**")
            st.dataframe(X_one, use_container_width=True)

            if show_shap:
                st.divider()
                st.write("### 🔍 Explication (SHAP – top facteurs)")

                # Récupère le preprocessor + modèle XGB
                preprocessor = model.named_steps["preprocessor"]
                clf = model.named_steps["classifier"]

                X_one_t = preprocessor.transform(X_one)
                X_one_dense = X_one_t.toarray() if hasattr(X_one_t, "toarray") else np.asarray(X_one_t)

                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_one_dense)[0]

                # Feature names post-encodage
                try:
                    feat_names = preprocessor.get_feature_names_out()
                except Exception:
                    feat_names = np.array([f"f{i}" for i in range(len(shap_vals))])

                # Top contributions absolues
                contrib = pd.DataFrame({
                    "feature": feat_names,
                    "shap_value": shap_vals,
                    "abs": np.abs(shap_vals),
                }).sort_values("abs", ascending=False).head(15)

                st.dataframe(contrib[["feature", "shap_value"]], use_container_width=True)

                # Plot bar local
                fig, ax = plt.subplots()
                ax.barh(contrib["feature"][::-1], contrib["shap_value"][::-1])
                ax.set_title("Contributions SHAP (local)")
                ax.set_xlabel("Impact sur le score (log-odds)")
                st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()

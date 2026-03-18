import json
from pathlib import Path

import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    import shap
except ImportError:
    shap = None

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
POLICY_PATH = ARTIFACTS_DIR / "policy.json"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zcdFLHEi2fZ59BvTpb2osfRQ2B8pvJxG"

DISPLAY_NAMES = {
    "loan_amnt": "Montant du prêt",
    "person_income": "Revenu annuel",
    "loan_int_rate": "Taux d'intérêt",
    "loan_percent_income": "Taux d'endettement",
    "person_age": "Âge",
    "person_emp_length": "Ancienneté emploi",
    "cb_person_cred_hist_length": "Historique de crédit",
    "loan_grade": "Grade de crédit",
    "person_home_ownership": "Situation logement",
    "loan_intent": "Objet du prêt",
    "cb_person_default_on_file": "Antécédent de défaut",
}


def load_model():
    if joblib is None or not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_policy():
    default_policy = {
        "t_accept": 0.10,
        "t_reject": 0.30,
    }

    if not POLICY_PATH.exists():
        return default_policy

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = json.load(f)

    return {
        "t_accept": float(policy.get("t_accept", 0.10)),
        "t_reject": float(policy.get("t_reject", 0.30)),
    }


model = load_model()
policy = load_policy()


def model_available():
    return model is not None


def shap_available():
    return shap is not None and model is not None


def prepare_input_dataframe(data):
    debt_ratio = float(data["debt_ratio"]) / 100.0

    cb_default_raw = str(data.get("cb_person_default_on_file", "")).strip()
    cb_default = "Y" if cb_default_raw in {"1", "Oui", "OUI", "Y", "YES"} else "N"

    row = {
        "loan_amnt": float(data["loan_amnt"]),
        "person_income": float(data["person_income"]),
        "loan_int_rate": float(data["loan_int_rate"]),
        "loan_percent_income": debt_ratio,
        "person_age": float(data["person_age"]),
        "person_emp_length": float(data["person_emp_length"]),
        "cb_person_cred_hist_length": float(data["cb_person_cred_hist_length"]),
        "loan_grade": str(data.get("loan_grade", "")).strip().upper(),
        "person_home_ownership": str(data.get("person_home_ownership", "")).strip().upper(),
        "loan_intent": str(data.get("loan_intent", "")).strip().upper(),
        "cb_person_default_on_file": cb_default,
    }

    return pd.DataFrame([row])


def predict_score(data):
    if model is None:
        raise FileNotFoundError(
            f"Modèle introuvable. Ajoute '{MODEL_PATH.name}' dans le dossier artifacts."
        )

    X = prepare_input_dataframe(data)
    proba = model.predict_proba(X)[0][1]
    return float(proba)


def decision_from_proba(p, t_accept=None, t_reject=None):

    if t_accept is None:
        t_accept = policy["t_accept"]

    if t_reject is None:
        t_reject = policy["t_reject"]

    if p < t_accept:
        return "ACCEPT", "Faible"

    if p < t_reject:
        return "REVIEW", "Modéré"

    return "REJECT", "Élevé"


def interpretation(decision):

    if decision == "ACCEPT":
        return "Risque faible, dossier éligible à une acceptation automatique."

    if decision == "REVIEW":
        return "Risque intermédiaire, analyse manuelle recommandée."

    return "Risque élevé de défaut selon la politique actuelle."


def risk_factors(data):

    negative = []
    positive = []

    debt_ratio = float(data["debt_ratio"])
    loan_int_rate = float(data["loan_int_rate"])
    income = float(data["person_income"])
    emp_length = float(data["person_emp_length"])
    cred_hist = float(data["cb_person_cred_hist_length"])
    age = float(data["person_age"])
    grade = str(data.get("loan_grade", "")).upper()
    home = str(data.get("person_home_ownership", "")).upper()

    if debt_ratio >= 45:
        negative.append("Taux d'endettement élevé")
    elif debt_ratio <= 25:
        positive.append("Taux d'endettement maîtrisé")

    if loan_int_rate >= 12:
        negative.append("Taux d'intérêt élevé")
    elif loan_int_rate <= 8:
        positive.append("Taux d'intérêt modéré")

    if income < 25000:
        negative.append("Revenu annuel limité")
    elif income >= 40000:
        positive.append("Revenu confortable")

    if emp_length < 2:
        negative.append("Faible ancienneté professionnelle")
    elif emp_length >= 3:
        positive.append("Ancienneté emploi stable")

    if cred_hist < 2:
        negative.append("Historique de crédit limité")
    elif cred_hist >= 4:
        positive.append("Historique de crédit établi")

    if grade in {"D", "E", "F", "G"}:
        negative.append("Grade de crédit défavorable")

    if home == "RENT":
        negative.append("Situation locative")

    if age >= 25:
        positive.append("Profil d'âge plus mature")

    return negative, positive


def compute_shap_explanations(data):
    return [], []


def build_result(data):

    p = predict_score(data)

    decision, risk = decision_from_proba(p)

    # Traduction française
    decision_fr_map = {
        "ACCEPT": "Accepté",
        "REVIEW": "À examiner",
        "REJECT": "Refusé",
    }

    decision_fr = decision_fr_map.get(decision, decision)

    explanation = interpretation(decision)

    negative, positive = risk_factors(data)

    return {
        "prob": round(p * 100, 2),
        "decision": decision,
        "decision_fr": decision_fr,
        "risk": risk,
        "interpretation": explanation,
        "neg": negative,
        "pos": positive,
        "thresholds": {
            "accept": round(policy["t_accept"] * 100, 2),
            "reject": round(policy["t_reject"] * 100, 2),
        },
        "shap": {
            "enabled": False,
            "risk_up": [],
            "risk_down": [],
        },
    }
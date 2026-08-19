"""
Moteur de scoring crédit.

Ce module encapsule tout ce qui touche au modèle ML :
- chargement de l'artefact versionné (jamais de téléchargement "à la volée"
  depuis une URL publique non contrôlée : le modèle doit être livré avec
  l'application, voir DEPLOYMENT.md) ;
- calcul de la probabilité de défaut à partir des mêmes features que celles
  utilisées à l'entraînement ;
- application de la politique de décision (ACCEPT / REVIEW / REJECT) ;
- explicabilité SHAP par requête.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from django.conf import settings

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import shap
    import numpy as np
except ImportError:  # pragma: no cover
    shap = None
    np = None

logger = logging.getLogger("scoring")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(getattr(settings, "MODEL_PATH", BASE_DIR / "artifacts" / "model.joblib"))
POLICY_PATH = Path(getattr(settings, "POLICY_PATH", BASE_DIR / "artifacts" / "policy.json"))

DISPLAY_NAMES = {
    "loan_amnt": "Montant du prêt",
    "person_income": "Revenu annuel",
    "loan_int_rate": "Taux d'intérêt",
    "loan_percent_income": "Ratio prêt/revenu",
    "person_age": "Âge",
    "person_emp_length": "Ancienneté emploi",
    "cb_person_cred_hist_length": "Historique de crédit",
    "loan_grade": "Grade de crédit",
    "person_home_ownership": "Situation logement",
    "loan_intent": "Objet du prêt",
    "cb_person_default_on_file": "Antécédent de défaut",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_fetch_model_from_url() -> None:
    """
    Repli optionnel : si MODEL_URL est configurée (stockage objet interne,
    signé, contrôlé par la banque) et que le fichier n'est pas déjà présent,
    on le télécharge puis on vérifie son empreinte SHA-256 attendue
    (MODEL_SHA256). En fonctionnement normal, cette fonction ne fait rien :
    le modèle doit être packagé dans l'image Docker.
    """
    model_url = getattr(settings, "MODEL_URL", "")
    if not model_url or MODEL_PATH.exists():
        return

    expected_sha256 = getattr(settings, "MODEL_SHA256", "")
    if not expected_sha256:
        logger.error(
            "MODEL_URL est configurée mais MODEL_SHA256 ne l'est pas : "
            "téléchargement refusé (pas de vérification d'intégrité possible)."
        )
        return

    import urllib.request

    if not model_url.lower().startswith("https://"):
        logger.error("MODEL_URL doit être une URL https:// (schéma refusé: %s).", model_url.split(":", 1)[0])
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".tmp")
    logger.info("Téléchargement du modèle depuis la source configurée...")
    urllib.request.urlretrieve(model_url, tmp_path)  # nosec B310

    actual_sha256 = _sha256(tmp_path)
    if actual_sha256 != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "Empreinte SHA-256 du modèle téléchargé invalide (attendu=%s, obtenu=%s). "
            "Fichier rejeté.", expected_sha256, actual_sha256,
        )
        return

    tmp_path.rename(MODEL_PATH)
    logger.info("Modèle téléchargé et vérifié avec succès.")


def load_model():
    if joblib is None:
        logger.error("joblib n'est pas installé, le scoring est indisponible.")
        return None

    _maybe_fetch_model_from_url()

    if not MODEL_PATH.exists():
        logger.error("Artefact modèle introuvable: %s", MODEL_PATH)
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Échec du chargement du modèle depuis %s", MODEL_PATH)
        return None


def load_policy() -> dict:
    default_policy = {"t_accept": 0.10, "t_reject": 0.30}

    if not POLICY_PATH.exists():
        return default_policy

    try:
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            policy = json.load(f)
    except Exception:
        logger.exception("Échec de lecture de la politique de décision, repli sur les valeurs par défaut.")
        return default_policy

    return {
        "t_accept": float(policy.get("t_accept", default_policy["t_accept"])),
        "t_reject": float(policy.get("t_reject", default_policy["t_reject"])),
    }


def _compute_model_version() -> str:
    """
    Version du modèle exposée dans l'API/l'audit trail. Priorité :
    1. metadata explicite (artifacts/model_metadata.json), pour un vrai
       pipeline MLOps (numéro de run, date d'entraînement, AUC...) ;
    2. à défaut, empreinte SHA-256 courte du fichier modèle (garantit qu'on
       sait toujours *exactement* quel binaire a produit une décision,
       même sans metadata explicite).
    """
    metadata_path = MODEL_PATH.parent / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            version = meta.get("version")
            if version:
                return str(version)
        except Exception:
            logger.warning("model_metadata.json illisible, repli sur le hash du fichier.")

    if MODEL_PATH.exists():
        return f"sha256:{_sha256(MODEL_PATH)[:12]}"

    return "unknown"


model = load_model()
policy = load_policy()
MODEL_VERSION = _compute_model_version()

_shap_explainer = None
if shap is not None and model is not None:
    try:
        _shap_explainer = shap.TreeExplainer(model.named_steps["classifier"])
    except Exception:
        logger.exception("Impossible d'initialiser l'explainer SHAP.")
        _shap_explainer = None


def model_available() -> bool:
    return model is not None


def shap_available() -> bool:
    return _shap_explainer is not None


def get_model_version() -> str:
    return MODEL_VERSION


def prepare_input_dataframe(data: dict) -> pd.DataFrame:
    """
    Construit la ligne de features attendue par le pipeline sklearn, avec
    exactement la même sémantique que les données d'entraînement.

    Important : `loan_percent_income` (feature utilisée par le modèle) est
    RECALCULÉE ici à partir de loan_amnt / person_income, et n'est PAS prise
    telle quelle depuis la saisie utilisateur. `debt_ratio` saisi par le
    client représente son endettement global déclaré (toutes charges de
    crédit confondues) : c'est une donnée business utile pour les règles de
    garde-fou et l'affichage, mais ce n'est pas la feature sur laquelle le
    modèle a été entraîné. Les confondre revient à empoisonner le modèle
    avec une donnée déclarative non vérifiée.
    """
    income = float(data["person_income"])
    loan_amnt = float(data["loan_amnt"])
    loan_percent_income = (loan_amnt / income) if income > 0 else 0.0

    cb_default_raw = str(data.get("cb_person_default_on_file", "")).strip()
    cb_default = "Y" if cb_default_raw in {"1", "Oui", "OUI", "Y", "YES"} else "N"

    row = {
        "loan_amnt": loan_amnt,
        "person_income": income,
        "loan_int_rate": float(data["loan_int_rate"]),
        "loan_percent_income": loan_percent_income,
        "person_age": float(data["person_age"]),
        "person_emp_length": float(data["person_emp_length"]),
        "cb_person_cred_hist_length": float(data["cb_person_cred_hist_length"]),
        "loan_grade": str(data.get("loan_grade", "")).strip().upper(),
        "person_home_ownership": str(data.get("person_home_ownership", "")).strip().upper(),
        "loan_intent": str(data.get("loan_intent", "")).strip().upper(),
        "cb_person_default_on_file": cb_default,
    }

    return pd.DataFrame([row])


def predict_score(data: dict) -> float:
    if model is None:
        raise FileNotFoundError(
            f"Modèle introuvable. Ajoute '{MODEL_PATH.name}' dans le dossier artifacts."
        )

    X = prepare_input_dataframe(data)
    proba = model.predict_proba(X)[0][1]
    return float(proba)


def decision_from_proba(p: float, t_accept: float | None = None, t_reject: float | None = None):
    if t_accept is None:
        t_accept = policy["t_accept"]

    if t_reject is None:
        t_reject = policy["t_reject"]

    if p < t_accept:
        return "ACCEPT", "Faible"

    if p < t_reject:
        return "REVIEW", "Modéré"

    return "REJECT", "Élevé"


def interpretation(decision: str) -> str:
    if decision == "ACCEPT":
        return "Risque faible, dossier éligible à une acceptation automatique."

    if decision == "REVIEW":
        return "Risque intermédiaire, analyse manuelle recommandée."

    return "Risque élevé de défaut selon la politique actuelle."


def risk_factors(data: dict):
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

    # Règle prudentielle HCSF (France) : au-delà de 35% d'endettement, le
    # dossier est considéré à risque même si le modèle ne le détecte pas
    # (le modèle ne reçoit pas cette donnée déclarative brute, cf.
    # prepare_input_dataframe).
    if debt_ratio >= 35:
        negative.append("Taux d'endettement déclaré supérieur au seuil prudentiel (35%)")
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


def compute_shap_explanations(data: dict, top_n: int = 5):
    """
    Calcule les contributions SHAP réelles pour cette requête et retourne
    les `top_n` facteurs qui augmentent le risque et ceux qui le réduisent,
    avec des noms lisibles.
    """
    if _shap_explainer is None or model is None:
        return [], []

    try:
        X = prepare_input_dataframe(data)
        preprocessor = model.named_steps["preprocessor"]
        X_t = preprocessor.transform(X)
        X_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_dense.shape[1])]

        shap_values = _shap_explainer.shap_values(X_dense)
        if isinstance(shap_values, list):
            # binaire : on garde la contribution vers la classe positive (défaut)
            shap_values = shap_values[-1]

        contributions = list(zip(feature_names, shap_values[0]))

        def humanize(raw_name: str) -> str:
            # les noms transformés ressemblent à "num__loan_amnt" ou
            # "cat__loan_grade_B" : on retire le préfixe et on tente une
            # correspondance avec DISPLAY_NAMES.
            base = raw_name.split("__", 1)[-1]
            for key, label in DISPLAY_NAMES.items():
                if base == key or base.startswith(key + "_"):
                    suffix = base[len(key):].lstrip("_")
                    return f"{label}" + (f" ({suffix})" if suffix else "")
            return base

        risk_up = sorted(contributions, key=lambda kv: kv[1], reverse=True)[:top_n]
        risk_down = sorted(contributions, key=lambda kv: kv[1])[:top_n]

        risk_up = [{"feature": humanize(k), "impact": round(float(v), 4)} for k, v in risk_up if v > 0]
        risk_down = [{"feature": humanize(k), "impact": round(float(v), 4)} for k, v in risk_down if v < 0]

        return risk_up, risk_down
    except Exception:
        logger.exception("Échec du calcul SHAP pour cette requête.")
        return [], []


def build_result(data: dict) -> dict:
    p = predict_score(data)

    decision, risk = decision_from_proba(p)

    decision_fr_map = {
        "ACCEPT": "Accepté",
        "REVIEW": "À examiner",
        "REJECT": "Refusé",
    }

    decision_fr = decision_fr_map.get(decision, decision)

    explanation = interpretation(decision)

    negative, positive = risk_factors(data)

    risk_up, risk_down = compute_shap_explanations(data)

    return {
        "prob": round(p * 100, 2),
        "decision": decision,
        "decision_fr": decision_fr,
        "risk": risk,
        "interpretation": explanation,
        "neg": negative,
        "pos": positive,
        "model_version": MODEL_VERSION,
        "thresholds": {
            "accept": round(policy["t_accept"] * 100, 2),
            "reject": round(policy["t_reject"] * 100, 2),
        },
        "shap": {
            "enabled": shap_available(),
            "risk_up": risk_up,
            "risk_down": risk_down,
        },
    }

import json
import urllib.request
from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None


BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
POLICY_PATH = ARTIFACTS_DIR / "policy.json"

# ✅ TON LIEN GOOGLE DRIVE (déjà corrigé)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zcdFLHEi2fZ59BvTpb2osfRQ2B8pvJxG"


def load_model():
    if joblib is None:
        print("❌ joblib n'est pas installé")
        return None

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 📥 Télécharger le modèle si absent
    if not MODEL_PATH.exists():
        try:
            print(f"⬇️ Téléchargement du modèle depuis: {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"✅ Modèle téléchargé dans: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Erreur téléchargement modèle: {e}")
            return None

    # 📦 Charger le modèle
    try:
        print(f"📦 Chargement du modèle depuis: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None


def load_policy():
    if not POLICY_PATH.exists():
        return {"threshold": 0.5}

    try:
        with open(POLICY_PATH) as f:
            return json.load(f)
    except Exception:
        return {"threshold": 0.5}


# 🔥 Chargement au démarrage
model = load_model()
policy = load_policy()


def score_client(features: dict):
    if model is None:
        return {
            "error": "Le modèle n'est pas disponible. Ajoute artifacts/model.joblib."
        }

    try:
        import pandas as pd

        df = pd.DataFrame([features])
        proba = model.predict_proba(df)[0][1]
        decision = "ACCORD" if proba >= policy.get("threshold", 0.5) else "REFUS"

        return {
            "probability": round(float(proba), 4),
            "decision": decision,
        }

    except Exception as e:
        return {"error": str(e)}
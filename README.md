# Credit Scoring Platform — Risk Modeling & Décision bancaire

Plateforme de scoring crédit de bout en bout : pipeline ML (XGBoost + SHAP)
pour prédire le risque de défaut, politique de décision à 3 zones
(Accept / Review / Reject), et application Django sécurisée (formulaire
analyste, API interne, dashboard portefeuille) prête pour un environnement
bancaire.

## Sommaire
- [Architecture](#architecture)
- [Modélisation](#modélisation)
- [Sécurité](#sécurité)
- [Démarrage rapide](#démarrage-rapide)
- [Tests & qualité](#tests--qualité)
- [Déploiement](#déploiement)

## Architecture

```
credit_scoring/
├── config/            # Configuration Django (settings, urls, wsgi)
├── scoring/           # Application Django
│   ├── views.py        # Formulaire, dashboard, API, healthz
│   ├── services.py      # Moteur de scoring (feature engineering, SHAP)
│   ├── forms.py          # Validation métier
│   ├── models.py          # Historique de scoring (piste d'audit)
│   ├── auth.py              # Authentification API par clé
│   └── tests/                # Suite de tests (pytest)
├── src/                # Pipeline d'entraînement (EDA, preprocessing,
│                        # modélisation, politique de décision, SHAP)
├── artifacts/          # Artefact modèle versionné + metadata + policy
├── app/streamlit_app.py # Prototype exploratoire (hors périmètre prod)
├── Dockerfile / docker-compose.yml / entrypoint.sh
└── render.yaml          # Blueprint de déploiement Render
```

**Flux applicatif**
1. Un analyste se connecte (`/login/`, obligatoire) et saisit un dossier
   (`/`), ou un système du SI bancaire appelle `POST /api/score/` avec une
   clé API (`X-API-Key`).
2. `scoring/services.py` construit les features exactement comme à
   l'entraînement (le ratio prêt/revenu est **recalculé** à partir de
   `loan_amnt` et `person_income`, jamais pris tel quel depuis une saisie
   libre), calcule la probabilité de défaut et applique la politique de
   décision (`artifacts/policy.json`).
3. Chaque décision est journalisée dans `ScoreHistory` avec : analyste (ou
   `API`), IP, version du modèle, horodatage — piste d'audit exploitable
   par la conformité.
4. Le dashboard (`/dashboard/`) agrège le portefeuille : taux
   d'acceptation, exposition, perte attendue, répartition du risque.

## Modélisation

- **Données** : 32 581 lignes, cible `loan_status` (défaut ≈ 21.8%)
- **Baseline** : Logistic Regression — AUC ROC ≈ 0.867
- **Modèle retenu** : XGBoost — AUC test ≈ 0.927 (voir
  `artifacts/model_metadata.json` pour les métriques complètes :
  CV AUC, Brier score, statistique KS)
- **Politique de décision** (optimisée pour ce modèle) :
  `t_accept=0.09`, `t_reject=0.20` → taux de défaut des dossiers acceptés
  ramené de ~21.8% à ~3.96%
- **Explicabilité** : SHAP `TreeExplainer`, calculé **en temps réel pour
  chaque requête** (pas un stub) et renvoyé dans la réponse API
  (`shap.risk_up` / `shap.risk_down`)

Pour ré-entraîner à partir des scripts `src/` :
```bash
python src/eda.py
python src/preprocessing.py
python src/modeling.py
python src/modeling_xgb.py
python src/threshold_optimization_xgb.py
python src/explainability_shap.py
```
Le nouvel artefact doit ensuite être copié dans `artifacts/model.joblib`
avec un `artifacts/model_metadata.json` à jour (`version`, `sha256`,
métriques) avant d'être redéployé — voir [DEPLOYMENT.md](DEPLOYMENT.md).

## Sécurité

Points appliqués (voir aussi `config/settings.py`) :
- Authentification obligatoire sur le formulaire et le dashboard
  (`@login_required`), clé API (`X-API-Key`) réellement vérifiée sur
  `/api/score/` (fail-closed si non configurée)
- Rate limiting sur le login et l'API (`django-ratelimit`)
- En production (`DEBUG=False`) : redirection HTTPS, HSTS, cookies
  `Secure`/`HttpOnly`, `X-Frame-Options: DENY`
- `SECRET_KEY` et `DATABASE_URL` obligatoires en production (l'application
  refuse de démarrer sinon plutôt que de tourner avec des valeurs par
  défaut non sécurisées)
- Postgres en production (SQLite réservé au développement local)
- Piste d'audit complète sur chaque décision (analyste, IP, version modèle)
- Scan de sécurité automatique en CI (`bandit`) + lint (`ruff`)

## Démarrage rapide

```bash
cp .env.example .env    # renseigner SECRET_KEY, SCORING_API_KEY...
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Avec Docker (Postgres inclus) :
```bash
cp .env.example .env
docker compose up --build
```

## Tests & qualité

```bash
pytest scoring/tests/ -v --cov=scoring --cov-report=term-missing
ruff check scoring src config
bandit -r scoring config -ll
python manage.py check --deploy
```

## Déploiement

Voir [DEPLOYMENT.md](DEPLOYMENT.md) pour le guide complet (Docker Compose
auto-hébergé, Render, Railway) et la checklist de mise en production.

# Fonctionnement du projet — Credit Scoring Platform

Ce document explique **comment le projet fonctionne concrètement**, du
jeu de données brut jusqu'à la décision affichée à l'analyste. Pour
l'installation et le déploiement, voir `README.md` et `DEPLOYMENT.md`.

## Sommaire
1. [Vue d'ensemble](#1-vue-densemble)
2. [Les données](#2-les-données)
3. [Le pipeline de Machine Learning](#3-le-pipeline-de-machine-learning)
4. [La politique de décision](#4-la-politique-de-décision)
5. [L'explicabilité (SHAP)](#5-lexplicabilité-shap)
6. [L'application Django](#6-lapplication-django)
7. [Le cycle de vie d'une requête](#7-le-cycle-de-vie-dune-requête)
8. [Le modèle de données](#8-le-modèle-de-données)
9. [La sécurité, en détail](#9-la-sécurité-en-détail)
10. [Stack technique](#10-stack-technique)

---

## 1. Vue d'ensemble

Le projet répond à un problème métier simple à énoncer, complexe à bien
faire : **étant donné un dossier de crédit, quelle est la probabilité que
le client fasse défaut, et quelle décision en tirer ?**

Il se découpe en deux blocs qui communiquent via un seul point de contact,
le fichier `artifacts/model.joblib` :

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│   BLOC DATA SCIENCE (src/)   │         │   BLOC APPLICATIF (scoring/)  │
│                              │         │                                │
│  CSV → nettoyage → features  │  model  │  Formulaire web / API REST    │
│  → entraînement XGBoost      │ .joblib │  → charge le modèle           │
│  → optimisation de seuils    │────────▶│  → calcule une probabilité    │
│  → analyse SHAP               │         │  → applique la politique      │
│                              │         │  → journalise la décision      │
└─────────────────────────────┘         └──────────────────────────────┘
      (offline, ponctuel)                    (online, à chaque requête)
```

Le bloc data science tourne **hors ligne**, à chaque ré-entraînement. Il
produit un artefact figé (le pipeline sklearn sérialisé). Le bloc
applicatif tourne **en ligne**, charge cet artefact une fois au démarrage
et l'utilise pour scorer chaque nouveau dossier en quelques millisecondes.

---

## 2. Les données

Fichier : `data/credit_risk_dataset.csv` — 32 581 lignes, 12 colonnes.

| Colonne | Description |
|---|---|
| `person_age` | Âge du demandeur |
| `person_income` | Revenu annuel |
| `person_home_ownership` | Statut logement (RENT / OWN / MORTGAGE / OTHER) |
| `person_emp_length` | Ancienneté professionnelle (années) |
| `loan_intent` | Objet du prêt (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION) |
| `loan_grade` | Grade de crédit interne (A à G) |
| `loan_amnt` | Montant demandé |
| `loan_int_rate` | Taux d'intérêt |
| `loan_percent_income` | Ratio montant du prêt / revenu |
| `cb_person_default_on_file` | Antécédent de défaut connu (Y/N) |
| `cb_person_cred_hist_length` | Longueur de l'historique de crédit |
| `loan_status` | **Cible** : 1 = défaut, 0 = remboursé (≈ 21.8% de défaut) |

**Nettoyage** (`src/preprocessing.py`) : suppression des doublons,
correction des valeurs aberrantes (âges > 100 ans, anciennetés
professionnelles impossibles), imputation des valeurs manquantes par la
médiane. Le jeu est ensuite séparé en train/test de façon stratifiée sur
la cible, pour garder la même proportion de défauts dans les deux
échantillons.

---

## 3. Le pipeline de Machine Learning

### 3.1 Feature engineering

Le pipeline sklearn (`ColumnTransformer`) traite deux types de variables :
- **Numériques** (`person_age`, `person_income`, `loan_amnt`,
  `loan_percent_income`, etc.) → standardisées (`StandardScaler`) pour la
  baseline, laissées brutes pour XGBoost (les arbres n'ont pas besoin de
  normalisation).
- **Catégorielles** (`loan_grade`, `person_home_ownership`,
  `loan_intent`, `cb_person_default_on_file`) → encodage one-hot
  (`OneHotEncoder`, `handle_unknown="ignore"` pour ne pas planter sur une
  catégorie jamais vue).

Tout ce traitement est encapsulé dans un `sklearn.Pipeline` unique
(`preprocessor` + `classifier`), ce qui garantit qu'**exactement la même
transformation** est appliquée à l'entraînement et à l'inférence — c'est
ce pipeline complet (pas juste le classifieur) qui est sérialisé dans
`model.joblib`.

### 3.2 Deux modèles comparés

| Modèle | Fichier | AUC ROC (test) |
|---|---|---|
| Baseline — Logistic Regression | `src/modeling.py` | ≈ 0.867 |
| Modèle retenu — XGBoost | `src/modeling_xgb.py` | ≈ 0.927 |

La baseline sert de référence simple et interprétable pour juger si la
complexité additionnelle de XGBoost se justifie (c'est le cas : +6 points
d'AUC). Les métriques complètes du modèle retenu (AUC en validation
croisée, Brier score, statistique de Kolmogorov-Smirnov, tailles
train/test) sont figées dans `artifacts/model_metadata.json` au moment de
l'entraînement — c'est ce fichier qui fait foi pour savoir *quel* modèle
tourne en production et avec quelles performances mesurées.

### 3.3 Ce que le modèle voit réellement

Point important, corrigé dans cette version du projet : la feature
`loan_percent_income` que le modèle utilise pour prédire n'est **pas**
une valeur saisie librement par l'utilisateur. Elle est recalculée à
chaque requête, côté serveur, à partir de deux champs vérifiables :

```python
loan_percent_income = loan_amnt / person_income
```

(voir `scoring/services.py::prepare_input_dataframe`). Le champ
« taux d'endettement déclaré » saisi dans le formulaire est une donnée
métier différente (endettement global du client, toutes charges
confondues) : elle sert de garde-fou d'affichage (règle prudentielle à
35%, cf. `risk_factors()`), mais n'entre jamais dans le calcul du modèle
sous couvert d'être la feature d'entraînement.

---

## 4. La politique de décision

Une probabilité de défaut seule ne suffit pas à un métier : il faut la
transformer en **décision opérationnelle**. Le projet utilise une
politique à trois zones (`scoring/services.py::decision_from_proba`,
logique reproduite pour l'analyse offline dans `src/decision_policy.py`) :

```
proba défaut < t_accept        → ACCEPT   (risque faible)
t_accept ≤ proba < t_reject    → REVIEW   (analyse manuelle)
proba défaut ≥ t_reject        → REJECT   (risque élevé)
```

Les seuils (`artifacts/policy.json`, actuellement `t_accept=0.09`,
`t_reject=0.20`) ne sont pas arbitraires : ils sont choisis en balayant
différentes combinaisons sur le jeu de test (`src/threshold_optimization_xgb.py`)
pour arbitrer entre deux objectifs concurrents :
- **minimiser le taux de défaut parmi les dossiers acceptés** (le vrai
  risque financier pour la banque),
- **garder un taux de révision manuelle raisonnable** (~20%, sinon les
  analystes sont noyés sous les dossiers à examiner à la main).

Avec les seuils retenus : le taux de défaut parmi les acceptés tombe à
**≈ 3.96%**, contre 21.8% sans aucun filtrage — c'est le chiffre qui
mesure concrètement l'apport du modèle.

## 5. L'explicabilité (SHAP)

Un score seul ("17.9% de risque") n'est pas actionnable pour un
analyste : il faut savoir **pourquoi**. Le projet utilise
[SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations),
qui décompose la prédiction en la contribution de chaque feature.

Deux usages, à deux moments différents :

- **Offline, à l'entraînement** (`src/explainability_shap.py`) : génère
  des vues globales sur tout le jeu de test — quelles features pèsent le
  plus en moyenne sur les prédictions du modèle (bar chart, beeswarm) —
  utile pour la documentation modèle et la revue par un data scientist.

- **Online, à chaque requête** (`scoring/services.py::compute_shap_explanations`) :
  un `shap.TreeExplainer` est instancié une seule fois au démarrage de
  l'application (coûteux à créer, bon marché à réutiliser), puis appelé à
  chaque scoring pour décomposer *cette prédiction précise*. Le résultat
  est retourné dans la réponse (`shap.risk_up` / `shap.risk_down`) avec
  des noms de features lisibles (ex. `"Ratio prêt/revenu"` plutôt que
  `num__loan_percent_income`), triés par impact décroissant.

C'est cette seconde partie qui alimente concrètement l'API et l'écran de
résultat : chaque décision est accompagnée des facteurs qui l'ont fait
pencher vers le risque ou au contraire vers la sécurité.

---

## 6. L'application Django

### 6.1 Organisation des apps

- **`config/`** — configuration globale (settings, routage racine, WSGI).
  Toute la configuration variable (secrets, base de données, sécurité)
  vient de variables d'environnement, jamais de valeurs en dur.
- **`scoring/`** — l'unique application métier :
  - `services.py` — le moteur de scoring (chargement modèle, prédiction,
    politique, SHAP). Aucune dépendance à Django : ce module pourrait être
    réutilisé tel quel dans un autre contexte (script batch, notebook).
  - `forms.py` — validation des entrées (bornes métier, cohérence
    âge/ancienneté professionnelle).
  - `views.py` — les vues HTTP : formulaire (`home`), tableau de bord
    (`dashboard`), API (`api_score`), export (`export_csv`/`export_pdf`),
    supervision (`healthz`).
  - `models.py` — `ScoreHistory`, la table qui journalise chaque décision.
  - `auth.py` — vérification de la clé API par comparaison en temps
    constant (`hmac.compare_digest`, pour éviter les attaques par mesure
    de timing).

### 6.2 Deux points d'entrée pour scorer un dossier

1. **Formulaire web** (`/`, `home` dans `views.py`) — destiné à un
   analyste humain connecté. Remplit `CreditForm`, affiche le résultat
   avec les facteurs de risque, enregistre l'historique avec
   `source="WEB"` et l'analyste identifié.

2. **API REST interne** (`POST /api/score/`) — destinée à un autre
   système du SI bancaire (moteur de décision, CRM...). Authentifiée par
   clé API dans l'en-tête `X-API-Key`, pas de session utilisateur. Reçoit
   et renvoie du JSON, enregistre l'historique avec `source="API"`.

Les deux chemins convergent vers **la même fonction** `build_result()`
dans `services.py` : aucune divergence de logique métier possible entre
le formulaire et l'API.

---

## 7. Le cycle de vie d'une requête

Exemple pour un appel API :

```
1. POST /api/score/  { loan_amnt: 10000, person_income: 35000, ... }
2. auth.is_valid_api_key() vérifie X-API-Key (sinon 401)
3. CreditForm valide et nettoie les données (sinon 400 + détail des erreurs)
4. services.prepare_input_dataframe() reconstruit les features
   (recalcule loan_percent_income, normalise les catégories en majuscules...)
5. model.predict_proba() → probabilité de défaut
6. decision_from_proba() → ACCEPT / REVIEW / REJECT selon policy.json
7. compute_shap_explanations() → facteurs qui expliquent CE score précis
8. risk_factors() → règles métier lisibles (endettement, taux, ancienneté...)
9. ScoreHistory.objects.create(...) → piste d'audit (IP, version modèle, source)
10. Réponse JSON renvoyée au système appelant
```

Chaque étape peut échouer proprement (modèle absent → 500 explicite,
JSON invalide → 400, clé API invalide → 401) sans jamais faire planter le
processus applicatif.

---

## 8. Le modèle de données

Une seule table métier, `ScoreHistory` :

| Champ | Rôle |
|---|---|
| `client_number`, `loan_amnt`, `person_income`, ... | Copie des données saisies (traçabilité — on doit pouvoir rejouer une décision) |
| `probability`, `decision`, `risk` | Résultat du scoring |
| `analyst` | Qui a déclenché le scoring (NULL si via API) |
| `source` | `WEB` ou `API` |
| `model_version` | Empreinte ou tag de version du modèle utilisé — répond à la question "quel modèle a produit cette décision ?" même après plusieurs ré-entraînements |
| `request_ip` | IP d'origine, à des fins d'audit/sécurité |
| `created_at` | Horodatage, indexé (les requêtes du dashboard filtrent massivement par date) |

Le dashboard (`views.dashboard`) agrège cette table côté base de données
(`Avg`, `Count`, `Sum` via l'ORM Django) plutôt qu'en itérant en Python,
pour rester performant même avec un historique volumineux.

---

## 9. La sécurité, en détail

- **Authentification à deux niveaux** : session Django classique pour les
  humains (`@login_required`), clé API partagée pour les systèmes
  (`X-API-Key` comparée en temps constant). Aucun accès anonyme aux
  données clients.
- **Fail-closed** : si `SCORING_API_KEY` n'est pas configurée,
  `is_valid_api_key()` renvoie toujours `False` — l'API se ferme plutôt
  que de s'ouvrir par défaut.
- **Rate limiting** (`django-ratelimit`) sur `/login/` (10 tentatives/min
  par IP) et `/api/score/` (60 requêtes/min par clé), pour limiter le
  bruteforce et les abus.
- **Séparation dev/prod automatique** : dès que `DEBUG=False`, Django
  active HSTS, cookies `Secure`/`HttpOnly`, redirection HTTPS — sans
  action manuelle supplémentaire.
- **Intégrité du modèle** : si un modèle est récupéré depuis une source
  externe (`MODEL_URL`), son empreinte SHA-256 est vérifiée avant
  utilisation (`MODEL_SHA256`) ; sinon le téléchargement est rejeté.

---

## 10. Stack technique

| Composant | Choix | Pourquoi |
|---|---|---|
| Framework web | Django 5.1 | ORM mature, admin intégré pour la piste d'audit, écosystème sécurité éprouvé |
| Serveur d'application | Gunicorn | Standard pour servir du WSGI Django en production |
| Fichiers statiques | WhiteNoise | Sert les statiques directement depuis l'app, sans CDN externe à configurer pour démarrer |
| Base de données | PostgreSQL (prod) / SQLite (dev) | Postgres pour la concurrence et la fiabilité attendues en prod ; SQLite pour un démarrage local sans dépendance |
| ML | scikit-learn + XGBoost | Pipeline unique reproductible ; XGBoost pour la performance sur données tabulaires |
| Explicabilité | SHAP | Référence académique et industrielle pour l'explicabilité des modèles à base d'arbres |
| Rapports | ReportLab | Génération de PDF côté serveur, sans dépendance à un moteur de rendu externe |
| Conteneurisation | Docker (utilisateur non-root) | Portabilité, reproductibilité, healthcheck intégré |
| Tests | pytest + pytest-django | Couverture des règles métier (`services`, `forms`), de la sécurité (`test_api.py`) et de la politique de décision pure (`test_decision_policy.py`, sans base de données) |
| Qualité | ruff (lint) + bandit (sécurité) | Intégrés en CI, avant tout déploiement |

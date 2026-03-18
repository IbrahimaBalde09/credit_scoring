# Credit Scoring – Data Science & IA (Risk Modeling)

## Objectif
Construire un modèle de scoring crédit pour prédire le risque de défaut (loan_status=1) et définir une politique de décision bancaire (Accept / Review / Reject).

## Données
- 32 581 lignes, 12 variables
- Cible : loan_status (0=non défaut, 1=défaut)
- Proportion défaut : ~21.8%
- Nettoyage : suppression doublons + correction anomalies (âge/ancienneté) + imputation médiane

## Méthode
### Baseline
- Pipeline sklearn : preprocessing (OneHotEncoder + StandardScaler) + Logistic Regression
- AUC ROC : ~0.867

### Modèle avancé
- Pipeline sklearn : preprocessing (OneHotEncoder) + XGBoost
- AUC ROC : ~0.940

## Politique de décision (3 zones)
- ACCEPT si proba défaut < t_accept
- REVIEW si t_accept <= proba < t_reject
- REJECT si proba défaut >= t_reject

Optimisation pour viser review ~ 15–25% et minimiser le taux de défaut des acceptés.

Meilleure policy (XGBoost) :
- t_accept=0.10, t_reject=0.30
- Accept rate ≈ 60%
- Review rate ≈ 20%
- Accepted default rate ≈ 3.14% (vs 21.8% global)

## Explicabilité (SHAP)
- SHAP global bar: reports/figures/shap_global_bar.png
- SHAP beeswarm: reports/figures/shap_summary_beeswarm.png
- SHAP local waterfall: reports/figures/shap_local_waterfall.png

## Résultats clés
- Réduction du risque accepté (default rate) : ~21.8% → ~3.14%
- Modèle robuste + interprétable (SHAP)
- Prêt pour intégration API / batch scoring

## Structure projet
- src/ : scripts
- reports/figures/ : figures SHAP
- models/ : modèles sauvegardés (optionnel)

## Run
```bash
python src/eda.py
python src/preprocessing.py
python src/modeling.py
python src/modeling_xgb.py
python src/threshold_optimization_xgb.py
python src/explainability_shap.py

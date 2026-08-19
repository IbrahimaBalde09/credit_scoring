# Guide de déploiement

Trois options, de la plus simple à la plus contrôlée. Dans tous les cas,
génère d'abord de vraies valeurs pour `SECRET_KEY` et `SCORING_API_KEY` :

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Option A — Docker Compose (auto-hébergé, VM ou serveur interne)

C'est l'option la plus proche d'un vrai déploiement bancaire on-premise :
tu contrôles entièrement la base Postgres et le réseau.

1. Copier et renseigner l'environnement :
   ```bash
   cp .env.example .env
   # éditer .env : SECRET_KEY, SCORING_API_KEY, DJANGO_SUPERUSER_*
   ```
2. Vérifier que `artifacts/model.joblib`, `artifacts/policy.json` et
   `artifacts/model_metadata.json` sont présents (ils sont montés en
   lecture seule dans le conteneur `web`).
3. Lancer :
   ```bash
   docker compose --env-file .env up --build -d
   ```
   `docker-compose.yml` démarre Postgres (avec un healthcheck), applique
   les migrations, collecte les statiques, crée le premier compte
   analyste (si `DJANGO_SUPERUSER_*` est renseigné) puis lance gunicorn.
4. Vérifier :
   ```bash
   curl http://localhost:8000/healthz/
   ```
5. En production réelle, mettre un reverse proxy TLS devant (Nginx,
   Caddy, Traefik) et pointer `ALLOWED_HOSTS` / `CSRF_TRUSTED_ORIGINS` sur
   ton nom de domaine.

**Sauvegardes** : le volume `pgdata` contient toutes les données
(historique de scoring). Planifie un `pg_dump` régulier hors du conteneur.

---

## Option B — Render (PaaS, le plus rapide pour une démo/POC)

Le projet inclut un blueprint `render.yaml` prêt à l'emploi, configuré sur
le **plan gratuit** (pas de carte bancaire nécessaire).

⚠️ Limites du plan gratuit à connaître : le service web se met en veille
après 15 minutes d'inactivité (redémarre en 30-60s au prochain accès), et
la base Postgres gratuite expire après 90 jours (à recréer, ou à passer
sur un plan payant avant l'échéance si le projet devient réel).

1. Pousser le code sur un dépôt Git (GitHub/GitLab).
2. Sur [render.com](https://render.com) : **New > Blueprint**, sélectionner
   le dépôt. Render détecte `render.yaml` et propose de créer :
   - le service web (build via `Dockerfile`)
   - une base Postgres managée, reliée automatiquement via `DATABASE_URL`
3. `SECRET_KEY` et `SCORING_API_KEY` sont générés automatiquement par
   Render (`generateValue: true`). Va dans l'onglet *Environment* du
   service pour les consulter/les communiquer aux systèmes appelants.
4. **Important — artefact modèle** : `artifacts/model.joblib` doit être
   présent dans l'image construite (il est inclus dans ce dépôt et n'est
   pas exclu par `.dockerignore`). Pour un modèle plus volumineux en
   production, préfère Git LFS ou une variable `MODEL_URL` +
   `MODEL_SHA256` pointant vers un stockage objet privé (S3/GCS signé) —
   voir `scoring/services.py::_maybe_fetch_model_from_url`.
5. Une fois déployé, vérifier `https://<ton-service>.onrender.com/healthz/`.
6. Créer le premier compte analyste :
   ```bash
   # depuis le Shell Render du service, ou en local avec le même DATABASE_URL
   python manage.py createsuperuser
   ```

---

## Option C — Railway

1. `railway init` puis `railway link` sur le repo.
2. Ajouter un plugin **PostgreSQL** (Railway injecte automatiquement
   `DATABASE_URL`).
3. Définir les variables : `SECRET_KEY`, `SCORING_API_KEY`, `DEBUG=False`,
   `ALLOWED_HOSTS=<ton-domaine>.up.railway.app`.
4. Railway détecte le `Dockerfile` et build automatiquement.
5. `railway run python manage.py createsuperuser` pour le premier compte.

---

## Checklist avant mise en production

- [ ] `DEBUG=False`
- [ ] `SECRET_KEY` généré aléatoirement, jamais commité
- [ ] `SCORING_API_KEY` généré aléatoirement, communiqué de façon sécurisée
      aux systèmes appelants (pas par email en clair)
- [ ] `DATABASE_URL` pointe vers un Postgres managé avec sauvegardes
      automatiques
- [ ] `ALLOWED_HOSTS` et `CSRF_TRUSTED_ORIGINS` limités au(x) domaine(s) réel(s)
- [ ] TLS actif de bout en bout (`SECURE_SSL_REDIRECT=True`)
- [ ] Premier compte analyste créé, mot de passe fort, pas de compte partagé
- [ ] `SENTRY_DSN` configuré si tu veux du monitoring d'erreurs
- [ ] `artifacts/model_metadata.json` à jour (version, date d'entraînement,
      métriques) — c'est ce qui apparaît dans l'audit trail et l'API
- [ ] Sauvegardes de la base testées (restauration, pas juste l'export)
- [ ] CI verte (`ruff`, `bandit`, `pytest`, `check --deploy`) avant tout
      déploiement
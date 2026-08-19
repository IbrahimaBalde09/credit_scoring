"""
Configuration Django — Credit Scoring Platform.

Principes appliqués :
- 12-factor app : toute la configuration variable vient de l'environnement,
  jamais de valeurs sensibles en dur dans le code.
- "Secure by default" : dès que DEBUG=False, les protections de sécurité
  production (HTTPS, cookies sécurisés, HSTS...) sont activées automatiquement.
- Base de données : SQLite en local par défaut, Postgres en production via
  la variable DATABASE_URL (standard 12-factor).
"""
from pathlib import Path
import os
import sys

import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Sécurité de base
# --------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-dev-only-never-use-in-prod")
DEBUG = os.environ.get("DEBUG", "False") == "True"

if not DEBUG and SECRET_KEY == "django-insecure-dev-only-never-use-in-prod":
    raise RuntimeError(
        "SECRET_KEY n'est pas défini en production. "
        "Génère une vraie valeur et passe-la via la variable d'environnement SECRET_KEY."
    )

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]
ALLOWED_HOSTS.append(".onrender.com")
ALLOWED_HOSTS.append(".railway.app")

render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    ALLOWED_HOSTS.append(render_host)

CSRF_TRUSTED_ORIGINS = [
    o.strip() for o in os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()
]

# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "scoring",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --------------------------------------------------------------------------
# Base de données
# --------------------------------------------------------------------------
# En local (pas de DATABASE_URL) : SQLite, pratique pour développer.
# En production : Postgres, obligatoire via DATABASE_URL (jamais SQLite,
# qui ne supporte pas les accès concurrents attendus d'un service bancaire).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=600,
            conn_health_checks=True,
            ssl_require=os.environ.get("DATABASE_SSL_REQUIRE", "False") == "True",
        )
    }
else:
    if not DEBUG and "test" not in sys.argv and "pytest" not in sys.modules:
        raise RuntimeError(
            "DATABASE_URL n'est pas défini en production. "
            "Configure une base Postgres (voir DEPLOYMENT.md)."
        )
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --------------------------------------------------------------------------
# Authentification
# --------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 10}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"

# Session : expire à la fermeture du navigateur + timeout d'inactivité
# (bonne pratique pour un outil manipulant des données financières clients).
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = int(os.environ.get("SESSION_COOKIE_AGE", 60 * 30))  # 30 min

# --------------------------------------------------------------------------
# Sécurité production (activée automatiquement dès que DEBUG=False)
# --------------------------------------------------------------------------
if not DEBUG:
    SECURE_SSL_REDIRECT = os.environ.get("SECURE_SSL_REDIRECT", "True") == "True"
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    CSRF_COOKIE_HTTPONLY = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    SECURE_HSTS_SECONDS = 31536000  # 1 an
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = "DENY"

# --------------------------------------------------------------------------
# Internationalisation
# --------------------------------------------------------------------------
LANGUAGE_CODE = "fr-fr"
TIME_ZONE = "Europe/Paris"
USE_I18N = True
USE_TZ = True

# --------------------------------------------------------------------------
# Fichiers statiques
# --------------------------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# --------------------------------------------------------------------------
# API interne (M2M) — clé partagée pour /api/score/
# --------------------------------------------------------------------------
# Aucune valeur par défaut : si non configurée, l'API refuse toute requête
# (fail-closed), voir scoring/auth.py.
SCORING_API_KEY = os.environ.get("SCORING_API_KEY", "")

# --------------------------------------------------------------------------
# Rate limiting (django-ratelimit) — protège /login/ et /api/score/
# --------------------------------------------------------------------------
RATELIMIT_ENABLE = os.environ.get("RATELIMIT_ENABLE", "True") == "True"
RATELIMIT_USE_CACHE = "default"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

# --------------------------------------------------------------------------
# Modèle ML — artefacts versionnés
# --------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "artifacts" / "model.joblib"))
POLICY_PATH = os.environ.get("POLICY_PATH", str(BASE_DIR / "artifacts" / "policy.json"))
# URL de repli optionnelle (ex: stockage objet interne S3/GCS signé) utilisée
# uniquement si le fichier n'est pas déjà présent dans l'image/le volume.
# Laisser vide en fonctionnement normal : le modèle doit être livré avec
# l'image Docker (voir DEPLOYMENT.md), pas téléchargé au démarrage.
MODEL_URL = os.environ.get("MODEL_URL", "")
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "")

# --------------------------------------------------------------------------
# Monitoring (Sentry, optionnel)
# --------------------------------------------------------------------------
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.django import DjangoIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[DjangoIntegration()],
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            send_default_pii=False,
            environment=os.environ.get("ENVIRONMENT", "production"),
        )
    except ImportError:
        pass

# --------------------------------------------------------------------------
# Logging structuré (audit — traçabilité des décisions de crédit)
# --------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": os.environ.get("LOG_LEVEL", "INFO"),
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.environ.get("DJANGO_LOG_LEVEL", "WARNING"),
            "propagate": False,
        },
        "scoring": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "scoring.audit": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

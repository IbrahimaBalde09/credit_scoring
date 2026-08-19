#!/bin/sh
set -e

echo "==> Application des migrations..."
python manage.py migrate --noinput

echo "==> Collecte des fichiers statiques..."
python manage.py collectstatic --noinput

if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo "==> Vérification du compte superutilisateur initial..."
    python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
username = '$DJANGO_SUPERUSER_USERNAME'
if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(
        username=username,
        email='$DJANGO_SUPERUSER_EMAIL',
        password='$DJANGO_SUPERUSER_PASSWORD',
    )
    print(f'Superutilisateur {username} créé.')
else:
    print(f'Superutilisateur {username} existe déjà.')
"
fi

echo "==> Démarrage de gunicorn..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WEB_CONCURRENCY:-3} \
    --timeout 60 \
    --access-logfile - \
    --error-logfile -

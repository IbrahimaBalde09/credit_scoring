"""Authentification simple par clé API pour l'endpoint machine-à-machine
scoring/api/score/. Pensé pour une intégration interne au SI bancaire
(un autre backend qui appelle ce service), pas pour un accès public.
"""
import hmac

from django.conf import settings


def is_valid_api_key(request) -> bool:
    expected = settings.SCORING_API_KEY
    if not expected:
        # Aucune clé configurée -> API désactivée par sécurité (fail-closed)
        return False

    provided = request.headers.get("X-API-Key", "")
    return hmac.compare_digest(str(provided), str(expected))

import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from scoring.models import ScoreHistory

VALID_PAYLOAD = {
    "client_number": "CLT-API-0001",
    "loan_amnt": 10000,
    "person_income": 35000,
    "loan_int_rate": 9.5,
    "debt_ratio": 20,
    "person_age": 30,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": 0,
    "cb_person_cred_hist_length": 4,
    "loan_grade": "B",
}


class ApiScoreTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_missing_api_key_is_rejected(self):
        response = self.client.post(
            reverse("api_score"), data=json.dumps(VALID_PAYLOAD), content_type="application/json"
        )
        self.assertEqual(response.status_code, 401)

    def test_wrong_api_key_is_rejected(self):
        response = self.client.post(
            reverse("api_score"),
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
            HTTP_X_API_KEY="wrong-key",
        )
        self.assertEqual(response.status_code, 401)

    def test_valid_request_with_api_key_succeeds(self):
        from django.conf import settings

        response = self.client.post(
            reverse("api_score"),
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
            HTTP_X_API_KEY=settings.SCORING_API_KEY,
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("decision", body)
        self.assertIn("probability", body)
        self.assertEqual(ScoreHistory.objects.filter(client_number="CLT-API-0001").count(), 1)
        self.assertEqual(ScoreHistory.objects.first().source, "API")

    def test_invalid_payload_returns_400(self):
        from django.conf import settings

        bad_payload = {**VALID_PAYLOAD, "person_age": 5}  # < min_value=18
        response = self.client.post(
            reverse("api_score"),
            data=json.dumps(bad_payload),
            content_type="application/json",
            HTTP_X_API_KEY=settings.SCORING_API_KEY,
        )
        self.assertEqual(response.status_code, 400)


class DashboardAuthTests(TestCase):
    def test_dashboard_requires_login(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_home_requires_login(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 302)

    def test_dashboard_accessible_when_logged_in(self):
        User = get_user_model()
        User.objects.create_user(username="analyst", password="testpass123!")
        self.client.login(username="analyst", password="testpass123!")
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)


class HealthzTests(TestCase):
    def test_healthz_ok(self):
        response = self.client.get(reverse("healthz"))
        self.assertIn(response.status_code, (200, 503))
        body = response.json()
        self.assertIn("model_loaded", body)

from django.contrib.auth import views as auth_views
from django.urls import path

from .views import home, dashboard, api_score, export_csv, export_pdf, healthz, RateLimitedLoginView

urlpatterns = [
    path("", home, name="home"),
    path("dashboard/", dashboard, name="dashboard"),
    path("api/score/", api_score, name="api_score"),
    path("dashboard/export/csv/", export_csv, name="export_csv"),
    path("dashboard/export/pdf/", export_pdf, name="export_pdf"),
    path("healthz/", healthz, name="healthz"),
    path("login/", RateLimitedLoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]

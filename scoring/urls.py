from django.urls import path
from .views import home, dashboard, api_score, export_csv, export_pdf

urlpatterns = [
    path("", home, name="home"),
    path("dashboard/", dashboard, name="dashboard"),
    path("api/score/", api_score, name="api_score"),
    path("dashboard/export/csv/", export_csv, name="export_csv"),
    path("dashboard/export/pdf/", export_pdf, name="export_pdf"),
]
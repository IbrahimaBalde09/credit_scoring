# config/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Auth (si tu as accounts)
    path("accounts/", include("accounts.urls")),

    # Planning
    path("planning/", include("planning.urls")),
]

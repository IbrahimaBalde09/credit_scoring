from django.contrib import admin

from .models import ScoreHistory


@admin.register(ScoreHistory)
class ScoreHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "client_number",
        "decision",
        "probability",
        "risk",
        "loan_amnt",
        "person_income",
        "analyst",
        "source",
        "model_version",
    )
    list_filter = ("decision", "risk", "source", "created_at", "loan_grade", "person_home_ownership")
    search_fields = ("client_number", "loan_grade", "loan_intent", "person_home_ownership")
    ordering = ("-created_at",)
    readonly_fields = [f.name for f in ScoreHistory._meta.fields]

    def has_add_permission(self, request):
        # L'historique de scoring ne doit être créé que via le moteur de scoring,
        # jamais saisi manuellement dans l'admin (intégrité de l'audit trail).
        return False

    def has_change_permission(self, request, obj=None):
        return False

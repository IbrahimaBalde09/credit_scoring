from django.contrib import admin
from .models import ScoreHistory


@admin.register(ScoreHistory)
class ScoreHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "decision",
        "probability",
        "risk",
        "loan_amnt",
        "person_income",
    )
    list_filter = ("decision", "risk", "created_at", "loan_grade", "person_home_ownership")
    search_fields = ("loan_grade", "loan_intent", "person_home_ownership")
    ordering = ("-created_at",)
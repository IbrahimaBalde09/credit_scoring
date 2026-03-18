from django.db import models


class ScoreHistory(models.Model):
    DECISION_CHOICES = [
        ("ACCEPT", "Accepté"),
        ("REVIEW", "À examiner"),
        ("REJECT", "Refusé"),
    ]

    client_number = models.CharField("Numéro client", max_length=50)

    loan_amnt = models.FloatField("Montant du prêt")
    person_income = models.FloatField("Revenu annuel")
    loan_int_rate = models.FloatField("Taux d'intérêt")
    debt_ratio = models.FloatField("Taux d'endettement (%)")
    person_age = models.PositiveIntegerField("Âge")
    person_emp_length = models.FloatField("Ancienneté emploi")
    cb_person_cred_hist_length = models.FloatField("Historique de crédit")

    loan_grade = models.CharField("Grade de crédit", max_length=10, blank=True)
    person_home_ownership = models.CharField("Situation logement", max_length=30, blank=True)
    loan_intent = models.CharField("Objet du prêt", max_length=50, blank=True)
    cb_person_default_on_file = models.CharField("Antécédent de défaut", max_length=5, blank=True)

    probability = models.FloatField("Probabilité de défaut")
    decision = models.CharField("Décision", max_length=10, choices=DECISION_CHOICES)
    risk = models.CharField("Niveau de risque", max_length=20)

    created_at = models.DateTimeField("Date de scoring", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Historique de scoring"
        verbose_name_plural = "Historiques de scoring"

    def __str__(self):
        return f"{self.client_number} - {self.get_decision_display()} - {self.probability:.2f}% - {self.created_at:%Y-%m-%d %H:%M}"
from django import forms

# Ces choix doivent impérativement couvrir toutes les catégories vues par le
# modèle à l'entraînement (voir data/credit_risk_dataset.csv). Une catégorie
# manquante ici serait rejetée par le formulaire alors qu'elle est valide
# pour le modèle -> régression testée dans scoring/tests/test_forms.py.
GRADE_CHOICES = [
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("D", "D"),
    ("E", "E"),
    ("F", "F"),
    ("G", "G"),
]

HOME_CHOICES = [
    ("RENT", "Location"),
    ("OWN", "Propriétaire"),
    ("MORTGAGE", "Crédit immobilier"),
    ("OTHER", "Autre"),
]

LOAN_INTENT_CHOICES = [
    ("PERSONAL", "Personnel"),
    ("EDUCATION", "Éducation"),
    ("MEDICAL", "Médical"),
    ("VENTURE", "Business"),
    ("HOMEIMPROVEMENT", "Amélioration logement"),
    ("DEBTCONSOLIDATION", "Rachat de crédit"),
]

YES_NO = [
    (1, "Oui"),
    (0, "Non"),
]

# Âge minimum légal pour travailler (utilisé pour la règle de cohérence
# âge / ancienneté professionnelle ci-dessous).
MIN_WORKING_AGE = 16


class CreditForm(forms.Form):
    client_number = forms.CharField(
        label="Numéro client",
        max_length=50,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Ex. CLT-2026-0001"})
    )

    loan_amnt = forms.IntegerField(
        label="Montant du prêt (€)",
        min_value=500,
        max_value=100000,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    person_income = forms.IntegerField(
        label="Revenu annuel (€)",
        min_value=1000,
        max_value=1000000,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    loan_int_rate = forms.FloatField(
        label="Taux d'intérêt (%)",
        min_value=0,
        max_value=40,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    debt_ratio = forms.FloatField(
        label="Taux d'endettement déclaré (%)",
        min_value=0,
        max_value=100,
        help_text="Endettement global déclaré par le client (toutes charges de crédit confondues). "
                   "Le ratio prêt/revenu utilisé par le modèle est recalculé automatiquement à partir "
                   "du montant du prêt et du revenu.",
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    person_age = forms.IntegerField(
        label="Âge",
        min_value=18,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    person_home_ownership = forms.ChoiceField(
        label="Situation logement",
        choices=HOME_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"})
    )

    person_emp_length = forms.IntegerField(
        label="Ancienneté emploi (années)",
        min_value=0,
        max_value=50,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    loan_intent = forms.ChoiceField(
        label="Objet du prêt",
        choices=LOAN_INTENT_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"})
    )

    cb_person_default_on_file = forms.ChoiceField(
        label="Antécédent de défaut",
        choices=YES_NO,
        widget=forms.Select(attrs={"class": "form-control"})
    )

    cb_person_cred_hist_length = forms.IntegerField(
        label="Historique de crédit (années)",
        min_value=0,
        max_value=50,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    loan_grade = forms.ChoiceField(
        label="Grade de crédit",
        choices=GRADE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"})
    )

    def clean(self):
        cleaned_data = super().clean()
        age = cleaned_data.get("person_age")
        emp_length = cleaned_data.get("person_emp_length")

        # Règle de cohérence métier : on ne peut pas avoir travaillé plus
        # longtemps que (âge - âge légal minimum de travail). Ex: un client
        # de 20 ans ne peut pas avoir 40 ans d'ancienneté professionnelle.
        # Cette incohérence, si non détectée, fausserait silencieusement le
        # score (le modèle ferait confiance à une donnée impossible).
        if age is not None and emp_length is not None:
            max_plausible_emp_length = age - MIN_WORKING_AGE
            if emp_length > max_plausible_emp_length:
                self.add_error(
                    "person_emp_length",
                    f"Incohérent avec l'âge déclaré ({age} ans) : l'ancienneté "
                    f"professionnelle ne peut pas dépasser {max_plausible_emp_length} ans.",
                )

        return cleaned_data

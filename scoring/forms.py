from django import forms

GRADE_CHOICES = [
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("D", "D"),
    ("E", "E"),
]

HOME_CHOICES = [
    ("RENT", "Location"),
    ("OWN", "Propriétaire"),
    ("MORTGAGE", "Crédit immobilier"),
]

LOAN_INTENT_CHOICES = [
    ("PERSONAL", "Personnel"),
    ("EDUCATION", "Éducation"),
    ("MEDICAL", "Médical"),
    ("VENTURE", "Business"),
    ("HOMEIMPROVEMENT", "Amélioration logement"),
]

YES_NO = [
    (1, "Oui"),
    (0, "Non"),
]


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
        label="Taux d'endettement (%)",
        min_value=0,
        max_value=100,
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
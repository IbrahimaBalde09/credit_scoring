from scoring.forms import CreditForm


VALID_PAYLOAD = {
    "client_number": "CLT-0001",
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


def test_valid_form_is_valid():
    form = CreditForm(data=VALID_PAYLOAD)
    assert form.is_valid(), form.errors


def test_all_dataset_categories_are_accepted():
    """Régression : les choix du formulaire doivent couvrir toutes les
    catégories réellement présentes dans les données d'entraînement."""
    for grade in ["A", "B", "C", "D", "E", "F", "G"]:
        data = {**VALID_PAYLOAD, "loan_grade": grade}
        assert CreditForm(data=data).is_valid()

    for home in ["RENT", "OWN", "MORTGAGE", "OTHER"]:
        data = {**VALID_PAYLOAD, "person_home_ownership": home}
        assert CreditForm(data=data).is_valid()

    for intent in ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]:
        data = {**VALID_PAYLOAD, "loan_intent": intent}
        assert CreditForm(data=data).is_valid()


def test_incoherent_employment_length_rejected():
    data = {**VALID_PAYLOAD, "person_age": 20, "person_emp_length": 40}
    form = CreditForm(data=data)
    assert not form.is_valid()
    assert "person_emp_length" in form.errors


def test_missing_client_number_invalid():
    data = {**VALID_PAYLOAD}
    del data["client_number"]
    form = CreditForm(data=data)
    assert not form.is_valid()

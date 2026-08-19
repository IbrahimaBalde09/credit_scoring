from scoring import services


VALID_DATA = {
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


def test_model_is_available():
    assert services.model_available() is True


def test_build_result_returns_expected_keys():
    result = services.build_result(VALID_DATA)
    for key in ("prob", "decision", "decision_fr", "risk", "thresholds", "model_version", "shap"):
        assert key in result
    assert 0 <= result["prob"] <= 100
    assert result["decision"] in {"ACCEPT", "REVIEW", "REJECT"}


def test_decision_from_proba_boundaries():
    services.policy["t_accept"] = 0.1
    services.policy["t_reject"] = 0.3
    assert services.decision_from_proba(0.05)[0] == "ACCEPT"
    assert services.decision_from_proba(0.2)[0] == "REVIEW"
    assert services.decision_from_proba(0.5)[0] == "REJECT"


def test_high_risk_profile_is_rejected():
    risky = {
        **VALID_DATA,
        "debt_ratio": 60,
        "loan_int_rate": 20,
        "loan_grade": "G",
        "cb_person_default_on_file": 1,
        "person_income": 12000,
    }
    result = services.build_result(risky)
    assert result["decision"] == "REJECT"


def test_low_risk_profile_is_accepted():
    safe = {
        **VALID_DATA,
        "debt_ratio": 8,
        "loan_int_rate": 6,
        "loan_grade": "A",
        "cb_person_default_on_file": 0,
        "person_income": 90000,
        "cb_person_cred_hist_length": 15,
    }
    result = services.build_result(safe)
    assert result["decision"] == "ACCEPT"

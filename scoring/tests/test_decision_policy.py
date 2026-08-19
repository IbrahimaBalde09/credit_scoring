"""Tests unitaires purs (pas de DB) pour la logique de politique de décision."""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from decision_policy import apply_policy, policy_metrics  # noqa: E402


def test_apply_policy_accept_below_threshold():
    proba = np.array([0.01, 0.05, 0.5, 0.9])
    decisions = apply_policy(proba, t_accept=0.1, t_reject=0.6)
    assert list(decisions) == [0, 0, 1, 2]


def test_apply_policy_reject_above_threshold():
    proba = np.array([0.7, 0.8])
    decisions = apply_policy(proba, t_accept=0.1, t_reject=0.6)
    assert list(decisions) == [2, 2]


def test_policy_metrics_rates_sum_to_one():
    decisions = np.array([0, 0, 1, 2, 2])
    y_true = np.array([0, 0, 1, 1, 0])
    m = policy_metrics(decisions, y_true)
    total_rate = m["accept_rate"] + m["review_rate"] + m["reject_rate"]
    assert total_rate == pytest.approx(1.0)
    assert m["accepted_count"] == 2
    assert m["rejected_count"] == 2


def test_policy_metrics_no_accepted_returns_nan():
    decisions = np.array([1, 1, 2])
    y_true = np.array([0, 1, 1])
    m = policy_metrics(decisions, y_true)
    assert np.isnan(m["accepted_default_rate"])

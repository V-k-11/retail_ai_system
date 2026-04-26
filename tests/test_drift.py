import pytest

from src.monitoring.drift import check_drift


def test_check_drift_returns_true_below_threshold():
    assert check_drift({"accuracy": 0.7}, threshold=0.75) is True


def test_check_drift_returns_false_at_threshold():
    assert check_drift({"accuracy": 0.75}, threshold=0.75) is False


def test_check_drift_requires_metric():
    with pytest.raises(ValueError):
        check_drift({"roc_auc": 0.8}, metric_name="accuracy")


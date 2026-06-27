# tests/test_backtest_metrics.py
import backtest


def test_hit_rate_true_when_argmax_matches():
    assert backtest.hit_rate((0.6, 0.3, 0.1), 0) is True
    assert backtest.hit_rate((0.2, 0.3, 0.5), 0) is False


def test_metrics_shape_and_values():
    recs = [{"p": (0.7, 0.2, 0.1), "y": 0}, {"p": (0.2, 0.2, 0.6), "y": 2}]
    m = backtest.metrics(recs)
    assert set(m) == {"hit_rate", "log_loss", "brier"}
    assert m["hit_rate"] == 1.0           # both argmax-correct
    assert m["log_loss"] > 0

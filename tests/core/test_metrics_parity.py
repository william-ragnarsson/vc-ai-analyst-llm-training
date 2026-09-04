"""Parity: the hand-written metrics against their scikit-learn oracle.

sklearn is a dev-only dependency and never ships (see the "oracle" entry in AGENTS.md).
It exists so that "I re-derived average precision" is a checkable claim rather than an
assertion -- tie handling in particular, which is where a hand-rolled PR-AUC usually
diverges by a percent or two without anyone noticing.

Run with ``make test-parity``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core import metrics as m

sklearn_metrics = pytest.importorskip("sklearn.metrics", reason="sklearn is a dev extra")

pytestmark = pytest.mark.parity


def _cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260823)
    cases = []

    y = (rng.random(500) < 0.26).astype(np.int8)
    cases.append(("imbalanced_continuous", y, rng.random(500)))

    y = (rng.random(500) < 0.26).astype(np.int8)
    score = np.clip(0.25 + 0.4 * y + rng.normal(0, 0.15, 500), 0.001, 0.999)
    cases.append(("informative_ranker", y, score))

    # Heavy ties are the realistic case here: six ordinal criteria produce many
    # identical score vectors, so many identical predicted probabilities.
    y = (rng.random(300) < 0.3).astype(np.int8)
    cases.append(("heavy_ties", y, np.round(rng.random(300), 1)))

    y = (rng.random(200) < 0.5).astype(np.int8)
    cases.append(("all_tied", y, np.full(200, 0.42)))

    return cases


CASES = _cases()
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize(("_name", "y", "score"), CASES, ids=IDS)
class TestAgainstSklearn:
    def test_pr_auc(self, _name: str, y: np.ndarray, score: np.ndarray) -> None:
        assert m.pr_auc(y, score) == pytest.approx(
            sklearn_metrics.average_precision_score(y, score), abs=1e-12
        )

    def test_roc_auc(self, _name: str, y: np.ndarray, score: np.ndarray) -> None:
        assert m.roc_auc(y, score) == pytest.approx(
            sklearn_metrics.roc_auc_score(y, score), abs=1e-12
        )

    def test_brier_and_log_loss(self, _name: str, y: np.ndarray, score: np.ndarray) -> None:
        assert m.brier(y, score) == pytest.approx(
            sklearn_metrics.brier_score_loss(y, score), abs=1e-12
        )
        assert m.log_loss(y, score) == pytest.approx(
            sklearn_metrics.log_loss(y, score, labels=[0, 1]), abs=1e-9
        )

    def test_threshold_metrics(self, _name: str, y: np.ndarray, score: np.ndarray) -> None:
        pred = (score >= 0.5).astype(np.int8)
        assert m.accuracy(y, pred) == pytest.approx(sklearn_metrics.accuracy_score(y, pred))
        assert m.balanced_accuracy(y, pred) == pytest.approx(
            sklearn_metrics.balanced_accuracy_score(y, pred)
        )
        assert m.f1(y, pred) == pytest.approx(sklearn_metrics.f1_score(y, pred, zero_division=0))
        assert m.mcc(y, pred) == pytest.approx(
            sklearn_metrics.matthews_corrcoef(y, pred), abs=1e-12
        )
        assert m.confusion(y, pred) == tuple(
            sklearn_metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
        )

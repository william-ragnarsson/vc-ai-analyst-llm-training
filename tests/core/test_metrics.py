"""Tests for the scoring module.

The reference lines get the most attention here: at 25.9% prevalence the failure mode
is not a wrong formula, it is reporting 0.74 accuracy as if it meant something.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core import metrics as m


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    """A 26%-positive problem with a mediocre but real ranker."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.26).astype(np.int8)
    score = np.clip(0.25 + 0.35 * y + rng.normal(0, 0.2, 400), 0.001, 0.999)
    return y, score


class TestReferenceLines:
    def test_prevalence_and_majority(self) -> None:
        y = np.array([1, 0, 0, 0], dtype=np.int8)
        assert m.prevalence(y) == 0.25
        assert m.majority_accuracy(y) == 0.75

    def test_always_pass_hits_the_floor_exactly(self, imbalanced: tuple) -> None:
        """The trap this project exists to avoid: a constant model looking competent."""
        y, _ = imbalanced
        always_no = np.zeros_like(y)
        assert m.accuracy(y, always_no) == pytest.approx(m.majority_accuracy(y))
        # ...and the metrics that refuse to be fooled by it.
        assert m.balanced_accuracy(y, always_no) == 0.5
        assert m.mcc(y, always_no) == 0.0
        assert m.f1(y, always_no) == 0.0

    def test_random_ranker_pr_auc_is_the_prevalence(self) -> None:
        rng = np.random.default_rng(7)
        y = (rng.random(20000) < 0.26).astype(np.int8)
        assert m.pr_auc(y, rng.random(20000)) == pytest.approx(m.prevalence(y), abs=0.01)


class TestRankingMetrics:
    def test_perfect_and_inverted_ranking(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        assert m.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
        assert m.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
        assert m.pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_all_tied_scores_give_chance_auc(self) -> None:
        """Every learner that predicts a constant lands here; it must not look good."""
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        assert m.roc_auc(y, np.full(4, 0.5)) == 0.5
        assert m.pr_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)

    def test_pr_auc_is_zero_without_positives(self) -> None:
        assert m.pr_auc(np.zeros(5, dtype=np.int8), np.random.default_rng(0).random(5)) == 0.0

    def test_roc_auc_is_nan_for_one_class(self) -> None:
        assert np.isnan(m.roc_auc(np.ones(5, dtype=np.int8), np.linspace(0, 1, 5)))


class TestCalibration:
    def test_brier_bounds(self) -> None:
        y = np.array([1, 1, 0, 0], dtype=np.int8)
        assert m.brier(y, y.astype(float)) == 0.0
        assert m.brier(y, np.full(4, 0.5)) == 0.25

    def test_log_loss_of_confident_and_wrong_is_finite(self) -> None:
        """Clipping matters: one inf would poison the mean over folds."""
        loss = m.log_loss(np.array([1, 0], dtype=np.int8), np.array([0.0, 1.0]))
        assert np.isfinite(loss) and loss > 30


class TestThresholds:
    def test_best_threshold_beats_the_default_when_0_5_is_wrong(self) -> None:
        """A ranker whose probabilities never reach 0.5 predicts "pass" for everything
        at the default cutoff, despite ranking perfectly."""
        y = np.array([0, 0, 0, 1, 1], dtype=np.int8)
        score = np.array([0.05, 0.10, 0.15, 0.30, 0.35])
        assert m.f1(y, (score >= 0.5).astype(np.int8)) == 0.0
        best = m.best_threshold(y, score)
        assert m.f1(y, (score >= best).astype(np.int8)) == 1.0


class TestEvaluate:
    def test_bundle_is_self_consistent(self, imbalanced: tuple) -> None:
        y, score = imbalanced
        scores = m.evaluate(y, score, threshold=0.4)
        assert scores.n == len(y)
        assert scores.n_pos == int(y.sum())
        assert scores.threshold == 0.4
        assert scores.accuracy == m.accuracy(y, (score >= 0.4).astype(np.int8))
        assert scores.pr_auc == m.pr_auc(y, score)
        assert set(scores.to_dict()) >= {"accuracy", "pr_auc", "roc_auc", "brier"}

    def test_rejects_nan_and_shape_mismatch(self) -> None:
        y = np.array([0, 1], dtype=np.int8)
        with pytest.raises(ValueError, match="NaN"):
            m.evaluate(y, np.array([0.5, np.nan]))
        with pytest.raises(ValueError, match="labels"):
            m.evaluate(y, np.array([0.5, 0.5, 0.5]))

    def test_rejects_non_binary_labels(self) -> None:
        with pytest.raises(ValueError, match="0 and 1"):
            m.evaluate(np.array([0, 2]), np.array([0.5, 0.5]))

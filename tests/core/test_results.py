"""Tests for the runner and the results file.

The learners here are throwaway stand-ins defined in the test -- the real ones start at
rung 0 in ROADMAP.md. What is under test is the harness: that folds are scored against
held-out rows only, that a result carries enough provenance to be trusted, and that a
rerun reproduces it exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from vcml.core.results import (
    RunResult,
    cross_validate,
    load_result,
    reference_lines,
    save_result,
)
from vcml.core.schema import Dataset
from vcml.core.splits import session_kfold
from vcml.models.base import BaseLearner, Learner, NotFittedError, check_X, check_Xy


@dataclass
class ConstantLearner(BaseLearner):
    """Predicts the training base rate for everyone. The shape of a useless model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_Xy(X, y)
        self.n_features_ = X.shape[1]
        self.rate_ = float(y.mean())
        return self._fitted()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.full(check_X(X, self.n_features_).shape[0], self.rate_)


@dataclass
class SingleFeatureLearner(BaseLearner):
    """Ranks by one column, scaled to [0, 1]. Weak but genuinely informative."""

    column: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, _ = check_Xy(X, y)
        self.n_features_ = X.shape[1]
        col = X[:, self.column]
        self.lo_ = float(np.nanmin(col))
        self.hi_ = float(np.nanmax(col))
        return self._fitted()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        col = check_X(X, self.n_features_)[:, self.column]
        span = max(self.hi_ - self.lo_, 1e-12)
        return np.clip(np.nan_to_num((col - self.lo_) / span, nan=0.5), 0.0, 1.0)


@pytest.fixture
def run(toy_ds: Dataset) -> RunResult:
    plan = session_kfold(toy_ds, n_folds=4)
    return cross_validate(SingleFeatureLearner, toy_ds, plan, experiment="test_single_feature")


class TestBaseLearner:
    def test_protocol_is_satisfied(self) -> None:
        assert isinstance(ConstantLearner(), Learner)

    def test_params_are_hyperparameters_not_fitted_state(self, toy_ds: Dataset) -> None:
        learner = SingleFeatureLearner(column=2).fit(toy_ds.X, toy_ds.y)
        assert learner.params() == {"column": 2}
        assert "lo_" not in learner.params()

    def test_predict_before_fit_raises(self, toy_ds: Dataset) -> None:
        with pytest.raises(NotFittedError):
            ConstantLearner().predict_proba(toy_ds.X)

    def test_wrong_width_at_predict_time_raises(self, toy_ds: Dataset) -> None:
        learner = ConstantLearner().fit(toy_ds.X, toy_ds.y)
        with pytest.raises(ValueError, match="fitted on"):
            learner.predict_proba(toy_ds.X[:, :3])

    def test_nan_rejection_is_opt_in(self, toy_ds: Dataset) -> None:
        with pytest.raises(ValueError, match="NaN"):
            check_Xy(toy_ds.X, toy_ds.y, allow_nan=False)


class TestReferenceLines:
    def test_floor_and_ceiling_bracket_the_task(self, toy_ds: Dataset) -> None:
        ref = reference_lines(toy_ds)
        assert 0.0 < ref.prevalence < 0.5
        assert ref.majority_accuracy == pytest.approx(1 - ref.prevalence)
        assert ref.majority_accuracy < ref.ceiling_accuracy <= 1.0
        assert ref.headroom > 0


class TestCrossValidate:
    def test_scores_every_held_out_row_once(self, run: RunResult, toy_ds: Dataset) -> None:
        assert run.pooled.n == toy_ds.n
        assert run.n_folds == 4
        assert len(run.fold_scores) == 4

    def test_records_full_provenance(self, run: RunResult, toy_ds: Dataset) -> None:
        assert run.dataset_fingerprint == toy_ds.fingerprint
        assert run.split_id == session_kfold(toy_ds, n_folds=4).split_id
        assert run.learner == "SingleFeatureLearner"
        assert run.params == {"column": 0}

    def test_a_constant_model_lands_on_the_floor(self, toy_ds: Dataset) -> None:
        """The result the whole reference-line machinery exists to make obvious."""
        plan = session_kfold(toy_ds, n_folds=4)
        result = cross_validate(ConstantLearner, toy_ds, plan, experiment="test_constant")
        assert result.pooled.accuracy == pytest.approx(result.reference.majority_accuracy, abs=0.02)
        assert result.beats_floor_by == pytest.approx(0.0, abs=0.02)
        assert result.pooled.mcc == 0.0

    def test_pooling_can_fake_ranking_signal_for_a_constant_model(self, toy_ds: Dataset) -> None:
        """The one artefact of pooled scoring, pinned so it stays visible.

        A base-rate predictor outputs one number per fold, and those numbers differ
        because each fold's training prevalence differs. Pooled, that between-fold
        variation looks like ranking signal -- the pooled ROC-AUC leaves 0.5 despite the
        model ranking nothing. Per fold, every prediction is tied and the AUC is exactly
        0.5, which is why fold means are recorded next to the pooled value.
        """
        plan = session_kfold(toy_ds, n_folds=4)
        result = cross_validate(ConstantLearner, toy_ds, plan, experiment="test_constant")

        assert result.fold_mean("roc_auc") == pytest.approx(0.5)
        assert result.fold_std("roc_auc") == pytest.approx(0.0)
        assert result.pooled.roc_auc != pytest.approx(0.5, abs=0.01)

    def test_an_informative_model_beats_the_floor_on_ranking(self, run: RunResult) -> None:
        assert run.pooled.roc_auc > 0.55
        assert run.pooled.pr_auc > run.reference.prevalence

    def test_a_fresh_learner_per_fold(self, toy_ds: Dataset) -> None:
        """A factory, not an instance -- otherwise fold 2 inherits fold 1's fit."""
        seen: list[int] = []

        @dataclass
        class Counting(ConstantLearner):
            def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
                seen.append(len(y))
                return super().fit(X, y)

        cross_validate(Counting, toy_ds, session_kfold(toy_ds, n_folds=4), experiment="t")
        assert len(seen) == 4

    def test_mismatched_split_plan_is_refused(self, toy_ds: Dataset) -> None:
        """The failure that silently indexes the wrong rows if it is not caught."""
        other = toy_ds.replace_X(toy_ds.X + 1.0, toy_ds.feature_names)
        plan = session_kfold(other, n_folds=4)
        with pytest.raises(ValueError, match="different dataset"):
            cross_validate(ConstantLearner, toy_ds, plan, experiment="t")

    def test_wrong_prediction_count_is_refused(self, toy_ds: Dataset) -> None:
        @dataclass
        class Truncating(ConstantLearner):
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return super().predict_proba(X)[:-1]

        with pytest.raises(ValueError, match="probabilities for"):
            cross_validate(Truncating, toy_ds, session_kfold(toy_ds, n_folds=4), experiment="t")


class TestReproducibility:
    def test_rerunning_reproduces_the_digest(self, toy_ds: Dataset, run: RunResult) -> None:
        """The claim the project rests on: the same code on the same data gives the
        same number, and the digest is how that is checked rather than asserted."""
        again = cross_validate(
            SingleFeatureLearner,
            toy_ds,
            session_kfold(toy_ds, n_folds=4),
            experiment="test_single_feature",
        )
        assert again.digest == run.digest

    def test_a_different_learner_changes_the_digest(self, toy_ds: Dataset) -> None:
        plan = session_kfold(toy_ds, n_folds=4)
        a = cross_validate(SingleFeatureLearner, toy_ds, plan, experiment="e")
        b = cross_validate(lambda: SingleFeatureLearner(column=2), toy_ds, plan, experiment="e")
        assert a.digest != b.digest

    def test_timestamps_stay_out_of_the_deterministic_block(self, run: RunResult) -> None:
        block = json.dumps(run.run_block())
        assert "created_at" not in block
        assert "runtime_seconds" not in block


class TestResultsFile:
    def test_round_trips_through_disk(self, run: RunResult, tmp_path: Path) -> None:
        path = save_result(run, results_dir=tmp_path)
        assert path.name == "test_single_feature.json"

        loaded = load_result(path)
        assert loaded["digest"] == run.digest
        assert loaded["run"]["pooled"]["accuracy"] == pytest.approx(run.pooled.accuracy)
        assert loaded["run"]["reference"]["ceiling_accuracy"] > 0
        assert set(loaded["env"]) >= {"created_at", "git_sha", "python", "numpy"}

    def test_json_is_plain_and_sorted(self, run: RunResult, tmp_path: Path) -> None:
        """No numpy scalars leaking into the file, and stable key order so a diff
        between two results is readable."""
        text = save_result(run, results_dir=tmp_path).read_text()
        assert "np.float64" not in text
        assert json.loads(text) == json.loads(text)

    def test_roadmap_line_carries_floor_and_ceiling(self, run: RunResult) -> None:
        line = run.roadmap_line()
        assert "acc" in line and "PR-AUC" in line and "ceiling" in line

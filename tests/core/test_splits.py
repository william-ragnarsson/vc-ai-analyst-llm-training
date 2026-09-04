"""Tests for split construction.

These run on the toy dataset, because what is being tested is the *mechanism* -- that
sessions stay whole, that folds are deterministic, that ids change when membership does.
The one test that touches real data checks that all 23 sessions are actually used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from vcml.core.config import MASTER_SEED
from vcml.core.schema import Dataset
from vcml.core.splits import (
    SplitError,
    assert_no_leakage,
    forward_chaining,
    session_kfold,
)


class TestSessionKFold:
    def test_partitions_every_row_exactly_once(self, toy_ds: Dataset) -> None:
        plan = session_kfold(toy_ds, n_folds=4)
        val_rows = np.concatenate([f.val for f in plan.folds])
        assert np.array_equal(np.sort(val_rows), np.arange(toy_ds.n))

    def test_sessions_never_span_a_fold(self, toy_ds: Dataset) -> None:
        """The core guarantee: scores within a session are relative to that session,
        so a session on both sides of the boundary leaks the comparison."""
        for fold in session_kfold(toy_ds, n_folds=4).folds:
            train_sessions = set(toy_ds.groups[fold.train].tolist())
            val_sessions = set(toy_ds.groups[fold.val].tolist())
            assert not (train_sessions & val_sessions)

    def test_train_and_val_are_complementary(self, toy_ds: Dataset) -> None:
        for fold in session_kfold(toy_ds, n_folds=4).folds:
            assert fold.n_train + fold.n_val == toy_ds.n
            assert not np.intersect1d(fold.train, fold.val).size

    def test_folds_are_roughly_balanced(self, toy_ds: Dataset) -> None:
        plan = session_kfold(toy_ds, n_folds=4)
        sizes = np.array([f.n_val for f in plan.folds])
        assert sizes.max() - sizes.min() <= toy_ds.n // len(plan.folds)

    def test_same_seed_same_folds(self, toy_ds: Dataset) -> None:
        a = session_kfold(toy_ds, n_folds=4, seed=MASTER_SEED)
        b = session_kfold(toy_ds, n_folds=4, seed=MASTER_SEED)
        assert a.split_id == b.split_id
        assert all(np.array_equal(x.val, y.val) for x, y in zip(a.folds, b.folds, strict=True))

    def test_different_seed_different_folds(self, toy_ds: Dataset) -> None:
        assert session_kfold(toy_ds, seed=1).split_id != session_kfold(toy_ds, seed=2).split_id

    def test_too_many_folds_is_an_error(self, toy_factory: Callable[..., Dataset]) -> None:
        small = toy_factory(n_sessions=3, rows_per_session=10)
        with pytest.raises(SplitError, match="cannot fill"):
            session_kfold(small, n_folds=5)


class TestForwardChaining:
    def test_validation_is_always_in_the_future(self, toy_ds: Dataset) -> None:
        """The honest question: could the rubric have predicted the *next* session?"""
        for fold in forward_chaining(toy_ds, n_folds=4).folds:
            assert toy_ds.groups[fold.train].max() < toy_ds.groups[fold.val].min()

    def test_training_set_grows(self, toy_ds: Dataset) -> None:
        sizes = [f.n_train for f in forward_chaining(toy_ds, n_folds=4).folds]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_is_deterministic_without_a_seed(self, toy_ds: Dataset) -> None:
        assert forward_chaining(toy_ds).split_id == forward_chaining(toy_ds).split_id

    def test_refuses_when_history_is_too_short(self, toy_factory: Callable[..., Dataset]) -> None:
        short = toy_factory(n_sessions=5, rows_per_session=10)
        with pytest.raises(SplitError, match="forward-chaining"):
            forward_chaining(short, n_folds=5, min_train_sessions=3)


class TestSplitPlan:
    def test_id_tracks_membership_not_just_name(self, toy_ds: Dataset) -> None:
        a = session_kfold(toy_ds, n_folds=4)
        b = session_kfold(toy_ds, n_folds=5)
        assert a.split_id != b.split_id

    def test_id_tracks_the_dataset(self, toy_ds: Dataset) -> None:
        """A plan built on different data must not silently validate against it."""
        perturbed = toy_ds.replace_X(toy_ds.X + 1.0, toy_ds.feature_names)
        assert session_kfold(toy_ds).split_id != session_kfold(perturbed).split_id

    def test_to_dict_is_json_ready(self, toy_ds: Dataset) -> None:
        d = session_kfold(toy_ds, n_folds=4).to_dict()
        assert d["n_folds"] == 4
        assert len(d["folds"]) == 4
        assert set(d) >= {"name", "seed", "split_id", "dataset_fingerprint"}

    def test_both_plans_pass_the_leakage_gate(self, toy_ds: Dataset) -> None:
        assert_no_leakage(toy_ds, session_kfold(toy_ds, n_folds=4))
        assert_no_leakage(toy_ds, forward_chaining(toy_ds, n_folds=4))


class TestOnRealData:
    def test_all_sessions_are_used(self, real_ds: Dataset) -> None:
        plan = session_kfold(real_ds, n_folds=5)
        covered = {int(s) for f in plan.folds for s in np.unique(real_ds.groups[f.val])}
        assert covered == set(range(23))
        assert sum(f.n_val for f in plan.folds) == 769

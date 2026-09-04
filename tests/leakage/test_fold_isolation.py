"""The leakage gate. Run with ``make leakage``.

Three distinct leaks are possible in this project and each gets a test here:

1. A row appearing in both train and validation.
2. A *session* spanning the boundary -- subtler, because the rows differ, but the
   scores inside one session are relative to each other, so the comparison leaks.
3. A synthetic row derived from a validation row sitting in the training set. This is
   the one augmentation makes easy to do by accident, and it is invisible in the
   metrics: it just makes every augmented model look better than it is.

Each test also checks that the guard *fires* on a deliberately broken plan. A leakage
check that has never failed is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core.schema import Dataset
from vcml.core.splits import (
    Fold,
    SplitError,
    SplitPlan,
    assert_no_leakage,
    forward_chaining,
    session_kfold,
)

pytestmark = pytest.mark.leakage


@pytest.fixture
def augmented(toy_ds: Dataset) -> Dataset:
    """The toy dataset plus 40 synthetic children of known real rows."""
    rng = np.random.default_rng(1)
    parents = rng.choice(toy_ds.n, size=(40, 2), replace=False).astype(np.int32)
    X_syn = np.nan_to_num(toy_ds.X[parents[:, 0]], nan=3.0)
    y_syn = toy_ds.y[parents[:, 0]]
    return toy_ds.concat_synthetic(X_syn, y_syn, parents)


class TestRowAndSessionIsolation:
    def test_clean_plans_pass(self, toy_ds: Dataset) -> None:
        assert_no_leakage(toy_ds, session_kfold(toy_ds, n_folds=4))
        assert_no_leakage(toy_ds, forward_chaining(toy_ds, n_folds=4))

    def test_overlapping_rows_are_caught(self, toy_ds: Dataset) -> None:
        plan = session_kfold(toy_ds, n_folds=4)
        broken = SplitPlan(
            name="broken",
            seed=0,
            dataset_fingerprint=toy_ds.fingerprint,
            folds=(
                Fold(
                    index=0,
                    train=np.arange(toy_ds.n, dtype=np.int32),
                    val=plan.folds[0].val,
                ),
            ),
        )
        with pytest.raises(SplitError, match="both train and val"):
            assert_no_leakage(toy_ds, broken)

    def test_a_split_session_is_caught(self, toy_ds: Dataset) -> None:
        """Rows are disjoint here -- only the session check catches this one."""
        session_zero = np.flatnonzero(toy_ds.groups == 0).astype(np.int32)
        half = len(session_zero) // 2
        broken = SplitPlan(
            name="split_session",
            seed=0,
            dataset_fingerprint=toy_ds.fingerprint,
            folds=(Fold(index=0, train=session_zero[:half], val=session_zero[half:]),),
        )
        with pytest.raises(SplitError, match="span the split"):
            assert_no_leakage(toy_ds, broken)


class TestSyntheticIsolation:
    def test_synthetic_rows_never_reach_validation(self, augmented: Dataset) -> None:
        for fold in session_kfold(augmented, n_folds=4).folds:
            assert not augmented.is_synthetic[fold.val].any()

    def test_real_rows_are_still_fully_covered(self, augmented: Dataset) -> None:
        """Augmentation must not quietly shrink the evaluation set."""
        plan = session_kfold(augmented, n_folds=4)
        val_rows = np.concatenate([f.val for f in plan.folds])
        real_rows = np.flatnonzero(~augmented.is_synthetic)
        assert np.array_equal(np.sort(val_rows), real_rows)

    def test_every_synthetic_train_row_has_its_parents_present(self, augmented: Dataset) -> None:
        parents = np.asarray(augmented.meta["synthetic_parents"], dtype=np.int32)
        synthetic = np.flatnonzero(augmented.is_synthetic)
        lookup = {int(r): parents[i] for i, r in enumerate(synthetic)}

        for fold in session_kfold(augmented, n_folds=4).folds:
            train_set = set(fold.train.tolist())
            for row in fold.train[augmented.is_synthetic[fold.train]]:
                assert set(lookup[int(row)].tolist()) <= train_set

    def test_a_descendant_of_a_validation_row_is_caught(self, augmented: Dataset) -> None:
        """The leak augmentation makes easy: a jittered copy of a held-out startup
        sitting in the training set, teaching the model the answer."""
        plan = session_kfold(augmented, n_folds=4)
        fold = plan.folds[0]
        smuggled = int(np.flatnonzero(augmented.is_synthetic)[0])
        parents = np.asarray(augmented.meta["synthetic_parents"], dtype=np.int32)
        parent_row = int(parents[0, 0])

        broken = SplitPlan(
            name="smuggled",
            seed=0,
            dataset_fingerprint=augmented.fingerprint,
            folds=(
                Fold(
                    index=0,
                    train=np.array(sorted({*fold.train.tolist(), smuggled}), dtype=np.int32),
                    val=np.array(sorted({*fold.val.tolist(), parent_row}), dtype=np.int32),
                ),
            ),
        )
        with pytest.raises(SplitError):
            assert_no_leakage(augmented, broken)

    def test_untagged_synthetic_rows_are_rejected(self, toy_ds: Dataset) -> None:
        """Bypassing ``concat_synthetic`` must fail loudly, not silently disable the
        parent check -- which is why AGENTS.md makes that method the only door in."""
        origin = toy_ds.origin.copy()
        origin[:5] = -1
        untagged = Dataset(
            X=toy_ds.X,
            y=toy_ds.y,
            groups=toy_ds.groups,
            origin=origin,
            missing_mask=toy_ds.missing_mask,
            feature_names=toy_ds.feature_names,
            meta={},
        )
        with pytest.raises(SplitError, match="synthetic_parents"):
            session_kfold(untagged, n_folds=4)

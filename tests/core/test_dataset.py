"""Tests for the Dataset container, the loader, and the duplicate audit."""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core.data import duplicate_audit, load_dataset
from vcml.core.schema import Dataset


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load_dataset()


class TestLoader:
    def test_shape_and_balance(self, ds: Dataset) -> None:
        assert ds.n == 769
        assert ds.n_pos == 199
        assert ds.prevalence == pytest.approx(0.2588, abs=1e-3)

    def test_criteria_come_first_and_funding_is_logged(self, ds: Dataset) -> None:
        assert ds.feature_names[:6] == (
            "Team",
            "Technology",
            "Market",
            "Value Proposition",
            "Competitive Advantage",
            "Socially Impactful",
        )
        assert "log1p_funding_usd" in ds.feature_names

    def test_leaky_columns_are_absent(self, ds: Dataset) -> None:
        """The whitelist must keep the second reviewer's verdict and the analyst's
        own notes out of X. Either would make the task trivial and meaningless."""
        joined = " ".join(ds.feature_names).lower()
        assert "wout" not in joined
        assert "note" not in joined

    def test_missing_indicators_are_binary(self, ds: Dataset) -> None:
        for j, name in enumerate(ds.feature_names):
            if name.startswith("missing_"):
                assert set(np.unique(ds.X[:, j]).tolist()) <= {0.0, 1.0}

    def test_origin_starts_as_identity(self, ds: Dataset) -> None:
        """Every row is real and is its own source until an augmenter runs."""
        assert np.array_equal(ds.origin, np.arange(ds.n))
        assert not ds.is_synthetic.any()

    def test_sessions_are_contiguous_ids(self, ds: Dataset) -> None:
        assert sorted(np.unique(ds.groups).tolist()) == list(range(23))


class TestImmutability:
    def test_arrays_are_write_disabled(self, ds: Dataset) -> None:
        """Experiments share a loaded dataset; in-place edits would corrupt folds."""
        for array in (ds.X, ds.y, ds.groups, ds.origin):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_subset_preserves_origin(self, ds: Dataset) -> None:
        idx = np.array([5, 10, 200])
        sub = ds.subset(idx)
        assert sub.n == 3
        assert np.array_equal(sub.origin, idx)
        assert np.array_equal(sub.y, ds.y[idx])

    def test_fingerprint_is_stable_and_content_sensitive(self, ds: Dataset) -> None:
        assert ds.fingerprint == load_dataset().fingerprint
        perturbed = ds.X.copy()
        perturbed[0, 0] += 1.0
        assert ds.replace_X(perturbed, ds.feature_names).fingerprint != ds.fingerprint


class TestDuplicateAudit:
    def test_known_duplicate_structure(self, ds: Dataset) -> None:
        report = duplicate_audit(ds)
        assert report.n_rows == 769
        assert report.n_unique_vectors == 637
        assert report.n_contradictory_vectors == 35
        assert report.n_rows_in_contradictions == 105

    def test_ceiling_is_below_one_and_above_the_majority_rate(self, ds: Dataset) -> None:
        """35 score vectors carry both verdicts, so perfect accuracy is impossible
        from the scores alone. Every model is reported against this bound."""
        report = duplicate_audit(ds)
        assert report.ceiling_accuracy == pytest.approx(0.949, abs=1e-3)
        assert 1 - ds.prevalence < report.ceiling_accuracy < 1.0

    def test_ceiling_is_one_when_no_contradictions(self) -> None:
        n = 8
        ds = Dataset(
            X=np.arange(n, dtype=np.float64).reshape(n, 1),
            y=np.array([0, 1] * (n // 2), dtype=np.int8),
            groups=np.zeros(n, dtype=np.int32),
            origin=np.arange(n, dtype=np.int32),
            missing_mask=np.zeros((n, 1), dtype=bool),
            feature_names=("Team",),
            meta={},
        )
        assert duplicate_audit(ds).ceiling_accuracy == 1.0


class TestSyntheticRows:
    def test_concat_marks_rows_synthetic(self, ds: Dataset) -> None:
        X_syn = np.zeros((3, ds.d))
        parents = np.array([[0], [1], [2]], dtype=np.int32)
        grown = ds.concat_synthetic(X_syn, np.ones(3, dtype=np.int8), parents)
        assert grown.n == ds.n + 3
        assert grown.is_synthetic.sum() == 3
        assert (grown.origin[-3:] == -1).all()
        # Real rows keep their identity, so leakage checks still work.
        assert np.array_equal(grown.origin[: ds.n], ds.origin)

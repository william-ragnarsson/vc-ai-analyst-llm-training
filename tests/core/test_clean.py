"""Tests for the cleaner and its drop ledger.

The sample size is quoted throughout the paper, so it is pinned here. If a future
change to the cleaning rules moves it, these tests fail and the paper's numbers get
re-derived deliberately rather than drifting.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core.clean import (
    NO_SCORES,
    NO_VERDICT,
    NO_VERDICT_NO_SCORES,
    clean_raw,
)
from vcml.core.config import CRITERIA, CRITERION_MAX, CRITERION_MIN, RAW_CSV

# Expected provenance: 822 raw rows -> 769 usable.
EXPECTED_RAW = 822
EXPECTED_DROPPED = {NO_VERDICT_NO_SCORES: 48, NO_VERDICT: 3, NO_SCORES: 2}
EXPECTED_N = 769
EXPECTED_POSITIVES = 199


@pytest.fixture(scope="module")
def cleaned() -> tuple[dict[str, np.ndarray], object, dict[str, object]]:
    """Reads the confidential raw export directly, so it skips without it."""
    if not RAW_CSV.exists():
        pytest.skip("needs data/raw/ (confidential; see AGENTS.md)")
    return clean_raw()


class TestDropLedger:
    def test_row_provenance_is_exact(self, cleaned) -> None:
        _, ledger, _ = cleaned
        assert ledger.n_raw == EXPECTED_RAW
        assert ledger.counts == EXPECTED_DROPPED
        assert ledger.n_dropped == sum(EXPECTED_DROPPED.values()) == 53
        assert ledger.n_retained == EXPECTED_N

    def test_ledger_accounts_for_every_dropped_row(self, cleaned) -> None:
        _, ledger, _ = cleaned
        for reason, count in ledger.counts.items():
            assert len(ledger.dropped_names[reason]) == count

    def test_class_balance(self, cleaned) -> None:
        columns, _, _ = cleaned
        y = columns["y"]
        assert len(y) == EXPECTED_N
        assert int(y.sum()) == EXPECTED_POSITIVES
        assert y.mean() == pytest.approx(0.2588, abs=1e-3)


class TestLabels:
    def test_labels_are_binary(self, cleaned) -> None:
        columns, _, _ = cleaned
        assert set(np.unique(columns["y"]).tolist()) == {0, 1}

    def test_no_unlabelled_row_survives(self, cleaned) -> None:
        """The original cleaner's bug: an empty verdict silently became 'No'."""
        columns, ledger, _ = cleaned
        # Every retained row had a real yes/no; the blanks are all in the ledger.
        assert ledger.counts[NO_VERDICT] + ledger.counts[NO_VERDICT_NO_SCORES] == 51
        assert len(columns["y"]) == EXPECTED_N


class TestCriteria:
    def test_scores_are_nan_or_in_range(self, cleaned) -> None:
        columns, _, _ = cleaned
        for c in CRITERIA:
            values = columns[c]
            present = values[~np.isnan(values)]
            assert present.min() >= CRITERION_MIN
            assert present.max() <= CRITERION_MAX

    def test_zero_is_missing_not_a_rating(self, cleaned) -> None:
        """0 meant 'not filled in'. It must never appear as a numeric score."""
        columns, _, _ = cleaned
        for c in CRITERIA:
            assert not np.any(columns[c] == 0.0)

    def test_every_retained_row_has_at_least_one_score(self, cleaned) -> None:
        columns, _, _ = cleaned
        scores = np.column_stack([columns[c] for c in CRITERIA])
        assert not np.any(np.all(np.isnan(scores), axis=1))


class TestFunding:
    def test_missing_and_zero_are_distinguished(self, cleaned) -> None:
        """Before the fix these were conflated: 171 'zeros' were really 98 absent
        values plus 73 startups that had genuinely raised nothing."""
        columns, _, _ = cleaned
        funding = columns["funding_usd"]
        assert int(np.isnan(funding).sum()) == 98
        assert int((funding[~np.isnan(funding)] == 0.0).sum()) == 73

    def test_scale_is_plausible_after_unit_correction(self, cleaned) -> None:
        columns, _, _ = cleaned
        present = columns["funding_usd"][~np.isnan(columns["funding_usd"])]
        assert present.max() == 350_000_000.0
        assert 50_000 <= np.median(present) <= 500_000

    def test_correlation_sign_is_negative(self, cleaned) -> None:
        """The headline consequence of the unit fix.

        Uncorrected, funding correlated +0.08 with the verdict; corrected it is
        -0.11, i.e. an early-stage fund's analyst mildly preferred *less*-funded
        startups. A regression in the unit logic would flip this back.
        """
        columns, _, _ = cleaned
        funding = np.nan_to_num(columns["funding_usd"], nan=0.0)
        r = np.corrcoef(np.log1p(funding), columns["y"])[0, 1]
        assert r < 0
        assert r == pytest.approx(-0.111, abs=0.01)


class TestSessions:
    def test_sessions_are_chronological(self, cleaned) -> None:
        """Jan-Apr then Nov-Dec. A string sort would put [10/12] before [28/01],
        which would silently corrupt the temporal split."""
        _, _, meta = cleaned
        labels = meta["session_labels"]
        assert labels[0].startswith("[28/01]")
        assert labels[-1].startswith("[10/12]")

    def test_group_ids_cover_all_sessions(self, cleaned) -> None:
        columns, _, meta = cleaned
        groups = columns["groups"]
        assert groups.min() >= 0
        assert groups.max() < len(meta["session_labels"])

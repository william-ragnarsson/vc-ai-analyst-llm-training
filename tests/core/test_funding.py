"""Tests for the funding parse + per-session unit correction.

These guard the single most consequential data fix in the project: the correction
flips the sign of funding's correlation with the target, so a regression here would
silently invert a published conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.core.funding import (
    SessionUnitAmbiguity,
    infer_session_factors,
    parse_funding,
    rescale_funding_by_session,
)


class TestParseFunding:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The export uses a fixed "%.3f" format, so '.' is always a decimal
            # point. The original cleaner's split(".")[0] discarded the fraction.
            ("$350000000.000", 350_000_000.0),
            ("$28600000.000", 28_600_000.0),
            ("$50000.000", 50_000.0),
            ("$0.000", 0.0),
            # In a thousands-denominated session these are 800 and 1.5 *thousand*;
            # the unit conversion happens in rescale_funding_by_session, not here.
            ("$800.000", 800.0),
            ("$1.500", 1.5),
            # No decimal point at all.
            ("$750", 750.0),
            ("1200", 1200.0),
            # Comma thousands separators.
            ("$1,250,000", 1_250_000.0),
            ("$1.5", 1.5),
            ("$10.25", 10.25),
        ],
    )
    def test_parses_known_formats(self, raw: str, expected: float) -> None:
        assert parse_funding(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", None, "n/a", "-"])
    def test_absent_values_become_none(self, raw: str | None) -> None:
        assert parse_funding(raw) is None

    def test_zero_is_a_value_not_a_missing_marker(self) -> None:
        # A startup that has raised nothing is data, not absence.
        assert parse_funding("$0.000") == 0.0
        assert parse_funding("$0.000") is not None


class TestSessionFactors:
    def test_thousands_session_detected(self) -> None:
        values = [800.0, 750.0, 900.0, 250.0]
        factors = infer_session_factors(values, [0, 0, 0, 0])
        assert factors[0] == 1000.0

    def test_dollar_session_detected(self) -> None:
        values = [5_000_000.0, 350_000_000.0, 70_000_000.0]
        factors = infer_session_factors(values, [1, 1, 1])
        assert factors[1] == 1.0

    def test_sessions_are_scaled_independently(self) -> None:
        values = [800.0, 900.0, 5_000_000.0, 70_000_000.0]
        groups = [0, 0, 1, 1]
        factors = infer_session_factors(values, groups)
        assert factors == {0: 1000.0, 1: 1.0}

    def test_session_with_no_funding_data_is_unscaled(self) -> None:
        factors = infer_session_factors([None, None], [3, 3])
        assert factors[3] == 1.0

    def test_mixed_units_raise_rather_than_guess(self) -> None:
        # If the partition assumption ever breaks, stop loudly.
        with pytest.raises(SessionUnitAmbiguity, match="mixes units"):
            infer_session_factors([800.0, 50_000_000.0], [0, 0])


class TestRescale:
    def test_applies_factor_and_preserves_missing(self) -> None:
        values = [800.0, None, 250.0]
        rescaled, factors = rescale_funding_by_session(values, [0, 0, 0])
        assert factors[0] == 1000.0
        assert rescaled[0] == 800_000.0
        assert np.isnan(rescaled[1])
        assert rescaled[2] == 250_000.0

    def test_zero_survives_rescaling_as_zero(self) -> None:
        rescaled, _ = rescale_funding_by_session([0.0, 800.0], [0, 0])
        assert rescaled[0] == 0.0
        assert not np.isnan(rescaled[0])

    def test_dollar_session_unchanged(self) -> None:
        rescaled, _ = rescale_funding_by_session([350_000_000.0], [0])
        assert rescaled[0] == 350_000_000.0

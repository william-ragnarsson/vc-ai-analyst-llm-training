"""Parsing and unit-correction for the funding column.

This module exists because the funding figures in the raw export are wrong in two
independent ways, and getting them right changes the sign of the feature's effect.

Bug 1 -- truncation at the decimal point
----------------------------------------
The export writes every amount with a fixed three decimal places ("%.3f"), so
"$350000000.000" is $350M and "$800.000" is 800.0 in that session's units. The
original cleaner did::

    funding_clean.split(".")[0]      # "$1.500" -> "1"

which silently discards the fractional part. Confirmed against the source: the
smallest non-zero values in dollar-denominated sessions are "$10000.000",
"$30000.000", "$50000.000" -- round dollar amounts with a ".000" tail -- so the
trailing group is a decimal, never a thousands separator.

Bug 2 -- per-session unit conventions
-------------------------------------
Different pitch sessions were typed up in different units. Some rows record whole
dollars ("$350000000.000" = $350M, correct for a late-stage battery company), others
record thousands ("$800.000" = $800k). Crucially the convention is *consistent within
a session* and the two groups are cleanly separated:

    small-unit sessions   max out at        900
    large-unit sessions   start at    5 000 000

Four orders of magnitude of clear air, no session mixing the two. That makes the
correction rule unambiguous: if a session's largest value is under 1000, every value
in that session is in thousands and must be multiplied by 1000.

Effect of the fix
-----------------
Before: median $150, max $350M, corr(log funding, verdict) = +0.082
After:  median $100k, q90 $3M, max $350M, corr(log funding, verdict) = -0.111

The sign flips. Uncorrected, the data says an analyst at an early-stage fund preferred
better-funded startups; corrected, it says the opposite, which is what the fund's
mandate would predict. A data-cleaning bug had inverted the conclusion.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

# A session whose max is below this is assumed to be denominated in thousands.
# Chosen to sit in the empty band between the two observed clusters (900 | 5_000_000).
UNIT_THRESHOLD: float = 1_000.0
THOUSANDS_FACTOR: float = 1_000.0

# Guards for the partition assumption. If a session ever contains values both well
# below and well above the threshold, the "one unit per session" premise has broken
# and we must not guess.
_AMBIGUITY_LOW: float = 1_000.0
_AMBIGUITY_HIGH: float = 1_000_000.0

_CURRENCY_RE = re.compile(r"[^0-9.,\-]")


class SessionUnitAmbiguity(ValueError):
    """A pitch session contains values in what look like two different units.

    Raised rather than warned deliberately. Silently mis-scaling funding by 1000x
    would be invisible in every downstream metric.
    """


def parse_funding(raw: str | float | None) -> float | None:
    """Parse one raw funding cell into that session's units, or None if absent.

    The '.' is always a decimal point -- the export uses a fixed "%.3f" format::

        parse_funding("$350000000.000")  -> 350000000.0
        parse_funding("$800.000")        -> 800.0     (x1000 later, see below)
        parse_funding("$1.500")          -> 1.5
        parse_funding("$0.000")          -> 0.0
        parse_funding("")                -> None

    This returns the number as written. Converting to dollars is the job of
    :func:`rescale_funding_by_session`, because the unit depends on the session.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return None if isinstance(raw, float) and np.isnan(raw) else float(raw)

    text = _CURRENCY_RE.sub("", str(raw).strip())
    if not text:
        return None

    # Commas are thousands separators in this export; strip them outright.
    text = text.replace(",", "")
    if not text or text == "-":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def infer_session_factors(
    values: Sequence[float | None],
    groups: Sequence[int],
) -> dict[int, float]:
    """Infer the unit multiplier for each pitch session.

    Returns ``{session_id: factor}`` where factor is 1000.0 for sessions recorded in
    thousands and 1.0 otherwise. Sessions with no funding data at all get 1.0.

    Raises:
        SessionUnitAmbiguity: if any session holds values on both sides of the gap,
            which would mean the one-unit-per-session premise no longer holds.
    """
    per_session: dict[int, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        if value is not None and value > 0:
            per_session.setdefault(int(group), []).append(float(value))

    factors: dict[int, float] = {}
    for session in set(int(g) for g in groups):
        observed = per_session.get(session, [])
        if not observed:
            factors[session] = 1.0
            continue

        low = [v for v in observed if v < _AMBIGUITY_LOW]
        high = [v for v in observed if v > _AMBIGUITY_HIGH]
        if low and high:
            raise SessionUnitAmbiguity(
                f"session {session} mixes units: {len(low)} value(s) below "
                f"{_AMBIGUITY_LOW:,.0f} and {len(high)} above {_AMBIGUITY_HIGH:,.0f}. "
                "The one-unit-per-session assumption no longer holds; inspect the "
                "source data rather than letting the pipeline guess."
            )

        factors[session] = THOUSANDS_FACTOR if max(observed) < UNIT_THRESHOLD else 1.0

    return factors


def rescale_funding_by_session(
    values: Sequence[float | None],
    groups: Sequence[int],
) -> tuple[np.ndarray, dict[int, float]]:
    """Apply the per-session unit correction.

    Returns:
        (rescaled, factors) where ``rescaled`` is a float64 array with NaN for absent
        values, and ``factors`` maps session id to the multiplier that was applied.

    Note that 0.0 is a legitimate value (a startup that has raised nothing) and is
    kept as 0.0; only a genuinely missing cell becomes NaN.
    """
    factors = infer_session_factors(values, groups)
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, (value, group) in enumerate(zip(values, groups, strict=True)):
        if value is not None:
            out[i] = float(value) * factors[int(group)]
    return out, factors

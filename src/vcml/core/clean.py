"""Raw scoresheet -> analysis-ready table, with an auditable record of every dropped row.

The original cleaner (kept at ``legacy/clean_data.py``) had one catastrophic behaviour:
an empty verdict cell became ``"0"``, i.e. "reject". 51 of 822 rows have no verdict,
and 48 of those were never reviewed at all. Training on them teaches the model that a
blank row means "pass" -- roughly 6% of the data was fabricated supervision.

This module drops those rows instead, and emits a ledger accounting for every single
one so the paper can state its sample size with provenance rather than assertion::

    822 raw rows
     -48  no verdict and no scores  (never reviewed)
     -3   scored but no verdict recorded
     -2   verdict recorded but no scores
    ----
     769  retained  (199 invest / 570 pass, 25.9% positive)

Column selection is a *whitelist* (``config.BASE_FEATURES``). Two raw columns would
destroy the study if they leaked into X -- see the note in ``config.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vcml.core.config import (
    CRITERIA,
    CRITERION_MAX,
    CRITERION_MIN,
    FUNDING_COL,
    GROUP_COL,
    LABEL_COL,
    NAME_COL,
    PROCESSED_DIR,
    RAW_CSV,
)
from vcml.core.funding import parse_funding, rescale_funding_by_session

# Drop reasons, in ledger order.
NO_VERDICT_NO_SCORES = "no_verdict_and_no_scores"
NO_VERDICT = "scored_but_no_verdict"
NO_SCORES = "verdict_but_no_scores"

# Session labels look like "[27/3] Friday Energy Session" -> day 27, month 3.
_SESSION_DATE_RE = re.compile(r"^\[(\d{1,2})/(\d{1,2})\]")


@dataclass(frozen=True, slots=True)
class DropLedger:
    """Accounting for every row that did not survive cleaning."""

    n_raw: int
    counts: dict[str, int]
    dropped_names: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return sum(self.counts.values())

    @property
    def n_retained(self) -> int:
        return self.n_raw - self.n_dropped

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "n_dropped": self.n_dropped,
            "n_retained": self.n_retained,
        }


def _session_sort_key(label: str) -> tuple[int, int, str]:
    """Order sessions chronologically.

    The internship ran Jan-Apr and then Nov-Dec, so a plain string sort would put
    "[10/12]" before "[28/01]". Sorting by (month, day) recovers the real sequence,
    which the temporal split depends on.
    """
    match = _SESSION_DATE_RE.match(label)
    if not match:
        return (99, 99, label)
    day, month = int(match.group(1)), int(match.group(2))
    return (month, day, label)


def _parse_criterion(value: str | None) -> float:
    """Score 1-5, or NaN when the cell was blank or 0.

    0 is *not* a rating. It is how the spreadsheet recorded "not filled in". Feeding
    it as a number would place it below the worst real score and every model would
    extrapolate it as "worse than terrible".
    """
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if not (CRITERION_MIN <= number <= CRITERION_MAX):
        return np.nan
    return number


def _parse_label(value: str | None) -> int | None:
    text = (value or "").strip().lower()
    if text == "yes":
        return 1
    if text == "no":
        return 0
    return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_raw(raw_path: Path = RAW_CSV) -> tuple[dict[str, np.ndarray], DropLedger, dict[str, Any]]:
    """Read the raw export and return (columns, ledger, meta).

    ``columns`` holds aligned arrays for the retained rows:
        - one float64 array per criterion (NaN where missing)
        - ``funding_usd``  float64, NaN where absent, 0.0 where genuinely zero
        - ``y``            int8
        - ``groups``       int32 session index, ordered chronologically
    """
    with raw_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    session_labels = sorted(
        {(row.get(GROUP_COL) or "").strip() for row in rows}, key=_session_sort_key
    )
    session_index = {label: i for i, label in enumerate(session_labels)}

    # Funding units are inferred per session, so this must run over ALL raw rows
    # before any are dropped -- a dropped row still carries unit evidence.
    all_groups = [session_index[(row.get(GROUP_COL) or "").strip()] for row in rows]
    all_funding, factors = rescale_funding_by_session(
        [parse_funding(row.get(FUNDING_COL)) for row in rows], all_groups
    )

    counts: dict[str, int] = {NO_VERDICT_NO_SCORES: 0, NO_VERDICT: 0, NO_SCORES: 0}
    dropped_names: dict[str, list[str]] = {k: [] for k in counts}
    keep: list[int] = []

    for i, row in enumerate(rows):
        label = _parse_label(row.get(LABEL_COL))
        scores = [_parse_criterion(row.get(c)) for c in CRITERIA]
        has_scores = not all(np.isnan(s) for s in scores)
        name = " ".join((row.get(NAME_COL) or "").split()) or f"<row {i}>"

        if label is None and not has_scores:
            reason: str | None = NO_VERDICT_NO_SCORES
        elif label is None:
            reason = NO_VERDICT
        elif not has_scores:
            reason = NO_SCORES
        else:
            reason = None

        if reason is None:
            keep.append(i)
        else:
            counts[reason] += 1
            dropped_names[reason].append(name)

    ledger = DropLedger(n_raw=len(rows), counts=counts, dropped_names=dropped_names)

    columns: dict[str, np.ndarray] = {}
    for c in CRITERIA:
        columns[c] = np.array([_parse_criterion(rows[i].get(c)) for i in keep], dtype=np.float64)
    columns["funding_usd"] = all_funding[keep]
    columns["y"] = np.array([_parse_label(rows[i].get(LABEL_COL)) for i in keep], dtype=np.int8)
    columns["groups"] = np.array([all_groups[i] for i in keep], dtype=np.int32)

    meta: dict[str, Any] = {
        "session_labels": session_labels,
        "session_factors": {session_labels[k]: v for k, v in sorted(factors.items())},
        "row_names": [" ".join((rows[i].get(NAME_COL) or "").split()) for i in keep],
        "raw_sha256": _file_sha256(raw_path),
        "cleaner_sha256": _file_sha256(Path(__file__)),
    }
    return columns, ledger, meta


def build_processed(raw_path: Path = RAW_CSV, out_dir: Path = PROCESSED_DIR) -> dict[str, Any]:
    """`make data`. Write the cleaned table, the drop ledger, and a manifest."""
    columns, ledger, meta = clean_raw(raw_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_names = [*CRITERIA, "funding_usd", "y", "groups"]
    csv_path = out_dir / "reviews.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(field_names)
        for i in range(ledger.n_retained):
            writer.writerow(
                ["" if np.isnan(v) else v for v in (columns[f][i] for f in field_names)]
            )

    (out_dir / "drop_ledger.json").write_text(json.dumps(ledger.to_dict(), indent=2))

    y = columns["y"]
    manifest: dict[str, Any] = {
        "n": int(ledger.n_retained),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_sessions": len(meta["session_labels"]),
        "raw_sha256": meta["raw_sha256"],
        "cleaner_sha256": meta["cleaner_sha256"],
        "session_labels": meta["session_labels"],
        "session_factors": meta["session_factors"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":  # pragma: no cover
    m = build_processed()
    print(f"n = {m['n']}  positives = {m['n_positive']}  ({m['prevalence']:.1%})")

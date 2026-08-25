"""Loading the cleaned table into a :class:`Dataset`, plus the duplicate audit.

The duplicate audit is worth reading before modelling. The six criteria are ordinal
1-5, so there are only 15,625 possible score vectors for 769 rows -- collisions are
guaranteed. What matters is how often two startups with *identical* scoresheets got
*opposite* verdicts, because no model reading only those scores can separate them.
That quantity is a Bayes-error bound: a hard ceiling on achievable accuracy.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vcml.core.clean import build_processed
from vcml.core.config import BASE_FEATURES, CRITERIA, PROCESSED_DIR
from vcml.core.schema import Dataset

FundingTransform = Literal["raw", "log1p", "none"]


class StaleDataError(RuntimeError):
    """The processed table is missing or was built from different inputs."""


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """How much of the data is un-separable by the features alone."""

    n_rows: int
    n_unique_vectors: int
    n_repeated_vectors: int
    n_rows_in_repeats: int
    n_contradictory_vectors: int
    n_rows_in_contradictions: int
    ceiling_accuracy: float

    def summary(self) -> str:
        return (
            f"{self.n_rows} rows -> {self.n_unique_vectors} distinct score vectors; "
            f"{self.n_contradictory_vectors} vectors carry both verdicts "
            f"({self.n_rows_in_contradictions} rows). "
            f"Ceiling accuracy from scores alone: {self.ceiling_accuracy:.1%}"
        )


def duplicate_audit(ds: Dataset, columns: tuple[str, ...] = CRITERIA) -> DuplicateReport:
    """Quantify exact-duplicate feature vectors and the irreducible error they imply.

    The ceiling is computed by giving an oracle the majority verdict for each distinct
    score vector. Rows that disagree with their group's majority can never be got
    right, so::

        ceiling = sum over vectors of max(n_yes, n_no) / n_rows

    This is optimistic (it assumes perfect memorisation of the training distribution),
    which is exactly what makes it a *ceiling*.
    """
    idx = [ds.feature_names.index(c) for c in columns if c in ds.feature_names]
    key_matrix = ds.X[:, idx]

    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(key_matrix):
        groups[tuple(row.tolist())].append(i)

    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    contradictory = {k: v for k, v in groups.items() if len(set(ds.y[v].tolist())) > 1}
    correct = sum(int(np.bincount(ds.y[v], minlength=2).max()) for v in groups.values())

    return DuplicateReport(
        n_rows=ds.n,
        n_unique_vectors=len(groups),
        n_repeated_vectors=len(repeated),
        n_rows_in_repeats=sum(len(v) for v in repeated.values()),
        n_contradictory_vectors=len(contradictory),
        n_rows_in_contradictions=sum(len(v) for v in contradictory.values()),
        ceiling_accuracy=correct / ds.n,
    )


def _read_processed(processed_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    csv_path = processed_dir / "reviews.csv"
    manifest_path = processed_dir / "manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        raise StaleDataError(
            f"{csv_path} not found. Run `make data` (or python -m vcml.core.clean)."
        )

    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        field_names = list(reader.fieldnames or [])

    columns: dict[str, np.ndarray] = {}
    for name in field_names:
        raw = [r[name] for r in rows]
        if name == "y":
            columns[name] = np.array([int(float(v)) for v in raw], dtype=np.int8)
        elif name == "groups":
            columns[name] = np.array([int(float(v)) for v in raw], dtype=np.int32)
        else:
            columns[name] = np.array(
                [np.nan if v == "" else float(v) for v in raw], dtype=np.float64
            )

    manifest = json.loads(manifest_path.read_text())
    return columns, manifest


def load_dataset(
    *,
    processed_dir: Path = PROCESSED_DIR,
    features: tuple[str, ...] = BASE_FEATURES,
    funding_transform: FundingTransform = "log1p",
    add_missing_indicators: bool = True,
    rebuild: bool = False,
) -> Dataset:
    """Load the canonical dataset.

    Args:
        features: whitelist of columns to place in X. Defaults to the six criteria
            plus funding. Nothing outside ``config.BASE_FEATURES`` should be added
            without reading the leakage note in ``config.py``.
        funding_transform: ``log1p`` by default. Funding spans $0 to $350M, so the
            raw variable is unusable by any linear model.
        add_missing_indicators: append a binary column per feature that has missing
            values. Lets a model distinguish "not scored" from an imputed value.
        rebuild: force a rebuild of the processed table first.
    """
    if rebuild or not (processed_dir / "reviews.csv").exists():
        build_processed(out_dir=processed_dir)

    columns, manifest = _read_processed(processed_dir)

    matrix: list[np.ndarray] = []
    names: list[str] = []
    for name in features:
        if name not in columns:
            raise KeyError(f"unknown feature {name!r}; available: {sorted(columns)}")
        values = columns[name]
        if name == "funding_usd":
            if funding_transform == "log1p":
                values = np.log1p(values)
                name = "log1p_funding_usd"
            elif funding_transform == "none":
                continue
        matrix.append(values)
        names.append(name)

    X = np.column_stack(matrix)
    missing_mask = np.isnan(X)

    if add_missing_indicators:
        indicator_cols = [j for j in range(X.shape[1]) if missing_mask[:, j].any()]
        if indicator_cols:
            indicators = missing_mask[:, indicator_cols].astype(np.float64)
            X = np.hstack([X, indicators])
            names.extend(f"missing_{names[j]}" for j in indicator_cols)
            missing_mask = np.hstack([missing_mask, np.zeros(indicators.shape, dtype=bool)])

    n = X.shape[0]
    meta: dict[str, Any] = {
        "manifest": manifest,
        "session_labels": manifest.get("session_labels", []),
        "n_raw": 822,
    }

    return Dataset(
        X=X,
        y=columns["y"],
        groups=columns["groups"],
        origin=np.arange(n, dtype=np.int32),
        missing_mask=missing_mask,
        feature_names=tuple(names),
        meta=meta,
    )

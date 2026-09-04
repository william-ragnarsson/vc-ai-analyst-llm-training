"""Shared fixtures.

Two kinds of dataset live here and the distinction matters.

``real_ds`` is the actual scoresheet. It requires ``data/raw/`` (confidential, not
committed) or an already-built ``data/processed/``, so it *skips* rather than fails when
neither is present. Only tests that pin published numbers -- 769 rows, ceiling 0.949 --
should use it.

``toy_ds`` is generated from ``MASTER_SEED`` and shipped with the repo. Everything that
tests *mechanism* rather than *findings* -- splits, metrics, leakage, learner behaviour
-- uses it, so the suite runs green on a fresh clone and in CI where the raw export will
never exist.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from vcml.core.config import MASTER_SEED, PROCESSED_DIR, RAW_CSV
from vcml.core.data import load_dataset
from vcml.core.schema import Dataset

TOY_FEATURES: tuple[str, ...] = (
    "Team",
    "Technology",
    "Market",
    "Value Proposition",
    "Competitive Advantage",
    "Socially Impactful",
    "log1p_funding_usd",
)


def real_data_available() -> bool:
    """True if the confidential export or a built processed table is on this machine."""
    return RAW_CSV.exists() or (PROCESSED_DIR / "reviews.csv").exists()


requires_real_data = pytest.mark.skipif(
    not real_data_available(),
    reason="needs data/raw/ or data/processed/ (confidential; see AGENTS.md)",
)


def make_toy_dataset(
    *,
    n_sessions: int = 12,
    rows_per_session: int = 20,
    seed: int = MASTER_SEED,
    missing_rate: float = 0.05,
    prevalence: float = 0.26,
) -> Dataset:
    """A structurally faithful stand-in for the real scoresheet.

    Mirrors the properties the code actually depends on: six ordinal 1-5 criteria with
    NaN for unrated, a heavy-tailed funding column, contiguous session ids in
    chronological order, roughly 26% positives, and a real but imperfect signal so a
    learner scores above the floor and below 1.0.
    """
    rng = np.random.default_rng(seed)
    n = n_sessions * rows_per_session

    criteria = rng.integers(1, 6, size=(n, 6)).astype(np.float64)
    funding = np.expm1(rng.normal(11.0, 2.0, size=n)).clip(0.0, 3.5e8)

    # Verdict driven mostly by Team and Market, plus noise -- so the task is learnable
    # but not separable, exactly like the real data.
    weights = np.array([0.9, 0.3, 0.7, 0.4, 0.2, 0.1])
    logits = criteria @ weights + rng.normal(0.0, 1.4, size=n)
    cutoff = float(np.quantile(logits, 1.0 - prevalence))
    y = (logits >= cutoff).astype(np.int8)

    # Unrated criteria become NaN *after* the label is drawn: missingness here is
    # incidental, as in the real export, not itself a signal.
    mask = rng.random((n, 6)) < missing_rate
    criteria[mask] = np.nan

    X = np.column_stack([criteria, np.log1p(funding)])
    groups = np.repeat(np.arange(n_sessions, dtype=np.int32), rows_per_session)

    return Dataset(
        X=X,
        y=y,
        groups=groups,
        origin=np.arange(n, dtype=np.int32),
        missing_mask=np.isnan(X),
        feature_names=TOY_FEATURES,
        meta={"toy": True, "seed": seed},
    )


@pytest.fixture(scope="session")
def toy_ds() -> Dataset:
    return make_toy_dataset()


@pytest.fixture(scope="session")
def toy_factory() -> Callable[..., Dataset]:
    """For tests that need a different shape (more sessions, no missing values, ...)."""
    return make_toy_dataset


@pytest.fixture(scope="session")
def real_ds() -> Dataset:
    if not real_data_available():
        pytest.skip("needs data/raw/ or data/processed/ (confidential; see AGENTS.md)")
    return load_dataset()

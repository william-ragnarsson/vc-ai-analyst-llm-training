"""Cross-validation splits. Deterministic, session-aware, and leakage-proof.

Design notes
------------
*Rows are not independent -- sessions are.* Every startup was scored inside a pitch
session, against the other startups in that session. The analyst's 4 on a Tuesday is not
the same 4 as on a Thursday: it is partly a ranking within that day's batch. A plain
random split therefore puts near-siblings on both sides of the fold boundary and
inflates every score. Whole sessions move together here.

*Two split kinds, answering two different questions.*
  ``session_kfold``     -- "could this rubric predict a verdict at all?" Sessions are
                           shuffled into folds. Uses all 23 sessions as validation data.
  ``forward_chaining``  -- "could it have predicted the *next* session?" Trains only on
                           earlier sessions. Strictly harder and strictly more honest,
                           because that is the only way the model could ever be used.

*Synthetic rows are train-only, and only behind their parents.* A jittered copy of a
validation row sitting in the training set is the same leak as the row itself. Rows with
``origin == -1`` never enter a validation fold, and enter a training fold only if every
one of their parents is already there. ``make leakage`` asserts this.

*Determinism.* Fold membership descends from ``MASTER_SEED`` and the dataset
fingerprint. Two runs on the same data produce the same folds, and the ``split_id``
recorded in every results JSON proves two numbers were computed on the same partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vcml.core.config import MASTER_SEED
from vcml.core.schema import Dataset, fingerprint_arrays


class SplitError(ValueError):
    """The requested split cannot be built from this dataset."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/validation partition, as row indices into the parent dataset."""

    index: int
    train: np.ndarray  # int32 row indices
    val: np.ndarray  # int32 row indices

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_val(self) -> int:
        return int(self.val.size)


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """A named, reproducible set of folds over one dataset."""

    name: str
    seed: int
    folds: tuple[Fold, ...]
    dataset_fingerprint: str

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def split_id(self) -> str:
        """Short hash over the actual membership -- two plans agree iff their ids do."""
        arrays = [a for fold in self.folds for a in (fold.train, fold.val)]
        return fingerprint_arrays(
            *arrays, extra=f"{self.name}|{self.seed}|{self.dataset_fingerprint}"
        )[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "split_id": self.split_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "folds": [
                {"index": f.index, "n_train": f.n_train, "n_val": f.n_val} for f in self.folds
            ],
        }

    def summary(self) -> str:
        sizes = ", ".join(f"{f.n_val}" for f in self.folds)
        return f"{self.name}[{self.split_id}] {self.n_folds} folds, val sizes: {sizes}"


def _synthetic_parents(ds: Dataset) -> dict[int, np.ndarray]:
    """Map each synthetic row index to the parent row indices it was built from."""
    synthetic = np.flatnonzero(ds.is_synthetic)
    if synthetic.size == 0:
        return {}
    parents = ds.meta.get("synthetic_parents")
    if parents is None:
        raise SplitError(
            f"{synthetic.size} rows are tagged synthetic but meta['synthetic_parents'] "
            "is absent. Synthetic rows must be added via Dataset.concat_synthetic."
        )
    parents = np.asarray(parents, dtype=np.int32)
    if parents.shape[0] != synthetic.size:
        raise SplitError(f"{parents.shape[0]} parent records for {synthetic.size} synthetic rows")
    return {int(row): parents[i] for i, row in enumerate(synthetic)}


def _rows_for_sessions(
    ds: Dataset, train_sessions: set[int], val_sessions: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a session assignment into row indices, placing synthetic rows safely."""
    parents = _synthetic_parents(ds)
    real = ~ds.is_synthetic

    train_real = np.flatnonzero(real & np.isin(ds.groups, list(train_sessions)))
    val_real = np.flatnonzero(real & np.isin(ds.groups, list(val_sessions)))

    if parents:
        # A synthetic row is admissible only where all of its ancestors already are.
        train_set = set(train_real.tolist())
        admissible = [
            row
            for row, ancestry in parents.items()
            if set(np.asarray(ancestry).ravel().tolist()) <= train_set
        ]
        train_rows = np.sort(np.concatenate([train_real, np.array(admissible, dtype=np.int64)]))
    else:
        train_rows = train_real

    return train_rows.astype(np.int32), val_real.astype(np.int32)


def _balanced_session_folds(ds: Dataset, n_folds: int, seed: int) -> list[list[int]]:
    """Assign whole sessions to folds, keeping positives evenly spread.

    Sessions vary in size (a dozen to sixty startups) and in how generous the analyst
    was that day. Assigning them round-robin at random leaves folds with wildly
    different base rates, which makes fold-to-fold variance mostly an artefact of the
    partition. Largest-remaining-first greedy assignment fixes that cheaply.
    """
    real = ~ds.is_synthetic
    sessions = np.unique(ds.groups[real])
    if len(sessions) < n_folds:
        raise SplitError(f"{len(sessions)} sessions cannot fill {n_folds} folds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sessions))
    sessions = sessions[order]

    counts = {int(s): int((real & (ds.groups == s)).sum()) for s in sessions}
    positives = {int(s): int(ds.y[real & (ds.groups == s)].sum()) for s in sessions}
    # Largest sessions placed first: the last, smallest ones then fine-tune the balance.
    ordered = sorted(counts, key=lambda s: (-counts[s], s))

    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    bucket_n = [0] * n_folds
    bucket_pos = [0] * n_folds
    for session in ordered:
        target = min(range(n_folds), key=lambda b: (bucket_n[b], bucket_pos[b], b))
        buckets[target].append(session)
        bucket_n[target] += counts[session]
        bucket_pos[target] += positives[session]
    return buckets


def session_kfold(
    ds: Dataset, *, n_folds: int = 5, seed: int = MASTER_SEED, name: str = "session_kfold"
) -> SplitPlan:
    """K-fold CV where whole pitch sessions, never individual rows, move between folds."""
    buckets = _balanced_session_folds(ds, n_folds, seed)
    all_sessions = {int(s) for s in np.unique(ds.groups[~ds.is_synthetic])}

    folds = []
    for i, held_out in enumerate(buckets):
        val_sessions = set(held_out)
        train, val = _rows_for_sessions(ds, all_sessions - val_sessions, val_sessions)
        if val.size == 0:
            raise SplitError(f"fold {i} has an empty validation set")
        folds.append(Fold(index=i, train=train, val=val))

    return SplitPlan(name=name, seed=seed, folds=tuple(folds), dataset_fingerprint=ds.fingerprint)


def forward_chaining(
    ds: Dataset,
    *,
    n_folds: int = 5,
    min_train_sessions: int = 3,
    name: str = "forward_chaining",
) -> SplitPlan:
    """Train on the past, validate on the future.

    Sessions are chronologically ordered by ``clean.py``, so fold *k* trains on
    everything before a cut point and validates on the block just after it. No seed:
    time supplies the ordering, so this split is deterministic by construction.
    """
    sessions = sorted(int(s) for s in np.unique(ds.groups[~ds.is_synthetic]))
    n_sessions = len(sessions)
    if n_sessions < min_train_sessions + n_folds:
        raise SplitError(
            f"{n_sessions} sessions cannot support {n_folds} forward-chaining folds "
            f"with a {min_train_sessions}-session warm-up"
        )

    # Evenly spaced cut points from the warm-up boundary to the final session.
    cuts = np.linspace(min_train_sessions, n_sessions, n_folds + 1).astype(int)

    folds = []
    for i in range(n_folds):
        train_sessions = set(sessions[: cuts[i]])
        val_sessions = set(sessions[cuts[i] : cuts[i + 1]])
        if not val_sessions:
            raise SplitError(f"fold {i} has an empty validation set")
        train, val = _rows_for_sessions(ds, train_sessions, val_sessions)
        folds.append(Fold(index=i, train=train, val=val))

    return SplitPlan(name=name, seed=0, folds=tuple(folds), dataset_fingerprint=ds.fingerprint)


def assert_no_leakage(ds: Dataset, plan: SplitPlan) -> None:
    """Raise if any fold violates the guarantees this module claims to provide.

    Called by ``make leakage`` and cheap enough to call from a runner. Checks that
    train and validation are disjoint, that no session spans the boundary, that no
    synthetic row is being validated on, and that every synthetic training row's
    ancestors are in the same training set.
    """
    parents = _synthetic_parents(ds)

    for fold in plan.folds:
        overlap = np.intersect1d(fold.train, fold.val)
        if overlap.size:
            raise SplitError(f"fold {fold.index}: {overlap.size} rows in both train and val")

        shared = np.intersect1d(ds.groups[fold.train], ds.groups[fold.val])
        if shared.size:
            raise SplitError(f"fold {fold.index}: sessions {shared.tolist()} span the split")

        if ds.is_synthetic[fold.val].any():
            raise SplitError(f"fold {fold.index}: synthetic rows reached the validation set")

        train_set = set(fold.train.tolist())
        for row in fold.train[ds.is_synthetic[fold.train]]:
            ancestry = set(np.asarray(parents[int(row)]).ravel().tolist())
            if not ancestry <= train_set:
                raise SplitError(
                    f"fold {fold.index}: synthetic row {int(row)} descends from a "
                    "row outside its training set"
                )

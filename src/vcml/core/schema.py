"""The ``Dataset`` container passed to every model and experiment.

Design notes
------------
*Immutable.* All arrays are write-disabled. Experiments run in parallel and share a
loaded dataset; a model that quietly standardised ``X`` in place would corrupt every
subsequent fold with no error message.

*NaN means missing.* Never 0 -- see ``config.CRITERIA``.

*Fingerprinted.* The sha256 over the arrays goes into every results JSON, so the paper
build can refuse to mix numbers computed against different versions of the data.

*Origin-tracked.* ``origin[i]`` is the index of the real row that row *i* came from,
or -1 if the row is synthetic. This is what makes augmentation leakage detectable:
the runner can assert that no synthetic row reached a validation fold and that no
validation row was an ancestor of a training row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


def fingerprint_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    """Stable sha256 over array contents plus an optional string."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
    digest.update(extra.encode())
    return digest.hexdigest()


def _freeze(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class Dataset:
    """An immutable feature matrix, target, and the metadata needed to split it safely."""

    X: np.ndarray  # (n, d) float64; NaN == missing
    y: np.ndarray  # (n,) int8 in {0, 1}
    groups: np.ndarray  # (n,) int32 pitch-session index, chronologically ordered
    origin: np.ndarray  # (n,) int32; >= 0 -> source row index, -1 -> synthetic
    missing_mask: np.ndarray  # (n, d) bool, True where the raw value was absent
    feature_names: tuple[str, ...]
    meta: Mapping[str, Any]

    def __post_init__(self) -> None:
        n, d = self.X.shape
        if self.y.shape != (n,):
            raise ValueError(f"y has shape {self.y.shape}, expected ({n},)")
        if self.groups.shape != (n,):
            raise ValueError(f"groups has shape {self.groups.shape}, expected ({n},)")
        if self.origin.shape != (n,):
            raise ValueError(f"origin has shape {self.origin.shape}, expected ({n},)")
        if self.missing_mask.shape != (n, d):
            raise ValueError(
                f"missing_mask has shape {self.missing_mask.shape}, expected ({n},{d})"
            )
        if len(self.feature_names) != d:
            raise ValueError(f"{len(self.feature_names)} feature names for {d} columns")

        for name in ("X", "y", "groups", "origin", "missing_mask"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    # --- shape helpers ------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())

    @property
    def prevalence(self) -> float:
        """Base rate of the positive class -- the floor any PR-AUC must beat."""
        return float(self.y.mean())

    @property
    def is_synthetic(self) -> np.ndarray:
        return self.origin < 0

    @property
    def fingerprint(self) -> str:
        return fingerprint_arrays(self.X, self.y, self.groups, extra="|".join(self.feature_names))

    # --- derivation ---------------------------------------------------------
    def subset(self, idx: np.ndarray) -> Dataset:
        """Row subset, preserving origin so leakage checks survive splitting."""
        idx = np.asarray(idx)
        return replace(
            self,
            X=self.X[idx],
            y=self.y[idx],
            groups=self.groups[idx],
            origin=self.origin[idx],
            missing_mask=self.missing_mask[idx],
        )

    def replace_X(self, X: np.ndarray, feature_names: tuple[str, ...]) -> Dataset:
        """Swap the feature matrix, e.g. after a preprocessing transform."""
        if X.shape[0] != self.n:
            raise ValueError(f"X has {X.shape[0]} rows, dataset has {self.n}")
        return replace(
            self,
            X=X,
            feature_names=feature_names,
            missing_mask=np.zeros(X.shape, dtype=bool),
        )

    def concat_synthetic(
        self, X_syn: np.ndarray, y_syn: np.ndarray, parents: np.ndarray
    ) -> Dataset:
        """Append augmented rows, tagged ``origin = -1`` and with parents recorded.

        ``parents`` is (n_synthetic, k) of indices into the *current* dataset. The
        runner uses it to assert no synthetic row descends from a validation row.
        """
        if X_syn.shape[1] != self.d:
            raise ValueError(f"synthetic X has {X_syn.shape[1]} columns, expected {self.d}")
        if len(y_syn) != len(X_syn):
            raise ValueError("X_syn and y_syn length mismatch")

        meta = dict(self.meta)
        meta["synthetic_parents"] = np.asarray(parents, dtype=np.int32)
        return replace(
            self,
            X=np.vstack([self.X, X_syn]),
            y=np.concatenate([self.y, np.asarray(y_syn, dtype=np.int8)]),
            # Synthetic rows inherit their first parent's session so grouped CV stays
            # coherent; they are excluded from validation folds regardless.
            groups=np.concatenate(
                [self.groups, self.groups[np.asarray(parents, dtype=np.int32)[:, 0]]]
            ),
            origin=np.concatenate([self.origin, np.full(len(X_syn), -1, dtype=np.int32)]),
            missing_mask=np.vstack([self.missing_mask, np.zeros(X_syn.shape, dtype=bool)]),
            meta=meta,
        )

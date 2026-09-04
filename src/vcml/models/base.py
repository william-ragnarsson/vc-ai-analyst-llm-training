"""The one interface every learner implements.

Design notes
------------
*Small on purpose.* ``fit`` / ``predict_proba`` / ``params`` is the whole contract. It
is deliberately the sklearn estimator shape, because ``make test-parity`` compares each
hand-written learner against its sklearn oracle and a shared shape makes that comparison
a two-line test rather than an adapter.

*Probabilities, not labels.* Every learner returns ``P(invest)``. Thresholding is a
separate decision made at scoring time -- at 25.9% prevalence the interesting question
is usually "how well are startups ranked", not "what happens at 0.5".

*Hyperparameters are dataclass fields; fitted state is not.* ``params()`` reads the
dataclass fields, so it captures exactly the configuration needed to reproduce a run and
never the learned weights. Subclasses set fitted attributes in ``fit`` with a trailing
underscore, sklearn-style.

*NaN is the learner's problem.* ``Dataset.X`` carries NaN for unrated criteria (see the
"0 is not a score" rule in AGENTS.md), and this base class does not impute. Each learner
declares what it does about missing values -- Naive Bayes can skip them natively, a
gradient learner cannot -- and that choice is part of the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np


class NotFittedError(RuntimeError):
    """``predict_proba`` was called before ``fit``."""


@runtime_checkable
class Learner(Protocol):
    """Structural type accepted by the runner, the parity tests, and the experiments."""

    @property
    def name(self) -> str: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def params(self) -> dict[str, Any]: ...


class BaseLearner(ABC):
    """Convenience base providing ``name``, ``params`` and ``predict``.

    Subclass it as a dataclass whose fields are the hyperparameters::

        @dataclass
        class MyLearner(BaseLearner):
            alpha: float = 1.0

            def fit(self, X, y):
                self.weights_ = ...
                return self._fitted()

            def predict_proba(self, X):
                self._check_fitted()
                return ...
    """

    _is_fitted: bool = False

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Learn from ``X`` (n, d) and ``y`` (n,), and return ``self``."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(y == 1)`` as a (n,) float64 array."""

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    def params(self) -> dict[str, Any]:
        """The hyperparameters, JSON-ready. Recorded in every results file."""
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    # --- fitted-state bookkeeping -------------------------------------------
    def _fitted(self) -> Self:
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.name} has not been fitted")


def check_Xy(
    X: np.ndarray, y: np.ndarray, *, allow_nan: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a training pair and return it as (float64, int8).

    ``allow_nan=False`` is how a learner that cannot handle missing values says so
    loudly at fit time, instead of silently producing NaN weights.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"{y.shape[0]} labels for {X.shape[0]} rows")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if not allow_nan and np.isnan(X).any():
        raise ValueError(
            "X contains NaN (unrated criteria). This learner needs an explicit "
            "imputation choice -- see vcml.augment.impute."
        )
    return X, y.astype(np.int8)


def check_X(X: np.ndarray, n_features: int, *, allow_nan: bool = True) -> np.ndarray:
    """Validate a prediction matrix against the width seen during fit."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} columns, fitted on {n_features}")
    if not allow_nan and np.isnan(X).any():
        raise ValueError("X contains NaN and this learner cannot handle missing values")
    return X

"""Finite-difference checking for hand-derived gradients.

The point of this project is deriving the learning algorithms by hand, which means the
derivations have to be *checked* rather than trusted. A wrong gradient rarely crashes --
it converges to a slightly wrong optimum and quietly reports a slightly wrong number.

The check compares an analytic gradient against a central difference::

    df/dx_i ~= (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)

Central rather than forward differences because the error term is O(eps^2) instead of
O(eps), which buys about eight digits of agreement instead of four. The comparison is on
*relative* error, since a loss gradient's scale varies by orders of magnitude between
parameters.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class GradientMismatch(AssertionError):
    """The analytic gradient disagrees with the numerical one."""


def numeric_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient of ``f`` at ``x``.

    Costs ``2 * x.size`` evaluations, so use it on small problems -- a handful of
    features and a few dozen rows is plenty to catch a sign error or a missing term.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (f(x + step) - f(x - step)) / (2.0 * eps)
    return grad


def check_gradient(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1e-6,
    tol: float = 1e-6,
) -> float:
    """Raise :class:`GradientMismatch` if analytic and numerical gradients disagree.

    Returns the maximum relative error, so a caller can report how close it was.
    """
    x = np.asarray(x, dtype=np.float64)
    analytic = np.asarray(grad(x), dtype=np.float64)
    numerical = numeric_gradient(f, x, eps=eps)

    if analytic.shape != numerical.shape:
        raise GradientMismatch(
            f"analytic gradient has shape {analytic.shape}, expected {numerical.shape}"
        )

    # Relative to the larger of the two magnitudes, with a floor so near-zero
    # components do not divide a tiny absolute error into a huge relative one.
    scale = np.maximum(np.maximum(np.abs(analytic), np.abs(numerical)), 1.0)
    error = float(np.max(np.abs(analytic - numerical) / scale))

    if not np.isfinite(error) or error > tol:
        worst = int(np.argmax(np.abs(analytic - numerical) / scale))
        raise GradientMismatch(
            f"max relative error {error:.3e} exceeds tol {tol:.3e}; "
            f"worst component {worst}: analytic {analytic.ravel()[worst]:.8f} "
            f"vs numerical {numerical.ravel()[worst]:.8f}"
        )
    return error

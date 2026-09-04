"""Tests for the finite-difference gradient checker itself. Run with ``make gradcheck``.

The checker is the instrument every hand-derived gradient will be measured with, so it
is calibrated first: it must accept a correct gradient at ~1e-9 and reject the specific
mistakes that actually happen -- a dropped factor, a flipped sign, a missing term.

Each gradient learner adds its own case here as it lands (rung 2 onward).
"""

from __future__ import annotations

import numpy as np
import pytest

from vcml.models.gradcheck import GradientMismatch, check_gradient, numeric_gradient

pytestmark = pytest.mark.gradcheck


def _logistic_problem(
    n: int = 60, d: int = 4, seed: int = 20260823
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.3).astype(np.float64)
    return X, y


def _nll(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Mean negative log-likelihood of logistic regression, computed stably."""
    z = X @ weights
    # log(1 + exp(z)) via logaddexp, so large |z| does not overflow.
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _nll_grad(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The hand-derived gradient: ``X^T (sigma(Xw) - y) / n``."""
    p = 1.0 / (1.0 + np.exp(-(X @ weights)))
    return X.T @ (p - y) / X.shape[0]


class TestAcceptsCorrectGradients:
    def test_quadratic(self) -> None:
        a = np.array([2.0, -3.0, 0.5])
        err = check_gradient(lambda x: float(np.sum(a * x**2)), lambda x: 2 * a * x, np.ones(3))
        assert err < 1e-8

    def test_logistic_nll(self) -> None:
        X, y = _logistic_problem()
        w = np.array([0.3, -0.7, 0.1, 0.9])
        err = check_gradient(lambda v: _nll(X, y, v), lambda v: _nll_grad(X, y, v), w, tol=1e-7)
        assert err < 1e-7

    def test_holds_at_saturated_weights(self) -> None:
        """Large weights push sigma() to 0/1, where a naive implementation loses all
        its precision. The check must still pass, or it cannot be trusted later."""
        X, y = _logistic_problem()
        w = np.array([12.0, -9.0, 7.0, -11.0])
        assert check_gradient(lambda v: _nll(X, y, v), lambda v: _nll_grad(X, y, v), w) < 1e-6


class TestCatchesWrongGradients:
    @pytest.mark.parametrize(
        ("label", "broken"),
        [
            ("sign flip", lambda X, y, w: -_nll_grad(X, y, w)),
            ("missing 1/n", lambda X, y, w: _nll_grad(X, y, w) * X.shape[0]),
            ("factor of two", lambda X, y, w: 2.0 * _nll_grad(X, y, w)),
            ("dropped y term", lambda X, y, w: X.T @ (1.0 / (1.0 + np.exp(-(X @ w)))) / len(y)),
        ],
    )
    def test_common_derivation_mistakes(self, label: str, broken: object) -> None:
        X, y = _logistic_problem()
        w = np.array([0.3, -0.7, 0.1, 0.9])
        with pytest.raises(GradientMismatch):
            check_gradient(
                lambda v: _nll(X, y, v),
                lambda v: broken(X, y, v),  # type: ignore[operator]
                w,
            )

    def test_shape_mismatch_is_reported(self) -> None:
        with pytest.raises(GradientMismatch, match="shape"):
            check_gradient(lambda x: float(x.sum()), lambda x: np.ones(2), np.ones(3))

    def test_error_message_names_the_worst_component(self) -> None:
        with pytest.raises(GradientMismatch, match="worst component"):
            check_gradient(
                lambda x: float(np.sum(x**2)), lambda x: np.array([2 * x[0], 0.0]), np.ones(2)
            )


class TestNumericGradient:
    def test_matches_a_known_derivative(self) -> None:
        grad = numeric_gradient(lambda x: float(np.sum(np.sin(x))), np.array([0.0, np.pi / 2]))
        assert grad == pytest.approx([1.0, 0.0], abs=1e-7)

    def test_works_on_matrices(self) -> None:
        """Later learners parameterise with 2-D weights, so shapes must pass through."""
        w = np.arange(6, dtype=np.float64).reshape(2, 3)
        grad = numeric_gradient(lambda m: float(np.sum(m**2)), w)
        assert grad.shape == w.shape
        assert grad == pytest.approx(2 * w, abs=1e-6)

"""Scoring. Depends on nothing but numpy, so every other module may import it.

Design notes
------------
*Accuracy alone is misleading here.* Prevalence is 25.9%, so a model that predicts
"pass" for every startup scores 74.1% and has learned nothing. Every number this module
produces is therefore reported next to two reference lines: the majority-class floor
(:func:`majority_accuracy`) and the duplicate ceiling from ``data.duplicate_audit``.
A result without both is not interpretable.

*PR-AUC over ROC-AUC for the headline.* With a 1:3 class ratio the ROC curve flatters
a model by rewarding it for the abundant negatives. Average precision has the positive
class in both terms, and its own floor is exactly the prevalence.

*Hand-written, parity-checked.* These are re-derivations, not wrappers -- ``average
precision`` here is the step-wise sum sklearn computes, tie-handling included, and
``make test-parity`` pins it against ``sklearn.metrics``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Probabilities are clipped before any log, so a confident-and-wrong prediction costs a
# large finite number rather than inf (which would poison the mean over folds). Machine
# epsilon specifically, because that is what sklearn.metrics.log_loss clips at and
# test-parity compares the two to 1e-9 -- a different constant here is a silent
# disagreement of ~0.1 nats whenever a model predicts exactly 0 or 1.
_EPS: float = float(np.finfo(np.float64).eps)


def _as_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    return y.astype(np.int8)


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing their mean rank. A local stand-in for scipy.rankdata."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


# --- reference lines --------------------------------------------------------
def prevalence(y_true: np.ndarray) -> float:
    """Positive base rate. This is exactly the PR-AUC of a random ranker."""
    return float(_as_binary(y_true).mean())


def majority_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the more common class. The floor to beat."""
    p = prevalence(y_true)
    return max(p, 1.0 - p)


# --- threshold metrics ------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((_as_binary(y_true) == _as_binary(y_pred)).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)``."""
    t, p = _as_binary(y_true), _as_binary(y_pred)
    return (
        int(((t == 0) & (p == 0)).sum()),
        int(((t == 0) & (p == 1)).sum()),
        int(((t == 1) & (p == 0)).sum()),
        int(((t == 1) & (p == 1)).sum()),
    )


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of recall on each class -- immune to the 1:3 imbalance."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float((tpr + tnr) / 2.0)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tp = confusion(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = confusion(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation. The one threshold metric that is hard to game by
    predicting a constant: any constant prediction scores exactly 0."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


# --- ranking and calibration metrics ----------------------------------------
def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision: ``sum (R_n - R_{n-1}) * P_n`` over distinct thresholds.

    The step-wise sum rather than trapezoidal interpolation, matching
    ``sklearn.metrics.average_precision_score``. Interpolating a PR curve is
    optimistic -- there is no model that achieves the interpolated points.
    """
    t = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(t.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-s, kind="mergesort")
    t, s = t[order], s[order]

    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    # Only the last row of each group of tied scores is a realisable operating point.
    last_of_tie = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, fp = tp[last_of_tie], fp[last_of_tie]

    prec = tp / (tp + fp)
    rec = tp / n_pos
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Via the Mann-Whitney U statistic, so tied scores are handled exactly."""
    t = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos, n_neg = int(t.sum()), int((t == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(s)
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is a coin flip."""
    t = _as_binary(y_true).astype(np.float64)
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((p - t) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    t = _as_binary(y_true).astype(np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The cutoff maximising F1, searched over the observed scores.

    Reported alongside the 0.5 numbers because at 25.9% prevalence 0.5 is an arbitrary
    and usually bad operating point -- a model can rank well and still predict "pass"
    for everything at 0.5.
    """
    s = np.asarray(y_score, dtype=np.float64).ravel()
    candidates = np.unique(s)
    if len(candidates) == 0:
        return 0.5
    scores = [f1(y_true, (s >= c).astype(np.int8)) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


# --- the bundle -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Scores:
    """Every metric for one set of predictions, at one threshold."""

    n: int
    n_pos: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self, floor: float | None = None, ceiling: float | None = None) -> str:
        line = (
            f"acc {self.accuracy:.3f} | bal {self.balanced_accuracy:.3f} | "
            f"F1 {self.f1:.3f} | PR-AUC {self.pr_auc:.3f}"
        )
        if floor is not None:
            line += f" | floor {floor:.3f}"
        if ceiling is not None:
            line += f" | ceiling {ceiling:.3f}"
        return line


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> Scores:
    """Score one set of predicted probabilities."""
    t = _as_binary(y_true)
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if p.shape != t.shape:
        raise ValueError(f"{p.shape} probabilities for {t.shape} labels")
    if np.isnan(p).any():
        raise ValueError("predicted probabilities contain NaN")

    y_pred = (p >= threshold).astype(np.int8)
    return Scores(
        n=int(t.size),
        n_pos=int(t.sum()),
        threshold=float(threshold),
        accuracy=accuracy(t, y_pred),
        balanced_accuracy=balanced_accuracy(t, y_pred),
        precision=precision(t, y_pred),
        recall=recall(t, y_pred),
        f1=f1(t, y_pred),
        mcc=mcc(t, y_pred),
        pr_auc=pr_auc(t, p),
        roc_auc=roc_auc(t, p),
        brier=brier(t, p),
        log_loss=log_loss(t, p),
    )

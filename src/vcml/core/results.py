"""Running a learner over a split plan, and writing the result down so it counts.

Design notes
------------
*A number without its provenance is not evidence.* Every result records the dataset
fingerprint, the split id, the seed, the learner's hyperparameters and the git sha. Two
accuracies are comparable only if the first two agree; the paper build can check that
mechanically instead of trusting that nothing moved between runs.

*Every result carries its own floor and ceiling.* ``reference_lines`` attaches the
majority-class accuracy (74.1%) and the duplicate ceiling (94.9%) to the run, so a
number can never be read without the two lines that make it meaningful.

*Deterministic on disk.* The JSON splits into ``run`` (deterministic: rerunning the same
learner on the same data reproduces it byte for byte, which is what
:attr:`RunResult.digest` pins) and ``env`` (timestamps, wall-clock, versions -- useful,
but never part of the comparison).

*Pooled out-of-fold scores are the headline, but both are recorded.* The mean of five
fold PR-AUCs is not the PR-AUC of the model: with ~150 validation rows per fold and
25.9% prevalence, per-fold ranking metrics are extremely noisy. The pooled score,
computed once over every held-out prediction, is the stable one.

Pooling has one artefact worth knowing about, and it is why the fold means are kept
alongside. A model whose output is constant *within* a fold but differs *between* folds
(any base-rate predictor, since each fold's training prevalence differs) gets a pooled
ROC-AUC away from 0.5 -- the between-fold variation looks like ranking signal. Per fold
it correctly scores exactly 0.5. So when ``fold_mean("roc_auc")`` is 0.5 and the pooled
value is not, the model is ranking nothing and the pooled number is an artefact.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vcml.core.config import MASTER_SEED, RESULTS_DIR
from vcml.core.data import duplicate_audit
from vcml.core.metrics import Scores, best_threshold, evaluate, majority_accuracy, prevalence
from vcml.core.schema import Dataset, fingerprint_arrays
from vcml.core.splits import SplitPlan, assert_no_leakage

if TYPE_CHECKING:  # core must not depend on models at runtime
    from vcml.models.base import Learner

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReferenceLines:
    """The two numbers every score must be read against."""

    n: int
    n_pos: int
    prevalence: float
    majority_accuracy: float
    ceiling_accuracy: float

    @property
    def headroom(self) -> float:
        """How much accuracy is actually available above the trivial model."""
        return self.ceiling_accuracy - self.majority_accuracy

    def summary(self) -> str:
        return (
            f"floor {self.majority_accuracy:.3f} (majority) .. "
            f"ceiling {self.ceiling_accuracy:.3f} (duplicates) -- "
            f"{self.headroom:.3f} of headroom"
        )


def reference_lines(ds: Dataset) -> ReferenceLines:
    """Floor and ceiling for this dataset."""
    return ReferenceLines(
        n=ds.n,
        n_pos=ds.n_pos,
        prevalence=prevalence(ds.y),
        majority_accuracy=majority_accuracy(ds.y),
        ceiling_accuracy=duplicate_audit(ds).ceiling_accuracy,
    )


def git_sha() -> str:
    """Current commit, or ``"unknown"`` outside a repo. Never fails a run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays into something ``json`` will accept."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class RunResult:
    """One learner, one dataset, one split plan -- and everything needed to trust it."""

    experiment: str
    learner: str
    params: dict[str, Any]
    dataset_fingerprint: str
    split_name: str
    split_id: str
    n_folds: int
    seed: int
    threshold: float
    pooled: Scores
    fold_scores: tuple[Scores, ...]
    reference: ReferenceLines
    best_f1_threshold: float
    git_sha: str = "unknown"
    runtime_seconds: float = 0.0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # --- derived reporting --------------------------------------------------
    def fold_values(self, metric: str) -> np.ndarray:
        return np.array([getattr(s, metric) for s in self.fold_scores], dtype=np.float64)

    def fold_mean(self, metric: str) -> float:
        """Mean across folds. Read next to the pooled value -- see the module docstring
        on why a 0.5 fold mean with a non-0.5 pooled AUC means "ranks nothing"."""
        return float(np.nanmean(self.fold_values(metric)))

    def fold_std(self, metric: str) -> float:
        """Fold-to-fold spread. Large values mean the headline number is fragile."""
        return float(np.nanstd(self.fold_values(metric)))

    @property
    def fold_accuracy_std(self) -> float:
        return self.fold_std("accuracy")

    @property
    def beats_floor_by(self) -> float:
        return self.pooled.accuracy - self.reference.majority_accuracy

    @property
    def ceiling_gap(self) -> float:
        return self.reference.ceiling_accuracy - self.pooled.accuracy

    def run_block(self) -> dict[str, Any]:
        """The deterministic half of the JSON -- no clocks, no versions."""
        block: dict[str, Any] = _jsonable(
            {
                "schema": SCHEMA_VERSION,
                "experiment": self.experiment,
                "learner": self.learner,
                "params": self.params,
                "dataset_fingerprint": self.dataset_fingerprint,
                "split": {
                    "name": self.split_name,
                    "id": self.split_id,
                    "n_folds": self.n_folds,
                    "seed": self.seed,
                },
                "threshold": self.threshold,
                "best_f1_threshold": self.best_f1_threshold,
                "pooled": self.pooled.to_dict(),
                "fold_mean": {
                    k: self.fold_mean(k)
                    for k in ("accuracy", "balanced_accuracy", "f1", "pr_auc", "roc_auc")
                },
                "fold_std": {
                    k: self.fold_std(k)
                    for k in ("accuracy", "balanced_accuracy", "f1", "pr_auc", "roc_auc")
                },
                "folds": [s.to_dict() for s in self.fold_scores],
                "reference": asdict(self.reference),
                "notes": self.notes,
                "extra": self.extra,
            }
        )
        return block

    @property
    def digest(self) -> str:
        """Hash of the deterministic block. Equal digests == a reproduced run."""
        payload = json.dumps(self.run_block(), sort_keys=True, separators=(",", ":"))
        return fingerprint_arrays(np.array([0]), extra=payload)[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run_block(),
            "digest": self.digest,
            "env": {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "runtime_seconds": round(self.runtime_seconds, 4),
                "git_sha": self.git_sha,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }

    def summary(self) -> str:
        scores = self.pooled.summary(
            self.reference.majority_accuracy, self.reference.ceiling_accuracy
        )
        return (
            f"{self.learner} on {self.split_name}: {scores}"
            f"\n  {self.beats_floor_by:+.3f} vs floor, {self.ceiling_gap:.3f} below ceiling, "
            f"fold acc sd {self.fold_accuracy_std:.3f}"
        )

    def roadmap_line(self) -> str:
        """The one-liner that goes next to this rung's box in ROADMAP.md."""
        return (
            f"{self.pooled.accuracy:.3f} acc / {self.pooled.pr_auc:.3f} PR-AUC "
            f"(floor {self.reference.majority_accuracy:.3f} / "
            f"{self.reference.prevalence:.3f}, ceiling {self.reference.ceiling_accuracy:.3f})"
        )


def cross_validate(
    make_learner: Callable[[], Learner],
    ds: Dataset,
    plan: SplitPlan,
    *,
    experiment: str,
    threshold: float = 0.5,
    check_leakage: bool = True,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> RunResult:
    """Fit a fresh learner per fold and score the held-out predictions.

    Args:
        make_learner: a zero-argument factory. A factory rather than an instance so no
            fold can ever see state fitted on another fold.
        plan: from ``splits.session_kfold`` or ``splits.forward_chaining``. Its
            fingerprint must match ``ds``, otherwise the folds index different rows.
        check_leakage: run ``splits.assert_no_leakage`` first. Left on by default;
            it costs milliseconds and it is the guarantee the whole project rests on.
    """
    if plan.dataset_fingerprint != ds.fingerprint:
        raise ValueError(
            "split plan was built for a different dataset "
            f"({plan.dataset_fingerprint[:12]} vs {ds.fingerprint[:12]}). "
            "Rebuild the plan, or the folds index the wrong rows."
        )
    if check_leakage:
        assert_no_leakage(ds, plan)

    started = time.perf_counter()
    fold_scores: list[Scores] = []
    oof_true: list[np.ndarray] = []
    oof_prob: list[np.ndarray] = []
    learner_name = make_learner().name
    params: dict[str, Any] = {}

    for fold in plan.folds:
        learner = make_learner()
        learner.fit(ds.X[fold.train], ds.y[fold.train])
        proba = np.asarray(learner.predict_proba(ds.X[fold.val]), dtype=np.float64).ravel()
        if proba.shape[0] != fold.n_val:
            raise ValueError(
                f"fold {fold.index}: learner returned {proba.shape[0]} probabilities "
                f"for {fold.n_val} rows"
            )
        fold_scores.append(evaluate(ds.y[fold.val], proba, threshold=threshold))
        oof_true.append(ds.y[fold.val])
        oof_prob.append(proba)
        params = learner.params()

    y_true = np.concatenate(oof_true)
    y_prob = np.concatenate(oof_prob)

    return RunResult(
        experiment=experiment,
        learner=learner_name,
        params=_jsonable(params),
        dataset_fingerprint=ds.fingerprint,
        split_name=plan.name,
        split_id=plan.split_id,
        n_folds=plan.n_folds,
        seed=plan.seed,
        threshold=threshold,
        pooled=evaluate(y_true, y_prob, threshold=threshold),
        fold_scores=tuple(fold_scores),
        reference=reference_lines(ds),
        best_f1_threshold=best_threshold(y_true, y_prob),
        git_sha=git_sha(),
        runtime_seconds=time.perf_counter() - started,
        notes=notes,
        extra=extra or {},
    )


def save_result(result: RunResult, *, results_dir: Path = RESULTS_DIR) -> Path:
    """Write ``results/<experiment>.json``. Sorted keys, so diffs are readable."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.experiment}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_result(path: Path) -> dict[str, Any]:
    """Read a results file back. Used by the figure and paper builds."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def default_seed() -> int:
    """Every random draw in the project descends from this."""
    return MASTER_SEED

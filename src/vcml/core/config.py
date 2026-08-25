"""Paths, seeds, and the column whitelist. Imported by everything; depends on nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --- Paths ------------------------------------------------------------------
# config.py -> core -> vcml -> src -> repo root
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_CSV: Final[Path] = DATA_DIR / "raw" / "plug-and-play-raw.csv"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SPLITS_DIR: Final[Path] = DATA_DIR / "splits"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
PAPER_DIR: Final[Path] = REPO_ROOT / "paper"

# --- Determinism ------------------------------------------------------------
# Every random draw in the project descends from this one integer. Changing it
# changes every reported number, so it is versioned with the code.
MASTER_SEED: Final[int] = 20260823

# --- Columns ----------------------------------------------------------------
# The six review criteria, each scored 1-5 by the analyst. Order is fixed: it
# determines column order in X and therefore the dataset fingerprint.
CRITERIA: Final[tuple[str, ...]] = (
    "Team",
    "Technology",
    "Market",
    "Value Proposition",
    "Competitive Advantage",
    "Socially Impactful",
)

LABEL_COL: Final[str] = "Invest William"
GROUP_COL: Final[str] = "Pitch Session"
NAME_COL: Final[str] = "Name"
FUNDING_COL: Final[str] = "Funding amount"

# Feature whitelist. Nothing outside this tuple may enter the model matrix X.
#
# This is a whitelist rather than a blacklist because the raw CSV contains two
# columns that would destroy the study if they leaked in:
#
#   "Invest Wout"  -- a second reviewer's verdict. It agrees with the target on
#                     650 of 656 co-labelled rows (99.1%), so a model given it
#                     would score ~99% while having learned nothing.
#   "Notes William" -- the analyst's own written rationale, which is evaluative:
#                     "strong" appears in 37.7% of yes-notes vs 7.0% of no-notes.
#                     Predicting a verdict from the reasoning behind it is circular.
#
# A blacklist would let a future column silently become a feature. This cannot.
BASE_FEATURES: Final[tuple[str, ...]] = (*CRITERIA, "funding_usd")

# A criterion recorded as 0 (or blank) means "not filled in", NOT a rating of zero.
# It is loaded as NaN: 0 sits below the 1-5 scale, and every model would happily
# extrapolate it as "worse than terrible".
MISSING_SENTINEL: Final[str] = "0"
CRITERION_MIN: Final[int] = 1
CRITERION_MAX: Final[int] = 5

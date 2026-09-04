# vcml — VC pitch-scoring project

Predicting one analyst's invest/pass verdicts on startup pitches from his own 1-5 criteria
scores plus funding amount. The point of the project is deriving and validating a learning
algorithm by hand, not shipping a production model — see "oracle" below.

## Commands

- `make setup` — install locked deps (`requirements.lock`), then the package (`dev`,`paper` extras)
- `make data` — rebuild `data/processed/` from `data/raw/` (needs the gitignored raw CSV locally)
- `make check` — lint + typecheck + test-core; this is what CI runs and what every PR must pass
- `make test-core` — the fast test gate
- `make test-parity` — hand-written learners vs. their sklearn oracle
- `make gradcheck` — finite-difference check of hand-derived gradients
- `make leakage` — asserts no train/val fold sees a synthetic row whose real parent is in the other fold
- `make lint` / `make format` / `make typecheck` — ruff + `mypy --strict`

## Layout

- `src/vcml/core/` — FROZEN. Import order `config → funding → clean → data/schema`, one
  direction only. Shared by every experiment, so a bug here is silent and repo-wide.
  `mypy --strict` applies here and in `models/`. Changing `config.MASTER_SEED` or the order of
  `CRITERIA` invalidates every existing results JSON (see fingerprinting below) — don't do it casually.
- `src/vcml/models/` — hand-written NumPy learners, no sklearn at runtime. One file per learner.
  Shared across experiments once imported, so keep interfaces stable.
- `src/vcml/experiments/` — one new file per experiment, discovered by directory scan — nothing
  to register centrally, by design, so parallel work never collides on one shared file. Freely
  disposable.
- `src/vcml/augment/` — synthetic-data generators. Any synthetic row must be added via
  `Dataset.concat_synthetic` so it's tagged `origin=-1` with `meta["synthetic_parents"]` — the
  leakage check depends on this.
- `legacy/clean_data.py` — the original buggy cleaner, kept only as the documented "before". Do
  not import from it.

## Hard rules

- **Feature whitelist.** Only `config.BASE_FEATURES` may enter model input `X`. Never add a raw
  column without reading the comment in `config.py` first — two columns look like ordinary
  features and are actually leaks: `Invest Wout` (a second reviewer's verdict, 99.1% agreement
  with the label) and `Notes William` (the analyst's own written rationale for the verdict).
  Whitelist, not blacklist, specifically so a new raw column is safe by default.
- **0 is not a score.** A criterion of `0` or blank means "not rated" and is loaded as NaN,
  never as a real 0 — 0 sits below the valid 1-5 range. Feeding it as a number previously taught
  a model that "unrated" means "worse than the worst startup."
- **Funding parsing has two independent historical bugs, both fixed in `funding.py`.** Re-read
  that file's docstring before touching it: naive `.split(".")[0]` parsing truncates real
  cents/dollars (the export always has a literal 3-decimal suffix, never a thousands separator),
  and funding units differ *per session* (some rows whole dollars, some thousands — inferred
  from a clean 4-orders-of-magnitude gap; never guessed when ambiguous, `SessionUnitAmbiguity`
  is raised instead). Getting this wrong previously flipped corr(funding, verdict) from +0.08 to -0.11.
- **`Dataset` is immutable** (`X.setflags(write=False)`), so parallel experiments can share one
  loaded copy safely. Don't work around the resulting `ValueError` — produce a new `Dataset` via
  `.subset()` / `.replace_X()` instead.
- **Check `duplicate_audit().ceiling_accuracy` before trusting any accuracy number.** Six
  ordinal 1-5 criteria give only 15,625 possible score vectors for 769 rows, so exact duplicate
  vectors with opposite verdicts are expected — that ceiling is the max any model reading only
  the scores can reach.
- **`data/raw/` is confidential** (real startup names + personal verdicts) — never commit it,
  never print/log full rows containing the `Name` column.

## Glossary

- **Oracle** — a trusted reference implementation (e.g. `sklearn`) used only to check a
  hand-written learner's correctness (`make test-parity`); it never ships. This is why sklearn
  lives under the `dev` extra in `pyproject.toml`, not `dependencies`.
- **Fingerprint** — `Dataset.fingerprint` is a sha256 over the array bytes; results built from
  differently-fingerprinted data are not comparable, even if nothing else looks different.
- **origin** — per-row provenance on a `Dataset`: real rows carry their index, synthetic rows
  carry `-1` plus a parent pointer in `meta["synthetic_parents"]`. The leakage check keys off this.

## Adding something new

A new experiment or model is one new file in `experiments/` or `models/` — nothing else to
register. Run `make test-core` (and `make leakage` / `make test-parity` / `make gradcheck` if
they apply to what you touched) before considering it done, then `make check`.

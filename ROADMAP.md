# Roadmap

The sequence, and nothing else. The rules live in [AGENTS.md](AGENTS.md).

**What's next** is always the topmost unchecked box. A box is ticked only when
`make check` is green, a fingerprinted JSON is in `results/`, and the headline number is
written on the line — so the ladder below doubles as the paper's results table.

Format: `acc / PR-AUC (floor acc / floor PR-AUC, ceiling)`.
Floor = majority class and prevalence. Ceiling = 0.949, the duplicate bound.

---

## Foundation

- [x] `core/metrics.py` — scores plus the floor/ceiling reference lines
- [x] `core/splits.py` — session-grouped and forward-chaining CV, leakage-proof
- [x] `models/base.py` — the `Learner` interface every rung implements
- [x] `models/gradcheck.py` — finite-difference checking for hand-derived gradients
- [x] `core/results.py` — fingerprinted results JSON + the cross-validation runner
- [x] `tests/conftest.py` — toy dataset, so the suite runs without the confidential raw CSV
- [x] `.github/workflows/ci.yml` — runs `make check`; all four make gates now honest
- [x] `README.md` — the confidentiality section `.gitignore` cites

## The ladder

Serial. Each rung must earn its complexity against the one below it.

- [ ] **0 · baselines** — `models/baselines.py`: majority, prevalence, single-best criterion
- [ ] **1 · naive bayes** — `models/naive_bayes.py`: categorical NB over the six criteria; parity vs `CategoricalNB`
- [ ] **2 · logistic regression** — `models/logistic.py`: first hand-derived gradient; first NaN and scaling decision
- [ ] **3 · ordinal logistic** — `models/ordinal.py`: does respecting the 1–5 ordering beat treating scores as continuous?
- [ ] **4 · decision tree** — `models/tree.py`: hand-written CART; first model that can express interactions
- [ ] **5 · bagged trees** — `models/forest.py`: does variance reduction help at n=769?
- [ ] **6 · small MLP** — `models/mlp.py`: the capacity question, honestly answered

## Supporting experiments

- [ ] `experiments/exp_ceiling.py` — the duplicate ceiling as a first-class result *(before rung 0)*
- [ ] `experiments/exp_permutation_canary.py` — shuffled labels must score at chance *(after rung 2)*
- [ ] `experiments/exp_funding_effect.py` — funding in/out, raw vs log1p; the corr −0.111 story
- [ ] `experiments/exp_feature_ablation.py` — one criterion dropped at a time *(after the best rung is known)*
- [ ] `experiments/exp_learning_curve.py` — accuracy vs. n; would more data help?
- [ ] `experiments/exp_error_analysis.py` — are the errors the contradictory duplicates?
- [ ] `core/harness.py` + `vcml` console entrypoint — directory-scan discovery *(around rung 4)*

## Augmentation

The one lane worth a parallel worktree — it does not inform the next rung.

- [ ] `augment/impute.py` — NaN strategies compared, not assumed
- [ ] `augment/jitter.py` — ordinal-aware synthetic rows via `Dataset.concat_synthetic`
- [ ] `augment/resample.py` — class imbalance at 25.9% prevalence
- [ ] `experiments/exp_augment_sweep.py` — does synthetic data help at all?

## Paper and portfolio

- [ ] `paper/figures.py` — figures generated from `results/*.json`, never hand-edited
- [ ] `make paper` — results → figures → sections
- [ ] `paper/sections/*.md` — methods, results, discussion, limitations
- [ ] Portfolio visuals — the ladder chart, the funding-bug before/after, the ablation panel
- [ ] Anonymised data release decision (drop `Name`/`url`, relabel sessions `S01..S23`)

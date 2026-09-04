# vcml — recovering a VC analyst's decision function

Can one analyst's invest/pass verdicts be predicted from his own 1–5 scoring rubric?

The data is 769 startup pitches from the Plug and Play deal pipeline, each scored by the
same analyst on six criteria — Team, Technology, Market, Value Proposition, Competitive
Advantage, Socially Impactful — plus the funding amount sought, and each ending in a
recorded invest or pass. 199 of the 769 are invests: a base rate of 25.9%.

The point of the project is deriving and validating the learning algorithms by hand.
Every model is written in NumPy from the derivation up; scikit-learn appears only as a
correctness oracle in the test suite and never ships. What is being demonstrated is the
method — measured, reproducible, adversarial to its own results — not a production model.

## The two numbers that frame everything

**Floor: 74.1%.** Predicting "pass" for every startup is right 74.1% of the time. Any
accuracy near that has learned nothing, so every result in this repo is reported next to
it.

**Ceiling: 94.9%.** Six ordinal criteria give only 5^6 = 15,625 possible score vectors
for 769 pitches, so identical scoresheets are common: 637 distinct vectors, and 35 of
them carry *both* verdicts across 105 rows. Those rows cannot be separated by any model
reading only the scores. The Bayes-error bound that follows — 94.9% — is the real target,
and the honest headroom is the 20.8 points between the two.

Run `python -c "from vcml.core.data import *; print(duplicate_audit(load_dataset()).summary())"`
to reproduce both.

## What is already established

- **822 raw rows → 769 usable.** 53 dropped, every one accounted for in
  `data/processed/drop_ledger.json`: 48 with neither verdict nor scores, 3 scored but
  never decided, 2 decided but never scored.
- **A criterion of 0 means "not rated", not a rating of zero.** It loads as NaN. Fed as a
  number it had been teaching models that an unrated startup is worse than the worst one.
- **Funding had two independent parsing bugs**, both fixed in `core/funding.py`. Naive
  `.split(".")[0]` truncation mangled the export's literal 3-decimal suffix, and funding
  units differ *per pitch session* — some rows whole dollars, some thousands. Correcting
  them flips corr(funding, verdict) from +0.08 to −0.11: the sign of a published
  conclusion depended on a parsing detail.
- **Two columns in the raw export are leaks, not features.** `Invest Wout` is a second
  reviewer's verdict (99.1% agreement with the target) and `Notes William` is the
  analyst's own written rationale. `config.BASE_FEATURES` is a whitelist precisely so a
  newly added raw column is safe by default.

## Getting started

```
make setup    # venv + locked deps
make data     # rebuild data/processed/ (needs the raw CSV; see below)
make check    # lint + mypy --strict + tests — what CI runs
```

Other gates: `make test-parity` (hand-written code vs. its sklearn oracle),
`make gradcheck` (finite-difference checks of hand-derived gradients), `make leakage`
(no information crosses a train/validation boundary).

[ROADMAP.md](ROADMAP.md) is the work queue and the results table. [AGENTS.md](AGENTS.md)
has the layout and the hard rules.

## Confidentiality

`data/raw/` is **not committed and must not be**. It contains named startups from a real
deal pipeline alongside one person's private verdicts on them. Nothing in this repo may
print or log a full row containing the `Name` column, and `data/processed/` is gitignored
for the same reason even though it is derived.

The consequence is that a third party cannot currently reproduce the numbers above. The
test suite is built so this does not silently rot: tests that pin published figures skip
without the raw export, and everything testing *mechanism* runs against a generated toy
dataset (`tests/conftest.py`), so CI stays meaningful on a clone with no data at all.

Publishing an anonymised release — drop `Name` and `url`, relabel sessions `S01..S23` —
is the open decision that would make the paper reproducible. It is tracked at the bottom
of the roadmap and has not been taken.

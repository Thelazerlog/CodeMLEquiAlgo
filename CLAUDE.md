# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IVADO ÉquiAlgo hackathon challenge (24 h): a fair student scholarship model. Diagnose regional bias in a historical grant model, correct it, and propose a production monitoring plan. All data is synthetic. Source of truth for the rules: `README.md` (EN), `LISEZMOI.md` (FR), `consignes-en.pdf` / `consignes-fr.pdf`. Code, comments and the baseline notebook are in French (no accents in identifiers/data values).

## Setup / commands

Windows machine, Python 3.12 available (3.10+ required).

```bash
python -m venv venv
venv\Scripts\activate            # PowerShell; bash: source venv/Scripts/activate
pip install -r requirements.txt  # pandas, numpy, scikit-learn, fairlearn, matplotlib, jupyter, shap
jupyter notebook baseline_model.ipynb
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb   # run a notebook headless
```

There are no tests, linter or build. All paths in the notebook are relative (`data/...`), so run from the repo root.

## Data

- `data/donnees_demandes.csv` — 10,000 historical applications with label `decision_octroi` (committee decision, ~40% granted).
- `data/candidats_evaluation.csv` — 4,000 applications to score, no label.
- Sensitive attribute: `region_administrative`, 5 values: `Montreal`, `Capitale-Nationale` (group "Centre") vs `Bas-Saint-Laurent`, `Cote-Nord`, `Gaspesie-Iles-de-la-Madeleine` (group "Eloignee"). The baseline's `groupe_region()` builds this binary grouping; fairness metrics are computed on it.
- Features: `cote_r_equivalent` (merit, 15–40), `programme_etudes` (5 cats), `code_postal_3`, `revenu_familial_estime`, `heures_travail_semaine`, `distance_domicile_campus_km`, `premiere_generation_universitaire`.
- Proxies for region: postal code, distance, hours worked, household income. Dropping `region_administrative` (and postal code) barely moves the parity gap (0.188 → 0.181 → 0.173) — "fairness through unawareness" does not work.

## Baseline (`baseline_model.ipynb`)

RandomForest (300 trees, `min_samples_leaf=20`, `random_state=42`) on one-hot features incl. region; 70/30 stratified split, ~88% accuracy vs `decision_octroi`. `encoder()` one-hot encodes `CATEGORIELLES` and **reindexes target columns to the training columns** — keep this alignment in any new pipeline. The last cell writes a valid `predictions.csv` template. Fairness audit uses `fairlearn.metrics.MetricFrame`, `demographic_parity_difference`, TPR by group.

## Critical constraints (the grading depends on these)

1. **Budget:** grant rate on the 4,000 evaluation candidates must be within **36%–44%**, or the 35 technical points score zero. Always check `predictions.mean()` before writing the file; prefer a rank/top-k or calibrated threshold to hit ~40% exactly.
2. **`decision_octroi` is biased, not the target.** Scoring is against a hidden, independently built "real merit" reference standard. Maximizing agreement with `decision_octroi` reproduces the bias. TPR/equal-opportunity measured against `decision_octroi` only measures fidelity to the committee.
3. **Automated scoring (35 pts):** Equity 20 pts = share of the baseline **equal-opportunity gap (0.270)** closed; Utility 15 pts = agreement with the reference standard (scaled between random budget-respecting draw and perfect). HxBuddy shows only indicative macro F1 / accuracy vs real merit.
4. Demographic parity and equal opportunity can't both hold; pick one (scorer uses equal opportunity) and justify it.

## Deliverables (repo root)

- `predictions.csv` — columns `id_candidat,decision_octroi`, 4,000 rows + header, values 0/1, same IDs as `candidats_evaluation.csv`.
- `audit_rapport.ipynb` — bias measurement, chosen fairness metrics with justification, proxy variables found.
- `model_corrige.py` or `.ipynb` — mitigation plus a **Pareto front plot** (fairness vs utility) sweeping the fairness constraint (e.g. `fairlearn.postprocessing.ThresholdOptimizer` — seconds; `fairlearn.reductions.ExponentiatedGradient` — minutes).
- `presentation.pdf` — 5-minute pitch.

Jury-scored sections: diagnostic rigour (25), governance & ethics incl. monitoring plan (25), pitch & code quality (15).

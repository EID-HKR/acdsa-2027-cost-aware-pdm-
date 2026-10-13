# ACDSA 2027 — reproducibility materials (public distribution edition)

**Paper:** *From Smart Hospitals to Smart Homes: Cost-Aware Predictive Maintenance for Home Medical Equipment* (ACDSA 2027, paper 290).

## Dataset attribution and rights
The source is the publicly accessible **Predictive Medical Equipment Maintenance System (PMEMS)** project by **Swarnodip Nag**: https://github.com/Swarnodip-Nag/PMEMS . The analysis uses **only observations with `equipment_type == "Ventilator"`**, resulting in **220 samples** in the dataset copy used for this reconstruction. No other equipment categories are analyzed. The source data are synthetic, with rule-generated `failure_probability` and illustrative `cost_implications`; they are not clinically validated observations.

**The original dataset and the 220-row ventilator extract are NOT included in this public package.** Public availability of a repository does not itself grant redistribution rights. Obtain the data directly from the original author’s repository and respect its terms. This package contains only our independently developed analysis code, figures, derived summary statistics and documentation.

The exact original repository commit corresponding to the supplied 220-row workbook was **not verified**. For strict reproducibility, archive the upstream commit ID, download date and SHA-256 hash of the locally obtained source file when known. If upstream data change, the results may differ.

## How to reproduce
1. Obtain the original PMEMS data from https://github.com/Swarnodip-Nag/PMEMS . Download the dataset locally. If needed, export it to `.csv` or `.xlsx` while **preserving row order**. If using the supplied study subset from a permitted local source, its worksheet is named `Ventilators`.
2. Ensure the file includes the column `equipment_type`, with 220 rows labelled `Ventilator` after filtering, and the required feature and target columns.
3. Place the local file next to the script as `ventilator_data.xlsx`, **or** pass its path with `--data`.
4. Set up Python and run:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python reconstruct_acdsa.py --data /path/to/local_original_data.xlsx
python verify.py
```

The script accepts `.xlsx` or `.csv`, selects only `Ventilator` rows, and refuses to proceed if the subset size differs from 220. It writes `ac_dsa_reconstruction_metrics.json` and five figures in the script directory. The verification script checks those derived outputs against documented expectations. On a different upstream dataset revision, verification may fail as intended.

## Experiment specification
- Nine numerical predictors: `temperature_avg`, `humidity_avg`, `last_maintenance_days`, `uptime_hours`, `sensor_1`, `sensor_2`, `vibration`, `voltage_fluctuation`, `pressure`.
- `failure_probability` is a synthetic regression target; `cost_implications` is used only for decision costs, not as a predictor.
- 80/20 training/test split (`random_state=42`); `GradientBoostingRegressor` defaults except `random_state=42`.
- Risk predictions clipped to [0, 1]; grouped feature importance is the sum of `feature_importances_` over environment/history/technical groups.
- Evaluate expected costs on 44 held-out cases. Service if `p_i > t`, otherwise cost `p_i * C_f,i`; `t` from 0.05 through 0.95 in increments of 0.05; ties use the first minimum (lowest threshold).
- Hospital/baseline: preventive cost 2000, failure costs as given in dataset. Home: preventive cost 4000, failure costs multiplied by 1.5. Environmental counterfactuals: temperature ±5 and humidity ±10, holding other features constant. All costs and perturbations are illustrative.

## Scientific scope
This is an **independent reconstruction** developed after the original experiment code became unavailable. It reproduces the **revised** computational case study, not the original missing implementation. High model scores indicate fit to synthetic labels, **not** clinical performance. The common-threshold optimum is only within the evaluated policy class, not necessarily the device-specific optimum. The included derived outputs were generated from the author's locally provided 220-row ventilator workbook; the workbook itself is deliberately omitted.

## Contents
- `reconstruct_acdsa.py`: runnable model, cost policy and figure generation.
- `verify.py`: computational assertions against documented expected outputs.
- `requirements.txt`: dependency versions from the tested environment.
- `fig2_reconstruction.png` ... `fig6_reconstruction.png`: derived figures.
- `ac_dsa_reconstruction_metrics.json`, `verification_results.json`: derived outputs and verification summary.
- `index.html`: static landing page ready for GitHub Pages.

## Publishing
Upload this directory to a public GitHub repository, enable Pages for the main branch root, and optionally archive a release with Zenodo. **No repository, public website or DOI has yet been created.** Do not upload your locally obtained original dataset unless permission is established.

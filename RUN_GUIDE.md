# Telecom Customer Churn Timing Analysis: Execution & Metrics Guide

This guide explains how to run the project on Windows, how the execution pipeline works, and where to inspect and interpret all generated metrics, statistical tables, and visualizations.

---

## 1. Quick Start: Running the Project on Windows

The Python virtual environment has already been configured for Windows with Python 3.12 and all 28 required dependencies installed.

### Option A: Direct Command (Fastest)

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\sa\Downloads\AIBA\AIBA
.\.venv\Scripts\python.exe run_study.py
```

### Option B: Activated Shell

```powershell
cd C:\Users\sa\Downloads\AIBA\AIBA
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python run_study.py
```

> The study runs in approximately 1 to 2 minutes. It loads 7,032 customer records, expands them into 228,000+ person-month observation rows, and evaluates 4 distinct models across 5 random cross-validation seeds (`42`, `7`, `19`, `101`, `2024`).

---

## 2. Methodology & Evaluated Models

The project studies **when** telecom customers leave, rather than just **if** they leave, comparing four survival and hazard models:

| Model Identifier | Display Name | Architecture |
| :--- | :--- | :--- |
| `screened` | **Screened Hazard (Proposed)** | Discrete-time hazard model with forward stepwise discovery of attribute-by-tenure interactions validated via Bonferroni-corrected likelihood-ratio tests on a held-out confirmation half. |
| `level` | **Level Hazard** | Standard discrete-time logistic hazard model assuming constant covariate effects across all tenure months. |
| `cox` | **Cox Proportional Hazards** | Continuous-time semi-parametric Cox model evaluating baseline cumulative hazard with constant hazard ratios. |
| `gbm` | **Boosted Hazard** | Non-parametric `HistGradientBoostingClassifier` trained on tenure month and customer attributes. |

---

## 3. Directory Map of Outputs

Every run clears and regenerates the `results/` folder:

```
AIBA/AIBA/results/
├── summary.json                               # Master JSON export with experiment configuration & results
├── metrics_summary.csv                        # Aggregated means & standard deviations across all 5 splits
├── metrics_by_split.csv                       # Detailed metric scores for each seed and model
├── paired_differences.csv                     # Statistical differences comparing Screened vs benchmarks
├── confirmation_by_split.csv                  # LR and Wald confirmation tests on held-out test splits
├── selection_frequency.csv                    # Selection count of interaction terms across 5 splits
├── confirmation_coefficients_primary.csv      # Odds ratios and 95% CIs for primary seed (42)
├── level_coefficients_primary.csv             # Baseline level coefficients
├── renewal_negative_control.csv               # Empirical test of contract renewal window effects
├── ph_diagnostics.csv                         # Grambsch-Therneau Proportional Hazards test results
├── empirical_hazard.csv                       # Raw monthly hazard rates by charge quartile & payment
├── window_calibration_*.csv                   # Subgroup calibration error per tenure window
└── fig_*.png                                  # High-resolution (300 DPI) publication figures
```

---

## 4. Where to Find and How to Interpret Key Metrics

### A. Overall Model Performance (`results/metrics_summary.csv`)

This file provides the primary benchmark across the 4 models averaged over all 5 random test splits:

| Metric | What It Measures | Target | Key Observation |
| :--- | :--- | :--- | :--- |
| **`c_index_mean`** | Concordance Index: ranking discrimination of churn timing. | Higher is better (0.5 = random, 1.0 = perfect) | All models achieve strong discrimination (~0.86). |
| **`ibs_1_24_mean`** | Integrated Brier Score over months 1 to 24 (IPCW probability error). | Lower is better | **Screened Hazard achieves lowest IBS (0.0812)** vs Cox (0.0823) and Level (0.0821). |
| **`brier_1_mean`** | Month 1 Brier Score: immediate onboarding exit probability error. | Lower is better | **Screened (0.0427)** matches GBM (0.0427) and significantly outperforms Cox (0.0442) and Level (0.0442). |
| **`wce_m1_mean`** | Window Calibration Error in Month 1: absolute predicted vs observed gap across customer subgroups. | Lower is better | **Screened (0.0122)** cuts calibration error by ~23% compared to Level (0.0158) and Cox (0.0152). |
| **`wce_all_mean`** | Overall Calibration Error across all windows (months 1, 2-6, 7-24). | Lower is better | Screened (0.0020) and GBM (0.0020) exhibit superior calibration across sub-populations. |

```csv
model,c_index_mean,c_index_std,ibs_1_24_mean,ibs_1_24_std,brier_1_mean,brier_1_std,wce_m1_mean,wce_all_mean
screened,0.8625,0.0033,0.0812,0.0021,0.0427,0.0017,0.0122,0.0020
level,0.8633,0.0029,0.0821,0.0021,0.0442,0.0018,0.0158,0.0021
cox,0.8635,0.0030,0.0823,0.0021,0.0442,0.0018,0.0152,0.0022
gbm,0.8584,0.0025,0.0820,0.0017,0.0427,0.0016,0.0120,0.0020
```

---

### B. Head-to-Head Model Comparisons (`results/paired_differences.csv`)

Quantifies whether the Screened model's improvements are statistically robust across seeds:

- **`mean_diff`**: Average difference (`Screened - Competitor`). For error metrics (Brier, IBS, WCE), negative values indicate the Screened model is superior.
- **`splits_screened_better`**: Out of 5 random splits, how many times the Screened model won.
  - In `brier_1`: Screened beat Cox and Level in **5 out of 5 splits** (`mean_diff = -0.0015`).
  - In `wce_m1`: Screened beat Cox and Level in **5 out of 5 splits** (`mean_diff = -0.0030` to `-0.0036`).
  - In `ibs_1_24`: Screened beat Cox and Level in **5 out of 5 splits** (`mean_diff = -0.0009` to `-0.0011`).

---

### C. Factor Selection Stability (`results/selection_frequency.csv`)

Shows which time-varying interactions were discovered by the forward screening algorithm:

| Variable & Window | Description | Selection Frequency | Empirical Meaning |
| :--- | :--- | :---: | :--- |
| `charges_z@m1` | Monthly charge in Month 1 | **5 / 5 splits** | High first-month billing shock is the single strongest and most persistent driver of immediate exit. |
| `fiber@m1` | Fiber optic service in Month 1 | **4 / 5 splits** | Fiber subscribers face initial onboarding/installation churn risk. |
| `charges_z@m2_6` | Monthly charge in Months 2 to 6 | **4 / 5 splits** | Billing sensitivity persists throughout early tenure. |
| `credit_card@m1` | Automatic credit card in Month 1 | 1 / 5 splits | Sporadic effect. |

---

### D. Hypothesis Testing on Confirmation Data (`results/confirmation_by_split.csv`)

The Screen-Then-Confirm framework guards against false positives by testing discovered terms on an independent held-out half:
- **Likelihood-Ratio Test (`lr_stat`, `lr_p`)**: p-values range from $2.0 \times 10^{-16}$ to $2.5 \times 10^{-11}$, rejecting the null hypothesis of constant hazard across all splits.
- **Clustered Wald Test (`wald_stat`, `wald_p`)**: Robust clustered variance accounts for repeat observations per customer; all splits confirm statistical significance ($p < 10^{-10}$).

---

### E. Negative Control: Contract Renewal (`results/renewal_negative_control.csv`)

Tests the common domain hypothesis that customers churn primarily at contract expiration (months 11-13 for 1-year contracts, months 23-25 for 2-year contracts):
- **Odds Ratio**: $1.015$ (95% CI: $[0.372, 2.767]$)
- **p-value**: $0.977$
- **Conclusion**: There is **no evidence** of an isolated renewal cliff in this dataset; churn is driven by early tenure experience rather than expiration month.

---

## 5. Visual Artifacts Explained

The pipeline generates publication-ready figures in `results/`:

1. **Study Pipeline Architecture (`results/fig_pipeline.png`)**: Visualizes the 50/50 discovery-confirmation split, forward interaction screening under Bonferroni threshold, and validation scoring.
2. **Empirical Hazard Profiles (`results/fig_empirical.png`)**: Shows monthly exit hazards: Panel (a) breaks down hazard rates by monthly charge quartile, illustrating the sharp Month-1 departure spike for top-quartile bills. Panel (b) compares manual check payments against automatic payment methods.
3. **Effect Sizes & Odds Ratios (`results/fig_forest.png`)**: Forest plot with 95% confidence intervals showing the magnitude of tenure-localised interaction terms identified during confirmation.
4. **IPCW Brier Score by Horizon (`results/fig_brier.png`)**: Tracks predictive probability error across months 1, 3, 6, 12, and 24, demonstrating where the Screened model gains an advantage over Cox and Level models.
5. **Month-1 Calibration Accuracy (`results/fig_month1.png`)**: Compares observed month-1 exit rates against model predictions across payment methods (Bank transfer, Credit card, Electronic check, Mailed check).

---

## 6. How to Extend or Customize the Study

All configuration constants are located at the top of `run_study.py`:

- **Change evaluation horizons**: Modify `BRIER_MONTHS = (1, 3, 6, 12, 24)`
- **Change tenure windows**: Edit `WINDOWS = {"m1": (1, 1), "m2_6": (2, 6), "m7_24": (7, 24)}`
- **Modify random seeds**: Edit `SEEDS = (42, 7, 19, 101, 2024)`
- **Adjust Bonferroni discovery threshold**: Alter `ALPHA = 0.05`

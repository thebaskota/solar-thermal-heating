# Gas Consumption Prediction — ML Pipeline

Predict weekly gas consumption from solar-thermal heating data using PCA-based regression, a polynomial benchmark, random forest, and physics-based HDD models.

**Canonical analysis:** run scripts `01_preprocessing.py` through `08_model_comparison.py`. The file `solar_heating_analysis_reproducible.py` is deprecated (kept for reference only).

## Setup

```bash
cd project_task_2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `Heating-data.csv` in the project root (tab-separated).

## Run the pipeline

```bash
python 01_preprocessing.py
python 02_exploratory_analysis.py
python 03_pca_analysis.py
python 04_linear_regression.py
python 05_polynomial_regression.py
python 06_random_forest.py
python 07_hdd_model.py
python 08_model_comparison.py
```

Or as one chain:

```bash
python 01_preprocessing.py && \
python 02_exploratory_analysis.py && \
python 03_pca_analysis.py && \
python 04_linear_regression.py && \
python 05_polynomial_regression.py && \
python 06_random_forest.py && \
python 07_hdd_model.py && \
python 08_model_comparison.py
```

## What each step does

| Step | Script | Output folder |
|------|--------|---------------|
| 1 | Load, validate, sort raw data | `outputs/01_preprocessing/` |
| 2 | EDA + VIF multicollinearity (motivates PCA) | `outputs/02_exploratory_analysis/` |
| 3 | PCA exploration (scree, loadings, variance) — set `N_PCA_COMPONENTS` in `model_outputs.py` | `outputs/03_pca_analysis/` |
| 4 | Temperature-only + PCA linear baselines; Ramsey RESET on PCA linear | `outputs/04_linear_regression/` |
| 5 | Polynomial regression (degree 2 on PCA features) | `outputs/05_polynomial_regression/` |
| 6 | Random forest (tuned via time-series CV) | `outputs/06_random_forest/` |
| 7 | HDD threshold + HDD + solar (CV-tuned balance temperature) | `outputs/07_hdd_model/` |
| 8 | Lean comparison of six models + holdout R² chart | `outputs/08_model_comparison/` |

Step 7 uses preprocessed data only (not PCA). Steps 4–6 fit scaler + PCA inside sklearn `Pipeline` on raw predictors so each CV fold is fit without future training data.

## Methodology

- **Train/test split:** 80/20 chronological (278 train / 70 test weeks).
- **Hyperparameter tuning:** 5-fold time-series CV on the training partition only.
- **PCA:** `N_PCA_COMPONENTS = 3` from step 3 scree plot; refit inside pipelines per CV fold.
- **HDD:** Two variants tuned over base temperatures 10–20 °C (step 1 °C): HDD-only and HDD + solar yield.
- **Functional-form tests:** Ramsey RESET and nested F-test on PCA-linear (step 4, statsmodels).
- **Reproducibility:** Fixed `random_state=42` in `model_outputs.py`.

## Outputs

Standard per-model files (via `model_outputs.py`):

- `model_evaluation.csv` — train/test R², MSE, RMSE, MAE, BIC (where applicable)
- `cv_summary.csv` — tuning CV mean ± std
- `test_predictions.csv` — held-out predictions with dates
- Diagnostic plots

Step-specific extras:

| Step | Key files |
|------|-----------|
| 4 | `baseline_comparison.csv`, `pca_linear/functional_form_diagnostics.csv` |
| 6 | `rf_hyperparameter_search.csv`, `feature_importance.csv` |
| 7 | `hdd_variant_comparison.csv`, `hdd_solar/hdd_ols_inference.csv`, `temperature_hdd_narrative.png` |
| 8 | `model_comparison.csv`, `model_r2_comparison.png`, `final_holdout_predictions.csv` |

## Report

The submission report is [`reports/final_report.md`](reports/final_report.md). Regenerate figures by running the full pipeline, then export to PDF if needed.

## Notes

- `model_outputs.py` is a shared helper — do not run it directly.
- `outputs/` is gitignored; re-run the pipeline to regenerate results.

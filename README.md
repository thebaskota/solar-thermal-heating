# Gas Consumption Prediction

Predict weekly gas consumption from solar-thermal heating data using PCA-based regression, a polynomial benchmark, random forest, and physics-based heating-degree-day (HDD) models.

## Setup

Requires **Python 3.9+**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `Heating-data.csv` in the project root (tab-separated).

## Run the pipeline

Run the numbered scripts in order:

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

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_preprocessing.py` | Load, validate (including weekly date spacing), and sort raw data |
| 2 | `02_exploratory_analysis.py` | Exploratory analysis and VIF multicollinearity checks (development sample for model-motivating stats) |
| 3 | `03_pca_analysis.py` | PCA exploration (scree plot, loadings, variance) |
| 4 | `04_linear_regression.py` | Temperature-only and PCA linear baselines |
| 5 | `05_polynomial_regression.py` | Polynomial regression (degree 2 on PCA features, OLS) |
| 6 | `06_random_forest.py` | Random forest tuned via time-series cross-validation |
| 7 | `07_hdd_model.py` | HDD threshold and HDD + solar models (CV-tuned balance temperature) |
| 8 | `08_model_comparison.py` | Comparison of all six models on the holdout set |

Step 7 uses preprocessed data only (not PCA). Steps 4–6 fit a scaler and PCA inside sklearn `Pipeline` objects on raw predictors so each CV fold is fit without future training data.

Each script writes evaluation metrics, predictions, and diagnostic plots when run. Shared helpers and configuration live in `model_outputs.py` (import only — do not run directly).

## Methodology

- **Train/test split:** 80/20 chronological (278 train / 70 test weeks).
- **Hyperparameter tuning:** 5-fold time-series CV on the training partition only.
- **PCA:** `N_PCA_COMPONENTS = 3` (set in `model_outputs.py` from step 3 scree plot); refit inside pipelines per CV fold.
- **HDD:** Two variants tuned over base temperatures 10–20 °C (step 1 °C): HDD-only and HDD + solar yield.
- **Reproducibility:** Fixed `random_state=42` in `model_outputs.py`.
- **BIC:** Computed on the development sample only (classical in-sample criterion); not reported for holdout or random forest.

## Project structure

```
├── 01_preprocessing.py
├── 02_exploratory_analysis.py
├── 03_pca_analysis.py
├── 04_linear_regression.py
├── 05_polynomial_regression.py
├── 06_random_forest.py
├── 07_hdd_model.py
├── 08_model_comparison.py
├── model_outputs.py      # shared config and output helpers
├── requirements.txt
└── Heating-data.csv      # input data
```

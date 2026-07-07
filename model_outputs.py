"""
Shared helpers for saving comparable regression model outputs.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.stattools import durbin_watson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
STABILITY_CV_FOLDS = 4
N_PCA_COMPONENTS = 3
POLY_INCLUDE_BIAS = False

DATE_COLUMN = "Date"
TARGET_COLUMN = "Gas consumption [kWh/day]"
TEMP_COLUMN = "Outdoor temperature [°C]"
SOLAR_COLUMN = "Solar yield [kWh/day]"
VALVE_COLUMN = "Valve [h/day]"

PREDICTOR_COLUMNS = [
    "Sunshine duration [h/day]",
    TEMP_COLUMN,
    SOLAR_COLUMN,
    "Solar pump [h/day]",
    VALVE_COLUMN,
]

HDD_BASE_TEMPS = list(np.arange(10.0, 20.1, 1.0))

RF_PARAM_GRID = {
    "regressor__n_estimators": [250],
    "regressor__max_depth": [2, 3, None],
    "regressor__min_samples_leaf": [1, 3],
    "regressor__max_features": [1.0],
}

HDD_BASE_TEMP_GRID = {
    "hdd__base_temperature": HDD_BASE_TEMPS,
}

HDD_SOLAR_BASE_TEMP_GRID = {
    "features__base_temperature": HDD_BASE_TEMPS,
}

EVALUATION_FILE = "model_evaluation.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
COEFFICIENTS_FILE = "model_coefficients.csv"
CV_SUMMARY_FILE = "cv_summary.csv"
HYPERPARAMETERS_FILE = "hyperparameters.json"

PREPROCESSED_FILE = Path("outputs") / "01_preprocessing" / "preprocessed_data.csv"


def chronological_split(df, test_size=TEST_SIZE):
    """Return split index for chronological train/test partition."""
    return int(len(df) * (1 - test_size))


def calculate_bic(y_true, y_pred, n_params):
    """BIC = n * ln(RSS/n) + k * ln(n). Lower is better."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    if rss <= 0:
        rss = 1e-12
    return float(n * np.log(rss / n) + n_params * np.log(n))


def calculate_metrics(y_true, y_pred, n_params=None):
    metrics = {
        "R²": r2_score(y_true, y_pred),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
    if n_params is not None:
        metrics["BIC"] = calculate_bic(y_true, y_pred, n_params)
    return metrics


def build_pca_pipeline(regressor, n_components=N_PCA_COMPONENTS):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("regressor", regressor),
    ])


def build_raw_pipeline(regressor):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def build_pca_poly_pipeline(degree, n_components=N_PCA_COMPONENTS):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("poly", PolynomialFeatures(degree=degree, include_bias=POLY_INCLUDE_BIAS)),
        ("regressor", LinearRegression()),
    ])


def poly_n_params(model):
    """Intercept + polynomial feature count (include_bias=False → add constant)."""
    return model.named_steps["poly"].n_output_features_ + 1


class HDDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_temperature=15.0):
        self.base_temperature = base_temperature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            temp = X[TEMP_COLUMN].to_numpy(dtype=float)
        else:
            temp = X[:, PREDICTOR_COLUMNS.index(TEMP_COLUMN)].astype(float)
        return np.maximum(0, self.base_temperature - temp).reshape(-1, 1)


class HDDPlusSolarOnlyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_temperature=15.0):
        self.base_temperature = base_temperature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            temp = X[TEMP_COLUMN].to_numpy(dtype=float)
            solar = X[SOLAR_COLUMN].to_numpy(dtype=float)
        else:
            temp = X[:, PREDICTOR_COLUMNS.index(TEMP_COLUMN)].astype(float)
            solar = X[:, PREDICTOR_COLUMNS.index(SOLAR_COLUMN)].astype(float)
        hdd = np.maximum(0, self.base_temperature - temp)
        return np.column_stack([hdd, solar])


class HDDPlusSolarValveTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_temperature=15.0):
        self.base_temperature = base_temperature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            temp = X[TEMP_COLUMN].to_numpy(dtype=float)
            solar = X[SOLAR_COLUMN].to_numpy(dtype=float)
            valve = X[VALVE_COLUMN].to_numpy(dtype=float)
        else:
            temp = X[:, PREDICTOR_COLUMNS.index(TEMP_COLUMN)].astype(float)
            solar = X[:, PREDICTOR_COLUMNS.index(SOLAR_COLUMN)].astype(float)
            valve = X[:, PREDICTOR_COLUMNS.index(VALVE_COLUMN)].astype(float)
        hdd = np.maximum(0, self.base_temperature - temp)
        return np.column_stack([hdd, solar, valve])


def build_hdd_pipeline(variant="hdd_solar"):
    if variant == "hdd_only":
        return Pipeline([
            ("hdd", HDDTransformer()),
            ("model", LinearRegression()),
        ])
    if variant == "hdd_solar":
        return Pipeline([
            ("features", HDDPlusSolarOnlyTransformer()),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    if variant == "hdd_solar_valve":
        return Pipeline([
            ("features", HDDPlusSolarValveTransformer()),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    raise ValueError(f"Unknown HDD variant: {variant}")


def hdd_base_temp_grid(variant):
    if variant == "hdd_only":
        return HDD_BASE_TEMP_GRID
    return HDD_SOLAR_BASE_TEMP_GRID


def hdd_n_params(variant):
    return {"hdd_only": 2, "hdd_solar": 3, "hdd_solar_valve": 4}[variant]


def fit_pca_ols_diagnostics(X_train, y_train, n_components=N_PCA_COMPONENTS):
    scaler = StandardScaler().fit(X_train)
    pcs_train = PCA(n_components=n_components).fit_transform(scaler.transform(X_train))
    ols_lin = sm.OLS(y_train, sm.add_constant(pcs_train)).fit()
    poly_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(pcs_train)
    ols_poly = sm.OLS(y_train, sm.add_constant(poly_train)).fit()
    return pcs_train, ols_lin, ols_poly


def save_functional_form_diagnostics(output_dir, ols_lin, ols_poly):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reset = linear_reset(ols_lin, power=3, use_f=True)
    fstat, fpvalue, df_diff = ols_poly.compare_f_test(ols_lin)
    dw = durbin_watson(ols_lin.resid)
    residuals = np.asarray(ols_lin.resid)
    acf_lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) > 1 else np.nan

    rows = [
        {"test": "Ramsey RESET", "statistic": "F", "value": float(reset.fvalue), "p_value": float(reset.pvalue)},
        {"test": "Nested F (linear vs quadratic PCA)", "statistic": "F", "value": float(fstat), "p_value": float(fpvalue), "df_diff": float(df_diff)},
        {"test": "Durbin-Watson", "statistic": "DW", "value": dw, "p_value": np.nan},
        {"test": "Residual lag-1 ACF", "statistic": "ACF", "value": acf_lag1, "p_value": np.nan},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "functional_form_diagnostics.csv", index=False)

    max_lag = min(12, len(residuals) - 1)
    if max_lag > 0:
        lags = np.arange(1, max_lag + 1)
        acfs = [float(np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]) for lag in lags]
        conf = 1.96 / np.sqrt(len(residuals))
        plt.figure(figsize=(8, 4))
        plt.stem(lags, acfs, basefmt=" ")
        plt.axhline(conf, linestyle="--", linewidth=1)
        plt.axhline(-conf, linestyle="--", linewidth=1)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Lag [weeks]")
        plt.ylabel("Residual autocorrelation")
        plt.title("Development residual autocorrelation — PCA + linear")
        plt.tight_layout()
        plt.savefig(output_dir / "residual_acf_development.png", dpi=150)
        plt.close()

    return rows


def save_ols_inference_table(output_dir, X_train, y_train, term_labels):
    output_dir = Path(output_dir)
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    pd.DataFrame({
        "term": ["intercept", *term_labels],
        "coefficient": ols.params,
        "std_error": ols.bse,
        "t": ols.tvalues,
        "p_value": ols.pvalues,
    }).to_csv(output_dir / "hdd_ols_inference.csv", index=False)
    return ols


def save_test_coverage_table(output_dir, train_df, test_df, predictors=None):
    predictors = predictors or PREDICTOR_COLUMNS
    rows = []
    for col in predictors:
        tr_min, tr_max = train_df[col].min(), train_df[col].max()
        te_min, te_max = test_df[col].min(), test_df[col].max()
        outside = int(((test_df[col] < tr_min) | (test_df[col] > tr_max)).sum())
        rows.append({
            "predictor": col,
            "train_min": tr_min,
            "train_max": tr_max,
            "test_min": te_min,
            "test_max": te_max,
            "test_points_outside_train_range": outside,
        })
    path = Path(output_dir) / "test_range_coverage.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_expanding_stability_cv(model, X_train, y_train, n_splits=STABILITY_CV_FOLDS):
    outer_cv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train), start=1):
        est = clone(model)
        if isinstance(X_train, pd.DataFrame):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        else:
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        est.fit(X_tr, y_train[tr_idx])
        pred = est.predict(X_va)
        y_va = y_train[va_idx]
        fold_rows.append({
            "fold": fold,
            "n_train": len(tr_idx),
            "n_validation": len(va_idx),
            "R2": r2_score(y_va, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_va, pred))),
            "MAE": float(mean_absolute_error(y_va, pred)),
        })
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "stability_cv_r2_mean": float(fold_df["R2"].mean()),
        "stability_cv_r2_std": float(fold_df["R2"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "stability_cv_rmse_mean": float(fold_df["RMSE"].mean()),
        "stability_cv_rmse_std": float(fold_df["RMSE"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "stability_cv_mae_mean": float(fold_df["MAE"].mean()),
        "stability_cv_fold_r2": " | ".join(f"{x:.3f}" for x in fold_df["R2"]),
    }
    return summary, fold_df


def time_series_cv_summary(estimator, X_train, y_train, tscv):
    """Mean ± SD for R², RMSE, MAE across time-series CV folds."""
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    cv = cross_validate(estimator, X_train, y_train, cv=tscv, scoring=scoring)
    return {
        "cv_r2_mean": float(cv["test_r2"].mean()),
        "cv_r2_std": float(cv["test_r2"].std()),
        "cv_mse_mean": float(-cv["test_neg_mse"].mean()),
        "cv_mse_std": float(-cv["test_neg_mse"].std()),
        "cv_rmse_mean": float(-cv["test_neg_rmse"].mean()),
        "cv_rmse_std": float(-cv["test_neg_rmse"].std()),
        "cv_mae_mean": float(-cv["test_neg_mae"].mean()),
        "cv_mae_std": float(-cv["test_neg_mae"].std()),
    }


def save_cv_summary(output_dir, cv_summary):
    pd.DataFrame([cv_summary]).to_csv(Path(output_dir) / CV_SUMMARY_FILE, index=False)


def save_hyperparameters(output_dir, params_dict):
    path = Path(output_dir) / HYPERPARAMETERS_FILE
    serialisable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in params_dict.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


def print_model_performance(train_metrics, test_metrics):
    metric_names = ["R²", "MSE", "RMSE", "MAE"]
    if "BIC" in train_metrics:
        metric_names.append("BIC")
    print(f"{'Metric':<10} {'Training':>12} {'Testing':>12}")
    print("-" * 36)
    for name in metric_names:
        if name not in train_metrics:
            continue
        t, v = train_metrics[name], test_metrics[name]
        if name in ("R²",):
            print(f"{name:<10} {t:>12.4f} {v:>12.4f}")
        elif name == "BIC":
            print(f"{name:<10} {t:>12.1f} {v:>12.1f}")
        else:
            print(f"{name:<10} {t:>12.2f} {v:>12.2f}")


def load_model_metrics(model_name, eval_path):
    df = pd.read_csv(eval_path)
    train = df.loc[df["split"] == "train"].iloc[0]
    test = df.loc[df["split"] == "test"].iloc[0]
    row = {
        "Model": model_name,
        "Train_R²": train["R²"],
        "Train_MSE": train.get("MSE", np.nan),
        "Train_RMSE": train["RMSE"],
        "Train_MAE": train["MAE"],
        "Test_R²": test["R²"],
        "Test_MSE": test.get("MSE", np.nan),
        "Test_RMSE": test["RMSE"],
        "Test_MAE": test["MAE"],
    }
    if "BIC" in train.index:
        row["Train_BIC"] = train["BIC"]
        row["Test_BIC"] = test["BIC"]
    return row


def save_model_coefficients(output_dir, terms, coefficients):
    pd.DataFrame({"term": terms, "coefficient": coefficients}).to_csv(
        Path(output_dir) / COEFFICIENTS_FILE, index=False
    )


def save_model_results(
    output_dir,
    model_name,
    dates,
    y_actual,
    y_all_pred,
    split_date,
    test_dates,
    y_test,
    y_test_pred,
    train_metrics,
    test_metrics,
):
    """Save standard metrics CSV, test predictions CSV, and diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(output_dir / EVALUATION_FILE, index=False)

    residuals = y_test - y_test_pred
    pd.DataFrame({
        DATE_COLUMN: test_dates.values if hasattr(test_dates, "values") else test_dates,
        "actual": y_test.values if hasattr(y_test, "values") else y_test,
        "predicted": y_test_pred,
        "residual": residuals.values if hasattr(residuals, "values") else residuals,
    }).to_csv(output_dir / TEST_PREDICTIONS_FILE, index=False)

    dates = pd.to_datetime(dates)
    y_actual = np.asarray(y_actual)
    y_all_pred = np.asarray(y_all_pred)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)
    residuals = np.asarray(residuals)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y = x")
    plt.xlabel("Actual Gas Consumption [kWh/day]")
    plt.ylabel("Predicted Gas Consumption [kWh/day]")
    plt.title(f"Actual vs Predicted — {model_name} (Testing Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_actual, "o-", color="steelblue", label="Actual", markersize=3, linewidth=1)
    plt.plot(dates, y_all_pred, "o-", color="darkorange", label="Predicted", markersize=3, linewidth=1, alpha=0.85)
    plt.axvline(pd.to_datetime(split_date), color="green", linestyle="--", linewidth=1.5, label="Train/test split")
    plt.xlabel("Date")
    plt.ylabel("Gas Consumption [kWh/day]")
    plt.title(f"Actual vs Predicted Over Time — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Gas Consumption [kWh/day]")
    plt.ylabel("Residual (Actual − Predicted) [kWh/day]")
    plt.title(f"Residual Plot — {model_name} (Testing Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Residual (Actual − Predicted) [kWh/day]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Residuals — {model_name} (Testing Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=150)
    plt.close()


def durbin_watson(residuals):
    residuals = np.asarray(residuals)
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


def save_residual_diagnostics(output_dir, model_name, y_test_pred, y_test, temperature_test):
    """Extended residual diagnostics: vs temperature, autocorrelation, Durbin-Watson."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)
    temperature_test = np.asarray(temperature_test)
    residuals = y_test - y_test_pred
    dw = durbin_watson(residuals)

    pd.DataFrame({"statistic": ["Durbin-Watson"], "value": [dw]}).to_csv(
        output_dir / "durbin_watson.csv", index=False
    )

    bands = [
        ("cold", temperature_test < 5),
        ("cool", (temperature_test >= 5) & (temperature_test < 12)),
        ("transition", (temperature_test >= 12) & (temperature_test < 18)),
        ("warm", temperature_test >= 18),
    ]
    band_rows = []
    for label, mask in bands:
        if mask.sum() == 0:
            continue
        band_rows.append({
            "temperature_band": label,
            "n_weeks": int(mask.sum()),
            "mean_residual": float(residuals[mask].mean()),
            "std_residual": float(residuals[mask].std()),
            "mean_abs_residual": float(np.abs(residuals[mask]).mean()),
        })
    pd.DataFrame(band_rows).to_csv(output_dir / "residual_by_temperature_band.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(temperature_test, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Outdoor Temperature [°C]")
    plt.ylabel("Residual (Actual − Predicted) [kWh/day]")
    plt.title(f"Residuals vs Temperature — {model_name} (Testing Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_temperature.png", dpi=150)
    plt.close()

    max_lag = min(20, len(residuals) - 1)
    if max_lag > 0:
        acf = [float(np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]) for lag in range(1, max_lag + 1)]
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, max_lag + 1), acf, color="steelblue", edgecolor="black")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xlabel("Lag (weeks)")
        plt.ylabel("Autocorrelation")
        plt.title(f"Residual Autocorrelation — {model_name} (Testing Set)")
        plt.tight_layout()
        plt.savefig(output_dir / "residual_autocorrelation.png", dpi=150)
        plt.close()


def init_plot_style():
    sns.set_style("whitegrid")

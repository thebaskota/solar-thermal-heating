"""
Step 7 — Physics-informed HDD linear regression + interpretation.

Select HDD base temperature via training CV; fit HDD-only and HDD + solar yield.
Run after 06_random_forest.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures

from model_outputs import (
    CV_FOLDS,
    DATE_COLUMN,
    PREDICTOR_COLUMNS,
    PREPROCESSED_FILE,
    SOLAR_COLUMN,
    TARGET_COLUMN,
    TEMP_COLUMN,
    build_hdd_pipeline,
    calculate_metrics,
    chronological_split,
    hdd_base_temp_grid,
    hdd_n_params,
    init_plot_style,
    print_model_performance,
    save_cv_summary,
    save_hyperparameters,
    save_model_coefficients,
    save_model_results,
    save_ols_inference_table,
    save_residual_diagnostics,
    save_test_coverage_table,
    time_series_cv_summary,
)

OUTPUT_DIR = Path("outputs") / "07_hdd_model"

HDD_VARIANTS = [
    ("hdd_only", "HDD Threshold", "hdd_threshold"),
    ("hdd_solar", "HDD + Solar Yield", "hdd_solar"),
]


def make_hdd(outdoor_temp, base_temp):
    return np.maximum(0.0, base_temp - outdoor_temp)


def build_hdd_features(df, base_temp):
    return pd.DataFrame({
        "HDD": make_hdd(df[TEMP_COLUMN].values, base_temp),
        SOLAR_COLUMN: df[SOLAR_COLUMN].values,
    })


def fit_hdd_variant(
    variant_key,
    model_name,
    subdir,
    train_df,
    test_df,
    df,
    split_idx,
    output_dir,
    tscv,
):
    variant_dir = output_dir / subdir
    variant_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[PREDICTOR_COLUMNS]
    X_test = test_df[PREDICTOR_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_hdd_pipeline(variant_key)
    search = GridSearchCV(
        pipeline,
        hdd_base_temp_grid(variant_key),
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    search.fit(X_train, y_train)
    best_base = float(search.best_params_[list(search.best_params_.keys())[0]])

    print(f"\n--- {model_name} ---")
    print(f"  Selected base temperature: {best_base:.1f}°C (CV RMSE = {-search.best_score_:.2f})")

    model = search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(df[PREDICTOR_COLUMNS])

    n_params = hdd_n_params(variant_key)
    train_metrics = calculate_metrics(y_train, y_train_pred, n_params)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print_model_performance(train_metrics, test_metrics)

    save_hyperparameters(variant_dir, {
        "variant": variant_key,
        "base_temperature_c": best_base,
        "best_cv_rmse": float(-search.best_score_),
    })

    cv_summary = time_series_cv_summary(model, X_train, y_train, tscv)
    save_cv_summary(variant_dir, cv_summary)

    save_model_results(
        variant_dir,
        model_name,
        df[DATE_COLUMN],
        df[TARGET_COLUMN],
        y_all_pred,
        df[DATE_COLUMN].iloc[split_idx],
        test_df[DATE_COLUMN],
        y_test,
        y_test_pred,
        train_metrics,
        test_metrics,
    )

    reg = model.named_steps["model"]
    if variant_key == "hdd_only":
        save_model_coefficients(
            variant_dir,
            ["intercept", "HDD"],
            [reg.intercept_, reg.coef_[0]],
        )
    else:
        save_model_coefficients(
            variant_dir,
            ["intercept", "HDD", SOLAR_COLUMN],
            [reg.intercept_, reg.coef_[0], reg.coef_[1]],
        )
        hdd_features = build_hdd_features(train_df, best_base)
        save_ols_inference_table(
            variant_dir,
            hdd_features.values,
            y_train,
            [f"HDD (base {best_base:.1f} °C)", SOLAR_COLUMN],
        )

    if variant_key == "hdd_solar":
        save_residual_diagnostics(
            variant_dir, model_name, y_test_pred, y_test, test_df[TEMP_COLUMN].values,
        )
        param_name = list(search.best_params_.keys())[0]
        search_records = []
        for i, params in enumerate(search.cv_results_["params"]):
            search_records.append({
                "base_temperature": params[param_name],
                "cv_rmse_mean": float(-search.cv_results_["mean_test_score"][i]),
            })
        pd.DataFrame(search_records).sort_values("base_temperature").to_csv(
            output_dir / "hdd_base_temperature_search.csv", index=False,
        )

    return best_base, {
        "Model": model_name,
        "variant": variant_key,
        "Base_Temperature": best_base,
        "CV_R²_Mean": cv_summary["cv_r2_mean"],
        "CV_RMSE_Mean": cv_summary["cv_rmse_mean"],
        "Train_R²": train_metrics["R²"],
        "Train_RMSE": train_metrics["RMSE"],
        "Train_MAE": train_metrics["MAE"],
        "Test_R²": test_metrics["R²"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
        "Dev_BIC": train_metrics.get("BIC"),
    }


def save_fig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hdd_search(search_df, best_base, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(search_df["base_temperature"], search_df["cv_rmse_mean"], "o-", color="steelblue", linewidth=2)
    plt.axvline(x=best_base, color="green", linestyle="--", label=f"Selected: {best_base:.1f}°C")
    plt.title("HDD Base Temperature Search (training CV RMSE)")
    plt.xlabel("Base temperature [°C]")
    plt.ylabel("CV RMSE [kWh/day]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def threshold_stats(df, base_temp):
    above = df[df[TEMP_COLUMN] >= base_temp]
    below = df[df[TEMP_COLUMN] < base_temp]
    return {
        "base_temperature_c": base_temp,
        "mean_gas_above_base": above[TARGET_COLUMN].mean(),
        "mean_gas_below_base": below[TARGET_COLUMN].mean(),
        "pct_gas_below_100_above_base": (above[TARGET_COLUMN] < 100).mean() * 100,
        "summer_floor_median_kwh": above[TARGET_COLUMN].median(),
    }


def plot_temperature_hdd_narrative(df, train_df, base_temp, output_dir):
    temp = df[TEMP_COLUMN].values
    gas = df[TARGET_COLUMN].values
    hdd = make_hdd(temp, base_temp)
    thresh = threshold_stats(df, base_temp)

    X_train = train_df[[TEMP_COLUMN]].values
    y_train = train_df[TARGET_COLUMN].values
    temp_line = np.linspace(temp.min(), temp.max(), 300)

    curves = {}
    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(degree, include_bias=False)
        model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
        curves[degree] = model.predict(poly.transform(temp_line.reshape(-1, 1)))

    hdd_train = make_hdd(train_df[TEMP_COLUMN].values, base_temp).reshape(-1, 1)
    hdd_model = LinearRegression().fit(hdd_train, y_train)
    hdd_line = np.linspace(0, hdd.max(), 300).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(temp, gas, alpha=0.4, s=18, c="steelblue", edgecolors="none")
    for degree in curves:
        ax.plot(temp_line, curves[degree], lw=2, label=["Linear", "Quadratic", "Cubic"][degree - 1])
    ax.axvline(base_temp, color="green", ls="--", lw=1.5, label=f"Balance point ({base_temp}°C)")
    ax.axhline(thresh["summer_floor_median_kwh"], color="gray", ls=":", lw=1.2)
    ax.set(xlabel="Outdoor Temperature [°C]", ylabel="Gas [kWh/day]",
           title="A) Gas vs temperature — plateau + slope")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(hdd, gas, alpha=0.4, s=18, c=hdd, cmap="coolwarm", edgecolors="none")
    ax.plot(hdd_line, hdd_model.predict(hdd_line), "k-", lw=2.5,
            label=f"Gas = {hdd_model.intercept_:.0f} + {hdd_model.coef_[0]:.1f}·HDD")
    ax.set(xlabel=f"HDD (base {base_temp}°C)", ylabel="Gas [kWh/day]",
           title="B) Gas vs HDD — linear fit")
    ax.legend(fontsize=8)

    lin_resid = gas - LinearRegression().fit(X_train, y_train).predict(temp.reshape(-1, 1))
    ax = axes[2]
    ax.scatter(temp, lin_resid, alpha=0.4, s=18, c="steelblue", edgecolors="none")
    ax.axhline(0, color="red", ls="--")
    ax.axvline(base_temp, color="green", ls="--", lw=1.5)
    ax.set(xlabel="Outdoor Temperature [°C]", ylabel="Residual (actual − linear temp fit)",
           title="C) Linear temp model misses summer plateau")

    fig.suptitle("Piecewise heating demand — HDD linearises the relationship", fontsize=12, y=1.02)
    save_fig(output_dir / "temperature_hdd_narrative.png")


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    print(f"Loading: {script_dir / PREPROCESSED_FILE}")
    df = pd.read_csv(script_dir / PREPROCESSED_FILE).sort_values(DATE_COLUMN).reset_index(drop=True)

    split_idx = chronological_split(df)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nTraining set: {len(train_df)}  |  Testing set: {len(test_df)}")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    variant_rows = []
    best_base = None

    print("\n--- HDD variant search (training CV only) ---")
    for variant_key, model_name, subdir in HDD_VARIANTS:
        base_temp, row = fit_hdd_variant(
            variant_key, model_name, subdir,
            train_df, test_df, df, split_idx, output_dir, tscv,
        )
        variant_rows.append(row)
        if variant_key == "hdd_solar":
            best_base = base_temp

    pd.DataFrame(variant_rows).to_csv(output_dir / "hdd_variant_comparison.csv", index=False)
    save_test_coverage_table(output_dir, train_df, test_df)

    search_path = output_dir / "hdd_base_temperature_search.csv"
    if search_path.exists() and best_base is not None:
        plot_hdd_search(pd.read_csv(search_path), best_base, output_dir / "hdd_validation_performance.png")

    print(f"\n--- HDD interpretation (base = {best_base:.1f}°C) ---")
    plot_temperature_hdd_narrative(df, train_df, best_base, output_dir)
    pd.DataFrame([threshold_stats(df, best_base)]).to_csv(
        output_dir / "hdd_threshold_statistics.csv", index=False
    )

    print(f"\nSaved outputs to: {output_dir}")
    print("Next step: run 08_model_comparison.py")


if __name__ == "__main__":
    main()

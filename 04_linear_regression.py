"""
Step 4 — Linear regression baselines.

Fits and compares:
  - Temperature-only linear
  - PCA + linear regression (task-required)

Run after 03_pca_analysis.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from model_outputs import (
    CV_FOLDS,
    DATE_COLUMN,
    PREDICTOR_COLUMNS,
    PREPROCESSED_FILE,
    TARGET_COLUMN,
    TEMP_COLUMN,
    build_pca_pipeline,
    calculate_metrics,
    chronological_split,
    init_plot_style,
    print_model_performance,
    save_cv_summary,
    save_model_coefficients,
    save_model_results,
    save_residual_diagnostics,
    time_series_cv_summary,
)

OUTPUT_DIR = Path("outputs") / "04_linear_regression"


def run_model(
    script_dir,
    name,
    model,
    X_train,
    X_test,
    X_all,
    y_train,
    y_test,
    y_all,
    df,
    split_idx,
    output_subdir,
    n_params=None,
    cv_estimator=None,
    coef_terms=None,
    coef_values=None,
    save_residuals=False,
    temp_test=None,
):
    out = script_dir / OUTPUT_DIR / output_subdir
    out.mkdir(parents=True, exist_ok=True)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X_all)

    train_metrics = calculate_metrics(y_train, y_train_pred, n_params)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print(f"\n--- {name} ---")
    print_model_performance(train_metrics, test_metrics)

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_est = cv_estimator if cv_estimator is not None else model
    cv_summary = time_series_cv_summary(cv_est, X_train, y_train, tscv)
    save_cv_summary(out, cv_summary)
    print(f"  CV R² = {cv_summary['cv_r2_mean']:.4f} ± {cv_summary['cv_r2_std']:.4f}")

    save_model_results(
        out,
        name,
        df[DATE_COLUMN],
        y_all,
        y_all_pred,
        df[DATE_COLUMN].iloc[split_idx],
        df[DATE_COLUMN].iloc[split_idx:],
        y_test,
        y_test_pred,
        train_metrics,
        test_metrics,
    )

    if coef_terms is not None and coef_values is not None:
        save_model_coefficients(out, coef_terms, coef_values)

    if save_residuals and temp_test is not None:
        save_residual_diagnostics(out, name, y_test_pred, y_test, temp_test)

    row = {
        "Model": name,
        "n_params": n_params if n_params is not None else np.nan,
        "Train_R²": train_metrics["R²"],
        "Train_MSE": train_metrics["MSE"],
        "Train_RMSE": train_metrics["RMSE"],
        "Train_MAE": train_metrics["MAE"],
        "Test_R²": test_metrics["R²"],
        "Test_MSE": test_metrics["MSE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
        "CV_R²_Mean": cv_summary["cv_r2_mean"],
        "CV_R²_Std": cv_summary["cv_r2_std"],
        "CV_RMSE_Mean": cv_summary["cv_rmse_mean"],
        "CV_RMSE_Std": cv_summary["cv_rmse_std"],
    }
    if "BIC" in train_metrics:
        row["Dev_BIC"] = train_metrics["BIC"]
    return row


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
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    y_all = df[TARGET_COLUMN]
    temp_test = test_df[TEMP_COLUMN].values

    print(f"Training set: {len(train_df)}  |  Testing set: {len(test_df)}")

    rows = []
    shared = dict(
        script_dir=script_dir, y_train=y_train, y_test=y_test, y_all=y_all,
        df=df, split_idx=split_idx, temp_test=temp_test,
    )

    # Temperature-only
    temp_model = LinearRegression()
    rows.append(run_model(
        name="Temperature-Only Linear", model=temp_model,
        X_train=train_df[[TEMP_COLUMN]], X_test=test_df[[TEMP_COLUMN]], X_all=df[[TEMP_COLUMN]],
        output_subdir="temperature_only", n_params=2, **shared,
    ))
    save_model_coefficients(
        output_dir / "temperature_only",
        ["intercept", TEMP_COLUMN],
        [temp_model.intercept_, temp_model.coef_[0]],
    )

    # PCA + linear (pipeline for CV; fit on raw predictors)
    pca_pipe = build_pca_pipeline(LinearRegression())
    rows.append(run_model(
        name="PCA + Linear Regression", model=pca_pipe,
        X_train=train_df[PREDICTOR_COLUMNS], X_test=test_df[PREDICTOR_COLUMNS],
        X_all=df[PREDICTOR_COLUMNS],
        output_subdir="pca_linear", n_params=4, cv_estimator=pca_pipe,
        save_residuals=True, **shared,
    ))
    pca_reg = pca_pipe.named_steps["regressor"]
    pc_names = [f"PC{i}" for i in range(1, len(pca_reg.coef_) + 1)]
    save_model_coefficients(
        output_dir / "pca_linear",
        ["intercept", *pc_names],
        [pca_reg.intercept_, *pca_reg.coef_],
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print(f"\nSaved baseline comparison: {output_dir / 'baseline_comparison.csv'}")
    print("Next step: run 05_polynomial_regression.py")


if __name__ == "__main__":
    main()

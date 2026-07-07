"""
Step 5 — Polynomial regression (degree 2) on PCA features.

Scaler + PCA + quadratic features + Ridge are refit inside a sklearn Pipeline on
raw predictors (train-only within each CV fold). Ridge alpha is tuned by
time-series CV to stabilise early-fold estimates.
Run after 04_linear_regression.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model_outputs import (
    CV_FOLDS,
    DATE_COLUMN,
    N_PCA_COMPONENTS,
    POLY_RIDGE_PARAM_GRID,
    PREDICTOR_COLUMNS,
    PREPROCESSED_FILE,
    TARGET_COLUMN,
    TEMP_COLUMN,
    build_pca_poly_pipeline,
    calculate_metrics,
    chronological_split,
    init_plot_style,
    poly_n_params,
    print_model_performance,
    save_cv_summary,
    save_hyperparameters,
    save_model_results,
    save_residual_diagnostics,
    time_series_cv_summary,
)

MODEL_NAME = "Polynomial Regression (degree=2)"
POLY_DEGREE = 2
OUTPUT_DIR = Path("outputs") / "05_polynomial_regression"


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    print(f"Loading: {script_dir / PREPROCESSED_FILE}")
    df = pd.read_csv(script_dir / PREPROCESSED_FILE).sort_values(DATE_COLUMN).reset_index(drop=True)

    split_idx = chronological_split(df)
    temp_test = df.iloc[split_idx:][TEMP_COLUMN].values

    X = df[PREDICTOR_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nModel: {MODEL_NAME}")
    print(f"Predictors: {PREDICTOR_COLUMNS} → PCA (k={N_PCA_COMPONENTS}) → polynomial degree {POLY_DEGREE}")

    pipeline = build_pca_poly_pipeline(POLY_DEGREE)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    search = GridSearchCV(
        pipeline,
        POLY_RIDGE_PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    search.fit(X_train, y_train)

    best_alpha = float(search.best_params_["regressor__alpha"])
    print("\n--- Ridge alpha search (training CV) ---")
    print(f"Selected alpha: {best_alpha:g}  (CV RMSE = {-search.best_score_:.2f} kWh/day)")

    pd.DataFrame(search.cv_results_)[[
        "param_regressor__alpha",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]].sort_values("rank_test_score").to_csv(
        output_dir / "poly_ridge_alpha_search.csv", index=False,
    )

    model = search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X)
    n_params = poly_n_params(model)

    train_metrics = calculate_metrics(y_train, y_train_pred, n_params)
    test_metrics = calculate_metrics(y_test, y_test_pred, n_params)
    print_model_performance(train_metrics, test_metrics)

    cv_summary = time_series_cv_summary(model, X_train, y_train, tscv)
    save_cv_summary(output_dir, cv_summary)
    print(f"  CV R² = {cv_summary['cv_r2_mean']:.4f} ± {cv_summary['cv_r2_std']:.4f}")

    save_hyperparameters(output_dir, {
        "degree": POLY_DEGREE,
        "n_pca_components": N_PCA_COMPONENTS,
        "ridge_alpha": best_alpha,
    })

    save_model_results(
        output_dir,
        MODEL_NAME,
        df[DATE_COLUMN],
        y,
        y_all_pred,
        df[DATE_COLUMN].iloc[split_idx],
        df[DATE_COLUMN].iloc[split_idx:],
        y_test,
        y_test_pred,
        train_metrics,
        test_metrics,
    )
    save_residual_diagnostics(output_dir, MODEL_NAME, y_test_pred, y_test, temp_test)

    print(f"\nSaved outputs to: {output_dir}")
    print("Next step: run 06_random_forest.py")


if __name__ == "__main__":
    main()

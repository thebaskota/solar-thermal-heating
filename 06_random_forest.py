"""
Step 6 — Random forest regression on PCA features.

Scaler + PCA + random forest are refit inside a sklearn Pipeline on raw
predictors. Tuned via time-series CV on the training set.
Run after 05_polynomial_regression.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model_outputs import (
    CV_FOLDS,
    DATE_COLUMN,
    N_PCA_COMPONENTS,
    PREDICTOR_COLUMNS,
    PREPROCESSED_FILE,
    RANDOM_STATE,
    RF_PARAM_GRID,
    TARGET_COLUMN,
    TEMP_COLUMN,
    build_pca_pipeline,
    calculate_metrics,
    chronological_split,
    init_plot_style,
    print_model_performance,
    save_cv_summary,
    save_hyperparameters,
    save_model_results,
    save_residual_diagnostics,
    time_series_cv_summary,
)

MODEL_NAME = "Random Forest Regression"
OUTPUT_DIR = Path("outputs") / "06_random_forest"


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    print(f"Loading: {script_dir / PREPROCESSED_FILE}")
    df = pd.read_csv(script_dir / PREPROCESSED_FILE).sort_values(DATE_COLUMN).reset_index(drop=True)

    X = df[PREDICTOR_COLUMNS]
    y = df[TARGET_COLUMN]

    split_idx = chronological_split(df)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    temp_test = df.iloc[split_idx:][TEMP_COLUMN].values

    pc_names = [f"PC{i}" for i in range(1, N_PCA_COMPONENTS + 1)]

    print(f"\nModel: {MODEL_NAME} (training-set CV tuning)")
    print(f"Predictors: {PREDICTOR_COLUMNS} → PCA (k={N_PCA_COMPONENTS})")

    pipeline = build_pca_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    )
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    search = GridSearchCV(
        pipeline,
        RF_PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    best = search.best_params_
    print("\n--- Hyperparameter search (training CV) ---")
    print(f"Best CV RMSE: {-search.best_score_:.2f} kWh/day")
    print(
        f"Selected: max_depth={best['regressor__max_depth']}, "
        f"min_samples_leaf={best['regressor__min_samples_leaf']}, "
        f"n_estimators={best['regressor__n_estimators']}"
    )

    pd.DataFrame(search.cv_results_)[[
        "param_regressor__max_depth",
        "param_regressor__min_samples_leaf",
        "param_regressor__n_estimators",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]].sort_values("rank_test_score").to_csv(
        output_dir / "rf_hyperparameter_search.csv", index=False,
    )

    save_hyperparameters(output_dir, {
        "max_depth": best["regressor__max_depth"],
        "min_samples_leaf": best["regressor__min_samples_leaf"],
        "n_estimators": best["regressor__n_estimators"],
        "n_pca_components": N_PCA_COMPONENTS,
    })

    model = search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X)

    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n--- Model performance ---")
    print_model_performance(train_metrics, test_metrics)

    cv_summary = time_series_cv_summary(model, X_train, y_train, tscv)
    save_cv_summary(output_dir, cv_summary)

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

    pd.DataFrame({
        "feature": pc_names,
        "importance": model.named_steps["regressor"].feature_importances_,
    }).to_csv(output_dir / "feature_importance.csv", index=False)

    print(f"\nSaved outputs to: {output_dir}")
    print("Next step: run 07_hdd_model.py")


if __name__ == "__main__":
    main()

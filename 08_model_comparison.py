"""
Step 8 — Compare six candidate models (lean aggregation).

Aggregates holdout results from steps 4–7. Read-only — no re-tuning.
Run after 07_hdd_model.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_outputs import (
    CV_SUMMARY_FILE,
    DATE_COLUMN,
    EVALUATION_FILE,
    TEST_PREDICTIONS_FILE,
    init_plot_style,
    load_model_metrics,
)

OUTPUT_DIR = Path("outputs") / "08_model_comparison"
HDD_ROOT = Path("outputs") / "07_hdd_model"
LINEAR_ROOT = Path("outputs") / "04_linear_regression"

MODEL_SPECS = [
    ("Temperature-Only Linear", LINEAR_ROOT / "temperature_only"),
    ("HDD Threshold", HDD_ROOT / "hdd_threshold"),
    ("HDD + Solar Yield", HDD_ROOT / "hdd_solar"),
    ("PCA + Linear Regression", LINEAR_ROOT / "pca_linear"),
    ("Polynomial Regression (degree=2)", Path("outputs") / "05_polynomial_regression"),
    ("Random Forest Regression", Path("outputs") / "06_random_forest"),
]


def load_cv_summary(model_dir):
    path = model_dir / CV_SUMMARY_FILE
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0]
    return {
        "CV_R²_Mean": row.get("cv_r2_mean", np.nan),
        "CV_R²_Std": row.get("cv_r2_std", np.nan),
        "CV_RMSE_Mean": row.get("cv_rmse_mean", np.nan),
        "CV_RMSE_Std": row.get("cv_rmse_std", np.nan),
    }


def plot_holdout_r2(comparison_df, output_path):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_df, x="Model", y="Test_R²", color="steelblue")
    plt.title("Final Holdout R² Comparison (70 weeks)")
    plt.xlabel("Model")
    plt.ylabel("Test R²")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_holdout_predictions(script_dir):
    merged = None
    for model_name, model_dir in MODEL_SPECS:
        pred_path = script_dir / model_dir / TEST_PREDICTIONS_FILE
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        col_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        frame = pred_df[[DATE_COLUMN, "actual", "predicted"]].rename(
            columns={"predicted": col_name},
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[[DATE_COLUMN, col_name]], on=DATE_COLUMN, how="outer")
    return merged


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    rows = []
    for model_name, model_dir in MODEL_SPECS:
        eval_path = script_dir / model_dir / EVALUATION_FILE
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing evaluation file: {eval_path}\nRun all prior steps first.")
        entry = load_model_metrics(model_name, eval_path)
        entry.update(load_cv_summary(script_dir / model_dir))
        rows.append(entry)

    comparison_df = pd.DataFrame(rows)
    comparison_df["R2_gap"] = comparison_df["Train_R²"] - comparison_df["Test_R²"]

    print("--- Model comparison (held-out test set) ---")
    display_cols = ["Model", "Train_R²", "Test_R²", "Test_RMSE", "Test_MAE", "Dev_BIC", "CV_R²_Mean", "CV_R²_Std"]
    print(comparison_df[display_cols].round(4).to_string(index=False))

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    plot_holdout_r2(comparison_df, output_dir / "model_r2_comparison.png")

    holdout = build_holdout_predictions(script_dir)
    if holdout is not None:
        holdout.to_csv(output_dir / "final_holdout_predictions.csv", index=False)

    best_row = comparison_df.sort_values(["Test_R²", "Test_RMSE"], ascending=[False, True]).iloc[0]
    print(f"\nHighest test R²: {best_row['Model']} (R²={best_row['Test_R²']:.4f})")
    print("Note: Dev_BIC is computed on the development sample only (classical in-sample BIC).")
    print("Note: Dev_BIC is not defined for Random Forest.")
    print(f"\nSaved outputs to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

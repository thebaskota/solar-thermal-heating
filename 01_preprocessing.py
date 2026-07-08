"""
Step 1 — Data preprocessing for the solar-thermal heating dataset.

Loads Heating-data.csv, validates columns, checks missing values and duplicate
dates, sorts chronologically, and saves the cleaned dataset.
"""

from pathlib import Path

import pandas as pd

from model_outputs import (
    DATE_COLUMN,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)

DATA_FILE = "Heating-data.csv"
OUTPUT_DIR = Path("outputs") / "01_preprocessing"
OUTPUT_FILE = OUTPUT_DIR / "preprocessed_data.csv"
VALIDATION_REPORT = OUTPUT_DIR / "validation_report.txt"

REQUIRED_COLUMNS = [DATE_COLUMN, TARGET_COLUMN, *PREDICTOR_COLUMNS]


def validate_dataframe(df):
    """Run input validation checks; return list of report lines."""
    lines = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    lines.append(f"Required columns present: {len(REQUIRED_COLUMNS)}")

    model_cols = [TARGET_COLUMN, *PREDICTOR_COLUMNS]
    missing_counts = df[model_cols].isna().sum()
    lines.append("Missing values per column:")
    for col in model_cols:
        lines.append(f"  {col}: {missing_counts[col]}")
    if missing_counts.sum() > 0:
        raise ValueError("Missing values found in model columns — resolve before continuing.")

    dup_dates = df[DATE_COLUMN].duplicated().sum()
    lines.append(f"Duplicate dates: {dup_dates}")
    if dup_dates > 0:
        raise ValueError(f"Found {dup_dates} duplicate dates — resolve before continuing.")

    lines.append(f"Observations: {len(df)}")
    return lines


def main():
    script_dir = Path(__file__).parent
    data_path = script_dir / DATA_FILE
    output_dir = script_dir / OUTPUT_DIR
    output_file = script_dir / OUTPUT_FILE
    report_file = script_dir / VALIDATION_REPORT

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")

    report_lines = validate_dataframe(df)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    diffs = df[DATE_COLUMN].diff().dropna()
    gap_count = int((diffs != pd.Timedelta(days=7)).sum())
    report_lines.append(f"Non-weekly gaps: {gap_count}")
    if gap_count > 0:
        raise ValueError("Date spacing is not exactly weekly.")

    report_lines.append(f"Date range: {df[DATE_COLUMN].iloc[0].date()} to {df[DATE_COLUMN].iloc[-1].date()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    report_file.write_text("\n".join(report_lines) + "\n")

    for line in report_lines:
        print(line)
    print(f"\nSaved: {output_file}")
    print(f"Validation report: {report_file}")
    print("Next step: run 02_exploratory_analysis.py")


if __name__ == "__main__":
    main()

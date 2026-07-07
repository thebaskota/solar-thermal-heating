"""
Step 2 — Exploratory data analysis.

Describes the dataset and motivates PCA:
  - Gas drivers: temperature plateau, seasonality, correlations
  - Multicollinearity (VIF) among predictors → PCA in step 03

Run after 01_preprocessing.py.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_outputs import (
    DATE_COLUMN,
    PREDICTOR_COLUMNS,
    SOLAR_COLUMN,
    TARGET_COLUMN,
    TEMP_COLUMN,
    chronological_split,
    init_plot_style,
)

SHORT_LABELS = [c.split("[")[0].strip() for c in PREDICTOR_COLUMNS]
ALL_COLUMNS = PREDICTOR_COLUMNS + [TARGET_COLUMN]

INPUT_FILE = Path("outputs") / "01_preprocessing" / "preprocessed_data.csv"
OUTPUT_DIR = Path("outputs") / "02_exploratory_analysis"

BALANCE_TEMP_C = 15.5  # observed plateau in the data; HDD formalised in step 07


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_vif(X_df):
    rows = []
    cols = list(X_df.columns)
    for col in cols:
        others = [c for c in cols if c != col]
        y = X_df[col].values
        pred = LinearRegression().fit(X_df[others].values, y).predict(X_df[others].values)
        r2 = r2_score(y, pred)
        rows.append({"variable": col, "VIF": np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows)


def plateau_stats(df):
    above = df[df[TEMP_COLUMN] >= BALANCE_TEMP_C]
    below = df[df[TEMP_COLUMN] < BALANCE_TEMP_C]
    return {
        "balance_temp_c": BALANCE_TEMP_C,
        "weeks_above": len(above),
        "weeks_below": len(below),
        "mean_gas_above": above[TARGET_COLUMN].mean(),
        "mean_gas_below": below[TARGET_COLUMN].mean(),
        "median_gas_above": above[TARGET_COLUMN].median(),
        "pct_gas_below_100_above": (above[TARGET_COLUMN] < 100).mean() * 100,
    }


def plot_gas_vs_temperature(df, output_dir):
    temp = df[TEMP_COLUMN].values
    gas = df[TARGET_COLUMN].values
    plateau = plateau_stats(df)

    slope, intercept, r, _, _ = stats.linregress(temp, gas)
    x_line = np.linspace(temp.min(), temp.max(), 200)

    plt.figure(figsize=(9, 6))
    plt.scatter(temp, gas, alpha=0.45, s=22, c="steelblue", edgecolors="none")
    plt.plot(x_line, slope * x_line + intercept, "r-", lw=2, label=f"Linear fit (r={r:.2f})")
    plt.axvline(BALANCE_TEMP_C, color="green", ls="--", lw=1.5,
                label=f"Observed breakpoint (~{BALANCE_TEMP_C}°C)")
    plt.axhline(plateau["median_gas_above"], color="gray", ls=":", lw=1.2,
                label=f"Summer median ({plateau['median_gas_above']:.0f} kWh/day)")
    plt.annotate(
        "Heating zone\n(gas rises with cold)",
        xy=(4, 450), fontsize=9, ha="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    plt.annotate(
        f"Baseline zone\n({plateau['pct_gas_below_100_above']:.0f}% below 100 kWh/day)",
        xy=(20, 85), fontsize=9, ha="center",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.85),
    )
    plt.xlabel("Outdoor Temperature [°C]")
    plt.ylabel("Gas Consumption [kWh/day]")
    plt.title("Gas vs Outdoor Temperature — piecewise pattern (summer plateau)")
    plt.legend(fontsize=8)
    save_fig(output_dir / "gas_vs_temperature.png")


def plot_correlation_heatmap(df, output_dir):
    corr = df[ALL_COLUMNS].corr()
    corr.reset_index(names="variable").to_csv(output_dir / "correlation_matrix.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        xticklabels=SHORT_LABELS + ["Gas"], yticklabels=SHORT_LABELS + ["Gas"],
        square=True, linewidths=0.5,
    )
    plt.title("Correlation Matrix (predictors and gas consumption)")
    save_fig(output_dir / "correlation_heatmap.png")


def plot_seasonal_drivers_timeseries(df, output_dir):
    """Gas, outdoor temperature, and solar yield over time (reference-style Figure 2)."""
    split_idx = chronological_split(df)
    holdout_start = pd.to_datetime(df[DATE_COLUMN].iloc[split_idx])
    dates = pd.to_datetime(df[DATE_COLUMN])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    series = [
        (TARGET_COLUMN, "Gas Consumption [kWh/day]", "steelblue"),
        (TEMP_COLUMN, "Outdoor Temperature [°C]", "coral"),
        (SOLAR_COLUMN, "Solar Yield [kWh/day]", "goldenrod"),
    ]
    for ax, (col, ylabel, color) in zip(axes, series):
        ax.plot(dates, df[col], color=color, linewidth=1.2)
        ax.axvline(holdout_start, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_ylabel(ylabel)
    axes[0].set_title("Weekly time series — seasonal anti-phase relationship")
    axes[-1].set_xlabel("Date")
    axes[0].legend(["Holdout start (2025-02-16)"], loc="upper right", fontsize=8)
    save_fig(output_dir / "seasonal_drivers_timeseries.png")


def plot_weekly_gas_timeseries(df, output_dir):
    split_idx = chronological_split(df)
    holdout_start = pd.to_datetime(df[DATE_COLUMN].iloc[split_idx])
    dates = pd.to_datetime(df[DATE_COLUMN])

    plt.figure(figsize=(12, 5))
    plt.plot(dates, df[TARGET_COLUMN], color="steelblue", linewidth=1.2, label="Weekly gas consumption")
    plt.axvline(holdout_start, color="crimson", linestyle="--", linewidth=1.5, label="Holdout start (2025-02-16)")
    plt.xlabel("Date")
    plt.ylabel("Gas Consumption [kWh/day]")
    plt.title("Weekly Gas Consumption Over Time")
    plt.legend(loc="upper right")
    save_fig(output_dir / "weekly_gas_timeseries.png")


def plot_seasonal_gas(df, output_dir):
    months = pd.to_datetime(df[DATE_COLUMN]).dt.month
    monthly = df.groupby(months)[TARGET_COLUMN].agg(["mean", "std"]).reset_index(names=["month", "mean", "std"])
    monthly.to_csv(output_dir / "monthly_gas.csv", index=False)

    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 5))
    plt.bar(monthly["month"], monthly["mean"], color="steelblue", edgecolor="black", alpha=0.85)
    plt.errorbar(monthly["month"], monthly["mean"], yerr=monthly["std"], fmt="none", color="black", capsize=3)
    plt.xticks(range(1, 13), names)
    plt.xlabel("Month")
    plt.ylabel("Mean Gas Consumption [kWh/day]")
    plt.title("Seasonal Gas Consumption (mean ± 1 std)")
    save_fig(output_dir / "seasonal_gas.png")


def save_eda_summary(df, vif_df, output_dir):
    corr = df[ALL_COLUMNS].corr()
    gas = df[TARGET_COLUMN]
    months = pd.to_datetime(df[DATE_COLUMN]).dt.month
    monthly = df.groupby(months)[TARGET_COLUMN].mean()
    plateau = plateau_stats(df)

    rows = [
        {"metric": "n_weeks", "value": len(df)},
        {"metric": "date_start", "value": df[DATE_COLUMN].iloc[0]},
        {"metric": "date_end", "value": df[DATE_COLUMN].iloc[-1]},
        {"metric": "gas_mean_kwh", "value": gas.mean()},
        {"metric": "gas_std_kwh", "value": gas.std()},
        {"metric": "gas_min_kwh", "value": gas.min()},
        {"metric": "gas_max_kwh", "value": gas.max()},
        {"metric": "winter_mean_gas_dec_feb", "value": monthly[[12, 1, 2]].mean()},
        {"metric": "summer_mean_gas_jun_aug", "value": monthly[[6, 7, 8]].mean()},
        {"metric": "r_temp_gas", "value": corr.loc[TEMP_COLUMN, TARGET_COLUMN]},
        {"metric": "r_solar_gas", "value": corr.loc[SOLAR_COLUMN, TARGET_COLUMN]},
        {"metric": "r_sunshine_gas", "value": corr.loc[PREDICTOR_COLUMNS[0], TARGET_COLUMN]},
        {"metric": "r_solar_pump_gas", "value": corr.loc["Solar pump [h/day]", TARGET_COLUMN]},
        {"metric": "r_valve_gas", "value": corr.loc["Valve [h/day]", TARGET_COLUMN]},
        {"metric": "r_solar_yield_pump", "value": corr.loc[SOLAR_COLUMN, "Solar pump [h/day]"]},
        {"metric": "balance_temp_c", "value": plateau["balance_temp_c"]},
        {"metric": "mean_gas_above_balance", "value": plateau["mean_gas_above"]},
        {"metric": "mean_gas_below_balance", "value": plateau["mean_gas_below"]},
        {"metric": "pct_gas_below_100_above_balance", "value": plateau["pct_gas_below_100_above"]},
    ]
    for _, row in vif_df.iterrows():
        rows.append({"metric": f"vif_{row['variable'].split('[')[0].strip()}", "value": row["VIF"]})

    pd.DataFrame(rows).to_csv(output_dir / "eda_summary.csv", index=False)
    vif_df.assign(short_label=vif_df["variable"].str.split("[").str[0].str.strip()).to_csv(
        output_dir / "vif_multicollinearity.csv", index=False
    )


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    print(f"Loading: {script_dir / INPUT_FILE}")
    df = pd.read_csv(script_dir / INPUT_FILE).sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"Dataset: {len(df)} weeks  ({df[DATE_COLUMN].iloc[0]} to {df[DATE_COLUMN].iloc[-1]})")

    vif_df = compute_vif(df[PREDICTOR_COLUMNS])
    plot_gas_vs_temperature(df, output_dir)
    plot_seasonal_drivers_timeseries(df, output_dir)
    plot_weekly_gas_timeseries(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_seasonal_gas(df, output_dir)
    save_eda_summary(df, vif_df, output_dir)

    print("\n--- VIF (multicollinearity) ---")
    for _, row in vif_df.iterrows():
        print(f"  {row['variable'].split('[')[0].strip():22} VIF = {row['VIF']:.1f}")

    print("\n--- Key correlations with gas ---")
    corr = df[ALL_COLUMNS].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    for col, r in corr.items():
        print(f"  {col.split('[')[0].strip():22} r = {r:+.3f}")

    print(f"\nOutputs ({len(list(output_dir.iterdir()))} files):")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")
    print("\nPredictors are redundant (high VIF) → PCA in step 03.")
    print("Next: python 03_pca_analysis.py")


if __name__ == "__main__":
    main()

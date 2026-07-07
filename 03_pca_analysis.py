"""
Step 3 — PCA exploration.

Follows step 02 (EDA showed overlapping predictors → PCA needed). Fits PCA on the
training set: variance table, scree plot, loadings, PC scatter.
Set N_PCA_COMPONENTS in model_outputs.py from the scree plot, then run step 04.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model_outputs import (
    DATE_COLUMN,
    PREDICTOR_COLUMNS,
    TEST_SIZE,
    chronological_split,
    init_plot_style,
)

SHORT_LABELS = [c.split("[")[0].strip() for c in PREDICTOR_COLUMNS]

INPUT_FILE = Path("outputs") / "01_preprocessing" / "preprocessed_data.csv"
OUTPUT_DIR = Path("outputs") / "03_pca_analysis"


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    init_plot_style()

    print(f"Loading: {script_dir / INPUT_FILE}")
    df = pd.read_csv(script_dir / INPUT_FILE).sort_values(DATE_COLUMN).reset_index(drop=True)

    split_idx = chronological_split(df, TEST_SIZE)
    train_df = df.iloc[:split_idx]
    X = train_df[PREDICTOR_COLUMNS]

    print(f"Training set: {len(train_df)}  "
          f"({train_df[DATE_COLUMN].iloc[0]} to {train_df[DATE_COLUMN].iloc[-1]})")
    print(f"Test set held out: {len(df) - len(train_df)} observations")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)
    ev = pca.explained_variance_ratio_
    cv = np.cumsum(ev)
    scores = pca.transform(X_scaled)
    n_pcs = len(ev)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PREDICTOR_COLUMNS,
        columns=[f"PC{i}" for i in range(1, n_pcs + 1)],
    )

    print("\n--- Explained variance (training set) ---")
    for i in range(n_pcs):
        print(f"  PC{i + 1}: {ev[i] * 100:.2f}%  |  cumulative: {cv[i] * 100:.2f}%")

    pd.DataFrame({
        "component": [f"PC{i}" for i in range(1, n_pcs + 1)],
        "explained_variance": ev,
        "cumulative_variance": cv,
    }).to_csv(output_dir / "pca_variance.csv", index=False)

    loadings.reset_index(names="Variable").to_csv(output_dir / "pca_loadings.csv", index=False)

    pcs = np.arange(1, n_pcs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, ev, "o-", color="steelblue")
    plt.title("Scree Plot (Training Set)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(pcs)
    save_fig(output_dir / "scree_plot.png")

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, cv, "o-", color="darkorange")
    plt.title("Cumulative Explained Variance (Training Set)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.ylim(0, 1.05)
    save_fig(output_dir / "cumulative_variance_plot.png")

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, yticklabels=SHORT_LABELS, cbar_kws={"label": "Loading"})
    plt.title("PCA Loadings Heatmap (Training Set)")
    save_fig(output_dir / "pca_loadings_heatmap.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, edgecolors="k", linewidths=0.3)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2 (Training Set)")
    save_fig(output_dir / "pca_scatter_plot.png")

    print(f"\nSaved outputs to: {output_dir}")
    print("Next step: run 04_linear_regression.py")


if __name__ == "__main__":
    main()

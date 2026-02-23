"""
Exploratory data analysis for fingerprint-score relationships.

Generates plots and statistics to understand the data before ML training:
  - Docking score distribution
  - Fingerprint bit variance and density
  - PCA projections colored by docking score
  - Top correlated bits per fingerprint type
  - Inter-fingerprint similarity
  - Mutual information feature importance preview
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC/headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from fingerprints import FingerprintSet

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def run_full_eda(
    fp_sets: list,
    scores: np.ndarray,
    names: list,
    output_dir: str = "plots",
) -> pd.DataFrame:
    """
    Run the full EDA pipeline. Saves plots to output_dir.
    Returns a summary DataFrame of per-fingerprint statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting EDA on %d molecules, %d fingerprint types", len(scores), len(fp_sets))

    # 1. Score distribution
    plot_score_distribution(scores, output_dir)

    # 2. Per-fingerprint analysis
    summary_rows = []
    for fp_set in fp_sets:
        row = analyze_fingerprint(fp_set, scores, output_dir)
        summary_rows.append(row)

    # 3. Inter-fingerprint correlation
    plot_inter_fp_correlation(fp_sets, output_dir)

    # 4. Combined PCA comparison
    plot_pca_comparison(fp_sets, scores, output_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "eda_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("EDA summary saved to %s", summary_path)

    return summary_df


def plot_score_distribution(scores: np.ndarray, output_dir: str) -> None:
    """Histogram and box plot of docking scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(scores, bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
    axes[0].set_xlabel("Docking Score (kcal/mol)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Docking Score Distribution")
    axes[0].axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.2f}")
    axes[0].axvline(np.median(scores), color="orange", linestyle="--", label=f"Median: {np.median(scores):.2f}")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(scores, vert=True)
    axes[1].set_ylabel("Docking Score (kcal/mol)")
    axes[1].set_title("Score Box Plot")

    plt.tight_layout()
    path = os.path.join(output_dir, "score_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved score distribution plot to %s", path)


def analyze_fingerprint(
    fp_set: FingerprintSet,
    scores: np.ndarray,
    output_dir: str,
) -> dict:
    """
    Analyze a single fingerprint type:
      - Bit density and variance
      - Top correlated bits (Pearson and Spearman)
      - Mutual information
      - PCA colored by score
    Returns a summary dict.
    """
    X = fp_set.matrix.astype(np.float64)
    name = fp_set.name

    # Bit statistics
    bit_sums = X.sum(axis=0)
    bit_variance = X.var(axis=0)
    nonzero_bits = np.count_nonzero(bit_sums)
    mean_density = X.mean()

    # Remove zero-variance bits for correlation analysis
    var_mask = bit_variance > 0
    X_var = X[:, var_mask]
    n_informative = X_var.shape[1]

    # Pearson correlation of each bit with docking score
    pearson_corrs = np.zeros(n_informative)
    spearman_corrs = np.zeros(n_informative)
    for j in range(n_informative):
        pearson_corrs[j], _ = stats.pearsonr(X_var[:, j], scores)
        spearman_corrs[j], _ = stats.spearmanr(X_var[:, j], scores)

    # Top 20 most correlated bits (by absolute Pearson)
    top_idx = np.argsort(np.abs(pearson_corrs))[::-1][:20]
    top_pearson = pearson_corrs[top_idx]
    top_spearman = spearman_corrs[top_idx]

    # Mutual information (subsample if large)
    mi_n = min(n_informative, 500)
    mi_idx = np.argsort(np.abs(pearson_corrs))[::-1][:mi_n]
    mi_scores = mutual_info_regression(
        X_var[:, mi_idx], scores, random_state=42, n_neighbors=5
    )

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EDA: {name}", fontsize=14)

    # Bit density histogram
    axes[0, 0].hist(bit_sums, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Bit frequency (molecules with bit ON)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Bit Frequency Distribution")

    # Pearson correlation distribution
    axes[0, 1].hist(pearson_corrs, bins=50, edgecolor="black", alpha=0.7, color="#FF9800")
    axes[0, 1].set_xlabel("Pearson r (bit vs docking score)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Bit-Score Correlation Distribution")
    axes[0, 1].axvline(0, color="black", linestyle="-", alpha=0.3)

    # Top correlated bits bar chart
    bit_labels = [f"bit_{top_idx[k]}" for k in range(len(top_idx))]
    colors = ["#4CAF50" if v < 0 else "#F44336" for v in top_pearson]
    axes[1, 0].barh(range(len(top_pearson)), top_pearson, color=colors)
    axes[1, 0].set_yticks(range(len(top_pearson)))
    axes[1, 0].set_yticklabels(bit_labels, fontsize=7)
    axes[1, 0].set_xlabel("Pearson r")
    axes[1, 0].set_title("Top 20 Correlated Bits")
    axes[1, 0].axvline(0, color="black", linestyle="-", alpha=0.3)

    # Mutual information bar chart (top 20)
    mi_top_idx = np.argsort(mi_scores)[::-1][:20]
    mi_labels = [f"bit_{mi_idx[k]}" for k in mi_top_idx]
    axes[1, 1].barh(range(len(mi_top_idx)), mi_scores[mi_top_idx], color="#9C27B0")
    axes[1, 1].set_yticks(range(len(mi_top_idx)))
    axes[1, 1].set_yticklabels(mi_labels, fontsize=7)
    axes[1, 1].set_xlabel("Mutual Information")
    axes[1, 1].set_title("Top 20 Bits by Mutual Information")

    plt.tight_layout()
    path = os.path.join(output_dir, f"eda_{name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved %s EDA plot to %s", name, path)

    return {
        "fingerprint": name,
        "total_bits": fp_set.n_bits,
        "nonzero_bits": nonzero_bits,
        "informative_bits": n_informative,
        "mean_density": round(mean_density, 4),
        "max_abs_pearson": round(float(np.max(np.abs(pearson_corrs))), 4),
        "max_abs_spearman": round(float(np.max(np.abs(spearman_corrs))), 4),
        "mean_abs_pearson": round(float(np.mean(np.abs(pearson_corrs))), 4),
        "max_mutual_info": round(float(np.max(mi_scores)), 4),
    }


def plot_pca_comparison(
    fp_sets: list,
    scores: np.ndarray,
    output_dir: str,
) -> None:
    """PCA of each fingerprint type, colored by docking score, in a grid."""
    n = len(fp_sets)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, fp_set in enumerate(fp_sets):
        X = fp_set.matrix.astype(np.float64)
        # Remove zero-variance columns for PCA
        var_mask = X.var(axis=0) > 0
        X_filtered = X[:, var_mask]

        if X_filtered.shape[1] < 2:
            axes[i].text(0.5, 0.5, "Too few features", ha="center", va="center")
            axes[i].set_title(fp_set.name)
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        scatter = axes[i].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=scores,
            cmap="RdYlGn_r",
            s=8,
            alpha=0.6,
        )
        axes[i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[i].set_title(fp_set.name)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.colorbar(scatter, ax=axes[:n], label="Docking Score (kcal/mol)", shrink=0.8)
    fig.suptitle("PCA of Fingerprints Colored by Docking Score", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "pca_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved PCA comparison to %s", path)


def plot_inter_fp_correlation(fp_sets: list, output_dir: str) -> None:
    """
    Heatmap showing pairwise similarity between fingerprint types.
    Uses mean Tanimoto-like overlap on the bit vectors.
    """
    n = len(fp_sets)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Compute mean Jaccard similarity across molecules
            Xi = fp_sets[i].matrix.astype(bool)
            Xj = fp_sets[j].matrix

            # For different-sized FPs, use PCA-reduced correlation
            if Xi.shape[1] != Xj.shape[1]:
                # Use mean bit density correlation as proxy
                di = Xi.mean(axis=1)
                dj = Xj.mean(axis=1)
                sim_matrix[i, j], _ = stats.pearsonr(di, dj)
            else:
                # Bit-wise agreement rate
                agreement = (Xi == Xj.astype(bool)).mean()
                sim_matrix[i, j] = agreement

    names = [fp.name for fp in fp_sets]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_title("Inter-Fingerprint Similarity")
    plt.tight_layout()
    path = os.path.join(output_dir, "inter_fp_similarity.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved inter-FP similarity heatmap to %s", path)

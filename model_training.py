"""
ML model training and comparison for docking score prediction.

Trains multiple regression models on each fingerprint type using
cross-validation, then compares performance across all combinations.
"""

import logging
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from fingerprints import FingerprintSet

logger = logging.getLogger(__name__)

# Suppress convergence warnings from MLPRegressor during CV
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def get_models() -> dict:
    """
    Return a dict of model_name -> sklearn estimator.
    These are tuned for molecular fingerprint regression tasks.
    """
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for a single fold."""
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(y_true) > 2:
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def cross_validate_model(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    scale: bool = True,
) -> dict:
    """
    Run k-fold cross-validation for a single model.
    Returns dict with mean and std of each metric.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        from sklearn.base import clone

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        metrics = _compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

    # Aggregate across folds
    result = {"model": model_name}
    for key in fold_metrics[0]:
        values = [fm[key] for fm in fold_metrics]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)

    return result


def train_and_compare(
    fp_sets: list,
    scores: np.ndarray,
    n_folds: int = 5,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Train all models on all fingerprint types with cross-validation.
    Returns a DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = get_models()
    all_results = []

    total_combos = len(fp_sets) * len(models)
    combo_i = 0

    for fp_set in fp_sets:
        X = fp_set.matrix.astype(np.float64)

        # Remove zero-variance features
        var_mask = X.var(axis=0) > 0
        X_filtered = X[:, var_mask]

        logger.info(
            "Training on %s: %d samples x %d features (from %d bits)",
            fp_set.name,
            X_filtered.shape[0],
            X_filtered.shape[1],
            fp_set.n_bits,
        )

        for model_name, model in models.items():
            combo_i += 1
            t0 = time.time()

            # SVR is slow on high-dimensional data; skip if > 5000 features
            if model_name == "SVR" and X_filtered.shape[1] > 5000:
                logger.info(
                    "  [%d/%d] Skipping %s on %s (too many features for SVR)",
                    combo_i, total_combos, model_name, fp_set.name,
                )
                continue

            result = cross_validate_model(
                model_name, model, X_filtered, scores, n_folds=n_folds
            )
            result["fingerprint"] = fp_set.name
            result["n_features"] = X_filtered.shape[1]
            result["time_sec"] = round(time.time() - t0, 1)

            all_results.append(result)

            logger.info(
                "  [%d/%d] %s + %s: RMSE=%.3f +/- %.3f, R2=%.3f, Pearson=%.3f (%.1fs)",
                combo_i,
                total_combos,
                fp_set.name,
                model_name,
                result["rmse_mean"],
                result["rmse_std"],
                result["r2_mean"],
                result["pearson_r_mean"],
                result["time_sec"],
            )

    results_df = pd.DataFrame(all_results)

    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values("rmse_mean").reset_index(drop=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved model comparison to %s", csv_path)

    # Generate comparison plots
    plot_model_comparison(results_df, output_dir)
    plot_heatmaps(results_df, output_dir)
    plot_scatter_predictions(fp_sets, scores, models, output_dir)

    return results_df


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """Bar charts comparing models across fingerprints."""
    metrics = [
        ("rmse_mean", "RMSE (lower is better)", True),
        ("r2_mean", "R² (higher is better)", False),
        ("pearson_r_mean", "Pearson r (higher is better)", False),
        ("mae_mean", "MAE (lower is better)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (metric, label, ascending) in zip(axes, metrics):
        pivot = results_df.pivot_table(
            index="fingerprint", columns="model", values=metric, aggfunc="first"
        )
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Model Comparison Across Fingerprint Types", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison_bars.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved model comparison bar charts to %s", path)


def plot_heatmaps(results_df: pd.DataFrame, output_dir: str) -> None:
    """Heatmaps of RMSE and R² for fingerprint x model combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title, cmap in [
        (axes[0], "rmse_mean", "RMSE (lower is better)", "YlOrRd"),
        (axes[1], "r2_mean", "R² (higher is better)", "YlGn"),
    ]:
        pivot = results_df.pivot_table(
            index="fingerprint", columns="model", values=metric, aggfunc="first"
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(output_dir, "performance_heatmaps.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved performance heatmaps to %s", path)


def plot_scatter_predictions(
    fp_sets: list,
    scores: np.ndarray,
    models: dict,
    output_dir: str,
) -> None:
    """
    Scatter plot of predicted vs actual scores for the best model
    on each fingerprint type (using a single train/test split for visualization).
    """
    from sklearn.model_selection import train_test_split

    n_fps = len(fp_sets)
    fig, axes = plt.subplots(2, (n_fps + 1) // 2, figsize=(6 * ((n_fps + 1) // 2), 10))
    axes = axes.flatten()

    # Use Random Forest as the representative model for scatter plots
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=3,
        n_jobs=-1, random_state=42,
    )

    for i, fp_set in enumerate(fp_sets):
        X = fp_set.matrix.astype(np.float64)
        var_mask = X.var(axis=0) > 0
        X_filtered = X[:, var_mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, scores, test_size=0.2, random_state=42
        )

        from sklearn.base import clone

        model = clone(rf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = _compute_metrics(y_test, y_pred)

        axes[i].scatter(y_test, y_pred, alpha=0.5, s=15, edgecolors="none")
        lims = [
            min(y_test.min(), y_pred.min()) - 0.5,
            max(y_test.max(), y_pred.max()) + 0.5,
        ]
        axes[i].plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].set_xlabel("Actual Score")
        axes[i].set_ylabel("Predicted Score")
        axes[i].set_title(
            f"{fp_set.name}\nR²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}"
        )
        axes[i].legend(fontsize=8)

    for j in range(n_fps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Random Forest: Predicted vs Actual Docking Scores", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved prediction scatter plots to %s", path)


def print_best_models(results_df: pd.DataFrame, top_n: int = 5) -> None:
    """Print the top N model+fingerprint combinations."""
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MODEL + FINGERPRINT COMBINATIONS (by RMSE)")
    print(f"{'='*70}")
    print(
        f"  {'Rank':<5} {'Fingerprint':<20} {'Model':<20} "
        f"{'RMSE':<10} {'R²':<10} {'Pearson r':<10}"
    )
    print(f"  {'-'*5} {'-'*20} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for i, row in results_df.head(top_n).iterrows():
        print(
            f"  {i+1:<5} {row['fingerprint']:<20} {row['model']:<20} "
            f"{row['rmse_mean']:<10.4f} {row['r2_mean']:<10.4f} "
            f"{row['pearson_r_mean']:<10.4f}"
        )

    print(f"\n{'='*70}")
    print(f"WORST {top_n} COMBINATIONS (by RMSE)")
    print(f"{'='*70}")
    for i, row in results_df.tail(top_n).iterrows():
        print(
            f"  {row['fingerprint']:<20} {row['model']:<20} "
            f"RMSE={row['rmse_mean']:.4f}  R²={row['r2_mean']:.4f}"
        )
    print()

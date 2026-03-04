# Vina_ML_Pipeline — API Reference

This document describes every module, class, and function in the ML pipeline
for predicting docking scores from molecular fingerprints.

---

## Table of Contents

- [Entry Point](#entry-point)
  - [run_ml_pipeline.py](#run_ml_pipelinepy)
- [Modules](#modules)
  - [data_loader.py](#data_loaderpy)
  - [fingerprints.py](#fingerprintspy)
  - [eda.py](#edapy)
  - [model_training.py](#model_trainingpy)

---

## Entry Point

### `run_ml_pipeline.py`

Orchestrates the four-phase ML pipeline: data loading → fingerprint generation →
exploratory data analysis → model training and comparison.

**Run with:**
```bash
# Synthetic test data
python run_ml_pipeline.py --synthetic --n-molecules 1000

# Real data from Vina docking results
python run_ml_pipeline.py --scores results.csv --molecules ligands.sdf

# SMILES input
python run_ml_pipeline.py --scores results.csv --smiles ligands.smi

# Skip EDA, only train models
python run_ml_pipeline.py --synthetic --skip-eda

# Skip training, only run EDA
python run_ml_pipeline.py --synthetic --skip-training
```

#### CLI Arguments

| Argument | Description |
|---|---|
| `--scores` | Path to Vina results CSV (must have ligand and energy columns). |
| `--molecules` | Path to SDF file with ligand structures. |
| `--smiles` | Path to SMILES file (space-separated) or CSV with a `smiles` column. |
| `--synthetic` | Generate synthetic data for testing (no input files needed). |
| `--n-molecules` | Number of synthetic molecules (default: 500). |
| `--skip-eda` | Skip exploratory data analysis. |
| `--skip-training` | Skip model training (EDA only). |
| `--folds` | Number of cross-validation folds (default: 5). |
| `--output-dir`, `-o` | Output directory (default: `results/`). |
| `--plot-dir` | EDA plot directory (default: `<output-dir>/plots/`). |
| `--verbose`, `-v` | Enable debug logging. |
| `--seed` | Random seed (default: 42). |

#### Functions

| Function | Description |
|---|---|
| `setup_logging(verbose)` | Configures logging level. |
| `parse_args()` | Parses all CLI arguments. |
| `main()` | Runs the four-phase pipeline. Returns 0 on success, 1 on error. |

---

## Modules

### `data_loader.py`

Handles loading docking results and molecular structures from various file formats.
Also provides synthetic data generation for testing.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `load_vina_results` | `(csv_path) → DataFrame` | Loads a Vina results CSV. Auto-detects columns containing "ligand" and "energy"/"score". Drops rows with non-numeric scores. Returns a DataFrame with `ligand` and `docking_score` columns. |
| `load_molecules_from_sdf` | `(sdf_path) → dict[str, Mol]` | Loads molecules from an SDF file using RDKit's `SDMolSupplier`. Returns a dict mapping molecule name (`_Name` property) to RDKit `Mol` object. Removes hydrogens. |
| `load_molecules_from_smiles` | `(smiles_path) → dict[str, Mol]` | Loads from a whitespace-separated SMILES file (format: `SMILES NAME`). Skips lines starting with `#`. Logs warnings for unparseable SMILES. |
| `load_molecules_from_smiles_csv` | `(csv_path) → dict[str, Mol]` | Loads from a CSV with `smiles` and `name`/`ligand`/`id` columns. Auto-detects column names. |
| `merge_scores_and_molecules` | `(scores_df, mol_dict) → (mols, scores, names)` | Matches docking scores to molecules by ligand name. Strips `.pdbqt` extensions and `_out` suffixes for flexible matching. Returns aligned lists of Mol objects, score array, and name list. Raises `ValueError` if no matches found. |
| `generate_synthetic_dataset` | `(n_molecules=500, seed=42) → (mols, scores, names)` | Generates a test dataset from 30 drug-like SMILES templates. Assigns synthetic docking scores correlated with molecular properties (MW, LogP, TPSA, HBD, HBA, rotatable bonds) plus Gaussian noise. Scores are clamped to [-12, -1] kcal/mol. |

**Integration point:** The `load_vina_results()` function accepts CSV output from both
`run_docking.py` (large-scale campaign) and `run_optimize.py` (best_docking_scores.csv).

---

### `fingerprints.py`

Generates multiple molecular fingerprint types using RDKit's `rdFingerprintGenerator` API.
Each fingerprint type produces a numpy feature matrix suitable for ML training.

#### Classes

##### `FingerprintSet`
Dataclass holding a named fingerprint matrix.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Fingerprint identifier (e.g., `"Morgan_r2_2048"`). |
| `matrix` | `ndarray` | Binary feature matrix, shape `(n_molecules, n_bits)`, dtype `uint8`. |
| `n_bits` | `int` | Number of bits in the fingerprint. |
| `description` | `str` | Human-readable description. |

#### Functions

| Function | Signature | Fingerprint Type | Bits |
|---|---|---|---|
| `generate_morgan_fp` | `(mols, radius=2, n_bits=2048)` | Morgan/ECFP circular fingerprints. Radius 2 ≈ ECFP4, radius 3 ≈ ECFP6. Encodes circular atom environments. | 2048 |
| `generate_rdkit_fp` | `(mols, n_bits=2048)` | RDKit topological/path-based fingerprints. Enumerates all linear paths up to a maximum length. | 2048 |
| `generate_atompair_fp` | `(mols, n_bits=2048)` | Atom pair fingerprints. Encodes pairs of atoms and the shortest path distance between them. | 2048 |
| `generate_torsion_fp` | `(mols, n_bits=2048)` | Topological torsion fingerprints. Encodes sequences of four consecutively bonded non-hydrogen atoms. | 2048 |
| `generate_maccs_fp` | `(mols)` | MACCS structural keys. 166 predefined substructure patterns (plus bit 0). Fixed-size, not hashed. | 167 |
| `generate_all_fingerprints` | `(mols) → list[FingerprintSet]` | Generates all six fingerprint types (Morgan r=2, Morgan r=3, RDKit, AtomPair, TopTorsion, MACCS). Logs the number of non-zero bits per type. |

**Why multiple fingerprint types?** Different fingerprints encode different structural
information. Morgan captures local atom environments, RDKit captures paths, AtomPair
captures distance relationships, MACCS captures specific substructures. Comparing them
reveals which structural representation best predicts binding affinity for a given target.

---

### `eda.py`

Exploratory data analysis for understanding fingerprint-score relationships before
ML training. All plots use the `Agg` backend (headless-safe) and are saved as PNG.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `run_full_eda` | `(fp_sets, scores, names, output_dir) → DataFrame` | Runs the complete EDA pipeline: score distribution, per-fingerprint analysis, inter-fingerprint correlation, PCA comparison. Returns a summary DataFrame with per-fingerprint statistics. |
| `plot_score_distribution` | `(scores, output_dir)` | Histogram and box plot of docking scores with mean/median lines. |
| `analyze_fingerprint` | `(fp_set, scores, output_dir) → dict` | Per-fingerprint analysis. Computes: bit frequency distribution, Pearson and Spearman correlations of each bit vs docking score, mutual information of top bits. Generates a 4-panel plot (bit frequency histogram, correlation distribution, top 20 correlated bits, top 20 mutual information bits). Returns a summary dict with statistics. |
| `plot_pca_comparison` | `(fp_sets, scores, output_dir)` | PCA of each fingerprint type in a grid layout. Points colored by docking score (red = poor binder, green = good binder). Shows explained variance ratios. Removes zero-variance features before PCA. |
| `plot_inter_fp_correlation` | `(fp_sets, output_dir)` | Heatmap showing pairwise similarity between fingerprint types. Uses bit-wise agreement rate for same-sized fingerprints, or density correlation for different-sized ones. |

**EDA summary statistics per fingerprint:**

| Statistic | Description |
|---|---|
| `total_bits` | Total bits in the fingerprint. |
| `nonzero_bits` | Bits that are ON in at least one molecule. |
| `informative_bits` | Bits with non-zero variance (useful for ML). |
| `mean_density` | Average fraction of bits ON per molecule. |
| `max_abs_pearson` | Strongest single-bit Pearson correlation with score. |
| `max_abs_spearman` | Strongest single-bit Spearman correlation. |
| `mean_abs_pearson` | Average absolute Pearson correlation across all bits. |
| `max_mutual_info` | Highest mutual information between a bit and the score. |

---

### `model_training.py`

Trains multiple regression models on each fingerprint type using k-fold
cross-validation, then compares performance.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `get_models` | `() → dict[str, estimator]` | Returns a dict of seven pre-configured sklearn estimators (see table below). |
| `_compute_metrics` | `(y_true, y_pred) → dict` | Computes RMSE, MAE, R², Pearson r (with p-value), and Spearman r (with p-value) for a single fold. |
| `cross_validate_model` | `(model_name, model, X, y, n_folds=5, scale=True) → dict` | Runs k-fold CV for one model. Applies `StandardScaler` per fold (fit on train, transform test). Returns mean and std of each metric across folds. |
| `train_and_compare` | `(fp_sets, scores, n_folds=5, output_dir="results") → DataFrame` | Main comparison function. Trains all 7 models on all 6 fingerprint types (42 combinations). Removes zero-variance features. Skips SVR if >5000 features (too slow). Saves results CSV and generates comparison plots. Returns a DataFrame sorted by RMSE. |
| `plot_model_comparison` | `(results_df, output_dir)` | 2×2 grid of grouped bar charts: RMSE, R², Pearson r, MAE. Each bar group = one fingerprint type, bars = models. |
| `plot_heatmaps` | `(results_df, output_dir)` | Side-by-side heatmaps of RMSE and R² for all fingerprint × model combinations. |
| `plot_scatter_predictions` | `(fp_sets, scores, models, output_dir)` | Predicted vs actual scatter plots using Random Forest on each fingerprint type (80/20 train/test split). Shows R² and RMSE per panel. |
| `print_best_models` | `(results_df, top_n=5)` | Prints the top N and worst N model+fingerprint combinations to stdout. |

#### Models

| Model | Key Hyperparameters | Notes |
|---|---|---|
| **RandomForest** | 200 trees, min_samples_leaf=3 | Ensemble of decision trees. Handles high-dimensional sparse data well. |
| **GradientBoosting** | 200 trees, max_depth=5, lr=0.1, subsample=0.8 | Sequential boosting. Often best for tabular data. |
| **Ridge** | alpha=1.0 | L2-regularized linear regression. Fast baseline. |
| **Lasso** | alpha=0.01, max_iter=5000 | L1-regularized linear regression. Performs feature selection (drives coefficients to zero). |
| **SVR** | RBF kernel, C=10, epsilon=0.1 | Support vector regression. Skipped if >5000 features. |
| **KNN** | k=5, distance-weighted | k-nearest neighbors. Uses Tanimoto-like distance in fingerprint space. |
| **MLP** | (256, 128) hidden layers, ReLU, early stopping | Neural network. May capture non-linear relationships. |

#### Metrics

| Metric | Description | Better |
|---|---|---|
| **RMSE** | Root mean squared error in kcal/mol. | Lower |
| **MAE** | Mean absolute error in kcal/mol. | Lower |
| **R²** | Coefficient of determination. 1.0 = perfect, 0.0 = predicts the mean. | Higher |
| **Pearson r** | Linear correlation between predicted and actual scores. | Higher |
| **Spearman r** | Rank correlation. Measures whether the model preserves the ranking of compounds. | Higher |

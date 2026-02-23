#!/usr/bin/env python
"""
ML pipeline for predicting docking scores from molecular fingerprints.

Loads Vina docking results, generates multiple fingerprint types via RDKit,
runs exploratory data analysis, then trains and compares several ML models.

Usage:
    # With real data (Vina CSV + SDF of ligands):
    python run_ml_pipeline.py \
        --scores output/results.csv \
        --molecules ligands.sdf \
        --output-dir results/

    # With real data (Vina CSV + SMILES file):
    python run_ml_pipeline.py \
        --scores output/results.csv \
        --smiles ligands.smi \
        --output-dir results/

    # With synthetic data for testing:
    python run_ml_pipeline.py --synthetic --n-molecules 1000

    # Skip EDA, only train models:
    python run_ml_pipeline.py --synthetic --skip-eda

    # Custom CV folds:
    python run_ml_pipeline.py --synthetic --folds 10
"""

import argparse
import logging
import sys
import time

import numpy as np


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML pipeline: predict docking scores from molecular fingerprints",
    )

    # Data source (mutually exclusive groups)
    data_group = parser.add_argument_group("Data source")
    data_group.add_argument(
        "--scores",
        help="Path to Vina results CSV (columns: ligand, best_energy_kcal)",
    )
    data_group.add_argument(
        "--molecules",
        help="Path to SDF file containing ligand structures",
    )
    data_group.add_argument(
        "--smiles",
        help="Path to SMILES file (space-separated: SMILES NAME) or CSV with smiles column",
    )
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing (no input files needed)",
    )
    data_group.add_argument(
        "--n-molecules",
        type=int,
        default=500,
        help="Number of synthetic molecules to generate (default: 500)",
    )

    # Pipeline options
    pipeline_group = parser.add_argument_group("Pipeline options")
    pipeline_group.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis, go straight to model training",
    )
    pipeline_group.add_argument(
        "--skip-training",
        action="store_true",
        help="Only run EDA, skip model training",
    )
    pipeline_group.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory for results and plots (default: results/)",
    )
    output_group.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for EDA plots (default: <output-dir>/plots/)",
    )

    # General
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    np.random.seed(args.seed)
    t_start = time.time()

    plot_dir = args.plot_dir or f"{args.output_dir}/plots"

    # ---------------------------------------------------------------
    # Phase 1: Load data
    # ---------------------------------------------------------------
    logger.info("Phase 1: Loading data")

    if args.synthetic:
        from data_loader import generate_synthetic_dataset

        mols, scores, names = generate_synthetic_dataset(
            n_molecules=args.n_molecules, seed=args.seed
        )
    elif args.scores:
        from data_loader import (
            load_molecules_from_sdf,
            load_molecules_from_smiles,
            load_molecules_from_smiles_csv,
            load_vina_results,
            merge_scores_and_molecules,
        )

        scores_df = load_vina_results(args.scores)

        if args.molecules:
            mol_dict = load_molecules_from_sdf(args.molecules)
        elif args.smiles:
            if args.smiles.endswith(".csv"):
                mol_dict = load_molecules_from_smiles_csv(args.smiles)
            else:
                mol_dict = load_molecules_from_smiles(args.smiles)
        else:
            logger.error("Provide --molecules (SDF) or --smiles with --scores")
            return 1

        mols, scores, names = merge_scores_and_molecules(scores_df, mol_dict)
    else:
        logger.error("Provide --synthetic or --scores with molecule data")
        return 1

    logger.info(
        "Loaded %d molecules, scores range: [%.2f, %.2f] kcal/mol",
        len(mols),
        scores.min(),
        scores.max(),
    )

    # ---------------------------------------------------------------
    # Phase 2: Generate fingerprints
    # ---------------------------------------------------------------
    logger.info("Phase 2: Generating fingerprints")
    from fingerprints import generate_all_fingerprints

    fp_sets = generate_all_fingerprints(mols)

    # ---------------------------------------------------------------
    # Phase 3: Exploratory Data Analysis
    # ---------------------------------------------------------------
    if not args.skip_eda:
        logger.info("Phase 3: Exploratory Data Analysis")
        from eda import run_full_eda

        eda_summary = run_full_eda(fp_sets, scores, names, output_dir=plot_dir)
        print("\nEDA Summary:")
        print(eda_summary.to_string(index=False))
        print()
    else:
        logger.info("Phase 3: Skipped (--skip-eda)")

    # ---------------------------------------------------------------
    # Phase 4: Model Training & Comparison
    # ---------------------------------------------------------------
    if not args.skip_training:
        logger.info("Phase 4: Model Training & Comparison (%d-fold CV)", args.folds)
        from model_training import print_best_models, train_and_compare

        results_df = train_and_compare(
            fp_sets, scores, n_folds=args.folds, output_dir=args.output_dir
        )
        print_best_models(results_df, top_n=5)
    else:
        logger.info("Phase 4: Skipped (--skip-training)")

    t_end = time.time()
    logger.info("Pipeline complete in %.1fs", t_end - t_start)

    return 0


if __name__ == "__main__":
    sys.exit(main())

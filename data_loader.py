"""
Data loading: parse Vina docking results and load molecular structures.

Supports multiple input formats:
  - Vina results CSV + SDF file of ligands
  - Vina results CSV + directory of SDF/MOL2 files
  - SMILES file with docking scores
  - Synthetic data generation for testing
"""

import csv
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)


def load_vina_results(csv_path: str) -> pd.DataFrame:
    """
    Load docking results from the Vina MPI pipeline CSV output.
    Expected columns: rank, ligand, best_energy_kcal, ...
    Returns DataFrame with at least 'ligand' and 'docking_score' columns.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "ligand" in lower:
            col_map[col] = "ligand"
        elif "energy" in lower or "score" in lower:
            col_map[col] = "docking_score"

    df = df.rename(columns=col_map)

    if "ligand" not in df.columns or "docking_score" not in df.columns:
        raise ValueError(
            f"CSV must contain ligand and energy/score columns. Found: {list(df.columns)}"
        )

    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["docking_score"])
    if len(df) < n_before:
        logger.warning(
            "Dropped %d rows with non-numeric docking scores", n_before - len(df)
        )

    logger.info("Loaded %d docking results from %s", len(df), csv_path)
    return df


def load_molecules_from_sdf(sdf_path: str) -> dict:
    """
    Load molecules from an SDF file.
    Returns dict mapping molecule name -> RDKit Mol object.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols = {}
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(mols)}"
        mols[name] = mol

    logger.info("Loaded %d molecules from %s", len(mols), sdf_path)
    return mols


def load_molecules_from_smiles(smiles_path: str) -> dict:
    """
    Load molecules from a SMILES file (tab or space separated: SMILES NAME).
    Returns dict mapping name -> RDKit Mol object.
    """
    mols = {}
    with open(smiles_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                smiles, name = parts[0], f"mol_{line_num}"
            else:
                smiles, name = parts[0], parts[1]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning("Failed to parse SMILES on line %d: %s", line_num, smiles)
                continue
            mols[name] = mol

    logger.info("Loaded %d molecules from %s", len(mols), smiles_path)
    return mols


def load_molecules_from_smiles_csv(csv_path: str) -> dict:
    """
    Load molecules from a CSV with 'smiles' and 'name' (or 'ligand') columns.
    Returns dict mapping name -> RDKit Mol object.
    """
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "smiles" in lower or "smi" in lower:
            col_map[col] = "smiles"
        elif "name" in lower or "ligand" in lower or "id" in lower:
            col_map[col] = "name"
    df = df.rename(columns=col_map)

    if "smiles" not in df.columns:
        raise ValueError(f"CSV must have a SMILES column. Found: {list(df.columns)}")
    if "name" not in df.columns:
        df["name"] = [f"mol_{i}" for i in range(len(df))]

    mols = {}
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            mols[row["name"]] = mol
        else:
            logger.warning("Failed to parse SMILES: %s", row["smiles"])

    logger.info("Loaded %d molecules from %s", len(mols), csv_path)
    return mols


def merge_scores_and_molecules(
    scores_df: pd.DataFrame,
    mol_dict: dict,
) -> tuple:
    """
    Match docking scores to molecules by ligand name.
    Returns (mols_list, scores_array, names_list) for matched entries.
    """
    # Strip .pdbqt extension from ligand names for matching
    scores_df = scores_df.copy()
    scores_df["ligand_key"] = scores_df["ligand"].apply(
        lambda x: os.path.splitext(str(x))[0]
    )

    matched_mols = []
    matched_scores = []
    matched_names = []

    for _, row in scores_df.iterrows():
        key = row["ligand_key"]
        # Try exact match, then without _out suffix
        mol = mol_dict.get(key) or mol_dict.get(key.replace("_out", ""))
        if mol is not None:
            matched_mols.append(mol)
            matched_scores.append(row["docking_score"])
            matched_names.append(key)

    scores_array = np.array(matched_scores, dtype=np.float64)

    logger.info(
        "Matched %d/%d ligands with molecular structures",
        len(matched_mols),
        len(scores_df),
    )

    if len(matched_mols) == 0:
        raise ValueError("No ligands matched between scores and molecule files")

    return matched_mols, scores_array, matched_names


# ---------------------------------------------------------------------------
# Synthetic data generation for testing without real docking data
# ---------------------------------------------------------------------------

# Drug-like SMILES for synthetic dataset generation
_SAMPLE_SMILES = [
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    "OC(=O)c1ccccc1O",  # salicylic acid
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
    "CC(=O)NC1=CC=C(C=C1)O",  # acetaminophen
    "C1CCCCC1",  # cyclohexane
    "c1ccc(cc1)C(=O)O",  # benzoic acid
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # isoproterenol
    "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1",  # atropine
    "CC(=O)Oc1ccc(cc1)C(C)C",  # ibuprofen ester
    "O=C(O)c1cc(O)c(O)c(O)c1",  # gallic acid
    "c1ccc(cc1)-c1ccccc1",  # biphenyl
    "CC(C)(C)c1ccc(cc1)O",  # 4-tert-butylphenol
    "OC(=O)C=Cc1ccccc1",  # cinnamic acid
    "c1ccc2[nH]ccc2c1",  # indole
    "c1ccncc1",  # pyridine
    "C1=CC=C(C=C1)N",  # aniline
    "CC(=O)c1ccccc1",  # acetophenone
    "OC(=O)c1ccc(N)cc1",  # PABA
    "c1ccc(cc1)S",  # thiophenol
    "CC(O)CC(=O)O",  # 3-hydroxybutyric acid
    "c1ccc(cc1)C#N",  # benzonitrile
    "CC(=O)OCC",  # ethyl acetate
    "c1ccc(cc1)Cl",  # chlorobenzene
    "CC(C)O",  # isopropanol
    "CCCCCCCC",  # octane
    "c1ccc(cc1)F",  # fluorobenzene
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
]


def generate_synthetic_dataset(
    n_molecules: int = 500,
    seed: int = 42,
) -> tuple:
    """
    Generate a synthetic dataset for testing the ML pipeline.

    Creates molecules by enumerating variations of known drug-like SMILES
    and assigns synthetic docking scores correlated with molecular properties
    (MW, LogP, TPSA) to simulate realistic structure-activity relationships.

    Returns (mols_list, scores_array, names_list).
    """
    rng = np.random.RandomState(seed)

    # Build a pool of molecules from sample SMILES + random modifications
    base_mols = []
    for smi in _SAMPLE_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            base_mols.append(mol)

    if not base_mols:
        raise RuntimeError("Failed to parse any sample SMILES")

    # Repeat and shuffle to get n_molecules
    mols = []
    names = []
    idx = 0
    while len(mols) < n_molecules:
        mol = base_mols[idx % len(base_mols)]
        mols.append(mol)
        names.append(f"ligand_{len(mols):05d}")
        idx += 1

    mols = mols[:n_molecules]
    names = names[:n_molecules]

    # Generate synthetic scores correlated with molecular properties
    # Docking scores typically range from -12 to -3 kcal/mol
    scores = np.zeros(n_molecules)
    for i, mol in enumerate(mols):
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)

        # Synthetic score: weighted combination + noise
        # Negative values (lower = better binding)
        score = (
            -0.01 * mw
            - 0.3 * logp
            - 0.005 * tpsa
            + 0.2 * hbd
            - 0.1 * hba
            + 0.15 * rotatable
            + rng.normal(0, 0.8)
        )
        # Clamp to realistic range
        scores[i] = np.clip(score, -12.0, -1.0)

    logger.info(
        "Generated synthetic dataset: %d molecules, scores range [%.2f, %.2f]",
        n_molecules,
        scores.min(),
        scores.max(),
    )

    return mols, scores, names

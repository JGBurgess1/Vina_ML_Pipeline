"""
Molecular fingerprint generation using RDKit.

Generates multiple fingerprint types for a list of RDKit Mol objects,
returning numpy arrays suitable for ML training.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

logger = logging.getLogger(__name__)


@dataclass
class FingerprintSet:
    """A named fingerprint matrix with metadata."""
    name: str
    matrix: np.ndarray  # shape (n_molecules, n_bits)
    n_bits: int
    description: str


def _safe_fp_generation(mols, generator_fn, name):
    """
    Generate fingerprints with per-molecule error handling.
    Returns (matrix, failed_indices).
    """
    fps = []
    failed = []
    for i, mol in enumerate(mols):
        try:
            fp = generator_fn(mol)
            fps.append(fp)
        except Exception as e:
            logger.warning("Failed to generate %s FP for mol %d: %s", name, i, e)
            failed.append(i)
            fps.append(None)
    return fps, failed


def generate_morgan_fp(
    mols: list,
    radius: int = 2,
    n_bits: int = 2048,
) -> FingerprintSet:
    """Morgan/ECFP circular fingerprints."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    matrix = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        matrix[i] = gen.GetFingerprintAsNumPy(mol)
    name = f"Morgan_r{radius}_{n_bits}"
    return FingerprintSet(
        name=name,
        matrix=matrix,
        n_bits=n_bits,
        description=f"Morgan circular FP (radius={radius}, {n_bits} bits)",
    )


def generate_rdkit_fp(
    mols: list,
    n_bits: int = 2048,
) -> FingerprintSet:
    """RDKit topological/path-based fingerprints."""
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    matrix = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        matrix[i] = gen.GetFingerprintAsNumPy(mol)
    return FingerprintSet(
        name=f"RDKit_{n_bits}",
        matrix=matrix,
        n_bits=n_bits,
        description=f"RDKit topological FP ({n_bits} bits)",
    )


def generate_atompair_fp(
    mols: list,
    n_bits: int = 2048,
) -> FingerprintSet:
    """Atom pair fingerprints."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    matrix = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        matrix[i] = gen.GetFingerprintAsNumPy(mol)
    return FingerprintSet(
        name=f"AtomPair_{n_bits}",
        matrix=matrix,
        n_bits=n_bits,
        description=f"Atom pair FP ({n_bits} bits)",
    )


def generate_torsion_fp(
    mols: list,
    n_bits: int = 2048,
) -> FingerprintSet:
    """Topological torsion fingerprints."""
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    matrix = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        matrix[i] = gen.GetFingerprintAsNumPy(mol)
    return FingerprintSet(
        name=f"TopTorsion_{n_bits}",
        matrix=matrix,
        n_bits=n_bits,
        description=f"Topological torsion FP ({n_bits} bits)",
    )


def generate_maccs_fp(mols: list) -> FingerprintSet:
    """MACCS structural keys (166 bits, fixed)."""
    n_bits = 167  # MACCS keys are indexed 0-166
    matrix = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        for bit in fp.GetOnBits():
            matrix[i, bit] = 1
    return FingerprintSet(
        name="MACCS_167",
        matrix=matrix,
        n_bits=n_bits,
        description="MACCS structural keys (167 bits)",
    )


def generate_all_fingerprints(mols: list) -> list:
    """
    Generate all supported fingerprint types for a list of molecules.
    Returns a list of FingerprintSet objects.
    """
    generators = [
        ("Morgan_r2", lambda m: generate_morgan_fp(m, radius=2, n_bits=2048)),
        ("Morgan_r3", lambda m: generate_morgan_fp(m, radius=3, n_bits=2048)),
        ("RDKit", lambda m: generate_rdkit_fp(m, n_bits=2048)),
        ("AtomPair", lambda m: generate_atompair_fp(m, n_bits=2048)),
        ("TopTorsion", lambda m: generate_torsion_fp(m, n_bits=2048)),
        ("MACCS", lambda m: generate_maccs_fp(m)),
    ]

    fp_sets = []
    for name, gen_fn in generators:
        logger.info("Generating %s fingerprints for %d molecules...", name, len(mols))
        fp_set = gen_fn(mols)
        nonzero_bits = np.count_nonzero(fp_set.matrix.sum(axis=0))
        logger.info(
            "  %s: %d bits, %d non-zero across dataset (%.1f%%)",
            fp_set.name,
            fp_set.n_bits,
            nonzero_bits,
            100.0 * nonzero_bits / fp_set.n_bits,
        )
        fp_sets.append(fp_set)

    return fp_sets

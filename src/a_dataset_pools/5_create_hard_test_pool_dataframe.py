"""
Create a DataFrame from the hard test pool for easy access in benchmark creation.

The hard test pool serves as the basis for the benchmark dataset. To make it easily
accessible, we store it as a DataFrame containing basic information about each
molecule including SMILES, complexity, heavy atom count, and IUPAC names.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem.GraphDescriptors import BertzCT
from tqdm import tqdm


SUBMISSION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SUBMISSION_ROOT.parent.parent / "data" / "dataset_pools"
DEFAULT_INPUT_PATH = DATA_ROOT / "processed" / "pubchem_train_test_pools.pkl"
DEFAULT_OUTPUT_PATH = DATA_ROOT / "processed" / "hard_test_pool_dataframe.pkl"


@dataclass
class CreateHardTestPoolDataFrameConfig:
    """Paths and parameters for creating the hard test pool DataFrame."""

    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    nbr_processes: int = 100


def _compute_bertz_complexity(mol: Chem.Mol) -> float | None:
    """
    Compute Bertz complexity for a given RDKit molecule.

    Args:
        mol: The RDKit molecule object.

    Returns:
        Bertz complexity of the molecule, or None if computation fails.
    """
    bertz_complexity = BertzCT(mol)
    return bertz_complexity


def _compute_nbr_heavy_atoms(mol: Chem.Mol) -> int | None:
    """
    Compute the number of heavy atoms in a given RDKit molecule.

    Args:
        mol: The RDKit molecule object.

    Returns:
        Number of heavy atoms in the molecule.
    """
    return mol.GetNumHeavyAtoms() if mol else None


def _create_dataframe_from_pool(
    smiles_list: List[str], iupac_dict: dict, nbr_processes: int
) -> pd.DataFrame:
    """
    Create a DataFrame from a list of SMILES with molecular properties.

    Args:
        smiles_list: List of SMILES strings.
        iupac_dict: Dictionary mapping SMILES to IUPAC names.
        nbr_processes: Number of parallel processes to use.

    Returns:
        DataFrame with SMILES, mol objects, complexity, heavy atoms, and IUPAC names.
    """
    # Create rdkit mol objects
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    assert None not in mols, "Some SMILES could not be converted to rdkit mol objects."

    # Compute Bertz complexity
    with Pool(nbr_processes) as pool:
        bertz_complexities = list(
            tqdm(
                pool.imap(_compute_bertz_complexity, mols),
                total=len(mols),
                desc="Computing Bertz complexities",
            )
        )

    # Compute number of heavy atoms
    with Pool(nbr_processes) as pool:
        nbr_heavy_atoms = list(
            tqdm(
                pool.imap(_compute_nbr_heavy_atoms, mols),
                total=len(mols),
                desc="Computing number of heavy atoms",
            )
        )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "mol": mols,
            "complexity": bertz_complexities,
            "nbr_heavy_atoms": nbr_heavy_atoms,
        }
    )

    # Include IUPAC names if available
    df["iupac_name"] = df["smiles"].map(iupac_dict)

    # Create complexity bins: 0-100, 100-200, ..., 900-1000, 1000+
    # Bins are preliminarily defined here and can be adjusted as needed
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1e9]
    labels = [f"{i}-{i+100}" for i in range(0, 1000, 100)] + ["1000+"]
    df["complexity_bin"] = pd.cut(df["complexity"], bins=bins, labels=labels, right=False)

    return df


def create_hard_test_pool_dataframe(cfg: CreateHardTestPoolDataFrameConfig) -> None:
    """Create and save a DataFrame from the hard test pool."""

    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        with input_path.open("rb") as handle:
            hard_test_set, _, _, iupac_dict = pickle.load(handle)
    else:
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    hard_test_df = _create_dataframe_from_pool(hard_test_set, iupac_dict, cfg.nbr_processes)

    with output_path.open("wb") as handle:
        pickle.dump(hard_test_df, handle)


if __name__ == "__main__":
    create_hard_test_pool_dataframe(CreateHardTestPoolDataFrameConfig())

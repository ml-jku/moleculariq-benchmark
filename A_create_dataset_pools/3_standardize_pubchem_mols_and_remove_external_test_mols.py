"""
Canonicalise PubChem molecules and remove overlaps with external benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pickle

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm


SUBMISSION_ROOT = Path(__file__).resolve().parent
DATA_DIR = SUBMISSION_ROOT / "data"
DEFAULT_PUBCHEM_PATH = DATA_DIR / "intermediate" / "pubchem_smiles_and_iupacs.pkl"
DEFAULT_EXTERNAL_PATH = DATA_DIR / "external" / "external_test_set_molecules.pkl"
DEFAULT_OUTPUT_PATH = DATA_DIR / "processed" / "filtered_smiles_and_iupacs.pkl"

@dataclass
class FilterPubChemConfig:
    pubchem_mol_path: Path = DEFAULT_PUBCHEM_PATH
    external_test_mol_path: Path = DEFAULT_EXTERNAL_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH


def _normalise_smiles(smiles_list: Iterable[str]) -> List[str]:
    normalised: List[str] = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            normalised.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue
    return normalised

def _load_pickle(path: Path) -> Tuple[List[str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def filter_pubchem_mols(cfg: FilterPubChemConfig) -> Tuple[List[str], dict]:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    RDLogger.DisableLog("rdApp.*")

    smiles_list, iupac_dict = _load_pickle(Path(cfg.pubchem_mol_path))

    if Path(cfg.external_test_mol_path).exists():
        with Path(cfg.external_test_mol_path).open("rb") as handle:
            external_test_smiles = pickle.load(handle)
    else:
        raise FileNotFoundError(f"External test molecule file not found: {cfg.external_test_mol_path}")

    canonical_external = set(_normalise_smiles(external_test_smiles))
    canonical_pubchem = set(_normalise_smiles(smiles_list))

    filtered_smiles = sorted(canonical_pubchem - canonical_external)
    if not filtered_smiles:
        raise ValueError("No PubChem molecules remain after filtering.")

    with Path(cfg.output_path).open("wb") as handle:
        pickle.dump((filtered_smiles, iupac_dict), handle)

    return filtered_smiles, iupac_dict


if __name__ == "__main__":
    filter_pubchem_mols(FilterPubChemConfig())

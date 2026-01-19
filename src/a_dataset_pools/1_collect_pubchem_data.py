"""
Extract molecules in terms of SMILES and IUPAC information from locally stored 
PubChem SDF files.

SDF files are not included in this repository due to their large size. Please download
the PubChem SDF archive from:
https://pubchem.ncbi.nlm.nih.gov/docs/downloads#section=From-the-PubChem-FTP-Site.

If no SDF files are found in the specified directory, a small pseudo dataset is
created from a pseudo sdf (see data/dataset_pools/pseudo_sdf) to allow end-to-end testing of the
pipeline. The pseudo sdf file is one the many pubchem data chunks.

Usage:
- Provide file_path to the SDF files (or run the script with the one provided SDF)
- Make sure output directory exists
- Run the script
"""

from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, List, Tuple

import gzip
import os
import pickle
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MolToSmiles
from tqdm import tqdm


SUBMISSION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SUBMISSION_ROOT.parent.parent / "data" / "dataset_pools"
DEFAULT_RAW_DIR = DATA_ROOT / "pubchem_raw_sdf"
DEFAULT_PSEUDO_DIR = DATA_ROOT / "pseudo_sdf"
DEFAULT_SAVE_PATH = DATA_ROOT / "intermediate" / "pubchem_smiles_and_iupacs.pkl"


@dataclass
class PubChemCollectionConfig:
    """Paths and parameters controlling the collection step."""

    data_dir: Path = DEFAULT_RAW_DIR
    pseudo_dir: Path = DEFAULT_PSEUDO_DIR
    save_path: Path = DEFAULT_SAVE_PATH
    nbr_processes: int = 4


def _contains_carbon(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def _is_multifragment(mol: Chem.Mol) -> bool:
    frags = Chem.GetMolFrags(mol)
    return len(frags) > 1


def _process_pubchem_sdf_file(root_dir: str, file: str) -> Tuple[List[str], DefaultDict[str, List[str]]]:
    file_path = os.path.join(root_dir, file)

    file_smiles_list: List[str] = []
    file_iupac_dict: DefaultDict[str, List[str]] = defaultdict(list)

    try:
        with gzip.open(file_path, "rb") as handle:
            suppl = Chem.ForwardSDMolSupplier(handle)
            for mol in suppl:
                if mol is None:
                    continue
                if not _contains_carbon(mol):
                    continue
                if _is_multifragment(mol):
                    continue

                try:
                    iupac = mol.GetProp("PUBCHEM_IUPAC_NAME")
                except Exception:
                    iupac = None

                try:
                    smiles = MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    smiles = MolToSmiles(mol)
                except Exception:
                    smiles = None

                if smiles is None:
                    continue

                file_smiles_list.append(smiles)
                if iupac:
                    file_iupac_dict[smiles].append(iupac)
    except Exception:
        # Swallow file-level errors so the batch job can continue.
        return [], defaultdict(list)

    return file_smiles_list, file_iupac_dict


def _extend_dict(target: DefaultDict[str, List[str]], items: Iterable[Tuple[str, List[str]]]) -> None:
    for smiles, names in items:
        target[smiles].extend(names)


def collect_pubchem_data(cfg: PubChemCollectionConfig) -> None:
    """Collect SMILES/IUPAC pairs from a directory of ``*.sdf.gz`` files."""

    data_dir = Path(cfg.data_dir)
    save_path = Path(cfg.save_path)
    nbr_processes = max(1, cfg.nbr_processes)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    RDLogger.DisableLog("rdApp.*")

    iupac_dict: DefaultDict[str, List[str]] = defaultdict(list)
    smiles_list: List[str] = []
    sdf_found = False

    if data_dir.exists():
        for root, _, files in os.walk(data_dir):
            sdf_files = [f for f in files if f.endswith(".sdf.gz")]
            if not sdf_files:
                continue

            sdf_found = True
            func = partial(_process_pubchem_sdf_file, root)

            with Pool(nbr_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(func, sdf_files),
                        total=len(sdf_files),
                        desc=f"Processing SDFs in {root}",
                    )
                )

            files_smiles_list, files_single_iupac_dict = zip(*results) if results else ([], [])
            for file_iupac in files_single_iupac_dict:
                _extend_dict(iupac_dict, file_iupac.items())

            for file_smiles_list in files_smiles_list:
                smiles_list.extend(file_smiles_list)

    if not sdf_found or not smiles_list:
        # Fallback: read from pseudo_sdf directory
        pseudo_dir = Path(cfg.pseudo_dir)
        if pseudo_dir.exists():
            for root, _, files in os.walk(pseudo_dir):
                sdf_files = [f for f in files if f.endswith(".sdf.gz")]
                if not sdf_files:
                    continue

                func = partial(_process_pubchem_sdf_file, root)

                with Pool(nbr_processes) as pool:
                    results = list(
                        tqdm(
                            pool.imap(func, sdf_files),
                            total=len(sdf_files),
                            desc=f"Processing pseudo SDFs in {root}",
                        )
                    )

                files_smiles_list, files_single_iupac_dict = zip(*results) if results else ([], [])
                for file_iupac in files_single_iupac_dict:
                    _extend_dict(iupac_dict, file_iupac.items())

                for file_smiles_list in files_smiles_list:
                    smiles_list.extend(file_smiles_list)

    with save_path.open("wb") as handle:
        pickle.dump([smiles_list, iupac_dict], handle)


if __name__ == "__main__":
    collect_pubchem_data(PubChemCollectionConfig())

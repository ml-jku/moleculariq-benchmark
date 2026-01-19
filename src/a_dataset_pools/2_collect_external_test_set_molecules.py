"""
Aggregate SMILES strings from external test set snapshots stored locally.

The allow for a better integration with other existing benchmarks, we collect included
molecules in the llasmaol, ChemIQ, ChemDFM, and Ether0 test sets.

The collected SMILES are stored in a pickle file for later use (filtering).

Note: Since the pooled molecules stem from snapshots of existing benchmarks, we cannot
guarantee that all molecules of updated benchmark versions are included.
"""

from dataclasses import dataclass
from pathlib import Path

import pickle

from utils.llasmol_testset import get_llasmol_test_set
from utils.chemIQ_testset import get_chemiq_test_set
from utils.chemDFM_testset import get_chemDFM_test_set
from utils.ether0_testset import get_ether0_test_set



SUBMISSION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SUBMISSION_ROOT.parent.parent / "data" / "dataset_pools"
DATA_DIR = DATA_ROOT / "external"
DEFAULT_SAVE_PATH = DATA_ROOT / "external" / "external_test_set_molecules.pkl"


@dataclass
class ExternalTestSetConfig:
    save_path: Path = DEFAULT_SAVE_PATH

def collect_external_test_set_molecules(cfg: ExternalTestSetConfig) -> list:
    """
    Collects molecules from external test sets and returns a list of standardized 
    SMILES.
    
    Log file: Report nbr of molecules collected from each test set.
    """
    
    # intialize the list of smiles
    smiles_list = list()
    
    # collect llasmol test set molecules
    print('Process llasmol molecules ...')
    llasmol_smiles = get_llasmol_test_set()
    smiles_list.extend(llasmol_smiles)

    #chemiq
    print('Process chemiq molecules ...')
    chemIQ_smiles = get_chemiq_test_set()
    smiles_list.extend(chemIQ_smiles)
    
    # chemdfm
    print('Process chemDFM molecules ...')
    chemDFM_smiles = get_chemDFM_test_set()
    smiles_list.extend(chemDFM_smiles)
    
    # ether0
    print('Process ether0 molecules ...')
    ether0_smiles = get_ether0_test_set()
    smiles_list.extend(ether0_smiles)

    # remove duplicates
    smiles_list = list(set(smiles_list))
        
    # save the list to a file
    with open(cfg.save_path, 'wb') as f:
        pickle.dump(smiles_list, f)
    
    return smiles_list

#---------------------------------------------------------------------------------------
# Run script
if __name__ == "__main__":
    cfg = ExternalTestSetConfig()
    collect_external_test_set_molecules(cfg)

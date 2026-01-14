"""
This file includes the ChemIQ test set, which has been used in the publication:
Assessing the Chemical Intelligence of Large Language Models 
https://arxiv.org/pdf/2505.07735
"""

#---------------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = SUBMISSION_ROOT.parent.parent / "data" / "dataset_pools"
DEFAULT_CHEMIQ_PATH = DATA_ROOT / "external" / "chemIQ_data"

@dataclass
class ChemIQTestSetConfig:
    data_path: Path = DEFAULT_CHEMIQ_PATH
    file_names: Tuple[str] = ("additional_smiles_to_iupac.jsonl",
                              "chemiq.jsonl")

#---------------------------------------------------------------------------------------
# Dependencies
import os
import json
from rdkit import Chem

#---------------------------------------------------------------------------------------
# Define function to get ChemIQ test set
def get_chemiq_test_set(cfg: ChemIQTestSetConfig=ChemIQTestSetConfig()) -> list:
    """
    Returns the ChemIQ test set as a list of SMILES strings.
    """
    smiles_list = []
    for file_name in cfg.file_names:
        file_path = os.path.join(cfg.data_path, file_name)
        
        # Define keywords for filtering
        keywords = ['scaffold_smiles', 'smiles_random', 'smiles', 'scaffold', 
                    'reactant_1', 'reactant_1_random', 'smiles2', 'true_smi_canonical', 
                    'reactant_2_random', 'product_random', 'smiles1', 
                    'core_molecule_smi', 'reactant_2', 'product']
        
        # read jsonl file
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Check if any of the keywords are in the data
                for keyword in keywords:
                    if keyword in data["meta_data"].keys():
                        smiles = data["meta_data"][keyword]
                        if isinstance(smiles, str):
                            smiles_list.append(smiles)
                        elif isinstance(smiles, list):
                            raise ValueError(f"Expected string but got {type(smiles)}")
        
    # Remove duplicates
    smiles_list = list(set(smiles_list))    
        
    return smiles_list

#---------------------------------------------------------------------------------------
# Debugging
if __name__ == "__main__":
    cfg = ChemIQTestSetConfig()
    smiles_list = get_chemiq_test_set(cfg)
    print("Number of SMILES in ChemIQ test set:", len(smiles_list))
    print("Done.")
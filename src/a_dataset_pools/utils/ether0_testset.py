"""
This file includs the Ether0 test set, which has been used in the publication:
https://www.futurehouse.org/research-announcements/ether0-a-scientific-reasoning-model-for-chemistry
https://huggingface.co/datasets/futurehouse/ether0-benchmark
"""

#---------------------------------------------------------------------------------------
# Config
from dataclasses import dataclass
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = SUBMISSION_ROOT.parent.parent / "data" / "dataset_pools"
DEFAULT_ETHER0_PATH = DATA_ROOT / "external" / "ether0_data" / "test-00000-of-00001.parquet"

@dataclass
class Ether0TestSetConfig:
    data_path: str = str(DEFAULT_ETHER0_PATH)
    
#---------------------------------------------------------------------------------------
# Dependencies
from .check_string_for_smiles import filter_smiles
import pandas as pd
import re
from rdkit import RDLogger
from rdkit import Chem
#---------------------------------------------------------------------------------------
# Define function to get Ether0 test set
def get_ether0_test_set(cfg: Ether0TestSetConfig=Ether0TestSetConfig()) -> list:
    """
    Returns the Ether0 test set as a list of SMILES strings.
    """
    
    # Ignore warnings from RDKit
    RDLogger.DisableLog('rdApp.*')
    
    # Initialize output list
    smiles_list = list()
    
    # Read the CSV file
    df = pd.read_parquet(cfg.data_path)
    
    # Extract columns
    problem = df['problem'].tolist()
    solution = df['solution'].tolist()
    ideal = df['ideal'].tolist()
    unformatted = df['unformatted'].tolist()
    
    # Extract SMILES from the problem text
    problem_chunks = list()
    for text in problem:
        chunked_text = _chunk_problem_text(text)
        problem_chunks.extend(chunked_text)

    problem_smiles = [filter_smiles(s) for s in problem_chunks]
    problem_smiles = list(filter(lambda x: x is not None, problem_smiles))
    
    # Extract SMILES from the solution text
    solution_smiles = [_apply_regex_filter_to_get_solution_from_text(text) for text in solution]
    solution_smiles = [filter_smiles(s) for s in solution_smiles]
    solution_smiles = list(filter(lambda x: x is not None, solution_smiles))
    
    # Extract SMILES from the ideal text
    ideal_smiles = [filter_smiles(text) for text in ideal]
    ideal_smiles = list(filter(lambda x: x is not None, ideal_smiles))
    
    # Extract SMILES from the unformatted text
    unformatted_smiles = [filter_smiles(text) for text in unformatted]
    unformatted_smiles = list(filter(lambda x: x is not None, unformatted_smiles))

    # Combine all SMILES
    smiles_list.extend(problem_smiles)
    smiles_list.extend(solution_smiles)
    smiles_list.extend(ideal_smiles)
    smiles_list.extend(unformatted_smiles)
    # Remove duplicates
    smiles_list = list(set(smiles_list))
    
    return smiles_list

def _chunk_problem_text(text:str)-> list:
    """
    Splits the text by white spaces and removes potential dots at the end
    """
    
    # Split the text by white spaces
    chunks = text.split()
    
    # Remove potential dots at the end of each chunk
    chunks = [chunk.rstrip('.') for chunk in chunks]
    
    return chunks

def _apply_regex_filter_to_get_solution_from_text(text: str) -> str:
    """
    Applies a regex filter to extract the solution from the text.
    Solutions are assumed within this format: !:!solution!:!
    """
    
    # Define the regex pattern to match the solution
    pattern = r'!:!(.*?)!:!'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    else:
        return None

#---------------------------------------------------------------------------------------
# Debugging
if __name__ == "__main__":
    cfg = Ether0TestSetConfig()
    smiles_list = get_ether0_test_set(cfg)
    print(smiles_list[:10])
    print("Number of SMILES in Ether0 test set:", len(smiles_list))
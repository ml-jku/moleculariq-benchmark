"""
This file includes the chemDFM test set, which has been used in the publication:
https://arxiv.org/abs/2401.14818
https://github.com/OpenDFM/ChemDFM/tree/main/ChemLLMBench_eval_data
"""

#---------------------------------------------------------------------------------------
# Config
from dataclasses import dataclass
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHEMDFM_PATH = SUBMISSION_ROOT / "data" / "external" / "chemDFM_data" / "ChemLLMBench_eval_data"

@dataclass
class ChemDFMTestSetConfig:
    data_base_path: str = str(DEFAULT_CHEMDFM_PATH) + "/"
    mol_design_file: str = "text_based_molecule_design.jsonl"
    prop_bace_file: str = "MolecularPropertyPrediction/bace.jsonl"
    prop_bbbp_file: str = "MolecularPropertyPrediction/bbbp.jsonl"
    prop_clintox_file: str = "MolecularPropertyPrediction/clintox.jsonl"
    prop_hiv_file: str = "MolecularPropertyPrediction/hiv.jsonl"
    prop_tox21_file: str = "MolecularPropertyPrediction/tox21.jsonl"
    rec_iupac_smiles_file: str = "MoleculeRecognition/iupac_to_smiles.jsonl"
    rec_smiles_iupac_file: str = "MoleculeRecognition/smiles_to_iupac.jsonl"
    rec_smiles_formula_file: str = ("MoleculeRecognition/"
                                    "smiles_to_molecular_formula.jsonl")
    rec_mol_captioning_file: str = "MoleculeRecognition/molecule_captioning.jsonl"
    
#---------------------------------------------------------------------------------------
# Dependencies
from .check_string_for_smiles import filter_smiles
import json
import re
from typing import Literal
from rdkit import Chem
from rdkit import RDLogger
#---------------------------------------------------------------------------------------
# Define function to get chemDFM test set
def get_chemDFM_test_set(cfg: ChemDFMTestSetConfig=ChemDFMTestSetConfig()) -> list:
    """
    Returns the chemDFM test set as a list of SMILES strings.
    """
    smiles_list = list()
    
    # Process the molecule design task
    mol_design_path = cfg.data_base_path + cfg.mol_design_file
    mol_design_smiles = _process_mol_design_task(mol_design_path)
    smiles_list.extend(mol_design_smiles)
    
    # Molecular property predictions
    bace_path = cfg.data_base_path + cfg.prop_bace_file
    bbbp_path = cfg.data_base_path + cfg.prop_bbbp_file
    clintox_path = cfg.data_base_path + cfg.prop_clintox_file
    hiv_path = cfg.data_base_path + cfg.prop_hiv_file
    tox21_path = cfg.data_base_path + cfg.prop_tox21_file
    
    bace_smiles = _process_property_prediction_task(bace_path)
    bbbp_smiles = _process_property_prediction_task(bbbp_path)
    clintox_smiles = _process_property_prediction_task(clintox_path)
    hiv_smiles = _process_property_prediction_task(hiv_path)
    tox21_smiles = _process_property_prediction_task(tox21_path)

    property_smiles = (bace_smiles + bbbp_smiles + clintox_smiles + hiv_smiles + 
                       tox21_smiles)
    smiles_list.extend(property_smiles)
    
    # Molecule recognition tasks
    rec_iupac_smiles_path = cfg.data_base_path + cfg.rec_iupac_smiles_file
    rec_smiles_iupac_path = cfg.data_base_path + cfg.rec_smiles_iupac_file
    rec_smiles_formula_path = cfg.data_base_path + cfg.rec_smiles_formula_file
    rec_mol_captioning_path = cfg.data_base_path + cfg.rec_mol_captioning_file
    smiles_iupacToSmiles = _process_recognition_task(rec_iupac_smiles_path, "iupac_to_smiles")
    smiles_smilesToIupac = _process_recognition_task(rec_smiles_iupac_path, "smiles_to_iupac")
    smiles_smilesToFormula = _process_recognition_task(rec_smiles_formula_path, 
                                               "smiles_to_formula")
    smiles_molCaptioning = _process_recognition_task(rec_mol_captioning_path, 
                                               "molecule_captioning")
    recognition_smiles = (smiles_iupacToSmiles + smiles_smilesToIupac + 
                          smiles_smilesToFormula + smiles_molCaptioning)
    smiles_list.extend(recognition_smiles)

    return smiles_list

def _process_mol_design_task(path:str) -> list:
    """
    Processes a single molecule design task from the chemDFM test set.
    Returns a list of SMILES strings.
    """
    
    # Read the JSONL file and extract the answers
    # From a quick check we found SMILES just appear in the "answer" field
    answer_list = list()
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            answer = data["answer"]
            assert isinstance(answer, list)
            assert len(answer) == 1, "Expected exactly one answer in the molecule design task."
            answer_list.append(answer[0])
    
    # Filter the answers to get valid SMILES
    valid_smiles = [filter_smiles(s) for s in answer_list]
    valid_smiles = list(filter(lambda x: x is not None, valid_smiles))

    return valid_smiles

def _process_property_prediction_task(path: str) -> list:
    """
    Processes a single molecular property prediction task from the chemDFM test set.
    Returns a list of SMILES strings.
    
    From a quick check we found SMILES just appear in the question field.
    SMILES are presented with this format:
    "\nSMILES: smiles_string\n"
    """
    # Load the JSONL file
    task_list = list()
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            
            # Extract SMILES using regex
            smiles_match = re.search(r'SMILES:\s*([^\n]+)\n', question)
            if smiles_match:
                smiles = smiles_match.group(1).strip()
                valid_smiles = filter_smiles(smiles)
                if valid_smiles:
                    task_list.append(valid_smiles)
    return task_list

recognition_task_choices = Literal["iupac_to_smiles", "smiles_to_iupac", 
                                   "smiles_to_formula", "molecule_captioning"]
def _process_recognition_task(path: str, task_type: recognition_task_choices) -> list:
    """
    Processes a single molecule recognition task from the chemDFM test set.
    Returns a list of SMILES strings.
    
    The location where to find SMILES strings depends on the task type:
    - iupac to smiles: answer field
    - molecule captioning: input field
    - smiles to iupac: question field, 
      SMILES are presented with this format: "... by SMILES: smiles_string."
    - smiles to formula: question field, 
      SMILES are presented with this format: "... The molecule is smiles_string."

    """
    task_smiles = list()
    if task_type == "iupac_to_smiles":
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                answer = data["answer"]
                assert isinstance(answer, str)
                smiles = filter_smiles(answer)
                if smiles:
                    task_smiles.append(smiles)
    elif task_type == "smiles_to_iupac":
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                smiles_match = re.search(r'SMILES:\s*([^\n]+)\.', question)
                if smiles_match:
                    smiles = smiles_match.group(1).strip()
                    valid_smiles = filter_smiles(smiles)
                    if valid_smiles:
                        task_smiles.append(valid_smiles)
    elif task_type == "smiles_to_formula":
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                smiles_match = re.search(r'The molecule is\s*([^\n]+)\.', question)
                if smiles_match:
                    smiles = smiles_match.group(1).strip()
                    valid_smiles = filter_smiles(smiles)
                    if valid_smiles:
                        task_smiles.append(valid_smiles)
    elif task_type == "molecule_captioning":
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                input_text = data["input"]
                valid_smiles = filter_smiles(input_text)
                if valid_smiles:
                    task_smiles.append(valid_smiles)
    else:
        raise ValueError(f"Invalid task type: {task_type}."
                          f"Expected one of {recognition_task_choices}.")

    return task_smiles

#---------------------------------------------------------------------------------------
# Debugging
if __name__ == "__main__":
    cfg = ChemDFMTestSetConfig()
    smiles_list = get_chemDFM_test_set(cfg)
    print(f"Collected {len(smiles_list)} SMILES from the chemDFM test set.")
    print(smiles_list[:5])  # Print first 5 SMILES for verification

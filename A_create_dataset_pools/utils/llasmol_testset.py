"""
This file provides access to the llasmol test set.
"""

#---------------------------------------------------------------------------------------
# Dependencies
from huggingface_hub import hf_hub_download
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import zipfile
import json

#---------------------------------------------------------------------------------------
# Define function
def get_llasmol_test_set() -> list:
    """
    Returns the llasmol test set as a list of SMILES strings.
    """
    # Disable RDKit warnings for invalid SMILES
    RDLogger.DisableLog('rdApp.*')
    zip_path = hf_hub_download(
        repo_id="osunlp/SMolInstruct",
        filename="data.zip",
        repo_type="dataset"
    )

    dataset = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        test_files = [name for name in zip_ref.namelist()
                     if name.startswith('raw/test/') and name.endswith('.jsonl')]

        for test_file in test_files:
            with zip_ref.open(test_file, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.decode('utf-8')))

    smiles_list = []

    for entry in dataset:
        task = entry.get("task", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        # Tasks where input contains SMILES
        if (task.startswith("name_conversion-s2") or
            task.startswith("property_prediction") or
            task == "molecule_captioning" or
            task == "forward_synthesis" or
            task == "retrosynthesis"):
            smiles = input_text.strip()
            # Try parsing the whole string first
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(smiles)
            else:
                # Try splitting by both . and ; for multi-component molecules and salts
                # Replace ; with . to have a single separator
                smiles_normalized = smiles.replace(";", ".")
                for smiles_part in smiles_normalized.split("."):
                    smiles_part = smiles_part.strip()
                    if smiles_part:
                        mol = Chem.MolFromSmiles(smiles_part)
                        if mol is not None:
                            smiles_list.append(smiles_part)

        # Tasks where output contains SMILES
        if (task.startswith("name_conversion-i2s") or
            task.startswith("name_conversion-f2s") or
            task == "molecule_generation" or
            task == "forward_synthesis" or
            task == "retrosynthesis"):
            smiles = output_text.strip()
            # Try parsing the whole string first
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(smiles)
            else:
                # Try splitting by both . and ; for multi-component molecules and salts
                # Replace ; with . to have a single separator
                smiles_normalized = smiles.replace(";", ".")
                for smiles_part in smiles_normalized.split("."):
                    smiles_part = smiles_part.strip()
                    if smiles_part:
                        mol = Chem.MolFromSmiles(smiles_part)
                        if mol is not None:
                            smiles_list.append(smiles_part)

    return smiles_list
    
#---------------------------------------------------------------------------------------
# Debugging
if __name__ == "__main__":

    llasmol_test_mols = get_llasmol_test_set()
    print("Done.")


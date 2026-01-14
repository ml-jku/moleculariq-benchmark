"""
This files includes a helper function to check whether a string is a valid SMILES
string that can be processed by RDKit.
"""

from rdkit import Chem
#---------------------------------------------------------------------------------------
# Define the function to filter SMILES strings

def filter_smiles(potential_smiles: str) -> bool:
    """
    Checks if the potential SMILES string can be processed by RDKit.
    """
    
    try:
        mol = Chem.MolFromSmiles(potential_smiles)
        if mol is not None:
            return potential_smiles
        else:
            return None
    except:
        return None
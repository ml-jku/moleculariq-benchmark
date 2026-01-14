"""
This script computes the properties, i.e. the task-specific ground truth values, 
for all molecules in the chosen molecule pool.
This is needed as a preprocessing step because later the benchmark will be sampled
using inverse value counts as weights to ensure a diverse benchmark.
"""

from pathlib import Path

from dataclasses import dataclass


SUBMISSION_ROOT = Path(__file__).resolve().parent
POOLS_DIR = SUBMISSION_ROOT.parent / "A_create_dataset_pools" / "data" / "processed"
PROPERTIES_DIR = SUBMISSION_ROOT / "data"


@dataclass
class PropertyCreationConfig:
    input_path: Path = POOLS_DIR / "hard_test_pool_dataframe.pkl"
    output_path: Path = PROPERTIES_DIR / "properties.pkl"

#---------------------------------------------------------------------------------------
# Imports
from B_create_benchmark.solver.solver import SymbolicSolver
from B_create_benchmark.task_names import task_names

import pandas as pd
from functools import partial
import multiprocessing as mp
import os
import random
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ENABLE_PANDARALLEL = os.environ.get("ENABLE_PANDARALLEL", "0").lower() in {"1", "true", "yes"}
if ENABLE_PANDARALLEL:  # pragma: no cover - optional fast path
    try:
        from pandarallel import pandarallel  # type: ignore

        PANDARALLEL_AVAILABLE = True
    except ImportError:
        pandarallel = None  # type: ignore
        PANDARALLEL_AVAILABLE = False
else:
    pandarallel = None  # type: ignore
    PANDARALLEL_AVAILABLE = False


def _series_apply(series, func):
    if PANDARALLEL_AVAILABLE and hasattr(series, "parallel_apply"):
        return series.parallel_apply(func)
    return series.apply(func)


def _df_apply_rows(df, func):
    if PANDARALLEL_AVAILABLE and hasattr(df, "parallel_apply"):
        return df.parallel_apply(func, axis=1)
    return df.apply(func, axis=1)


def _safe_map(func, values, n_workers: int):
    if n_workers <= 1:
        return [func(v) for v in values]
    try:
        with mp.Pool(processes=n_workers) as pool:
            return pool.map(func, values)
    except (OSError, PermissionError):  # pragma: no cover - sandbox fallback
        return [func(v) for v in values]
#---------------------------------------------------------------------------------------
# Functions
def transform_smiles(smiles: str, seed: int = None) -> str:
    """
    Transform SMILES with 50% chance of randomization and 50% chance of kekulization.

    Args:
        smiles: Original SMILES string
        seed: Random seed for reproducibility

    Returns:
        Transformed SMILES string
    """
    if seed is not None:
        random.seed(seed)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # 50% chance to randomize SMILES
        randomize = random.random() < 0.5

        # 50% chance to kekulize (independent of randomization)
        kekulize = random.random() < 0.5

        if randomize and kekulize:
            return Chem.MolToSmiles(mol, doRandom=True, kekuleSmiles=True)
        elif randomize:
            return Chem.MolToSmiles(mol, doRandom=True)
        elif kekulize:
            return Chem.MolToSmiles(mol, kekuleSmiles=True)
        else:
            return Chem.MolToSmiles(mol)  # Canonical form
    except:
        # If any error occurs, return original SMILES
        return smiles


def compute_properties(config: PropertyCreationConfig = PropertyCreationConfig()):
    if PANDARALLEL_AVAILABLE:
        pandarallel.initialize(progress_bar=False)

    n_workers = os.cpu_count() or 1

    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        df = pd.read_pickle(input_path)
    else:
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    # Filtering
    filtered = df[(df.nbr_heavy_atoms >= 5) & (df.nbr_heavy_atoms <= 50)]
    if not filtered.empty:
        df = filtered
    else:
        raise ValueError("No molecules left after filtering by heavy atom count.")
    df = df[df.smiles.apply(lambda x: isinstance(x, str) and len(x) < 100)].reset_index(drop=True)
    df = df[~df.smiles.isna()].reset_index(drop=True)
    
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Subsetting
    # df = df.iloc[:100000]
    print(f"Loaded {len(df)} molecules.")

    # Store original SMILES
    df['original_smiles'] = df['smiles']

    # Transform SMILES with reproducible randomness
    print("Transforming SMILES with randomization/kekulization...")
    df['smiles'] = _df_apply_rows(df, lambda row: transform_smiles(row['smiles'], seed=row.name))

    # Define solver
    solver = SymbolicSolver()
    
    # Compute properties
    for task_name in task_names:
        print(f"Computing property '{task_name}' ...")
        
        if task_name == "ring":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_ring_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_ring_indices)
        elif task_name ==  "fused_ring":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_fused_ring_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_fused_ring_indices)
        elif task_name == "bridgehead":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_bridgehead_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_bridgehead_indices)
        elif task_name == "smallest_largest_ring_size":
            # smallest ring size
            func_count = partial(solver.get_smallest_or_largest_ring_count, smallest=True)
            func_index = partial(solver.get_smallest_or_largest_ring_indices, smallest=True)
            df[f"{task_name}_smallest_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_smallest_index"] = _series_apply(df["smiles"], func_index)
            # largest ring size
            func_count = partial(solver.get_smallest_or_largest_ring_count, smallest=False)
            func_index = partial(solver.get_smallest_or_largest_ring_indices, smallest=False)
            df[f"{task_name}_largest_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_largest_index"] = _series_apply(df["smiles"], func_index)
        elif task_name == "chain_termini":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_chain_termini_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_chain_termini_indices)
        elif task_name == "branch_point":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_branch_point_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_branch_point_indices)
        elif task_name == "aromatic_ring":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_aromatic_ring_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_aromatic_ring_indices)
        elif task_name == "aliphatic_ring":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_aliphatic_ring_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_aliphatic_ring_indices)
        elif task_name == "heterocycle":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_heterocycle_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_heterocycle_indices)
        elif task_name == "saturated_ring":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_saturated_ring_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_saturated_ring_indices)
        elif task_name == "csp3_carbon":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_csp3_carbon_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_csp3_carbon_indices)
        elif task_name == "longest_carbon_chain":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_longest_carbon_chain_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_longest_carbon_chain_indices)
        elif task_name == "r_s_stereocenter":
            func_count = partial(solver.get_r_or_s_stereocenter_count, r_count=True)
            func_index = partial(solver.get_r_or_s_stereocenter_indices, r_indices=True)
            df[f"{task_name}_r_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_r_index"] = _series_apply(df["smiles"], func_index)
            func_count = partial(solver.get_r_or_s_stereocenter_count, r_count=False)
            func_index = partial(solver.get_r_or_s_stereocenter_indices, r_indices=False)
            df[f"{task_name}_s_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_s_index"] = _series_apply(df["smiles"], func_index)
        elif task_name == "unspecified_stereocenter":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_unspecified_stereocenter_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_unspecified_stereocenter_indices)
        elif task_name == "e_z_stereochemistry_double_bond":
            func_count = partial(solver.get_e_z_stereochemistry_double_bond_count, e_count=True)
            func_index = partial(solver.get_e_z_stereochemistry_double_bond_indices, e_indices=True)
            df[f"{task_name}_e_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_e_index"] = _series_apply(df["smiles"], func_index)
            func_count = partial(solver.get_e_z_stereochemistry_double_bond_count, e_count=False)
            func_index = partial(solver.get_e_z_stereochemistry_double_bond_indices, e_indices=False)
            df[f"{task_name}_z_count"] = _series_apply(df["smiles"], func_count)
            df[f"{task_name}_z_index"] = _series_apply(df["smiles"], func_index)
        elif task_name == "stereochemistry_unspecified_double_bond":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_stereochemistry_unspecified_double_bond_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_stereochemistry_unspecified_double_bond_indices)
        elif task_name == "stereocenter":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_stereocenter_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_stereocenter_indices)
        elif task_name == "carbon_atom":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_carbon_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_carbon_indices)
        elif task_name == "hetero_atom":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_hetero_atom_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_hetero_atom_indices)
        elif task_name == "halogen_atom":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_halogen_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_halogen_indices)
        elif task_name == "heavy_atom":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_heavy_atom_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_heavy_atom_indices)
        elif task_name == "hydrogen_atom":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_hydrogen_count)
            # hydrogen indices is not possible and therefore omitted
        elif task_name == "molecular_formula":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_molecular_formula)
            df[f"{task_name}_value"] = df[f"{task_name}_count"]
        elif task_name == "hba":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_hba_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_hba_indices)
        elif task_name == "hbd":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_hbd_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_hbd_indices)
        elif task_name == "rotatable_bond":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_rotatable_bond_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_rotatable_bond_indices)
        elif task_name == "oxidation_state":
            elements =  ['C', 'N', 'O', 'P', 'S']
            max_oxidation_choices = [True, False]
            max_ox_mapping = {True: "max", False: "min"}
            for element in elements:
                for max_ox in max_oxidation_choices:
                    func_count = partial(solver.get_oxidation_state_count, element=element, max_oxidation=max_ox)
                    func_index = partial(solver.get_oxidation_state_indices, element=element, max_oxidation=max_ox)
                    df[f"{task_name}_{element}_{max_ox_mapping[max_ox]}_count"] = _series_apply(df["smiles"], func_count)
                    df[f"{task_name}_{element}_{max_ox_mapping[max_ox]}_index"] = _series_apply(df["smiles"], func_index)
        elif task_name == "functional_group":
            #fg_data = [solver.get_functional_group_count_and_indices(smiles) 
            #           for smiles in df["smiles"]]
            fg_data = _safe_map(solver.get_functional_group_count_and_indices, df["smiles"], n_workers)
            df = pd.concat([df, pd.DataFrame(fg_data)], axis=1)
        elif task_name == "brics_decomposition":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_brics_fragment_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_brics_bond_indices)
        elif task_name == "template_based_reaction_prediction":
            #react_data = [solver.get_reaction_counts_and_indices(smiles) 
            #              for smiles in df["smiles"]]
            react_data = _safe_map(solver.get_reaction_counts_and_indices, df["smiles"], n_workers)
            df = pd.concat([df, pd.DataFrame(react_data)], axis=1)
        elif task_name == "murcko_scaffold":
            df[f"{task_name}_count"] = _series_apply(df["smiles"], solver.get_murcko_scaffold_count)
            df[f"{task_name}_index"] = _series_apply(df["smiles"], solver.get_murcko_scaffold_indices)
            df[f"{task_name}_value"] = _series_apply(df["smiles"], solver.get_murcko_scaffold_value)
        else:
            raise ValueError(f"Unknown task name '{task_name}'")
        
    # Save results to pkl file
    df.to_pickle(output_path)
    print(f"Saved results to '{output_path}'")

#---------------------------------------------------------------------------------------
# Run script
if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    compute_properties()
    end_time = datetime.now()
    print("Total time in minutes:", (end_time - start_time).total_seconds()/60.0)

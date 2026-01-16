"""
Constraint satisfaction index building.

Vectorized index building for (property, value) -> set(smiles) mapping.
This is the key speedup from the turbo generator.
"""

from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.smiles import canonicalize_smiles
from ..utils.properties import to_python_scalar


# Standalone functional group boolean properties to skip (use _nbrInstances instead)
STANDALONE_FG_PROPS = {
    "aldehyde", "ketone", "carboxylic_acid", "ester", "amide", "lactone",
    "lactam", "acyl_chloride", "acyl_bromide", "acyl_fluoride", "acyl_iodide",
    "anhydride", "primary_amine", "secondary_amine", "tertiary_amine",
    "quaternary_ammonium", "imine", "nitrile", "nitro", "nitroso", "azide",
    "hydrazine", "hydrazone", "oxime", "isocyanate", "isothiocyanate",
    "carbodiimide", "urea", "thiourea", "guanidine", "alcohol", "phenol",
    "ether", "aryl_ether", "epoxide", "peroxide", "hydroperoxide", "enol",
    "hemiacetal", "hemiketal", "acetal", "ketal", "thiol", "thioether",
    "disulfide", "sulfoxide", "sulfone", "sulfonic_acid", "sulfonamide",
    "sulfonic_ester", "thioketone", "thioaldehyde", "alkyl_fluoride",
    "alkyl_chloride", "alkyl_bromide", "alkyl_iodide", "aryl_fluoride",
    "aryl_chloride", "aryl_bromide", "aryl_iodide", "vinyl_halide",
    "trifluoromethyl", "perfluoroalkyl", "phosphine", "phosphine_oxide",
    "phosphonium", "phosphonic_acid", "phosphate", "phosphonate",
    "phosphoramide", "alkene", "alkyne", "allene", "aromatic",
    "conjugated_system", "enamine", "enol_ether", "ketene", "carbene",
    "michael_acceptor", "alpha_beta_unsaturated", "benzene", "naphthalene",
    "pyridine", "pyrrole", "furan", "thiophene", "imidazole", "pyrazole",
    "oxazole", "thiazole", "pyrimidine", "indole", "quinoline", "isoquinoline",
    "methyl", "ethyl", "propyl", "isopropyl", "butyl", "isobutyl", "sec_butyl",
    "tert_butyl", "primary_carbon", "secondary_carbon", "tertiary_carbon",
    "quaternary_carbon", "boc", "cbz", "fmoc", "tosyl", "mesyl", "triflate",
    "carbonate", "carbamate", "thioester", "imide", "sulfonyl_chloride",
    "isonitrile", "n_oxide", "hydroxylamine", "diazo", "nitrone", "aziridine",
    "silyl_ether", "silane", "silyl_enol_ether", "boronic_acid", "boronic_ester",
    "borate", "vinyl_ether", "orthoester", "allyl", "benzyl", "propargyl",
}

# Columns to skip entirely
SKIP_COLS = {
    "smiles", "mol", "iupac_name", "original_smiles", "canonical_smiles",
    "molecular_formula", "complexity", "nbr_heavy_atoms", "complexity_bin",
    "InChI", "molecular_formula_value",
}

# Suffixes to skip
SKIP_SUFFIX = {"_indices", "_index", "_list", "_array", "_products"}


def build_constraint_index(
    properties_df: pd.DataFrame,
    skip_list_properties: bool = True,
    show_progress: bool = True,
) -> Dict[tuple, Set[str]]:
    """
    Vectorized build of (property, value) -> set(canonical_smiles).

    Skips array-like/dict/unhashable/NaN; keeps template_*_success and
    functional_group_*_nbrInstances.

    Args:
        properties_df: DataFrame with molecular properties
        skip_list_properties: Whether to skip list/array valued properties
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping (property, value) tuples to sets of SMILES
    """
    print("Building constraint satisfaction index (vectorized)...")

    df = properties_df.copy()

    # Ensure canonical_smiles column once
    if "canonical_smiles" not in df.columns:
        if "original_smiles" in df.columns and df["original_smiles"].notna().any():
            df["canonical_smiles"] = df["original_smiles"]
        else:
            df["canonical_smiles"] = df["smiles"].map(canonicalize_smiles)

    idx: Dict[tuple, Set[str]] = {}

    columns = df.columns
    if show_progress:
        columns = tqdm(columns, desc="  Indexing columns")

    for col in columns:
        # Skip explicitly listed columns
        if col in SKIP_COLS:
            continue

        # Skip columns with invalid suffixes
        if any(col.endswith(sfx) for sfx in SKIP_SUFFIX):
            continue

        # Skip standalone functional group boolean properties
        if col in STANDALONE_FG_PROPS:
            continue

        # Handle functional_group_* properties
        if col.startswith("functional_group_"):
            # Only keep functional_group_*_nbrInstances
            if not col.endswith("_nbrInstances"):
                continue

        # Handle template_based_reaction_* properties
        if col.startswith("template_based_reaction_"):
            # Only keep template_based_reaction_*_success
            if not col.endswith("_success"):
                continue

        s = df[col]
        # Filter out NaN and complex types
        mask_valid = ~s.isna()
        if skip_list_properties:
            mask_valid &= ~s.apply(
                lambda v: isinstance(v, (list, np.ndarray, dict))
            )

        sub = df.loc[mask_valid, ["canonical_smiles", col]].copy()
        if sub.empty:
            continue

        # Normalize values once
        sub[col] = sub[col].map(to_python_scalar)

        # Group by value -> set of smiles
        groups = sub.groupby(col, sort=False)["canonical_smiles"].apply(set)
        for val, smi_set in groups.items():
            try:
                hash(val)
            except TypeError:
                continue
            if not smi_set:
                continue
            idx[(col, val)] = smi_set

    print(f"  Indexed {len(idx):,} unique (property, value) constraints")
    print(f"  Covering {len(df):,} molecules")
    return idx


def get_conflicting_properties(prop: str) -> Set[str]:
    """
    Return set of properties that conflict with the given property.

    Conflicting properties provide redundant information when used together.
    For example, molecular_formula already defines all elemental counts.

    Args:
        prop: Property name

    Returns:
        Set of property names that conflict with prop
    """
    # Define conflict groups
    molecular_formula_conflicts = {
        "carbon_atom_count",
        "hydrogen_atom_count",
        "hetero_atom_count",
        "heavy_atom_count",
        "halogen_atom_count",
    }

    # If prop is molecular_formula, return all its conflicts
    if prop == "molecular_formula":
        return molecular_formula_conflicts

    # If prop is one of the elemental counts, it conflicts with molecular_formula
    if prop in molecular_formula_conflicts:
        return {"molecular_formula"}

    # No conflicts for other properties
    return set()


def has_conflicting_properties(properties: list) -> bool:
    """
    Check if a list of properties contains any conflicting pairs.

    Args:
        properties: List of property names

    Returns:
        True if there are conflicts, False otherwise
    """
    conflicting_groups = [
        {
            "molecular_formula",
            "carbon_atom_count",
            "hydrogen_atom_count",
            "hetero_atom_count",
            "heavy_atom_count",
            "halogen_atom_count",
        },
    ]
    for group in conflicting_groups:
        if len(set(properties) & group) >= 2:
            return True
    return False


def has_duplicate_properties(constraints: list) -> bool:
    """
    Check if constraints list has duplicate property names.

    Args:
        constraints: List of (property, value) tuples

    Returns:
        True if duplicates exist, False otherwise
    """
    props = [p for p, _ in constraints]
    return len(props) != len(set(props))

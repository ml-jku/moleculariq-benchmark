"""
Constraint validation utilities.

Validates constraints against molecules using the reward system.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.b_benchmark.rewards.constraint_reward import multi_constraint_generation_reward


def validate_constraints(smiles: str, constraints: List[Dict[str, Any]]) -> bool:
    """
    Validate that all constraints hold for the given SMILES.

    Uses the reward system to verify constraint satisfaction.

    Args:
        smiles: SMILES string to validate
        constraints: List of constraint dictionaries with property, operator, value

    Returns:
        True if all constraints are satisfied, False otherwise
    """
    try:
        result = multi_constraint_generation_reward(
            smiles, constraints, return_details=False
        )
        return result == 1.0
    except Exception:
        return False


def check_constraints_satisfiable(
    constraints: List[Dict[str, Any]],
    properties_df: pd.DataFrame,
    max_check: int = 1000,
) -> Tuple[bool, int]:
    """
    Check if constraints are satisfiable (have at least one matching molecule).

    Args:
        constraints: List of constraint dicts with property, operator, value
        properties_df: Properties dataframe to check against
        max_check: Maximum number of molecules to check

    Returns:
        Tuple of (is_satisfiable, num_matches)
    """
    if len(properties_df) == 0:
        return False, 0

    mask = pd.Series([True] * len(properties_df), index=properties_df.index)

    for constraint in constraints:
        prop = constraint["property"]
        operator = constraint.get("operator", "=")
        value = constraint["value"]

        if prop not in properties_df.columns:
            return False, 0

        col = properties_df[prop]

        # Skip NaN and array values
        valid_mask = ~col.isna()
        if len(col) > 0:
            first_val = col.iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                valid_mask &= col.apply(
                    lambda x: not isinstance(x, (list, np.ndarray))
                    if x is not None
                    else True
                )

        try:
            if operator == "=":
                constraint_mask = col == value
            elif operator == ">":
                constraint_mask = col > value
            elif operator == "<":
                constraint_mask = col < value
            elif operator == ">=":
                constraint_mask = col >= value
            elif operator == "<=":
                constraint_mask = col <= value
            elif operator == "range":
                min_val = constraint.get("min_value")
                max_val = constraint.get("max_value")
                if min_val is None or max_val is None:
                    return False, 0
                constraint_mask = (col >= min_val) & (col <= max_val)
            else:
                continue

            mask &= constraint_mask & valid_mask
        except (TypeError, ValueError):
            return False, 0

    num_matches = mask.sum()
    return num_matches > 0, int(num_matches)


def compute_match_rate(
    df: pd.DataFrame, constraints: List[Dict[str, Any]]
) -> float:
    """
    Compute the fraction of molecules matching all constraints.

    Args:
        df: Properties dataframe
        constraints: List of constraint dictionaries

    Returns:
        Match rate (0.0 to 1.0)
    """
    if len(df) == 0:
        return 0.0

    mask = pd.Series(True, index=df.index)
    for constraint in constraints:
        prop = constraint["property"]
        if prop not in df.columns:
            return 0.0

        operator = constraint.get("operator", "=")

        if operator == "=":
            mask &= df[prop] == constraint["value"]
        elif operator == ">":
            mask &= df[prop] > constraint["value"]
        elif operator == "<":
            mask &= df[prop] < constraint["value"]
        elif operator == ">=":
            mask &= df[prop] >= constraint["value"]
        elif operator == "<=":
            mask &= df[prop] <= constraint["value"]
        elif operator == "range":
            mask &= (df[prop] >= constraint["min_value"]) & (
                df[prop] <= constraint["max_value"]
            )

        if not mask.any():
            return 0.0

    return float(mask.mean())

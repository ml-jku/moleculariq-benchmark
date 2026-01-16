"""
Constraint combination scoring.

Sophisticated scoring that considers cumulative reduction, size bonuses,
and rarity bonuses. This is the quality scoring from the turbo generator.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ..utils.properties import is_zero_value


def count_zeros_in_constraints(constraints: List[Tuple[str, Any]]) -> int:
    """
    Count how many constraints have zero values.

    Args:
        constraints: List of (property, value) tuples

    Returns:
        Number of zero-valued constraints
    """
    return sum(1 for prop, val in constraints if is_zero_value(val))


def count_special_properties(constraints: List[Tuple[str, Any]]) -> int:
    """
    Count special properties (reaction templates, FG instances).

    Args:
        constraints: List of (property, value) tuples

    Returns:
        Number of special property constraints
    """
    count = 0
    for prop, val in constraints:
        if prop.startswith("template_based_reaction_") and prop.endswith("_success"):
            count += 1
        elif prop.startswith("functional_group_") and prop.endswith("_nbrInstances"):
            count += 1
    return count


def get_k_thresholds(k: int) -> Dict[str, Any]:
    """
    Get threshold parameters for constraint count k.

    Args:
        k: Number of constraints

    Returns:
        Dictionary with min_molecules, max_molecules, ideal_range
    """
    if k == 5:
        return dict(min_molecules=5, max_molecules=1500, ideal_range=(20, 120))
    elif k == 3:
        return dict(min_molecules=8, max_molecules=1200, ideal_range=(40, 180))
    else:  # k == 2
        return dict(min_molecules=10, max_molecules=1200, ideal_range=(60, 220))


def get_k_min_intersection(k: int, global_min: int = 10) -> int:
    """
    Get minimum intersection size for constraint count k.

    Args:
        k: Number of constraints
        global_min: Default minimum

    Returns:
        Minimum intersection size
    """
    return 10 if k == 2 else 8 if k == 3 else 5


def get_rr_cutoff_for_k(k: int) -> float:
    """
    Get reduction ratio cutoff for pre-screening.

    Args:
        k: Number of constraints

    Returns:
        Reduction ratio threshold
    """
    return 2.0 if k <= 3 else 1.5


def zero_is_informative(
    combo: List[Tuple[str, Any]],
    constraint_index: Dict[Tuple[str, Any], Set[str]],
    min_ratio: float = 1.4,
) -> bool:
    """
    Check if zero-valued constraints in combo are actually informative.

    A zero constraint is informative if it significantly reduces the
    candidate set (by at least min_ratio).

    Args:
        combo: List of (property, value) constraint tuples
        constraint_index: Mapping from constraints to satisfying SMILES sets
        min_ratio: Minimum reduction ratio to consider informative

    Returns:
        True if zeros are informative, False otherwise
    """
    zeros = [c for c in combo if is_zero_value(c[1])]
    if not zeros:
        return True
    nonzeros = [c for c in combo if not is_zero_value(c[1])]
    if not nonzeros:
        return False

    base = constraint_index.get(nonzeros[0], set())
    for c in nonzeros[1:]:
        base = base & constraint_index.get(c, set())
        if not base:
            return False

    for z in zeros:
        zset = constraint_index.get(z, set())
        inter = base & zset
        if not inter:
            return False
        if len(base) / len(inter) < min_ratio:
            return False
        base = inter
    return True


def score_constraint_combination(
    constraints: List[Tuple[str, Any]],
    constraint_index: Dict[Tuple[str, Any], Set[str]],
    total_molecules: int,
    zero_limit: int = 1,
    max_special: int = 3,
    min_molecules: int = 10,
    max_molecules: int = 1000,
    ideal_range: Tuple[int, int] = (50, 200),
) -> Tuple[float, int, float]:
    """
    Score a constraint combination for quality.

    Scoring considers:
    - Zero-value limits
    - Special property limits
    - Satisfying molecule count (must be in bounds)
    - Cumulative reduction (information gain)
    - Size bonuses for ideal ranges
    - Rarity bonuses for rare combinations

    Args:
        constraints: List of (property, value) tuples
        constraint_index: Mapping from constraints to satisfying SMILES sets
        total_molecules: Total number of molecules in dataset
        zero_limit: Maximum allowed zero-valued constraints
        max_special: Maximum allowed special properties
        min_molecules: Minimum satisfying molecules required
        max_molecules: Maximum satisfying molecules allowed
        ideal_range: Ideal range of satisfying molecules

    Returns:
        Tuple of (score, n_satisfying, match_rate)
        Score of 0.0 means invalid/rejected combination
    """
    # Zero-value limit
    if count_zeros_in_constraints(constraints) > zero_limit:
        return 0.0, 0, 0.0

    # Special properties limit
    if count_special_properties(constraints) > max_special:
        return 0.0, 0, 0.0

    # Fetch sets
    sets = [constraint_index.get(c, set()) for c in constraints]
    if not all(sets):
        return 0.0, 0, 0.0

    # Full intersection
    inter = sets[0]
    for s in sets[1:]:
        inter = inter & s
        if not inter:
            return 0.0, 0, 0.0

    n_satisfying = len(inter)
    match_rate = n_satisfying / total_molecules if total_molecules > 0 else 0.0

    # Bounds
    if n_satisfying < min_molecules or n_satisfying > max_molecules:
        return 0.0, n_satisfying, match_rate

    # Cumulative reduction (step-by-step)
    cumulative_reduction = 1.0
    current = sets[0]
    for next_set in sets[1:]:
        new_inter = current & next_set
        if not new_inter:
            return 0.0, 0, 0.0
        reduction = len(current) / len(new_inter)
        cumulative_reduction *= reduction
        current = new_inter

    # Size bonus
    size_score = 1.0
    if ideal_range[0] <= n_satisfying <= ideal_range[1]:
        size_score = 2.0
    elif (n_satisfying >= min_molecules and n_satisfying < ideal_range[0]) or (
        n_satisfying > ideal_range[1] and n_satisfying <= max_molecules // 2
    ):
        size_score = 1.5

    # Rarity bonus
    rarity_score = 1.0
    if match_rate < 0.01:
        rarity_score = 2.0
    elif match_rate < 0.05:
        rarity_score = 1.5
    elif match_rate < 0.10:
        rarity_score = 1.2

    score = cumulative_reduction * size_score * rarity_score
    return score, n_satisfying, match_rate


def normalize_prop_for_caps(p: str) -> str:
    """
    Normalize property name for cap tracking.

    Args:
        p: Property name

    Returns:
        Normalized property name
    """
    if p.startswith("functional_group_") and p.endswith("_nbrInstances"):
        return "functional_group"
    if p.startswith("template_based_reaction_") and p.endswith("_success"):
        return "reaction_template"
    if p == "molecular_formula_count":
        return "molecular_formula"
    return p


def combo_norm_props(constraints: List[Tuple[str, Any]]) -> Set[str]:
    """
    Get normalized property names from constraints for cap checking.

    Args:
        constraints: List of (property, value) tuples

    Returns:
        Set of normalized property names
    """
    return {normalize_prop_for_caps(p) for (p, _v) in constraints}

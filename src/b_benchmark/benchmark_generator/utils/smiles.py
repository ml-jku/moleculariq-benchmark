"""
SMILES utility functions for canonicalization and transformation.
"""

import random
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    if not smiles or smiles == "...":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def transform_smiles(
    smiles: str,
    rng: random.Random,
    randomize_prob: float = 0.5,
    kekulize_prob: float = 0.5,
    return_flags: bool = False
) -> Tuple[str, str, bool, bool] | Tuple[str, str]:
    """
    Transform SMILES with optional randomization and kekulization.

    Args:
        smiles: Original SMILES string
        rng: Random number generator
        randomize_prob: Probability of randomizing SMILES order (0.0-1.0)
        kekulize_prob: Probability of using Kekule form (0.0-1.0)
        return_flags: If True, return (original, transformed, randomized, kekulized)

    Returns:
        Tuple of (original_smiles, transformed_smiles) or
        Tuple of (original_smiles, transformed_smiles, was_randomized, was_kekulized) if return_flags=True
    """
    was_randomized = False
    was_kekulized = False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if return_flags:
                return smiles, smiles, False, False
            return smiles, smiles

        # Randomly decide on transformations
        was_randomized = rng.random() < randomize_prob
        was_kekulized = rng.random() < kekulize_prob

        if was_randomized and was_kekulized:
            transformed = Chem.MolToSmiles(mol, doRandom=True, kekuleSmiles=True)
        elif was_randomized:
            transformed = Chem.MolToSmiles(mol, doRandom=True)
        elif was_kekulized:
            transformed = Chem.MolToSmiles(mol, kekuleSmiles=True)
        else:
            transformed = Chem.MolToSmiles(mol)  # Canonical form

        if return_flags:
            return smiles, transformed, was_randomized, was_kekulized
        return smiles, transformed
    except Exception:
        if return_flags:
            return smiles, smiles, False, False
        return smiles, smiles


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def _find_ring_closures(smiles: str) -> List[Tuple[int, int, str]]:
    """
    Find all ring closure tokens in a SMILES string.

    Returns list of (start_pos, end_pos, ring_number_str) tuples.
    Ring closures are digits 1-9 or %XX patterns that follow atoms or bonds.

    In SMILES notation, ring closures can follow:
    - Atom symbols (C, N, c, n, etc.)
    - Bracket close ]
    - Other ring closures (consecutive ring closures like C12)
    - Bond characters (-, =, #, :, /, \\) for specifying ring bond types
    """
    closures = []

    # Valid characters that can precede a ring closure digit
    # This includes: letters, ], digits, and bond characters
    bond_chars = {"-", "=", "#", ":", "/", "\\", "~"}

    # Pattern for extended ring notation: %XX where XX is two digits
    extended_pattern = re.compile(r"%(\d{2})")

    # First, find all extended ring closures (%XX)
    for match in extended_pattern.finditer(smiles):
        start = match.start()
        end = match.end()
        ring_num = match.group(1)  # The two-digit number
        closures.append((start, end, ring_num))

    # Now find single-digit ring closures (1-9)
    # These appear after atoms, bonds, ], or other ring closures, but not inside brackets
    i = 0
    in_bracket = False
    while i < len(smiles):
        char = smiles[i]

        if char == "[":
            in_bracket = True
        elif char == "]":
            in_bracket = False
        elif char == "%":
            # Skip extended notation (already handled)
            if i + 2 < len(smiles) and smiles[i + 1 : i + 3].isdigit():
                i += 3
                continue
        elif char.isdigit() and not in_bracket:
            # Check if this is a ring closure (digit after valid ring closure context)
            if i > 0:
                prev_char = smiles[i - 1]
                # Ring closures follow: letters, ], digits, or bond characters
                if (
                    prev_char.isalpha()
                    or prev_char == "]"
                    or prev_char.isdigit()
                    or prev_char in bond_chars
                ):
                    # Check not already in extended list
                    already_found = any(
                        start <= i < end for start, end, _ in closures
                    )
                    if not already_found:
                        closures.append((i, i + 1, char))
        i += 1

    # Sort by position
    closures.sort(key=lambda x: x[0])
    return closures


def has_rings(smiles: str) -> bool:
    """
    Check if a SMILES string contains any ring closures.

    Args:
        smiles: SMILES string to check

    Returns:
        True if the SMILES has at least one ring, False otherwise
    """
    closures = _find_ring_closures(smiles)
    return len(closures) > 0


def enumerate_rings_in_smiles(
    smiles: str, rng: random.Random, max_attempts: int = 10
) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Replace sequential ring closure numbers with random integers 1-99.

    In SMILES notation, ring closures use digits to pair atoms forming rings.
    The actual numbers are arbitrary - they just need to match. This function
    replaces the original ring numbers with random ones (using %XX format
    for numbers > 9) to test if models understand ring notation.

    The function ensures the output SMILES is different from the input.
    If the random assignment produces the same SMILES, it will resample.

    Args:
        smiles: Original SMILES string
        rng: Random number generator for reproducibility
        max_attempts: Maximum resampling attempts if result equals original

    Returns:
        Tuple of (ring_enumerated_smiles, mapping_dict)
        - ring_enumerated_smiles: SMILES with randomized ring numbers, or None if failed
        - mapping_dict: {original_num: new_num} e.g., {'1': 42, '2': 17}

    Example:
        Input: "c1ccc2ccccc2c1" (naphthalene with rings 1, 2)
        Output: ("c%17ccc%42ccccc%42c%17", {'1': 17, '2': 42})
    """
    if not smiles:
        return None, {}

    closures = _find_ring_closures(smiles)
    if not closures:
        # No rings, return original
        return smiles, {}

    # Collect unique ring numbers
    unique_ring_nums = set()
    for _, _, ring_num in closures:
        unique_ring_nums.add(ring_num)

    sorted_unique_nums = sorted(unique_ring_nums)
    n_unique = len(sorted_unique_nums)

    # Try multiple times to get a different SMILES
    for attempt in range(max_attempts):
        # Generate random replacements (1-99, no duplicates)
        available_nums = list(range(1, 100))
        rng.shuffle(available_nums)

        # Create mapping from old ring numbers to new ones
        mapping: Dict[str, int] = {}
        for i, old_num in enumerate(sorted_unique_nums):
            mapping[old_num] = available_nums[i]

        # Build new SMILES by replacing ring closures
        # Process from end to start to preserve positions
        result = list(smiles)
        for start, end, old_num in reversed(closures):
            new_num = mapping[old_num]
            # Format: single digit for 1-9, %XX for 10-99
            if new_num <= 9:
                new_token = str(new_num)
            else:
                new_token = f"%{new_num:02d}"
            result[start:end] = list(new_token)

        new_smiles = "".join(result)

        # Check if result is different from original
        if new_smiles == smiles:
            continue  # Resample

        # Validate with RDKit - the canonical SMILES should be identical
        try:
            original_mol = Chem.MolFromSmiles(smiles)
            new_mol = Chem.MolFromSmiles(new_smiles)

            if original_mol is None or new_mol is None:
                return None, {}

            # Verify they represent the same molecule
            original_canonical = Chem.MolToSmiles(original_mol)
            new_canonical = Chem.MolToSmiles(new_mol)

            if original_canonical != new_canonical:
                return None, {}

        except Exception:
            return None, {}

        return new_smiles, mapping

    # All attempts produced the same SMILES (very unlikely with 99 options)
    return None, {}

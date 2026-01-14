#!/usr/bin/env python3
"""
Complete benchmark dataset generator for molecular reasoning tasks.
Generates count/index/constraint tasks and creates HuggingFace dataset.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi
from tqdm import tqdm


# Add repo root to path so imports work from any location
import sys
SUBMISSION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.b_benchmark.questions import TASKS
from src.b_benchmark.natural_language.mappings import parse_natural_language, COUNT_MAPPINGS, INDEX_MAPPINGS
from src.b_benchmark.natural_language.formatter import format_constraint, format_count_query, format_index_query
from src.b_benchmark.column_category_map import COUNT_MAP, INDEX_MAP, CONSTRAINT_MAP, COUNT_TO_INDEX_MAP
from src.b_benchmark.rewards.constraint_reward import multi_constraint_generation_reward


# NOTE: Using explicit category maps bundled with the submission instead of
# dynamic detection for better consistency and control.


DEFAULT_PROPERTIES_PATH = REPO_ROOT / "data" / "benchmark" / "properties.pkl"


def transform_smiles(smiles: str, rng: random.Random) -> Tuple[str, str]:
    """
    Return original SMILES without randomisation. The tuple shape is kept so downstream
    code can continue to unpack into (original, transformed) while both entries are identical.

    Args:
        smiles: Original SMILES string
        rng: Random number generator for reproducibility

    Returns:
        Tuple of (original_smiles, transformed_smiles)
    """
    return smiles, smiles


def load_and_prepare_data(pickle_path: str) -> pd.DataFrame:
    """
    Load moleculariq properties and add complexity bins.

    Args:
        pickle_path: Path to the properties pickle file

    Returns:
        DataFrame with complexity_bin column added
    """
    with tqdm(total=2, desc="Loading data", unit="step") as pbar:
        df = pd.read_pickle(pickle_path)
        pbar.update(1)
        pbar.set_description("Preparing bins")

        complexity_lims = [0, 250, 1000, np.inf]
        df['complexity_bin'] = pd.cut(
            df['complexity'],
            bins=complexity_lims,
            labels=['0-250', '250-1000', '1000-inf'],
            right=False
        )
        pbar.update(1)

    return df


def validate_constraints(smiles: str, constraints: List[Dict[str, Any]]) -> bool:
    """Return True if all constraints hold for the given SMILES."""
    try:
        result = multi_constraint_generation_reward(smiles, constraints, return_details=False)
        return result == 1.0
    except Exception:
        return False


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy scalar types to native Python numbers/bools."""
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonicalize_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def get_count_properties(df: pd.DataFrame, include_reactions: bool = False) -> List[str]:
    """Get all count properties from the dataframe."""
    count_props = []
    for col in df.columns:
        if '_count' in col and '_index' not in col:
            # Exclude reaction templates unless explicitly included
            if include_reactions or not col.startswith('template_based_reaction_prediction_'):
                count_props.append(col)
    return count_props


def get_index_properties(df: pd.DataFrame, include_reactions: bool = False) -> List[str]:
    """Get all index properties from the dataframe."""
    index_props = []
    for col in df.columns:
        if '_index' in col:
            # Exclude reaction templates unless explicitly included
            if include_reactions or not col.startswith('template_based_reaction_prediction_'):
                index_props.append(col)
    return index_props


# categorize_properties function removed - now using explicit COUNT_MAP, INDEX_MAP from column_category_map.py


def select_property_with_inverse_frequency(
    available_props: List[str],
    property_usage_counter: Dict[str, int],
    category_usage_counter: Dict[str, int],
    row_data: Dict[str, Any],
    rng: random.Random,
    apply_zero_bias: bool = True
) -> str:
    """
    Select a property using two-level inverse frequency weighting:
    1. First select category with inverse frequency
    2. Then select property within category with inverse frequency + optional zero bias

    Args:
        available_props: List of available properties to choose from
        property_usage_counter: Dict tracking usage count per property
        category_usage_counter: Dict tracking usage count per category
        row_data: Row data containing property values for zero bias
        rng: Random number generator for reproducibility
        apply_zero_bias: Whether to apply zero/empty bias

    Returns:
        Selected property name
    """
    if len(available_props) == 0:
        raise ValueError("No available properties to select from")

    if len(available_props) == 1:
        return available_props[0]

    # Step 1: Group properties by category
    available_categories = {}
    for prop in available_props:
        cat = get_property_category(prop)
        if cat not in available_categories:
            available_categories[cat] = []
        available_categories[cat].append(prop)

    # Step 2: Select category with inverse frequency weighting
    categories = list(available_categories.keys())
    if len(categories) == 1:
        chosen_category = categories[0]
    else:
        cat_weights = [1.0 / (1 + category_usage_counter.get(cat, 0)) for cat in categories]
        cat_weights = np.array(cat_weights)
        cat_weights = cat_weights / cat_weights.sum()
        chosen_category = rng.choices(categories, weights=cat_weights, k=1)[0]

    # Step 3: Select property within category with inverse frequency + zero bias
    props_in_category = available_categories[chosen_category]
    if len(props_in_category) == 1:
        return props_in_category[0]

    prop_weights = []
    for prop in props_in_category:
        base_weight = 1.0 / (1 + property_usage_counter.get(prop, 0))

        # Apply zero bias if requested
        if apply_zero_bias and prop in row_data:
            if prop.endswith('_count'):
                # For count properties, reduce weight if value is 0
                if row_data[prop] == 0:
                    base_weight *= 0.5
            elif prop.endswith('_index'):
                # For index properties, reduce weight if list is empty
                value = row_data[prop]
                if not isinstance(value, list) or len(value) == 0:
                    base_weight *= 0.5

        prop_weights.append(base_weight)

    prop_weights = np.array(prop_weights)
    prop_weights = prop_weights / prop_weights.sum()
    chosen_prop = rng.choices(props_in_category, weights=prop_weights, k=1)[0]

    return chosen_prop


def _compute_inverse_frequency_arrays(
    df: pd.DataFrame,
    property_col: str,
    min_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return index labels and normalised weights for inverse-frequency sampling."""
    series = df[property_col]

    filtered = series.dropna()
    if filtered.empty:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)

    value_counts = filtered.value_counts()

    if min_count > 1:
        valid_values = value_counts[value_counts >= min_count].index
        filtered = filtered[filtered.isin(valid_values)]
        if filtered.empty:
            filtered = series.dropna()
            value_counts = filtered.value_counts()

    if filtered.empty:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)

    inv_weights = filtered.map(lambda x: 1.0 / value_counts[x])
    zero_mask = filtered == 0
    if zero_mask.any():
        inv_weights.loc[zero_mask] *= 0.5

    weights = inv_weights.to_numpy(dtype=np.float64)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        weights = np.full(len(filtered), 1.0 / len(filtered), dtype=np.float64)
    else:
        weights /= weight_sum

    labels = filtered.index.to_numpy(copy=False)
    weights = weights.astype(np.float32, copy=False)

    return labels, weights


def sample_with_inverse_frequency(
    df: pd.DataFrame,
    property_col: str,
    n_samples: int,
    min_count: int = 100,
    seed: int = 42,
    cache: Optional[Dict[Any, Tuple[np.ndarray, np.ndarray]]] = None,
    cache_key: Optional[Any] = None
) -> pd.DataFrame:
    """
    Sample using inverse frequency weighting, underweighting zeros.

    Args:
        df: DataFrame to sample from
        property_col: Column to use for weighting
        n_samples: Number of samples to draw
        min_count: Minimum value count to include
        seed: Random seed

    Returns:
        Sampled DataFrame
    """
    cache_entry = None
    if cache is not None and cache_key is not None:
        cache_entry = cache.get(cache_key)

    if cache_entry is None:
        labels, weights = _compute_inverse_frequency_arrays(
            df,
            property_col,
            min_count
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = (labels, weights)
    else:
        labels, weights = cache_entry

    if len(labels) == 0:
        return df.iloc[0:0]

    n_actual = min(n_samples, len(labels))
    if n_actual == 0:
        return df.iloc[0:0]

    rng_local = np.random.default_rng(seed)
    weights_array = weights.astype(np.float64, copy=False)
    weights_sum = weights_array.sum()
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        weights_array = np.full(len(weights_array), 1.0 / len(weights_array), dtype=np.float64)
    else:
        weights_array /= weights_sum
    replace = n_actual > len(labels)
    choice_indices = rng_local.choice(
        len(labels),
        size=n_actual,
        replace=replace,
        p=weights_array
    )
    selected_labels = np.atleast_1d(labels[choice_indices])

    return df.loc[selected_labels]


def _prime_sampling_cache(
    df_bins_cache: Dict[Any, pd.DataFrame],
    valid_bins: List[Any],
    sampling_cache: Dict[Tuple[Any, str, int], Tuple[np.ndarray, np.ndarray]],
    *,
    min_count: int,
    workers: int
) -> None:
    """Prime inverse-frequency cache entries in parallel."""
    tasks: List[Tuple[Any, str]] = []
    for bin_name in valid_bins:
        df_bin = df_bins_cache[bin_name]
        for category_name, category_properties in COUNT_MAP.items():
            if category_name == 'reaction_success':
                continue
            for prop in category_properties:
                if prop in df_bin.columns:
                    tasks.append((bin_name, prop))

    if not tasks or workers <= 0:
        return

    desc = f"Priming single-task cache ({workers} workers)"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {}
        for bin_name, prop in tasks:
            df_bin = df_bins_cache[bin_name]
            future = executor.submit(_compute_inverse_frequency_arrays, df_bin, prop, min_count)
            future_to_key[future] = (bin_name, prop)

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc=desc, unit="prop"):
            bin_name, prop = future_to_key[future]
            positions, weights = future.result()
            sampling_cache[(bin_name, prop, min_count)] = (positions, weights)


def generate_paired_single_tasks(
    df: pd.DataFrame,
    n_samples_per_bin: int = 10,
    seed: int = 42,
    *,
    prime_workers: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Generate paired count and index tasks for the same molecule-property combinations.
    Each selected molecule-property pair generates both a count and index task.
    Uses explicit COUNT_MAP and COUNT_TO_INDEX_MAP for pairing.

    Args:
        df: DataFrame with properties
        n_samples_per_bin: Samples per property per complexity bin
        seed: Random seed
        prime_workers: Optional number of worker threads to precompute sampling weights

    Returns:
        Tuple of (task_list, molecule_property_mapping)
        - task_list: List of task dictionaries (both count and index)
        - molecule_property_mapping: Dict mapping molecule SMILES to properties used
    """
    tasks = []
    molecule_property_mapping = {}  # Track which properties are used for each molecule
    complexity_bins = df['complexity_bin'].unique()

    # Pre-slice dataframe per bin so we only compute filters once
    df_bins_cache: Dict[Any, pd.DataFrame] = {}
    valid_bins: List[Any] = []
    for bin_name in complexity_bins:
        df_bin = df[df['complexity_bin'] == bin_name]
        if len(df_bin) == 0:
            continue
        df_bins_cache[bin_name] = df_bin
        valid_bins.append(bin_name)

    # Question templates for both task types
    count_templates = TASKS['single_count']['question_templates']
    index_templates = TASKS['single_index_identification']['question_templates']

    rng = random.Random(seed)
    sampling_cache: Dict[Tuple[Any, str, int], Tuple[np.ndarray, np.ndarray]] = {}

    if prime_workers > 0:
        _prime_sampling_cache(
            df_bins_cache,
            valid_bins,
            sampling_cache,
            min_count=1,
            workers=prime_workers
        )

    total_single_iterations = 0
    for bin_name in valid_bins:
        df_bin = df_bins_cache[bin_name]
        available_categories = 0
        for category_name, category_properties in COUNT_MAP.items():
            if category_name == 'reaction_success':
                continue
            if any(p in df_bin.columns for p in category_properties):
                available_categories += 1
        total_single_iterations += available_categories * n_samples_per_bin

    with tqdm(total=total_single_iterations,
              desc="Generating single count/index tasks",
              unit="sample") as progress_bar:
        for bin_name in valid_bins:
            df_bin = df_bins_cache[bin_name]

            property_usage_counter = {}
            category_usage_counter = {}

            for category_name, category_properties in COUNT_MAP.items():
                if category_name == 'reaction_success':
                    continue

                available_props = [p for p in category_properties if p in df_bin.columns]
                if not available_props:
                    continue

                for sample_idx in range(n_samples_per_bin):
                    progress_bar.update(1)

                    prop_weights = []
                    for p in available_props:
                        usage = property_usage_counter.get(p, 0)
                        weight = 1.0 / (1 + usage)
                        prop_weights.append(weight)

                    prop_weights = [w / sum(prop_weights) for w in prop_weights]
                    prop = rng.choices(available_props, weights=prop_weights, k=1)[0]

                    cache_key = (bin_name, prop, 1)
                    sampled = sample_with_inverse_frequency(
                        df_bin,
                        prop,
                        n_samples=1,
                        min_count=1,
                        seed=seed + hash(f"{prop}_{sample_idx}_{bin_name}") % 1000,
                        cache=sampling_cache,
                        cache_key=cache_key
                    )

                    for _, row in sampled.iterrows():
                        # Transform SMILES for the question
                        original_smiles, transformed_smiles = transform_smiles(row['smiles'], rng)
                        molecule_key = original_smiles

                        # Track this property for this molecule
                        if molecule_key not in molecule_property_mapping:
                            molecule_property_mapping[molecule_key] = {
                                'properties': [],
                                'complexity': float(row['complexity']),
                                'complexity_bin': str(row['complexity_bin']),
                                'iupac_name': row['iupac_name'] if isinstance(row['iupac_name'], list) else [row['iupac_name']],
                                'transformed_smiles': transformed_smiles,
                                'row_data': row.to_dict()  # Store full row for multi-task use
                            }
                        molecule_property_mapping[molecule_key]['properties'].append(prop)

                        # Update usage counters
                        property_usage_counter[prop] = property_usage_counter.get(prop, 0) + 1
                        category_usage_counter[category_name] = category_usage_counter.get(category_name, 0) + 1

                        # Generate COUNT task
                        count_natural_name = get_natural_language_name(prop, 'count')
                        count_template = random.Random(seed + len(tasks)).choice(count_templates)
                        normalized_template = count_template.replace('{count_type}', '{count_types}')
                        count_question = format_count_query(
                            transformed_smiles,
                            [count_natural_name],
                            template=normalized_template,
                            include_key_hint=True,
                            key_names=[prop]
                        )

                        # Handle molecular formula specially
                        if 'molecular_formula' in prop:
                            target_value = str(row[prop])
                        else:
                            target_value = int(row[prop])

                        count_target_dict = {prop: target_value}
                        count_task = {
                            "task_type": "single_count",
                            "original_smiles": original_smiles,
                            "smiles": transformed_smiles,
                            "complexity": float(row['complexity']),
                            "complexity_bin": str(row['complexity_bin']),
                            "iupac_name": row['iupac_name'] if isinstance(row['iupac_name'], list) else [row['iupac_name']],
                            "question": count_question,
                            "category": get_property_category(prop),
                            "property": prop,
                            "target": count_target_dict,
                            "natural_language_answer": format_natural_language_answer(count_target_dict, "single_count"),
                            "supercategory": f"single_count_{get_property_category(prop)}"
                        }
                        tasks.append(count_task)

                        # Generate INDEX task using COUNT_TO_INDEX_MAP for pairing
                        if prop in COUNT_TO_INDEX_MAP:
                            index_prop = COUNT_TO_INDEX_MAP[prop]
                            if index_prop in row:
                                index_value = row[index_prop]
                                # Always create index task, even for empty lists
                                # Empty lists are valid answers (means "none present")
                                if isinstance(index_value, list):
                                    index_natural_name = get_natural_language_name(index_prop, 'index')
                                    index_template = random.Random(seed + len(tasks)).choice(index_templates)
                                    normalized_index_template = index_template.replace('{index_type}', '{index_types}')
                                    index_question = format_index_query(
                                        transformed_smiles,
                                        [index_natural_name],
                                        template=normalized_index_template,
                                        include_key_hint=True,
                                        key_names=[index_prop]
                                    )

                                    index_target_dict = {index_prop: index_value}
                                    index_task = {
                                        "task_type": "single_index",
                                        "original_smiles": original_smiles,
                                        "smiles": transformed_smiles,
                                        "complexity": float(row['complexity']),
                                        "complexity_bin": str(row['complexity_bin']),
                                        "iupac_name": row['iupac_name'] if isinstance(row['iupac_name'], list) else [row['iupac_name']],
                                        "question": index_question,
                                        "category": get_property_category(index_prop),
                                        "property": index_prop,
                                        "target": index_target_dict,
                                        "natural_language_answer": format_natural_language_answer(index_target_dict, "single_index"),
                                        "supercategory": f"single_index_{get_property_category(index_prop)}"
                                    }
                                    tasks.append(index_task)

    return tasks, molecule_property_mapping


def generate_multi_count_tasks(
    df: pd.DataFrame,
    molecule_property_mapping: Dict[str, Dict[str, Any]],
    n_samples_per_bin: int = 10,
    n_properties: List[int] = [2, 3, 5],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate multi count tasks by selecting from molecules already used in single tasks.
    Adds additional properties to create multi-property tasks.
    Ensures at most one zero value and no duplicate properties.

    Args:
        df: DataFrame with properties
        molecule_property_mapping: Dict of molecules used in single tasks with their properties
        n_samples_per_bin: Samples per combination per complexity bin
        n_properties: List of numbers of properties to combine (e.g., [2, 3, 5])
        seed: Random seed

    Returns:
        List of task dictionaries
    """
    tasks = []
    count_properties = get_count_properties(df)

    # Group molecules by complexity bin for sampling
    molecules_by_bin = {}
    for mol_smiles, mol_data in molecule_property_mapping.items():
        bin_name = mol_data['complexity_bin']
        if bin_name not in molecules_by_bin:
            molecules_by_bin[bin_name] = []
        molecules_by_bin[bin_name].append((mol_smiles, mol_data))

    # No longer need to categorize properties - we use explicit maps from column_category_map

    # Question templates
    templates = TASKS['multi_count']['question_templates']

    rng = random.Random(seed)
    total_targets = sum(
        len(n_properties) * n_samples_per_bin
        for molecules_in_bin in molecules_by_bin.values()
        if len(molecules_in_bin) > 0
    )

    if total_targets == 0:
        return tasks

    with tqdm(total=total_targets,
              desc="Generating multi-count tasks",
              unit="task") as progress_bar:
        for bin_name, molecules_in_bin in molecules_by_bin.items():
            if len(molecules_in_bin) == 0:
                continue

            # Track property and category usage per bin
            property_usage_counter = {}
            category_usage_counter = {}

            for n_props in n_properties:
                attempts = 0
                successes = 0
                max_attempts = n_samples_per_bin * 10  # Allow extra retries for constrained sampling

                while successes < n_samples_per_bin and attempts < max_attempts:
                    attempts += 1

                    # Select a molecule that was used in single tasks
                    mol_smiles, mol_data = rng.choice(molecules_in_bin)
                    row_data = mol_data['row_data']
                    existing_props = mol_data['properties']

                    # Start with properties already used in single tasks
                    selected_props = rng.sample(existing_props, min(len(existing_props), n_props))

                    # If we need more properties, add new ones using weighted selection
                    if len(selected_props) < n_props:
                        available_props = [
                            p for p in count_properties
                            if p in row_data and p not in selected_props
                        ]
                        n_needed = n_props - len(selected_props)
                        if len(available_props) >= n_needed:
                            additional_props = []
                            available_props_copy = available_props.copy()

                            for _ in range(n_needed):
                                chosen_prop = select_property_with_inverse_frequency(
                                    available_props_copy,
                                    property_usage_counter,
                                    category_usage_counter,
                                    row_data,
                                    rng,
                                    apply_zero_bias=True
                                )
                                additional_props.append(chosen_prop)
                                available_props_copy.remove(chosen_prop)

                                property_usage_counter[chosen_prop] = property_usage_counter.get(chosen_prop, 0) + 1
                                cat = get_property_category(chosen_prop)
                                category_usage_counter[cat] = category_usage_counter.get(cat, 0) + 1

                            selected_props.extend(additional_props)
                        else:
                            continue  # Can't get enough properties for this molecule

                    if len(selected_props) != n_props:
                        continue

                    zero_count = sum(1 for prop in selected_props if row_data[prop] == 0)
                    if zero_count > 1:
                        continue

                    transformed_smiles = mol_data['transformed_smiles']
                    natural_names = [get_natural_language_name(p, 'count') for p in selected_props]
                    template = rng.choice(templates)
                    # Format using the formatter function with key hints
                    question = format_count_query(
                        transformed_smiles,
                        natural_names,
                        template=template,
                        include_key_hint=True,
                        key_names=selected_props
                    )

                    target = {}
                    for prop in selected_props:
                        if 'molecular_formula' in prop:
                            target[prop] = str(row_data[prop])
                        else:
                            target[prop] = int(row_data[prop])

                    selected_categories = [get_property_category(prop) for prop in selected_props]

                    task = {
                        "task_type": "multi_count",
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data['complexity'],
                        "complexity_bin": mol_data['complexity_bin'],
                        "iupac_name": mol_data['iupac_name'],
                        "question": question,
                        "n_properties": len(selected_props),
                        "categories": selected_categories,
                        "properties": selected_props,
                        "target": target,
                        "natural_language_answer": format_natural_language_answer(target, "multi_count"),
                        "supercategory": f"multi_count_nbr_{len(selected_props)}"
                    }

                    tasks.append(task)
                    successes += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'bin': str(bin_name),
                        'n_props': n_props
                    }, refresh=False)

    return tasks


def generate_multi_index_tasks(
    df: pd.DataFrame,
    molecule_property_mapping: Dict[str, Dict[str, Any]],
    n_samples_per_bin: int = 10,
    n_properties: List[int] = [2, 3, 5],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate multi index tasks by selecting from molecules already used in single tasks.
    Adds additional index properties to create multi-property tasks.
    Ensures at most one empty index list and no duplicate properties.

    Args:
        df: DataFrame with properties
        molecule_property_mapping: Dict of molecules used in single tasks with their properties
        n_samples_per_bin: Samples per combination per complexity bin
        n_properties: List of numbers of properties to combine (e.g., [2, 3, 5])
        seed: Random seed

    Returns:
        List of task dictionaries
    """
    tasks = []
    index_properties = get_index_properties(df)

    # Group molecules by complexity bin for sampling
    molecules_by_bin = {}
    for mol_smiles, mol_data in molecule_property_mapping.items():
        bin_name = mol_data['complexity_bin']
        if bin_name not in molecules_by_bin:
            molecules_by_bin[bin_name] = []
        molecules_by_bin[bin_name].append((mol_smiles, mol_data))

    # Question templates
    templates = TASKS['multi_index_identification']['question_templates']

    rng = random.Random(seed)
    total_targets = sum(
        len(n_properties) * n_samples_per_bin
        for molecules_in_bin in molecules_by_bin.values()
        if len(molecules_in_bin) > 0
    )

    if total_targets == 0:
        return tasks

    with tqdm(total=total_targets,
              desc="Generating multi-index tasks",
              unit="task") as progress_bar:
        for bin_name, molecules_in_bin in molecules_by_bin.items():
            if len(molecules_in_bin) == 0:
                continue

            property_usage_counter = {}
            category_usage_counter = {}

            for n_props in n_properties:
                attempts = 0
                successes = 0
                max_attempts = n_samples_per_bin * 10

                while successes < n_samples_per_bin and attempts < max_attempts:
                    attempts += 1

                    mol_smiles, mol_data = rng.choice(molecules_in_bin)
                    row_data = mol_data['row_data']

                    existing_count_props = mol_data['properties']
                    existing_index_props = []
                    for count_prop in existing_count_props:
                        index_prop = count_prop.replace('_count', '_index')
                        if index_prop in row_data and index_prop in index_properties:
                            value = row_data[index_prop]
                            if isinstance(value, list) and len(value) > 0:
                                existing_index_props.append(index_prop)

                    if len(existing_index_props) == 0:
                        continue

                    selected_props = rng.sample(
                        existing_index_props,
                        min(len(existing_index_props), n_props)
                    )

                    if len(selected_props) < n_props:
                        available_props = []
                        for prop in index_properties:
                            if prop in row_data and prop not in selected_props:
                                value = row_data[prop]
                                if isinstance(value, list) and len(value) > 0:
                                    available_props.append(prop)

                        n_needed = n_props - len(selected_props)
                        if len(available_props) >= n_needed:
                            additional_props = []
                            available_props_copy = available_props.copy()

                            for _ in range(n_needed):
                                chosen_prop = select_property_with_inverse_frequency(
                                    available_props_copy,
                                    property_usage_counter,
                                    category_usage_counter,
                                    row_data,
                                    rng,
                                    apply_zero_bias=True
                                )
                                additional_props.append(chosen_prop)
                                available_props_copy.remove(chosen_prop)

                                property_usage_counter[chosen_prop] = property_usage_counter.get(chosen_prop, 0) + 1
                                cat = get_property_category(chosen_prop)
                                category_usage_counter[cat] = category_usage_counter.get(cat, 0) + 1

                            selected_props.extend(additional_props)
                        else:
                            continue

                    if len(selected_props) != n_props:
                        continue

                    empty_count = 0
                    for prop in selected_props:
                        value = row_data[prop]
                        if not isinstance(value, list) or len(value) == 0:
                            empty_count += 1

                    if empty_count > 1:
                        continue

                    transformed_smiles = mol_data['transformed_smiles']
                    natural_names = [get_natural_language_name(p, 'index') for p in selected_props]
                    template = rng.choice(templates)
                    question = format_index_query(
                        transformed_smiles,
                        natural_names,
                        template=template,
                        include_key_hint=True,
                        key_names=selected_props
                    )

                    target = {prop: row_data[prop] for prop in selected_props}
                    selected_categories = [get_property_category(prop) for prop in selected_props]

                    task = {
                        "task_type": "multi_index",
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data['complexity'],
                        "complexity_bin": mol_data['complexity_bin'],
                        "iupac_name": mol_data['iupac_name'],
                        "question": question,
                        "n_properties": len(selected_props),
                        "categories": selected_categories,
                        "properties": selected_props,
                        "target": target,
                        "natural_language_answer": format_natural_language_answer(target, "multi_index"),
                        "supercategory": f"multi_index_nbr_{len(selected_props)}"
                    }

                    tasks.append(task)
                    successes += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'bin': str(bin_name),
                        'n_props': n_props
                    }, refresh=False)

    return tasks


def generate_paired_multi_tasks(
    df: pd.DataFrame,
    molecule_property_mapping: Dict[str, Dict[str, Any]],
    n_samples_per_bin: int = 10,
    n_properties: List[int] = [2, 3, 5],
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate paired multi-count and multi-index tasks for the same molecules.
    Ensures that each molecule selected generates both count and index tasks.

    Args:
        df: DataFrame with properties
        molecule_property_mapping: Dict of molecules used in single tasks with their properties
        n_samples_per_bin: Samples per combination per complexity bin
        n_properties: List of numbers of properties to combine (e.g., [2, 3, 5])
        seed: Random seed

    Returns:
        Tuple of (count_tasks, index_tasks) - paired lists
    """
    count_tasks = []
    index_tasks = []

    count_properties = get_count_properties(df)
    index_properties = get_index_properties(df)

    # Group molecules by complexity bin for sampling
    molecules_by_bin = {}
    for mol_smiles, mol_data in molecule_property_mapping.items():
        bin_name = mol_data['complexity_bin']
        if bin_name not in molecules_by_bin:
            molecules_by_bin[bin_name] = []
        molecules_by_bin[bin_name].append((mol_smiles, mol_data))

    # Question templates
    count_templates = TASKS['multi_count']['question_templates']
    index_templates = TASKS['multi_index_identification']['question_templates']

    rng = random.Random(seed)
    total_targets = sum(
        len(n_properties) * n_samples_per_bin
        for molecules_in_bin in molecules_by_bin.values()
        if len(molecules_in_bin) > 0
    )

    if total_targets == 0:
        return count_tasks, index_tasks

    with tqdm(total=total_targets,
              desc="Generating paired multi count/index tasks",
              unit="pair") as progress_bar:
        for bin_name, molecules_in_bin in molecules_by_bin.items():
            if len(molecules_in_bin) == 0:
                continue

            property_usage_counter = {}
            category_usage_counter = {}
            used_combinations = set()

            for n_props in n_properties:
                attempts = 0
                successes = 0
                max_attempts = n_samples_per_bin * 15

                while successes < n_samples_per_bin and attempts < max_attempts:
                    attempts += 1

                    mol_smiles, mol_data = rng.choice(molecules_in_bin)
                    row_data = mol_data['row_data']
                    existing_props = mol_data['properties']

                    selected_count_props = rng.sample(
                        existing_props,
                        min(len(existing_props), n_props)
                    )

                    if len(selected_count_props) < n_props:
                        available_count_props = [
                            p for p in count_properties
                            if p in row_data and p not in selected_count_props
                        ]
                        n_needed = n_props - len(selected_count_props)
                        if len(available_count_props) >= n_needed:
                            additional_props = []
                            available_props_copy = available_count_props.copy()
                            for _ in range(n_needed):
                                chosen_prop = select_property_with_inverse_frequency(
                                    available_props_copy,
                                    property_usage_counter,
                                    category_usage_counter,
                                    row_data,
                                    rng,
                                    apply_zero_bias=True
                                )
                                additional_props.append(chosen_prop)
                                available_props_copy.remove(chosen_prop)
                            selected_count_props.extend(additional_props)
                        else:
                            continue

                    if len(selected_count_props) != n_props:
                        continue

                    combination_key = (mol_smiles, frozenset(selected_count_props))
                    if combination_key in used_combinations:
                        continue

                    zero_count = sum(1 for prop in selected_count_props if row_data[prop] == 0)
                    if zero_count > 1:
                        continue

                    selected_index_props: List[Optional[str]] = []
                    for count_prop in selected_count_props:
                        index_prop = count_prop.replace('_count', '_index')
                        if index_prop in row_data and index_prop in index_properties:
                            selected_index_props.append(index_prop)
                        else:
                            selected_index_props.append(None)

                    valid_index_props = [p for p in selected_index_props if p is not None]
                    if len(valid_index_props) < n_props // 2:
                        continue

                    final_index_props: List[str] = []
                    for index_prop in selected_index_props:
                        if index_prop is not None:
                            final_index_props.append(index_prop)
                        else:
                            available_alternatives = [
                                p for p in index_properties
                                if p in row_data and p not in final_index_props
                                and isinstance(row_data[p], list) and len(row_data[p]) > 0
                            ]
                            if available_alternatives:
                                alt_prop = select_property_with_inverse_frequency(
                                    available_alternatives,
                                    property_usage_counter,
                                    category_usage_counter,
                                    row_data,
                                    rng,
                                    apply_zero_bias=True
                                )
                                final_index_props.append(alt_prop)

                    if len(final_index_props) < n_props:
                        continue

                    empty_index_count = 0
                    for prop in final_index_props:
                        value = row_data[prop]
                        if not isinstance(value, list) or len(value) == 0:
                            empty_index_count += 1
                    if empty_index_count > 1:
                        continue

                    transformed_smiles = mol_data['transformed_smiles']

                    count_natural_names = [get_natural_language_name(p, 'count') for p in selected_count_props]
                    count_template = rng.choice(count_templates)
                    count_question = format_count_query(
                        transformed_smiles,
                        count_natural_names,
                        template=count_template,
                        include_key_hint=True,
                        key_names=selected_count_props
                    )

                    count_target = {}
                    for prop in selected_count_props:
                        count_target[prop] = str(row_data[prop]) if 'molecular_formula' in prop else int(row_data[prop])

                    count_categories = [get_property_category(prop) for prop in selected_count_props]
                    count_task = {
                        "task_type": "multi_count",
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data['complexity'],
                        "complexity_bin": mol_data['complexity_bin'],
                        "iupac_name": mol_data['iupac_name'],
                        "question": count_question,
                        "n_properties": len(selected_count_props),
                        "categories": count_categories,
                        "properties": selected_count_props,
                        "target": count_target,
                        "natural_language_answer": format_natural_language_answer(count_target, "multi_count"),
                        "supercategory": f"multi_count_nbr_{len(selected_count_props)}"
                    }

                    index_natural_names = [get_natural_language_name(p, 'index') for p in final_index_props]
                    index_template = rng.choice(index_templates)
                    index_question = format_index_query(
                        transformed_smiles,
                        index_natural_names,
                        template=index_template,
                        include_key_hint=True,
                        key_names=final_index_props
                    )

                    index_target = {prop: row_data[prop] for prop in final_index_props}
                    index_categories = [get_property_category(prop) for prop in final_index_props]
                    index_task = {
                        "task_type": "multi_index",
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data['complexity'],
                        "complexity_bin": mol_data['complexity_bin'],
                        "iupac_name": mol_data['iupac_name'],
                        "question": index_question,
                        "n_properties": len(final_index_props),
                        "categories": index_categories,
                        "properties": final_index_props,
                        "target": index_target,
                        "natural_language_answer": format_natural_language_answer(index_target, "multi_index"),
                        "supercategory": f"multi_index_nbr_{len(final_index_props)}"
                    }

                    count_tasks.append(count_task)
                    index_tasks.append(index_task)
                    successes += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'bin': str(bin_name),
                        'n_props': n_props
                    }, refresh=False)

                    used_combinations.add(combination_key)

                    for prop in selected_count_props:
                        property_usage_counter[prop] = property_usage_counter.get(prop, 0) + 1
                        cat = get_property_category(prop)
                        category_usage_counter[cat] = category_usage_counter.get(cat, 0) + 1

                    for prop in final_index_props:
                        property_usage_counter[prop] = property_usage_counter.get(prop, 0) + 1
                        cat = get_property_category(prop)
                        category_usage_counter[cat] = category_usage_counter.get(cat, 0) + 1

    return count_tasks, index_tasks


def generate_single_constraint_tasks(
    single_count_tasks: List[Dict[str, Any]],
    df: pd.DataFrame,
    n_samples_per_category: int = 10,
    seed: int = 42,
    *,
    variant_mode: str = 'exact',
    return_exact_seeds: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Generate single-constraint tasks with optional exact/flexible variants.

    Args:
        single_count_tasks: Seed count tasks to source properties from.
        df: Complete dataframe of properties.
        n_samples_per_category: Number of constraint seeds per category.
        seed: RNG seed for reproducibility.
        variant_mode: 'exact', 'flexible', or 'both'.
        return_exact_seeds: When True also return list of exact (equality) tasks.

    Returns:
        Either a list of tasks or (tasks, exact_seed_tasks).
    """
    rng = random.Random(seed)
    variant_key = variant_mode.lower()
    if variant_key not in {'exact', 'flexible', 'both'}:
        raise ValueError("variant_mode must be one of {'exact', 'flexible', 'both'}")

    include_exact = variant_key in {'exact', 'both'}
    include_flexible = variant_key in {'flexible', 'both'}

    constraint_tasks: List[Dict[str, Any]] = []
    exact_seed_tasks: List[Dict[str, Any]] = []
    percentile_cache: Dict[Tuple[str, str], np.ndarray] = {}
    constraint_column_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    count_tasks_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for task in single_count_tasks:
        count_tasks_by_category.setdefault(task['category'], []).append(task)

    print(f"  Found {len(count_tasks_by_category)} categories from count tasks")

    sampled_constraints: List[Tuple[Tuple[str, Any], Dict[str, Any]]] = []

    for category, category_tasks in count_tasks_by_category.items():
        unique_constraints: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}

        for task in category_tasks:
            property_name = task['property']
            if property_name.startswith('functional_group_') and property_name.endswith('_count'):
                constraint_key = (property_name, task['target'][property_name])
            else:
                constraint_key = (property_name, task['target'][property_name])
            unique_constraints.setdefault(constraint_key, []).append(task)

        category_samples: List[Tuple[Tuple[str, Any], Dict[str, Any]]] = []
        # Track property/value combos we have already selected for this category
        existing_constraints: Set[Tuple[str, Any]] = set(unique_constraints.keys())

        if len(unique_constraints) >= n_samples_per_category:
            sampled_keys = rng.sample(list(unique_constraints.keys()), n_samples_per_category)
        else:
            sampled_keys = list(unique_constraints.keys())

        for key in sampled_keys:
            category_samples.append((key, rng.choice(unique_constraints[key])))

        # If we still need more constraints for this category, augment directly from the dataframe
        while len(category_samples) < n_samples_per_category:
            needed = n_samples_per_category - len(category_samples)

            additional_constraints = sample_additional_constraints_for_category(
                df, category, needed, existing_constraints, rng,
                column_cache=constraint_column_cache
            )

            for constraint_data in additional_constraints:
                key = (constraint_data['property'], constraint_data['value'])
                category_samples.append((key, constraint_data))
                if len(category_samples) >= n_samples_per_category:
                    break

            if len(category_samples) >= n_samples_per_category:
                break

            # Fallback: try global sampling and filter by category
            if not additional_constraints:
                fallback_candidates = sample_additional_constraints_global(
                    df,
                    n_needed=needed,
                    existing_constraints=existing_constraints,
                    rng=rng
                )

                added_from_fallback = False
                for constraint_data in fallback_candidates:
                    prop = constraint_data['property']
                    if get_property_category(prop) != category:
                        continue

                    key = (prop, constraint_data['value'])
                    if key in existing_constraints:
                        continue

                    existing_constraints.add(key)
                    category_samples.append((key, constraint_data))
                    added_from_fallback = True

                    if len(category_samples) >= n_samples_per_category:
                        break

                if not added_from_fallback:
                    # No more unique constraints could be found for this category
                    if len(category_samples) < n_samples_per_category:
                        print(
                            f"    Warning: category '{category}' only has "
                            f"{len(category_samples)} unique constraints (target={n_samples_per_category})"
                        )
                    break

        sampled_constraints.extend(category_samples)

    special_categories: List[Tuple[str, List[str]]] = []
    if 'murcko_scaffold_value' in df.columns:
        special_categories.append(('murcko_scaffold_value', ['murcko_scaffold_value']))
    reaction_props = [col for col in df.columns if col.endswith('_success')]
    if reaction_props:
        special_categories.append(('reaction_success', reaction_props))

    for category_name, _ in special_categories:
        additional_constraints = sample_additional_constraints_for_category(
            df, category_name, n_samples_per_category, set(), rng,
            column_cache=constraint_column_cache
        )
        for constraint_data in additional_constraints:
            key = (constraint_data['property'], constraint_data['value'])
            sampled_constraints.append((key, constraint_data))
        print(f"  Added {len(additional_constraints)} constraints for special category: {category_name}")

    print(f"  Total sampled constraint tasks: {len(sampled_constraints)}")

    templates = TASKS['constraint_generation']['question_templates']

    for constraint_key, task_data in tqdm(sampled_constraints, desc="Creating single constraint tasks", unit="task"):
        property_name, raw_value = constraint_key

        if 'original_smiles' in task_data:
            original_smiles = task_data['original_smiles']
            complexity_bin = task_data['complexity_bin']
        else:
            original_smiles = task_data['smiles']
            complexity_bin = task_data['complexity_bin']

        _, transformed_smiles = transform_smiles(original_smiles, rng)
        transformation_type = 'unknown'

        constraint_property = property_name
        constraint_value = raw_value

        if property_name.startswith('functional_group_') and property_name.endswith('_count'):
            base_name = property_name.replace('_count', '')
            mapped_property = base_name + '_nbrInstances'
            if mapped_property in df.columns:
                constraint_property = mapped_property
                row_match = df[df['smiles'] == original_smiles]
                if len(row_match) > 0 and pd.notna(row_match.iloc[0][constraint_property]):
                    constraint_value = row_match.iloc[0][constraint_property]

        constraint_value = _to_python_scalar(constraint_value)

        constraint_dict = {
            'type': constraint_property,
            'operator': '=',
            'value': constraint_value
        }
        constraint_text = format_constraint(constraint_dict, use_varied_phrasing=False)
        question_template = rng.choice(templates)
        question_exact = question_template.format(constraint=constraint_text)

        if isinstance(constraint_property, str) and constraint_property.startswith('template_based_reaction_prediction_'):
            category = 'reaction_templates'
        elif constraint_property in ['murcko_scaffold_value', 'molecular_formula']:
            category = constraint_property
        else:
            category = get_property_category(constraint_property)

        # Get complexity and iupac_name from task_data if available
        complexity_val = task_data.get('complexity') if 'complexity' in task_data else None
        iupac_val = task_data.get('iupac_name') if 'iupac_name' in task_data else None

        # Determine supercategory based on property type
        if constraint_property.endswith('_nbrInstances'):
            supercategory_suffix = 'functional_group_nbrInstances'
        elif constraint_property.startswith('template_based_reaction_prediction_'):
            supercategory_suffix = 'reaction_templates'
        else:
            supercategory_suffix = category

        exact_task = {
            'task_type': 'single_constraint_generation',
            'question': question_exact,
            'category': category,
            'property': constraint_property,
            'constraints': [{
                'property': constraint_property,
                'operator': '=',
                'value': constraint_value
            }],
            'answer': transformed_smiles,
            'original_smiles': original_smiles,
            'smiles': transformed_smiles,  # Add smiles field
            'complexity': complexity_val,  # Add complexity from source task
            'complexity_bin': complexity_bin,
            'iupac_name': iupac_val,  # Add iupac_name from source task
            'transformation_type': transformation_type,
            'natural_language_answer': constraint_text,
            'supercategory': f'single_constraint_gen_exact_{supercategory_suffix}',
            # Add placeholder fields for multi-task properties
            'categories': None,  # Single tasks don't have categories (plural)
            'properties': None,  # Single tasks don't have properties (plural)
            'n_properties': None,  # Only for count/index tasks
            'target': None,  # Constraint tasks don't have targets
            'n_constraints': 1  # Single constraint task has 1 constraint
        }

        exact_seed_tasks.append(exact_task)
        if include_exact:
            constraint_tasks.append(exact_task)

        if include_flexible:
            df_bin = df[df['complexity_bin'] == complexity_bin]
            if len(df_bin) == 0:
                df_bin = df[df['complexity_bin'].astype(str) == str(complexity_bin)]
            if len(df_bin) == 0:
                df_bin = df

            flexible_task = None
            for _ in range(10):
                operator = select_random_operator(constraint_property, constraint_value, df_bin, rng)
                if operator == '=':
                    continue

                # Handle range operator differently
                if operator == 'range':
                    min_val, max_val = adjust_values_for_range_operator(
                        constraint_property,
                        constraint_value,
                        df_bin,
                        rng,
                        percentile_cache=percentile_cache,
                        cache_key=(str(complexity_bin), constraint_property)
                    )
                    min_val = _to_python_scalar(min_val)
                    max_val = _to_python_scalar(max_val)

                    variant_constraint = {
                        'property': constraint_property,
                        'operator': 'range',
                        'min_value': min_val,
                        'max_value': max_val
                    }

                    variant_dict = {
                        'type': constraint_property,
                        'operator': 'range',
                        'min_value': min_val,
                        'max_value': max_val
                    }
                else:
                    # Handle other operators (>, <, >=, <=)
                    adjusted_value = adjust_value_for_operator(
                        constraint_property,
                        constraint_value,
                        operator,
                        df_bin,
                        rng,
                        percentile_cache=percentile_cache,
                        cache_key=(str(complexity_bin), constraint_property)
                    )
                    adjusted_value = _to_python_scalar(adjusted_value)

                    variant_constraint = {
                        'property': constraint_property,
                        'operator': operator,
                        'value': adjusted_value
                    }

                    variant_dict = {
                        'type': constraint_property,
                        'operator': operator,
                        'value': adjusted_value
                    }

                if not validate_constraints(transformed_smiles, [variant_constraint]):
                    continue

                variant_text = format_constraint(variant_dict, use_varied_phrasing=False)
                variant_question = rng.choice(templates).format(constraint=variant_text)

                flexible_task = {
                    'task_type': 'single_constraint_generation',
                    'question': variant_question,
                    'category': category,
                    'property': constraint_property,
                    'constraints': [variant_constraint],
                    'answer': transformed_smiles,
                    'original_smiles': original_smiles,
                    'smiles': transformed_smiles,  # Add smiles field
                    'complexity': complexity_val,  # Use same complexity from source task
                    'complexity_bin': complexity_bin,
                    'iupac_name': iupac_val,  # Use same iupac_name from source task
                    'transformation_type': transformation_type,
                    'natural_language_answer': variant_text,
                    'supercategory': f'single_constraint_gen_flexible_{supercategory_suffix}',
                    # Add placeholder fields for multi-task properties
                    'categories': None,  # Single tasks don't have categories (plural)
                    'properties': None,  # Single tasks don't have properties (plural)
                    'n_properties': None,  # Only for count/index tasks
                    'target': None,  # Constraint tasks don't have targets
                    'n_constraints': 1  # Single constraint task has 1 constraint
                }
                break

            if flexible_task is not None:
                constraint_tasks.append(flexible_task)

    rng.shuffle(constraint_tasks)

    if include_flexible:
        exact_map: Dict[Tuple[str, str], str] = {
            (task['original_smiles'], task['property']): task['answer']
            for task in exact_seed_tasks
        }
        for task in constraint_tasks:
            if task['constraints'][0]['operator'] != '=':
                key = (task['original_smiles'], task['property'])
                expected = exact_map.get(key)
                if expected is None:
                    raise ValueError(
                        f"No matching exact constraint found for flexible variant {key}"
                    )
                if canonicalize_smiles(task['answer']) != canonicalize_smiles(expected):
                    raise ValueError(
                        f"Flexible constraint answer mismatch for {key}: {task['answer']} vs {expected}"
                    )

    if return_exact_seeds:
        return constraint_tasks, exact_seed_tasks

    return constraint_tasks


def sample_additional_constraints_global(
    df: pd.DataFrame,
    n_needed: int,
    existing_constraints: Set[Tuple[str, Any]],
    rng: random.Random
) -> List[Dict[str, Any]]:
    """
    Sample additional constraint tasks globally using inverse frequency weighting.

    Args:
        df: Full dataframe
        n_needed: Number of additional constraints needed
        existing_constraints: Set of (property, value) tuples to avoid
        rng: Random number generator

    Returns:
        List of constraint data dictionaries
    """
    # Get all constraint-suitable properties
    constraint_properties = []

    # 1. Count properties (excluding reaction templates for count, but we'll add them separately)
    count_properties = get_count_properties(df, include_reactions=False)
    # For functional groups, use nbrInstances instead of count for constraints
    for prop in count_properties:
        if prop.startswith('functional_group_') and prop.endswith('_count'):
            # Replace with nbrInstances version
            base_name = prop.replace('_count', '')
            nbrInstances_prop = base_name + '_nbrInstances'
            if nbrInstances_prop in df.columns:
                constraint_properties.append(nbrInstances_prop)
        else:
            constraint_properties.append(prop)

    # 2. Reaction success properties (boolean constraints)
    reaction_properties = [col for col in df.columns
                          if col.startswith('template_based_reaction_prediction_')
                          and col.endswith('_success')]
    constraint_properties.extend(reaction_properties)

    # 3. Special value properties (string constraints)
    if 'murcko_scaffold_value' in df.columns:
        constraint_properties.append('murcko_scaffold_value')
    if 'molecular_formula' in df.columns:
        constraint_properties.append('molecular_formula')

    # Track property usage for inverse frequency
    property_usage_counter = {}
    category_usage_counter = {}

    additional_constraints = []
    attempts = 0
    max_attempts = n_needed * 30

    while len(additional_constraints) < n_needed and attempts < max_attempts:
        attempts += 1

        # Sample a molecule from entire dataset
        row = df.sample(1, random_state=rng.randint(0, 2**32-1)).iloc[0]

        # Select property with inverse frequency
        available_props = [p for p in constraint_properties if p in row and pd.notna(row[p])]
        if not available_props:
            continue

        # For reaction and special properties, don't use the weighted selection
        # since they're not in the category system
        reaction_props = [p for p in available_props if p.startswith('template_based_reaction_prediction_')]
        special_props = [p for p in available_props if p in ['murcko_scaffold_value', 'molecular_formula']]
        regular_props = [p for p in available_props if p not in reaction_props and p not in special_props]

        # Mix property types: 60% regular, 30% reactions, 10% special
        prop_type_choice = rng.random()
        if prop_type_choice < 0.6 and regular_props:
            # Regular properties with inverse frequency
            property_name = select_property_with_inverse_frequency(
                regular_props,
                property_usage_counter,
                category_usage_counter,
                row,
                rng,
                apply_zero_bias=True
            )
        elif prop_type_choice < 0.9 and reaction_props:
            # Reaction properties - uniform sampling
            property_name = rng.choice(reaction_props)
        elif special_props:
            # Special properties - uniform sampling
            property_name = rng.choice(special_props)
        elif regular_props:
            # Fallback to regular if others not available
            property_name = select_property_with_inverse_frequency(
                regular_props,
                property_usage_counter,
                category_usage_counter,
                row,
                rng,
                apply_zero_bias=True
            )
        else:
            continue

        value = row[property_name]

        # Convert numpy types to Python types for JSON serialization
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            value = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            value = float(value)
        elif isinstance(value, np.bool_):
            value = bool(value)

        # Check uniqueness
        constraint_key = (property_name, value)
        if constraint_key in existing_constraints:
            continue

        # Create constraint data
        constraint_data = {
            'property': property_name,
            'value': value,
            'smiles': row['smiles'],
            'complexity_bin': str(row['complexity_bin'])  # Ensure complexity_bin is string
        }

        additional_constraints.append(constraint_data)
        existing_constraints.add(constraint_key)

        # Update usage counters
        property_usage_counter[property_name] = property_usage_counter.get(property_name, 0) + 1

        # Get category (handle special properties)
        if property_name.startswith('template_based_reaction_prediction_'):
            category = 'reaction_templates'
        elif property_name in ['murcko_scaffold_value', 'molecular_formula']:
            category = 'special_values'
        else:
            category = get_property_category(property_name)

        category_usage_counter[category] = category_usage_counter.get(category, 0) + 1

    if len(additional_constraints) < n_needed:
        print(f"    Warning: Could only find {len(additional_constraints)} unique constraints out of {n_needed} requested")

    return additional_constraints


def _get_constraint_property_cache(
    df: pd.DataFrame,
    property_name: str,
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cached non-null values and positional indices for a property."""
    cached = cache.get(property_name)
    if cached is not None:
        return cached

    if property_name not in df.columns:
        entry = (np.array([], dtype=object), np.array([], dtype=int))
        cache[property_name] = entry
        return entry

    series = df[property_name]
    mask = series.notna()
    mask_np = mask.to_numpy()
    if not mask_np.any():
        entry = (np.array([], dtype=object), np.array([], dtype=int))
        cache[property_name] = entry
        return entry

    values = series.to_numpy()
    positions = np.nonzero(mask_np)[0].astype(int)
    python_values = np.empty(len(positions), dtype=object)
    for idx, pos in enumerate(positions):
        python_values[idx] = _to_python_scalar(values[pos])

    entry = (python_values, positions)
    cache[property_name] = entry
    return entry


def sample_additional_constraints_for_category(
    df: pd.DataFrame,
    category: str,
    n_needed: int,
    existing_constraints: Set[Tuple[str, Any]],
    rng: random.Random,
    *,
    column_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
) -> List[Dict[str, Any]]:
    """
    Sample additional constraint tasks for a specific category.

    Args:
        df: Full dataframe
        category: Category to sample from
        n_needed: Number of additional constraints needed
        existing_constraints: Set of (property, value) tuples to avoid
        rng: Random number generator

    Returns:
        List of constraint data dictionaries
    """
    # Get properties for this category from the category maps
    if category in COUNT_MAP:
        category_properties = COUNT_MAP[category]
    elif category in CONSTRAINT_MAP:
        category_properties = CONSTRAINT_MAP[category]
    elif category == 'murcko_scaffold_value':
        # Special case for murcko scaffold value
        category_properties = ['murcko_scaffold_value'] if 'murcko_scaffold_value' in df.columns else []
    elif category == 'reaction_success':
        # Special case for reaction success properties
        category_properties = [col for col in df.columns if col.endswith('_success')]
    elif category == 'functional_group_instances':
        category_properties = [
            col for col in df.columns
            if col.startswith('functional_group_') and col.endswith('_nbrInstances')
        ]
    else:
        # Fall back to searching for properties by category
        category_properties = []
        for prop in df.columns:
            if get_property_category(prop) == category:
                category_properties.append(prop)

    if not category_properties:
        return []

    if column_cache is None:
        column_cache = {}

    additional_constraints = []
    max_attempts = n_needed * 10

    for _ in range(max_attempts):
        # Select a random property from this category
        property_name = rng.choice(category_properties)

        # Handle functional groups specially
        if property_name.startswith('functional_group_') and property_name.endswith('_count'):
            # For functional groups, use nbrInstances for constraints
            base_name = property_name.replace('_count', '')
            constraint_property = base_name + '_nbrInstances'
            if constraint_property in df.columns:
                property_name = constraint_property

        if property_name not in df.columns:
            continue

        values_array, positions_array = _get_constraint_property_cache(df, property_name, column_cache)
        if len(values_array) == 0:
            continue

        sample_idx = rng.randint(0, len(values_array) - 1)
        value = values_array[sample_idx]

        # Check if this constraint already exists
        constraint_key = (property_name, value)
        if constraint_key in existing_constraints:
            continue

        row_position = int(positions_array[sample_idx])
        row = df.iloc[row_position]

        constraint_data = {
            'property': property_name,
            'value': value,
            'smiles': row['smiles'],
            'complexity_bin': str(row['complexity_bin'])
        }

        additional_constraints.append(constraint_data)
        existing_constraints.add(constraint_key)

        if len(additional_constraints) >= n_needed:
            break

    return additional_constraints


def _apply_single_constraint(series: pd.Series, constraint: Dict[str, Any]) -> pd.Series:
    operator = constraint['operator']
    if operator == '=':
        return series == constraint['value']
    if operator == '>':
        return series > constraint['value']
    if operator == '<':
        return series < constraint['value']
    if operator == '>=':
        return series >= constraint['value']
    if operator == '<=':
        return series <= constraint['value']
    if operator == 'range':
        low = constraint.get('min_value')
        high = constraint.get('max_value')
        if low is None or high is None:
            raise ValueError("range constraint requires min_value and max_value")
        return (series >= low) & (series <= high)
    raise ValueError(f"Unsupported operator: {operator}")


def _match_rate(df: pd.DataFrame, constraints: List[Dict[str, Any]]) -> float:
    if len(df) == 0:
        return 0.0

    mask = pd.Series(True, index=df.index)
    for constraint in constraints:
        prop = constraint['property']
        if prop not in df.columns:
            return 0.0
        mask &= _apply_single_constraint(df[prop], constraint)
        if not mask.any():
            return 0.0

    return float(mask.mean())


def _is_zero_value(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 0.0
    if isinstance(value, (bool, np.bool_)):
        return value is False
    return False


def _category_for_prop(prop: str) -> str:
    if prop.startswith('template_based_reaction_prediction_') and prop.endswith('_success'):
        return 'reaction_templates'
    if prop in {'murcko_scaffold_value', 'molecular_formula'}:
        return prop
    return get_property_category(prop)


def _format_constraints_nl(
    templates: List[str],
    constraints: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[str, str]:
    parts = []
    for constraint in constraints:
        data = {'type': constraint['property'], 'operator': constraint['operator']}
        if constraint['operator'] == 'range':
            data['min_value'] = constraint['min_value']
            data['max_value'] = constraint['max_value']
        else:
            data['value'] = constraint['value']
        parts.append(format_constraint(data, use_varied_phrasing=False))

    if len(parts) == 1:
        constraint_text = parts[0]
    elif len(parts) == 2:
        constraint_text = f"{parts[0]} and {parts[1]}"
    else:
        constraint_text = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    question = rng.choice(templates).format(constraint=constraint_text)
    return question, constraint_text


def _map_count_prop_to_constraint(prop: str, row: pd.Series, df: pd.DataFrame) -> Tuple[str, Any]:
    if prop.startswith('functional_group_') and prop.endswith('_count'):
        base = prop.replace('_count', '')
        alt = base + '_nbrInstances'
        if alt in df.columns and pd.notna(row.get(alt, np.nan)):
            return alt, _to_python_scalar(row[alt])
    return prop, _to_python_scalar(row[prop])


def _hardest_subset_k(
    exact_constraints: List[Dict[str, Any]],
    k: int,
    df_global: pd.DataFrame,
    *,
    zero_limit: int,
    require_diversity: bool,
    rng: random.Random,
    tries: int = 120
) -> Optional[List[Dict[str, Any]]]:
    if len(exact_constraints) < k:
        return None

    indices = list(range(len(exact_constraints)))
    categories = [_category_for_prop(c['property']) for c in exact_constraints]
    zero_flags = [_is_zero_value(c['value']) for c in exact_constraints]

    best_choice: Optional[List[Dict[str, Any]]] = None
    best_rate = float('inf')

    for _ in range(tries):
        rng.shuffle(indices)
        picked: List[Dict[str, Any]] = []
        used_categories: Set[str] = set()
        zero_count = 0

        for idx in indices:
            if len(picked) == k:
                break

            constraint = exact_constraints[idx]
            category = categories[idx]
            is_zero = zero_flags[idx]

            if require_diversity and category in used_categories:
                continue
            if is_zero and zero_count >= zero_limit:
                continue

            picked.append(constraint)
            used_categories.add(category)
            if is_zero:
                zero_count += 1

        if len(picked) != k:
            continue

        rate = _match_rate(df_global, picked)
        if 0.0 < rate < best_rate:
            best_rate = rate
            best_choice = [dict(c) for c in picked]

    return best_choice


def generate_multi_constraint_generation_tasks_exact_global_pool(
    multi_count_tasks: List[Dict[str, Any]],
    single_constraint_tasks: List[Dict[str, Any]],
    df: pd.DataFrame,
    n_per_k: Dict[int, int],
    seed: int = 42,
    *,
    target_rate_min: float = 0.0,
    target_rate_max: float = 0.03,
    only_from_multicount: bool = True,
    include_special: bool = True,
    include_extra_counts: bool = True,
    require_category_diversity: bool = True,
    zero_limit: int = 1,
    additional_special_samples: int = 90
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    templates = TASKS['constraint_generation']['question_templates']

    df_by_smiles = {smiles: row for smiles, row in df.set_index('smiles').iterrows()}

    if only_from_multicount:
        smiles_pool: Set[str] = {task['original_smiles'] for task in multi_count_tasks}
    else:
        smiles_pool = set(df['smiles'].tolist())

    # Include molecules appearing in the single-constraint seeds even if they weren't used in multi-count tasks
    for task in single_constraint_tasks:
        smiles = task.get('original_smiles') or task.get('smiles')
        if smiles:
            smiles_pool.add(smiles)

    reaction_columns = [
        col for col in df.columns
        if col.startswith('template_based_reaction_prediction_') and col.endswith('_success')
    ]
    all_count_props = get_count_properties(df, include_reactions=False)

    multi_count_by_smiles: Dict[str, List[Dict[str, Any]]] = {}
    for task in multi_count_tasks:
        multi_count_by_smiles.setdefault(task['original_smiles'], []).append(task)

    single_constraints_by_smiles: Dict[str, List[Dict[str, Any]]] = {}
    for task in single_constraint_tasks:
        constraints = task.get('constraints') or []
        if not constraints:
            continue
        # Single tasks should have one constraint; skip non-exact seeds
        constraint = constraints[0]
        if constraint.get('operator') != '=':
            continue
        smiles = task.get('original_smiles') or task.get('smiles')
        if not smiles:
            continue
        prop = constraint.get('property')
        if not prop:
            continue
        value = constraint.get('value')
        single_constraints_by_smiles.setdefault(smiles, []).append({
            'property': prop,
            'operator': '=',
            'value': _to_python_scalar(value)
        })

    if include_special and additional_special_samples > 0:
        existing_special_keys: Set[Tuple[str, Any]] = set(
            (constraint['property'], constraint['value'])
            for constraints in single_constraints_by_smiles.values()
            for constraint in constraints
        )

        column_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        special_category_names = [
            'murcko_scaffold_value',
            'molecular_formula',
            'reaction_success',
            'functional_group_instances'
        ]

        for category_name in special_category_names:
            extra_constraints = sample_additional_constraints_for_category(
                df,
                category_name,
                additional_special_samples,
                existing_special_keys,
                rng,
                column_cache=column_cache
            )

            for constraint_data in extra_constraints:
                prop = constraint_data['property']
                value = _to_python_scalar(constraint_data['value'])
                smiles = constraint_data['smiles']
                existing_special_keys.add((prop, value))
                single_constraints_by_smiles.setdefault(smiles, []).append({
                    'property': prop,
                    'operator': '=',
                    'value': value
                })
                smiles_pool.add(smiles)

    def _constraints_for_smiles(smiles: str) -> List[Dict[str, Any]]:
        row = df_by_smiles.get(smiles)
        if row is None:
            return []

        seen: Set[str] = set()
        candidates: List[Dict[str, Any]] = []

        for constraint in single_constraints_by_smiles.get(smiles, []):
            prop = constraint['property']
            if prop not in df.columns or prop in seen:
                continue
            if pd.isna(row.get(prop, np.nan)):
                continue
            candidates.append(dict(constraint))
            seen.add(prop)

        for task in multi_count_by_smiles.get(smiles, []):
            for prop in task.get('properties', []):
                if prop in row.index and pd.notna(row[prop]):
                    mapped_prop, mapped_value = _map_count_prop_to_constraint(prop, row, df)
                    if mapped_prop not in df.columns or mapped_prop in seen or pd.isna(mapped_value):
                        continue
                    candidates.append({
                        'property': mapped_prop,
                        'operator': '=',
                        'value': _to_python_scalar(mapped_value)
                    })
                    seen.add(mapped_prop)

        if include_extra_counts:
            for prop in all_count_props:
                if prop in row.index and pd.notna(row[prop]):
                    mapped_prop, mapped_value = _map_count_prop_to_constraint(prop, row, df)
                    if mapped_prop not in df.columns or mapped_prop in seen or pd.isna(mapped_value):
                        continue
                    candidates.append({
                        'property': mapped_prop,
                        'operator': '=',
                        'value': _to_python_scalar(mapped_value)
                    })
                    seen.add(mapped_prop)

        if include_special:
            if 'murcko_scaffold_value' in df.columns:
                value = row.get('murcko_scaffold_value', np.nan)
                if not pd.isna(value) and 'murcko_scaffold_value' not in seen:
                    candidates.append({
                        'property': 'murcko_scaffold_value',
                        'operator': '=',
                        'value': _to_python_scalar(value)
                    })
                    seen.add('murcko_scaffold_value')

            if 'molecular_formula' in df.columns:
                value = row.get('molecular_formula', np.nan)
                if not pd.isna(value) and 'molecular_formula' not in seen:
                    candidates.append({
                        'property': 'molecular_formula',
                        'operator': '=',
                        'value': _to_python_scalar(value)
                    })
                    seen.add('molecular_formula')

            for col in reaction_columns:
                value = row.get(col, np.nan)
                if not pd.isna(value) and col not in seen:
                    candidates.append({
                        'property': col,
                        'operator': '=',
                        'value': _to_python_scalar(value)
                    })
                    seen.add(col)

        return candidates

    tasks: List[Dict[str, Any]] = []
    used_keys: Set[Tuple[str, Tuple[str, ...]]] = set()

    total_exact_targets = sum(n_per_k.values())
    progress = tqdm(total=total_exact_targets, desc="Generating exact multi-constraint tasks", unit="task") if total_exact_targets > 0 else None

    try:
        for k, target_total in sorted(n_per_k.items()):
            if k < 2:
                raise ValueError("k must be >= 2 for multi-constraint generation")

            produced = 0
            attempts = 0
            max_attempts = max(5000, target_total * 120)

            smiles_list = list(smiles_pool)
            rng.shuffle(smiles_list)

            while produced < target_total and attempts < max_attempts:
                attempts += 1
                smiles = rng.choice(smiles_list)
                row = df_by_smiles.get(smiles)
                if row is None:
                    continue

                candidates = _constraints_for_smiles(smiles)
                if len(candidates) < k:
                    continue

                pick = _hardest_subset_k(
                    candidates,
                    k,
                    df,
                    zero_limit=zero_limit,
                    require_diversity=require_category_diversity,
                    rng=rng
                )
                if pick is None:
                    continue

                rate = _match_rate(df, pick)
                if rate == 0.0 or rate < target_rate_min or rate > target_rate_max:
                    continue

                properties_sorted = tuple(sorted(constraint['property'] for constraint in pick))
                combo_key = (smiles, properties_sorted)
                if combo_key in used_keys:
                    continue

                _, transformed_smiles = transform_smiles(smiles, rng)
                if not validate_constraints(transformed_smiles, pick):
                    continue

                categories = [_category_for_prop(constraint['property']) for constraint in pick]
                question, natural_language = _format_constraints_nl(templates, pick, rng)

                complexity_value = float(row['complexity']) if 'complexity' in row and not pd.isna(row['complexity']) else None
                complexity_bin = str(row['complexity_bin']) if 'complexity_bin' in row and not pd.isna(row['complexity_bin']) else None
                iupac_raw = row.get('iupac_name') if 'iupac_name' in row else None
                if isinstance(iupac_raw, list):
                    iupac_value = iupac_raw
                elif pd.isna(iupac_raw) if isinstance(iupac_raw, (float, np.floating)) else iupac_raw is None:
                    iupac_value = None
                else:
                    iupac_value = [iupac_raw]

                task = {
                    'task_type': 'multi_constraint_generation',
                    'question': question,
                    'n_constraints': k,
                    'constraints': [dict(constraint) for constraint in pick],
                    'answer': transformed_smiles,
                    'original_smiles': smiles,
                    'smiles': transformed_smiles,
                    'complexity': complexity_value,
                    'complexity_bin': complexity_bin,
                    'iupac_name': iupac_value,
                    'transformation_type': None,
                    'categories': categories,
                    'properties': [constraint['property'] for constraint in pick],
                    'natural_language_answer': natural_language,
                    'supercategory': f'multi_constraint_generation_exact_global_nbr_{k}',
                    'category': None,
                    'property': None,
                    'target': None,
                    'n_properties': None
                }

                tasks.append(task)
                used_keys.add(combo_key)
                produced += 1
                if progress is not None:
                    progress.update(1)

        if produced < target_total:
            raise RuntimeError(
                f"GLOBAL exact-only: built {produced}/{target_total} tasks for k={k}. "
                f"Consider relaxing target_rate_max={target_rate_max} or diversity constraints."
            )
    finally:
        if progress is not None:
            progress.close()

    random.Random(seed).shuffle(tasks)
    return tasks


def generate_multi_constraint_generation_tasks(
    multi_count_tasks: List[Dict[str, Any]],
    single_constraint_tasks: List[Dict[str, Any]],
    df: pd.DataFrame,
    n_samples_per_combination: Union[int, Dict[int, int]] = 100,
    n_constraints: List[int] = [2, 3, 5],
    seed: int = 42,
    *,
    variant_mode: str = 'exact',
    target_rate_min: float = 0.0,
    target_rate_max: float = 0.03,
    only_from_multicount: bool = True,
    include_special: bool = True,
    include_extra_counts: bool = True,
    require_category_diversity: bool = True,
    zero_limit: int = 1,
    additional_special_samples: int = 90
) -> List[Dict[str, Any]]:
    """Generate global exact multi-constraint tasks and optional flexible variants.

    Builds a candidate pool per molecule using both the paired multi-count tasks and
    the exact single-constraint seeds, optionally augmenting special properties with
    `additional_special_samples` extra (property, value) pairs before selecting hard
    combinations with low global match rates. Flexible variants are derived from the
    exact seeds when requested.

    Args:
        multi_count_tasks: Paired multi-count tasks used to source count properties.
        single_constraint_tasks: Exact single-constraint seeds (one per property/value)
            that should always be considered in the candidate pool.
        df: Property dataframe.
        n_samples_per_combination: Target number of exact tasks per constraint cardinality.
        n_constraints: Constraint counts to generate (e.g., [2, 3, 5]).
        seed: RNG seed.
        variant_mode: 'exact', 'flexible', or 'both'.
        target_rate_min/target_rate_max: Bounds on the global match rate used when
            selecting constraint subsets (smaller rate ⇒ harder task).
        only_from_multicount: When True, start from the union of molecules seen in
            multi-count tasks plus any from the single-constraint seeds. When False,
            allow any molecule in the dataframe.
        include_special/include_extra_counts: Control whether scaffold/formula/reaction
            constraints and unused count properties are eligible additions.
        require_category_diversity: Enforce distinct categories within each subset.
        zero_limit: Maximum number of zero-valued constraints per subset.
        additional_special_samples: Extra number of special-category constraints to
            sample globally (per category) before building the multi-constraint pool.
    """

    variant_key = variant_mode.lower()
    if variant_key not in {'exact', 'flexible', 'both'}:
        raise ValueError("variant_mode must be one of {'exact', 'flexible', 'both'}")

    rng = random.Random(seed)
    if isinstance(n_samples_per_combination, dict):
        n_per_k = {k: n_samples_per_combination.get(k, 0) for k in n_constraints}
    else:
        n_per_k = {k: n_samples_per_combination for k in n_constraints}

    exact_tasks = generate_multi_constraint_generation_tasks_exact_global_pool(
        multi_count_tasks,
        single_constraint_tasks,
        df,
        n_per_k,
        seed=seed,
        target_rate_min=target_rate_min,
        target_rate_max=target_rate_max,
        only_from_multicount=only_from_multicount,
        include_special=include_special,
        include_extra_counts=include_extra_counts,
        require_category_diversity=require_category_diversity,
        zero_limit=zero_limit,
        additional_special_samples=additional_special_samples
    )

    include_exact = variant_key in {'exact', 'both'}
    include_flexible = variant_key in {'flexible', 'both'}

    tasks: List[Dict[str, Any]] = []
    if include_exact:
        tasks.extend(exact_tasks)

    if include_flexible:
        percentile_cache: Dict[Tuple[str, str], np.ndarray] = {}
        expected_map = {
            (task['original_smiles'], tuple(sorted(task['properties']))): task['answer']
            for task in exact_tasks
        }

        templates = TASKS['constraint_generation']['question_templates']
        flexible_tasks: List[Dict[str, Any]] = []

        for exact_task in exact_tasks:
            base_constraints = exact_task['constraints']
            bin_name = exact_task.get('complexity_bin')
            if bin_name is not None and 'complexity_bin' in df.columns:
                df_bin = df[df['complexity_bin'].astype(str) == str(bin_name)]
            else:
                df_bin = df
            if len(df_bin) == 0:
                df_bin = df

            variant: Optional[List[Dict[str, Any]]] = None

            for _ in range(10):
                candidate_constraints: List[Dict[str, Any]] = []
                diff_operator = False
                valid = True

                for base_constraint in base_constraints:
                    prop = base_constraint['property']
                    base_value = base_constraint.get('value')

                    operator = '='
                    for _ in range(5):
                        operator = select_random_operator(prop, base_value, df_bin, rng)
                        if operator != '=':
                            break

                    if operator == '=':
                        valid = False
                        break

                    cache_key = (str(bin_name), prop)

                    if operator == 'range':
                        min_val, max_val = adjust_values_for_range_operator(
                            prop,
                            base_value,
                            df_bin,
                            rng,
                            percentile_cache=percentile_cache,
                            cache_key=cache_key
                        )
                        min_val = _to_python_scalar(min_val)
                        max_val = _to_python_scalar(max_val)
                        candidate_constraints.append({
                            'property': prop,
                            'operator': 'range',
                            'min_value': min_val,
                            'max_value': max_val
                        })
                    else:
                        adjusted_value = adjust_value_for_operator(
                            prop,
                            base_value,
                            operator,
                            df_bin,
                            rng,
                            percentile_cache=percentile_cache,
                            cache_key=cache_key
                        )
                        adjusted_value = _to_python_scalar(adjusted_value)
                        candidate_constraints.append({
                            'property': prop,
                            'operator': operator,
                            'value': adjusted_value
                        })

                    if operator != '=':
                        diff_operator = True

                if not valid or not diff_operator or len(candidate_constraints) != len(base_constraints):
                    continue

                if validate_constraints(exact_task['answer'], candidate_constraints):
                    variant = candidate_constraints
                    break

            if variant is None:
                continue

            question, natural_text = _format_constraints_nl(templates, variant, rng)

            flexible_task = {
                **{k: exact_task[k] for k in exact_task if k not in {'question', 'constraints', 'natural_language_answer', 'supercategory'}},
                'task_type': 'multi_constraint_generation',
                'question': question,
                'constraints': variant,
                'natural_language_answer': natural_text,
                'supercategory': f"multi_constraint_generation_flexible_nbr_{len(variant)}"
            }
            flexible_tasks.append(flexible_task)

        for flex_task in flexible_tasks:
            key = (flex_task['original_smiles'], tuple(sorted(flex_task['properties'])))
            expected = expected_map.get(key)
            if expected is None:
                raise ValueError(f"No matching exact multi-constraint found for flexible variant {key}")
            if canonicalize_smiles(flex_task['answer']) != canonicalize_smiles(expected):
                raise ValueError(
                    f"Flexible multi-constraint answer mismatch for {key}: {flex_task['answer']} vs {expected}"
                )

        tasks.extend(flexible_tasks)

    random.Random(seed).shuffle(tasks)

    if variant_key == 'flexible':
        return [task for task in tasks if task['supercategory'].startswith('multi_constraint_generation_flexible')]

    return tasks


def select_random_operator(property_name: str, value: Any, df: pd.DataFrame, rng: random.Random) -> str:
    """
    Select a random operator for a constraint.

    Args:
        property_name: The property name
        value: The current value
        df: Dataframe to check value distribution
        rng: Random number generator

    Returns:
        Selected operator string
    """
    # Boolean properties only support equality
    if property_name.startswith('template_based_reaction_prediction_') and property_name.endswith('_success'):
        return '='

    # String/categorical properties only support equality
    if property_name in ['murcko_scaffold_value', 'murcko_scaffold_count', 'molecular_formula_count', 'molecular_formula']:
        return '='

    # Reaction success properties (any reaction template success)
    if '_success' in property_name:
        return '='

    # Numeric properties - handle zero values specially
    if isinstance(value, (int, float, np.integer, np.floating)) and value == 0:
        # For zero values, only = and > make sense
        # >= 0 is meaningless (everything satisfies it)
        # <= 0 means "exactly 0" so just use =
        # < 0 is impossible for counts
        # range with min=0 could work but is similar to >
        operators = ['=', '>']
    else:
        # For non-zero numeric properties, all operators make sense including range
        operators = ['=', '>', '<', '>=', '<=', 'range']

    return rng.choice(operators)


def adjust_value_for_operator(
    property_name: str,
    value: Any,
    operator: str,
    df: pd.DataFrame,
    rng: random.Random,
    *,
    percentile_cache: Optional[Dict[Any, np.ndarray]] = None,
    cache_key: Optional[Any] = None
) -> Any:
    """
    Adjust value based on operator to ensure some matches exist.

    Args:
        property_name: The property name
        value: The original value
        operator: The selected operator
        df: Dataframe to check value distribution
        rng: Random number generator

    Returns:
        Adjusted value
    """
    # No adjustment needed for equality
    if operator == '=':
        return value

    # No adjustment for non-numeric properties
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value

    # Get value distribution from dataframe
    if property_name not in df.columns:
        return value

    cached_sorted = None
    if percentile_cache is not None and cache_key is not None:
        cached_sorted = percentile_cache.get(cache_key)

    if cached_sorted is not None:
        values = cached_sorted
    else:
        values = df[property_name].dropna()
        if len(values) == 0:
            return value
        values = np.sort(values.values if isinstance(values, pd.Series) else np.array(values))
        if percentile_cache is not None and cache_key is not None:
            percentile_cache[cache_key] = values

    # For > and >=, we want a smaller value to ensure matches
    # MORE EXTREME: Use much lower percentiles for more challenging constraints
    if operator in ['>', '>=']:
        # Use a much lower percentile for more extreme constraints
        percentile = rng.uniform(2, 30)  # Was 10-50, now 2-30
        threshold = np.percentile(values, percentile)
        return type(value)(threshold)

    # For < and <=, we want a larger value to ensure matches
    # MORE EXTREME: Use much higher percentiles for more challenging constraints
    if operator in ['<', '<=']:
        # Use a much higher percentile for more extreme constraints
        percentile = rng.uniform(70, 98)  # Was 50-90, now 70-98
        threshold = np.percentile(values, percentile)
        return type(value)(threshold)

    return value


def adjust_values_for_range_operator(
    property_name: str,
    actual_value: Any,
    df: pd.DataFrame,
    rng: random.Random,
    *,
    percentile_cache: Optional[Dict[Any, np.ndarray]] = None,
    cache_key: Optional[Any] = None
) -> Tuple[Any, Any]:
    """
    Generate min_value and max_value for a range constraint.

    Args:
        property_name: The property name
        actual_value: The actual value that must be included in the range
        df: Dataframe to check value distribution
        rng: Random number generator
        percentile_cache: Optional cache for sorted values
        cache_key: Optional cache key

    Returns:
        Tuple of (min_value, max_value) where min_value <= actual_value <= max_value
    """
    # For non-numeric properties, can't create a range
    if not isinstance(actual_value, (int, float, np.integer, np.floating)):
        return actual_value, actual_value

    # Get value distribution from dataframe
    if property_name not in df.columns:
        # If no distribution data, create a simple range around the value
        if actual_value == 0:
            return 0, max(1, int(actual_value + 5))
        offset = max(1, abs(actual_value) * 0.3)
        return type(actual_value)(actual_value - offset), type(actual_value)(actual_value + offset)

    # Get cached or compute sorted values
    cached_sorted = None
    if percentile_cache is not None and cache_key is not None:
        cached_sorted = percentile_cache.get(cache_key)

    if cached_sorted is not None:
        values = cached_sorted
    else:
        values = df[property_name].dropna()
        if len(values) == 0:
            # No data, create simple range
            if actual_value == 0:
                return 0, max(1, int(actual_value + 5))
            offset = max(1, abs(actual_value) * 0.3)
            return type(actual_value)(actual_value - offset), type(actual_value)(actual_value + offset)
        values = np.sort(values.values if isinstance(values, pd.Series) else np.array(values))
        if percentile_cache is not None and cache_key is not None:
            percentile_cache[cache_key] = values

    # Find the percentile of the actual value
    actual_percentile = (values < actual_value).sum() / len(values) * 100

    # Create range based on where the actual value falls in the distribution
    # Using TIGHTER ranges for more challenging constraints (harder = fewer molecules satisfy)
    if actual_value == 0:
        # For zero values, create a tight range from 0 to a low percentile
        # Tighter range = harder constraint
        upper_percentile = rng.uniform(5, 15)  # Very tight upper bound
        max_val = np.percentile(values, upper_percentile)
        # Ensure max_val > 0 for a meaningful range
        max_val = max(1, max_val)
        return 0, type(actual_value)(max_val)

    elif actual_percentile < 30:
        # Value is in lower third - create tight range around the actual value
        # Tight window makes it challenging
        lower_percentile = max(0, actual_percentile - rng.uniform(2, 5))  # Very close below
        upper_percentile = min(100, actual_percentile + rng.uniform(5, 10))  # Small window above
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)
        min_val = min(min_val, actual_value)  # Ensure actual_value is included
        max_val = max(max_val, actual_value)

    elif actual_percentile > 70:
        # Value is in upper third - create tight range around the actual value
        # Tight window for challenging constraint
        lower_percentile = max(0, actual_percentile - rng.uniform(5, 10))  # Small window below
        upper_percentile = min(100, actual_percentile + rng.uniform(2, 5))  # Very close above
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)
        min_val = min(min_val, actual_value)  # Ensure actual_value is included
        max_val = max(max_val, actual_value)

    else:
        # Value is in middle - create NARROW range for harder constraint
        spread = rng.uniform(8, 15)  # Much tighter spread (was 40-60, now 8-15)
        lower_percentile = max(0, actual_percentile - spread/2)
        upper_percentile = min(100, actual_percentile + spread/2)
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)

    # Ensure the range includes the actual value with some buffer
    min_val = min(min_val, actual_value)
    max_val = max(max_val, actual_value)

    # Ensure min < max (add small buffer if they're equal)
    if min_val >= max_val:
        if actual_value == 0:
            max_val = max(1, min_val + 1)
        else:
            buffer = max(1, abs(actual_value) * 0.1)
            min_val = actual_value - buffer
            max_val = actual_value + buffer

    # Type conversion to match original type
    if isinstance(actual_value, (int, np.integer)):
        min_val = int(np.floor(min_val))
        max_val = int(np.ceil(max_val))
    else:
        min_val = type(actual_value)(min_val)
        max_val = type(actual_value)(max_val)

    return min_val, max_val


def find_molecules_matching_constraints(df: pd.DataFrame, constraints: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Find molecules in dataframe that match ALL constraints.

    Args:
        df: Dataframe to search
        constraints: List of constraint dictionaries

    Returns:
        DataFrame of matching molecules or None if no matches
    """
    # Start with a mask aligned with the dataframe index
    mask = pd.Series([True] * len(df), index=df.index)

    for constraint in constraints:
        property_name = constraint['property']
        operator = constraint['operator']
        value = constraint['value']

        # Skip if property doesn't exist
        if property_name not in df.columns:
            return None

        # Apply operator
        if operator == '=':
            mask &= (df[property_name] == value)
        elif operator == '>':
            mask &= (df[property_name] > value)
        elif operator == '<':
            mask &= (df[property_name] < value)
        elif operator == '>=':
            mask &= (df[property_name] >= value)
        elif operator == '<=':
            mask &= (df[property_name] <= value)
        else:
            # Unsupported operator
            return None

    matching = df[mask]

    if len(matching) == 0:
        return None

    return matching


def format_natural_language_answer(target: Dict[str, Any], task_type: str) -> str:
    """
    Format the target answer in natural language.

    Args:
        target: Dictionary of property names to values
        task_type: Type of task (count or index)

    Returns:
        Natural language answer string
    """
    answers = []

    for prop, value in target.items():
        # Get the natural language name
        if 'count' in task_type:
            nl_name = get_natural_language_name(prop, 'count')
        else:
            nl_name = get_natural_language_name(prop, 'index')

        # Format the value
        if task_type in ['single_count', 'multi_count']:
            # For count tasks, just show the number
            answers.append(f"{nl_name}: {value}")
        elif task_type in ['single_index', 'multi_index']:
            # For index tasks, format the list
            if isinstance(value, list) and len(value) > 0:
                indices_str = ", ".join(str(i) for i in value)
                answers.append(f"{nl_name}: {indices_str}")
            else:
                answers.append(f"{nl_name}: none")

    return "; ".join(answers)


def get_property_category(property_name: str) -> str:
    """
    Get the category name for a property using explicit maps from column_category_map.

    Args:
        property_name: The property name

    Returns:
        Category name string
    """
    # Check in COUNT_MAP
    for category, props in COUNT_MAP.items():
        if property_name in props:
            return category

    # Check in INDEX_MAP
    for category, props in INDEX_MAP.items():
        if property_name in props:
            return category

    # Check in CONSTRAINT_MAP
    for category, props in CONSTRAINT_MAP.items():
        if property_name in props:
            return category

    # If not found in any map, return the property name itself
    return property_name


def get_natural_language_name(property_name: str, task_type: str) -> str:
    """
    Get natural language name for a property.

    Args:
        property_name: Technical property name
        task_type: 'count' or 'index'

    Returns:
        Natural language description
    """
    # Use mappings from natural_language module
    if task_type == 'count' and property_name in COUNT_MAPPINGS:
        options = COUNT_MAPPINGS[property_name]
        return options[0] if isinstance(options, list) else options
    elif task_type == 'index' and property_name in INDEX_MAPPINGS:
        options = INDEX_MAPPINGS[property_name]
        return options[0] if isinstance(options, list) else options

    # Fallback: convert underscores to spaces and clean up
    name = property_name.replace('_', ' ')
    name = name.replace(' count', '').replace(' index', '')
    return name


def create_huggingface_dataset(all_tasks: List[Dict[str, Any]],
                              dataset_name: str = "anonymous/moleculariq_benchmark",
                              push_to_hub: bool = False,
                              private: bool = True) -> DatasetDict:
    """
    Create HuggingFace dataset from generated tasks.
    Creates 'test' split with all data, plus individual splits for each task_type.

    Args:
        all_tasks: List of all generated tasks
        dataset_name: Name for HuggingFace dataset
        push_to_hub: Whether to push to HuggingFace Hub
        private: Whether dataset should be private

    Returns:
        DatasetDict with test split (all data) and task-type splits
    """
    print("\nCreating HuggingFace dataset...")

    # Normalize data types to ensure consistency BEFORE grouping
    print("  Normalizing data types...")
    for task in all_tasks:
        # Ensure ALL fields are present in ALL tasks (required for HuggingFace DatasetDict)
        # Common fields
        if 'task_type' not in task:
            task['task_type'] = None
        if 'original_smiles' not in task:
            task['original_smiles'] = None
        if 'smiles' not in task:
            task['smiles'] = None
        if 'complexity' not in task:
            task['complexity'] = None
        if 'complexity_bin' not in task:
            task['complexity_bin'] = None
        if 'iupac_name' not in task:
            task['iupac_name'] = None
        if 'question' not in task:
            task['question'] = None
        if 'natural_language_answer' not in task:
            task['natural_language_answer'] = None
        if 'supercategory' not in task:
            task['supercategory'] = None
        if 'uid' not in task:
            task['uid'] = None

        # Single task fields
        if 'category' not in task:
            task['category'] = None
        if 'property' not in task:
            task['property'] = None

        # Multi task fields
        if 'categories' not in task:
            task['categories'] = None
        if 'properties' not in task:
            task['properties'] = None
        if 'n_properties' not in task:
            task['n_properties'] = None

        # Count/Index task fields
        if 'target' not in task:
            task['target'] = None

        # Constraint generation task fields
        if 'answer' not in task:
            task['answer'] = None
        if 'constraints' not in task:
            task['constraints'] = None
        if 'transformation_type' not in task:
            task['transformation_type'] = None
        if 'n_constraints' not in task:
            task['n_constraints'] = None

        # Convert target to JSON string to handle mixed types
        if task['target'] is not None:
            task['target'] = json.dumps(task['target'])

        # Convert constraints to JSON strings
        if task['constraints'] is not None:
            task['constraints'] = json.dumps(task['constraints'])

        # Ensure iupac_name is a string
        if task['iupac_name'] is not None:
            if isinstance(task['iupac_name'], list):
                task['iupac_name'] = task['iupac_name'][0] if task['iupac_name'] else None

    # Define explicit features schema for consistent typing across all splits
    features = Features({
        'uid': Value('string'),
        'task_type': Value('string'),
        'original_smiles': Value('string'),
        'smiles': Value('string'),
        'complexity': Value('float64'),
        'complexity_bin': Value('string'),
        'iupac_name': Value('string'),
        'question': Value('string'),
        'category': Value('string'),
        'property': Value('string'),
        'target': Value('string'),  # JSON string
        'natural_language_answer': Value('string'),
        'supercategory': Value('string'),
        'categories': Sequence(Value('string')),
        'properties': Sequence(Value('string')),
        'n_properties': Value('int64'),
        'answer': Value('string'),
        'constraints': Value('string'),  # JSON string
        'transformation_type': Value('string'),
        'n_constraints': Value('int64'),
    })

    # Group tasks by type for creating splits AFTER normalization
    task_groups = {}
    for task in all_tasks:
        task_type = task['task_type']
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(task)

    # Create dataset splits
    dataset_dict = {}

    # Add 'test' split with all data using explicit features
    dataset_dict['test'] = Dataset.from_list(all_tasks, features=features)
    print(f"  Created 'test' split with {len(all_tasks)} examples (full dataset)")

    # Add individual splits for each task_type with same features
    print("\nCreating task-specific splits:")
    for task_type, tasks in sorted(task_groups.items()):
        split_name = task_type  # Use task_type as split name
        dataset_dict[split_name] = Dataset.from_list(tasks, features=features)
        print(f"  Created '{split_name}' split with {len(tasks)} examples")

    dataset = DatasetDict(dataset_dict)

    # Print summary
    print(f"\nDataset summary:")
    print(f"  Total splits: {len(dataset_dict)}")
    print(f"  Main 'test' split: {len(all_tasks)} examples")
    print(f"  Task-specific splits: {len(task_groups)} splits")

    if push_to_hub:
        try:
            # Check if user is logged in
            api = HfApi()
            user_info = api.whoami()
            print(f"\nPushing to HuggingFace Hub as {user_info['name']}...")

            # Push dataset
            dataset.push_to_hub(
                dataset_name,
                private=private,
                commit_message="Benchmark dataset for molecular reasoning tasks with test and task-specific splits"
            )

            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{dataset_name}")
            if private:
                print("   (Dataset is PRIVATE)")
        except Exception as e:
            print(f"\n❌ Error pushing to hub: {e}")
            print("\nTo push the dataset, you need to:")
            print("1. Install huggingface-hub: pip install huggingface-hub")
            print("2. Login with: huggingface-cli login")
            print("3. Make sure you have write access to the organization")

    return dataset


def main():
    """Generate benchmark dataset with command-line arguments."""

    parser = argparse.ArgumentParser(
        description='Generate molecular reasoning benchmark dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data source
    parser.add_argument('--pickle-path', type=str,
                        default=str(DEFAULT_PROPERTIES_PATH),
                        help='Path to properties pickle file')

    # Sampling parameters
    parser.add_argument('--n-samples-per-bin', type=int, default=10,
                        help='Samples per property per complexity bin for single tasks')
    parser.add_argument('--n-samples-per-category', type=int, default=30,
                        help='Samples per category for constraint tasks')
    parser.add_argument('--n-samples-multi', type=int, default=100,
                        help='Samples per complexity bin for multi-property tasks')

    # Output options
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path for output JSON file (default: count_index_tasks.json)')
    parser.add_argument('--save-local', type=str, default=None,
                        help='Path to save HuggingFace dataset locally')

    # HuggingFace options
    parser.add_argument('--push-to-hub', action='store_true',
                        help='Push dataset to HuggingFace Hub')
    parser.add_argument('--dataset-name', type=str, default='anonymous/moleculariq_benchmark',
                        help='HuggingFace dataset name')
    parser.add_argument('--public', action='store_true',
                        help='Make dataset public (default is private)')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample molecules for testing')
    parser.add_argument('--sampling-prime-workers', type=int, default=0,
                        help='Thread workers for precomputing inverse-frequency caches (0 disables priming)')
    parser.add_argument('--single-constraint-variants', type=str, default='exact',
                        choices=['exact', 'flexible', 'both'],
                        help='Which single-constraint variants to emit')
    parser.add_argument('--multi-constraint-variants', type=str, default='exact',
                        choices=['exact', 'flexible', 'both'],
                        help='Which multi-constraint variants to emit')
    parser.add_argument('--subsample-cache-path', type=str, default=None,
                        help='Optional path to a cached subsampled pickle (created when missing)')

    args = parser.parse_args()

    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(__file__).parent / 'count_index_tasks.json'

    # Parameters from arguments
    n_samples_per_bin_single = args.n_samples_per_bin
    n_samples_per_bin_multi = args.n_samples_multi
    n_samples_per_category = args.n_samples_per_category
    seed = args.seed
    subsample_size = args.subsample
    single_variant_mode = args.single_constraint_variants
    multi_variant_mode = args.multi_constraint_variants

    print("Loading data...")
    subsample_applied = False
    cache_path = Path(args.subsample_cache_path) if args.subsample_cache_path else None
    load_path = Path(args.pickle_path)
    if cache_path and cache_path.exists():
        print(f"  Using subsample cache: {cache_path}")
        load_path = cache_path
    else:
        print(f"  Pickle path: {args.pickle_path}")

    try:
        df = load_and_prepare_data(load_path)
        print(f"Loaded {len(df)} molecules with {len(df.columns)} properties")
    except (EOFError, FileNotFoundError, Exception) as e:
        print(f"Could not load pickle file: {e}")
        print("Creating synthetic data for testing...")

        # Create synthetic dataset for testing
        n_molecules = 1000

        # Generate valid SMILES
        smiles_list = []
        for i in range(n_molecules):
            # Create simple valid SMILES
            carbon_count = (i % 10) + 1
            if i % 3 == 0:
                # Linear alkane
                smiles = 'C' * carbon_count
            elif i % 3 == 1:
                # Alcohol
                smiles = 'C' * max(1, carbon_count - 1) + 'O'
            else:
                # Ether
                smiles = 'C' * max(1, carbon_count // 2) + 'O' + 'C' * max(1, carbon_count // 2)
            smiles_list.append(smiles)

        data = {
            'smiles': smiles_list,
            'iupac_name': [f'molecule_{i}' for i in range(n_molecules)],
            'complexity': np.random.uniform(0, 2000, n_molecules)
        }

        # Add complexity bins
        data['complexity_bin'] = pd.cut(
            data['complexity'],
            bins=[0, 250, 1000, float('inf')],
            labels=['0-250', '250-1000', '1000-inf']
        )

        # Add various properties
        properties = []

        # Basic properties
        basic_props = ['hydrogen_atom_count', 'carbon_atom_count', 'ring_count',
                       'aromatic_ring_count', 'hba_count', 'hbd_count',
                       'rotatable_bond_count']
        properties.extend(basic_props)

        # Functional groups
        for i in range(10):
            properties.append(f'functional_group_{i}_count')

        # Oxidation states
        for element in ['C', 'N', 'O']:
            for state in ['max', 'min']:
                properties.append(f'oxidation_state_{element}_{state}_count')

        # Stereochemistry
        properties.extend(['r_s_stereocenter_r_count', 'r_s_stereocenter_s_count',
                          'e_z_stereochemistry_e_count', 'e_z_stereochemistry_z_count'])

        # Ring sizes
        properties.extend(['smallest_largest_ring_size_smallest_count',
                          'smallest_largest_ring_size_largest_count'])

        # Add properties with realistic distributions
        for prop in properties:
            if 'functional_group' in prop:
                # Many zeros for functional groups
                data[prop] = np.random.choice([0, 0, 0, 0, 1, 2], n_molecules)
            elif 'oxidation_state' in prop:
                # Some zeros for oxidation states
                data[prop] = np.random.choice([0, 0, 1, 2, 3], n_molecules)
            elif 'stereocenter' in prop or 'stereochemistry' in prop:
                # Rare for stereochemistry
                data[prop] = np.random.choice([0, 0, 0, 0, 0, 1], n_molecules)
            else:
                # Fewer zeros for basic properties
                data[prop] = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_molecules)

            # Add corresponding index properties
            if prop.endswith('_count'):
                index_prop = prop.replace('_count', '_index')
                data[index_prop] = [
                    list(range(val)) if val > 0 else []
                    for val in data[prop]
                ]

        df = pd.DataFrame(data)
        print(f"Created synthetic dataset with {len(df)} molecules and {len(properties)} properties")

    if cache_path and not cache_path.exists() and subsample_size is not None and len(df) > subsample_size:
        df_sub = df.sample(n=subsample_size, random_state=seed).reset_index(drop=True)
        df_sub.to_pickle(cache_path)
        print(f"Saved subsample of {len(df_sub)} molecules to {cache_path}")
        df = df_sub
        subsample_applied = True

    if cache_path and cache_path.exists() and load_path == cache_path:
        if subsample_size is not None and len(df) > subsample_size:
            df = df.sample(n=subsample_size, random_state=seed).reset_index(drop=True)
            subsample_applied = True

    # Subsample for faster testing
    if subsample_size is not None and len(df) > subsample_size and not subsample_applied:
        df = df.head(subsample_size)
        print(f"Subsampled to {len(df)} molecules for testing")

    # Generate paired single tasks (both count and index)
    print("\nGenerating paired single tasks (count + index)...")
    single_tasks, molecule_property_mapping = generate_paired_single_tasks(
        df,
        n_samples_per_bin=n_samples_per_bin_single,
        seed=seed,
        prime_workers=args.sampling_prime_workers
    )
    single_count_tasks = [t for t in single_tasks if t['task_type'] == 'single_count']
    single_index_tasks = [t for t in single_tasks if t['task_type'] == 'single_index']
    print(f"  Generated {len(single_count_tasks)} count tasks")
    print(f"  Generated {len(single_index_tasks)} index tasks")
    print(f"  Tracking {len(molecule_property_mapping)} molecules")

    # Multi-property numbers configuration
    n_properties = [2, 3, 5]  # For multi-tasks (both count and index)

    print(f"\nGenerating paired multi-tasks (n_properties={n_properties})...")
    print(f"  Target: {n_samples_per_bin_multi} tasks per complexity bin")

    multi_count, multi_index = generate_paired_multi_tasks(
        df, molecule_property_mapping,
        n_samples_per_bin=n_samples_per_bin_multi,
        n_properties=n_properties,
        seed=seed
    )
    print(f"  Generated {len(multi_count)} multi-count tasks")
    print(f"  Generated {len(multi_index)} multi-index tasks")
    print(f"  Perfect pairing: {len(multi_count) == len(multi_index)}")

    # Generate single constraint tasks
    print(f"\nGenerating single constraint generation tasks...")
    print(f"  Target: {n_samples_per_category} tasks per category")

    single_constraint_tasks, single_constraint_exact_seeds = generate_single_constraint_tasks(
        single_count_tasks,
        df,
        n_samples_per_category=n_samples_per_category,
        seed=seed,
        variant_mode=single_variant_mode,
        return_exact_seeds=True
    )
    print(f"  Generated {len(single_constraint_tasks)} single constraint tasks")
    exact_single = sum(1 for task in single_constraint_tasks if task['constraints'][0]['operator'] == '=')
    flexible_single = len(single_constraint_tasks) - exact_single
    print(f"    Breakdown → exact: {exact_single}, flexible: {flexible_single}")

    # Generate multi-constraint tasks - 300 for each constraint count (2, 3, 5)
    print(f"\nGenerating multi-constraint generation tasks (mode={multi_variant_mode})...")
    multi_constraint_tasks = generate_multi_constraint_generation_tasks(
        multi_count,
        single_constraint_exact_seeds,
        df,
        n_samples_per_combination=n_samples_per_category,
        n_constraints=[2, 3, 5],
        seed=seed,
        variant_mode=multi_variant_mode
    )
    print(f"  Total multi-constraint tasks: {len(multi_constraint_tasks)}")
    exact_multi = sum(
        1 for task in multi_constraint_tasks
        if all(constraint['operator'] == '=' for constraint in task['constraints'])
    )
    flexible_multi = len(multi_constraint_tasks) - exact_multi
    print(f"    Breakdown → exact: {exact_multi}, flexible: {flexible_multi}")

    # Combine all tasks
    all_tasks = single_tasks + multi_count + multi_index + single_constraint_tasks + multi_constraint_tasks

    # Assign unique identifiers before shuffling so downstream consumers can join/sort reliably
    uid_template = "task_{:08d}"
    for idx, task in enumerate(all_tasks):
        task['uid'] = uid_template.format(idx)

    # Shuffle
    random.Random(seed).shuffle(all_tasks)

    # Save to JSON
    print(f"\nSaving {len(all_tasks)} tasks to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_tasks, f, indent=2)

    # Print statistics
    print("\nTask statistics:")
    task_types = {}
    for task in all_tasks:
        task_type = task['task_type']
        task_types[task_type] = task_types.get(task_type, 0) + 1

    for task_type, count in sorted(task_types.items()):
        print(f"  {task_type}: {count}")

    print("\nComplexity bin distribution:")
    complexity_bins = {}
    for task in all_tasks:
        bin_name = task['complexity_bin']
        complexity_bins[bin_name] = complexity_bins.get(bin_name, 0) + 1

    for bin_name, count in sorted(complexity_bins.items()):
        print(f"  {bin_name}: {count}")

    # Print multi-constraint breakdown
    if 'multi_constraint_generation' in task_types:
        print("\nMulti-constraint generation breakdown by constraint count:")
        constraint_counts = {}
        for task in all_tasks:
            if task['task_type'] == 'multi_constraint_generation':
                n_constraints = len(task['constraints'])
                constraint_counts[n_constraints] = constraint_counts.get(n_constraints, 0) + 1
        for n, count in sorted(constraint_counts.items()):
            print(f"  {n} constraints: {count} tasks")

    # Create HuggingFace dataset if requested
    if args.push_to_hub or args.save_local:
        try:
            dataset = create_huggingface_dataset(
                all_tasks=all_tasks,
                dataset_name=args.dataset_name,
                push_to_hub=args.push_to_hub,
                private=not args.public
            )

            # Save locally if requested
            if args.save_local:
                dataset.save_to_disk(args.save_local)
                print(f"\n✅ Dataset saved locally to: {args.save_local}")

        except ImportError:
            print("\n⚠️  Warning: datasets and/or huggingface_hub not installed")
            print("   Install with: pip install datasets huggingface_hub")

    print(f"\n{'='*60}")
    print("BENCHMARK DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total tasks generated: {len(all_tasks)}")
    print(f"JSON file: {output_path}")
    if args.push_to_hub:
        print(f"HuggingFace dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()

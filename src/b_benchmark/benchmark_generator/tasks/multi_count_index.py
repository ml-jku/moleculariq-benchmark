"""
Multi count and index task generation.

Generates paired multi-count and multi-index tasks for the same molecules.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.b_benchmark.questions import TASKS
from src.b_benchmark.natural_language.formatter import format_count_query, format_index_query
from src.b_benchmark.column_category_map import COUNT_TO_INDEX_MAP
from ..lineage import UIDGenerator, PropertyTaskMapping

from ..utils.smiles import transform_smiles
from ..utils.properties import (
    get_property_category,
    get_natural_language_name,
    get_count_properties,
    get_index_properties,
    convert_fg_count_to_nbr_instances,
)
from ..core.sampling import select_property_with_inverse_frequency
from .single_count_index import format_natural_language_answer


def generate_paired_multi_tasks(
    df: pd.DataFrame,
    molecule_property_mapping: Dict[str, Dict[str, Any]],
    n_samples_per_bin: int = 10,
    n_properties: List[int] = None,
    seed: int = 42,
    uid_generator: Optional[UIDGenerator] = None,
    property_mapping: Optional[PropertyTaskMapping] = None,
    ultra_hard: bool = False,
    convert_fg_properties: bool = False,
    fg_conversion_probability: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate paired multi-count and multi-index tasks for the same molecules.

    Args:
        df: DataFrame with properties
        molecule_property_mapping: Dict of molecules used in single tasks
        n_samples_per_bin: Samples per combination per complexity bin
        n_properties: List of numbers of properties to combine (e.g., [2, 3, 5])
        seed: Random seed
        uid_generator: UID generator for task IDs
        property_mapping: Property task mapping for lineage tracking
        ultra_hard: Whether to use ultra-hard mode
        convert_fg_properties: Whether to convert FG count to nbrInstances
        fg_conversion_probability: Probability of conversion

    Returns:
        Tuple of (count_tasks, index_tasks)
    """
    if n_properties is None:
        n_properties = [2, 3, 5]

    count_tasks = []
    index_tasks = []

    count_properties = get_count_properties(df)
    index_properties = get_index_properties(df)

    # Group molecules by complexity bin
    molecules_by_bin: Dict[str, List[Tuple[str, Dict]]] = {}
    for mol_smiles, mol_data in molecule_property_mapping.items():
        bin_name = mol_data["complexity_bin"]
        if bin_name not in molecules_by_bin:
            molecules_by_bin[bin_name] = []
        molecules_by_bin[bin_name].append((mol_smiles, mol_data))

    count_templates = TASKS["multi_count"]["question_templates"]
    index_templates = TASKS["multi_index_identification"]["question_templates"]

    rng = random.Random(seed)
    total_targets = sum(
        len(n_properties) * n_samples_per_bin
        for molecules_in_bin in molecules_by_bin.values()
        if len(molecules_in_bin) > 0
    )

    if total_targets == 0:
        return count_tasks, index_tasks

    with tqdm(
        total=total_targets,
        desc="Generating paired multi count/index tasks",
        unit="pair",
    ) as progress_bar:
        for bin_name, molecules_in_bin in molecules_by_bin.items():
            if len(molecules_in_bin) == 0:
                continue

            property_usage_counter: Dict[str, int] = {}
            category_usage_counter: Dict[str, int] = {}
            used_combinations: set = set()

            for n_props in n_properties:
                attempts = 0
                successes = 0
                max_attempts = n_samples_per_bin * 15

                while successes < n_samples_per_bin and attempts < max_attempts:
                    attempts += 1

                    mol_smiles, mol_data = rng.choice(molecules_in_bin)
                    row_data = mol_data["row_data"]
                    existing_props = mol_data["properties"]

                    selected_count_props = rng.sample(
                        existing_props, min(len(existing_props), n_props)
                    )

                    if len(selected_count_props) < n_props:
                        available_count_props = [
                            p
                            for p in count_properties
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
                                    apply_zero_bias=True,
                                    ultra_hard=ultra_hard,
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

                    zero_count = sum(
                        1 for prop in selected_count_props if row_data[prop] == 0
                    )
                    if zero_count > 1:
                        continue

                    # Convert count properties to index properties
                    selected_index_props: List[Optional[str]] = []
                    for count_prop in selected_count_props:
                        index_prop = COUNT_TO_INDEX_MAP.get(count_prop)
                        if (
                            index_prop
                            and index_prop in row_data
                            and index_prop in index_properties
                        ):
                            selected_index_props.append(index_prop)
                        else:
                            selected_index_props.append(None)

                    valid_index_props = [p for p in selected_index_props if p is not None]
                    if len(valid_index_props) < n_props // 2:
                        continue

                    # Build final index property list
                    final_index_props: List[str] = []
                    for index_prop in selected_index_props:
                        if index_prop is not None:
                            final_index_props.append(index_prop)
                        else:
                            available_alternatives = [
                                p
                                for p in index_properties
                                if p in row_data
                                and p not in final_index_props
                                and isinstance(row_data[p], list)
                                and len(row_data[p]) > 0
                            ]
                            if available_alternatives:
                                alt_prop = select_property_with_inverse_frequency(
                                    available_alternatives,
                                    property_usage_counter,
                                    category_usage_counter,
                                    row_data,
                                    rng,
                                    apply_zero_bias=True,
                                    ultra_hard=ultra_hard,
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

                    transformed_smiles = mol_data["transformed_smiles"]

                    # Optionally convert FG count properties
                    converted_count_props = []
                    for p in selected_count_props:
                        if convert_fg_properties:
                            conv_p = convert_fg_count_to_nbr_instances(p)
                            if (
                                conv_p != p
                                and rng.random() < fg_conversion_probability
                            ):
                                converted_count_props.append(conv_p)
                            else:
                                converted_count_props.append(p)
                        else:
                            converted_count_props.append(p)

                    count_natural_names = [
                        get_natural_language_name(p, "count")
                        for p in converted_count_props
                    ]
                    count_template = rng.choice(count_templates)
                    count_question = format_count_query(
                        transformed_smiles,
                        count_natural_names,
                        template=count_template,
                        include_key_hint=True,
                        key_names=converted_count_props,
                    )

                    count_target = {}
                    for orig_prop, conv_prop in zip(
                        selected_count_props, converted_count_props
                    ):
                        count_target[conv_prop] = (
                            str(row_data[orig_prop])
                            if "molecular_formula" in orig_prop
                            else int(row_data[orig_prop])
                        )

                    count_categories = [
                        get_property_category(prop) for prop in converted_count_props
                    ]

                    count_uid = uid_generator.generate() if uid_generator else None
                    index_uid = uid_generator.generate() if uid_generator else None

                    count_parent_uids = []
                    if property_mapping:
                        count_parent_uids = property_mapping.find_parent_tasks(
                            mol_smiles, converted_count_props
                        )

                    count_task = {
                        "task_type": "multi_count",
                        "uid": count_uid,
                        "parent_uids": count_parent_uids,
                        "sibling_uids": [index_uid] if index_uid else [],
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data["complexity"],
                        "complexity_bin": mol_data["complexity_bin"],
                        "iupac_name": mol_data["iupac_name"],
                        "question": count_question,
                        "n_properties": len(converted_count_props),
                        "categories": count_categories,
                        "properties": converted_count_props,
                        "target": count_target,
                        "natural_language_answer": format_natural_language_answer(
                            count_target, "multi_count"
                        ),
                        "supercategory": f"multi_count_nbr_{len(converted_count_props)}",
                        "is_randomized": mol_data.get("is_randomized", False),
                        "is_kekulized": mol_data.get("is_kekulized", False),
                    }

                    index_natural_names = [
                        get_natural_language_name(p, "index") for p in final_index_props
                    ]
                    index_template = rng.choice(index_templates)
                    index_question = format_index_query(
                        transformed_smiles,
                        index_natural_names,
                        template=index_template,
                        include_key_hint=True,
                        key_names=final_index_props,
                    )

                    index_target = {prop: row_data[prop] for prop in final_index_props}
                    index_categories = [
                        get_property_category(prop) for prop in final_index_props
                    ]

                    index_parent_uids = []
                    if property_mapping:
                        index_parent_uids = property_mapping.find_parent_tasks(
                            mol_smiles, final_index_props
                        )

                    index_task = {
                        "task_type": "multi_index",
                        "uid": index_uid,
                        "parent_uids": index_parent_uids,
                        "sibling_uids": [count_uid] if count_uid else [],
                        "original_smiles": mol_smiles,
                        "smiles": transformed_smiles,
                        "complexity": mol_data["complexity"],
                        "complexity_bin": mol_data["complexity_bin"],
                        "iupac_name": mol_data["iupac_name"],
                        "question": index_question,
                        "n_properties": len(final_index_props),
                        "categories": index_categories,
                        "properties": final_index_props,
                        "target": index_target,
                        "natural_language_answer": format_natural_language_answer(
                            index_target, "multi_index"
                        ),
                        "supercategory": f"multi_index_nbr_{len(final_index_props)}",
                        "is_randomized": mol_data.get("is_randomized", False),
                        "is_kekulized": mol_data.get("is_kekulized", False),
                    }

                    count_tasks.append(count_task)
                    index_tasks.append(index_task)
                    successes += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {"bin": str(bin_name), "n_props": n_props}, refresh=False
                    )

                    used_combinations.add(combination_key)

                    for prop in selected_count_props:
                        property_usage_counter[prop] = (
                            property_usage_counter.get(prop, 0) + 1
                        )
                        cat = get_property_category(prop)
                        category_usage_counter[cat] = (
                            category_usage_counter.get(cat, 0) + 1
                        )

                    for prop in final_index_props:
                        property_usage_counter[prop] = (
                            property_usage_counter.get(prop, 0) + 1
                        )
                        cat = get_property_category(prop)
                        category_usage_counter[cat] = (
                            category_usage_counter.get(cat, 0) + 1
                        )

    return count_tasks, index_tasks

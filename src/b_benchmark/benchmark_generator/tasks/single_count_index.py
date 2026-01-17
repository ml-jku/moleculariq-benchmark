"""
Single count and index task generation.

Generates paired count and index tasks for the same molecule-property combinations.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.b_benchmark.questions import TASKS
from src.b_benchmark.natural_language.formatter import format_count_query, format_index_query
from src.b_benchmark.column_category_map import COUNT_MAP, COUNT_TO_INDEX_MAP
from ..lineage import UIDGenerator, PropertyTaskMapping

from ..utils.smiles import transform_smiles
from ..utils.properties import (
    get_property_category,
    get_natural_language_name,
    convert_fg_count_to_nbr_instances,
)
from ..core.sampling import (
    sample_with_inverse_frequency,
    compute_inverse_frequency_arrays,
)


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
        if "count" in task_type:
            nl_name = get_natural_language_name(prop, "count")
        else:
            nl_name = get_natural_language_name(prop, "index")

        if task_type in ["single_count", "multi_count"]:
            answers.append(f"{nl_name}: {value}")
        elif task_type in ["single_index", "multi_index"]:
            if isinstance(value, list) and len(value) > 0:
                indices_str = ", ".join(str(i) for i in value)
                answers.append(f"{nl_name}: {indices_str}")
            else:
                answers.append(f"{nl_name}: none")

    return "; ".join(answers)


def generate_paired_single_tasks(
    df: pd.DataFrame,
    n_samples_per_bin: int = 10,
    seed: int = 42,
    prime_workers: int = 0,
    uid_generator: Optional[UIDGenerator] = None,
    ultra_hard: bool = False,
    convert_fg_properties: bool = False,
    fg_conversion_probability: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], PropertyTaskMapping]:
    """
    Generate paired count and index tasks for the same molecule-property combinations.

    Each selected molecule-property pair generates both a count and index task.
    Uses explicit COUNT_MAP and COUNT_TO_INDEX_MAP for pairing.

    Args:
        df: DataFrame with properties
        n_samples_per_bin: Samples per property per complexity bin
        seed: Random seed
        prime_workers: Optional number of worker threads to precompute sampling weights
        uid_generator: UID generator for task IDs
        ultra_hard: Whether to use ultra-hard mode
        convert_fg_properties: Whether to convert FG count to nbrInstances
        fg_conversion_probability: Probability of conversion

    Returns:
        Tuple of (task_list, molecule_property_mapping, property_mapping)
    """
    tasks = []
    molecule_property_mapping = {}
    property_mapping = PropertyTaskMapping()
    complexity_bins = df["complexity_bin"].unique()

    # Pre-slice dataframe per bin
    df_bins_cache: Dict[Any, pd.DataFrame] = {}
    valid_bins: List[Any] = []
    for bin_name in complexity_bins:
        df_bin = df[df["complexity_bin"] == bin_name]
        if len(df_bin) == 0:
            continue
        df_bins_cache[bin_name] = df_bin
        valid_bins.append(bin_name)

    count_templates = TASKS["single_count"]["question_templates"]
    index_templates = TASKS["single_index_identification"]["question_templates"]

    rng = random.Random(seed)
    sampling_cache: Dict[Tuple[Any, str, int], Tuple] = {}

    # Calculate total iterations for progress bar
    total_single_iterations = 0
    for bin_name in valid_bins:
        df_bin = df_bins_cache[bin_name]
        available_categories = 0
        for category_name, category_properties in COUNT_MAP.items():
            if category_name == "reaction_success":
                continue
            if any(p in df_bin.columns for p in category_properties):
                available_categories += 1
        total_single_iterations += available_categories * n_samples_per_bin

    with tqdm(
        total=total_single_iterations,
        desc="Generating single count/index tasks",
        unit="sample",
    ) as progress_bar:
        for bin_name in valid_bins:
            df_bin = df_bins_cache[bin_name]

            property_usage_counter: Dict[str, int] = {}
            category_usage_counter: Dict[str, int] = {}

            for category_name, category_properties in COUNT_MAP.items():
                if category_name == "reaction_success":
                    continue

                available_props = [p for p in category_properties if p in df_bin.columns]
                if not available_props:
                    continue

                for sample_idx in range(n_samples_per_bin):
                    progress_bar.update(1)

                    # Select property with inverse frequency weighting
                    prop_weights = []
                    for p in available_props:
                        usage = property_usage_counter.get(p, 0)
                        weight = 1.0 / (1 + usage)
                        if ultra_hard:
                            weight = weight**2
                        prop_weights.append(weight)

                    total_weight = sum(prop_weights)
                    prop_weights = [w / total_weight for w in prop_weights]
                    prop = rng.choices(available_props, weights=prop_weights, k=1)[0]

                    cache_key = (bin_name, prop, 1)
                    sampled = sample_with_inverse_frequency(
                        df_bin,
                        prop,
                        n_samples=1,
                        min_count=1,
                        seed=seed + hash(f"{prop}_{sample_idx}_{bin_name}") % 1000,
                        cache=sampling_cache,
                        cache_key=cache_key,
                    )

                    for _, row in sampled.iterrows():
                        original_smiles, transformed_smiles, is_randomized, is_kekulized = transform_smiles(
                            row["smiles"], rng, return_flags=True
                        )
                        molecule_key = original_smiles

                        # Track this property for this molecule
                        if molecule_key not in molecule_property_mapping:
                            molecule_property_mapping[molecule_key] = {
                                "properties": [],
                                "complexity": float(row["complexity"]),
                                "complexity_bin": str(row["complexity_bin"]),
                                "iupac_name": (
                                    row["iupac_name"]
                                    if isinstance(row["iupac_name"], list)
                                    else [row["iupac_name"]]
                                ),
                                "transformed_smiles": transformed_smiles,
                                "is_randomized": is_randomized,
                                "is_kekulized": is_kekulized,
                                "row_data": row.to_dict(),
                            }
                        molecule_property_mapping[molecule_key]["properties"].append(prop)

                        # Update usage counters
                        property_usage_counter[prop] = (
                            property_usage_counter.get(prop, 0) + 1
                        )
                        category_usage_counter[category_name] = (
                            category_usage_counter.get(category_name, 0) + 1
                        )

                        # Generate COUNT task
                        prop_for_task = prop
                        if convert_fg_properties:
                            converted_prop = convert_fg_count_to_nbr_instances(prop)
                            if (
                                converted_prop != prop
                                and rng.random() < fg_conversion_probability
                            ):
                                prop_for_task = converted_prop

                        count_natural_name = get_natural_language_name(
                            prop_for_task, "count"
                        )
                        count_template = random.Random(seed + len(tasks)).choice(
                            count_templates
                        )
                        normalized_template = count_template.replace(
                            "{count_type}", "{count_types}"
                        )
                        count_question = format_count_query(
                            transformed_smiles,
                            [count_natural_name],
                            template=normalized_template,
                            include_key_hint=True,
                            key_names=[prop_for_task],
                        )

                        # Handle molecular formula specially
                        if "molecular_formula" in prop:
                            target_value = str(row[prop])
                        else:
                            target_value = int(row[prop])

                        count_target_dict = {prop_for_task: target_value}

                        count_uid = uid_generator.generate() if uid_generator else None

                        count_task = {
                            "task_type": "single_count",
                            "uid": count_uid,
                            "parent_uids": [],
                            "original_smiles": original_smiles,
                            "smiles": transformed_smiles,
                            "complexity": float(row["complexity"]),
                            "complexity_bin": str(row["complexity_bin"]),
                            "iupac_name": (
                                row["iupac_name"]
                                if isinstance(row["iupac_name"], list)
                                else [row["iupac_name"]]
                            ),
                            "question": count_question,
                            "category": get_property_category(prop_for_task),
                            "property": prop_for_task,
                            "target": count_target_dict,
                            "natural_language_answer": format_natural_language_answer(
                                count_target_dict, "single_count"
                            ),
                            "supercategory": f"single_count_{get_property_category(prop_for_task)}",
                            "is_randomized": is_randomized,
                            "is_kekulized": is_kekulized,
                        }
                        tasks.append(count_task)

                        if uid_generator:
                            property_mapping.add_single_task(
                                count_uid, original_smiles, prop_for_task, count_task
                            )

                        # Generate INDEX task using COUNT_TO_INDEX_MAP for pairing
                        if prop in COUNT_TO_INDEX_MAP:
                            index_prop = COUNT_TO_INDEX_MAP[prop]
                            if index_prop in row:
                                index_value = row[index_prop]
                                if isinstance(index_value, list):
                                    # Ultra-hard mode: skip index tasks with <= 5 indices
                                    if ultra_hard and len(index_value) <= 5:
                                        continue

                                    index_natural_name = get_natural_language_name(
                                        index_prop, "index"
                                    )
                                    index_template = random.Random(
                                        seed + len(tasks)
                                    ).choice(index_templates)
                                    normalized_index_template = index_template.replace(
                                        "{index_type}", "{index_types}"
                                    )
                                    index_question = format_index_query(
                                        transformed_smiles,
                                        [index_natural_name],
                                        template=normalized_index_template,
                                        include_key_hint=True,
                                        key_names=[index_prop],
                                    )

                                    index_target_dict = {index_prop: index_value}

                                    index_uid = (
                                        uid_generator.generate()
                                        if uid_generator
                                        else None
                                    )

                                    index_task = {
                                        "task_type": "single_index",
                                        "uid": index_uid,
                                        "parent_uids": (
                                            [count_uid] if count_uid else []
                                        ),
                                        "original_smiles": original_smiles,
                                        "smiles": transformed_smiles,
                                        "complexity": float(row["complexity"]),
                                        "complexity_bin": str(row["complexity_bin"]),
                                        "iupac_name": (
                                            row["iupac_name"]
                                            if isinstance(row["iupac_name"], list)
                                            else [row["iupac_name"]]
                                        ),
                                        "question": index_question,
                                        "category": get_property_category(index_prop),
                                        "property": index_prop,
                                        "target": index_target_dict,
                                        "natural_language_answer": format_natural_language_answer(
                                            index_target_dict, "single_index"
                                        ),
                                        "supercategory": f"single_index_{get_property_category(index_prop)}",
                                        "is_randomized": is_randomized,
                                        "is_kekulized": is_kekulized,
                                    }
                                    tasks.append(index_task)

                                    if uid_generator:
                                        property_mapping.add_single_task(
                                            index_uid,
                                            original_smiles,
                                            index_prop,
                                            index_task,
                                        )

    return tasks, molecule_property_mapping, property_mapping

"""
Single constraint task generation.

Generates single-constraint generation tasks with exact and flexible variants.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from moleculariq_core import TASKS, COUNT_MAP, CONSTRAINT_MAP, NaturalLanguageFormatter
from ..lineage import UIDGenerator

# Create formatter instance for formatting constraints
_formatter = NaturalLanguageFormatter()
format_constraint = _formatter.format_constraint

from ..utils.smiles import transform_smiles
from ..utils.properties import get_property_category, to_python_scalar
from ..core.validation import validate_constraints


def select_random_operator(
    property_name: str, value: Any, df: pd.DataFrame, rng: random.Random
) -> str:
    """Select a random operator for a constraint."""
    # Boolean properties only support equality
    if property_name.startswith("template_based_reaction_prediction_") and property_name.endswith("_success"):
        return "="

    # String/categorical properties only support equality
    if property_name in ["murcko_scaffold_value", "murcko_scaffold_count", "molecular_formula_count", "molecular_formula"]:
        return "="

    if "_success" in property_name:
        return "="

    # Numeric properties - handle zero values specially
    if isinstance(value, (int, float, np.integer, np.floating)) and value == 0:
        operators = ["=", ">"]
    else:
        operators = ["=", ">", "<", ">=", "<=", "range"]

    return rng.choice(operators)


def adjust_value_for_operator(
    property_name: str,
    value: Any,
    operator: str,
    df: pd.DataFrame,
    rng: random.Random,
    percentile_cache: Optional[Dict] = None,
    cache_key: Optional[Any] = None,
) -> Any:
    """Adjust value based on operator to ensure matches exist."""
    if operator == "=":
        return value

    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value

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

    if operator in [">", ">="]:
        percentile = rng.uniform(2, 30)
        threshold = np.percentile(values, percentile)
        return type(value)(threshold)

    if operator in ["<", "<="]:
        percentile = rng.uniform(70, 98)
        threshold = np.percentile(values, percentile)
        return type(value)(threshold)

    return value


def adjust_values_for_range_operator(
    property_name: str,
    actual_value: Any,
    df: pd.DataFrame,
    rng: random.Random,
    percentile_cache: Optional[Dict] = None,
    cache_key: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Generate min_value and max_value for a range constraint."""
    if not isinstance(actual_value, (int, float, np.integer, np.floating)):
        return actual_value, actual_value

    if property_name not in df.columns:
        if actual_value == 0:
            return 0, max(1, int(actual_value + 5))
        offset = max(1, abs(actual_value) * 0.3)
        return type(actual_value)(actual_value - offset), type(actual_value)(actual_value + offset)

    cached_sorted = None
    if percentile_cache is not None and cache_key is not None:
        cached_sorted = percentile_cache.get(cache_key)

    if cached_sorted is not None:
        values = cached_sorted
    else:
        values = df[property_name].dropna()
        if len(values) == 0:
            if actual_value == 0:
                return 0, max(1, int(actual_value + 5))
            offset = max(1, abs(actual_value) * 0.3)
            return type(actual_value)(actual_value - offset), type(actual_value)(actual_value + offset)
        values = np.sort(values.values if isinstance(values, pd.Series) else np.array(values))
        if percentile_cache is not None and cache_key is not None:
            percentile_cache[cache_key] = values

    actual_percentile = (values < actual_value).sum() / len(values) * 100

    if actual_value == 0:
        upper_percentile = rng.uniform(5, 15)
        max_val = np.percentile(values, upper_percentile)
        max_val = max(1, max_val)
        return 0, type(actual_value)(max_val)
    elif actual_percentile < 30:
        lower_percentile = max(0, actual_percentile - rng.uniform(2, 5))
        upper_percentile = min(100, actual_percentile + rng.uniform(5, 10))
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)
        min_val = min(min_val, actual_value)
        max_val = max(max_val, actual_value)
    elif actual_percentile > 70:
        lower_percentile = max(0, actual_percentile - rng.uniform(5, 10))
        upper_percentile = min(100, actual_percentile + rng.uniform(2, 5))
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)
        min_val = min(min_val, actual_value)
        max_val = max(max_val, actual_value)
    else:
        spread = rng.uniform(8, 15)
        lower_percentile = max(0, actual_percentile - spread / 2)
        upper_percentile = min(100, actual_percentile + spread / 2)
        min_val = np.percentile(values, lower_percentile)
        max_val = np.percentile(values, upper_percentile)

    min_val = min(min_val, actual_value)
    max_val = max(max_val, actual_value)

    if min_val >= max_val:
        if actual_value == 0:
            max_val = max(1, min_val + 1)
        else:
            buffer = max(1, abs(actual_value) * 0.1)
            min_val = actual_value - buffer
            max_val = actual_value + buffer

    if isinstance(actual_value, (int, np.integer)):
        min_val = int(np.floor(min_val))
        max_val = int(np.ceil(max_val))
    else:
        min_val = type(actual_value)(min_val)
        max_val = type(actual_value)(max_val)

    return min_val, max_val


def sample_additional_constraints_for_category(
    df: pd.DataFrame,
    category: str,
    n_needed: int,
    existing_constraints: Set[Tuple[str, Any]],
    rng: random.Random,
    column_cache: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Sample additional constraint tasks for a specific category."""
    if category in COUNT_MAP:
        category_properties = COUNT_MAP[category]
    elif category in CONSTRAINT_MAP:
        category_properties = CONSTRAINT_MAP[category]
    elif category == "murcko_scaffold_value":
        category_properties = ["murcko_scaffold_value"] if "murcko_scaffold_value" in df.columns else []
    elif category == "reaction_success":
        category_properties = [col for col in df.columns if col.endswith("_success")]
    elif category == "functional_group_instances":
        category_properties = [
            col for col in df.columns
            if col.startswith("functional_group_") and col.endswith("_nbrInstances")
        ]
    else:
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
        property_name = rng.choice(category_properties)

        if property_name.startswith("functional_group_") and property_name.endswith("_count"):
            base_name = property_name.replace("_count", "")
            constraint_property = base_name + "_nbrInstances"
            if constraint_property in df.columns:
                property_name = constraint_property

        if property_name not in df.columns:
            continue

        # Get or cache non-null values
        if property_name not in column_cache:
            series = df[property_name]
            mask = series.notna()
            mask_np = mask.to_numpy()
            if not mask_np.any():
                column_cache[property_name] = (np.array([], dtype=object), np.array([], dtype=int))
            else:
                values = series.to_numpy()
                positions = np.nonzero(mask_np)[0].astype(int)
                python_values = np.empty(len(positions), dtype=object)
                for idx, pos in enumerate(positions):
                    python_values[idx] = to_python_scalar(values[pos])
                column_cache[property_name] = (python_values, positions)

        values_array, positions_array = column_cache[property_name]
        if len(values_array) == 0:
            continue

        sample_idx = rng.randint(0, len(values_array) - 1)
        value = values_array[sample_idx]

        constraint_key = (property_name, value)
        if constraint_key in existing_constraints:
            continue

        row_position = int(positions_array[sample_idx])
        row = df.iloc[row_position]

        constraint_data = {
            "property": property_name,
            "value": value,
            "smiles": row["smiles"],
            "complexity_bin": str(row["complexity_bin"]),
        }

        additional_constraints.append(constraint_data)
        existing_constraints.add(constraint_key)

        if len(additional_constraints) >= n_needed:
            break

    return additional_constraints


def generate_single_constraint_tasks(
    single_count_tasks: List[Dict[str, Any]],
    df: pd.DataFrame,
    n_samples_per_category: int = 10,
    seed: int = 42,
    variant_mode: str = "exact",
    return_exact_seeds: bool = False,
    uid_generator: Optional[UIDGenerator] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Generate single-constraint tasks with optional exact/flexible variants.

    Args:
        single_count_tasks: Seed count tasks to source properties from
        df: Complete dataframe of properties
        n_samples_per_category: Number of constraint seeds per category
        seed: RNG seed
        variant_mode: 'exact', 'flexible', or 'both'
        return_exact_seeds: When True also return list of exact tasks
        uid_generator: UID generator for task IDs

    Returns:
        Either a list of tasks or (tasks, exact_seed_tasks)
    """
    rng = random.Random(seed)
    variant_key = variant_mode.lower()
    if variant_key not in {"exact", "flexible", "both"}:
        raise ValueError("variant_mode must be one of {'exact', 'flexible', 'both'}")

    include_exact = variant_key in {"exact", "both"}
    include_flexible = variant_key in {"flexible", "both"}

    constraint_tasks: List[Dict[str, Any]] = []
    exact_seed_tasks: List[Dict[str, Any]] = []
    percentile_cache: Dict = {}
    constraint_column_cache: Dict = {}

    # Build mapping for parent tracking
    count_task_uid_mapping: Dict[Tuple[str, str], str] = {}
    for task in single_count_tasks:
        if "uid" in task and "original_smiles" in task and "property" in task:
            key = (task["original_smiles"], task["property"])
            count_task_uid_mapping[key] = task["uid"]

    count_tasks_by_category: Dict[str, List[Dict]] = {}
    for task in single_count_tasks:
        count_tasks_by_category.setdefault(task["category"], []).append(task)

    print(f"  Found {len(count_tasks_by_category)} categories from count tasks")

    sampled_constraints: List[Tuple[Tuple[str, Any], Dict]] = []

    for category, category_tasks in count_tasks_by_category.items():
        unique_constraints: Dict[Tuple[str, Any], List[Dict]] = {}

        for task in category_tasks:
            property_name = task["property"]
            constraint_key = (property_name, task["target"][property_name])
            unique_constraints.setdefault(constraint_key, []).append(task)

        category_samples: List[Tuple[Tuple[str, Any], Dict]] = []
        existing_constraints: Set[Tuple[str, Any]] = set(unique_constraints.keys())

        if len(unique_constraints) >= n_samples_per_category:
            sampled_keys = rng.sample(list(unique_constraints.keys()), n_samples_per_category)
        else:
            sampled_keys = list(unique_constraints.keys())

        for key in sampled_keys:
            category_samples.append((key, rng.choice(unique_constraints[key])))

        # Augment if needed
        while len(category_samples) < n_samples_per_category:
            needed = n_samples_per_category - len(category_samples)
            additional_constraints = sample_additional_constraints_for_category(
                df, category, needed, existing_constraints, rng,
                column_cache=constraint_column_cache
            )

            for constraint_data in additional_constraints:
                key = (constraint_data["property"], constraint_data["value"])
                category_samples.append((key, constraint_data))
                if len(category_samples) >= n_samples_per_category:
                    break

            if not additional_constraints:
                break

        sampled_constraints.extend(category_samples)

    # Add special categories
    special_categories = []
    if "murcko_scaffold_value" in df.columns:
        special_categories.append(("murcko_scaffold_value", ["murcko_scaffold_value"]))
    reaction_props = [col for col in df.columns if col.endswith("_success")]
    if reaction_props:
        special_categories.append(("reaction_success", reaction_props))

    for category_name, _ in special_categories:
        additional_constraints = sample_additional_constraints_for_category(
            df, category_name, n_samples_per_category, set(), rng,
            column_cache=constraint_column_cache
        )
        for constraint_data in additional_constraints:
            key = (constraint_data["property"], constraint_data["value"])
            sampled_constraints.append((key, constraint_data))
        print(f"  Added {len(additional_constraints)} constraints for special category: {category_name}")

    print(f"  Total sampled constraint tasks: {len(sampled_constraints)}")

    templates = TASKS["constraint_generation"]["question_templates"]

    for constraint_key, task_data in tqdm(sampled_constraints, desc="Creating single constraint tasks", unit="task"):
        property_name, raw_value = constraint_key

        if "original_smiles" in task_data:
            original_smiles = task_data["original_smiles"]
            complexity_bin = task_data["complexity_bin"]
        else:
            original_smiles = task_data["smiles"]
            complexity_bin = task_data["complexity_bin"]

        _, transformed_smiles = transform_smiles(original_smiles, rng)
        transformation_type = "unknown"

        constraint_property = property_name
        constraint_value = raw_value

        if property_name.startswith("functional_group_") and property_name.endswith("_count"):
            base_name = property_name.replace("_count", "")
            mapped_property = base_name + "_nbrInstances"
            if mapped_property in df.columns:
                constraint_property = mapped_property
                row_match = df[df["smiles"] == original_smiles]
                if len(row_match) > 0 and pd.notna(row_match.iloc[0][constraint_property]):
                    constraint_value = row_match.iloc[0][constraint_property]

        constraint_value = to_python_scalar(constraint_value)

        # Calculate prevalence for this constraint
        total_molecules = len(df)
        if constraint_property in df.columns:
            n_satisfying = int((df[constraint_property] == constraint_value).sum())
        else:
            n_satisfying = 0
        prevalence = n_satisfying / total_molecules if total_molecules > 0 else 0.0

        constraint_dict = {
            "type": constraint_property,
            "operator": "=",
            "value": constraint_value,
        }
        constraint_text = format_constraint(constraint_dict, use_varied_phrasing=False)
        question_template = rng.choice(templates)
        question_exact = question_template.format(constraint=constraint_text)

        if isinstance(constraint_property, str) and constraint_property.startswith("template_based_reaction_prediction_"):
            category = "reaction_templates"
        elif constraint_property in ["murcko_scaffold_value", "molecular_formula"]:
            category = constraint_property
        else:
            category = get_property_category(constraint_property)

        complexity_val = task_data.get("complexity") if "complexity" in task_data else None
        iupac_val = task_data.get("iupac_name") if "iupac_name" in task_data else None

        if constraint_property.endswith("_nbrInstances"):
            supercategory_suffix = "functional_group_nbrInstances"
        elif constraint_property.startswith("template_based_reaction_prediction_"):
            supercategory_suffix = "reaction_templates"
        else:
            supercategory_suffix = category

        exact_uid = uid_generator.generate() if uid_generator else None
        parent_uids = []
        source_key = (original_smiles, task_data.get("property", property_name))
        if source_key in count_task_uid_mapping:
            parent_uids.append(count_task_uid_mapping[source_key])

        exact_task = {
            "task_type": "single_constraint_generation",
            "uid": exact_uid,
            "parent_uids": parent_uids,
            "question": question_exact,
            "category": category,
            "property": constraint_property,
            "constraints": [{
                "property": constraint_property,
                "operator": "=",
                "value": constraint_value,
            }],
            "answer": transformed_smiles,
            "original_smiles": original_smiles,
            "smiles": None,  # No input molecule for generation tasks
            "complexity": None,  # No input molecule to measure complexity
            "complexity_bin": None,  # No input molecule
            "iupac_name": iupac_val,
            "transformation_type": transformation_type,
            "natural_language_answer": constraint_text,
            "supercategory": f"single_constraint_gen_exact_{supercategory_suffix}",
            "categories": None,
            "properties": None,
            "n_properties": None,
            "target": None,
            "n_constraints": 1,
            "n_satisfying_molecules": n_satisfying,
            "prevalence": prevalence,
        }

        exact_seed_tasks.append(exact_task)
        if include_exact:
            constraint_tasks.append(exact_task)

        if include_flexible:
            df_bin = df[df["complexity_bin"] == complexity_bin]
            if len(df_bin) == 0:
                df_bin = df[df["complexity_bin"].astype(str) == str(complexity_bin)]
            if len(df_bin) == 0:
                df_bin = df

            flexible_task = None
            for _ in range(10):
                operator = select_random_operator(constraint_property, constraint_value, df_bin, rng)
                if operator == "=":
                    continue

                if operator == "range":
                    min_val, max_val = adjust_values_for_range_operator(
                        constraint_property, constraint_value, df_bin, rng,
                        percentile_cache=percentile_cache,
                        cache_key=(str(complexity_bin), constraint_property)
                    )
                    min_val = to_python_scalar(min_val)
                    max_val = to_python_scalar(max_val)

                    variant_constraint = {
                        "property": constraint_property,
                        "operator": "range",
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                    variant_dict = {
                        "type": constraint_property,
                        "operator": "range",
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                else:
                    adjusted_value = adjust_value_for_operator(
                        constraint_property, constraint_value, operator, df_bin, rng,
                        percentile_cache=percentile_cache,
                        cache_key=(str(complexity_bin), constraint_property)
                    )
                    adjusted_value = to_python_scalar(adjusted_value)

                    variant_constraint = {
                        "property": constraint_property,
                        "operator": operator,
                        "value": adjusted_value,
                    }
                    variant_dict = {
                        "type": constraint_property,
                        "operator": operator,
                        "value": adjusted_value,
                    }

                if not validate_constraints(transformed_smiles, [variant_constraint]):
                    continue

                # Calculate prevalence for flexible constraint
                if constraint_property in df.columns:
                    col = df[constraint_property]
                    if operator == "range":
                        flex_n_satisfying = int(((col >= min_val) & (col <= max_val)).sum())
                    elif operator == ">":
                        flex_n_satisfying = int((col > adjusted_value).sum())
                    elif operator == ">=":
                        flex_n_satisfying = int((col >= adjusted_value).sum())
                    elif operator == "<":
                        flex_n_satisfying = int((col < adjusted_value).sum())
                    elif operator == "<=":
                        flex_n_satisfying = int((col <= adjusted_value).sum())
                    else:
                        flex_n_satisfying = 0
                else:
                    flex_n_satisfying = 0
                flex_prevalence = flex_n_satisfying / total_molecules if total_molecules > 0 else 0.0

                variant_text = format_constraint(variant_dict, use_varied_phrasing=False)
                variant_question = rng.choice(templates).format(constraint=variant_text)

                flexible_uid = uid_generator.generate() if uid_generator else None
                flexible_parent_uids = [exact_uid] if exact_uid else []

                flexible_task = {
                    "task_type": "single_constraint_generation",
                    "uid": flexible_uid,
                    "parent_uids": flexible_parent_uids,
                    "question": variant_question,
                    "category": category,
                    "property": constraint_property,
                    "constraints": [variant_constraint],
                    "answer": transformed_smiles,
                    "original_smiles": original_smiles,
                    "smiles": None,  # No input molecule for generation tasks
                    "complexity": None,  # No input molecule to measure complexity
                    "complexity_bin": None,  # No input molecule
                    "iupac_name": iupac_val,
                    "transformation_type": transformation_type,
                    "natural_language_answer": variant_text,
                    "supercategory": f"single_constraint_gen_flexible_{supercategory_suffix}",
                    "categories": None,
                    "properties": None,
                    "n_properties": None,
                    "target": None,
                    "n_constraints": 1,
                    "n_satisfying_molecules": flex_n_satisfying,
                    "prevalence": flex_prevalence,
                }
                break

            if flexible_task is not None:
                constraint_tasks.append(flexible_task)

    rng.shuffle(constraint_tasks)

    if return_exact_seeds:
        return constraint_tasks, exact_seed_tasks

    return constraint_tasks

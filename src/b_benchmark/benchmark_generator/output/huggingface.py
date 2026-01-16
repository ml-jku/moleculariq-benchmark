"""
HuggingFace dataset creation.

Creates standardized HuggingFace datasets from generated tasks.
"""

import json
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Value
from rdkit import RDLogger

from ..utils.smiles import enumerate_rings_in_smiles, has_rings

# Suppress RDKit logging
RDLogger.DisableLog('rdApp.*')

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def _create_ring_enumeration_split(
    task_groups: Dict[str, List[Dict]],
    schema: Features,
    seed: int,
) -> Optional[Dataset]:
    """
    Create a single ring enumeration split combining count and index tasks.

    Only includes tasks where the SMILES has at least one ring. The ring closure
    numbers are randomized (1-99) while keeping the molecule structure identical.
    The enumerated SMILES is guaranteed to be different from the original.

    Args:
        task_groups: Dict mapping task_type to list of tasks
        schema: HuggingFace Features schema
        seed: Random seed for reproducibility

    Returns:
        Dataset with ring-enumerated tasks, or None if no tasks qualify
    """
    rng = random.Random(seed)
    all_ring_enum_tasks = []
    total_skipped_no_rings = 0
    total_skipped_enum_failed = 0

    # Process count and index tasks (generation tasks don't have input SMILES)
    for task_type in ["count", "index"]:
        if task_type not in task_groups:
            continue

        for task in task_groups[task_type]:
            # Get the SMILES from metadata
            metadata = json.loads(task.get("metadata", "{}"))
            smiles = metadata.get("smiles")

            if not smiles:
                continue

            # Check if SMILES has rings
            if not has_rings(smiles):
                total_skipped_no_rings += 1
                continue

            # Enumerate rings (ensures result is different from original)
            # Retry multiple times to maximize success rate
            max_retries = 10
            enumerated_smiles = None
            ring_mapping = None
            for _ in range(max_retries):
                enumerated_smiles, ring_mapping = enumerate_rings_in_smiles(smiles, rng)
                if enumerated_smiles is not None:
                    break

            if enumerated_smiles is None:
                total_skipped_enum_failed += 1
                continue

            # Create new task with enumerated SMILES
            new_task = task.copy()

            # Replace SMILES in question text
            original_question = task["question"]
            new_question = original_question.replace(smiles, enumerated_smiles)
            new_task["question"] = new_question

            # Update metadata with ring mapping info
            new_metadata = metadata.copy()
            new_metadata["smiles"] = enumerated_smiles
            new_metadata["ring_mapping"] = ring_mapping
            new_metadata["original_ring_smiles"] = smiles
            new_task["metadata"] = json.dumps(new_metadata)

            # Generate new UID for ring enumerated task
            new_task["uid"] = f"{task['uid']}_ring_enum"

            all_ring_enum_tasks.append(new_task)

    if all_ring_enum_tasks:
        print(
            f"    Combined {len(all_ring_enum_tasks)} tasks with rings "
            f"(skipped {total_skipped_no_rings} without rings, "
            f"{total_skipped_enum_failed} failed enumeration)"
        )
        return Dataset.from_list(all_ring_enum_tasks, features=schema)

    return None


def create_huggingface_dataset(
    all_tasks: List[Dict[str, Any]],
    dataset_name: str = "tschouis/moleculariq",
    push_to_hub: bool = False,
    private: bool = True,
    ring_enumeration: bool = False,
    seed: int = 42,
) -> DatasetDict:
    """
    Create HuggingFace dataset from generated tasks.

    Creates 'test' split with all data, plus individual splits for each task_type.
    Optionally creates ring enumeration splits where SMILES ring closure numbers
    are randomized.

    Args:
        all_tasks: List of all generated tasks
        dataset_name: Name for HuggingFace dataset
        push_to_hub: Whether to push to HuggingFace Hub
        private: Whether dataset should be private
        ring_enumeration: Whether to create additional ring enumeration splits
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with test split (all data) and task-type splits
    """
    print("\nCreating HuggingFace dataset...")

    # Create minimal tasks with clean schema
    print("  Creating minimal task records...")
    minimal_tasks = []
    for task in all_tasks:
        old_task_type = task.get("task_type", "")

        # Map old task_type to new simplified task_type (count, index, generation)
        if "count" in old_task_type:
            new_task_type = "count"
        elif "index" in old_task_type:
            new_task_type = "index"
        elif "constraint" in old_task_type or "generation" in old_task_type:
            new_task_type = "generation"
        else:
            new_task_type = old_task_type

        # Determine multi_task_load (1 for single, 2/3/5 for multi)
        if old_task_type.startswith("single_"):
            multi_task_load = 1
        elif old_task_type.startswith("multi_"):
            # For multi-constraint: use n_constraints
            # For multi-count/index: use n_properties
            multi_task_load = task.get("n_constraints") or task.get("n_properties") or 1
        else:
            multi_task_load = 1

        # Convert target to JSON string if needed
        target = task.get("target")
        if target is not None:
            target = json.dumps(target)

        # Convert constraints to JSON string if needed
        constraints = task.get("constraints")
        if constraints is not None and not isinstance(constraints, str):
            constraints = json.dumps(constraints)

        # Features = the specific property categories (was supercategory)
        features_val = task.get("supercategory")

        # Build metadata dict (contains SMILES transformation info for count/index tasks)
        if new_task_type != "generation":
            metadata = {
                "smiles": task.get("smiles"),  # Transformed SMILES used in question
                "is_randomized": task.get("is_randomized", False),
                "is_kekulized": task.get("is_kekulized", False),
            }
        else:
            # For generation tasks, include prevalence information
            metadata = {
                "prevalence": task.get("prevalence"),
                "n_satisfying_molecules": task.get("n_satisfying_molecules"),
            }

        minimal_task = {
            "uid": task.get("uid"),
            "task_type": new_task_type,
            "features": features_val,
            "question": task.get("question"),
            "target": target,  # For count/index tasks
            "constraints": constraints,  # For generation tasks
            "original_smiles": task.get("original_smiles"),  # Ground truth / canonical
            "complexity_bin": task.get("complexity_bin"),
            "multi_task_load": multi_task_load,
            "metadata": json.dumps(metadata),  # JSON string with transformation info
        }
        minimal_tasks.append(minimal_task)

    # Define minimal features schema
    schema = Features({
        "uid": Value("string"),
        "task_type": Value("string"),  # count, index, generation
        "features": Value("string"),  # specific property categories
        "question": Value("string"),
        "target": Value("string"),  # JSON string for count/index tasks
        "constraints": Value("string"),  # JSON string for generation tasks
        "original_smiles": Value("string"),  # Ground truth / canonical SMILES
        "complexity_bin": Value("string"),
        "multi_task_load": Value("int64"),  # 1, 2, 3, or 5
        "metadata": Value("string"),  # JSON with smiles, is_randomized, is_kekulized
    })

    # Group tasks by new task_type (count, index, generation)
    task_groups: Dict[str, List[Dict]] = {}
    for task in minimal_tasks:
        task_type = task["task_type"]
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(task)

    # Create dataset splits
    dataset_dict = {}

    # Add 'test' split with all data
    dataset_dict["test"] = Dataset.from_list(minimal_tasks, features=schema)
    print(f"  Created 'test' split with {len(minimal_tasks)} examples (full dataset)")

    # Add individual splits for each task_type (count, index, generation)
    print("\nCreating task-specific splits:")
    for task_type, tasks in sorted(task_groups.items()):
        split_name = task_type
        dataset_dict[split_name] = Dataset.from_list(tasks, features=schema)
        print(f"  Created '{split_name}' split with {len(tasks)} examples")

    # Create ring enumeration split if enabled
    if ring_enumeration:
        print("\nCreating ring enumeration split:")
        ring_enum_dataset = _create_ring_enumeration_split(
            task_groups, schema, seed
        )
        if ring_enum_dataset is not None:
            dataset_dict["ring_enum"] = ring_enum_dataset
            print(f"  Created 'ring_enum' split with {len(ring_enum_dataset)} examples")

    dataset = DatasetDict(dataset_dict)

    # Print summary
    print(f"\nDataset summary:")
    print(f"  Total splits: {len(dataset_dict)}")
    print(f"  Main 'test' split: {len(minimal_tasks)} examples")
    print(f"  Task-specific splits: {len(task_groups)} splits")

    if push_to_hub:
        if HfApi is None:
            raise ImportError(
                "huggingface-hub is required for pushing to hub. "
                "Install with: pip install huggingface-hub"
            )
        api = HfApi()
        user_info = api.whoami()
        print(f"\nPushing to HuggingFace Hub as {user_info['name']}...")

        dataset.push_to_hub(
            dataset_name,
            private=private,
            commit_message="Benchmark dataset for molecular reasoning tasks",
        )

        print(f"Successfully pushed dataset to: https://huggingface.co/datasets/{dataset_name}")
        if private:
            print("   (Dataset is PRIVATE)")

    return dataset

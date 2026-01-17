"""
JSON output for benchmark tasks.

Creates JSON files with the same lean schema as the HuggingFace dataset,
including a split field and ring enumeration tasks.
"""

import json
from typing import Any, Dict, List

from .huggingface import create_minimal_task, create_ring_enumeration_tasks


def save_benchmark_json(
    all_tasks: List[Dict[str, Any]],
    output_path: str,
    ring_enumeration: bool = True,
    seed: int = 42,
) -> None:
    """
    Save benchmark tasks as JSON with split field and ring enumeration.

    Output format: List of tasks with minimal schema and 'split' field indicating
    which split each task belongs to (count, index, generation, ring_enum).

    Args:
        all_tasks: List of raw tasks with all original fields
        output_path: Path to save the JSON file
        ring_enumeration: Whether to include ring enumeration tasks
        seed: Random seed for reproducibility
    """
    print("\nCreating JSON output with minimal schema...")

    # Transform all tasks to minimal format
    print("  Transforming tasks to minimal schema...")
    minimal_tasks = [create_minimal_task(task) for task in all_tasks]

    # Add split field based on task_type
    for task in minimal_tasks:
        task["split"] = task["task_type"]

    # Create ring enumeration tasks if enabled
    ring_enum_tasks = []
    if ring_enumeration:
        print("  Creating ring enumeration tasks...")
        ring_enum_tasks, skipped_no_rings, skipped_enum_failed = (
            create_ring_enumeration_tasks(minimal_tasks, seed)
        )

        # Add split field to ring enum tasks
        for task in ring_enum_tasks:
            task["split"] = "ring_enum"

        if ring_enum_tasks:
            print(
                f"    Created {len(ring_enum_tasks)} ring enumeration tasks "
                f"(skipped {skipped_no_rings} without rings, "
                f"{skipped_enum_failed} failed enumeration)"
            )

    # Combine all tasks
    output_tasks = minimal_tasks + ring_enum_tasks

    # Print summary by split
    split_counts: Dict[str, int] = {}
    for task in output_tasks:
        split = task["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    print("\n  Tasks by split:")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count}")
    print(f"    Total: {len(output_tasks)}")

    # Save to JSON
    print(f"\n  Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_tasks, f, indent=2)

    print(f"  Saved {len(output_tasks)} tasks to JSON")

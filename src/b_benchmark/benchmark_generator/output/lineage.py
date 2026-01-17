"""
Task lineage export utilities.

Exports task lineage information for tracking parent-child relationships.
"""

from typing import Any, Dict, List

from ..lineage import TaskLineageBuilder, build_task_lineage_from_tasks


def export_lineage(
    all_tasks: List[Dict[str, Any]],
    output_path: str,
) -> Dict[str, Any]:
    """
    Build and export task lineage to JSON file.

    Args:
        all_tasks: List of all generated tasks
        output_path: Path to save lineage JSON

    Returns:
        Lineage data dictionary
    """
    print("\nBuilding task lineage graph...")

    lineage_builder = build_task_lineage_from_tasks(all_tasks)
    lineage_data = lineage_builder.build_lineage_data()

    print(f"  Total tasks: {lineage_data['statistics']['total_tasks']}")
    print(f"  Tasks with parents: {lineage_data['statistics']['tasks_with_parents']}")
    print(f"  Tasks with children: {lineage_data['statistics']['tasks_with_children']}")

    lineage_builder.save_lineage(output_path)
    print(f"  Saved lineage to: {output_path}")

    return lineage_data


def print_lineage_summary(lineage_data: Dict[str, Any]) -> None:
    """
    Print a summary of the lineage data.

    Args:
        lineage_data: Lineage data dictionary
    """
    stats = lineage_data.get("statistics", {})

    print("\nLineage Summary:")
    print(f"  Total tasks: {stats.get('total_tasks', 0)}")
    print(f"  Root tasks: {stats.get('root_tasks', 0)}")
    print(f"  Tasks with parents: {stats.get('tasks_with_parents', 0)}")
    print(f"  Tasks with children: {stats.get('tasks_with_children', 0)}")

    if "task_types" in stats:
        print("\n  By task type:")
        for task_type, count in sorted(stats["task_types"].items()):
            print(f"    {task_type}: {count}")

"""
Task lineage tracking utilities for benchmark dataset generation.

Provides task ID generation and relationship tracking to trace how
multi-tasks and constraint tasks are built from single tasks.
"""

from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict
import json


class UIDGenerator:
    """Generate unique sequential UIDs for tasks."""

    def __init__(self, start: int = 0):
        self.counter: int = start

    def generate(self) -> str:
        """
        Generate unique UID for a task.

        Returns:
            Unique UID string (e.g., 'task_00000001')
        """
        uid = f"task_{self.counter:08d}"
        self.counter += 1
        return uid

    def get_count(self) -> int:
        """Get current counter value."""
        return self.counter


class PropertyTaskMapping:
    """Track which tasks use which properties for lineage resolution."""

    def __init__(self):
        # property -> [(uid, smiles, task_data), ...]
        self.property_to_tasks: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        # (smiles, property) -> uid (for fast lookup)
        self.smiles_property_to_task: Dict[Tuple[str, str], str] = {}

    def add_single_task(self, uid: str, smiles: str, property_name: str, task_data: Dict[str, Any]):
        """Register a single task using a property."""
        key = (smiles, property_name)
        self.property_to_tasks[property_name].append((uid, smiles, task_data))
        self.smiles_property_to_task[key] = uid

    def find_parent_tasks(self, smiles: str, properties: List[str]) -> List[str]:
        """
        Find parent task UIDs for a multi-task based on smiles and properties used.

        Args:
            smiles: SMILES string of the molecule
            properties: List of properties used in the multi-task

        Returns:
            List of parent task UIDs that contributed these properties
        """
        parent_uids = []
        for prop in properties:
            key = (smiles, prop)
            uid = self.smiles_property_to_task.get(key)
            if uid:
                parent_uids.append(uid)
        return list(set(parent_uids))  # Deduplicate


class TaskLineageBuilder:
    """Build comprehensive task lineage graphs and mappings."""

    def __init__(self):
        self.uid_to_task: Dict[str, Dict[str, Any]] = {}
        self.uid_to_parents: Dict[str, List[str]] = defaultdict(list)
        self.uid_to_children: Dict[str, List[str]] = defaultdict(list)
        self.molecule_to_tasks: Dict[str, List[str]] = defaultdict(list)
        self.property_to_tasks: Dict[str, List[str]] = defaultdict(list)

    def add_task(self, task: Dict[str, Any]):
        """Add a task to the lineage builder."""
        uid = task.get('uid')
        if not uid:
            return

        self.uid_to_task[uid] = task

        # Track molecule usage
        smiles = task.get('original_smiles') or task.get('smiles')
        if smiles:
            self.molecule_to_tasks[smiles].append(uid)

        # Track property usage
        if 'property' in task and task['property']:
            self.property_to_tasks[task['property']].append(uid)
        if 'properties' in task and task['properties']:
            for prop in task['properties']:
                self.property_to_tasks[prop].append(uid)

        # Track parent-child relationships
        parent_uids = task.get('parent_uids', [])
        if parent_uids:
            for parent_uid in parent_uids:
                self.uid_to_parents[uid].append(parent_uid)
                self.uid_to_children[parent_uid].append(uid)

    def build_lineage_data(self) -> Dict[str, Any]:
        """
        Build comprehensive lineage data structure.

        Returns:
            Dictionary with all lineage mappings
        """
        # Build task metadata (lightweight version without full task data)
        task_metadata = {}
        for uid, task in self.uid_to_task.items():
            task_metadata[uid] = {
                'task_type': task.get('task_type'),
                'category': task.get('category'),
                'categories': task.get('categories'),
                'property': task.get('property'),
                'properties': task.get('properties'),
                'supercategory': task.get('supercategory'),
                'smiles': task.get('original_smiles') or task.get('smiles'),
                'n_properties': task.get('n_properties'),
                'n_constraints': task.get('n_constraints')
            }

        return {
            'uid_to_parents': dict(self.uid_to_parents),
            'uid_to_children': dict(self.uid_to_children),
            'molecule_to_tasks': dict(self.molecule_to_tasks),
            'property_to_tasks': dict(self.property_to_tasks),
            'task_metadata': task_metadata,
            'statistics': {
                'total_tasks': len(self.uid_to_task),
                'total_molecules': len(self.molecule_to_tasks),
                'total_properties': len(self.property_to_tasks),
                'tasks_by_type': self._count_by_type(),
                'tasks_with_parents': sum(1 for parents in self.uid_to_parents.values() if parents),
                'tasks_with_children': sum(1 for children in self.uid_to_children.values() if children)
            }
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count tasks by type."""
        counts = defaultdict(int)
        for task in self.uid_to_task.values():
            task_type = task.get('task_type', 'unknown')
            counts[task_type] += 1
        return dict(counts)

    def save_lineage(self, filepath: str):
        """Save lineage data to JSON file."""
        lineage_data = self.build_lineage_data()
        with open(filepath, 'w') as f:
            json.dump(lineage_data, f, indent=2)

    def get_task_ancestors(self, uid: str) -> Set[str]:
        """Get all ancestor task UIDs (recursive parents)."""
        ancestors = set()
        queue = [uid]
        while queue:
            current = queue.pop(0)
            parents = self.uid_to_parents.get(current, [])
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors

    def get_task_descendants(self, uid: str) -> Set[str]:
        """Get all descendant task UIDs (recursive children)."""
        descendants = set()
        queue = [uid]
        while queue:
            current = queue.pop(0)
            children = self.uid_to_children.get(current, [])
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        return descendants


def map_count_property_to_index(count_property: str) -> Optional[str]:
    """
    Map a count property to its corresponding index property.

    Args:
        count_property: Count property name (e.g., 'ring_count')

    Returns:
        Corresponding index property name or None
    """
    if count_property.endswith('_count'):
        return count_property.replace('_count', '_index')
    return None


def build_task_lineage_from_tasks(all_tasks: List[Dict[str, Any]]) -> TaskLineageBuilder:
    """
    Build task lineage from a list of tasks with parent_task_ids.

    Args:
        all_tasks: List of all generated tasks

    Returns:
        TaskLineageBuilder with complete lineage data
    """
    builder = TaskLineageBuilder()
    for task in all_tasks:
        builder.add_task(task)
    return builder

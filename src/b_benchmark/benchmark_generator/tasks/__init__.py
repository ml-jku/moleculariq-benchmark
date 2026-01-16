"""Task generation modules for different task types."""

from .single_count_index import generate_paired_single_tasks
from .multi_count_index import generate_paired_multi_tasks
from .single_constraint import generate_single_constraint_tasks
from .multi_constraint import generate_multi_constraint_tasks

__all__ = [
    "generate_paired_single_tasks",
    "generate_paired_multi_tasks",
    "generate_single_constraint_tasks",
    "generate_multi_constraint_tasks",
]

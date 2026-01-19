"""
Property utility functions for category mapping and natural language names.
"""

from typing import Any, List

import numpy as np
import pandas as pd

from moleculariq_core import (
    COUNT_MAP,
    INDEX_MAP,
    CONSTRAINT_MAP,
    COUNT_TO_INDEX_MAP,
    COUNT_MAPPINGS,
    INDEX_MAPPINGS,
)


def to_python_scalar(value: Any) -> Any:
    """
    Convert numpy scalar types to native Python types.

    Args:
        value: Value to convert

    Returns:
        Python native type
    """
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def get_property_category(property_name: str) -> str:
    """
    Get the category name for a property using explicit maps.

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
    if task_type == "count" and property_name in COUNT_MAPPINGS:
        options = COUNT_MAPPINGS[property_name]
        return options[0] if isinstance(options, list) else options
    elif task_type == "index" and property_name in INDEX_MAPPINGS:
        options = INDEX_MAPPINGS[property_name]
        return options[0] if isinstance(options, list) else options

    # Fallback: convert underscores to spaces and clean up
    name = property_name.replace("_", " ")
    name = name.replace(" count", "").replace(" index", "")
    return name


def get_count_properties(df: pd.DataFrame, include_reactions: bool = False) -> List[str]:
    """
    Get all count properties from the dataframe.

    Args:
        df: DataFrame to inspect
        include_reactions: Whether to include reaction template properties

    Returns:
        List of count property names
    """
    count_props = []
    for col in df.columns:
        if "_count" in col and "_index" not in col:
            # Exclude reaction templates unless explicitly included
            if include_reactions or not col.startswith(
                "template_based_reaction_prediction_"
            ):
                count_props.append(col)
    return count_props


def get_index_properties(df: pd.DataFrame, include_reactions: bool = False) -> List[str]:
    """
    Get all index properties from the dataframe.

    Args:
        df: DataFrame to inspect
        include_reactions: Whether to include reaction template properties

    Returns:
        List of index property names
    """
    index_props = []
    for col in df.columns:
        if "_index" in col:
            # Exclude reaction templates unless explicitly included
            if include_reactions or not col.startswith(
                "template_based_reaction_prediction_"
            ):
                index_props.append(col)
    return index_props


def convert_fg_count_to_nbr_instances(property_name: str) -> str:
    """
    Convert functional group _count property to _nbrInstances property.

    Args:
        property_name: Property name to convert

    Returns:
        Converted property name or original if not applicable
    """
    if property_name.startswith("functional_group_") and property_name.endswith(
        "_count"
    ):
        return property_name.replace("_count", "_nbrInstances")
    return property_name


def is_zero_value(value: Any) -> bool:
    """
    Check if a value is considered "zero" for constraint purposes.

    Args:
        value: Value to check

    Returns:
        True if value is zero/false, False otherwise
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 0.0
    if isinstance(value, (bool, np.bool_)):
        return value is False
    return False

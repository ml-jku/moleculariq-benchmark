"""
Sampling utilities for inverse frequency weighting.

Provides balanced sampling across property values and categories.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.properties import get_property_category


def compute_inverse_frequency_arrays(
    df: pd.DataFrame,
    property_col: str,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute arrays for inverse-frequency sampling.

    Args:
        df: DataFrame to sample from
        property_col: Column to use for weighting
        min_count: Minimum value count to include

    Returns:
        Tuple of (index_labels, normalized_weights)
    """
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
    cache_key: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Sample using inverse frequency weighting, underweighting zeros.

    Args:
        df: DataFrame to sample from
        property_col: Column to use for weighting
        n_samples: Number of samples to draw
        min_count: Minimum value count to include
        seed: Random seed
        cache: Optional cache for precomputed weights
        cache_key: Key for cache lookup

    Returns:
        Sampled DataFrame
    """
    cache_entry = None
    if cache is not None and cache_key is not None:
        cache_entry = cache.get(cache_key)

    if cache_entry is None:
        labels, weights = compute_inverse_frequency_arrays(df, property_col, min_count)
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
        weights_array = np.full(
            len(weights_array), 1.0 / len(weights_array), dtype=np.float64
        )
    else:
        weights_array /= weights_sum
    replace = n_actual > len(labels)
    choice_indices = rng_local.choice(
        len(labels), size=n_actual, replace=replace, p=weights_array
    )
    selected_labels = np.atleast_1d(labels[choice_indices])

    return df.loc[selected_labels]


def select_property_with_inverse_frequency(
    available_props: List[str],
    property_usage_counter: Dict[str, int],
    category_usage_counter: Dict[str, int],
    row_data: Dict[str, Any],
    rng: random.Random,
    apply_zero_bias: bool = True,
    ultra_hard: bool = False,
    property_category_cache: Optional[Dict[str, str]] = None,
) -> str:
    """
    Select a property using two-level inverse frequency weighting.

    First selects category with inverse frequency, then property within
    category with inverse frequency + optional zero bias.

    Args:
        available_props: List of available properties to choose from
        property_usage_counter: Dict tracking usage count per property
        category_usage_counter: Dict tracking usage count per category
        row_data: Row data containing property values for zero bias
        rng: Random number generator for reproducibility
        apply_zero_bias: Whether to apply zero/empty bias
        ultra_hard: Whether to use ultra-hard mode
        property_category_cache: Optional pre-computed property->category mapping

    Returns:
        Selected property name
    """
    if len(available_props) == 0:
        raise ValueError("No available properties to select from")

    if len(available_props) == 1:
        return available_props[0]

    # Step 1: Group properties by category
    available_categories: Dict[str, List[str]] = {}
    for prop in available_props:
        if property_category_cache is not None:
            cat = property_category_cache.get(prop, get_property_category(prop))
        else:
            cat = get_property_category(prop)
        if cat not in available_categories:
            available_categories[cat] = []
        available_categories[cat].append(prop)

    # Step 2: Select category with inverse frequency weighting
    categories = list(available_categories.keys())
    if len(categories) == 1:
        chosen_category = categories[0]
    else:
        cat_weights = [
            1.0 / (1 + category_usage_counter.get(cat, 0)) for cat in categories
        ]
        cat_weights_arr = np.array(cat_weights)
        cat_weights_arr = cat_weights_arr / cat_weights_arr.sum()
        chosen_category = rng.choices(categories, weights=cat_weights_arr.tolist(), k=1)[0]

    # Step 3: Select property within category with inverse frequency + zero bias
    props_in_category = available_categories[chosen_category]
    if len(props_in_category) == 1:
        return props_in_category[0]

    prop_weights = []
    for prop in props_in_category:
        base_weight = 1.0 / (1 + property_usage_counter.get(prop, 0))

        # Ultra-hard mode: square the weight to heavily favor rarest properties
        if ultra_hard:
            base_weight = base_weight**2

        # Apply zero bias if requested
        if apply_zero_bias and prop in row_data:
            if prop.endswith("_count"):
                # For count properties, reduce weight if value is 0
                if row_data[prop] == 0:
                    base_weight *= 0.0 if ultra_hard else 0.5
            elif prop.endswith("_index"):
                # For index properties, reduce weight if list is empty
                value = row_data[prop]
                if not isinstance(value, list) or len(value) == 0:
                    base_weight *= 0.0 if ultra_hard else 0.5

        prop_weights.append(base_weight)

    prop_weights_arr = np.array(prop_weights)
    total = prop_weights_arr.sum()
    if total <= 0:
        # All weights are zero, fall back to uniform
        prop_weights_arr = np.ones(len(prop_weights_arr)) / len(prop_weights_arr)
    else:
        prop_weights_arr = prop_weights_arr / total
    chosen_prop = rng.choices(props_in_category, weights=prop_weights_arr.tolist(), k=1)[0]

    return chosen_prop

"""
Reward functions for count tasks (single and multi).
Ground truth is provided in the target dictionary - no recalculation.
"""

import json
import re
from typing import Any, Dict, Optional, Union

from .utils import are_same_molecular_formula, parse_natural_language_property


_NUMERIC_TOLERANCE = 1e-6


def _coerce_numeric(value: Any) -> Optional[float]:
    """Convert a value to a float when possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_dict_like(data: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse inputs that should represent dictionaries."""
    if isinstance(data, dict):
        return dict(data)

    if isinstance(data, str):
        text = data.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        # Fallback: parse "key:value" pairs separated by comma/semicolon/newline
        result: Dict[str, Any] = {}
        for segment in re.split(r'[;,\n]', text):
            if ':' not in segment:
                continue
            key, value = segment.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value

        if result:
            return result

    return None


def _normalize_count_dict(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dictionary keys using natural language mappings."""
    normalized: Dict[str, Any] = {}
    for key, value in raw_dict.items():
        normalized_key = parse_natural_language_property(str(key)).strip()
        normalized[normalized_key] = value
    return normalized


def _values_match(target_value: Any, predicted_value: Any, property_name: str = "") -> bool:
    """Compare count values with numeric tolerance and special handling for molecular formulas.

    Args:
        target_value: Expected value
        predicted_value: Model's predicted value
        property_name: The normalized property name (used to detect molecular formula)
    """
    # Special handling for molecular formulas
    if 'molecular_formula' in property_name:
        target_str = str(target_value).strip()
        predicted_str = str(predicted_value).strip()
        # Use the normalized comparison that handles different orders and formats
        return are_same_molecular_formula(target_str, predicted_str)

    # Numeric comparison
    target_num = _coerce_numeric(target_value)
    predicted_num = _coerce_numeric(predicted_value)

    if target_num is not None and predicted_num is not None:
        return abs(target_num - predicted_num) <= _NUMERIC_TOLERANCE

    # Fallback to strict string comparison
    return str(target_value).strip() == str(predicted_value).strip()


def _extract_single_count_value(predicted: Union[str, int, float, Dict], target_key: str) -> Optional[Any]:
    """Extract or infer the value for a single-count prediction."""
    if isinstance(predicted, (int, float)) and not isinstance(predicted, bool):
        return predicted

    parsed: Optional[Dict[str, Any]] = None

    if isinstance(predicted, str):
        # For molecular formula properties, treat the string as the value directly
        if 'molecular_formula' in target_key:
            # First try to parse as dict
            parsed = _parse_dict_like(predicted)
            if parsed is None:
                # If not a dict, treat the string as the formula value
                return predicted
        else:
            numeric = _coerce_numeric(predicted)
            if numeric is not None:
                return numeric
            parsed = _parse_dict_like(predicted)
    elif isinstance(predicted, dict):
        parsed = dict(predicted)

    if parsed is None:
        return None

    normalized_pred = _normalize_count_dict(parsed)

    if target_key in normalized_pred:
        return normalized_pred[target_key]

    if len(normalized_pred) == 1:
        return next(iter(normalized_pred.values()))

    return None


def _parse_multi_count_prediction(predicted: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse predictions for multi-count tasks into a normalized dictionary."""
    if isinstance(predicted, dict):
        parsed = dict(predicted)
    elif isinstance(predicted, str):
        parsed = _parse_dict_like(predicted)
    else:
        parsed = None

    if parsed is None:
        return None

    normalized = _normalize_count_dict(parsed)
    return normalized if normalized else None


def multi_count_dict_reward(
    predicted: Union[str, int, float, Dict],
    target: Union[str, Dict],
    *,
    return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Reward function for both single-count and multi-count tasks.
    Target dict IS the ground truth - we don't recalculate.

    When ``return_details`` is ``False`` (default), the function returns a binary
    reward: ``1.0`` if *every* requested property matches exactly, else ``0.0``.

    When ``return_details`` is ``True`` the function returns a dictionary with:

    ``reward``
        Binary reward (same as described above).
    ``details``
        Mapping of normalized property keys to a structure containing the target
        value, the predicted value (if available), and a boolean match flag.
    ``matched`` / ``total``
        Number of matching properties and the total requested.
    ``extra_predictions``
        Any normalized prediction keys that were not part of the target.

    For single-count tasks (target has 1 key), accepts:
    - Just a number: "5" or 5
    - Dict format: {"carbon": 5} or '{"carbon": 5}'

    For multi-count tasks, accepts:
    - Dict format: {"carbon": 5, "oxygen": 2}
    - JSON string: '{"carbon": 5, "oxygen": 2}'

    Args:
        predicted: Number, dictionary, or JSON string
        target: Dictionary {property: count} or JSON string
        return_details: If True, return the structured dictionary described above
            instead of a float

    Returns:
        Union[float, Dict[str, Any]]: Either the binary reward or a detail
        dictionary (see above).
    """
    target_dict_raw = _parse_dict_like(target)
    if target_dict_raw is None:
        return 0.0 if not return_details else {
            "reward": 0.0,
            "details": {},
            "matched": 0,
            "total": 0,
            "extra_predictions": {}
        }

    norm_target = _normalize_count_dict(target_dict_raw)
    if not norm_target:
        return 0.0 if not return_details else {
            "reward": 0.0,
            "details": {},
            "matched": 0,
            "total": 0,
            "extra_predictions": {}
        }

    # Single-count tasks
    if len(norm_target) == 1:
        target_key, target_value = next(iter(norm_target.items()))
        pred_value = _extract_single_count_value(predicted, target_key)
        match = False
        if pred_value is not None:
            match = _values_match(target_value, pred_value, target_key)

        reward = 1.0 if match else 0.0

        if return_details:
            details = {
                target_key: {
                    "target": target_value,
                    "predicted": pred_value,
                    "match": match
                }
            }
            # Capture extra predictions when dict-like input contains other keys
            norm_pred = _parse_multi_count_prediction(predicted)
            extra_predictions = {}
            if norm_pred:
                for key, value in norm_pred.items():
                    if key != target_key:
                        extra_predictions[key] = value

            return {
                "reward": reward,
                "details": details,
                "matched": 1 if match else 0,
                "total": 1,
                "extra_predictions": extra_predictions
            }

        return reward

    # Multi-count tasks
    norm_pred = _parse_multi_count_prediction(predicted)
    if not norm_pred:
        if return_details:
            details = {
                key: {
                    "target": value,
                    "predicted": None,
                    "match": False
                }
                for key, value in norm_target.items()
            }
            return {
                "reward": 0.0,
                "details": details,
                "matched": 0,
                "total": len(norm_target),
                "extra_predictions": {}
            }
        return 0.0

    details: Dict[str, Dict[str, Any]] = {}
    matched = 0

    for key, target_value in norm_target.items():
        predicted_value = norm_pred.get(key)
        match = (
            predicted_value is not None
            and _values_match(target_value, predicted_value, key)
        )
        if match:
            matched += 1
        details[key] = {
            "target": target_value,
            "predicted": predicted_value,
            "match": match
        }

    reward = 1.0 if matched == len(norm_target) else 0.0

    if return_details:
        extra_predictions = {
            key: value
            for key, value in norm_pred.items()
            if key not in norm_target
        }
        return {
            "reward": reward,
            "details": details,
            "matched": matched,
            "total": len(norm_target),
            "extra_predictions": extra_predictions
        }

    return reward


# Alias for backward compatibility
def single_count_reward(
    predicted: Union[str, int, float],
    target: Dict,
    *,
    return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Wrapper for single count tasks that proxies to ``multi_count_dict_reward``.

    Args:
        predicted: Predicted count value
        target: Dictionary with single property-count pair
        return_details: If True, return the full detail dictionary described in
            ``multi_count_dict_reward``

    Returns:
        Union[float, Dict[str, Any]]: Binary reward or detail dictionary.
    """
    return multi_count_dict_reward(
        predicted,
        target,
        return_details=return_details
    )

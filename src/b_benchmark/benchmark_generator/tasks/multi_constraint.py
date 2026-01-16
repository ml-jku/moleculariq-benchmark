"""
Multi-constraint task generation using the turbo approach.

This uses the sophisticated constraint scoring and parallel generation
from the intelligent turbo generator.
"""

import json
import random
import pickle
import tempfile
import platform
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from questions import TASKS
from natural_language.formatter import format_constraint
from B_create_benchmark.task_lineage import UIDGenerator

from ..utils.smiles import canonicalize_smiles
from ..utils.properties import get_property_category, to_python_scalar
from ..core.indexing import (
    build_constraint_index,
    has_conflicting_properties,
    has_duplicate_properties,
)
from ..core.scoring import (
    score_constraint_combination,
    zero_is_informative,
    count_zeros_in_constraints,
    get_k_thresholds,
    get_k_min_intersection,
    get_rr_cutoff_for_k,
    combo_norm_props,
)


# ---------------------------------------------------------------------------
# Globals shared by worker processes
# ---------------------------------------------------------------------------
_GLOBAL_INDEX: Dict[Tuple[str, Any], Set[str]] = None
_GLOBAL_ALL_CONSTRAINTS: List[Tuple[str, Any]] = None
_GLOBAL_FILTERED: List[Tuple[str, Any]] = None
_GLOBAL_TOTAL_MOLS: int = 0
_GLOBAL_PRESCREEN_SIZE: int = 5000
_GLOBAL_CAND_POOL_SIZE: int = 800
_GLOBAL_MIN_INTERSECTION: int = 10


def _init_worker(
    index_path: Optional[str],
    filtered_path: Optional[str],
    total_molecules: int,
    prescreen_size: int,
    cand_pool_size: int,
    min_intersection: int,
):
    """Initializer for 'spawn'/'forkserver' contexts."""
    global _GLOBAL_INDEX, _GLOBAL_ALL_CONSTRAINTS, _GLOBAL_FILTERED, _GLOBAL_TOTAL_MOLS
    global _GLOBAL_PRESCREEN_SIZE, _GLOBAL_CAND_POOL_SIZE, _GLOBAL_MIN_INTERSECTION
    if index_path is not None:
        with open(index_path, "rb") as f:
            _GLOBAL_INDEX = pickle.load(f)
        _GLOBAL_ALL_CONSTRAINTS = list(_GLOBAL_INDEX.keys())
    if filtered_path is not None:
        with open(filtered_path, "rb") as f:
            _GLOBAL_FILTERED = pickle.load(f)
    _GLOBAL_TOTAL_MOLS = int(total_molecules)
    _GLOBAL_PRESCREEN_SIZE = int(prescreen_size)
    _GLOBAL_CAND_POOL_SIZE = int(cand_pool_size)
    _GLOBAL_MIN_INTERSECTION = int(min_intersection)


def _score_seed_constraint_worker(args: Tuple) -> Tuple[str, Tuple[str, Any], List]:
    """Worker function for parallel constraint scoring."""
    (smiles, seed_constraint, k, max_special, top_n, max_attempts, worker_seed) = args

    rng = random.Random(worker_seed)

    constraint_index = _GLOBAL_INDEX
    all_pool = _GLOBAL_FILTERED or _GLOBAL_ALL_CONSTRAINTS
    total_molecules = _GLOBAL_TOTAL_MOLS
    prescreen_size = _GLOBAL_PRESCREEN_SIZE
    cand_pool_size = _GLOBAL_CAND_POOL_SIZE
    min_intersection = _GLOBAL_MIN_INTERSECTION
    k_min = get_k_min_intersection(k, min_intersection)

    seed_set = constraint_index.get(seed_constraint, set())
    if not seed_set:
        return (smiles, seed_constraint, [])

    seed_prop = seed_constraint[0]

    # Two-stage pre-screen
    if len(all_pool) <= prescreen_size:
        subset = list(all_pool)
    else:
        subset = rng.sample(all_pool, prescreen_size)

    good_cands: List[Tuple[Tuple[str, Any], float]] = []
    for c in subset:
        if c[0] == seed_prop:
            continue
        s = constraint_index.get(c, set())
        if not s:
            continue
        inter = seed_set & s
        if len(inter) < k_min:
            continue
        rr = min(len(seed_set), len(s)) / len(inter)
        rr_cut = get_rr_cutoff_for_k(k)
        if rr >= rr_cut:
            good_cands.append((c, rr))

    if not good_cands:
        good_cands = [(c, 1.0) for c in subset if c[0] != seed_prop]

    good_cands.sort(key=lambda x: x[1], reverse=True)
    cand_pool = [c for c, _ in good_cands[:cand_pool_size]]

    # Cache seed intersections
    seed_cap = {}
    for c in cand_pool:
        s = constraint_index.get(c, set())
        inter = seed_set & s
        if len(inter) >= k_min:
            seed_cap[c] = inter
    if not seed_cap:
        return (smiles, seed_constraint, [])
    cand_pool = list(seed_cap.keys())

    if not cand_pool:
        return (smiles, seed_constraint, [])

    scored_combinations = []
    seen_combos: Set[FrozenSet] = set()
    attempts = 0

    while attempts < max_attempts and len(scored_combinations) < top_n * 2:
        attempts += 1

        candidates = []
        props_seen = {seed_prop}
        tries = 0
        target_needed = k - 1
        while len(candidates) < target_needed and tries < 5 * target_needed:
            tries += 1
            c = rng.choice(cand_pool)
            p = c[0]
            if p in props_seen:
                continue
            props_seen.add(p)
            candidates.append(c)

        if len(candidates) < target_needed:
            continue

        combo = tuple([seed_constraint] + candidates)
        combo_fs = frozenset(combo)
        if combo_fs in seen_combos:
            continue
        seen_combos.add(combo_fs)

        props = [c[0] for c in combo]
        if has_conflicting_properties(props):
            continue
        if has_duplicate_properties(list(combo)):
            continue

        thr = get_k_thresholds(k)

        # Quick intersection chain
        current = seed_cap[candidates[0]]
        ok = True
        for c in candidates[1:]:
            current = current & constraint_index[c]
            if not current:
                ok = False
                break
        if not ok:
            continue
        n_sat_quick = len(current)
        if n_sat_quick < thr["min_molecules"] or n_sat_quick > thr["max_molecules"]:
            continue

        if not zero_is_informative(list(combo), constraint_index, min_ratio=1.4):
            continue

        score, n_sat, match_rate = score_constraint_combination(
            list(combo), constraint_index, total_molecules,
            zero_limit=1, max_special=max_special, **thr
        )

        if score > 0:
            scored_combinations.append((list(combo), score, n_sat, match_rate))

    if scored_combinations:
        scored_combinations.sort(key=lambda x: x[1], reverse=True)
        scored_combinations = scored_combinations[:top_n]

    return (smiles, seed_constraint, scored_combinations)


def extract_constraints_from_single_tasks(
    single_constraint_tasks: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[str, Any]]]:
    """Extract constraints from single-constraint tasks."""
    print("Extracting constraints from single-constraint tasks...")
    smiles_to_constraints: Dict[str, List[Tuple[str, Any]]] = {}

    for task in tqdm(single_constraint_tasks, desc="  Processing tasks"):
        smiles = task.get("answer") or task.get("original_smiles") or task.get("smiles")
        if not smiles:
            continue

        canon_smiles = canonicalize_smiles(smiles)
        if not canon_smiles:
            continue

        constraints_raw = task.get("constraints")
        if not constraints_raw:
            continue

        try:
            constraints = json.loads(constraints_raw) if isinstance(constraints_raw, str) else constraints_raw
            if len(constraints) == 1:
                c = constraints[0]
                prop = c.get("property")
                value = to_python_scalar(c.get("value"))
                if prop and value is not None:
                    if canon_smiles not in smiles_to_constraints:
                        smiles_to_constraints[canon_smiles] = []
                    smiles_to_constraints[canon_smiles].append((prop, value))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    print(f"  Extracted seed constraints for {len(smiles_to_constraints):,} molecules")
    return smiles_to_constraints


def find_best_combinations_from_global_space(
    smiles_to_constraints: Dict[str, List[Tuple[str, Any]]],
    constraint_index: Dict[Tuple[str, Any], Set[str]],
    properties_df: pd.DataFrame,
    k: int,
    max_special: int,
    top_n: int = 50,
    n_workers: int = 60,
    base_seed: int = 42,
    min_support: int = 10,
    max_support_fraction: float = 0.35,
    prescreen_size: int = 5000,
    cand_pool_size: int = 800,
    min_intersection: int = 10,
    chunksize: int = 8,
) -> Dict[str, List[Tuple[List[Tuple[str, Any]], float, int]]]:
    """Find best constraint combinations using parallel scoring."""
    print(f"\nFinding best {k}-constraint combinations from GLOBAL constraint space...")

    # Flatten seeds
    all_seeds: List[Tuple[str, Tuple[str, Any]]] = []
    for smiles, constraints in smiles_to_constraints.items():
        for c in constraints:
            all_seeds.append((smiles, c))

    print(f"  Starting with {len(all_seeds):,} seed constraints from {len(smiles_to_constraints):,} molecules")
    print(f"  Global constraint space: {len(constraint_index):,} unique (property, value) pairs")
    print(f"  Using {n_workers} parallel workers...")

    # Build filtered pool by support
    total_molecules = len(properties_df)
    support = {c: len(s) for c, s in constraint_index.items()}
    max_support = int(max_support_fraction * total_molecules)
    filtered_pool = [c for c, n in support.items() if min_support <= n <= max_support]

    def freq_score(n):
        f = n / max(total_molecules, 1)
        return -abs(f - 0.05)
    filtered_pool.sort(key=lambda c: freq_score(support[c]), reverse=True)

    print(f"  Filtered candidate pool: {len(filtered_pool):,} of {len(constraint_index):,} total")

    # Prepare worker args
    top_combos = top_n
    max_attempts_per_seed = 2000 if k == 5 else (800 if filtered_pool else 1500)
    worker_args = []
    for i, (smiles, seed) in enumerate(all_seeds):
        worker_args.append((
            smiles, seed, k, max_special, top_combos, max_attempts_per_seed, base_seed + i
        ))

    # Setup multiprocessing
    prefer_fork = (platform.system() != "Windows")
    created_temp_files = []

    if prefer_fork:
        global _GLOBAL_INDEX, _GLOBAL_ALL_CONSTRAINTS, _GLOBAL_FILTERED, _GLOBAL_TOTAL_MOLS
        global _GLOBAL_PRESCREEN_SIZE, _GLOBAL_CAND_POOL_SIZE, _GLOBAL_MIN_INTERSECTION
        _GLOBAL_INDEX = constraint_index
        _GLOBAL_ALL_CONSTRAINTS = list(constraint_index.keys())
        _GLOBAL_FILTERED = filtered_pool
        _GLOBAL_TOTAL_MOLS = total_molecules
        _GLOBAL_PRESCREEN_SIZE = prescreen_size
        _GLOBAL_CAND_POOL_SIZE = cand_pool_size
        _GLOBAL_MIN_INTERSECTION = min_intersection

        ctx = mp.get_context("fork")
        executor_kwargs = dict(mp_context=ctx, initializer=None, initargs=())
    else:
        idx_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        flt_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        idx_tmp.close()
        flt_tmp.close()
        with open(idx_tmp.name, "wb") as f:
            pickle.dump(constraint_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(flt_tmp.name, "wb") as f:
            pickle.dump(filtered_pool, f, protocol=pickle.HIGHEST_PROTOCOL)
        created_temp_files.extend([idx_tmp.name, flt_tmp.name])

        ctx = mp.get_context("spawn")
        executor_kwargs = dict(
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(idx_tmp.name, flt_tmp.name, total_molecules, prescreen_size, cand_pool_size, min_intersection)
        )

    # Parallel execution
    seed_combinations: Dict[Tuple[str, Tuple[str, Any]], List] = {}

    try:
        with ProcessPoolExecutor(max_workers=n_workers, **executor_kwargs) as executor:
            results = list(tqdm(
                executor.map(_score_seed_constraint_worker, worker_args, chunksize=chunksize),
                total=len(worker_args), desc=f"  Scoring k={k}"
            ))
    finally:
        import os
        for p in created_temp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

    # Aggregate
    for smiles, seed_constraint, scored in results:
        if scored:
            seed_combinations[(smiles, seed_constraint)] = [
                (combo, score, n_sat) for (combo, score, n_sat, _mr) in scored
            ]

    print(f"  Generated combinations for {len(seed_combinations):,} seed entries")

    # Group per molecule
    molecule_combinations: Dict[str, List] = {}
    for (smiles, _seed), combos in seed_combinations.items():
        if smiles not in molecule_combinations:
            molecule_combinations[smiles] = []
        for combo, score, n_sat in combos:
            molecule_combinations[smiles].append((combo, score, n_sat))

    for smiles in molecule_combinations:
        molecule_combinations[smiles].sort(key=lambda x: x[1], reverse=True)
        molecule_combinations[smiles] = molecule_combinations[smiles][:top_n]

    print(f"  Found combinations for {len(molecule_combinations):,} molecules")

    # Stats
    if molecule_combinations:
        all_scores = [score for combos in molecule_combinations.values() for (_c, score, _n) in combos]
        all_nsat = [n for combos in molecule_combinations.values() for (_c, _s, n) in combos]
        if all_scores:
            print(f"  Score stats: min={min(all_scores):.2f}, max={max(all_scores):.2f}, "
                  f"mean={np.mean(all_scores):.2f}, median={np.median(all_scores):.2f}")
        if all_nsat:
            print(f"  Satisfying mols stats: min={min(all_nsat)}, max={max(all_nsat)}, "
                  f"mean={np.mean(all_nsat):.1f}, median={int(np.median(all_nsat))}")

    return molecule_combinations


def _format_constraints_nl(
    templates: List[str],
    constraints: List[Dict[str, Any]],
    rng: random.Random,
    include_key_hint: bool = True,
) -> Tuple[str, str]:
    """Format constraints as natural language question."""
    parts = []
    for constraint in constraints:
        data = {"type": constraint["property"], "operator": constraint["operator"]}
        if constraint["operator"] == "range":
            data["min_value"] = constraint["min_value"]
            data["max_value"] = constraint["max_value"]
        else:
            data["value"] = constraint["value"]
        parts.append(format_constraint(data, use_varied_phrasing=False))

    if len(parts) == 1:
        constraint_text = parts[0]
    elif len(parts) == 2:
        constraint_text = f"{parts[0]} and {parts[1]}"
    else:
        constraint_text = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    question = rng.choice(templates).format(constraint=constraint_text)

    if include_key_hint:
        key_hints = [
            " Return the result as a JSON object using the key `smiles`.",
            " Respond with a JSON mapping whose key is `smiles`.",
            " Provide your answer as JSON with `smiles` as the key.",
            " Give the generated molecule in a JSON object keyed by `smiles`.",
        ]
        question += rng.choice(key_hints)

    return question, constraint_text


def generate_multi_constraint_tasks(
    single_constraint_tasks: List[Dict[str, Any]],
    properties_df: pd.DataFrame,
    n_tasks_by_k: Dict[int, int],
    seed: int = 42,
    top_combinations_per_molecule: int = 50,
    n_workers: int = 60,
    min_support: int = 10,
    max_support_fraction: float = 0.35,
    prescreen_size: int = 5000,
    cand_pool_size: int = 800,
    min_intersection: int = 10,
    chunksize: int = 8,
    prop_caps: Optional[Dict[int, Dict[str, float]]] = None,
    uid_generator: Optional[UIDGenerator] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multi-constraint tasks using the turbo approach.

    Args:
        single_constraint_tasks: Single constraint tasks as seeds
        properties_df: Properties dataframe
        n_tasks_by_k: Dict mapping k to number of tasks
        seed: Random seed
        top_combinations_per_molecule: Top combinations to keep per molecule
        n_workers: Number of parallel workers
        min_support: Minimum support for constraints
        max_support_fraction: Maximum support fraction
        prescreen_size: Prescreen sample size
        cand_pool_size: Candidate pool size
        min_intersection: Minimum intersection size
        chunksize: Chunk size for parallel execution
        prop_caps: Optional property caps
        uid_generator: UID generator

    Returns:
        List of multi-constraint tasks
    """
    print("=" * 80)
    print("MULTI-CONSTRAINT TASK GENERATION (TURBO)")
    print("=" * 80)

    # Build canonical smiles lookup
    if "canonical_smiles" not in properties_df.columns:
        properties_df = properties_df.copy()
        if "original_smiles" in properties_df.columns and properties_df["original_smiles"].notna().any():
            properties_df["canonical_smiles"] = properties_df["original_smiles"]
        else:
            properties_df["canonical_smiles"] = properties_df["smiles"].map(canonicalize_smiles)

    smi2row = {smi: idx for idx, smi in enumerate(properties_df["canonical_smiles"])}

    # Phase 1: Index
    print("\n[PHASE 1] Building constraint satisfaction index...")
    constraint_index = build_constraint_index(properties_df)

    # Phase 2: Extract seeds and score
    print("\n[PHASE 2] Scoring constraint combinations...")
    smiles_to_constraints = extract_constraints_from_single_tasks(single_constraint_tasks)
    if not smiles_to_constraints:
        print("  ERROR: No molecules with single-constraint seeds found!")
        return []

    all_combinations: Dict[int, Dict[str, List]] = {}

    for k in sorted(n_tasks_by_k.keys()):
        print(f"\n  Scoring combinations for k={k}...")
        max_special = 1 if k in (2, 3) else 3

        combinations_for_k = find_best_combinations_from_global_space(
            smiles_to_constraints=smiles_to_constraints,
            constraint_index=constraint_index,
            properties_df=properties_df,
            k=k,
            max_special=max_special,
            top_n=top_combinations_per_molecule,
            n_workers=n_workers,
            base_seed=seed + 17 * k,
            min_support=min_support,
            max_support_fraction=max_support_fraction,
            prescreen_size=prescreen_size,
            cand_pool_size=cand_pool_size,
            min_intersection=min_intersection,
            chunksize=chunksize,
        )
        all_combinations[k] = combinations_for_k

    # Phase 3: Generate tasks
    print("\n[PHASE 3] Generating tasks from high-quality combinations...")
    generated_tasks: List[Dict[str, Any]] = []
    templates = TASKS["constraint_generation"]["question_templates"]
    total_molecules = len(properties_df)

    for k in sorted(n_tasks_by_k.keys()):
        n_tasks = n_tasks_by_k[k]
        molecule_combinations = all_combinations[k]

        if not molecule_combinations:
            print(f"  ERROR: No valid combinations found for k={k}")
            continue

        # Build sampling pool
        sampling_pool: List[Tuple[str, List, float, int]] = []
        for smiles, combos in molecule_combinations.items():
            for combo, score, n_sat in combos:
                if n_sat > 0:
                    sampling_pool.append((smiles, combo, score, n_sat))

        sampling_pool.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Generating {n_tasks} tasks with k={k} constraints...")
        print(f"  Sampling pool: {len(sampling_pool)} pairs")

        cap_counters = Counter()
        cap_limits = {}
        if prop_caps and k in prop_caps:
            for prop, frac in prop_caps[k].items():
                cap_limits[prop] = int(round(frac * n_tasks))

        rng = random.Random(seed + 101 * k)
        generated = []
        seen = set()
        attempts = 0
        max_attempts = n_tasks * 35 if k == 5 else n_tasks * 10

        with tqdm(total=n_tasks, desc=f"  k={k} tasks") as pbar:
            while len(generated) < n_tasks and attempts < max_attempts and sampling_pool:
                attempts += 1

                # Exponential bias toward top
                idx = int(rng.expovariate(1.0 / max(1, int(len(sampling_pool) * 0.3))))
                idx = min(idx, len(sampling_pool) - 1)
                canon_smiles, constraints, score, n_sat = sampling_pool[idx]

                if count_zeros_in_constraints(constraints) > 1:
                    continue

                sig = (canon_smiles, frozenset(constraints))
                if sig in seen:
                    continue
                seen.add(sig)

                if cap_limits:
                    norm_props = combo_norm_props(constraints)
                    if any((p in cap_limits) and (cap_counters[p] >= cap_limits[p]) for p in norm_props):
                        continue

                # Build task
                row_idx = smi2row.get(canon_smiles)
                if row_idx is None:
                    continue
                molecule_row = properties_df.iloc[row_idx]

                constraint_dicts = [{"property": p, "operator": "=", "value": v} for p, v in constraints]

                question, natural_language_text = _format_constraints_nl(templates, constraint_dicts, rng)
                categories = [get_property_category(c["property"]) for c in constraint_dicts]

                uid = uid_generator.generate() if uid_generator else f"task_{len(generated):08d}"

                task = {
                    "task_type": "multi_constraint_generation",
                    "uid": uid,
                    "parent_uids": [],
                    "question": question,
                    "smiles": None,  # No input molecule for generation tasks
                    "constraints": json.dumps(constraint_dicts),
                    "n_constraints": k,
                    "categories": categories,
                    "complexity": None,  # No input molecule to measure complexity
                    "complexity_bin": None,  # No input molecule
                    "supercategory": f"multi_constraint_generation_{k}_constraints",
                    "natural_language_answer": natural_language_text,
                    "original_smiles": canon_smiles,
                    "iupac_name": None,
                    "target": None,
                    "category": None,
                    "property": None,
                    "properties": None,
                    "n_properties": None,
                    "answer": None,
                    "transformation_type": None,
                    "n_satisfying_molecules": n_sat,
                    "prevalence": n_sat / total_molecules if total_molecules > 0 else 0.0,
                }

                generated.append(task)
                pbar.update(1)

                if cap_limits:
                    for p in combo_norm_props(constraints):
                        if p in cap_limits:
                            cap_counters[p] += 1

        generated_tasks.extend(generated)
        print(f"  Generated {len(generated)}/{n_tasks} tasks for k={k}")

    # Add parent UIDs
    _add_parent_uids_to_multi_constraint_tasks(generated_tasks, single_constraint_tasks)

    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE: {len(generated_tasks)} total tasks")
    print("=" * 80)

    return generated_tasks


def _add_parent_uids_to_multi_constraint_tasks(
    multi_constraint_tasks: List[Dict[str, Any]],
    single_constraint_tasks: List[Dict[str, Any]],
) -> None:
    """Add parent UIDs to multi-constraint tasks."""
    print("\n[POST-PROCESSING] Adding parent UIDs...")

    # Build mapping
    constraint_to_uid: Dict[Tuple[str, str, Any], str] = {}
    for task in single_constraint_tasks:
        uid = task.get("uid")
        if not uid:
            continue
        smiles = task.get("original_smiles")
        if not smiles:
            continue

        constraints_raw = task.get("constraints")
        if not constraints_raw:
            continue

        try:
            constraints = json.loads(constraints_raw) if isinstance(constraints_raw, str) else constraints_raw
            if len(constraints) != 1:
                continue
            constraint = constraints[0]
            prop = constraint.get("property")
            value = to_python_scalar(constraint.get("value"))
            if prop and value is not None:
                key = (smiles, prop, value)
                constraint_to_uid[key] = uid
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    print(f"  Built mapping for {len(constraint_to_uid):,} unique constraints")

    # Update tasks
    updated_count = 0
    for task in multi_constraint_tasks:
        smiles = task.get("original_smiles")
        if not smiles:
            continue

        constraints_raw = task.get("constraints")
        if not constraints_raw:
            continue

        try:
            constraints = json.loads(constraints_raw) if isinstance(constraints_raw, str) else constraints_raw
            parent_uids = []
            for constraint in constraints:
                prop = constraint.get("property")
                value = to_python_scalar(constraint.get("value"))
                if prop and value is not None:
                    key = (smiles, prop, value)
                    parent_uid = constraint_to_uid.get(key)
                    if parent_uid:
                        parent_uids.append(parent_uid)

            if parent_uids:
                task["parent_uids"] = list(set(parent_uids))
                updated_count += 1
            else:
                task["parent_uids"] = []
        except (json.JSONDecodeError, TypeError, KeyError):
            task["parent_uids"] = []

    print(f"  Updated {updated_count:,} tasks with parent UIDs")

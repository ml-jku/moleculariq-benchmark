#!/usr/bin/env python3
"""
Benchmark Dataset Generator - Main Entry Point

This is the single entry point for generating the complete benchmark dataset.
It uses the superior multi-constraint logic from the turbo generator.

Usage:
    python -m src.B_create_benchmark.benchmark_generator.main --help

Example:
    python -m src.B_create_benchmark.benchmark_generator.main \
        --pickle-path /path/to/properties.pkl \
        --output-path /path/to/output.json \
        --save-local /path/to/hf_dataset \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from B_create_benchmark.task_lineage import UIDGenerator, PropertyTaskMapping

from .config import BenchmarkConfig
from .tasks.single_count_index import generate_paired_single_tasks
from .tasks.multi_count_index import generate_paired_multi_tasks
from .tasks.single_constraint import generate_single_constraint_tasks
from .tasks.multi_constraint import generate_multi_constraint_tasks
from .output.huggingface import create_huggingface_dataset
from .output.lineage import export_lineage


def load_and_prepare_data(pickle_path: str) -> pd.DataFrame:
    """
    Load molecular properties and add complexity bins.

    Args:
        pickle_path: Path to the properties pickle file

    Returns:
        DataFrame with complexity_bin column added
    """
    with tqdm(total=2, desc="Loading data", unit="step") as pbar:
        df = pd.read_pickle(pickle_path)
        pbar.update(1)
        pbar.set_description("Preparing bins")

        complexity_lims = [0, 250, 1000, np.inf]
        df["complexity_bin"] = pd.cut(
            df["complexity"],
            bins=complexity_lims,
            labels=["0-250", "250-1000", "1000-inf"],
            right=False,
        )
        pbar.update(1)

    return df


def generate_benchmark_dataset(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Generate complete benchmark dataset.

    Args:
        config: Configuration object

    Returns:
        List of all generated tasks
    """
    print("=" * 80)
    print("BENCHMARK DATASET GENERATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Pickle path: {config.pickle_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  Seed: {config.seed}")
    print(f"  Ultra-hard mode: {config.ultra_hard}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    df = load_and_prepare_data(config.pickle_path)
    print(f"Loaded {len(df)} molecules with {len(df.columns)} properties")

    # Subsample for testing
    if config.subsample is not None and len(df) > config.subsample:
        df = df.sample(n=config.subsample, random_state=config.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} molecules for testing")

    # Initialize UID generator and property mapping
    uid_generator = UIDGenerator()
    property_mapping = PropertyTaskMapping()

    # ==================================================================
    # Phase 1: Generate single count/index tasks
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Generating paired single tasks (count + index)")
    print("=" * 60)

    single_tasks, molecule_property_mapping, property_mapping = generate_paired_single_tasks(
        df,
        n_samples_per_bin=config.n_samples_per_bin_single,
        seed=config.seed,
        prime_workers=config.sampling_prime_workers,
        uid_generator=uid_generator,
        ultra_hard=config.ultra_hard,
        convert_fg_properties=config.convert_fg_properties,
        fg_conversion_probability=config.fg_conversion_probability,
    )

    single_count_tasks = [t for t in single_tasks if t["task_type"] == "single_count"]
    single_index_tasks = [t for t in single_tasks if t["task_type"] == "single_index"]

    print(f"  Generated {len(single_count_tasks)} count tasks")
    print(f"  Generated {len(single_index_tasks)} index tasks")
    print(f"  Tracking {len(molecule_property_mapping)} molecules")

    # ==================================================================
    # Phase 2: Generate multi count/index tasks
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 2: Generating paired multi-tasks (n_properties={config.n_properties_list})")
    print("=" * 60)

    multi_count, multi_index = generate_paired_multi_tasks(
        df,
        molecule_property_mapping,
        n_samples_per_bin=config.n_samples_per_bin_multi,
        n_properties=config.n_properties_list,
        seed=config.seed,
        uid_generator=uid_generator,
        property_mapping=property_mapping,
        ultra_hard=config.ultra_hard,
        convert_fg_properties=config.convert_fg_properties,
        fg_conversion_probability=config.fg_conversion_probability,
    )

    print(f"  Generated {len(multi_count)} multi-count tasks")
    print(f"  Generated {len(multi_index)} multi-index tasks")
    print(f"  Perfect pairing: {len(multi_count) == len(multi_index)}")

    # ==================================================================
    # Phase 3: Generate single constraint tasks
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Generating single constraint tasks")
    print("=" * 60)

    single_constraint_tasks, single_constraint_exact_seeds = generate_single_constraint_tasks(
        single_count_tasks,
        df,
        n_samples_per_category=config.n_samples_per_category_constraint,
        seed=config.seed,
        variant_mode=config.single_constraint_variants,
        return_exact_seeds=True,
        uid_generator=uid_generator,
    )

    print(f"  Generated {len(single_constraint_tasks)} single constraint tasks")
    exact_single = sum(
        1 for task in single_constraint_tasks if task["constraints"][0]["operator"] == "="
    )
    flexible_single = len(single_constraint_tasks) - exact_single
    print(f"    Breakdown: exact={exact_single}, flexible={flexible_single}")

    # ==================================================================
    # Phase 4: Generate multi-constraint tasks (TURBO!)
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Generating multi-constraint tasks (TURBO)")
    print("=" * 60)

    multi_constraint_tasks = generate_multi_constraint_tasks(
        single_constraint_exact_seeds,
        df,
        n_tasks_by_k=config.n_tasks_per_k,
        seed=config.seed,
        top_combinations_per_molecule=config.top_combinations_per_molecule,
        n_workers=config.n_workers,
        min_support=config.min_support,
        max_support_fraction=config.max_support_fraction,
        prescreen_size=config.prescreen_size,
        cand_pool_size=config.cand_pool_size,
        min_intersection=config.min_intersection,
        chunksize=config.chunksize,
        prop_caps=config.prop_caps,
        uid_generator=uid_generator,
    )

    print(f"  Total multi-constraint tasks: {len(multi_constraint_tasks)}")

    # Combine all tasks
    all_tasks = (
        single_tasks
        + multi_count
        + multi_index
        + single_constraint_tasks
        + multi_constraint_tasks
    )

    # Shuffle
    random.Random(config.seed).shuffle(all_tasks)

    return all_tasks


def print_statistics(all_tasks: List[Dict[str, Any]]) -> None:
    """Print task statistics."""
    print("\n" + "=" * 60)
    print("TASK STATISTICS")
    print("=" * 60)

    # Task type distribution
    task_types: Dict[str, int] = {}
    for task in all_tasks:
        task_type = task["task_type"]
        task_types[task_type] = task_types.get(task_type, 0) + 1

    print("\nBy task type:")
    for task_type, count in sorted(task_types.items()):
        print(f"  {task_type}: {count}")

    # Complexity bin distribution (only for tasks with input molecules)
    print("\nBy complexity bin (non-generation tasks):")
    complexity_bins: Dict[str, int] = {}
    n_no_complexity = 0
    for task in all_tasks:
        bin_name = task.get("complexity_bin")
        if bin_name is None:
            n_no_complexity += 1
        else:
            complexity_bins[bin_name] = complexity_bins.get(bin_name, 0) + 1

    for bin_name, count in sorted(complexity_bins.items()):
        print(f"  {bin_name}: {count}")
    if n_no_complexity > 0:
        print(f"  (generation tasks without complexity: {n_no_complexity})")

    # Multi-constraint breakdown
    if "multi_constraint_generation" in task_types:
        print("\nMulti-constraint by constraint count:")
        constraint_counts: Dict[int, int] = {}
        for task in all_tasks:
            if task["task_type"] == "multi_constraint_generation":
                n_constraints = task.get("n_constraints", 0)
                constraint_counts[n_constraints] = constraint_counts.get(n_constraints, 0) + 1
        for n, count in sorted(constraint_counts.items()):
            print(f"  {n} constraints: {count} tasks")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate molecular reasoning benchmark dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    parser.add_argument(
        "--pickle-path",
        type=str,
        default="/system/user/publicdata/chemical_reasoning/moleculariq/properties_new.pkl",
        help="Path to properties pickle file",
    )

    # Sampling parameters
    parser.add_argument(
        "--n-samples-per-bin",
        type=int,
        default=10,
        help="Samples per property per complexity bin for single tasks",
    )
    parser.add_argument(
        "--n-samples-per-category",
        type=int,
        default=30,
        help="Samples per category for constraint tasks",
    )
    parser.add_argument(
        "--n-samples-multi",
        type=int,
        default=100,
        help="Samples per complexity bin for multi-property tasks",
    )
    parser.add_argument(
        "--ultra-hard",
        action="store_true",
        help="Generate ultra-hard variant (rarest properties, non-zero values)",
    )

    # Output options
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Path to save HuggingFace dataset locally",
    )
    parser.add_argument(
        "--lineage-output",
        type=str,
        default=None,
        help="Path to save lineage JSON",
    )

    # HuggingFace options
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="tschouis/moleculariq_benchmark",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default is private)",
    )

    # Constraint variant options
    parser.add_argument(
        "--single-constraint-variants",
        type=str,
        default="exact",
        choices=["exact", "flexible", "both"],
        help="Which single-constraint variants to emit",
    )
    parser.add_argument(
        "--multi-constraint-variants",
        type=str,
        default="exact",
        choices=["exact", "flexible", "both"],
        help="Which multi-constraint variants to emit",
    )

    # Multi-constraint turbo parameters
    parser.add_argument(
        "--n-workers",
        type=int,
        default=60,
        help="Parallel workers for multi-constraint generation",
    )
    parser.add_argument(
        "--top-combinations",
        type=int,
        default=50,
        help="Top combinations to keep per molecule",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Minimum number of molecules that must satisfy constraint combination",
    )
    parser.add_argument(
        "--max-support-fraction",
        type=float,
        default=0.35,
        help="Maximum fraction of molecules that can satisfy a constraint (filters too-easy constraints)",
    )
    parser.add_argument(
        "--n-tasks-per-k",
        type=str,
        default="2:300,3:300,5:300",
        help="Number of tasks per constraint count, format: 'k1:n1,k2:n2,...' (e.g., '2:300,3:300,5:300')",
    )
    parser.add_argument(
        "--prop-cap",
        type=str,
        action="append",
        default=None,
        help="Limit property representation in multi-constraint tasks. Format: 'property=fraction' "
             "(e.g., '--prop-cap molecular_formula=0.25'). Can be specified multiple times.",
    )

    # Ring enumeration option (enabled by default)
    ring_enum_group = parser.add_mutually_exclusive_group()
    ring_enum_group.add_argument(
        "--ring-enumeration",
        action="store_true",
        dest="ring_enumeration",
        help="Generate ring_enum split with ring-enumerated SMILES (default)",
    )
    ring_enum_group.add_argument(
        "--no-ring-enumeration",
        action="store_false",
        dest="ring_enumeration",
        help="Disable ring enumeration split generation",
    )
    parser.set_defaults(ring_enumeration=True)

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample molecules for testing",
    )
    parser.add_argument(
        "--sampling-prime-workers",
        type=int,
        default=0,
        help="Thread workers for precomputing inverse-frequency caches",
    )

    args = parser.parse_args()

    # Create config from args
    config = BenchmarkConfig.from_cli_args(args)

    # Set output path if not provided
    if config.output_path is None:
        config.output_path = str(Path(__file__).parent.parent / "benchmark_tasks.json")

    # Generate dataset
    all_tasks = generate_benchmark_dataset(config)

    # Print statistics
    print_statistics(all_tasks)

    # Save JSON
    print(f"\nSaving {len(all_tasks)} tasks to {config.output_path}")
    with open(config.output_path, "w") as f:
        json.dump(all_tasks, f, indent=2)

    # Save lineage
    if args.lineage_output:
        export_lineage(all_tasks, args.lineage_output)

    # Create HuggingFace dataset
    if args.push_to_hub or args.save_local:
        dataset = create_huggingface_dataset(
            all_tasks=all_tasks,
            dataset_name=args.dataset_name,
            push_to_hub=args.push_to_hub,
            private=not args.public,
            ring_enumeration=config.ring_enumeration,
            seed=config.seed,
        )

        if args.save_local:
            dataset.save_to_disk(args.save_local)
            print(f"\nDataset saved locally to: {args.save_local}")

    # Final summary
    print("\n" + "=" * 80)
    print("BENCHMARK DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Total tasks generated: {len(all_tasks)}")
    print(f"JSON file: {config.output_path}")
    if args.push_to_hub:
        print(f"HuggingFace dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()

#python -m src.B_create_benchmark.benchmark_generator.main --output-path /system/user/publicwork/bartmann/chemical_reasoning/tests/test_commit_data
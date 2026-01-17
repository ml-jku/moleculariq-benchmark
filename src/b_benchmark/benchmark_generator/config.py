"""
Configuration dataclass for benchmark dataset generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark dataset generation."""

    # Data source
    pickle_path: str = "data/benchmark/properties.pkl"

    # Output paths
    output_path: Optional[str] = None
    save_local: Optional[str] = None
    lineage_output: Optional[str] = None

    # HuggingFace options
    push_to_hub: bool = False
    dataset_name: str = "moleculariq"
    public: bool = False

    # Sampling parameters
    n_samples_per_bin_single: int = 10
    n_samples_per_bin_multi: int = 100
    n_samples_per_category_constraint: int = 30

    # Multi-property task configuration
    n_properties_list: List[int] = field(default_factory=lambda: [2, 3, 5])

    # Multi-constraint task configuration
    n_constraints_list: List[int] = field(default_factory=lambda: [2, 3, 5])
    n_tasks_per_k: Dict[int, int] = field(
        default_factory=lambda: {2: 300, 3: 300, 5: 300}
    )

    # Constraint generation parameters
    single_constraint_variants: str = "exact"  # 'exact', 'flexible', 'both'
    multi_constraint_variants: str = "exact"  # 'exact', 'flexible', 'both'
    target_rate_min: float = 0.0
    target_rate_max: float = 0.03

    # Ultra-hard mode
    ultra_hard: bool = False

    # Multi-constraint turbo parameters
    top_combinations_per_molecule: int = 50
    n_workers: int = 60
    min_support: int = 10
    max_support_fraction: float = 0.35
    prescreen_size: int = 5000
    cand_pool_size: int = 800
    min_intersection: int = 10
    chunksize: int = 8

    # Property caps for multi-constraint (limit overrepresentation)
    prop_caps: Dict[int, Dict[str, float]] = field(
        default_factory=lambda: {
            2: {"molecular_formula": 0.25},
            3: {"molecular_formula": 0.25},
            5: {"molecular_formula": 0.25},
        }
    )

    # Other options
    seed: int = 42
    subsample: Optional[int] = None
    sampling_prime_workers: int = 0

    # FG property conversion
    convert_fg_properties: bool = False
    fg_conversion_probability: float = 0.5

    # Ring enumeration - create additional splits with randomized ring closure numbers
    ring_enumeration: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.single_constraint_variants not in {"exact", "flexible", "both"}:
            raise ValueError(
                f"single_constraint_variants must be 'exact', 'flexible', or 'both', "
                f"got {self.single_constraint_variants}"
            )
        if self.multi_constraint_variants not in {"exact", "flexible", "both"}:
            raise ValueError(
                f"multi_constraint_variants must be 'exact', 'flexible', or 'both', "
                f"got {self.multi_constraint_variants}"
            )

        # Set default output path if not provided
        if self.output_path is None:
            self.output_path = "data/benchmark/benchmark_tasks.json"

        # Set default save_local path if not provided
        if self.save_local is None:
            self.save_local = "data/benchmark/hf_dataset"

        # Set default lineage output path if not provided
        if self.lineage_output is None:
            self.lineage_output = "data/benchmark/lineage.json"

        # Ultra-hard mode adjustments
        if self.ultra_hard:
            self.target_rate_max = 0.005  # Stricter threshold

    @classmethod
    def from_cli_args(cls, args) -> "BenchmarkConfig":
        """Create config from argparse namespace."""
        # Parse n_tasks_per_k from string format "2:300,3:300,5:300"
        n_tasks_per_k = {}
        n_tasks_str = getattr(args, "n_tasks_per_k", "2:300,3:300,5:300")
        for item in n_tasks_str.split(","):
            k, n = item.split(":")
            n_tasks_per_k[int(k)] = int(n)

        # Parse prop_caps from repeated --prop-cap arguments
        prop_caps: Dict[int, Dict[str, float]] = {}
        prop_cap_list = getattr(args, "prop_cap", None)
        if prop_cap_list:
            # Apply same caps to all k values
            parsed_caps: Dict[str, float] = {}
            for cap_str in prop_cap_list:
                prop, frac = cap_str.split("=")
                parsed_caps[prop.strip()] = float(frac)
            # Apply to all constraint counts
            for k in n_tasks_per_k.keys():
                prop_caps[k] = parsed_caps.copy()
        else:
            # Default caps
            prop_caps = {
                2: {"molecular_formula": 0.25},
                3: {"molecular_formula": 0.25},
                5: {"molecular_formula": 0.25},
            }

        return cls(
            pickle_path=args.pickle_path,
            output_path=args.output_path,
            save_local=args.save_local,
            lineage_output=getattr(args, "lineage_output", None),
            push_to_hub=args.push_to_hub,
            dataset_name=args.dataset_name,
            public=args.public,
            n_samples_per_bin_single=args.n_samples_per_bin,
            n_samples_per_bin_multi=args.n_samples_multi,
            n_samples_per_category_constraint=args.n_samples_per_category,
            single_constraint_variants=args.single_constraint_variants,
            multi_constraint_variants=args.multi_constraint_variants,
            ultra_hard=args.ultra_hard,
            seed=args.seed,
            subsample=args.subsample,
            sampling_prime_workers=args.sampling_prime_workers,
            n_workers=getattr(args, "n_workers", 60),
            top_combinations_per_molecule=getattr(args, "top_combinations", 50),
            min_support=getattr(args, "min_support", 10),
            max_support_fraction=getattr(args, "max_support_fraction", 0.35),
            n_tasks_per_k=n_tasks_per_k,
            prop_caps=prop_caps,
            ring_enumeration=getattr(args, "ring_enumeration", False),
        )

"""
Generate the benchmark dataset from precomputed molecular properties.

This script is the second step in the benchmark creation pipeline. It uses the
properties.pkl file (created by 1_compute_properties.py) to generate the full
benchmark dataset with various task types.

Prerequisites:
    - Run 1_compute_properties.py first to create data/benchmark/properties.pkl

Usage:
    python src/b_benchmark/2_create_benchmark.py
    python src/b_benchmark/2_create_benchmark.py --help

Output:
    - data/benchmark/benchmark_dataset.json
    - data/benchmark/hf_dataset/ (optional, if --save-local specified)
"""

import sys
from pathlib import Path

# Add repo root to path so imports work from any location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.b_benchmark.benchmark_generator.main import main

if __name__ == "__main__":
    main()

"""
Benchmark Dataset Generator Package

Clean, modular benchmark generation for molecular reasoning tasks.
Uses the superior multi-constraint logic from the turbo generator.

Usage:
    python -m src.B_create_benchmark.benchmark_generator.main --help
"""

from .config import BenchmarkConfig

__all__ = ["BenchmarkConfig"]

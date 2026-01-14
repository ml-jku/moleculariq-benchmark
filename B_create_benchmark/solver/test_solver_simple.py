#!/usr/bin/env python3
"""
Simplified solver test that can run quickly.
Usage: python test_solver_simple.py [--n_molecules N]
"""

import sys
import argparse
import json
import time
import random
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_solver_consistency import SolverConsistencyTester


def get_test_molecules(n=100):
    """Generate diverse test molecules."""
    molecules = []

    # Simple molecules
    molecules.extend([
        "CCO", "CC(C)O", "CC(=O)O", "c1ccccc1", "C1CCCCC1",
        "CCN", "CCC", "CCCC", "CC(C)(C)C", "C1CCC1"
    ])

    # Aromatic molecules
    molecules.extend([
        "c1ccccc1", "c1ccncc1", "c1ccccc1C(=O)O", "c1ccc(O)cc1",
        "c1ccc2ccccc2c1", "c1ccc(cc1)c2ccccc2"
    ])

    # Stereochemistry
    molecules.extend([
        "C[C@H](Cl)Br", "C[C@@H](Cl)Br", "CC(Cl)Br",
        "C/C=C/C", "C/C=C\\C", "CC=CC"
    ])

    # Ring systems
    molecules.extend([
        "C1CC2CCC1C2",  # norbornane
        "C1C2CC3CC1CC(C2)C3",  # adamantane
        "C12CCC1CC2",  # bicyclo
        "C12(CCC1)CCC2",  # spiro
    ])

    # Functional groups
    molecules.extend([
        "CC(=O)O", "CC(=O)OC", "CCO", "CCN", "CC(=O)N",
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    ])

    # Generate random alkanes
    for i in range(n - len(molecules)):
        length = random.randint(3, 10)
        if random.random() < 0.3:
            # Branched
            molecules.append("CC" + "C(C)" * (length // 2))
        else:
            # Linear
            molecules.append("C" * length)

    return molecules[:n]


def main():
    parser = argparse.ArgumentParser(description="Simple solver test")
    parser.add_argument('--n_molecules', type=int, default=100,
                       help='Number of molecules to test')
    parser.add_argument('--output', type=str, default='solver_test_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    print("="*60)
    print(f"SOLVER VALIDATION TEST")
    print(f"Testing {args.n_molecules} molecules")
    print("="*60)

    # Get test molecules
    test_molecules = get_test_molecules(args.n_molecules)

    # Initialize tester
    tester = SolverConsistencyTester()

    # Run tests
    print("\nRunning tests...")
    start_time = time.time()

    results = tester.run_comprehensive_test(test_molecules, verbose=False)

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total molecules tested: {results['summary']['total_molecules_tested']}")
    print(f"Overall pass rate: {results['summary']['overall_pass_rate']:.2f}%")
    print(f"Execution time: {elapsed:.1f} seconds")
    print("\nTest Category Pass Rates:")
    for test_name, stats in results['summary']['test_statistics'].items():
        print(f"  {test_name}: {stats['pass_rate']:.2f}%")

    # Save results
    output = {
        'n_molecules': args.n_molecules,
        'overall_pass_rate': results['summary']['overall_pass_rate'],
        'test_statistics': results['summary']['test_statistics'],
        'edge_case_stats': results['edge_case_stats'],
        'execution_time': elapsed
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
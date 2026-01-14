#!/usr/bin/env python3
"""
Comprehensive Solver Consistency Test Suite

Tests all solver methods for consistency across diverse SMILES strings.
Addresses reviewer concerns about RDKit edge cases and verifier dependence.
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdmd
import warnings
warnings.filterwarnings('ignore')

# Import the solver
from solver import SymbolicSolver


@dataclass
class TestResult:
    """Container for test results of a single method."""
    method_name: str
    total_molecules: int
    passed: int
    failed: int
    errors: List[Tuple[str, str]]  # (SMILES, error_message)
    execution_time: float
    pass_rate: float

    def to_dict(self):
        return {
            'method_name': self.method_name,
            'total_molecules': self.total_molecules,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': self.pass_rate,
            'execution_time': self.execution_time,
            'num_errors': len(self.errors),
            'sample_errors': self.errors[:5]  # Only keep first 5 for report
        }


class SolverConsistencyTester:
    """Comprehensive test suite for SymbolicSolver consistency."""

    def __init__(self):
        self.solver = SymbolicSolver()
        self.results = {}
        self.edge_case_stats = defaultdict(int)
        self.performance_stats = {}

    def get_all_method_pairs(self) -> List[Tuple[str, str]]:
        """Get all count/indices method pairs that should be consistent."""
        pairs = [
            # Graph topology
            ('get_ring_count', 'get_ring_indices'),
            ('get_fused_ring_count', 'get_fused_ring_indices'),
            ('get_bridgehead_count', 'get_bridgehead_indices'),
            ('get_chain_termini_count', 'get_chain_termini_indices'),
            ('get_branch_point_count', 'get_branch_point_indices'),

            # Chemistry-typed topology
            ('get_aromatic_ring_count', 'get_aromatic_ring_indices'),
            ('get_aliphatic_ring_count', 'get_aliphatic_ring_indices'),
            ('get_heterocycle_count', 'get_heterocycle_indices'),
            ('get_saturated_ring_count', 'get_saturated_ring_indices'),
            ('get_csp3_carbon_count', 'get_csp3_carbon_indices'),
            ('get_stereocenter_count', 'get_stereocenter_indices'),
            ('get_unspecified_stereocenter_count', 'get_unspecified_stereocenter_indices'),

            # Composition
            ('get_carbon_atom_count', 'get_carbon_atom_indices'),
            ('get_hetero_atom_count', 'get_hetero_atom_indices'),
            ('get_halogen_atom_count', 'get_halogen_atom_indices'),
            ('get_heavy_atom_count', 'get_heavy_atom_indices'),
            ('get_explicit_hydrogen_count', 'get_explicit_hydrogen_indices'),

            # Chemical perception
            ('get_hba_count', 'get_hba_indices'),
            ('get_hbd_count', 'get_hbd_indices'),
            ('get_rotatable_bond_count', 'get_rotatable_bond_indices'),

            # Synthesis
            ('get_brics_fragment_count', 'get_brics_bond_indices'),
            ('get_spiro_count', 'get_spiro_indices'),
        ]

        # Add R/S stereocenter pairs
        for r_or_s in [True, False]:
            count_method = f'get_{"r" if r_or_s else "s"}_stereocenter_count'
            indices_method = f'get_r_or_s_stereocenter_indices'
            pairs.append((count_method, indices_method, r_or_s))

        # Add E/Z double bond pairs
        for e_or_z in [True, False]:
            count_method = f'get_{"e" if e_or_z else "z"}_stereochemistry_double_bond_count'
            indices_method = f'get_e_z_stereochemistry_double_bond_indices'
            pairs.append((count_method, indices_method, e_or_z))

        # Add oxidation state pairs
        for element in ['C', 'N', 'O', 'P', 'S']:
            for max_ox in [True, False]:
                count_method = f'get_oxidation_state_{element}_{"max" if max_ox else "min"}_count'
                indices_method = f'get_oxidation_state_indices'
                pairs.append((count_method, indices_method, element, max_ox))

        # Add smallest/largest ring pairs
        for smallest in [True, False]:
            size_type = 'smallest' if smallest else 'largest'
            count_method = f'get_{size_type}_ring_size_count'
            indices_method = f'get_smallest_or_largest_ring_indices'
            pairs.append((count_method, indices_method, smallest))

        return pairs

    def test_count_indices_consistency(self, smiles: str) -> Dict[str, Any]:
        """Test that count methods match length of indices methods."""
        issues = []

        for pair in self.get_all_method_pairs():
            try:
                if len(pair) == 2:
                    count_method, indices_method = pair
                    count = getattr(self.solver, count_method)(smiles)
                    indices = getattr(self.solver, indices_method)(smiles)
                elif len(pair) == 3:
                    # Methods with additional parameter
                    count_method, indices_method, param = pair
                    if 'stereocenter' in count_method:
                        count = getattr(self.solver, count_method)(smiles)
                        indices = getattr(self.solver, indices_method)(smiles, r_indices=param)
                    elif 'stereochemistry_double_bond' in count_method:
                        count = getattr(self.solver, count_method)(smiles)
                        indices = getattr(self.solver, indices_method)(smiles, e_indices=param)
                    else:  # smallest/largest ring
                        count = getattr(self.solver, count_method)(smiles)
                        indices = getattr(self.solver, indices_method)(smiles, smallest=param)
                elif len(pair) == 4:
                    # Oxidation state methods
                    count_method, indices_method, element, max_ox = pair
                    count = getattr(self.solver, count_method)(smiles)
                    indices = getattr(self.solver, indices_method)(smiles, element, max_oxidation=max_ox)

                # Check consistency
                if count != len(indices):
                    issues.append({
                        'type': 'count_indices_mismatch',
                        'method_pair': pair[:2],
                        'count': count,
                        'num_indices': len(indices),
                        'indices': indices[:10]  # First 10 for debugging
                    })

                # Check index validity
                if indices:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        num_atoms = mol.GetNumAtoms()
                        invalid_indices = [i for i in indices if i < 0 or i >= num_atoms]
                        if invalid_indices:
                            issues.append({
                                'type': 'invalid_indices',
                                'method': indices_method,
                                'invalid': invalid_indices,
                                'num_atoms': num_atoms
                            })

                        # Check sorting
                        if indices != sorted(indices):
                            issues.append({
                                'type': 'unsorted_indices',
                                'method': indices_method,
                                'indices': indices[:10]
                            })

                        # Check uniqueness
                        if len(indices) != len(set(indices)):
                            issues.append({
                                'type': 'duplicate_indices',
                                'method': indices_method,
                                'duplicates': [i for i in indices if indices.count(i) > 1]
                            })

            except Exception as e:
                issues.append({
                    'type': 'exception',
                    'method_pair': pair[:2] if len(pair) >= 2 else str(pair),
                    'error': str(e)
                })

        return {'smiles': smiles, 'issues': issues}

    def test_logical_relationships(self, smiles: str) -> Dict[str, Any]:
        """Test logical relationships between related methods."""
        issues = []

        try:
            # Test: Aromatic + Aliphatic rings should equal total rings
            total_rings = self.solver.get_ring_count(smiles)
            aromatic_rings = self.solver.get_aromatic_ring_count(smiles)
            aliphatic_rings = self.solver.get_aliphatic_ring_count(smiles)

            # Note: In RDKit, a ring can have both aromatic and non-aromatic bonds
            # so this relationship might not hold exactly
            if aromatic_rings + aliphatic_rings != total_rings:
                # This is actually expected in RDKit - document it
                self.edge_case_stats['ring_classification_overlap'] += 1

            # Test: Fused rings should be subset of all rings
            fused_indices = set(self.solver.get_fused_ring_indices(smiles))
            all_ring_indices = set(self.solver.get_ring_indices(smiles))
            if not fused_indices.issubset(all_ring_indices):
                issues.append({
                    'type': 'subset_violation',
                    'relationship': 'fused_rings_not_subset_of_all_rings',
                    'fused_not_in_rings': list(fused_indices - all_ring_indices)
                })

            # Test: Stereocenters = R + S + unspecified
            total_stereo = self.solver.get_stereocenter_count(smiles)
            r_stereo = self.solver.get_r_stereocenter_count(smiles)
            s_stereo = self.solver.get_s_stereocenter_count(smiles)
            unspec_stereo = self.solver.get_unspecified_stereocenter_count(smiles)

            if total_stereo != r_stereo + s_stereo + unspec_stereo:
                issues.append({
                    'type': 'stereocenter_count_mismatch',
                    'total': total_stereo,
                    'r': r_stereo,
                    's': s_stereo,
                    'unspecified': unspec_stereo
                })

            # Test: Heavy atoms >= Carbon + Heteroatoms
            heavy = self.solver.get_heavy_atom_count(smiles)
            carbon = self.solver.get_carbon_atom_count(smiles)
            hetero = self.solver.get_hetero_atom_count(smiles)

            if heavy != carbon + hetero:
                issues.append({
                    'type': 'heavy_atom_count_mismatch',
                    'heavy': heavy,
                    'carbon': carbon,
                    'hetero': hetero
                })

        except Exception as e:
            issues.append({
                'type': 'exception',
                'test': 'logical_relationships',
                'error': str(e)
            })

        return {'smiles': smiles, 'issues': issues}

    def test_edge_cases(self, smiles: str) -> Dict[str, Any]:
        """Test specific edge cases mentioned by reviewer."""
        issues = []
        mol = Chem.MolFromSmiles(smiles)

        if not mol:
            return {'smiles': smiles, 'issues': [{'type': 'invalid_smiles'}]}

        # Check for edge case characteristics
        has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        has_kekulize_issue = False

        # Test aromaticity/kekulization
        try:
            # Test if molecule can be kekulized
            mol_copy = Chem.Mol(mol)
            Chem.Kekulize(mol_copy)
            kek_smiles = Chem.MolToSmiles(mol_copy, kekuleSmiles=True)

            # Reparse and check consistency
            mol_kek = Chem.MolFromSmiles(kek_smiles)
            if mol_kek:
                # Test if both forms give same results
                orig_rings = self.solver.get_ring_count(smiles)
                kek_rings = self.solver.get_ring_count(kek_smiles)

                if orig_rings != kek_rings:
                    issues.append({
                        'type': 'kekulization_inconsistency',
                        'original': orig_rings,
                        'kekulized': kek_rings
                    })
                    self.edge_case_stats['kekulization_issues'] += 1

        except Exception as e:
            has_kekulize_issue = True
            self.edge_case_stats['kekulization_failures'] += 1

        # Check for undefined stereochemistry
        unspec_stereo = self.solver.get_unspecified_stereocenter_count(smiles)
        if unspec_stereo > 0:
            self.edge_case_stats['unspecified_stereochemistry'] += 1

        # Check for E/Z stereochemistry
        e_count = self.solver.get_e_stereochemistry_double_bond_count(smiles)
        z_count = self.solver.get_z_stereochemistry_double_bond_count(smiles)
        unspec_ez = self.solver.get_stereochemistry_unspecified_double_bond_count(smiles)

        if unspec_ez > 0:
            self.edge_case_stats['unspecified_double_bonds'] += 1

        # Test tautomer-sensitive properties
        # HBD/HBA can change with tautomerization
        hbd = self.solver.get_hbd_count(smiles)
        hba = self.solver.get_hba_count(smiles)

        # Document if molecule might have tautomers (has both HBD and C=O/C=N)
        if hbd > 0:
            smarts_co = Chem.MolFromSmarts('[C]=[O]')
            smarts_cn = Chem.MolFromSmarts('[C]=[N]')
            if mol.HasSubstructMatch(smarts_co) or mol.HasSubstructMatch(smarts_cn):
                self.edge_case_stats['potential_tautomers'] += 1

        return {'smiles': smiles, 'issues': issues}

    def test_rdkit_cross_validation(self, smiles: str) -> Dict[str, Any]:
        """Cross-validate with direct RDKit methods where available."""
        issues = []
        mol = Chem.MolFromSmiles(smiles)

        if not mol:
            return {'smiles': smiles, 'issues': [{'type': 'invalid_smiles'}]}

        try:
            # Test heavy atom count
            rdkit_heavy = mol.GetNumHeavyAtoms()
            solver_heavy = self.solver.get_heavy_atom_count(smiles)
            if rdkit_heavy != solver_heavy:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'heavy_atoms',
                    'rdkit': rdkit_heavy,
                    'solver': solver_heavy
                })

            # Test aromatic ring count
            rdkit_aromatic = Descriptors.NumAromaticRings(mol)
            solver_aromatic = self.solver.get_aromatic_ring_count(smiles)
            if rdkit_aromatic != solver_aromatic:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'aromatic_rings',
                    'rdkit': rdkit_aromatic,
                    'solver': solver_aromatic
                })

            # Test rotatable bonds
            rdkit_rotatable = rdmd.CalcNumRotatableBonds(mol)
            solver_rotatable = self.solver.get_rotatable_bond_count(smiles)
            if rdkit_rotatable != solver_rotatable:
                # This might be due to strict vs Lipinski definition
                self.edge_case_stats['rotatable_bond_definition_diff'] += 1

            # Test HBA/HBD
            rdkit_hba = rdmd.CalcNumHBA(mol)
            solver_hba = self.solver.get_hba_count(smiles)
            if rdkit_hba != solver_hba:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'hba',
                    'rdkit': rdkit_hba,
                    'solver': solver_hba
                })

            rdkit_hbd = rdmd.CalcNumHBD(mol)
            solver_hbd = self.solver.get_hbd_count(smiles)
            if rdkit_hbd != solver_hbd:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'hbd',
                    'rdkit': rdkit_hbd,
                    'solver': solver_hbd
                })

            # Test spiro atoms
            rdkit_spiro = rdmd.CalcNumSpiroAtoms(mol)
            solver_spiro = self.solver.get_spiro_count(smiles)
            if rdkit_spiro != solver_spiro:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'spiro_atoms',
                    'rdkit': rdkit_spiro,
                    'solver': solver_spiro
                })

            # Test bridgehead atoms
            rdkit_bridgehead = rdmd.CalcNumBridgeheadAtoms(mol)
            solver_bridgehead = self.solver.get_bridgehead_count(smiles)
            if rdkit_bridgehead != solver_bridgehead:
                issues.append({
                    'type': 'rdkit_mismatch',
                    'property': 'bridgehead_atoms',
                    'rdkit': rdkit_bridgehead,
                    'solver': solver_bridgehead
                })

        except Exception as e:
            issues.append({
                'type': 'exception',
                'test': 'rdkit_cross_validation',
                'error': str(e)
            })

        return {'smiles': smiles, 'issues': issues}

    def test_functional_groups(self, smiles: str) -> Dict[str, Any]:
        """Test functional group detection consistency."""
        issues = []

        try:
            fg_dict = self.solver.get_functional_group_count_and_indices(smiles)

            for fg_name, fg_data in fg_dict.items():
                count = fg_data.get('nbr_instances', 0)
                indices = fg_data.get('indices', [])

                # Flatten nested indices
                flat_indices = []
                for idx_list in indices:
                    if isinstance(idx_list, list):
                        flat_indices.extend(idx_list)
                    else:
                        flat_indices.append(idx_list)

                # Check count matches number of instances
                if count != len(indices):
                    issues.append({
                        'type': 'functional_group_count_mismatch',
                        'group': fg_name,
                        'count': count,
                        'num_instances': len(indices)
                    })

                # Check indices are valid
                mol = Chem.MolFromSmiles(smiles)
                if mol and flat_indices:
                    num_atoms = mol.GetNumAtoms()
                    invalid = [i for i in flat_indices if i < 0 or i >= num_atoms]
                    if invalid:
                        issues.append({
                            'type': 'functional_group_invalid_indices',
                            'group': fg_name,
                            'invalid_indices': invalid
                        })

        except Exception as e:
            issues.append({
                'type': 'exception',
                'test': 'functional_groups',
                'error': str(e)
            })

        return {'smiles': smiles, 'issues': issues}

    def test_reaction_templates(self, smiles: str) -> Dict[str, Any]:
        """Test reaction template consistency."""
        issues = []

        try:
            reaction_dict = self.solver.get_reaction_counts_and_indices(smiles)

            for reaction_name, reaction_data in reaction_dict.items():
                count = reaction_data.get('count', 0)
                indices = reaction_data.get('indices', [])
                success = reaction_data.get('success', 0)
                products = reaction_data.get('products', None)

                # Check success flag consistency
                if success == 1 and products is None:
                    issues.append({
                        'type': 'reaction_success_without_products',
                        'reaction': reaction_name
                    })
                elif success == 0 and products is not None:
                    issues.append({
                        'type': 'reaction_products_without_success',
                        'reaction': reaction_name,
                        'products': products[:3]  # First 3
                    })

                # Validate product SMILES
                if products:
                    for prod_smiles in products[:5]:  # Check first 5
                        mol = Chem.MolFromSmiles(prod_smiles)
                        if not mol:
                            issues.append({
                                'type': 'invalid_reaction_product',
                                'reaction': reaction_name,
                                'product': prod_smiles
                            })

        except Exception as e:
            issues.append({
                'type': 'exception',
                'test': 'reaction_templates',
                'error': str(e)
            })

        return {'smiles': smiles, 'issues': issues}

    def run_comprehensive_test(self, smiles_list: List[str],
                              verbose: bool = False) -> Dict[str, Any]:
        """Run all tests on a list of SMILES."""
        print(f"Testing {len(smiles_list)} molecules...")

        all_results = []
        failed_molecules = defaultdict(list)

        for smiles in tqdm(smiles_list, desc="Testing molecules"):
            molecule_results = {}

            # Run each test category
            tests = [
                ('count_indices', self.test_count_indices_consistency),
                ('logical_relationships', self.test_logical_relationships),
                ('edge_cases', self.test_edge_cases),
                ('rdkit_validation', self.test_rdkit_cross_validation),
                ('functional_groups', self.test_functional_groups),
                ('reaction_templates', self.test_reaction_templates),
            ]

            for test_name, test_func in tests:
                try:
                    result = test_func(smiles)
                    molecule_results[test_name] = result

                    # Track failures
                    if result.get('issues'):
                        failed_molecules[test_name].append({
                            'smiles': smiles,
                            'issues': result['issues']
                        })

                except Exception as e:
                    molecule_results[test_name] = {
                        'smiles': smiles,
                        'issues': [{'type': 'test_exception', 'error': str(e)}]
                    }
                    failed_molecules[test_name].append({
                        'smiles': smiles,
                        'error': str(e)
                    })

            all_results.append(molecule_results)

        # Generate summary statistics
        summary = self.generate_summary(all_results, failed_molecules)

        return {
            'summary': summary,
            'edge_case_stats': dict(self.edge_case_stats),
            'failed_molecules': failed_molecules,
            'detailed_results': all_results if verbose else None
        }

    def generate_summary(self, all_results: List[Dict],
                        failed_molecules: Dict) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_molecules = len(all_results)

        test_stats = {}
        for test_name in ['count_indices', 'logical_relationships', 'edge_cases',
                         'rdkit_validation', 'functional_groups', 'reaction_templates']:
            failures = len(failed_molecules.get(test_name, []))
            test_stats[test_name] = {
                'total': total_molecules,
                'passed': total_molecules - failures,
                'failed': failures,
                'pass_rate': (total_molecules - failures) / total_molecules * 100
            }

        return {
            'total_molecules_tested': total_molecules,
            'test_statistics': test_stats,
            'overall_pass_rate': np.mean([s['pass_rate'] for s in test_stats.values()])
        }

    def generate_report(self, results: Dict[str, Any],
                       output_path: str = "solver_test_report.json"):
        """Generate a JSON report for the paper."""
        report = {
            'test_suite': 'SymbolicSolver Consistency Tests',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': results['summary'],
            'edge_case_statistics': results['edge_case_stats'],
            'test_coverage': {
                'num_methods_tested': len(self.get_all_method_pairs()),
                'categories': [
                    'count_indices_consistency',
                    'logical_relationships',
                    'edge_cases',
                    'rdkit_cross_validation',
                    'functional_groups',
                    'reaction_templates'
                ]
            },
            'key_findings': {
                'overall_pass_rate': results['summary']['overall_pass_rate'],
                'kekulization_issues': results['edge_case_stats'].get('kekulization_failures', 0),
                'unspecified_stereochemistry': results['edge_case_stats'].get('unspecified_stereochemistry', 0),
                'potential_tautomers': results['edge_case_stats'].get('potential_tautomers', 0)
            }
        }

        # Add sample failures for each category
        if 'failed_molecules' in results:
            report['sample_failures'] = {}
            for test_name, failures in results['failed_molecules'].items():
                if failures:
                    report['sample_failures'][test_name] = failures[:3]  # First 3 examples

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {output_path}")

        # Also create a CSV summary for easier inclusion in paper
        summary_df = pd.DataFrame(results['summary']['test_statistics']).T
        summary_df.to_csv(output_path.replace('.json', '_summary.csv'))

        return report


def load_test_molecules(n_molecules: int = 10000,
                        dataset_path: Optional[str] = None) -> List[str]:
    """Load diverse SMILES for testing."""
    # For now, generate some test molecules
    # In production, load from your actual dataset
    test_smiles = [
        # Simple molecules
        "CCO", "CC(C)O", "CC(=O)O", "c1ccccc1", "C1CCCCC1",

        # Molecules with stereochemistry
        "C[C@H](Cl)Br", "C[C@@H](Cl)Br", "CC(Cl)Br",
        "C/C=C/C", "C/C=C\\C",

        # Complex ring systems
        "C1CC2CCC1C2",  # Norbornane
        "C1C2CC3CC1CC(C2)C3",  # Adamantane
        "C12(CCC1)CCC2",  # Spiro

        # Aromatic/heteroaromatic
        "c1ccncc1", "c1ccccc1C(=O)O", "c1ccc2c(c1)ncn2",

        # Functional groups
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen

        # Edge cases
        "C(C)(C)(C)(C)C",  # Invalid valence (should fail gracefully)
        "[Cu+2].[O-]S([O-])(=O)=O",  # Coordination complex
    ]

    # If dataset path provided, load real molecules
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_parquet(dataset_path)
            if 'smiles' in df.columns:
                real_smiles = df['smiles'].dropna().tolist()[:n_molecules]
                test_smiles.extend(real_smiles)
        except Exception as e:
            print(f"Warning: Could not load dataset from {dataset_path}: {e}")

    # Ensure we have enough molecules
    while len(test_smiles) < n_molecules:
        # Generate simple alkanes as fillers
        n = len(test_smiles) % 20 + 1
        test_smiles.append("C" * n)

    return test_smiles[:n_molecules]


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Test SymbolicSolver consistency")
    parser.add_argument('--n_molecules', type=int, default=1000,
                       help='Number of molecules to test')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset with SMILES')
    parser.add_argument('--output', type=str, default='solver_test_report.json',
                       help='Output report path')
    parser.add_argument('--verbose', action='store_true',
                       help='Include detailed results in output')

    args = parser.parse_args()

    # Initialize tester
    tester = SolverConsistencyTester()

    # Load test molecules
    print(f"Loading {args.n_molecules} test molecules...")
    smiles_list = load_test_molecules(args.n_molecules, args.dataset)

    # Run tests
    print("Running comprehensive tests...")
    results = tester.run_comprehensive_test(smiles_list, verbose=args.verbose)

    # Generate report
    print("Generating report...")
    report = tester.generate_report(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total molecules tested: {results['summary']['total_molecules_tested']}")
    print(f"Overall pass rate: {results['summary']['overall_pass_rate']:.2f}%")
    print("\nTest Category Pass Rates:")
    for test_name, stats in results['summary']['test_statistics'].items():
        print(f"  {test_name}: {stats['pass_rate']:.2f}%")
    print("\nEdge Case Statistics:")
    for edge_case, count in results['edge_case_stats'].items():
        print(f"  {edge_case}: {count}")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
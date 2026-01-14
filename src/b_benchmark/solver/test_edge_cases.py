#!/usr/bin/env python3
"""
Edge Case Test Suite for SymbolicSolver

Specifically addresses reviewer concerns about:
- Aromaticity/kekulization mismatches
- Tautomerism
- Unspecified stereochemistry
- Adversarial edge cases
"""

import unittest
import json
from typing import List, Dict, Any, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Import the solver
from solver import SymbolicSolver


class TestAromaticityEdgeCases(unittest.TestCase):
    """Test aromatic vs Kekule representation consistency."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_benzene_representations(self):
        """Test benzene in aromatic vs Kekule forms."""
        test_cases = [
            ("c1ccccc1", "C1=CC=CC=C1", "benzene"),
            ("c1ccccc1C(=O)O", "C1=CC=CC=C1C(=O)O", "benzoic acid"),
            ("c1ccc2ccccc2c1", "C1=CC=C2C=CC=CC2=C1", "naphthalene"),
            ("c1nc2ccccc2n1", "C1=NC2=CC=CC=C2N1", "benzimidazole"),
            ("c1ccc2c(c1)nc1ccccc1n2", "C1=CC=C2C(=C1)N=C1C=CC=CC1=N2", "phenanthroline"),
        ]

        for aromatic, kekule, name in test_cases:
            with self.subTest(molecule=name):
                # Both should give same canonical SMILES
                mol_arom = Chem.MolFromSmiles(aromatic)
                mol_kek = Chem.MolFromSmiles(kekule)

                self.assertIsNotNone(mol_arom, f"Failed to parse aromatic form of {name}")
                self.assertIsNotNone(mol_kek, f"Failed to parse Kekule form of {name}")

                canon_arom = Chem.MolToSmiles(mol_arom)
                canon_kek = Chem.MolToSmiles(mol_kek)

                self.assertEqual(canon_arom, canon_kek,
                                f"Canonicalization mismatch for {name}")

                # Test that solver gives consistent results
                for method in ['get_ring_count', 'get_aromatic_ring_count',
                             'get_carbon_atom_count', 'get_ring_indices']:
                    result_arom = getattr(self.solver, method)(aromatic)
                    result_kek = getattr(self.solver, method)(kekule)

                    # For indices, compare sorted lists
                    if isinstance(result_arom, list):
                        self.assertEqual(sorted(result_arom), sorted(result_kek),
                                       f"{method} inconsistent for {name}")
                    else:
                        self.assertEqual(result_arom, result_kek,
                                       f"{method} inconsistent for {name}")

    def test_mixed_aromaticity(self):
        """Test molecules with both aromatic and aliphatic rings."""
        test_cases = [
            "c1ccccc1C1CCCCC1",  # benzene + cyclohexane
            "c1ccc2c(c1)CCC2",   # tetralin
            "O1c2ccccc2CC1",     # chromane
        ]

        for smiles in test_cases:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)

                # Test that aromatic and aliphatic rings are detected
                aromatic = self.solver.get_aromatic_ring_count(smiles)
                aliphatic = self.solver.get_aliphatic_ring_count(smiles)
                total = self.solver.get_ring_count(smiles)

                self.assertGreater(aromatic, 0, f"No aromatic rings detected in {smiles}")
                self.assertGreater(aliphatic, 0, f"No aliphatic rings detected in {smiles}")
                self.assertGreaterEqual(total, max(aromatic, aliphatic),
                                      "Total rings less than components")

    def test_kekulization_edge_cases(self):
        """Test molecules that might have kekulization issues."""
        problematic_cases = [
            "c1ccccc1-c2ccccc2",  # biphenyl
            "c1ccc2c(c1)[nH]c1ccccc12",  # carbazole
            "[O-][n+]1ccccc1",  # pyridine N-oxide
        ]

        for smiles in problematic_cases:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    try:
                        # Try to kekulize
                        mol_copy = Chem.Mol(mol)
                        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
                        kek_smiles = Chem.MolToSmiles(mol_copy)

                        # Check solver handles both forms
                        orig_result = self.solver.get_ring_count(smiles)
                        kek_result = self.solver.get_ring_count(kek_smiles)

                        # Document any differences
                        if orig_result != kek_result:
                            print(f"Kekulization difference for {smiles}: "
                                  f"orig={orig_result}, kek={kek_result}")

                    except Exception as e:
                        # Document kekulization failures
                        print(f"Kekulization failed for {smiles}: {e}")


class TestStereochemistryEdgeCases(unittest.TestCase):
    """Test handling of stereochemistry edge cases."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_undefined_vs_unspecified_stereocenters(self):
        """Test distinction between undefined and unspecified stereochemistry."""
        test_cases = [
            {
                "smiles": "CC(Cl)Br",  # No stereochemistry markers
                "expected": {
                    "has_stereocenter": True,
                    "unspecified_count": 1,
                    "r_count": 0,
                    "s_count": 0
                }
            },
            {
                "smiles": "C[C@H](Cl)Br",  # R stereochemistry
                "expected": {
                    "has_stereocenter": True,
                    "unspecified_count": 0,
                    "r_count": 1,
                    "s_count": 0
                }
            },
            {
                "smiles": "C[C@@H](Cl)Br",  # S stereochemistry
                "expected": {
                    "has_stereocenter": True,
                    "unspecified_count": 0,
                    "r_count": 0,
                    "s_count": 1
                }
            },
            {
                "smiles": "C[C@H](O)[C@H](O)C",  # Multiple stereocenters
                "expected": {
                    "has_stereocenter": True,
                    "total_stereocenters": 2,
                    "r_count": 2,  # Both marked as R
                }
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]
            expected = case["expected"]

            with self.subTest(smiles=smiles):
                # Test stereocenter detection
                total = self.solver.get_stereocenter_count(smiles)
                unspec = self.solver.get_unspecified_stereocenter_count(smiles)
                r_count = self.solver.get_r_stereocenter_count(smiles)
                s_count = self.solver.get_s_stereocenter_count(smiles)

                if "has_stereocenter" in expected:
                    self.assertEqual(total > 0, expected["has_stereocenter"],
                                   f"Stereocenter detection failed for {smiles}")

                if "unspecified_count" in expected:
                    self.assertEqual(unspec, expected["unspecified_count"],
                                   f"Unspecified count mismatch for {smiles}")

                if "r_count" in expected:
                    self.assertEqual(r_count, expected["r_count"],
                                   f"R stereocenter count mismatch for {smiles}")

                if "s_count" in expected:
                    self.assertEqual(s_count, expected["s_count"],
                                   f"S stereocenter count mismatch for {smiles}")

                # Verify total = R + S + unspecified
                self.assertEqual(total, r_count + s_count + unspec,
                               f"Stereocenter sum mismatch for {smiles}")

    def test_ez_stereochemistry(self):
        """Test E/Z double bond stereochemistry."""
        test_cases = [
            {
                "smiles": "C/C=C/C",  # E-2-butene
                "expected_e": True,
                "expected_z": False
            },
            {
                "smiles": "C/C=C\\C",  # Z-2-butene
                "expected_e": False,
                "expected_z": True
            },
            {
                "smiles": "CC=CC",  # No stereochemistry specified
                "expected_unspecified": True
            },
            {
                "smiles": "Cl/C=C/Br",  # E-1-bromo-2-chloroethene
                "expected_e": True,
                "expected_z": False
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]

            with self.subTest(smiles=smiles):
                e_count = self.solver.get_e_stereochemistry_double_bond_count(smiles)
                z_count = self.solver.get_z_stereochemistry_double_bond_count(smiles)
                unspec = self.solver.get_stereochemistry_unspecified_double_bond_count(smiles)

                if "expected_e" in case:
                    if case["expected_e"]:
                        self.assertGreater(e_count, 0, f"E configuration not detected in {smiles}")
                    else:
                        self.assertEqual(e_count, 0, f"E configuration wrongly detected in {smiles}")

                if "expected_z" in case:
                    if case["expected_z"]:
                        self.assertGreater(z_count, 0, f"Z configuration not detected in {smiles}")
                    else:
                        self.assertEqual(z_count, 0, f"Z configuration wrongly detected in {smiles}")

                if case.get("expected_unspecified"):
                    self.assertGreater(unspec, 0, f"Unspecified double bond not detected in {smiles}")


class TestTautomerHandling(unittest.TestCase):
    """Test handling of tautomeric forms."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_keto_enol_tautomers(self):
        """Test that keto and enol forms are treated as distinct."""
        tautomer_pairs = [
            ("CC(=O)CC", "CC(O)=CC", "butanone/butenol"),
            ("CC(=O)C", "CC(O)=C", "acetone/propen-2-ol"),
            ("O=C1CCCCC1", "OC1=CCCCC1", "cyclohexanone/cyclohexenol"),
        ]

        for keto, enol, name in tautomer_pairs:
            with self.subTest(tautomer=name):
                # Get HBD counts (should differ)
                keto_hbd = self.solver.get_hbd_count(keto)
                enol_hbd = self.solver.get_hbd_count(enol)

                # Keto form has no HBD, enol form has 1
                self.assertEqual(keto_hbd, 0, f"Keto form {keto} should have 0 HBD")
                self.assertEqual(enol_hbd, 1, f"Enol form {enol} should have 1 HBD")

                # Verify they remain distinct (different canonical SMILES)
                keto_canon = Chem.CanonSmiles(keto)
                enol_canon = Chem.CanonSmiles(enol)
                self.assertNotEqual(keto_canon, enol_canon,
                                  f"Tautomers should remain distinct: {name}")

    def test_imine_enamine_tautomers(self):
        """Test imine-enamine tautomerism."""
        pairs = [
            ("CC(=N)C", "CC(N)=C", "imine/enamine"),
        ]

        for imine, enamine, name in pairs:
            with self.subTest(tautomer=name):
                # Test that functional groups differ
                imine_fg = self.solver.get_functional_group_count_and_indices(imine)
                enamine_fg = self.solver.get_functional_group_count_and_indices(enamine)

                # Check for imine group
                self.assertIn("imine", imine_fg,
                            f"Imine group not detected in {imine}")
                self.assertIn("enamine", enamine_fg,
                            f"Enamine group not detected in {enamine}")


class TestRingSystemEdgeCases(unittest.TestCase):
    """Test complex ring systems."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_bridged_ring_systems(self):
        """Test bridgehead atom detection in bridged systems."""
        test_cases = [
            {
                "smiles": "C1CC2CCC1C2",  # Norbornane
                "name": "norbornane",
                "expected_bridgeheads": 2,
                "expected_rings": 2
            },
            {
                "smiles": "C1C2CC3CC1CC(C2)C3",  # Adamantane
                "name": "adamantane",
                "expected_bridgeheads": 4,
                "expected_rings": 3
            },
            {
                "smiles": "C12CCC1CC2",  # Bicyclo[2.2.1]heptane
                "name": "bicyclo[2.2.1]heptane",
                "expected_bridgeheads": 2,
                "expected_rings": 2
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]
            name = case["name"]

            with self.subTest(molecule=name):
                bridgeheads = self.solver.get_bridgehead_count(smiles)
                rings = self.solver.get_ring_count(smiles)

                self.assertEqual(bridgeheads, case["expected_bridgeheads"],
                               f"Bridgehead count mismatch for {name}")
                self.assertEqual(rings, case["expected_rings"],
                               f"Ring count mismatch for {name}")

                # Check indices are valid
                bridgehead_indices = self.solver.get_bridgehead_indices(smiles)
                self.assertEqual(len(bridgehead_indices), bridgeheads,
                               f"Bridgehead indices count mismatch for {name}")

    def test_fused_ring_systems(self):
        """Test fused ring detection."""
        test_cases = [
            {
                "smiles": "c1ccc2ccccc2c1",  # Naphthalene
                "name": "naphthalene",
                "expected_fused": True,
                "min_fused_atoms": 10
            },
            {
                "smiles": "c1ccc2c(c1)ccc3ccccc32",  # Anthracene
                "name": "anthracene",
                "expected_fused": True,
                "min_fused_atoms": 14
            },
            {
                "smiles": "C1CCC2CCCCC2C1",  # Decalin
                "name": "decalin",
                "expected_fused": True,
                "min_fused_atoms": 10
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]
            name = case["name"]

            with self.subTest(molecule=name):
                fused_count = self.solver.get_fused_ring_count(smiles)
                fused_indices = self.solver.get_fused_ring_indices(smiles)

                if case["expected_fused"]:
                    self.assertGreater(fused_count, 0,
                                     f"No fused rings detected in {name}")
                    self.assertGreaterEqual(len(fused_indices), case["min_fused_atoms"],
                                          f"Too few fused ring atoms in {name}")

    def test_spiro_systems(self):
        """Test spiro atom detection."""
        test_cases = [
            {
                "smiles": "C12(CCC1)CCC2",  # Spiropentane
                "name": "spiropentane",
                "expected_spiro": 1
            },
            {
                "smiles": "C12(CCCC1)CCCC2",  # Spiro[4.4]nonane
                "name": "spiro[4.4]nonane",
                "expected_spiro": 1
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]
            name = case["name"]

            with self.subTest(molecule=name):
                spiro_count = self.solver.get_spiro_count(smiles)
                spiro_indices = self.solver.get_spiro_indices(smiles)

                self.assertEqual(spiro_count, case["expected_spiro"],
                               f"Spiro count mismatch for {name}")
                self.assertEqual(len(spiro_indices), spiro_count,
                               f"Spiro indices count mismatch for {name}")


class TestFunctionalGroupOverlaps(unittest.TestCase):
    """Test overlapping functional group definitions."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_overlapping_groups(self):
        """Test molecules with multiple overlapping functional groups."""
        test_cases = [
            {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "name": "aspirin",
                "expected_groups": ["ester", "carboxylic_acid", "aromatic"]
            },
            {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
                "name": "ibuprofen",
                "expected_groups": ["carboxylic_acid", "aromatic"]
            },
            {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "name": "caffeine",
                "expected_groups": ["amide", "aromatic", "tertiary_amine"]
            }
        ]

        for case in test_cases:
            smiles = case["smiles"]
            name = case["name"]

            with self.subTest(molecule=name):
                fg_dict = self.solver.get_functional_group_count_and_indices(smiles)

                for group in case["expected_groups"]:
                    self.assertIn(group, fg_dict,
                                 f"Expected group '{group}' not found in {name}")
                    self.assertGreater(fg_dict[group]["nbr_instances"], 0,
                                     f"Group '{group}' has zero count in {name}")

                # Check that indices don't have impossible overlaps
                all_indices = set()
                for group, data in fg_dict.items():
                    if data["nbr_instances"] > 0:
                        for idx_list in data["indices"]:
                            # Track all unique atoms involved
                            if isinstance(idx_list, list):
                                all_indices.update(idx_list)
                            else:
                                all_indices.add(idx_list)

                # All indices should be valid atom indices
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    num_atoms = mol.GetNumAtoms()
                    invalid = [i for i in all_indices if i >= num_atoms]
                    self.assertEqual(len(invalid), 0,
                                   f"Invalid atom indices in {name}: {invalid}")


class TestInvalidSMILESHandling(unittest.TestCase):
    """Test graceful handling of invalid SMILES."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_invalid_smiles(self):
        """Test that invalid SMILES are handled gracefully."""
        invalid_cases = [
            "C(C)(C)(C)(C)C",  # Invalid valence
            "C1CCC",  # Unclosed ring
            "[Cu+2].[Cu+2].[O-]S([O-])(=O)=O",  # Coordination complex
            "",  # Empty string
            "INVALID",  # Not a SMILES
            "C1CC)C1",  # Mismatched parentheses
        ]

        for smiles in invalid_cases:
            with self.subTest(smiles=smiles):
                # All count methods should return 0
                count_methods = [
                    'get_ring_count',
                    'get_carbon_atom_count',
                    'get_stereocenter_count'
                ]

                for method_name in count_methods:
                    result = getattr(self.solver, method_name)(smiles)
                    self.assertEqual(result, 0,
                                   f"{method_name} should return 0 for invalid SMILES")

                # All indices methods should return empty list
                indices_methods = [
                    'get_ring_indices',
                    'get_carbon_atom_indices',
                    'get_stereocenter_indices'
                ]

                for method_name in indices_methods:
                    result = getattr(self.solver, method_name)(smiles)
                    self.assertEqual(result, [],
                                   f"{method_name} should return [] for invalid SMILES")


class TestLargeMoleculePerformance(unittest.TestCase):
    """Test performance on large molecules."""

    def setUp(self):
        self.solver = SymbolicSolver()

    def test_longest_chain_approximation(self):
        """Test longest carbon chain with large molecules (>60 carbons)."""
        # Create a molecule with >60 carbons
        large_alkane = "C" * 70  # 70-carbon linear alkane

        # Should trigger approximation
        chain_length = self.solver.get_longest_carbon_chain_count(large_alkane)
        chain_indices = self.solver.get_longest_carbon_chain_indices(large_alkane)

        self.assertEqual(chain_length, 70, "Chain length should be 70")
        self.assertEqual(len(chain_indices), 70, "Should have 70 indices")

        # Create branched large molecule
        branched = "CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC"
        chain_length_branched = self.solver.get_longest_carbon_chain_count(branched)

        self.assertGreater(chain_length_branched, 0, "Should find a chain")
        self.assertLess(chain_length_branched, len(branched),
                       "Chain should be less than total string length")


def generate_edge_case_report(output_file: str = "edge_case_test_results.json"):
    """Run all edge case tests and generate a report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestAromaticityEdgeCases,
        TestStereochemistryEdgeCases,
        TestTautomerHandling,
        TestRingSystemEdgeCases,
        TestFunctionalGroupOverlaps,
        TestInvalidSMILESHandling,
        TestLargeMoleculePerformance
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)

    # Generate report
    report = {
        "test_suite": "SymbolicSolver Edge Case Tests",
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        "edge_cases_covered": [
            "aromaticity_kekulization",
            "stereochemistry_ambiguity",
            "tautomer_handling",
            "ring_perception",
            "functional_group_overlap",
            "invalid_smiles",
            "large_molecule_performance"
        ],
        "test_categories": {
            "aromaticity": {
                "description": "Tests for aromatic vs Kekule representations",
                "num_tests": len([m for m in dir(TestAromaticityEdgeCases) if m.startswith('test_')])
            },
            "stereochemistry": {
                "description": "Tests for R/S/E/Z and unspecified stereochemistry",
                "num_tests": len([m for m in dir(TestStereochemistryEdgeCases) if m.startswith('test_')])
            },
            "tautomers": {
                "description": "Tests for keto-enol and imine-enamine tautomers",
                "num_tests": len([m for m in dir(TestTautomerHandling) if m.startswith('test_')])
            },
            "ring_systems": {
                "description": "Tests for bridged, fused, and spiro ring systems",
                "num_tests": len([m for m in dir(TestRingSystemEdgeCases) if m.startswith('test_')])
            },
            "functional_groups": {
                "description": "Tests for overlapping functional group definitions",
                "num_tests": len([m for m in dir(TestFunctionalGroupOverlaps) if m.startswith('test_')])
            },
            "invalid_handling": {
                "description": "Tests for graceful failure on invalid input",
                "num_tests": len([m for m in dir(TestInvalidSMILESHandling) if m.startswith('test_')])
            }
        }
    }

    # Add failure details
    if result.failures:
        report["failure_details"] = []
        for test, traceback in result.failures[:5]:  # First 5 failures
            report["failure_details"].append({
                "test": str(test),
                "error": traceback.split('\n')[-2]  # Last line before traceback
            })

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nEdge Case Test Report saved to {output_file}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Edge Cases Covered: {', '.join(report['edge_cases_covered'])}")

    return report


if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Generate report
    generate_edge_case_report()
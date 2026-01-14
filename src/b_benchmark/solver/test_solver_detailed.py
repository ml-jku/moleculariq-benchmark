#!/usr/bin/env python3
"""
Detailed solver testing to show actual values obtained.
"""

import sys
import pickle
import random
from pathlib import Path
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from solver import SymbolicSolver
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdmd


def test_detailed_values(properties_path: str, n_samples: int = 100):
    """Test solver and show actual values obtained."""

    # Load properties
    with open(properties_path, 'rb') as f:
        df = pickle.load(f)

    # Find SMILES column
    smiles_col = None
    for col in ['canonical_smiles', 'smiles', 'SMILES', 'original_smiles']:
        if col in df.columns:
            smiles_col = col
            break

    if not smiles_col:
        raise ValueError("No SMILES column found")

    # Sample molecules
    all_smiles = df[smiles_col].dropna().tolist()
    valid_smiles = [s for s in all_smiles if s and s != "..."]

    random.seed(42)
    test_molecules = random.sample(valid_smiles, min(n_samples, len(valid_smiles)))

    print(f"Testing {len(test_molecules)} molecules...")
    print("="*80)

    solver = SymbolicSolver()
    statistics = defaultdict(list)

    # Test first 10 molecules in detail
    for i, smiles in enumerate(test_molecules[:10]):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        print(f"\nMolecule {i+1}: {smiles[:50]}...")
        print("-"*40)

        # Basic properties
        ring_count = solver.get_ring_count(smiles)
        carbon_count = solver.get_carbon_atom_count(smiles)
        heavy_atom_count = solver.get_heavy_atom_count(smiles)

        print(f"  Ring count: {ring_count}")
        print(f"  Carbon atoms: {carbon_count}")
        print(f"  Heavy atoms: {heavy_atom_count}")

        # RDKit comparison
        rdkit_heavy = mol.GetNumHeavyAtoms()
        print(f"  RDKit heavy atoms: {rdkit_heavy} (match: {heavy_atom_count == rdkit_heavy})")

        # Stereochemistry
        total_stereo = solver.get_stereocenter_count(smiles)
        if total_stereo > 0:
            r_count = solver.get_r_or_s_stereocenter_count(smiles, r_count=True)
            s_count = solver.get_r_or_s_stereocenter_count(smiles, r_count=False)
            unspec = solver.get_unspecified_stereocenter_count(smiles)
            print(f"  Stereocenters: Total={total_stereo}, R={r_count}, S={s_count}, Unspecified={unspec}")
            print(f"    Sum check: {r_count + s_count + unspec} == {total_stereo} ✓")

        # Aromatic rings
        aromatic = solver.get_aromatic_ring_count(smiles)
        if aromatic > 0:
            rdkit_aromatic = Descriptors.NumAromaticRings(mol)
            print(f"  Aromatic rings: {aromatic} (RDKit: {rdkit_aromatic}, match: {aromatic == rdkit_aromatic})")

        # Special features
        bridgehead = solver.get_bridgehead_count(smiles)
        if bridgehead > 0:
            rdkit_bridgehead = rdmd.CalcNumBridgeheadAtoms(mol)
            print(f"  Bridgehead atoms: {bridgehead} (RDKit: {rdkit_bridgehead}, match: {bridgehead == rdkit_bridgehead})")

        spiro = solver.get_spiro_count(smiles)
        if spiro > 0:
            rdkit_spiro = rdmd.CalcNumSpiroAtoms(mol)
            print(f"  Spiro atoms: {spiro} (RDKit: {rdkit_spiro}, match: {spiro == rdkit_spiro})")

    # Collect statistics for all molecules
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)

    rdkit_matches = {'heavy_atoms': 0, 'aromatic_rings': 0, 'total': 0}
    stereochem_consistent = 0

    for smiles in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        rdkit_matches['total'] += 1

        # Test heavy atoms
        solver_heavy = solver.get_heavy_atom_count(smiles)
        rdkit_heavy = mol.GetNumHeavyAtoms()
        if solver_heavy == rdkit_heavy:
            rdkit_matches['heavy_atoms'] += 1

        # Test aromatic rings
        solver_aromatic = solver.get_aromatic_ring_count(smiles)
        rdkit_aromatic = Descriptors.NumAromaticRings(mol)
        if solver_aromatic == rdkit_aromatic:
            rdkit_matches['aromatic_rings'] += 1

        # Test stereochemistry consistency
        total = solver.get_stereocenter_count(smiles)
        r = solver.get_r_or_s_stereocenter_count(smiles, r_count=True)
        s = solver.get_r_or_s_stereocenter_count(smiles, r_count=False)
        unspec = solver.get_unspecified_stereocenter_count(smiles)
        if total == r + s + unspec:
            stereochem_consistent += 1

    print(f"Molecules tested: {rdkit_matches['total']}")
    print(f"Heavy atom count matches RDKit: {rdkit_matches['heavy_atoms']}/{rdkit_matches['total']} "
          f"({100*rdkit_matches['heavy_atoms']/rdkit_matches['total']:.1f}%)")
    print(f"Aromatic ring count matches RDKit: {rdkit_matches['aromatic_rings']}/{rdkit_matches['total']} "
          f"({100*rdkit_matches['aromatic_rings']/rdkit_matches['total']:.1f}%)")
    print(f"Stereochemistry internally consistent: {stereochem_consistent}/{rdkit_matches['total']} "
          f"({100*stereochem_consistent/rdkit_matches['total']:.1f}%)")

    # Test specific edge cases
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80)

    edge_cases = [
        ("c1ccccc1", "C1=CC=CC=C1", "Benzene aromatic vs Kekule"),
        ("CC(=O)CC", "CC(O)=CC", "Keto vs enol tautomer"),
        ("C[C@H](Cl)Br", "C[C@@H](Cl)Br", "R vs S stereoisomer"),
    ]

    for case1, case2, description in edge_cases:
        print(f"\n{description}:")
        print(f"  SMILES 1: {case1}")
        print(f"  SMILES 2: {case2}")

        # Compare key properties
        for method_name in ['get_carbon_atom_count', 'get_heavy_atom_count', 'get_ring_count']:
            method = getattr(solver, method_name)
            val1 = method(case1)
            val2 = method(case2)
            match = "✓" if val1 == val2 else "✗"
            print(f"    {method_name}: {val1} vs {val2} {match}")


if __name__ == "__main__":
    properties_path = '/system/user/publicdata/chemical_reasoning/moleculariq/properties_new.pkl'
    test_detailed_values(properties_path, n_samples=100)
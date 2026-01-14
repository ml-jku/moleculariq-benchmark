#!/usr/bin/env python3
"""
Generate diverse SMILES for comprehensive solver testing.

Creates a test set of 10,000 molecules with good coverage of edge cases.
"""

import os
import random
import pickle
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from tqdm import tqdm
import json


def categorize_molecule(mol: Chem.Mol, smiles: str) -> List[str]:
    """Categorize a molecule based on its properties."""
    categories = []

    if mol is None:
        return ["invalid"]

    # Size categories
    num_atoms = mol.GetNumHeavyAtoms()
    if num_atoms < 10:
        categories.append("small")
    elif num_atoms < 30:
        categories.append("medium")
    else:
        categories.append("large")

    # Aromaticity
    has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    if has_aromatic:
        categories.append("aromatic")

    # Stereochemistry
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if chiral_centers:
        categories.append("stereochemistry")
        if any(tag == '?' for _, tag in chiral_centers):
            categories.append("unspecified_stereochemistry")

    # Double bond stereochemistry
    if '/' in smiles or '\\' in smiles:
        categories.append("ez_stereochemistry")

    # Ring systems
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()

    if num_rings > 0:
        categories.append("rings")

        # Check for fused rings
        rings = [set(ring) for ring in ring_info.AtomRings()]
        for i, ring1 in enumerate(rings):
            for ring2 in rings[i+1:]:
                if len(ring1 & ring2) >= 2:
                    categories.append("fused_rings")
                    break

        # Check for bridged rings (simplified check)
        if num_rings >= 2:
            # Look for bridgehead atoms
            atom_ring_count = {}
            for ring in ring_info.AtomRings():
                for atom_idx in ring:
                    atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1

            if any(count > 2 for count in atom_ring_count.values()):
                categories.append("bridged_rings")

            # Check for spiro
            if any(count == 2 for count in atom_ring_count.values()):
                # Additional check: atom should be in exactly 2 rings
                for atom_idx, count in atom_ring_count.items():
                    if count == 2:
                        atom_rings = [r for r in rings if atom_idx in r]
                        if len(atom_rings) == 2 and len(atom_rings[0] & atom_rings[1]) == 1:
                            categories.append("spiro")
                            break

    # Heterocycles
    if num_rings > 0:
        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() not in ['C', 'H']:
                    categories.append("heterocycle")
                    break

    # Functional groups
    functional_group_smarts = {
        "alcohol": "[OH]",
        "amine": "[NX3;H2,H1;!$(NC=O)]",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "ester": "[#6][CX3](=O)[OX2H0][#6]",
        "ketone": "[CX3](=O)[CX4]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "amide": "[NX3][CX3](=[OX1])[#6]",
        "halogen": "[F,Cl,Br,I]"
    }

    for fg_name, smarts in functional_group_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            categories.append(f"fg_{fg_name}")

    # Special molecules
    if num_atoms > 60:
        categories.append("very_large")

    # Linear alkanes
    if not has_aromatic and num_rings == 0 and all(atom.GetSymbol() in ['C', 'H'] for atom in mol.GetAtoms()):
        categories.append("alkane")

    return categories


def generate_edge_case_molecules() -> List[str]:
    """Generate specific edge case molecules for testing."""
    edge_cases = []

    # Aromaticity edge cases
    aromatic_cases = [
        "c1ccccc1",  # benzene (aromatic)
        "C1=CC=CC=C1",  # benzene (Kekule)
        "c1ccc2ccccc2c1",  # naphthalene
        "c1ccc2c(c1)ncn2",  # benzimidazole
        "c1ccccc1-c2ccccc2",  # biphenyl
        "c1ccc2c(c1)[nH]c1ccccc12",  # carbazole
        "[O-][n+]1ccccc1",  # pyridine N-oxide
    ]
    edge_cases.extend(aromatic_cases)

    # Stereochemistry edge cases
    stereo_cases = [
        "C[C@H](Cl)Br",  # R stereocenter
        "C[C@@H](Cl)Br",  # S stereocenter
        "CC(Cl)Br",  # unspecified stereocenter
        "C[C@H](O)[C@H](O)C",  # multiple stereocenters
        "C/C=C/C",  # E double bond
        "C/C=C\\C",  # Z double bond
        "CC=CC",  # unspecified double bond
        "C[C@H]1CC[C@@H](C)CC1",  # ring with stereocenters
    ]
    edge_cases.extend(stereo_cases)

    # Tautomer pairs
    tautomer_cases = [
        "CC(=O)CC",  # keto form
        "CC(O)=CC",  # enol form
        "CC(=O)C",  # acetone
        "CC(O)=C",  # propen-2-ol
        "O=C1CCCCC1",  # cyclohexanone
        "OC1=CCCCC1",  # cyclohexenol
    ]
    edge_cases.extend(tautomer_cases)

    # Complex ring systems
    ring_cases = [
        "C1CC2CCC1C2",  # norbornane (bridged)
        "C1C2CC3CC1CC(C2)C3",  # adamantane
        "C12CCC1CC2",  # bicyclo[2.2.1]heptane
        "C12(CCC1)CCC2",  # spiropentane
        "C12(CCCC1)CCCC2",  # spiro[4.4]nonane
        "c1ccc2c(c1)CCC2",  # tetralin (fused aromatic/aliphatic)
        "O1c2ccccc2CC1",  # chromane
        "C1CCC2CCCCC2C1",  # decalin
    ]
    edge_cases.extend(ring_cases)

    # Large molecules
    large_cases = [
        "C" * 70,  # 70-carbon alkane
        "CC(C)" * 20 + "C",  # heavily branched alkane
        "c1ccccc1" * 5,  # polyphenyl
    ]
    edge_cases.extend(large_cases)

    # Functional group combinations
    fg_cases = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)NC1=CC=C(C=C1)O",  # acetaminophen
    ]
    edge_cases.extend(fg_cases)

    # Invalid/problematic SMILES
    invalid_cases = [
        "",  # empty
        "INVALID",  # not SMILES
        "C(C)(C)(C)(C)C",  # invalid valence
        "C1CCC",  # unclosed ring
        "[Cu+2].[O-]S([O-])(=O)=O",  # coordination complex
    ]
    edge_cases.extend(invalid_cases)

    return edge_cases


def select_diverse_molecules(molecules: List[Tuple[str, List[str]]],
                            n_total: int = 10000) -> List[str]:
    """Select diverse molecules based on categories."""

    # Count molecules per category
    category_counts = {}
    for smiles, categories in molecules:
        for cat in categories:
            if cat not in category_counts:
                category_counts[cat] = []
            category_counts[cat].append(smiles)

    print("\nCategory distribution:")
    for cat, mols in sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(mols)} molecules")

    # Target distribution
    targets = {
        "small": 1000,
        "medium": 2000,
        "large": 2000,
        "aromatic": 1000,
        "heterocycle": 1000,
        "stereochemistry": 1000,
        "rings": 1000,
        "fused_rings": 500,
        "bridged_rings": 200,
        "spiro": 100,
        "alkane": 500,
        "fg_carboxylic_acid": 300,
        "fg_ester": 300,
        "fg_amine": 300,
        "very_large": 100,
    }

    selected = set()
    selected_categories = {}

    # First pass: try to meet targets
    for cat, target in targets.items():
        if cat in category_counts:
            available = [m for m in category_counts[cat] if m not in selected]
            to_select = min(target, len(available))
            if to_select > 0:
                chosen = random.sample(available, to_select)
                selected.update(chosen)
                selected_categories[cat] = chosen
                print(f"  Selected {to_select} molecules for {cat}")

    # Fill remaining slots with random molecules
    all_smiles = [smiles for smiles, _ in molecules if smiles not in selected]
    remaining = n_total - len(selected)

    if remaining > 0 and all_smiles:
        additional = random.sample(all_smiles, min(remaining, len(all_smiles)))
        selected.update(additional)
        print(f"  Added {len(additional)} random molecules")

    return list(selected)


def load_molecules_from_dataset(dataset_path: str,
                               max_molecules: int = 50000) -> List[str]:
    """Load molecules from a dataset file."""
    molecules = []

    if dataset_path.endswith('.pkl'):
        # Load pickle file
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, pd.DataFrame):
            if 'smiles' in data.columns:
                molecules = data['smiles'].dropna().tolist()[:max_molecules]
            elif 'SMILES' in data.columns:
                molecules = data['SMILES'].dropna().tolist()[:max_molecules]

    elif dataset_path.endswith('.parquet'):
        # Load parquet file
        df = pd.read_parquet(dataset_path)
        if 'smiles' in df.columns:
            molecules = df['smiles'].dropna().tolist()[:max_molecules]

    elif dataset_path.endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(dataset_path)
        if 'smiles' in df.columns:
            molecules = df['smiles'].dropna().tolist()[:max_molecules]

    return molecules


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate diverse test molecules")
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset file (pkl, parquet, csv)')
    parser.add_argument('--output', type=str, default='test_molecules_10k.json',
                       help='Output file for test molecules')
    parser.add_argument('--n_molecules', type=int, default=10000,
                       help='Number of test molecules to generate')
    parser.add_argument('--properties_path', type=str,
                       default='/system/user/publicdata/chemical_reasoning/moleculariq/properties_new.pkl',
                       help='Path to properties dataframe')

    args = parser.parse_args()

    print("Generating diverse test molecules...")

    # Start with edge cases
    edge_cases = generate_edge_case_molecules()
    print(f"Generated {len(edge_cases)} edge case molecules")

    all_molecules = []

    # Load from dataset if provided
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading molecules from {args.dataset}...")
        dataset_molecules = load_molecules_from_dataset(args.dataset)
        print(f"Loaded {len(dataset_molecules)} molecules from dataset")
    else:
        # Try to load from default properties file
        if os.path.exists(args.properties_path):
            print(f"Loading molecules from {args.properties_path}...")
            dataset_molecules = load_molecules_from_dataset(args.properties_path)
            print(f"Loaded {len(dataset_molecules)} molecules from properties")
        else:
            print("No dataset found, using only edge cases and generated molecules")
            dataset_molecules = []

    # Categorize molecules
    print("\nCategorizing molecules...")
    categorized = []

    for smiles in tqdm(edge_cases + dataset_molecules[:50000], desc="Categorizing"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            categories = categorize_molecule(mol, smiles)
            categorized.append((smiles, categories))
        except Exception as e:
            # Invalid molecules go to "invalid" category
            categorized.append((smiles, ["invalid"]))

    # Select diverse set
    print(f"\nSelecting {args.n_molecules} diverse molecules...")
    selected_molecules = select_diverse_molecules(categorized, args.n_molecules)

    # Validate selection
    print(f"\nValidating {len(selected_molecules)} selected molecules...")
    valid_count = 0
    invalid_count = 0
    category_stats = {}

    for smiles in selected_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_count += 1
            categories = categorize_molecule(mol, smiles)
            for cat in categories:
                category_stats[cat] = category_stats.get(cat, 0) + 1
        else:
            invalid_count += 1

    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print("\nFinal category distribution:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    # Save to file
    output_data = {
        "n_molecules": len(selected_molecules),
        "valid_molecules": valid_count,
        "invalid_molecules": invalid_count,
        "category_distribution": category_stats,
        "molecules": selected_molecules
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(selected_molecules)} test molecules to {args.output}")

    # Also save just the SMILES list for easy loading
    smiles_file = args.output.replace('.json', '_smiles.txt')
    with open(smiles_file, 'w') as f:
        for smiles in selected_molecules:
            f.write(f"{smiles}\n")
    print(f"Saved SMILES list to {smiles_file}")

    return selected_molecules


if __name__ == "__main__":
    main()
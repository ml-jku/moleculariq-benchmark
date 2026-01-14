# MolecularIQ Benchmark

A comprehensive benchmark dataset for evaluating large language models on molecular reasoning tasks, featuring count, index, and constraint generation challenges across diverse chemical properties.

![MolecularIQ benchmark statistics](assets/img/moleculariq_statistics.png)

## Overview

MolecularIQ is a benchmark specifically designed to measure the structural reasoning abilities of large language models on molecules. Unlike many chemistry evaluation sets that rely on literature labels or surrogate predictors, MolecularIQ focuses only on tasks whose correctness can be verified algorithmically from the molecular graph itself. This makes it possible to distinguish genuine structural understanding from memorization or correlation-based answers.

### Key characteristics of MolecularIQ

## Repository Structure

```
moleculariq-benchmark/
├── src/                                # Source code
│   ├── a_dataset_pools/               # Stage A: Dataset pool creation
│   │   ├── 1_collect_pubchem_data.py
│   │   ├── 2_collect_external_test_set_molecules.py
│   │   ├── 3_standardize_pubchem_mols_and_remove_external_test_mols.py
│   │   ├── 4_create_train_test_pools.py
│   │   ├── 5_create_hard_test_pool_dataframe.py
│   │   └── utils/                     # External test set utilities
│   └── b_benchmark/                   # Stage B: Benchmark generation
│       ├── compute_properties.py      # Compute ground truth properties
│       ├── generate_benchmark_dataset.py  # Generate final dataset
│       ├── solver/                    # Property computation solvers
│       ├── natural_language/          # Question formatting
│       ├── rewards/                   # Evaluation metrics
│       └── questions.py               # Task definitions
├── data/                              # Data artifacts (not tracked)
│   ├── dataset_pools/                 # Molecule pools
│   │   ├── external/                  # External benchmark molecules
│   │   ├── intermediate/              # Pipeline intermediates
│   │   ├── processed/                 # Processed datasets
│   │   ├── pseudo_sdf/               # Sample SDF for testing
│   │   └── pubchem_raw_sdf/          # Raw PubChem SDF files
│   └── benchmark/                     # Generated benchmark data
│       └── properties.pkl
├── notebooks/                         # Analysis notebooks
│   └── overview_created_data.ipynb   # Data creation walkthrough
└── assets/                            # Documentation assets
    └── moleculariq_statistics.pdf
```

## Data Creation Pipeline

### Stage A: Dataset Pool Creation

1. **Collect PubChem Data** ([1_collect_pubchem_data.py](src/a_dataset_pools/1_collect_pubchem_data.py))
   - Extract SMILES and IUPAC names from PubChem SDF files
   - Filter molecules (carbon-containing, single-fragment)

2. **Collect External Test Sets** ([2_collect_external_test_set_molecules.py](src/a_dataset_pools/2_collect_external_test_set_molecules.py))
   - Aggregate molecules from LLaSMol, ChemDFM, Ether0, ChemIQ benchmarks

3. **Standardize and Filter** ([3_standardize_pubchem_mols_and_remove_external_test_mols.py](src/a_dataset_pools/3_standardize_pubchem_mols_and_remove_external_test_mols.py))
   - Canonicalize SMILES
   - Remove molecules present in external benchmarks

4. **Create Train/Test Pools** ([4_create_train_test_pools.py](src/a_dataset_pools/4_create_train_test_pools.py))
   - Cluster molecules using MinHash LSH on Morgan fingerprints
   - Split into: Training pool, Easy test set, Hard test set

5. **Create Hard Test Pool DataFrame** ([5_create_hard_test_pool_dataframe.py](src/a_dataset_pools/5_create_hard_test_pool_dataframe.py))
   - Build structured dataframe with molecular complexity metrics

### Stage B: Benchmark Generation

1. **Compute Properties** ([compute_properties.py](src/b_benchmark/compute_properties.py))
   - Calculate ground truth values for all chemical properties
   - Uses symbolic solver for accurate property computation

2. **Generate Benchmark Dataset** ([generate_benchmark_dataset.py](src/b_benchmark/generate_benchmark_dataset.py))
   - Sample diverse datapoints across complexity dimensions
   - Generate questions using natural language templates
   - Create count, index, and constraint generation tasks
   - Export to HuggingFace dataset format

## Getting Started

### Prerequisites

```bash
pip install rdkit pandas numpy tqdm datasets huggingface_hub
```

### Quick Start

1. **Download PubChem SDF files** (optional - a pseudo SDF is included for testing):
   ```bash
   # Download from https://pubchem.ncbi.nlm.nih.gov/docs/downloads
   # Place in data/dataset_pools/pubchem_raw_sdf/
   ```

2. **Run the data creation pipeline**:
   ```bash
   # Stage A: Create molecule pools
   cd src/a_dataset_pools
   python 1_collect_pubchem_data.py
   python 2_collect_external_test_set_molecules.py
   python 3_standardize_pubchem_mols_and_remove_external_test_mols.py
   python 4_create_train_test_pools.py
   python 5_create_hard_test_pool_dataframe.py

   # Stage B: Generate benchmark
   cd ../b_benchmark
   python compute_properties.py
   python generate_benchmark_dataset.py
   ```

3. **Explore the created data**:
   ```bash
   jupyter notebook notebooks/overview_created_data.ipynb
   ```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


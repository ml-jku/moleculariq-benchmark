# MolecularIQ Benchmark Submission Package

This folder contains everything needed to recreate the MolecularIQ benchmark that
accompanies our anonymous ICLR submission.  The code assumes the user supplies the
necessary source datasets (PubChem SDF shards and external benchmark SMILES).  Once
placed in the documented locations the pipeline can be executed end-to-end without
any additional dependencies outside this submission tree.

## Folder map

- `A_create_dataset_pools/`
  - `1_collect_pubchem_data.py` extracts SMILES/IUPAC pairs from `data/pubchem_raw_sdf`.
  - `2_collect_external_test_set_molecules.py` aggregates molecules from benchmark snapshots stored in `data/external/*.smi` and writes a merged pickle.
  - `3_standardize_pubchem_mols_and_remove_external_test_mols.py` canonicalises the PubChem cache and removes overlaps with the external sets (falls back to the original list when everything would be filtered away).
  - `4_create_train_test_pools.py` partitions the filtered molecules into train/easy/hard pools. LSH-based deduplication is used when `datasketch` is installed; otherwise a deterministic set difference fallback is applied.
- `B_create_benchmark/`
  - `A_compute_properties.py` evaluates every task-specific property for the hard pool and stores the results in `data/properties_new.pkl`. Multiprocessing and pandarallel are opt-in via `ENABLE_PANDARALLEL=1`.
  - `generate_benchmark_dataset.py` samples count/index/constraint tasks and produces the final JSON dataset. All configuration happens via CLI flags; defaults are tuned for the shipped sample data.
  - `solver/` contains the symbolic solvers and helper resources used by both the property computation and task generation stages.
- `natural_language/` text templates and formatters for phrasing tasks in natural language.
- `rewards/` reward functions used to verify constraint satisfaction.
- `questions.py` registry of task definitions consumed by the generator.
- `data/` (created at runtime) caches intermediate artefacts such as `properties_new.pkl` and `sample_tasks.json`.

## Quick start (sample data)

All commands assume the repository root as the working directory:

```bash
# 1. Build the molecule pools (requires local data sources)
python src/submission/A_create_dataset_pools/1_collect_pubchem_data.py
python src/submission/A_create_dataset_pools/2_collect_external_test_set_molecules.py
python src/submission/A_create_dataset_pools/3_standardize_pubchem_mols_and_remove_external_test_mols.py
python src/submission/A_create_dataset_pools/4_create_train_test_pools.py

# 2. Compute task properties for the hard pool
python src/submission/B_create_benchmark/A_compute_properties.py

# 3. Generate a compact benchmark (no constraint tasks, finishes in seconds)
python src/submission/B_create_benchmark/generate_benchmark_dataset.py \
  --pickle-path src/submission/B_create_benchmark/data/properties_new.pkl \
  --output-path src/submission/B_create_benchmark/data/sample_tasks.json \
  --n-samples-per-bin 1 --n-samples-per-category 0 --n-samples-multi 1
```

The final dataset will be written to
`src/submission/B_create_benchmark/data/sample_tasks.json`, and all intermediate
artefacts remain inside the submission folder.

## Using full datasets

1. Download the relevant PubChem `*.sdf.gz` shards and place them under
   `src/submission/A_create_dataset_pools/data/pubchem_raw_sdf/` (subdirectories are
   supported).
2. Place one SMILES-per-line text file for each external benchmark inside
   `src/submission/A_create_dataset_pools/data/external/` using the filenames
   referenced in `2_collect_external_test_set_molecules.py` (e.g.
   `llasmol_test_set.smi`).
3. Re-run the commands from the quick start section.  The outputs in
   `data/intermediate`, `data/processed`, and `B_create_benchmark/data` are
   overwritten automatically.

To speed up heavy runs you can enable optional parallel features:

- Set `ENABLE_PANDARALLEL=1` before calling
  `A_compute_properties.py` if `pandarallel` is installed.
- Install `datasketch` to re-enable MinHash-based filtering in
  `4_create_train_test_pools.py`.

## Dependencies

The scripts rely on standard scientific Python packages:
`rdkit`, `numpy`, `pandas`, `datasets`, `tqdm`, and `huggingface_hub`.  Optional
packages (`datasketch`, `pandarallel`) are detected automatically when present.


## Verification

Running the quick start commands with the required inputs produces
`sample_tasks.json` (or the specified output path) and stores every intermediate
artefact under `src/submission`.  No external services are contacted unless you
explicitly invoke `--push-to-hub`.

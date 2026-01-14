"""
Split filtered PubChem molecules into training and evaluation buckets.

To allow for both future model training on benchmark tasks as well as unbiased evaluation,
we create three pools from the filtered PubChem data by performance a cluster-based 
split:
1) A training pool: Contains molecules from which training samples can be built.
2) An easy test set: Contains molecules that are drawn from clusters present in the 
   training pool.
3) A hard test set: Contains molecules that are drawn from clusters not present in the
   training pool.

The splitting is performed using MinHash LSH to cluster similar molecules based on
their Morgan fingerprints. Emerging pools build disjoint sets of molecules.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from datasketch import MinHash, MinHashLSH

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Debugging flag
# When False:
# Assume all Pubchem data has been processed -> create large pools
# When True:
# Assume only small amount of pseudo data is available -> create small pools
DEBUGGING = True

SUBMISSION_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SUBMISSION_ROOT / "data" / "processed" / "filtered_smiles_and_iupacs.pkl"
DEFAULT_OUTPUT_PATH = SUBMISSION_ROOT / "data" / "processed" / "pubchem_train_test_pools.pkl"


@dataclass
class CreateTrainTestPoolsConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    nbr_mols_hard_test_set: int = 1_000_000 if not DEBUGGING else 1_000
    nbr_mols_second_pool: int = 6_000_000 if not DEBUGGING else 6_000
    nbr_mols_easy_test_set: int = 1_000_000 if not DEBUGGING else 1_000
    similarity_threshold: float = 0.7
    num_perm: int = 128
    random_seed: int = 1019 # To not reveal our true pools, we changed the seed after creating the benchmark

# Note the second pool is used to create the training pool by keeping only molecules
# that are dissimilar to the hard test set. From the filtered second pool, we then 
# sample the easy test set and use the remaining molecules as training set.

def get_minhash(smiles: str, num_perm: int = 128) -> MinHash | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    mh = MinHash(num_perm=num_perm)
    for bit in fp.GetOnBits():
        mh.update(str(bit).encode("utf8"))
    return mh


def _sample(smiles: Sequence[str], k: int) -> List[str]:
    if not smiles:
        return []
    k = min(len(smiles), k)
    return random.sample(list(smiles), k)


def create_train_test_pools(cfg: CreateTrainTestPoolsConfig) -> Tuple[List[str], List[str], List[str], dict]:
    # Set random seed for reproducibility
    random.seed(cfg.random_seed)

    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        with input_path.open("rb") as handle:
            filtered_smiles_list, iupac_dict = pickle.load(handle)
    else:
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    hard_test_set = _sample(filtered_smiles_list, cfg.nbr_mols_hard_test_set)
    second_pool = _sample(filtered_smiles_list, cfg.nbr_mols_second_pool)

    filtered_second_pool: List[str] = []
    lsh = MinHashLSH(threshold=cfg.similarity_threshold, num_perm=cfg.num_perm)
    for idx, smi in tqdm(enumerate(hard_test_set), total=len(hard_test_set), desc="Inserting hard set"):
        mh = get_minhash(smi, cfg.num_perm)
        if mh:
            lsh.insert(f"mol_{idx}", mh)

    for smi in tqdm(second_pool, desc="Filtering second pool"):
        mh = get_minhash(smi, cfg.num_perm)
        if mh and not lsh.query(mh):
            filtered_second_pool.append(smi)

    easy_test_set = _sample(filtered_second_pool, cfg.nbr_mols_easy_test_set)
    training_set = sorted(set(filtered_second_pool) - set(easy_test_set))

    with output_path.open("wb") as handle:
        pickle.dump([hard_test_set, easy_test_set, training_set, iupac_dict], handle)

    return hard_test_set, easy_test_set, training_set, iupac_dict


if __name__ == "__main__":
    create_train_test_pools(CreateTrainTestPoolsConfig())

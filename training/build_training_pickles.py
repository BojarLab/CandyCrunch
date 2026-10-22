"""
Helper script to convert the output of training/cc_training_data.py (train.pkl: labelled model inputs with their candidate
sets; pretrain/<project>.pkl: unlabelled ones) into CandyCrunch-ready training pickles: X/y tuples for training_script.py,
per-sample candidate masks and the pretraining inputs.  Adjust the paths in the configuration block before running.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import glycan_to_composition

cc_training_dir = Path("cc_training")
exclude_projects = set()               # GlycoPost_IDs left out of training and pretraining
output_dir = Path("prepared_datasets")
test_size = 0.15
random_state = 42

# Each row's settings as wrap_inference passed them, coded as process_for_inference codes them (anything else gets the last code)
MODE_MAP = {"negative": 1, "positive": 0}
LC_MAP = {"PGC": 0, "C18": 1}
MOD_MAP = {"reduced": 0, "permethylated": 1}
TRAP_MAP = {"linear": 0, "orbitrap": 1, "amazon": 2}

# The model's inputs in the order it takes them; compositional_vector, glycan_type and RT2 are the row's as the model got them at inference
FEATURE_COLUMNS = [
    "binned_intensities",
    "mz_remainder",
    "compositional_vector",
    "glycan_type",
    "RT2",
    "mode",
    "lc",
    "modification",
    "trap",
]
# Pretraining inputs keep the spectrum sparse (model_input: float32 rows of bin index, binned intensity and m/z remainder of each non-empty bin)
PRETRAIN_COLUMNS = ["model_input"] + FEATURE_COLUMNS[2:]



def _readable(glycan):
    # Structures glycowork cannot turn into a composition and a graph cannot be trained on (training_script.py drops them)
    try:
        glycan_to_composition(glycan)
        glycan_to_nxGraph(glycan)
        return True
    except Exception:
        return False

def _encode_settings(df):
    return df.assign(mode=df["mode"].map(MODE_MAP), lc=df["lc_type"].map(lambda v: LC_MAP.get(v, 2)),
                     modification=df["modification"].map(lambda v: MOD_MAP.get(v, 2)), trap=df["trap"].map(lambda v: TRAP_MAP.get(v, 3)))

def candidate_mask(df, glycans):
    # mask[i, j]: glycans[j] is in sample i's candidate set (its label and the isomers its map lists at its RT), for a loss on the set's summed probability
    index = {g: j for j, g in enumerate(glycans)}
    mask = np.zeros((len(df), len(glycans)), dtype=bool)
    for i, candidates in enumerate(df["glycan_candidates"]):
        mask[i, [index[g] for g in candidates]] = True
    return mask

def downcast_numeric(df):
    result = df.copy()
    int_cols = result.select_dtypes(include=["int", "uint", "int64", "uint64"]).columns
    float_cols = result.select_dtypes(include=["float", "float64"]).columns
    for col in int_cols:
        result[col] = pd.to_numeric(result[col], downcast="unsigned")
    for col in float_cols:
        result[col] = pd.to_numeric(result[col], downcast="float")
    return result

def tupleify(df, columns):
    return list(df[list(columns)].itertuples(index=False, name=None))


print("Loading labelled inputs")
train = pd.read_pickle(cc_training_dir / "train.pkl")
train = train[~train["GlycoPost_ID"].isin(exclude_projects)]
structures = set(train["glycan"]).union(*train["glycan_candidates"])
readable = {g for g in structures if _readable(g)}
print(f"Labelled inputs: {len(train)}, structures: {len(structures)}, unreadable: {len(structures) - len(readable)}")
# Unreadable structures leave the candidate sets: an input whose set is then empty goes, one whose nearest-RT label went takes the next of its set
train = train.assign(glycan_candidates=[[g for g in c if g in readable] for c in train["glycan_candidates"]])
train = train[train["glycan_candidates"].map(len) > 0]
train = train.assign(glycan=train["glycan_candidates"].map(lambda c: c[0]))
# The binned spectrum and m/z remainders the model got (the mean over the pooled spectra, which binning the pooled peak_d does not give back)
binned, remainders = [], []
for sparse in train["model_input"]:
    idx = sparse[0].astype(int)
    b = np.zeros(2048, dtype=np.float32)
    r = np.zeros(2048, dtype=np.float32)
    b[idx] = sparse[1]
    r[idx] = sparse[2]
    binned.append(b)
    remainders.append(r)
combined = _encode_settings(train.assign(binned_intensities=binned, mz_remainder=remainders)).reset_index(drop=True)
glycans = sorted(set().union(*combined["glycan_candidates"]))

print("Splitting train/test by run")
splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
train_idx, test_idx = next(splitter.split(combined, groups=combined["GlycoPost_ID"] + "/" + combined["filename"]))
train_df = combined.iloc[train_idx].reset_index(drop=True)
test_df = combined.iloc[test_idx].reset_index(drop=True)

train_df = downcast_numeric(train_df)
test_df = downcast_numeric(test_df)

print("Writing intermediate dataframes")
output_dir.mkdir(parents=True, exist_ok=True)
train_df.to_pickle(output_dir / "train_second.pkl")
test_df.to_pickle(output_dir / "test_second.pkl")

print("Serialising tuples for CandyCrunch")
X_train = tupleify(train_df, FEATURE_COLUMNS)
X_test = tupleify(test_df, FEATURE_COLUMNS)
y_train = train_df["glycan"].tolist()
y_test = test_df["glycan"].tolist()

with open(output_dir / "X_train.pkl", "wb") as fh:
    pickle.dump(X_train, fh)
with open(output_dir / "X_test.pkl", "wb") as fh:
    pickle.dump(X_test, fh)
with open(output_dir / "y_train.pkl", "wb") as fh:
    pickle.dump(y_train, fh)
with open(output_dir / "y_test.pkl", "wb") as fh:
    pickle.dump(y_test, fh)
with open(output_dir / "glycans.pkl", "wb") as fh:
    pickle.dump(glycans, fh)
with open(output_dir / "cand_mask_train.pkl", "wb") as fh:
    pickle.dump(candidate_mask(train_df, glycans), fh)
with open(output_dir / "cand_mask_test.pkl", "wb") as fh:
    pickle.dump(candidate_mask(test_df, glycans), fh)

print("Serializing pretraining inputs")
# One input per candidate composition of each unlabeled spectrum, as at inference; the spectrum stays sparse, as dense arrays of every unlabeled input
# exceed memory (densify as for the labeled inputs above)
(output_dir / "pretrain").mkdir(exist_ok=True)
for path in sorted((cc_training_dir / "pretrain").glob("*.pkl")):
    if path.stem in exclude_projects:
        continue
    pretrain = pd.read_pickle(path).explode(["compositional_vectors", "glycan_types"])
    pretrain = _encode_settings(pretrain.rename(columns={"compositional_vectors": "compositional_vector", "glycan_types": "glycan_type"}))
    with open(output_dir / "pretrain" / f"X_{path.stem}.pkl", "wb") as fh:
        pickle.dump(tupleify(pretrain, PRETRAIN_COLUMNS), fh)

print(f"Saved processed datasets to {output_dir.resolve()}")

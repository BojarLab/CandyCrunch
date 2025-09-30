"""
Helper script to convert a condensed spectra table (e.g. full_dataset.xlsx) and
its metadata into CandyCrunch-ready training pickles.  Adjust the paths in the
configuration block before running.
"""

import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from candycrunch.prediction import bin_intensities

full_dataset_path = Path("full_dataset.xlsx")
metadata_path = Path("training/file_checklist_template.csv")
legacy_train_path = None               # e.g. Path("prepared_datasets/train_legacy.pkl")
legacy_test_path = None                # e.g. Path("prepared_datasets/test_legacy.pkl")
output_dir = Path("prepared_datasets")
test_size = 0.15
random_state = 42

MODE_MAP = {"negative": 0, "positive": 1}
LC_MAP = {"PGC": 0, "C18": 1}
MOD_MAP = {"reduced": 0, "permethylated": 1}
TRAP_MAP = {"linear": 0, "orbitrap": 1, "amazon": 2}

FEATURE_COLUMNS = [
    "binned_intensities",
    "mz_remainder",
    "reducing_mass",
    "glycan_type",
    "RT",
    "mode",
    "lc",
    "modification",
    "trap",
]



def safe_peak_parse(value):
    if isinstance(value, dict):
        return {float(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, dict):
            return {float(k): float(v) for k, v in parsed.items()}
    return None

def normalise_spectrum(spec):
    total = float(sum(spec.values()))
    if total <= 0:
        return None
    return {mz: intensity / total for mz, intensity in spec.items()}

def process_peaks(df, min_mz=39.714, max_mz=3000.0, bin_num=2048):
    frames = np.linspace(min_mz, max_mz, bin_num)
    parsed = df["peak_d"].map(safe_peak_parse)
    parsed = parsed.map(lambda spec: None if spec is None else normalise_spectrum(spec))
    keep_mask = parsed.notnull()
    df = df.loc[keep_mask].copy()
    parsed = parsed.loc[keep_mask]
    binned, remainders = zip(*(bin_intensities(spec, frames) for spec in parsed))
    df["binned_intensities"] = [np.asarray(vec, dtype=np.float32) for vec in binned]
    df["mz_remainder"] = [np.asarray(vec, dtype=np.float32) for vec in remainders]
    return df

def process_retention_times(df):
    df = df.copy()
    df["RT"] = df["RT"].fillna(15)
    df = df[df["RT"] > 2]
    df["RT"] = df.groupby("filename")["RT"].transform(lambda rt: rt / max(rt.max(), 30.0))
    return df

def infer_glycan_types(df):
    def classify(g):
        if g.endswith(("GalNAc", "GalNAc6S", "GalNAcOS", "Fuc", "Man", "Gal")):
            return 0
        if "GlcNAc(b1-4)GlcNAc" in g:
            return 1
        if g.endswith(("Glc", "GlcOS", "GlcNAc", "Ins")):
            return 2
        return 3
    df = df.copy()
    df["glycan_type"] = df["glycan"].map(classify)
    return df

def attach_metadata(df, checklist):
    meta = checklist.copy()
    meta.columns = [col.strip() for col in meta.columns]
    meta = meta.set_index("GlycoPOST_ID")
    meta.index = meta.index.astype(str).str.strip().str.lower()

    def build_dict(column):
        if column not in meta.columns:
            return {}
        series = meta[column].fillna("").astype(str).str.lower().str.strip()
        return series.to_dict()

    mode_dict = build_dict("mode")
    lc_dict = build_dict("LC_type")
    mod_dict = build_dict("modification")
    trap_dict = build_dict("trap")

    def lookup(mapping, value, fallback):
        if not value:
            return fallback
        return mapping.get(value.lower(), fallback)

    def map_series(ids, source_dict, mapping, fallback):
        def mapper(gid):
            if gid in source_dict:
                return lookup(mapping, source_dict[gid], fallback)
            return fallback
        lowered = ids.fillna("").astype(str).str.strip().str.lower()
        return lowered.map(mapper).astype(int)

    df = df.copy()
    ids = df["GlycoPost_ID"].astype(str).str.strip()
    df["mode"] = map_series(ids, mode_dict, MODE_MAP, 2)
    df["lc"] = map_series(ids, lc_dict, LC_MAP, 2)
    df["modification"] = map_series(ids, mod_dict, MOD_MAP, 2)
    df["trap"] = map_series(ids, trap_dict, TRAP_MAP, 3)
    return df

def process_full_dataset(full_df, checklist):
    df = process_retention_times(full_df)
    df = infer_glycan_types(df)
    df = process_peaks(df)
    df = attach_metadata(df, checklist)
    return df

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


print("Loading condensed spectra")
full_df = pd.read_excel(full_dataset_path)
meta_df = pd.read_csv(metadata_path)
processed = process_full_dataset(full_df, meta_df)

frames = [processed]
if legacy_train_path:
    frames.append(pd.read_pickle(legacy_train_path))
if legacy_test_path:
    frames.append(pd.read_pickle(legacy_test_path))
combined = pd.concat(frames, ignore_index=True)

glycans = sorted(set(combined["glycan"]))
glycan_to_idx = {g: i for i, g in enumerate(glycans)}
combined["glycan"] = combined["glycan"].map(glycan_to_idx)

print("Splitting train/test by filename")
splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
train_idx, test_idx = next(splitter.split(combined, groups=combined["filename"]))
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

print(f"Saved processed datasets to {output_dir.resolve()}")

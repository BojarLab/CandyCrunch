"""Evaluation harness: every test set of test_CandyCrunch.TEST_DICTS plus held-out runs (by default the GlycoGauntlet private files), in parallel, with
content-addressed caches for the expensive pure steps and a trace of where each missed ground-truth peak was lost.

Usage (from the repo root):
    python tests/harness.py --out harness_results/base.json
    python tests/harness.py --out harness_results/new.json --compare harness_results/base.json
    python tests/harness.py --sets GPST000029,milk --trace_rows
Options: --format xlsx|mzml (mzml: .mzML/.mzXML/.raw/.d where a dataset has one, else its .xlsx), --n_jobs (parallel files; default: half the CPUs),
--private_spectra / --private_gt (folders with <run>.xlsx and <run>_solution.csv; default: GlycoGauntlet next to this repo), --cache (default:
~/.cache/candycrunch_harness), --no_cache, --stage_cache_gb (default 20), --files_per_process (default 4), --seeds 0,1,2 (only needed for code from before
content-seeded test-time augmentation).

Tune on DEV (the test sets); HELD-OUT (the private files) is only for confirming that a change generalizes. Each run is computed once: since test-time
augmentation is seeded per input, a code change only alters the inputs it touches, so seeds no longer need averaging.

Caches (keyed by the content of what goes in, plus the source of the code that computes it, so a stale entry is never hit): the loaded spectra of each
file (loader source), each CandyCrumbs and supporting_ions call (analysis.py and glycowork), the model's top-k for each model input (model.py,
weights, vocabulary), the outputs of condense_dataframe and assign_annotation_scores_pooled (the candycrunch code they reach, without comments, plus
candycrunch's module-level code and data, glycowork and the library versions), and the metric's and trace's glycowork calls. A change in prediction.py's
downstream steps therefore reruns in a fraction of the time; a change in analysis.py, the model or a cached step reruns those calls once. Caching the
steps after the model was measured and is not worth it: they take seconds, while hashing and storing their inputs and outputs took as long (2026-10-17;
42 files on 11 processes, warm: 286 s before, 150 s with the cached steps, chunked processes and glycowork memo, outputs identical).

Trace (stage where each ground-truth peak was lost, in pipeline order): no_ms2 (no MS2 spectrum within 0.5 m/z and 1 min), no_cluster, composition_not_candidate,
model_no_composition (no structure of the composition in the model's raw top 25), below_pred_thresh (in the top 25, but no prediction survives
pred_thresh / the class veto), domain_filter, other_composition_chosen (the spectrum is reported with another composition), crumbs_gate, deduplicated,
finalise (ppm / RT / merge steps), matched_elsewhere (a reported row is near, but the one-to-one matching gave it to another peak or it lies just outside the
metric's tolerance). Wrong structures: wrong_composition, not_in_vocabulary, not_in_top25, below_thresh (in the top 25, not among the row's predictions),
ranked_lower (among the row's predictions, not first); a hedged ground truth counts as present wherever a more specific version of it is. False positives: fp_same_composition (the GT has the composition at another RT), fp_new_composition.
"""
import argparse
import hashlib
import inspect
import json
import os
import pathlib
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
TRACE_ORDER = ['no_ms2', 'no_cluster', 'composition_not_candidate', 'model_no_composition', 'below_pred_thresh', 'domain_filter', 'other_composition_chosen',
               'crumbs_gate', 'deduplicated', 'finalise', 'matched_elsewhere']
WRONG_ORDER = ['wrong_composition', 'not_in_vocabulary', 'not_in_top25', 'below_thresh', 'ranked_lower']
FP_ORDER = ['fp_same_composition', 'fp_new_composition']


def freeze(x):
    # A deterministic, hashable form of a call's arguments (set order follows PYTHONHASHSEED for strings)
    if isinstance(x, (set, frozenset)):
        return ('__set__', sorted((freeze(v) for v in x), key = repr))
    if isinstance(x, dict):
        return ('__dict__', tuple((freeze(k), freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(freeze(v) for v in x))
    if isinstance(x, np.ndarray):
        return ('__array__', x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, np.generic):
        return x.item()
    return x


def digest(*parts):
    return hashlib.sha1(pickle.dumps(freeze(parts), protocol = 4)).hexdigest()


class Cache:
    # One pickle per (file, kind), loaded once per run and written back if anything was added; values are stored pickled, so every hit is a fresh copy

    def __init__(self, folder, kind, enabled):
        self.path = pathlib.Path(folder) / f'{kind}.pkl'
        self.enabled = enabled
        self.data = {}
        self.new = 0
        self.hits = 0
        self.misses = 0
        if enabled and self.path.exists():
            try:
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}

    def get(self, key):
        if not self.enabled or key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        return pickle.loads(self.data[key])

    def put(self, key, value):
        if self.enabled:
            self.data[key] = pickle.dumps(value, protocol = 4)
            self.new += 1

    def save(self):
        if self.enabled and self.new:
            self.path.parent.mkdir(parents = True, exist_ok = True)
            tmp = self.path.with_suffix('.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(self.data, f, protocol = 4)
            os.replace(tmp, self.path)


def glycowork_fingerprint():
    import glycowork
    root = pathlib.Path(glycowork.__file__).parent
    return digest(glycowork.__version__, [(str(p.relative_to(root)), p.stat().st_size, int(p.stat().st_mtime)) for p in sorted(root.rglob('*.py'))])


class Hasher:
    def __init__(self):
        self.sha = hashlib.sha1()

    def write(self, b):
        self.sha.update(b)


def content_digest(*parts):
    # A pickle without memo, so equal values give equal digests whatever objects they share (DataFrames with dict cells, attrs included)
    hasher = Hasher()
    pickler = pickle.Pickler(hasher, protocol = 5)
    pickler.fast = True
    pickler.dump(parts)
    return hasher.sha.hexdigest()


def code_names(code):
    return set(code.co_names).union(*[code_names(c) for c in code.co_consts if inspect.iscode(c)])


def normalized_source(obj):
    # Comments and formatting do not count
    import ast
    import textwrap
    return ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(obj))))


def source_closure(fn, orig):
    # The candycrunch functions and classes a function reaches through the names its code uses (the harness's wrappers stand for the functions they wrap)
    seen, sources, todo = set(), [], [fn]
    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        sources.append(f'{f.__module__}.{f.__qualname__}\n{normalized_source(f)}')
        codes = [f.__code__] if inspect.isfunction(f) else [m.__code__ for m in vars(f).values() if inspect.isfunction(m)]
        scope = f.__globals__ if inspect.isfunction(f) else vars(sys.modules[f.__module__])
        for name in set().union(*map(code_names, codes)):
            v = scope.get(name)
            if name in orig and v is not None and not (getattr(v, '__module__', None) or '').startswith('candycrunch'):
                v = orig[name]
            v = inspect.unwrap(v) if callable(v) else v
            if (inspect.isfunction(v) or inspect.isclass(v)) and (v.__module__ or '').startswith('candycrunch'):
                todo.append(v)
    return sorted(sources)


def environment_fingerprint():
    # Everything a pipeline step depends on besides its own code: the module-level code of candycrunch (constants), the package data (vocabulary, weights,
    # mappings), glycowork's code and data, the library versions and Python (whose float sum differs from 3.12)
    import ast
    import candycrunch
    import glycowork
    import networkx
    import scipy
    import torch
    mods = ['candycrunch.prediction', 'candycrunch.analysis', 'candycrunch.utils', 'candycrunch.model']
    top = [ast.unparse(ast.Module(body = [n for n in ast.parse(inspect.getsource(sys.modules[m])).body
                                          if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores = [])) for m in mods]
    files = lambda root, keep: [(str(p.relative_to(root)), p.stat().st_size, int(p.stat().st_mtime)) for p in sorted(root.rglob('*'))
                                if p.is_file() and '__pycache__' not in p.parts and keep(p)]
    return digest(top, files(pathlib.Path(candycrunch.__file__).parent, lambda p: p.suffix != '.py'), files(pathlib.Path(glycowork.__file__).parent, lambda p: True),
                  sys.version, np.__version__, pd.__version__, scipy.__version__, networkx.__version__, torch.__version__)


# Pipeline steps whose outputs are cached (keyed by their inputs and the code they reach): the slow steps before the model, which the reducing end and glycan
# class checks repeat (condense_dataframe ran 5 times at ~40 s on GPST000403); deisotoped (a cache keyed by object ids) and n_jobs do not change outputs
STAGES = ['condense_dataframe', 'assign_annotation_scores_pooled']
STAGE_IGNORED = {'deisotoped', 'n_jobs'}


def restore(obj, new):
    # Puts an argument's state after the call into the caller's object, as the call changed it in place; False if that cannot be done
    if isinstance(obj, (pd.DataFrame, pd.Series)) and type(new) is type(obj):
        object.__setattr__(obj, '_mgr', new._mgr)
        object.__setattr__(obj, '_attrs', new._attrs)
        if getattr(obj, '_item_cache', None) is not None:
            obj._item_cache.clear()
    elif isinstance(obj, dict) and isinstance(new, dict):
        obj.clear()
        obj.update(new)
    elif isinstance(obj, (list, set)) and type(new) is type(obj):
        obj.clear()
        obj.update(new) if isinstance(obj, set) else obj.extend(new)
    elif isinstance(obj, np.ndarray) and isinstance(new, np.ndarray) and new.shape == obj.shape and new.dtype == obj.dtype:
        obj[...] = new
    else:
        return False
    return True


class StageCache:
    # One pickle per (step, inputs, code), so outputs are only read when hit and old ones can be pruned by age. Arguments a step changes in place are
    # stored with its output and put back into the caller's objects on a hit; a step changing anything it cannot put back is not cached

    def __init__(self, folder, enabled):
        self.folder = pathlib.Path(folder) / 'stage'
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.mutating = set()

    def wrap(self, name, fn, env, orig, codes):
        sig = inspect.signature(fn)
        def wrapped(*args, **kwargs):
            if not self.enabled or name in self.mutating:
                return fn(*args, **kwargs)
            if name not in codes:
                codes[name] = digest(env, source_closure(fn, orig))
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = {k: v for k, v in bound.arguments.items() if k not in STAGE_IGNORED}
            try:
                before = {k: content_digest(v) for k, v in inputs.items()}
            except Exception:
                return fn(*args, **kwargs)
            key = digest(name, codes[name], before)
            path = self.folder / f'{name}_{key}.pkl'
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        out, changed = pickle.load(f)
                    for k, v in changed.items():
                        restore(inputs[k], v)
                        # a step returning the argument it changed returns the caller's object
                        out = inputs[k] if out is v else out
                    os.utime(path)
                    self.hits += 1
                    return out
                except Exception:
                    pass
            self.misses += 1
            out = fn(*args, **kwargs)
            changed = {k: v for k, v in inputs.items() if content_digest(v) != before[k]}
            if not all(isinstance(v, (pd.DataFrame, pd.Series, dict, list, set, np.ndarray)) for v in changed.values()):
                self.mutating.add(name)
                return out
            try:
                self.folder.mkdir(parents = True, exist_ok = True)
                tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                with open(tmp, 'wb') as f:
                    pickle.dump((out, changed), f, protocol = 5)
                os.replace(tmp, path)
            except Exception:
                pass
            return out
        wrapped.__name__ = name
        return wrapped

    def save(self):
        pass


def prune_stage_cache(cache_dir, max_gb):
    # Least recently used step outputs go first (a hit touches its file)
    files = sorted(((p.stat().st_mtime, p.stat().st_size, p) for p in pathlib.Path(cache_dir).glob('*/stage/*.pkl')), reverse = True)
    total = 0
    for _, size, p in files:
        total += size
        if total > max_gb * 1e9:
            p.unlink(missing_ok = True)


def install_caches(cp, cache_dir, enabled, captured, model = None, vocab = None):
    # Wraps the module globals wrap_inference looks up at call time: memoized loading, CandyCrumbs, supporting_ions and model top-k, and captures of the
    # intermediate tables of the first (main) call of each stage for the trace
    import torch
    import candycrunch.analysis as ca
    import candycrunch.model as cm
    import candycrunch.utils as cu
    gw = glycowork_fingerprint()
    crumbs_src = digest(pathlib.Path(ca.__file__).read_text(), gw)
    model, vocab = (cp.candycrunch if model is None else model), (cp.glycans if vocab is None else vocab)
    model_src = digest(pathlib.Path(cm.__file__).read_text(), [p.detach().numpy() for p in model.state_dict().values()], list(vocab))
    caches = {k: Cache(cache_dir, k, enabled) for k in ['load', 'crumbs', 'topk']}
    caches['stage'] = StageCache(cache_dir, enabled)
    # The unwrapped functions, kept on the module, so installing again (the next file in one process) wraps the originals and not the last wrappers
    if not hasattr(cp, '_harness_orig'):
        cp._harness_orig = {n: getattr(cp, n) for n in ['load_spectra_filepath', 'CandyCrumbs', 'supporting_ions', 'get_topk', 'condense_dataframe',
                                                        'assign_candidate_structures', 'assign_annotation_scores_pooled', 'process_for_inference', 'domain_filter',
                                                        'deduplicate_predictions', 'finalise_predictions']}
        cp._harness_env = environment_fingerprint()
    orig = cp._harness_orig
    inner = {n: caches['stage'].wrap(n, orig[n], cp._harness_env, orig, cp.__dict__.setdefault('_harness_codes', {})) for n in STAGES}
    for n in STAGES:
        setattr(cp, n, inner[n])
    loader_src = digest(pathlib.Path(cu.__file__).read_text(), [inspect.getsource(getattr(cp, n)) for n in ['load_spectra_filepath', 'process_mzML_stack', 'process_mzXML_stack',
                        'process_raw_stack', 'process_tdf_stack', 'process_yep_stack', 'refine_precursor_mz', 'refine_tof_precursors'] if n != 'load_spectra_filepath' and hasattr(cp, n)]
                        + [inspect.getsource(orig['load_spectra_filepath'])], gw)
    def first(name, value):
        if name not in captured:
            captured[name] = value
    def load_spectra_filepath(spectra_filepath, extract_ms1 = False):
        p = pathlib.Path(spectra_filepath)
        stat = [(str(q.relative_to(p)), q.stat().st_size, int(q.stat().st_mtime)) for q in sorted(p.rglob('*')) if q.is_file()] if p.is_dir() else (p.stat().st_size, int(p.stat().st_mtime))
        key = digest(p.name, stat, extract_ms1, loader_src)
        hit = caches['load'].get(key)
        if hit is None:
            df = orig['load_spectra_filepath'](spectra_filepath, extract_ms1 = extract_ms1)
            caches['load'].put(key, (df, dict(df.attrs)))
        else:
            df, attrs = hit
            df.attrs.update(attrs)
        first('loaded', df[[c for c in ['m/z', 'reducing_mass', 'RT', 'precursor_charge'] if c in df.columns]].copy())
        return df
    def memo(name):
        def wrapped(*args, **kwargs):
            key = digest(name, args, kwargs, crumbs_src)
            hit = caches['crumbs'].get(key)
            if hit is None:
                hit = orig[name](*args, **kwargs)
                caches['crumbs'].put(key, hit)
            return hit
        wrapped.__name__ = name
        return wrapped
    def get_topk(dataloader, run_model, glycans, k = 25, temp = False, temperature = cp.temperature):
        dataset = dataloader.dataset
        parts = dataset.datasets if isinstance(dataset, torch.utils.data.ConcatDataset) else [dataset]
        if run_model is not model or glycans is not vocab or any(getattr(d, 'transform_mz', None) or getattr(d, 'transform_rt', None) for d in parts):
            # Augmentation drawn at access time (code from before content-seeded augmentation) or a model the cache was not keyed on: no cache
            out = orig['get_topk'](dataloader, run_model, glycans, k = k, temp = temp, temperature = temperature)
            first('topk', out)
            return out
        settings = (k, temp, float(temperature), model_src)
        keys = [digest(b''.join(t.numpy().tobytes() for t in dataset[i][:9]), settings) for i in range(len(dataset))]
        found = [caches['topk'].get(key) for key in keys]
        missing = [i for i, f in enumerate(found) if f is None]
        if missing:
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, missing), batch_size = dataloader.batch_size, shuffle = False)
            preds, conf = orig['get_topk'](loader, run_model, glycans, k = k, temp = temp, temperature = temperature)
            for i, p, c in zip(missing, preds, conf):
                found[i] = (p, c)
                caches['topk'].put(keys[i], (p, c))
        out = ([f[0] for f in found], [f[1] for f in found])
        first('topk', out)
        return out
    def capture(name, out_index = None, copy_input = False):
        def wrapped(*args, **kwargs):
            # the input before the call, as some steps change it in place
            snapshot = args[0].copy() if copy_input or name == 'domain_filter' else None
            out = inner.get(name, orig[name])(*args, **kwargs)
            first(name, snapshot if copy_input else (out[out_index] if out_index is not None else out).copy())
            if name == 'domain_filter':
                first('domain_filter_in', snapshot)
            return out
        wrapped.__name__ = name
        return wrapped
    cp.load_spectra_filepath = load_spectra_filepath
    cp.CandyCrumbs = memo('CandyCrumbs')
    cp.supporting_ions = memo('supporting_ions')
    cp.get_topk = get_topk
    cp.condense_dataframe = capture('condense_dataframe')
    cp.assign_candidate_structures = capture('assign_candidate_structures')
    cp.assign_annotation_scores_pooled = capture('assign_annotation_scores_pooled')
    cp.process_for_inference = capture('process_for_inference', out_index = 1)
    cp.domain_filter = capture('domain_filter')
    cp.deduplicate_predictions = capture('deduplicate_predictions', copy_input = True)
    cp.finalise_predictions = capture('finalise_predictions', copy_input = True)
    return caches


class GlycoworkMemo:
    # The metric's and the trace's glycowork calls (canonicalize_iupac of the whole vocabulary, glycan similarities), shared by all files and kept on disk
    def __init__(self, cache_dir, enabled):
        self.path = pathlib.Path(cache_dir) / '_glycowork' / f'{glycowork_fingerprint()}.pkl'
        self.enabled = enabled
        self.data, self.new = {}, {}
        if enabled and self.path.exists():
            try:
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)
            except Exception:
                pass

    def wrap(self, fn):
        def wrapped(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            try:
                hash(key)
            except TypeError:
                return fn(*args, **kwargs)
            if key not in self.data:
                self.data[key] = self.new[key] = fn(*args, **kwargs)
            return self.data[key]
        return wrapped if self.enabled else fn

    def save(self):
        # merged with what other processes saved meanwhile
        if self.new:
            try:
                with open(self.path, 'rb') as f:
                    self.data.update(pickle.load(f))
            except Exception:
                pass
            self.data.update(self.new)
            self.path.parent.mkdir(parents = True, exist_ok = True)
            tmp = self.path.with_suffix(f'.{os.getpid()}.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(self.data, f, protocol = 5)
            os.replace(tmp, self.path)
            self.new = {}


def z1(mz, z, sign = -1):
    # Singly charged equivalent, as the metric compares masses (an unknown charge counts as singly charged in the run's ion mode)
    # Singly charged equivalent, as the metric compares masses (an unknown charge counts as singly charged in the run's ion mode)
    from candycrunch.prediction import PROTON_MASS
    z = np.where(np.isnan(z) | (z == 0), sign, z)
    return mz * np.abs(z) - (z - np.sign(z)) * PROTON_MASS


def near(mz, rt, z, gt_mz, gt_rt, gt_z, mass_tol = 0.5, rt_tol = 1.0, sign = -1):
    # Rows within the metric's tolerances of one ground-truth peak, as raw m/z or as singly charged equivalents
    mz, rt, z = np.asarray(mz, dtype = float), np.asarray(rt, dtype = float), np.asarray(z, dtype = float)
    close = (np.abs(mz - gt_mz) <= mass_tol) | (np.abs(z1(mz, z, sign) - z1(np.array([gt_mz]), np.array([gt_z], dtype = float), sign)[0]) <= mass_tol)
    return close & (np.abs(rt - gt_rt) <= rt_tol)


def trace_file(captured, preds, gt, rt_col, sign = -1, vocabulary = None):
    # Where each ground-truth peak was lost, what is wrong about each wrong structure, and what each false positive is
    # the test module's glycowork functions, which run_file memoizes
    from tests.test_CandyCrunch import match_spectra, MASS_TOLERANCE, RT_TOLERANCE, compare_glycans, canonicalize_iupac
    from candycrunch.prediction import get_comp, glycans
    vocabulary = glycans if vocabulary is None else vocabulary
    comp_key = lambda c: tuple(sorted((k, v) for k, v in c.items() if v)) if isinstance(c, dict) else None
    gt_comp = lambda g: comp_key(get_comp(g)) if isinstance(g, str) else None
    vocab = {canonicalize_iupac(g) for g in vocabulary}
    gt = gt.reset_index(drop = True)
    gt_z = gt['charge'].values.astype(float) if 'charge' in gt.columns else np.full(len(gt), float(sign))
    loaded = captured.get('loaded', pd.DataFrame(columns = ['m/z', 'RT']))
    lmz = loaded['m/z' if 'm/z' in loaded.columns else 'reducing_mass'].values
    lz = loaded['precursor_charge'].values.astype(float) if 'precursor_charge' in loaded.columns else np.full(len(loaded), np.nan)
    cond = captured.get('condense_dataframe', pd.DataFrame(columns = ['m/z', 'RT']))
    cmz = cond['m/z'].values if 'm/z' in cond.columns else cond.index.values
    cand = captured.get('assign_annotation_scores_pooled', captured.get('assign_candidate_structures', pd.DataFrame(columns = ['m/z', 'RT', 'spec_id', 'composition'])))
    cand_mz = cand['m/z'].values if 'm/z' in cand.columns else cand.index.values
    cand_z = cand['charge'].values.astype(float) * sign if 'charge' in cand.columns else np.full(len(cand), float(sign))
    # The model's raw top 25 per (spectrum, composition), merged over the test-time augmentation copies like wrap_inference does
    raw = {}
    pfi, topk = captured.get('process_for_inference'), captured.get('topk')
    if pfi is not None and topk is not None:
        first_rows = pfi.drop_duplicates('input_id')
        tta = first_rows['tta'].values
        order = np.concatenate([np.flatnonzero(tta), np.flatnonzero(~tta)])
        sizes = np.where(tta[order], 5, 1)
        per_input = {}
        for i, start, n in zip(order, np.cumsum(sizes) - sizes, sizes):
            best = {}
            for p, c in zip(topk[0][start:start + n], topk[1][start:start + n]):
                for g, x in zip(p, c):
                    best[g] = max(best.get(g, 0), x)
            per_input[first_rows['input_id'].iloc[i]] = best
        for spec, comp, iid in zip(pfi['spec_id'], pfi['composition'], pfi['input_id']):
            raw[(spec, comp_key(comp))] = per_input.get(iid, {})
    df_in = captured.get('domain_filter_in')
    df_after = captured.get('domain_filter')
    dedup_in = captured.get('deduplicate_predictions')
    fin_in = captured.get('finalise_predictions')
    out = preds.reset_index()
    out = out[out['top1_pred'].notna()].reset_index(drop = True)
    out_z = out['charge'].values.astype(float) if 'charge' in out.columns else np.full(len(out), float(sign))
    # The metric's one-to-one matching
    pairs = out[['m/z', 'RT']].round(2).values
    pairs_c = np.column_stack([z1(out['m/z'].values, out_z, sign), out['RT'].values]).round(2)
    gt_pairs = gt[['Mass', rt_col]].round(2).values
    gt_pairs_c = np.column_stack([z1(gt['Mass'].values, gt_z, sign), gt[rt_col].values]).round(2)
    matched = dict(match_spectra(gt_pairs, pairs, mass_threshold = MASS_TOLERANCE, rt_threshold = RT_TOLERANCE, array1_alt = gt_pairs_c, array2_alt = pairs_c))
    rows = []
    def row_preds(df, gmz, grt, gz, comp):
        # Rows of an intermediate table near the peak, and whether one of them has a prediction of the composition (any composition for open peaks)
        if df is None or not len(df):
            return False
        d = df.reset_index() if 'm/z' not in df.columns else df
        hit = near(d['m/z'].values, d['RT'].values, (d['charge'].values.astype(float) if 'charge' in d.columns else np.full(len(d), float(sign))), gmz, grt, gz, sign = sign)
        for p in d.loc[hit, 'predictions'] if 'predictions' in d.columns else []:
            if isinstance(p, list) and p and (comp is None or any(gt_comp(g) == comp for g, _ in p)):
                return True
        return False
    for i, g in gt.iterrows():
        gmz, grt, gzz, glycan = g['Mass'], g[rt_col], gt_z[i], g['glycan'] if isinstance(g['glycan'], str) else None
        comp = gt_comp(glycan)
        if i in matched:
            p = out.iloc[matched[i]]['top1_pred']
            if glycan is None or compare_glycans(glycan, p):
                continue
            # A wrong structure: which step put the true one out of reach
            if gt_comp(p) != comp:
                stage = 'wrong_composition'
            elif canonicalize_iupac(glycan) not in vocab and not any(compare_glycans(glycan, v, subsumes = True) for v in vocabulary if gt_comp(v) == comp):
                stage = 'not_in_vocabulary'
            else:
                r = out.iloc[matched[i]]
                specs = [k for k in raw if k[1] == comp]
                hits = [s for s in specs if s[0] in set(cand.loc[near(cand_mz, cand['RT'].values, cand_z, r['m/z'], r['RT'], out_z[matched[i]], sign = sign), 'spec_id'])] if 'spec_id' in cand.columns else []
                top = {x for s in hits for x in raw[s]}
                # a hedged ground truth is reachable through any more specific version of it
                if not any(compare_glycans(glycan, x, subsumes = True) for x in top):
                    stage = 'not_in_top25'
                elif not any(compare_glycans(glycan, x, subsumes = True) for x, _ in r['predictions']) if isinstance(r.get('predictions'), list) else True:
                    stage = 'below_thresh'
                else:
                    stage = 'ranked_lower'
            rows.append(dict(kind = 'wrong', stage = stage, mass = gmz, rt = grt, gt = glycan, pred = p))
            continue
        # A missed peak: the first step after which nothing near it is left
        if not near(lmz, loaded['RT'].values, lz, gmz, grt, gzz, sign = sign).any():
            stage = 'no_ms2'
        elif not near(cmz, cond['RT'].values, np.full(len(cond), np.nan), gmz, grt, gzz, sign = sign).any():
            stage = 'no_cluster'
        else:
            hit = near(cand_mz, cand['RT'].values, cand_z, gmz, grt, gzz, sign = sign) if len(cand) else np.zeros(0, dtype = bool)
            comps = {comp_key(c) for c in cand.loc[hit, 'composition']} if len(cand) else set()
            specs = set(cand.loc[hit, 'spec_id']) if len(cand) else set()
            if comp is not None and comp not in comps:
                stage = 'composition_not_candidate'
            elif raw and not any(gt_comp(x) == comp if comp is not None else True for s in specs for k, top in raw.items() if k[0] == s and (comp is None or k[1] == comp) for x in top):
                stage = 'model_no_composition'
            elif not row_preds(df_in, gmz, grt, gzz, comp):
                stage = 'below_pred_thresh'
            elif not row_preds(df_after, gmz, grt, gzz, comp):
                stage = 'domain_filter'
            elif not row_preds(dedup_in, gmz, grt, gzz, comp):
                d = df_after.reset_index() if df_after is not None and 'm/z' not in df_after.columns else df_after
                spec_rows = d[d['spec_id'].isin(specs)] if d is not None and 'spec_id' in d.columns else pd.DataFrame()
                stage = 'other_composition_chosen' if len(spec_rows) and any(
                    comp_key(c) != comp and isinstance(p, list) and p for c, p in zip(spec_rows['composition'], spec_rows['predictions'])) else 'crumbs_gate'
            elif not row_preds(fin_in, gmz, grt, gzz, comp):
                stage = 'deduplicated'
            elif not near(out['m/z'].values, out['RT'].values, out_z, gmz, grt, gzz, mass_tol = 1.0, rt_tol = 2.0, sign = sign).any():
                stage = 'finalise'
            else:
                stage = 'matched_elsewhere'
        rows.append(dict(kind = 'missed', stage = stage, mass = gmz, rt = grt, gt = glycan, pred = None))
    gt_comps = {gt_comp(x) for x in gt['glycan'] if isinstance(x, str)}
    for j in set(range(len(out))) - set(matched.values()):
        p = out.iloc[j]
        rows.append(dict(kind = 'fp', stage = 'fp_same_composition' if gt_comp(p['top1_pred']) in gt_comps else 'fp_new_composition', mass = p['m/z'], rt = p['RT'],
                         gt = None, pred = p['top1_pred']))
    return rows


def run_file(task):
    # One file in its own process: caches installed, the file predicted and scored, its trace computed
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import random
    import torch
    torch.set_num_threads(1)
    import candycrunch.prediction as cp
    import tests.test_CandyCrunch as tc
    from tests.test_CandyCrunch import evaluate_predictions, MASS_TOLERANCE, RT_TOLERANCE
    if not hasattr(tc, '_harness_memo'):
        tc._harness_memo = GlycoworkMemo(task['cache'], task['use_cache'])
        for n in ['canonicalize_iupac', 'compare_glycans', 'get_glycan_similarity']:
            setattr(tc, n, tc._harness_memo.wrap(getattr(tc, n)))
    out = []
    model, vocab = cp.candycrunch, cp.glycans
    if task['model']:
        # A fine-tuned checkpoint (the fine-tuning cell): weights and its vocabulary
        from candycrunch.model import CandyCrunch_CNN
        ckpt = torch.load(task['model'], map_location = 'cpu', weights_only = False)
        vocab = ckpt['glycans']
        model = CandyCrunch_CNN(2048, num_classes = len(vocab), input_precursor_dim = 12)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    for seed in task['seeds']:
        captured = {}
        caches = install_caches(cp, pathlib.Path(task['cache']) / task['label'].replace('/', '__'), task['use_cache'], captured, model = model, vocab = vocab)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        start = time.time()
        error = None
        try:
            preds = cp.wrap_inference(task['path'], **task['args'], supplement = True, experimental = True, **({'model': model, 'glycans': vocab} if task['model'] else {}))
        except Exception:
            import traceback
            error = traceback.format_exc()
            preds = pd.DataFrame(columns = ['top1_pred', 'charge', 'RT', 'predictions'], index = pd.Index([], name = 'm/z'))
        elapsed = time.time() - start
        if task['save_preds']:
            pathlib.Path(task['save_preds']).mkdir(parents = True, exist_ok = True)
            preds.to_pickle(pathlib.Path(task['save_preds']) / f"{task['label'].replace('/', '__')}_seed{seed}.pkl")
        for c in caches.values():
            c.save()
        gt = pd.read_csv(task['gt'])
        rt_col = 'RT' if 'RT' in gt.columns else task['run'] + '_RT'
        if task['run'] in gt.columns:
            gt = gt[gt[task['run']] > 0]
        f1, precision, recall, not_picked, wrong, tp, fp, fn, uneval = evaluate_predictions(preds.copy(), gt.copy(), rt_col, MASS_TOLERANCE, RT_TOLERANCE)
        try:
            trace = trace_file(captured, preds, gt, rt_col, sign = -1 if task['args'].get('max_charge', -3) < 0 else 1, vocabulary = vocab)
        except Exception:
            import traceback
            trace = [dict(kind = 'error', stage = traceback.format_exc(), mass = None, rt = None, gt = None, pred = None)]
        tc._harness_memo.save()
        out.append(dict(label = task['label'], set = task['set'], held_out = task['held_out'], seed = seed, f1 = float(f1), precision = float(precision),
                        recall = float(recall), tp = float(tp), fp = float(fp), fn = float(fn), not_picked = int(not_picked), wrong = int(wrong), time = elapsed,
                        cache = {k: (c.hits, c.misses) for k, c in caches.items()}, uncached = sorted(caches['stage'].mutating), trace = trace, error = error))
    return out


def run_files(tasks):
    # Several files in one process, so its imports (~20 s) are paid once
    return [r for t in tasks for r in run_file(t)]


def collect_tasks(args):
    from tests.test_CandyCrunch import TEST_DICTS, TEST_DATA_DIR, MS1_FORMATS
    tasks, names = [], Counter(d['name'] for d in TEST_DICTS)
    wanted = set(args.sets.split(',')) if args.sets else None
    for d in TEST_DICTS:
        name = d['name'] if names[d['name']] == 1 else f"{d['name']}_{d['args']['glycan_class']}"
        folder = pathlib.Path(TEST_DATA_DIR) / d.get('folder', d['name'])
        gt = folder / f"df_mz_{d['name']}.csv"
        if (wanted and name not in wanted and d['name'] not in wanted) or not gt.exists():
            continue
        files = [x for x in d.get('test_files', os.listdir(folder)) if 'df_mz' not in x and not x.startswith('.')]
        ms1 = [x for x in files if x.lower().endswith(MS1_FORMATS)]
        files = (ms1 or [x for x in files if x.lower().endswith('.xlsx')]) if args.format == 'mzml' else [x for x in files if x.lower().endswith(('.xlsx', '.mgf'))]
        for x in sorted(files):
            tasks.append(dict(label = f'{name}/{x}', set = name, held_out = False, path = str(folder / x), gt = str(gt), run = x.split('.')[0], args = d['args']))
    spectra, gts = pathlib.Path(args.private_spectra), pathlib.Path(args.private_gt)
    if not args.no_private and spectra.exists() and gts.exists():
        for x in sorted(spectra.glob('*.xlsx')):
            sol = gts / f'{x.stem}_solution.csv'
            name = f'GG_{x.stem}'
            if not sol.exists() or (wanted and name not in wanted and 'held_out' not in wanted):
                continue
            tasks.append(dict(label = f'{name}/{x.name}', set = name, held_out = True, path = str(x), gt = str(sol), run = x.stem,
                              args = {'glycan_class': 'N' if '_N_' in x.name or x.stem.endswith('_N') else 'O'}))
    return tasks


def solution_to_gt(task, tmp):
    # GlycoGauntlet solutions (m/z, RT, charge, top1_pred) in the ground-truth format of the test sets
    sol = pd.read_csv(task['gt'])
    if 'Mass' not in sol.columns:
        sol = sol.rename(columns = {'m/z': 'Mass', 'top1_pred': 'glycan'})
        sol[task['run']] = 1
        path = pathlib.Path(tmp) / f"{task['run']}_gt.csv"
        sol.to_csv(path, index = False)
        task['gt'] = str(path)
    return task


def summarize(results, prev = None):
    by_set = defaultdict(list)
    for r in results:
        by_set[r['set']].append(r)
    lines = [f"{'set':<34}{'F1':>7}{'P':>7}{'R':>7}{'TP':>7}{'FP':>6}{'FN':>7}{'time':>7}" + ('   dF1' if prev else '')]
    means = {}
    for name in sorted(by_set, key = lambda n: (by_set[n][0]['held_out'], n)):
        rs = by_set[name]
        means[name] = np.mean([r['f1'] for r in rs])
        delta = f"{means[name] - prev['sets'][name]:+8.4f}" if prev and name in prev.get('sets', {}) else ''
        lines.append(f"{name:<34}{means[name]:7.4f}{np.mean([r['precision'] for r in rs]):7.3f}{np.mean([r['recall'] for r in rs]):7.3f}"
                     f"{np.mean([r['tp'] for r in rs]):7.1f}{np.mean([r['fp'] for r in rs]):6.1f}{np.mean([r['fn'] for r in rs]):7.1f}{sum(r['time'] for r in rs):7.0f}{delta}"
                     + ('  ERROR' if any(r['error'] for r in rs) else ''))
    dev = [m for n, m in means.items() if not by_set[n][0]['held_out']]
    held = [m for n, m in means.items() if by_set[n][0]['held_out']]
    summary = dict(dev = float(np.mean(dev)) if dev else None, held_out = float(np.mean(held)) if held else None, sets = {k: float(v) for k, v in means.items()})
    for key, label in [('dev', 'DEV mean (tune on this)'), ('held_out', 'HELD-OUT mean (confirm only)')]:
        if summary[key] is not None:
            # against the earlier run's mean over the same sets (a run of a few sets compared with a full one)
            common = [n for n in means if by_set[n][0]['held_out'] == (key == 'held_out') and n in (prev or {}).get('sets', {})]
            delta = f"  ({np.mean([means[n] - prev['sets'][n] for n in common]):+.4f} over {len(common)} sets)" if common else ''
            lines.append(f"{label:<34}{summary[key]:7.4f}{delta}")
    # Trace: stage counts over all files (a seed's worth: the mean over seeds)
    n_seeds = max(len({r['seed'] for r in results}), 1)
    for kind, order, title in [('missed', TRACE_ORDER, 'Missed GT peaks by stage lost'), ('wrong', WRONG_ORDER, 'Wrong structures'), ('fp', FP_ORDER, 'False positives')]:
        lines.append(f'\n{title} (dev / held-out)')
        counts = {h: Counter(t['stage'] for r in results if r['held_out'] == h for t in r['trace'] if t['kind'] == kind) for h in (False, True)}
        for stage in order:
            if counts[False][stage] or counts[True][stage]:
                lines.append(f"  {stage:<32}{counts[False][stage] / n_seeds:7.1f}{counts[True][stage] / n_seeds:7.1f}")
    errors = [t['stage'] for r in results for t in r['trace'] if t['kind'] == 'error'] + [r['error'] for r in results if r['error']]
    if errors:
        lines.append(f'\n{len(errors)} errors, first:\n{errors[0]}')
    return '\n'.join(lines), summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CandyCrunch evaluation harness')
    parser.add_argument('--sets', default = None, help = 'comma-separated set names (e.g. GPST000029,milk,GG_3_N_reduced, or held_out); default: all')
    parser.add_argument('--format', default = 'xlsx', choices = ['xlsx', 'mzml'])
    parser.add_argument('--n_jobs', type = int, default = max(1, (os.cpu_count() or 2) // 2), help = 'parallel files (each holds one run in memory, ~1-3 GB)')
    parser.add_argument('--seeds', default = '0')
    parser.add_argument('--private_spectra', default = str(BASE_DIR.parent / 'Gothenburg' / 'Projects' / 'GlycoGauntlet' / 'improved_xlsx_2026-10-10' / 'private_test'))
    parser.add_argument('--private_gt', default = str(BASE_DIR.parent / 'Gothenburg' / 'Projects' / 'GlycoGauntlet' / 'refined_ground_truths' / 'private_test'))
    parser.add_argument('--no_private', action = 'store_true')
    parser.add_argument('--cache', default = str(pathlib.Path.home() / '.cache' / 'candycrunch_harness'))
    parser.add_argument('--no_cache', action = 'store_true')
    parser.add_argument('--stage_cache_gb', type = float, default = 20, help = 'size of the cached step outputs, least recently used pruned first')
    parser.add_argument('--files_per_process', type = int, default = 4, help = 'files per process beyond the n_jobs largest (its imports take ~20 s)')
    parser.add_argument('--out', default = None, help = 'JSON with every score and trace row')
    parser.add_argument('--compare', default = None, help = 'an earlier --out JSON; per-set F1 differences are printed')
    parser.add_argument('--trace_rows', action = 'store_true', help = 'also print every traced row')
    parser.add_argument('--save_preds', default = None, help = 'folder to pickle every output table into, for diffs between code versions')
    parser.add_argument('--model', default = None, help = 'a fine-tuned checkpoint (weights and vocabulary, from the fine-tuning cell) instead of the shipped model')
    args = parser.parse_args()
    # Set iteration order of strings follows the hash seed, which reaches the outputs (impute variants, subgraph order); the pool's spawned processes inherit it
    os.environ['PYTHONHASHSEED'] = '0'
    tmp = pathlib.Path(args.cache) / '_gt'
    tmp.mkdir(parents = True, exist_ok = True)
    seeds = [int(s) for s in args.seeds.split(',')]
    tasks = [solution_to_gt(dict(t, seeds = seeds, cache = args.cache, use_cache = not args.no_cache, save_preds = args.save_preds, model = args.model), tmp) for t in collect_tasks(args)]
    # Largest files first, so the pool does not end on one long file
    tasks.sort(key = lambda t: -sum(f.stat().st_size for f in pathlib.Path(t['path']).rglob('*') if f.is_file()) if pathlib.Path(t['path']).is_dir() else -os.path.getsize(t['path']))
    print(f'{len(tasks)} files, {args.n_jobs} processes')
    # Finished files go to a .partial.jsonl next to --out as they come in, so a run that dies (e.g. a process killed for memory) resumes where it stopped
    partial = pathlib.Path(args.out).with_suffix('.partial.jsonl') if args.out else None
    results = [json.loads(line) for line in open(partial)] if partial and partial.exists() else []
    done = {(r['label'], r['seed']) for r in results}
    tasks = [t for t in tasks if any((t['label'], s) not in done for s in t['seeds'])]
    start, n_jobs = time.time(), max(args.n_jobs, 1)
    while tasks:
        # The largest files each in a process of their own, the rest a few per process (its memory is returned when it ends; reusing processes through
        # max_tasks_per_child > 1 broke the pool on Windows); if a process dies, the files it did not finish are retried with one process fewer
        chunks = [[t] for t in tasks[:n_jobs]] + [tasks[i:i + args.files_per_process] for i in range(n_jobs, len(tasks), args.files_per_process)]
        try:
            with ProcessPoolExecutor(max_workers = n_jobs, mp_context = multiprocessing.get_context('spawn'), max_tasks_per_child = 1) as pool:
                futures = {pool.submit(run_files, c): c for c in chunks}
                for fut in as_completed(futures):
                    for r in fut.result():
                        results.append(r)
                        if partial:
                            partial.parent.mkdir(parents = True, exist_ok = True)
                            with open(partial, 'a') as f:
                                f.write(json.dumps(r, default = str) + '\n')
                        print(f"  {r['label']:<70} seed {r['seed']}  F1 {r['f1']:.4f}  {r['time']:5.0f} s  cache {r['cache']}" + ('  ERROR' if r['error'] else '')
                              + (f"  uncached (change inputs in a way that cannot be stored): {', '.join(r['uncached'])}" if r.get('uncached') else ''), flush = True)
                    tasks = [t for t in tasks if all(t is not u for u in futures[fut])]
        except BrokenProcessPool:
            if n_jobs == 1:
                raise
            n_jobs -= 1
            print(f'A process died (out of memory?); retrying {len(tasks)} files with {n_jobs} processes', flush = True)
    prune_stage_cache(args.cache, args.stage_cache_gb)
    prev = json.load(open(args.compare)) if args.compare else None
    text, summary = summarize(results, prev)
    print(f'\n{text}\n\nTotal {time.time() - start:.0f} s')
    if args.trace_rows:
        for r in sorted(results, key = lambda r: r['label']):
            for t in r['trace']:
                print(f"{r['label']:<55}{t['kind']:<8}{t['stage']:<28}{t['mass'] if t['mass'] is None else round(t['mass'], 2)!s:>9}{t['rt'] if t['rt'] is None else round(t['rt'], 2)!s:>7}  {t['gt'] or ''}  ->  {t['pred'] or ''}")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents = True, exist_ok = True)
        json.dump(dict(summary, results = results, settings = vars(args)), open(args.out, 'w'), indent = 1, default = str)
        partial.unlink(missing_ok = True)

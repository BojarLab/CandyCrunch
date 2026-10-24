"""Training data from the .cc files of mapped projects: each run's model inputs as CandyCrunch builds them (pooled spectra of MS1-refined precursors, one per
candidate composition), labelled from the project's map with pipeline_labels.label_run, one file per run, merged into a training and a pretraining table.

Usage (rerun whenever files arrive; finished runs are skipped unless their .cc, map, checklist row or this code changed; a run keeps its inputs, so a
changed map, checklist label or labelling rule relabels it without running the pipeline again, while a change of the package's inference code, glycowork or
the run's settings recomputes them):
    python training/cc_training_data.py [--data_root DIR] [--cc_root DIR] [--maps DIR] [--checklist CSV] [--exclusion CSV] [--out DIR] [--projects ID1,ID2]
                                        [--n_jobs 2] [--watch 30]
Paths not given are those of the data folder (--data_root, else $CANDYCRUNCH_DATA, else ../CandyCrunch or ../Gothenburg/Projects/CandyCrunch beside this
repo where one has cc_files/): <data_root>/cc_files, mz_maps, file_checklist.csv, exclusion_files.csv (else the GlycoGauntlet audit's list beside the data
folder; without either only the test runs of tests/ are left out) and the output <data_root>/cc_training.

Inputs: <cc_root>/<project>/**/<run>.cc (project = a map ID: GlycoPOST, PMC, MassIVE, ...), <maps>/df_mz_<project>[_<suffix>].csv (Mass, RT, glycan, optional
sample columns whose names are substrings of the run names, optional reducing_end (a REDUCING_ENDS value or 'custom <tag mass>'), sialic_derivatization,
trap and lc (PGC, C18, HILIC or other) per row, which set them for a run whose rows all agree) and the checklist row of the project (mode, LC_type, trap,
glycan_type, reducing_end, sample_prep, sialic_derivatization, mass_tag).
Per run: the project's maps of the class its name gives (NG/OG, N-/O-glycans, GSL, a sample number ending in N or O; map class from the suffix _N, _O, _GSL,
...), those with a sample column found in the run name (rows where it is > 0), else every row of a map that names no run; glycan class from the maps' suffixes,
else the run name, else the map's structures, else the checklist; map masses 0.5-1 off their structure's ion take its m/z; candidate compositions from the
species' kingdom for non-mammals; without a reducing end in the checklist, reduced and free are both tried and the one labelling more inputs is kept, and if
the checklist's settings label nothing although the map has structures, underivatized/permethylated x reduced/free are tried. label_run shifts map RTs onto the
run and labels RT-less map rows only on their composition's strongest inputs. Runs of test sets and held-out files (tests/data, the training exclusion list of
the GlycoGauntlet audit), and their copies (<run>_2, ... or the same inputs), are written but never merged; the other runs of their projects are training
data like any. Of runs with the same inputs (one acquisition downloaded under several names, also under several project IDs) only the first is merged
(runs.csv: duplicate_of, as project/run; the inputs' hash is kept beside each run as <run>.key). <out>/manifest.json records what the merged tables were made
from (code, glycowork, checklist, maps, exclusions, .cc loaders) and what they hold.
Outputs: <out>/runs/<project>/<run>.pkl (labelled and unlabelled inputs with settings, the inputs of every settings tried, and map_rows: what became of
each map row, with its map file and row), <out>/runs.csv (one line per run; map_row_status counts the map rows' fates), <out>/train.pkl (labeled inputs:
m/z, peak_d, RT, precursor_charge, glycan, glycan_candidates, glycan_match_status, composition, annotation_score, num_spectra, intensity, rt_range, spec_id,
filename, GlycoPost_ID, mode, trap, lc_type, modification (as wrap_inference passes them), model_input, RT2, compositional_vector, glycan_type (the spectrum,
RT, composition and class as the model got them; label_run), label_source, rt_error, mz_error (where the label came from), member_inputs and member_rts
(the spectra the input pools, for re-pooling subsets in training), activation, ms1_trap, rival_compositions, rival_vectors, rival_scores, rival_mapped
(label_run), reducing_end, sample_prep, mass_tag, sialic_derivatization, fragmentation, ion_source, lc_checklist, trap_checklist (the checklist's own
values; wrap_inference got PGC/C18/HILIC/other and linear/orbitrap/amazon/other)) and <out>/pretrain/<project>.pkl
(unlabelled pooled spectra without peak_d unless --pretrain_peaks, with compositions, compositional_vectors, glycan_types and charges of their candidates;
one file per project, as all of them together exceed the memory of a laptop).
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
NON_SAMPLE = {'Mass', 'RT', 'glycan', 'added_db', 'removed_db', 'glycan_class', 'rt_group', 'charge', 'reducing_end', 'sialic_derivatization', 'trap', 'lc'}
# Class of a map by the first part of its suffix (_N_rt: N)
SUFFIX_CLASSES = {'N': 'N', 'Nsulf': 'N', 'O': 'O', 'Osulf': 'O', 'GSL': 'lipid', 'milk': 'free'}
LOG_COLUMNS = ['project', 'run', 'path', 'excluded', 'maps', 'glycan_class', 'rt_shift', 'rt_alignment', 'rt_anchors', 'masses_rescued', 'settings_fitted', 'status',
               'error', 'settings', 'inputs', 'labelled', 'unique', 'structures', 'map_rows', 'map_rows_with_rt', 'map_row_status', 'unlabelled', 'inputs_cached',
               'seconds']
REDUCING_ENDS = {'reduced', 'reduced_D', '2AA', '2AB', 'procainamide', 'PA', 'APTS', 'aminoxyTMT', 'GirardP'}


def sample_columns(df):
    return [c for c in df.columns if c not in NON_SAMPLE and not str(c).startswith(('Unnamed', 'RT_'))]


def name_class(stem):
    # NG/OG, N-glycan/O-glycan, GSL in the run name, or a sample number ending in N or O (JC_171012Ze1N, JC_160226O3)
    s = stem.upper()
    for pattern, cls in [(r'(^|[^A-Z])(NG|N-?GLY)', 'N'), (r'(^|[^A-Z])(OG|O-?GLY)', 'O'), (r'(^|[^A-Z])(GSL|LIPID)', 'lipid'), (r'\dN\d*$', 'N'), (r'\dO\d*$', 'O')]:
        if re.search(pattern, s):
            return cls
    return None


def structure_class(glycans):
    from glycowork.motif.processing import get_class
    classes = pd.Series([get_class(g) for g in glycans if isinstance(g, str)]).replace({'lipid/free': 'free', 'repeat': None, '': None}).dropna()
    return classes.value_counts().index[0] if len(classes) else None


def maps_for_run(project, path, maps_dir):
    # Mapping rows for the run, its glycan class (or list of classes) and the names of the maps used
    maps = sorted(p for p in maps_dir.glob(f'df_mz_{project}*.csv') if p.stem == f'df_mz_{project}' or p.stem.startswith(f'df_mz_{project}_'))
    hint = name_class(path.stem)
    named, general = [], []
    for m in maps:
        df = pd.read_csv(m, encoding = 'utf-8-sig')
        cls = SUFFIX_CLASSES.get(m.stem[len(f'df_mz_{project}'):].lstrip('_').split('_')[0])
        # A map of another class than the run name says (N-glycan maps for an O-glycan run of the same samples) is not the run's, unless a column is the run
        # itself (the O-glycans of an N-glycan fraction)
        if hint and cls and cls != hint and path.stem not in map(str, df.columns):
            continue
        per_run = [c for c in df.columns if str(c).startswith('RT_')]
        if per_run and 'RT' not in df.columns:
            df['RT'] = df[per_run].apply(pd.to_numeric, errors = 'coerce').where(lambda x: x > 0).median(axis = 1)
        # A sample column at the start of the run name (its sample group, as in '6i. <sample>') before columns found elsewhere in it (a '2' or '4' in
        # 'Stage2_4'), then the longest, so 'S1' cannot claim 'S10'
        samples = sample_columns(df)
        cols = sorted((c for c in samples if str(c) in path.name), key = lambda c: (path.name.startswith(str(c)), len(str(c))), reverse = True)
        if cols:
            rows = df[pd.to_numeric(df[cols[0]], errors = 'coerce') > 0].copy()
            if f'RT_{cols[0]}' in rows.columns:
                own = pd.to_numeric(rows[f'RT_{cols[0]}'], errors = 'coerce')
                rows['RT'] = own.where(own > 0, rows['RT'])
            named.append((rows, cls, m.name, (path.name.startswith(str(cols[0])), len(str(cols[0])))))
        else:
            # Maps whose sample columns name no run (samples, not files) apply as a whole, as in the snapshot's extraction
            general.append((df[(df[samples].apply(pd.to_numeric, errors = 'coerce') > 0).any(axis = 1)] if samples else df, cls, m.name))
    # Maps naming the run win over maps for every run, and among them those naming it at its start, then by the longest column (one-character columns
    # such as 1 or 2 are in many run names besides the column of the run's own group)
    chosen = [c[:3] for c in named if c[3] == max(n[3] for n in named)] or general
    if not chosen:
        return None, hint, []
    # Each row with the map file and row it came from (0-based, below the header), so map_rows points curation at it
    mapping = pd.concat([c[0].assign(map_file = c[2], map_row = c[0].index) for c in chosen], ignore_index = True)
    classes = list(dict.fromkeys(c[1] for c in chosen if c[1]))
    cls = classes if len(classes) > 1 else (classes[0] if classes else hint or structure_class(mapping['glycan']))
    return mapping, cls, [c[2] for c in chosen]


def run_settings(row):
    # wrap_inference settings of a project from its checklist row (no modification: reducing end unknown)
    get = lambda k: row[k] if k in row and isinstance(row[k], str) and row[k].strip() else None
    mode = (get('mode') or 'negative').split(';')[0].strip()
    prep = get('sample_prep') or 'underivatized'
    end = get('reducing_end')
    settings = dict(max_charge = (4 if prep != 'underivatized' else 3) * (1 if mode == 'positive' else -1), sample_prep = prep,
                    lc = {'PGC': 'PGC', 'C18': 'C18', 'HILIC': 'HILIC'}.get(get('LC_type'), 'other'),
                    trap = {'linear': 'linear', 'orbitrap': 'orbitrap', 'amazon': 'amazon'}.get(get('trap'), 'other'))
    tag = pd.to_numeric(row.get('mass_tag'), errors = 'coerce')
    if end in REDUCING_ENDS:
        settings['modification'] = end
    elif end in ('free', 'native') or (end and end.startswith('custom')):
        settings.update(modification = 'custom', mass_tag = None if np.isnan(tag) else float(tag))
    sia = get('sialic_derivatization')
    if sia:
        settings['sialic_derivatization'] = tuple(re.findall(r"'([^']+)'", sia)) if sia.startswith('(') else sia
    return settings


def taxonomy(species):
    # Candidate compositions from the glycans of the run's kingdom when its species are no mammals (wrap_inference's default Mammalia lacks many
    # compositions of other animals); species unknown to glycowork count by their genus
    from glycowork.glycan_data.loader import df_species
    taxa = df_species.drop_duplicates('Species').set_index('Species')[['Genus', 'Class', 'Kingdom']]
    found = []
    for s in (species or '').split(';'):
        s = s.strip()
        hit = taxa.loc[s] if s in taxa.index else taxa[taxa['Genus'] == s.split('_')[0]].head(1).squeeze(axis = 0) if (taxa['Genus'] == s.split('_')[0]).any() else None
        if hit is not None and len(hit):
            found.append((hit['Class'], hit['Kingdom']))
    if not found or all(c == 'Mammalia' for c, _ in found) or len({k for _, k in found}) > 1:
        return {}
    return dict(taxonomy_level = 'Kingdom', taxonomy_filter = found[0][1])


def rescue_masses(mapping, settings):
    # Masses 0.5-1 off the nearest ion of their row's structure (rounded, the 13C peak, z = 2 m/z written 0.5 off) take that
    # ion's m/z, as the snapshot's extraction did; rows further off (structure and mass disagree) keep their mass and label nothing
    from glycowork.motif.tokenization import glycan_to_mass, get_ion_mzs
    modification = None if settings.get('modification') == 'custom' else settings.get('modification')
    masses, rescued = [], 0
    for glycan, mass in zip(mapping['glycan'], pd.to_numeric(mapping['Mass'], errors = 'coerce')):
        if isinstance(glycan, str) and not np.isnan(mass):
            try:
                neutral = glycan_to_mass(glycan, sample_prep = settings['sample_prep'], modification = modification,
                                         sialic_derivatization = settings.get('sialic_derivatization')) + (settings.get('mass_tag') or 0)
                ions = np.array(list(get_ion_mzs(neutral, max_charge = settings['max_charge'], adducts = ['Na+'] if settings['max_charge'] > 0 else None).values()))
                error = np.abs(ions - mass)
                if 0.5 < np.nanmin(error) <= 1.0:
                    mass, rescued = float(ions[np.nanargmin(error)]), rescued + 1
            except Exception:
                pass
        masses.append(mass)
    return mapping.assign(Mass = masses), rescued


def stamp(path, mapping_names, maps_dir, row, code):
    # What a run's output depends on: the .cc, its maps, its checklist row and the code turning it into inputs and labels
    h = hashlib.sha1(code)
    st = path.stat()
    h.update(f'{st.st_size} {st.st_mtime_ns}'.encode())
    for m in mapping_names:
        h.update((maps_dir / m).read_bytes())
    h.update(json.dumps({k: str(v) for k, v in row.items()}, sort_keys = True).encode())
    return h.hexdigest()


def convert_run(task):
    # One run in its own process: labelled and unlabelled inputs with settings, written to <out>/runs/<project>/<run>.pkl
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from training.pipeline_labels import label_inputs, run_inputs
    path, out, maps_dir = pathlib.Path(task['path']), pathlib.Path(task['out']), pathlib.Path(task['maps'])
    start = time.time()
    info = dict(project = task['project'], run = path.stem, path = str(path), excluded = task['excluded'], maps = ', '.join(task['map_names']), glycan_class = str(task['glycan_class']))
    target = out / 'runs' / task['project'] / f'{path.stem}.pkl'
    # The inputs of the last conversion, by what they depend on (inference code, glycowork, .cc, class and settings): a changed map or labelling rule
    # only relabels them
    cached, used, reused = {}, {}, 0
    if target.exists():
        try:
            cached = pd.read_pickle(target).get('inputs', {})
        except Exception:
            pass
    try:
        settings = dict(run_settings(task['row']), **taxonomy(task['row'].get('Species') if isinstance(task['row'].get('Species'), str) else None))
        ends = task['mapping']['reducing_end'] if task['mapping'] is not None and 'reducing_end' in task['mapping'].columns else pd.Series(dtype = object)
        if len(ends) and ends.notna().all() and ends.nunique() == 1 and ends.iloc[0] in REDUCING_ENDS:
            settings['modification'] = ends.iloc[0]
            settings.pop('mass_tag', None)
        elif len(ends) and ends.notna().all() and ends.nunique() == 1 and str(ends.iloc[0]).startswith('custom'):
            # 'custom <mass>': a tagged reducing end, as the CORA benzyl glycosides ('custom 76.0313' on permethylated glycans)
            tag = str(ends.iloc[0])[len('custom'):].strip()
            settings.update(modification = 'custom', mass_tag = float(tag) if tag else None)
        traps = task['mapping']['trap'] if task['mapping'] is not None and 'trap' in task['mapping'].columns else pd.Series(dtype = object)
        if len(traps) and traps.notna().all() and traps.nunique() == 1 and traps.iloc[0] in ('linear', 'orbitrap', 'amazon'):
            settings['trap'] = traps.iloc[0]
        lcs = task['mapping']['lc'] if task['mapping'] is not None and 'lc' in task['mapping'].columns else pd.Series(dtype = object)
        if len(lcs) and lcs.notna().all() and lcs.nunique() == 1 and lcs.iloc[0] in ('PGC', 'C18', 'HILIC', 'other'):
            settings['lc'] = lcs.iloc[0]
        sias = task['mapping']['sialic_derivatization'] if task['mapping'] is not None and 'sialic_derivatization' in task['mapping'].columns else pd.Series(dtype = object)
        if len(sias) and sias.notna().all() and sias.nunique() == 1:
            from glycowork.motif.tokenization import SIALIC_DERIVATIZATIONS
            if sias.iloc[0] in SIALIC_DERIVATIZATIONS:
                settings['sialic_derivatization'] = sias.iloc[0]
        tries = [settings] if 'modification' in settings else [dict(settings, modification = 'reduced'), dict(settings, modification = 'custom', mass_tag = None)]

        def attempt(s):
            nonlocal reused
            mapping, rescued = rescue_masses(task['mapping'], s) if task['mapping'] is not None else (pd.DataFrame(columns = ['Mass', 'RT', 'glycan']), 0)
            key = hashlib.sha1((task['inputs_stamp'] + json.dumps(s, sort_keys = True, default = str)).encode()).hexdigest()
            if key not in used:
                reused += key in cached
                used[key] = cached[key] if key in cached else run_inputs(str(path), task['glycan_class'], **s)
            return (*label_inputs(used[key], mapping, crumbs_thresh = s.get('crumbs_thresh', 3)), s, mapping, rescued)

        # The first of the most labelling settings
        best, fitted = max([attempt(s) for s in tries], key = lambda r: len(r[0])), False
        if best[0].empty and len(best[1]) and task['mapping'] is not None and task['mapping']['glycan'].notna().any():
            # The checklist's settings label nothing although the map has structures: the derivatization and reducing end that the map's masses fit
            done = [(s['sample_prep'], s['modification']) for s in tries]
            # A sialic acid derivatization only exists on underivatized glycans
            other = [attempt(dict({k: v for k, v in settings.items() if k != 'sialic_derivatization' or prep == 'underivatized'}, sample_prep = prep, modification = end,
                                  mass_tag = None, max_charge = (4 if prep != 'underivatized' else 3) * int(np.sign(settings['max_charge']))))
                     for prep in ('underivatized', 'permethylated') for end in ('reduced', 'custom') if (prep, end) not in done]
            if other and max(len(r[0]) for r in other):
                best, fitted = max(other, key = lambda r: len(r[0])), True
        labelled, unlabelled, settings, mapping, rescued = best
        map_rows = pd.DataFrame(labelled.attrs.pop('map_rows'))
        # A run without mapping rows (none of the project's maps names it) has no records, and its status counts are empty
        map_rows = map_rows if len(map_rows) else pd.DataFrame(columns = ['Mass', 'RT', 'glycan', 'status', 'inputs', 'labels'])
        info.update(rt_shift = labelled.attrs.get('rt_shift'), rt_alignment = labelled.attrs.get('rt_alignment'), rt_anchors = labelled.attrs.get('rt_anchors'),
                    masses_rescued = rescued, settings_fitted = fitted, inputs_cached = f'{reused}/{len(used)}')
        end = settings['modification'] if settings['modification'] != 'custom' else ('free' if settings.get('mass_tag') is None else f"custom {settings['mass_tag']}")
        meta = dict(filename = path.stem, GlycoPost_ID = task['project'], reducing_end = end, sample_prep = settings['sample_prep'], mass_tag = settings.get('mass_tag'),
                    sialic_derivatization = str(settings['sialic_derivatization']) if settings.get('sialic_derivatization') else None,
                    fragmentation = task['row'].get('fragmentation'), ion_source = task['row'].get('ionization'), lc_checklist = task['row'].get('LC_type'),
                    trap_checklist = task['row'].get('trap'))
        # glycan_type (labelled) and glycan_types (unlabelled) are label_run's: the class code the model got, the run's class
        labelled = labelled.assign(**meta)
        unlabelled = unlabelled.assign(**meta)
        mapped = mapping[mapping['glycan'].notna()]
        info.update(status = 'ok', settings = json.dumps({k: v for k, v in settings.items()}, default = str), inputs = len(labelled) + len(unlabelled),
                    labelled = len(labelled), unique = int((labelled['glycan_match_status'] == 'unique').sum()), structures = labelled['glycan'].nunique(),
                    map_rows = len(mapped), map_rows_with_rt = int(pd.to_numeric(mapped.get('RT', pd.Series(dtype = float)), errors = 'coerce').notna().sum()),
                    map_row_status = json.dumps(map_rows.loc[map_rows['status'] != 'no_structure', 'status'].value_counts().to_dict()), unlabelled = len(unlabelled))
        target.parent.mkdir(parents = True, exist_ok = True)
        tmp = target.with_suffix(f'.{os.getpid()}.tmp')
        pd.to_pickle(dict(labelled = labelled, unlabelled = unlabelled, settings = settings, info = info, stamp = task['stamp'], map_rows = map_rows, inputs = used), tmp)
        os.replace(tmp, target)
        # The stamp also beside the run, so rescans need not load the pickle
        target.with_suffix('.stamp').write_text(task['stamp'])
    except Exception:
        info.update(status = 'error', error = traceback.format_exc(limit = 3).strip().splitlines()[-1])
    info['seconds'] = round(time.time() - start, 1)
    return info


def convert_runs(tasks):
    return [convert_run(t) for t in tasks]


def excluded_stems(exclusion_csv):
    # Run names keep their dots (a stem cut at the first dot, as in '...-35C-0.6ul-...', matched no run); test runs that are not tracked (tests/untracked:
    # the timsTOF serum and GPST000750 sets, misprediction reports) count as well
    stems = {p.stem for folder in ('data', 'untracked') for p in (BASE_DIR / 'tests' / folder).rglob('*')
             if p.suffix.lower() in ('.raw', '.mzml', '.mzxml', '.xlsx', '.d', '.cc', '.mgf')}
    if exclusion_csv and pathlib.Path(exclusion_csv).exists():
        ex = pd.read_csv(exclusion_csv)
        stems |= {str(f).rsplit('.', 1)[0] for f in ex['filename']} | {str(f).rsplit('.', 1)[0] for f in ex.get('matched_to', pd.Series(dtype = str)).dropna()}
    return stems


def collect(args):
    cc_root, maps_dir, out = pathlib.Path(args.cc_root), pathlib.Path(args.maps), pathlib.Path(args.out)
    checklist = pd.read_csv(args.checklist, encoding_errors = 'replace')
    checklist['GlycoPOST_ID'] = checklist['GlycoPOST_ID'].astype(str).str.strip()
    checklist = checklist.drop_duplicates('GlycoPOST_ID').set_index('GlycoPOST_ID')
    wanted = set(args.projects.split(',')) if args.projects else None
    exclude = excluded_stems(args.exclusion)
    # What the inputs depend on: the package's inference code (not the GUI or CLI), glycowork's motif functions and glycan data (a dev install changes
    # without a version bump) and run_inputs; the labels also on the rest of the labelling and the conversion functions (none of it imported here: the
    # main process stays small)
    labelling = (BASE_DIR / 'training' / 'pipeline_labels.py').read_text(encoding = 'utf-8')
    glycowork = pathlib.Path(importlib.util.find_spec('glycowork').origin).parent
    inputs_code = hashlib.sha1(b''.join(p.read_bytes() for p in sorted((BASE_DIR / 'candycrunch').glob('*.py')) if p.name not in ('gui.py', 'cli.py')) + b''.join(
        p.read_bytes() for p in sorted(glycowork.joinpath('motif').glob('*.py')) + sorted(glycowork.joinpath('glycan_data').glob('*')) if p.is_file()) + next(
        ast.get_source_segment(labelling, f) for f in ast.parse(labelling).body if isinstance(f, ast.FunctionDef) and f.name == 'run_inputs').encode()).hexdigest()
    code = inputs_code.encode() + labelling.encode() + ''.join(
        inspect.getsource(f) for f in (sample_columns, name_class, structure_class, maps_for_run, run_settings, taxonomy, rescue_masses, convert_run)).encode()
    tasks, skipped = [], 0
    for project in sorted(p.name for p in cc_root.iterdir() if p.is_dir()):
        if wanted and project not in wanted:
            continue
        row = checklist.loc[project].to_dict() if project in checklist.index else {}
        for path in sorted(cc_root.joinpath(project).rglob('*.cc')):
            # Files still being downloaded or copied wait for the next scan
            if time.time() - path.stat().st_mtime < 120 or (args.exclude_runs and re.search(args.exclude_runs, path.stem)):
                continue
            mapping, cls, names = maps_for_run(project, path, maps_dir)
            if cls is None:
                cls = (str(row.get('glycan_type', '')).split(';')[0].strip() or 'N') if isinstance(row.get('glycan_type'), str) else 'N'
            target = out / 'runs' / project / f'{path.stem}.pkl'
            st = stamp(path, names, maps_dir, row, code)
            if target.exists():
                try:
                    side = target.with_suffix('.stamp')
                    if (side.read_text() if side.exists() else pd.read_pickle(target)['stamp']) == st:
                        skipped += 1
                        continue
                except Exception:
                    pass
            cc = path.stat()
            tasks.append(dict(project = project, path = str(path), out = str(out), maps = str(maps_dir), mapping = mapping, glycan_class = cls, map_names = names,
                              row = row, stamp = st, excluded = is_excluded(path.stem, exclude),
                              inputs_stamp = hashlib.sha1(f'{inputs_code} {cc.st_size} {cc.st_mtime_ns} {cls}'.encode()).hexdigest()))
    return tasks, skipped


def write(df, path):
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    df.to_pickle(tmp)
    os.replace(tmp, path)


def is_excluded(stem, exclude):
    # The download holds some files several times under suffixed names (<run>_2, _3, ...), copies of an excluded run among them
    return stem in exclude or re.sub(r'_\d+$', '', stem) in exclude


def merge(out, exclude, pretrain_peaks = False, exclude_runs = None, sources = None):
    # Every finished run into train.pkl / pretrain/<project>.pkl (excluded runs left out) and runs.csv; the pretraining tables leave out peak_d unless
    # asked (the model takes model_input, the runs keep the peak lists, and orbitrap runs' peak lists made one project's table 15 GB in memory)
    out = pathlib.Path(out)
    train, infos, n_pretrain = [], [], 0
    projects = {p: sorted(r for r in p.glob('*.pkl') if not (exclude_runs and re.search(exclude_runs, r.stem))) for p in sorted(out.joinpath('runs').iterdir()) if p.is_dir()}
    # Copies of one acquisition have the same inputs (156 of the first 1107 runs, the .cc differing only in the embedded name;
    # examples of shared runs under two project IDs: GPST000422/506, PMC7329767/_xyl, GPST000009/162, GPST000222/268), so runs are told apart by a hash of their
    # inputs over all projects, kept beside each run (<run>.key) as long as the run is not converted again
    keys = {}
    for runs in projects.values():
        for r in runs:
            side = r.with_suffix('.key')
            if side.exists() and side.stat().st_mtime_ns >= r.stat().st_mtime_ns:
                keys[r] = side.read_text()
                continue
            d = pd.read_pickle(r)
            inputs = pd.concat([d['labelled'].reset_index()[['m/z', 'RT', 'num_spectra']], d['unlabelled'].reset_index()[['m/z', 'RT', 'num_spectra']]])
            keys[r] = hashlib.sha1(np.round(inputs.sort_values(['RT', 'm/z']).to_numpy(float), 3).tobytes()).hexdigest() if len(inputs) else r.stem
            side.write_text(keys[r])
    # A copy of an excluded run is excluded too, in any project, and of the others only the first is merged
    excluded_keys = {keys[r] for runs in projects.values() for r in runs if is_excluded(r.stem, exclude)}
    first = {}
    for project, runs in projects.items():
        pretrain = []
        for r in runs:
            d = pd.read_pickle(r)
            info = dict(d['info'], excluded = keys[r] in excluded_keys, duplicate_of = first.get(keys[r]))
            first.setdefault(keys[r], f"{project.name}/{info['run']}")
            # The loader that wrote the run's .cc and the SHA-1 of the file it was extracted from (extract_spectra records it)
            if pathlib.Path(info['path']).exists():
                with np.load(info['path']) as f:
                    cc = json.loads(f['meta'].tobytes())
                info.update(cc_loader = cc.get('loader'), cc_source_sha1 = cc.get('source_sha1'))
            infos.append(info)
            if info['excluded'] or info['duplicate_of']:
                continue
            train.append(d['labelled'].reset_index())
            pretrain.append((d['unlabelled'] if pretrain_peaks else d['unlabelled'].drop(columns = 'peak_d', errors = 'ignore')).reset_index())
        if pretrain:
            write(pd.concat(pretrain, ignore_index = True), out / 'pretrain' / f'{project.name}.pkl')
            n_pretrain += sum(len(f) for f in pretrain)
    # Runs whose conversion failed have no pickle; their last log line goes into runs.csv
    log = out / 'log.csv'
    if log.exists():
        done = {(i['project'], i['run']) for i in infos}
        last = pd.read_csv(log).drop_duplicates(['project', 'run'], keep = 'last')
        infos += [r for r in last[last['status'] == 'error'].to_dict('records') if (r['project'], r['run']) not in done]
    pd.DataFrame(infos).to_csv(out / 'runs.csv', index = False)
    train = pd.concat(train, ignore_index = True) if train else pd.DataFrame()
    write(train, out / 'train.pkl')
    # What the tables were made from, so that a checkpoint of them can be told apart and made again, and what they hold
    import glycowork
    digest = lambda paths: hashlib.sha1(b''.join(p.read_bytes() for p in sorted(paths) if p.is_file())).hexdigest()
    git = lambda *a: subprocess.run(['git', '-C', str(BASE_DIR), *a], capture_output = True, text = True).stdout.strip()
    gw = pathlib.Path(glycowork.__file__).parent
    ok = [i for i in infos if i.get('status') == 'ok']
    manifest = dict(created = time.strftime('%Y-%m-%d %H:%M:%S'), python = sys.version.split()[0], git_commit = git('rev-parse', 'HEAD'),
                    git_changed = git('status', '--porcelain', '--', 'candycrunch', 'training').splitlines(),
                    code_sha1 = digest(list((BASE_DIR / 'candycrunch').glob('*.py')) + list((BASE_DIR / 'training').glob('*.py'))),
                    glycowork = dict(version = glycowork.__version__, sha1 = digest(list(gw.joinpath('motif').glob('*.py')) + list(gw.joinpath('glycan_data').glob('*')))),
                    sources = {k: dict(path = str(v), sha1 = digest([pathlib.Path(v)] if pathlib.Path(v).is_file() else pathlib.Path(v).glob('*.csv')))
                               for k, v in (sources or {}).items() if v and pathlib.Path(v).exists()},
                    exclude_runs = exclude_runs, excluded_stems = sorted(exclude),
                    runs = dict(merged = sum(not i['excluded'] and not i['duplicate_of'] for i in ok), excluded = sum(bool(i['excluded']) for i in ok),
                                duplicates = sum(bool(i['duplicate_of']) and not i['excluded'] for i in ok), failed = len(infos) - len(ok)),
                    cc_loaders = dict(Counter(str(i.get('cc_loader')) for i in ok if not i['excluded'] and not i['duplicate_of'])),
                    labelled = len(train), unlabelled = n_pretrain, structures = int(train['glycan'].nunique()) if len(train) else 0,
                    labelled_per_project = train['GlycoPost_ID'].value_counts().to_dict() if len(train) else {})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent = 1, default = str))
    return len(train), n_pretrain, len(infos)


def main(args):
    tasks, skipped = collect(args)
    print(f'{len(tasks)} runs to convert, {skipped} up to date', flush = True)
    if tasks:
        log = pathlib.Path(args.out) / 'log.csv'
        log.parent.mkdir(parents = True, exist_ok = True)
        # A log of other columns takes the current ones (appended lines would sit under the wrong header)
        if log.exists() and pd.read_csv(log, nrows = 0).columns.tolist() != LOG_COLUMNS:
            pd.read_csv(log).reindex(columns = LOG_COLUMNS).to_csv(log, index = False)
        # A few runs per process (its imports take ~20 s), each process fresh (reusing processes through max_tasks_per_child > 1 stalls the pool on Windows)
        chunks = [tasks[i:i + args.runs_per_process] for i in range(0, len(tasks), args.runs_per_process)]
        with ProcessPoolExecutor(max_workers = args.n_jobs, mp_context = multiprocessing.get_context('spawn'), max_tasks_per_child = 1) as pool:
            for fut in as_completed([pool.submit(convert_runs, c) for c in chunks]):
                for info in fut.result():
                    # One column order for every line (a failed run has fewer fields, which shifted its values under other columns)
                    pd.DataFrame([info]).reindex(columns = LOG_COLUMNS).to_csv(log, mode = 'a', header = not log.exists(), index = False)
                    print(f"  {info['project']}/{info['run']}: {info['status']}, {info.get('labelled', 0)} labelled / {info.get('inputs', 0)} inputs"
                          f" ({info.get('glycan_class')}, {info.get('maps') or 'no map'}) {info['seconds']} s" + (f" {info.get('error')}" if info['status'] == 'error' else ''), flush = True)
    # The merged tables only when a run changed (writing them takes minutes)
    if tasks or not (pathlib.Path(args.out) / 'train.pkl').exists():
        n_train, n_pretrain, n_runs = merge(args.out, excluded_stems(args.exclusion), args.pretrain_peaks, args.exclude_runs,
                                            dict(checklist = args.checklist, maps = args.maps, exclusion = args.exclusion))
        print(f'{n_runs} runs: {n_train} labelled inputs in train.pkl, {n_pretrain} unlabelled in pretrain/', flush = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CandyCrunch training data from mapped .cc files')
    parser.add_argument('--data_root', default = None, help = 'folder with cc_files/, mz_maps/, file_checklist.csv and exclusion_files.csv, which receives cc_training/; '
                        'default: $CANDYCRUNCH_DATA, else ../CandyCrunch or ../Gothenburg/Projects/CandyCrunch beside this repo')
    parser.add_argument('--cc_root', default = None, help = 'default: <data_root>/cc_files')
    parser.add_argument('--maps', default = None, help = 'default: <data_root>/mz_maps')
    parser.add_argument('--checklist', default = None, help = 'default: <data_root>/file_checklist.csv')
    parser.add_argument('--exclusion', default = None, help = "csv of files never to merge (test and held-out runs; column filename); default: <data_root>/exclusion_files.csv, "
                        "else the GlycoGauntlet audit's list beside the data folder; the test runs of tests/ are left out either way")
    parser.add_argument('--out', default = None, help = 'default: <data_root>/cc_training')
    parser.add_argument('--projects', default = None, help = 'comma-separated project folders; default: all')
    parser.add_argument('--n_jobs', type = int, default = 2, help = 'runs converted in parallel (each holds one run in memory, ~1-2 GB)')
    parser.add_argument('--runs_per_process', type = int, default = 10)
    parser.add_argument('--pretrain_peaks', action = 'store_true', help = 'keep the pooled peak lists (peak_d) in pretrain/*.pkl')
    parser.add_argument('--exclude_runs', default = None, help = 'regular expression of run names never converted or merged (runs of another kind, e.g. glycopeptide digests)')
    parser.add_argument('--watch', type = float, default = 0, help = 'minutes between rescans for new files (0: convert once and stop)')
    args = parser.parse_args()
    # Paths not given come from the data folder, so the script runs on anyone's copy of the data
    given = args.data_root or os.environ.get('CANDYCRUNCH_DATA')
    root = pathlib.Path(given) if given else next((p for p in [BASE_DIR.parent / 'CandyCrunch', BASE_DIR.parent / 'Gothenburg' / 'Projects' / 'CandyCrunch']
                                                   if (p / 'cc_files').is_dir()), None)
    if root is None and not all([args.cc_root, args.maps, args.checklist, args.out]):
        parser.error('no data folder found: pass --data_root (the folder with cc_files/, mz_maps/ and file_checklist.csv) or set CANDYCRUNCH_DATA')
    args.cc_root, args.maps, args.checklist, args.out = [getattr(args, k) or str(root / name) for k, name in
                                                         [('cc_root', 'cc_files'), ('maps', 'mz_maps'), ('checklist', 'file_checklist.csv'), ('out', 'cc_training')]]
    if args.exclusion is None:
        args.exclusion = next((str(p) for p in ([root / 'exclusion_files.csv', root.parent / 'GlycoGauntlet' / 'full20261003_audit' / 'exclusion_files.csv'] if root else [])
                               if p.exists()), None)
        if args.exclusion is None:
            print('No exclusion list found: only the test runs of tests/ are left out of train.pkl', flush = True)
    # Set iteration order reaches the inputs (as in the test harness), so conversions are reproducible
    os.environ['PYTHONHASHSEED'] = '0'
    while True:
        main(args)
        if not args.watch:
            break
        time.sleep(args.watch * 60)

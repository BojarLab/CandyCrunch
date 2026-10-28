"""Training data from runs with known glycans: the inputs CandyCrunch builds from a run (pooled spectra of MS1-refined precursors, one per candidate
composition), labelled from a mapping of the run's glycans (e.g. a GlycoPOST map), so that the model is trained on what it gets at inference.

On 24 runs with audited ground truths (2026-10-20), these labels were as often right per spectrum as single scans matched to the same maps (90.7%), with 2
instead of 105 wrong compositions and 9% instead of 26% of spectra away from any ground-truth peak; as candidate sets (every structure of the input's
composition the map lists within the RT window) the right structure was in the label for 97.7% of the spectra.
"""
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import stringify_dict
from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.tokenization import glycan_to_composition
from candycrunch.prediction import ISOTOPE_SPACING, acidic_count, bin_intensities, load_spectra_filepath, plain_composition, wrap_inference

SPECTRUM_COLUMNS = ['spec_id', 'peak_d', 'RT', 'rt_range', 'intensity', 'num_spectra', 'model_input', 'RT2']
# Where a label's structure came from: its mapping row's RT as the mapping gives it (fewer than 5 anchors), shifted by the median offset, by the fitted
# drift model, a row without RT, or a row whose RT was dropped as the anchors disagreed
LABEL_COLUMNS = ['label_source', 'rt_error', 'mz_error']


def run_inputs(spectra_filepath, glycan_class, **inference_kwargs):
    """the model inputs of a run, as label_run labels them\n
   | Arguments:
   | :-
   | spectra_filepath (string): run file as wrap_inference reads it (.cc, .raw, .d, .mzML, .mzXML, ...)
   | glycan_class (string): glycan class of the run, as for wrap_inference
   | **inference_kwargs: further wrap_inference arguments, as for label_run\n
   | Returns:
   | :-
   | Returns wrap_inference(return_inputs = True)'s inputs (attrs: its settings) with the spectrum the model got as model_input (float32 rows of bin index,
   | binned intensity and m/z remainder of each non-empty bin) and, per pooled spectrum, its member spectra as member_inputs (the same rows, the intensity
   | unnormalized, for each spectrum the pool averages) and member_rts, instead of the dense binned_intensities and mz_remainder; a run of several
   | activations gives the inputs of each activation's spectra, with their activation as a column
   """
    loaded = load_spectra_filepath(spectra_filepath, extract_ms1 = True)
    activations = loaded['activation'].map(lambda a: a if isinstance(a, str) else '') if 'activation' in loaded.columns else pd.Series('', index = loaded.index)
    if activations.nunique() < 2:
        inputs = wrap_inference(spectra_filepath, glycan_class, return_inputs = True, _loaded = loaded, **inference_kwargs)
    else:
        # A run fragmenting its precursors by several activations (CID and HCD scans of each) would pool them into one input, so each activation's spectra
        # are processed on their own (inference still pools them), with the analyzer that read them where the file records it (CID in the ion trap,
        # HCD in the orbitrap), their pooled spectra numbered apart
        parts = []
        for activation in sorted(set(activations)):
            spectra = loaded[(activations == activation).values].reset_index(drop = True)
            if 'analyzer' in spectra.columns and spectra['analyzer'].notna().any():
                spectra.attrs['detected_trap'] = {'tof': 'other'}.get(spectra['analyzer'].mode().iloc[0], spectra['analyzer'].mode().iloc[0])
            part = wrap_inference(spectra_filepath, glycan_class, return_inputs = True, _loaded = spectra, **inference_kwargs)
            if len(part):
                parts.append(part.assign(activation = activation or None, spec_id = part['spec_id'] + (parts[-1]['spec_id'].max() + 1 if parts else 0)))
        if not parts:
            return part
        inputs = pd.concat(parts)
        inputs.attrs = dict(parts[0].attrs, activation = None)
    if inputs.empty:
        return inputs
    # The model gets the mean of the pooled spectra's binned intensities and m/z remainders, which the pooled peak_d cannot give back; kept sparse, as a
    # float32 array of the bin index, binned intensity and m/z remainder of each non-empty bin (dense, an input takes 32 KB)
    inputs['model_input'] = [np.vstack([np.flatnonzero(k), b[k], r[k]]).astype(np.float32) for b, r in zip(inputs['binned_intensities'], inputs['mz_remainder'])
                             for k in [(b != 0) | (r != 0)]]
    if 'member_peaks' in inputs.columns:
        # The pooled input is the members' summed binned intensities, normalized, and the plain mean of their m/z remainders (empty members included), so a
        # random subset of them re-pools as the cluster would have (training on such resampled pools served the pipeline better than the pooled inputs,
        # harness DEV 0.743 vs 0.677 over two seeds); one entry per spectrum, shared by its rows of other candidate compositions
        step = (3000 - 39.714) / (2048 - 1)
        frames = np.array([39.714 + step * i for i in range(2048)])
        members = {}
        for sid, peaks, rts in zip(inputs['spec_id'], inputs['member_peaks'], inputs['member_rts']):
            if sid not in members:
                binned = [bin_intensities(p, frames) for p in peaks]
                members[sid] = ([np.vstack([np.flatnonzero(k), b[k], r[k]]).astype(np.float32) for b, r in binned for k in [(b != 0) | (r != 0)]],
                                np.array(rts, dtype = np.float32))
        inputs['member_inputs'] = [members[s][0] for s in inputs['spec_id']]
        inputs['member_rts'] = [members[s][1] for s in inputs['spec_id']]
    return inputs.drop(columns = ['binned_intensities', 'mz_remainder', 'member_peaks'], errors = 'ignore')


def label_run(spectra_filepath, mapping, glycan_class, mz_tolerance = 0.5, rt_window = 1.0, ppm_tolerance = 20, **inference_kwargs):
    """labels the model inputs of a run from a mapping of its glycans\n
   | Arguments:
   | :-
   | spectra_filepath (string): run file as wrap_inference reads it (.cc, .raw, .d, .mzML, .mzXML, ...)
   | mapping (dataframe): the run's glycans, as for label_inputs
   | glycan_class (string): glycan class of the run, as for wrap_inference
   | mz_tolerance (float): as for label_inputs; default:0.5
   | rt_window (float): as for label_inputs; default:1.0
   | ppm_tolerance (float): as for label_inputs; default:20
   | **inference_kwargs: further wrap_inference arguments; modification (and sample_prep, lc, ... where they differ from the defaults) have to be given,
   |                     as its reducing-end and glycan-class checks are skipped\n
   | Returns:
   | :-
   | Returns label_inputs' labeled inputs and unlabelled pooled spectra of the run's inputs (run_inputs)
   """
    return label_inputs(run_inputs(spectra_filepath, glycan_class, **inference_kwargs), mapping, mz_tolerance = mz_tolerance, rt_window = rt_window,
                        crumbs_thresh = inference_kwargs.get('crumbs_thresh', 3), ppm_tolerance = ppm_tolerance)


def label_inputs(inputs, mapping, mz_tolerance = 0.5, rt_window = 1.0, crumbs_thresh = 3, ppm_tolerance = 20):
    """labels the model inputs of a run from a mapping of its glycans\n
   | Arguments:
   | :-
   | inputs (dataframe): the run's model inputs, from run_inputs
   | mapping (dataframe): the run's glycans, columns Mass (m/z of the observed ion), glycan (IUPAC) and optionally RT (min; missing or NaN: any RT, but
   |                      only on the composition's most intense inputs at each charge above crumbs_thresh, as many as the mapping has rows of that composition;
   |                      on PGC two such rows differing only in a2-3 / a2-6 sialic linkages split two of them by elution order, a2-6 first); a row
   |                      without glycan but with Mass and RT labels nothing, yet keeps the inputs nearer it than any structured row of their mass
   | mz_tolerance (float): maximum distance between an input's m/z and a mapping Mass; default:0.5 (mapping masses are literature values)
   | rt_window (float): maximum distance between an input's RT and a mapping RT, after shifting the mapping's RTs onto the run (median offset of at least
   |                    5 compositions it lists once; with at least 10 of them, as a function of RT and of the number of charged residues where that fits
   |                    them better; if their median deviation from it exceeds 1 min, the mapping counts as RT-less, unless at least 5 and 40% of them agree within
   |                    1 min, whose median then shifts it); default:1.0 (2.0 only added labels away from the
   |                    glycans' peaks)
   | crumbs_thresh (float): annotation score an input needs to take a mapping row without RT, and to anchor the RT shift, as the pipeline's gate; default:3
   | ppm_tolerance (float): with accurate survey scans (the inputs' attrs accurate_ms1), the maximum distance in ppm of a mapping Mass, of its first two
   |                        13C peaks or of one 13C spacing below it at the input's charge instead of mz_tolerance, widened to half the last decimal the
   |                        mapping gives the Mass with (1171.58: 0.005) and at most mz_tolerance; default:20\n
   | Returns:
   | :-
   | (1) labeled inputs: index m/z, columns spec_id (pooled spectrum), peak_d, RT, rt_range, intensity, num_spectra, model_input (the spectrum as the model
   |     got it: float32 rows of bin index, binned intensity and m/z remainder of each non-empty bin), RT2 (the RT as the model got it), precursor_charge
   |     (the input's charge), composition, compositional_vector and glycan_type (as the model got them), annotation_score, glycan (structure of the
   |     nearest-RT mapping row), glycan_candidates (every structure of the input's composition the mapping lists within rt_window, nearest RT first: the
   |     label as a candidate set), glycan_match_status ('unique' or 'candidates'), label_source (the nearest row's RT: 'rt' as the mapping gives it,
   |     'rt_median' or 'rt_model' shifted onto the run, 'no_rt' a row without RT, 'rt_dropped' a mapping whose RTs disagreed with the run), rt_error
   |     (input RT minus the nearest row's shifted RT) and mz_error (input m/z minus its Mass), member_inputs and member_rts (the pooled spectra, where
   |     inputs has them), the settings the model encodes, as CandyCrunch passes them at inference: mode, trap, lc_type, modification, activation (the
   |     run's most frequent) and ms1_trap (survey scan analyzer), rival_compositions, rival_vectors, rival_scores and rival_mapped (the spectrum's other
   |     candidate compositions, their composition vectors and annotation scores, and whether the mapping lists them there too); attrs:
   |     rt_shift (the median shift applied to the mapping's RTs), rt_alignment ('none', 'median', 'model' or 'dropped'), rt_anchors (compositions
   |     the shift rests on), map_rows (records, one per mapping row: Mass, RT, glycan, any map_file / map_row columns of the mapping, status (labelled, or why
   |     not: no_structure, no_mass, unreadable, no_input_at_mass, other_composition, outside_rt, open_row_nearer (its inputs sit nearer a same-mass
   |     row without structure), not_strongest (a row without RT whose composition's
   |     strongest passing inputs were taken by its isomers or failed crumbs_thresh), spectrum_taken (its spectra kept another composition's label)),
   |     inputs (labeled inputs with it in their candidate set) and labels (of them, with it as the nearest row))
   | (2) unlabelled pooled spectra (for pretraining), one row per spectrum that got no label: index m/z, the spectrum columns of (1), compositions,
   |     compositional_vectors, glycan_types and charges (per candidate composition of the spectrum), and the settings columns
   """
    settings_columns = ['mode', 'trap', 'lc_type', 'modification', 'activation', 'ms1_trap']
    map_rows = mapping[[c for c in ['Mass', 'RT', 'glycan', 'map_file', 'map_row'] if c in mapping.columns]].assign(
        status = np.where(mapping['glycan'].isna(), 'no_structure', np.where(mapping['Mass'].isna(), 'no_mass', 'unreadable')), inputs = 0, labels = 0)
    if inputs.empty:
        # A run without inputs (no MS2 spectrum with a candidate composition, e.g. the GPI glycan infusions of GPST000193)
        empty = lambda columns: pd.DataFrame(columns = columns, index = pd.Index([], name = 'm/z'))
        labelled = empty(SPECTRUM_COLUMNS + ['precursor_charge', 'composition', 'compositional_vector', 'glycan_type', 'annotation_score', 'glycan', 'glycan_candidates',
                                             'glycan_match_status'] + LABEL_COLUMNS + settings_columns)
        map_rows.loc[map_rows['status'] == 'unreadable', 'status'] = 'no_input_at_mass'
        labelled.attrs.update(rt_shift = 0.0, rt_alignment = 'none', rt_anchors = 0, map_rows = map_rows.to_dict('records'))
        return labelled, empty(SPECTRUM_COLUMNS + ['compositions', 'compositional_vectors', 'glycan_types', 'charges'] + settings_columns)
    settings = inputs.attrs
    inputs = inputs.copy()
    rows = mapping[mapping['glycan'].notna() & mapping['Mass'].notna()]
    rts = pd.to_numeric(rows['RT'], errors = 'coerce') if 'RT' in rows.columns else pd.Series(np.nan, index = rows.index)
    keys, structures, masses, map_rts, row_ids = [], [], [], [], []
    for row_id, glycan, mass, rt in zip(rows.index, rows['glycan'], rows['Mass'], rts):
        # Mapping rows glycowork cannot read cannot label anything
        try:
            key, structure = tuple(sorted(glycan_to_composition(glycan).items())), canonicalize_iupac(glycan)
        except Exception:
            continue
        keys.append(key)
        structures.append(structure)
        masses.append(float(mass))
        map_rts.append(float(rt))
        row_ids.append(row_id)
    masses, map_rts = np.array(masses), np.array(map_rts)
    # Rows without a structure but with a Mass (composition-level rows, isomers the mapping leaves open) still own the peaks at their RT
    open_rows = mapping[mapping['glycan'].isna() & mapping['Mass'].notna()]
    open_masses = open_rows['Mass'].to_numpy(dtype = float)
    open_rts = pd.to_numeric(open_rows['RT'], errors = 'coerce').to_numpy(dtype = float) if 'RT' in open_rows.columns else np.full(len(open_rows), np.nan)
    # Which inputs fit which rows: within mz_tolerance of the Mass, with accurate survey scans within ppm_tolerance of it or of its first two 13C peaks at the
    # input's charge (13C picks: GPST000578 and 651 at z = 3), a Mass given with fewer decimals widened to half its last one (an integer Mass keeps
    # mz_tolerance). Further off is another ion, also with the row's composition among its candidates: at 0.5 m/z GPST000712's 720.321 took N3H3 [M+2Na]2+
    # 719.849, and in the audited test runs with accurate survey scans labels from their ground truths went to ions 42-660 ppm off its peaks within 1.5 min
    mz_values = inputs.index.to_numpy(dtype = float)

    def fit_matrix(masses):
        if not settings.get('accurate_ms1'):
            return np.abs(mz_values[:, None] - masses[None, :]) <= mz_tolerance
        decimals = np.array([next((d for d in range(7) if round(m, d) == m), 6) for m in masses])
        tolerances = np.minimum(mz_tolerance, np.maximum(ppm_tolerance * masses / 1e6, 0.5 * 10.0 ** -decimals))
        spacing = ISOTOPE_SPACING / np.nan_to_num(np.abs(inputs['charge'].to_numpy(dtype = float)), nan = 1.0)[:, None]
        fits = np.zeros((len(mz_values), len(masses)), bool)
        # One spacing below: the precursor picked one peak below the monoisotopic one (large z = 3-4 sialylated ions sit 1.00 Da light, with high annotation
        # scores and mostly a correct-mass input of the same glycan at the same RT)
        for k in range(-1, 3):
            fits |= np.abs(mz_values[:, None] - masses[None, :] - k * spacing) <= tolerances[None, :]
        return fits

    fits, open_fits = fit_matrix(masses), fit_matrix(open_masses)
    input_keys = [tuple(sorted(plain_composition(c).items())) if isinstance(c, dict) else None for c in inputs['composition']]
    # Inputs from the most intense down (spectrum counts without precursor intensities); crumbs_thresh as the pipeline's gate
    by_intensity = np.lexsort((-inputs['num_spectra'].to_numpy(dtype = float), -inputs['intensity'].fillna(0).to_numpy(dtype = float)))
    passing = inputs['annotation_score'].to_numpy() > crumbs_thresh
    # Mapping RTs come from one run of a study, and its other runs drift (up to 1.4 min between replicates of GPST000315): the run's shift is the median
    # offset of the mapping's compositions with a single RT peak (one structure, its charge states within 1 min) to their most intense passing input near
    # one of those rows' Mass, if at least 5 agree within a
    # median deviation of 1 min (audited test runs: 16 more labels right, one fewer wrong; mappings shifted by 1.3 / 2.5 min kept 95% / 92% of the right labels
    # instead of 29% / 9%; a fitted RT line instead broke runs with few anchors)
    with_rt = defaultdict(list)
    for k in range(len(keys)):
        if not np.isnan(map_rts[k]):
            with_rt[keys[k]].append(k)
    single = Counter({key: 1 if len({structures[k] for k in ks}) == 1 and np.ptp(map_rts[ks]) <= 1.0 else len(ks) for key, ks in with_rt.items()})
    first = {}
    for i in by_intensity:
        if passing[i] and single.get(input_keys[i]) == 1:
            first.setdefault(input_keys[i], []).append(i)
    offsets, anchor_rts, anchor_acid, anchored = [], [], [], set()
    for k in range(len(keys)):
        if not np.isnan(map_rts[k]) and single[keys[k]] == 1 and keys[k] not in anchored:
            hit = next((i for i in first.get(keys[k], []) if fits[i, k]), None)
            if hit is not None:
                anchored.add(keys[k])
                offsets.append(inputs['RT'].iloc[hit] - map_rts[k])
                anchor_rts.append(map_rts[k])
                anchor_acid.append(acidic_count(dict(keys[k])))
    rt_shift = float(np.median(offsets)) if len(offsets) >= 5 else 0.0
    alignment = 'median' if len(offsets) >= 5 else 'none'
    given_rts = map_rts.copy()
    n_acid = np.array([acidic_count(dict(k)) for k in keys], dtype = float)
    row_shift = np.full(len(map_rts), rt_shift)
    # Anchors that disagree mean the mapping's RTs come from another gradient or LC than the run, so its rows count as RT-less; with >= 10 anchors that is
    # judged after fitting the drift, as a gradient of another slope scatters the plain offsets as well
    scatter = float(np.median(np.abs(np.array(offsets) - np.median(offsets)))) if len(offsets) >= 5 else 0.0
    if len(offsets) >= 10:
        # On PGC the offset also moves with retention time and with every charged residue (on study maps, after the median shift, up to 88% of the
        # disialylated rows of a map sat over 1 min off), so with >= 10 anchors it is fitted as in wrap_inference_batch: a + b * RT + c * acidic residues,
        # each term only where the anchors span it, iteratively reweighted from the median (Huber, then Tukey's biweight), and used if its anchors scatter less
        y, x, a = np.array(offsets), np.array(anchor_rts), np.array(anchor_acid, dtype = float)
        use_rt, use_acid = bool(np.percentile(x, 90) - np.percentile(x, 10) >= 5), bool((a == 0).sum() >= 3 and (a > 0).sum() >= 3)
        if use_rt or use_acid:
            X, r = np.column_stack([np.ones(len(y))] + [x] * use_rt + [a] * use_acid), y - np.median(y)
            for it in range(30):
                u = np.abs(r) / (1.4826 * np.median(np.abs(r)) + 0.01)
                w = np.sqrt(np.clip(1.345 / np.maximum(u, 1e-9), 0, 1) if it < 20 else np.clip(1 - (u / 4.685) ** 2, 0, None) ** 2)
                beta = np.linalg.lstsq(X * w[:, None], y * w, rcond = None)[0]
                r = y - X @ beta
            if np.median(np.abs(r)) < np.median(np.abs(y - np.median(y))):
                coefs = iter(beta[1:])
                b, c = float(next(coefs)) if use_rt else 0.0, float(next(coefs)) if use_acid else 0.0
                row_shift = beta[0] + b * map_rts + c * np.minimum(n_acid, a.max())
                rt_shift = float(np.nanmedian(row_shift[~np.isnan(map_rts)]))
                alignment = 'model'
                scatter = float(np.median(np.abs(r - np.median(r))))
    if len(offsets) >= 5 and scatter > 1.0:
        # Unless enough anchors agree: in fraction runs and others with isomers the mapping does not list, a composition it lists once has its strongest
        # input at another isomer, 10-50 min off, while the rest still sit on the mapping's gradient
        o = np.array(offsets)
        agree = np.abs(o - o[np.argmax([np.sum(np.abs(o - x) <= 1.0) for x in o])]) <= 1.0
        if agree.sum() >= 5 and agree.mean() >= 0.4:
            rt_shift, alignment = float(np.median(o[agree])), 'median'
            row_shift = np.full(len(map_rts), rt_shift)
        else:
            rt_shift, map_rts, row_shift, alignment = 0.0, np.full(len(map_rts), np.nan), np.zeros(len(map_rts)), 'dropped'
    map_rts = map_rts + row_shift
    # The mapping rows of the input's composition within the windows (a mapping row without RT fits any RT)
    near = [[k for k in np.flatnonzero(fits[i]) if keys[k] == key and not abs(map_rts[k] - rt) > rt_window]
            for i, (rt, key) in enumerate(zip(inputs['RT'], input_keys))]
    windowed = {k for n in near for k in n}
    # An input nearer a same-mass row without structure (moved by the run's median shift) than a structured row is the open row's peak, not the structured
    # row's: an isomer's composition-level row 0.3-0.8 min from a structured one otherwise gives its peak the structured row's label
    if alignment != 'dropped' and len(open_masses):
        open_dist = np.array([np.nanmin(np.abs(open_rts[f] + rt_shift - rt), initial = np.inf) for f, rt in zip(open_fits, inputs['RT'])])
        near = [[k for k in n if not open_dist[i] < abs(map_rts[k] - rt)] for i, (n, rt) in enumerate(zip(near, inputs['RT']))]
    open_taken = windowed - {k for n in near for k in n}
    # A run holds about as many peaks of a composition as the mapping lists isomers of it, plus PGC tails, isotope picks and noise, so a row without RT only
    # labels the most intense inputs of its composition at each charge whose fragments pass the pipeline's crumbs_thresh, as many as the mapping has rows of
    # that composition (26 audited test runs with their mappings' RTs removed: 37% of labels off every ground-truth peak and 92% with the right structure in
    # the label; 10% and 97% with this rule; 6% and 97% with the RTs), counted per activation where a run's activations were processed apart (each holds
    # every peak once)
    isomers, seen = Counter(keys), Counter()
    acts = inputs['activation'].tolist() if 'activation' in inputs.columns else [None] * len(inputs)
    for i in by_intensity:
        kept = []
        for k in near[i]:
            if np.isnan(map_rts[k]):
                if not passing[i]:
                    continue
                seen[k, inputs['charge'].iloc[i], acts[i]] += 1
                if seen[k, inputs['charge'].iloc[i], acts[i]] > isomers[keys[k]]:
                    continue
            kept.append(k)
        near[i] = kept
    # Every structure of those rows, nearest RT first
    nearest = [sorted(n, key = lambda k: np.nan_to_num(abs(map_rts[k] - rt))) for n, rt in zip(near, inputs['RT'])]
    candidates = [list(dict.fromkeys(structures[k] for k in n)) for n in nearest]
    # On PGC a2-6 isomers elute before their a2-3 isomers (343 of 368 given pairs of the study maps), so where a composition's two rows without RT differ only
    # in sialic linkages, its two kept inputs at a charge more than 0.5 min apart split them: the earlier takes the row with fewer a2-3 linkages
    if settings.get('lc') == 'PGC':
        blank = lambda s: re.sub(r'(Neu5[AG]c)\(a2-[36]\)', r'\1', s)
        pairs = defaultdict(list)
        for i, n in enumerate(nearest):
            s = [structures[k] for k in n]
            if (len(n) == 2 and np.isnan(map_rts[n]).all() and blank(s[0]) == blank(s[1]) and not re.search(r'Neu5[AG]c\(a2-(\?|\d/)', s[0] + s[1])
                    and s[0].count('(a2-3)') != s[1].count('(a2-3)')):
                pairs[tuple(sorted(n)), inputs['charge'].iloc[i], acts[i]].append(i)
        for (rows, _, _), members in pairs.items():
            if len(members) == 2 and abs(inputs['RT'].iloc[members[0]] - inputs['RT'].iloc[members[1]]) > 0.5:
                early, late = sorted(members, key = lambda i: inputs['RT'].iloc[i])
                fewer, more = sorted(rows, key = lambda k: structures[k].count('(a2-3)'))
                candidates[early], candidates[late] = [structures[fewer]], [structures[more]]
    source = {'none': 'rt', 'median': 'rt_median', 'model': 'rt_model', 'dropped': 'rt_dropped'}[alignment]
    first_row = [n[0] if n else None for n in nearest]
    inputs = inputs.assign(glycan_candidates = candidates, precursor_charge = inputs['charge'], composition = [stringify_dict(c) for c in inputs['composition']],
                           label_source = [None if k is None else 'no_rt' if np.isnan(given_rts[k]) else source for k in first_row],
                           rt_error = [np.nan if k is None else rt - map_rts[k] for k, rt in zip(first_row, inputs['RT'])],
                           mz_error = [np.nan if k is None else mz - masses[k] for k, mz in zip(first_row, inputs.index)],
                           mode = settings['mode'], trap = settings['trap'], lc_type = settings['lc'], ms1_trap = settings.get('ms1_trap'),
                           activation = inputs['activation'] if 'activation' in inputs.columns else settings.get('activation'),
                           modification = 'permethylated' if settings['sample_prep'] == 'permethylated' else (
                               'reduced' if settings['modification'] is None and not settings['mass_tag'] else settings['modification']),
                           order = np.arange(len(inputs)))
    labelled = inputs[[len(c) > 0 for c in candidates]]
    # A pooled spectrum labelled under several compositions keeps the one its fragments support best
    labelled = labelled.sort_values('annotation_score', ascending = False, kind = 'stable')
    labelled = labelled[~labelled['spec_id'].duplicated()].sort_values('order')
    labelled = labelled.assign(glycan = [c[0] for c in labelled['glycan_candidates']],
                               glycan_match_status = ['unique' if len(c) == 1 else 'candidates' for c in labelled['glycan_candidates']])
    # What became of each mapping row, from the first step it failed (rows with inputs at their mass but of other compositions are the mass-check
    # failures map curation looks for)
    kept_orders = set(labelled['order'])
    key_ids = {}
    input_ids = np.array([key_ids.setdefault(k, len(key_ids)) for k in input_keys])
    takes, labels = defaultdict(list), Counter()
    for i, n in enumerate(near):
        for k in n:
            takes[k].append(i)
    for i, k in enumerate(first_row):
        if k is not None and i in kept_orders:
            labels[k] += 1
    for k, row_id in enumerate(row_ids):
        at_mass = fits[:, k]
        if not at_mass.any():
            status = 'no_input_at_mass'
        elif not (at_mass & (input_ids == key_ids.get(keys[k], -1))).any():
            status = 'other_composition'
        elif k not in windowed:
            status = 'outside_rt'
        elif not takes[k] and k in open_taken:
            status = 'open_row_nearer'
        elif not takes[k]:
            status = 'not_strongest'
        elif not any(i in kept_orders for i in takes[k]):
            status = 'spectrum_taken'
        else:
            status = 'labelled'
        map_rows.loc[row_id, ['status', 'inputs', 'labels']] = status, sum(i in kept_orders for i in takes[k]), labels[k]
    member_columns = [c for c in ['member_inputs', 'member_rts'] if c in labelled.columns]
    labelled = labelled[SPECTRUM_COLUMNS + ['precursor_charge', 'composition', 'compositional_vector', 'glycan_type', 'annotation_score', 'glycan', 'glycan_candidates',
                                            'glycan_match_status'] + LABEL_COLUMNS + member_columns + settings_columns]
    # The spectrum's other candidate compositions: negatives for a composition-level loss, unless rival_mapped (the mapping lists that composition there too)
    label_of = labelled.set_index('spec_id')['composition']
    rivals = inputs[inputs['spec_id'].isin(label_of.index)]
    rivals = dict(list(rivals[rivals['composition'].values != label_of.reindex(rivals['spec_id']).values].drop_duplicates(['spec_id', 'composition']).groupby('spec_id')))
    none = inputs.iloc[:0]
    labelled = labelled.assign(**{f'rival_{k}': [rivals.get(s, none)[c].tolist() for s in labelled['spec_id']] for k, c in
                                  [('compositions', 'composition'), ('vectors', 'compositional_vector'), ('scores', 'annotation_score')]},
                               rival_mapped = [[len(c) > 0 for c in rivals.get(s, none)['glycan_candidates']] for s in labelled['spec_id']])
    rest = inputs[~inputs['spec_id'].isin(labelled['spec_id'])].sort_values('order')
    unlabelled = rest.reset_index().groupby('spec_id', sort = False).agg(**{'m/z': ('m/z', 'first')}, **{c: (c, 'first') for c in SPECTRUM_COLUMNS[1:] + settings_columns},
                                                                       compositions = ('composition', list), compositional_vectors = ('compositional_vector', list),
                                                                       glycan_types = ('glycan_type', list), charges = ('precursor_charge', list))
    unlabelled = unlabelled.reset_index().set_index('m/z')[SPECTRUM_COLUMNS + ['compositions', 'compositional_vectors', 'glycan_types', 'charges'] + settings_columns]
    # map_rows as records: pandas compares attrs when it concatenates frames, which a dataframe in them would break
    labelled.attrs.update(rt_shift = rt_shift, rt_alignment = alignment, rt_anchors = len(offsets), map_rows = map_rows.to_dict('records'))
    return labelled, unlabelled

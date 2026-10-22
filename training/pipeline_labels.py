"""Training data from runs with known glycans: the inputs CandyCrunch builds from a run (pooled spectra of MS1-refined precursors, one per candidate
composition), labelled from a mapping of the run's glycans (e.g. a GlycoPOST map), so that the model is trained on what it gets at inference.

On 24 runs with audited ground truths (2026-10-20), these labels were as often right per spectrum as single scans matched to the same maps (90.7%), with 2
instead of 105 wrong compositions and 9% instead of 26% of spectra away from any ground-truth peak; as candidate sets (every structure of the input's
composition the map lists within the RT window) the right structure was in the label for 97.7% of the spectra.
"""
from collections import Counter
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import stringify_dict
from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.tokenization import glycan_to_composition
from candycrunch.prediction import wrap_inference

SPECTRUM_COLUMNS = ['spec_id', 'peak_d', 'RT', 'rt_range', 'intensity', 'num_spectra', 'model_input', 'RT2']


def label_run(spectra_filepath, mapping, glycan_class, mz_tolerance = 0.5, rt_window = 1.0, **inference_kwargs):
    """labels the model inputs of a run from a mapping of its glycans\n
   | Arguments:
   | :-
   | spectra_filepath (string): run file as wrap_inference reads it (.cc, .raw, .d, .mzML, .mzXML, ...)
   | mapping (dataframe): the run's glycans, columns Mass (m/z of the observed ion), glycan (IUPAC) and optionally RT (min; missing or NaN: any RT, but
   |                      only on the composition's most intense inputs at each charge above crumbs_thresh, as many as the mapping has rows of that composition)
   | glycan_class (string): glycan class of the run, as for wrap_inference
   | mz_tolerance (float): maximum distance between an input's m/z and a mapping Mass; default:0.5 (mapping masses are literature values)
   | rt_window (float): maximum distance between an input's RT and a mapping RT, after shifting the mapping's RTs onto the run (median offset of at least
   |                    5 compositions it lists once; if their median deviation exceeds 1 min, the mapping counts as RT-less); default:1.0 (2.0 only added
   |                    labels away from the glycans' peaks)
   | **inference_kwargs: further wrap_inference arguments; modification (and sample_prep, lc, ... where they differ from the defaults) have to be given,
   |                     as its reducing-end and glycan-class checks are skipped\n
   | Returns:
   | :-
   | (1) labeled inputs: index m/z, columns spec_id (pooled spectrum), peak_d, RT, rt_range, intensity, num_spectra, model_input (the spectrum as the model
   |     got it: float32 rows of bin index, binned intensity and m/z remainder of each non-empty bin), RT2 (the RT as the model got it), precursor_charge
   |     (the input's charge), composition, compositional_vector and glycan_type (as the model got them), annotation_score, glycan (structure of the
   |     nearest-RT mapping row), glycan_candidates (every structure of the input's composition the mapping lists within rt_window, nearest RT first: the
   |     label as a candidate set), glycan_match_status ('unique' or 'candidates'), and the settings the model encodes, as CandyCrunch passes them at
   |     inference: mode, trap, lc_type, modification; attrs['rt_shift'] holds the shift applied to the mapping's RTs
   | (2) unlabelled pooled spectra (for pretraining), one row per spectrum that got no label: index m/z, the spectrum columns of (1), compositions,
   |     compositional_vectors, glycan_types and charges (per candidate composition of the spectrum), and the settings columns
   """
    inputs = wrap_inference(spectra_filepath, glycan_class, return_inputs = True, **inference_kwargs)
    settings_columns = ['mode', 'trap', 'lc_type', 'modification']
    if inputs.empty:
        # A run without inputs (no MS2 spectrum with a candidate composition, e.g. the GPI glycan infusions of GPST000193)
        empty = lambda columns: pd.DataFrame(columns = columns, index = pd.Index([], name = 'm/z'))
        return (empty(SPECTRUM_COLUMNS + ['precursor_charge', 'composition', 'compositional_vector', 'glycan_type', 'annotation_score', 'glycan', 'glycan_candidates',
                                          'glycan_match_status'] + settings_columns),
                empty(SPECTRUM_COLUMNS + ['compositions', 'compositional_vectors', 'glycan_types', 'charges'] + settings_columns))
    settings = inputs.attrs
    # The model gets the mean of the pooled spectra's binned intensities and m/z remainders, which the pooled peak_d cannot give back; kept sparse, as a
    # float32 array of the bin index, binned intensity and m/z remainder of each non-empty bin (dense, an input takes 32 KB)
    inputs['model_input'] = [np.vstack([np.flatnonzero(k), b[k], r[k]]).astype(np.float32) for b, r in zip(inputs['binned_intensities'], inputs['mz_remainder'])
                             for k in [(b != 0) | (r != 0)]]
    rows = mapping[mapping['glycan'].notna() & mapping['Mass'].notna()]
    rts = pd.to_numeric(rows['RT'], errors = 'coerce') if 'RT' in rows.columns else pd.Series(np.nan, index = rows.index)
    keys, structures, masses, map_rts = [], [], [], []
    for glycan, mass, rt in zip(rows['glycan'], rows['Mass'], rts):
        # Mapping rows glycowork cannot read cannot label anything
        try:
            key, structure = tuple(sorted(glycan_to_composition(glycan).items())), canonicalize_iupac(glycan)
        except Exception:
            continue
        keys.append(key)
        structures.append(structure)
        masses.append(float(mass))
        map_rts.append(float(rt))
    masses, map_rts = np.array(masses), np.array(map_rts)
    input_keys = [tuple(sorted(c.items())) if isinstance(c, dict) else None for c in inputs['composition']]
    # Inputs from the most intense down (spectrum counts without precursor intensities); crumbs_thresh as the pipeline's gate
    by_intensity = np.lexsort((-inputs['num_spectra'].to_numpy(dtype = float), -inputs['intensity'].fillna(0).to_numpy(dtype = float)))
    crumbs_thresh = inference_kwargs.get('crumbs_thresh', 3)
    passing = inputs['annotation_score'].to_numpy() > crumbs_thresh
    # Mapping RTs come from one run of a study, and its other runs drift (up to 1.4 min between replicates of GPST000315): the run's shift is the median
    # offset of the mapping's compositions with a single RT row to their most intense passing input near that row's Mass, if at least 5 agree within a
    # median deviation of 1 min (audited test runs: 16 more labels right, one fewer wrong; mappings shifted by 1.3 / 2.5 min kept 95% / 92% of the right labels
    # instead of 29% / 9%; a fitted RT line instead broke runs with few anchors)
    single = Counter(keys[k] for k in range(len(keys)) if not np.isnan(map_rts[k]))
    first = {}
    for i in by_intensity:
        if passing[i] and single.get(input_keys[i]) == 1:
            first.setdefault(input_keys[i], []).append(i)
    offsets = []
    for k in range(len(keys)):
        if not np.isnan(map_rts[k]) and single[keys[k]] == 1:
            hit = next((i for i in first.get(keys[k], []) if abs(inputs.index[i] - masses[k]) <= mz_tolerance), None)
            if hit is not None:
                offsets.append(inputs['RT'].iloc[hit] - map_rts[k])
    rt_shift = float(np.median(offsets)) if len(offsets) >= 5 else 0.0
    # Anchors that disagree mean the mapping's RTs come from another gradient or LC than the run, so its rows count as RT-less
    if len(offsets) >= 5 and np.median(np.abs(np.array(offsets) - np.median(offsets))) > 1.0:
        rt_shift, map_rts = 0.0, np.full(len(map_rts), np.nan)
    map_rts = map_rts + rt_shift
    # The mapping rows of the input's composition within the windows (a mapping row without RT fits any RT)
    near = [[k for k in np.flatnonzero(np.abs(masses - mz) <= mz_tolerance) if keys[k] == key and not abs(map_rts[k] - rt) > rt_window]
            for mz, rt, key in zip(inputs.index, inputs['RT'], input_keys)]
    # A run holds about as many peaks of a composition as the mapping lists isomers of it, plus PGC tails, isotope picks and noise, so a row without RT only
    # labels the most intense inputs of its composition at each charge whose fragments pass the pipeline's crumbs_thresh, as many as the mapping has rows of
    # that composition (26 audited test runs with their mappings' RTs removed: 37% of labels off every ground-truth peak and 92% with the right structure in
    # the label; 10% and 97% with this rule; 6% and 97% with the RTs)
    isomers, seen = Counter(keys), Counter()
    for i in by_intensity:
        kept = []
        for k in near[i]:
            if np.isnan(map_rts[k]):
                if not passing[i]:
                    continue
                seen[k, inputs['charge'].iloc[i]] += 1
                if seen[k, inputs['charge'].iloc[i]] > isomers[keys[k]]:
                    continue
            kept.append(k)
        near[i] = kept
    # Every structure of those rows, nearest RT first
    candidates = [list(dict.fromkeys(structures[k] for k in sorted(n, key = lambda k: np.nan_to_num(abs(map_rts[k] - rt))))) for n, rt in zip(near, inputs['RT'])]
    inputs = inputs.assign(glycan_candidates = candidates, precursor_charge = inputs['charge'], composition = [stringify_dict(c) for c in inputs['composition']],
                           mode = settings['mode'], trap = settings['trap'], lc_type = settings['lc'],
                           modification = 'permethylated' if settings['sample_prep'] == 'permethylated' else (
                               'reduced' if settings['modification'] is None and not settings['mass_tag'] else settings['modification']),
                           order = np.arange(len(inputs)))
    labelled = inputs[[len(c) > 0 for c in candidates]]
    # A pooled spectrum labelled under several compositions keeps the one its fragments support best
    labelled = labelled.sort_values('annotation_score', ascending = False, kind = 'stable')
    labelled = labelled[~labelled['spec_id'].duplicated()].sort_values('order')
    labelled = labelled.assign(glycan = [c[0] for c in labelled['glycan_candidates']],
                               glycan_match_status = ['unique' if len(c) == 1 else 'candidates' for c in labelled['glycan_candidates']])
    labelled = labelled[SPECTRUM_COLUMNS + ['precursor_charge', 'composition', 'compositional_vector', 'glycan_type', 'annotation_score', 'glycan', 'glycan_candidates',
                                            'glycan_match_status'] + settings_columns]
    rest = inputs[~inputs['spec_id'].isin(labelled['spec_id'])].sort_values('order')
    unlabelled = rest.reset_index().groupby('spec_id', sort = False).agg(**{'m/z': ('m/z', 'first')}, **{c: (c, 'first') for c in SPECTRUM_COLUMNS[1:] + settings_columns},
                                                                       compositions = ('composition', list), compositional_vectors = ('compositional_vector', list),
                                                                       glycan_types = ('glycan_type', list), charges = ('precursor_charge', list))
    unlabelled = unlabelled.reset_index().set_index('m/z')[SPECTRUM_COLUMNS + ['compositions', 'compositional_vectors', 'glycan_types', 'charges'] + settings_columns]
    labelled.attrs['rt_shift'] = rt_shift
    return labelled, unlabelled

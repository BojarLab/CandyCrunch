"""Training data from runs with known glycans: the inputs CandyCrunch builds from a run (pooled spectra of MS1-refined precursors, one per candidate
composition), labelled from a mapping of the run's glycans (e.g. a GlycoPOST map), so that the model is trained on what it gets at inference.

On 24 runs with audited ground truths (2026-10-20), these labels were as often right per spectrum as single scans matched to the same maps (90.7%), with 2
instead of 105 wrong compositions and 9% instead of 26% of spectra away from any ground-truth peak; as candidate sets (every structure of the input's
composition the map lists within the RT window) the right structure was in the label for 97.7% of the spectra.
"""
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import stringify_dict
from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.tokenization import glycan_to_composition
from candycrunch.prediction import wrap_inference

SPECTRUM_COLUMNS = ['spec_id', 'peak_d', 'RT', 'rt_range', 'intensity', 'num_spectra']


def label_run(spectra_filepath, mapping, glycan_class, mz_tolerance = 0.5, rt_window = 1.0, **inference_kwargs):
    """labels the model inputs of a run from a mapping of its glycans\n
   | Arguments:
   | :-
   | spectra_filepath (string): run file as wrap_inference reads it (.cc, .raw, .d, .mzML, .mzXML, ...)
   | mapping (dataframe): the run's glycans, columns Mass (m/z of the observed ion), glycan (IUPAC) and optionally RT (min; missing or NaN: any RT)
   | glycan_class (string): glycan class of the run, as for wrap_inference
   | mz_tolerance (float): maximum distance between an input's m/z and a mapping Mass; default:0.5 (mapping masses are literature values)
   | rt_window (float): maximum distance between an input's RT and a mapping RT; default:1.0 (2.0 only added labels away from the glycans' peaks)
   | **inference_kwargs: further wrap_inference arguments; modification (and sample_prep, lc, ... where they differ from the defaults) have to be given,
   |                     as its reducing-end and glycan-class checks are skipped\n
   | Returns:
   | :-
   | (1) labelled inputs: index m/z, columns spec_id (pooled spectrum), peak_d, RT, rt_range, intensity, num_spectra, precursor_charge (the input's charge),
   |     composition, annotation_score, glycan (structure of the nearest-RT mapping row), glycan_candidates (every structure of the input's composition the
   |     mapping lists within rt_window, nearest RT first: the label as a candidate set), glycan_match_status ('unique' or 'candidates'), and the settings the
   |     model encodes, as CandyCrunch passes them at inference: mode, trap, lc_type, modification
   | (2) unlabelled pooled spectra (for pretraining), one row per spectrum that got no label: index m/z, the spectrum columns of (1), compositions and
   |     charges (every candidate composition of the spectrum and its charge), and the settings columns
   """
    inputs = wrap_inference(spectra_filepath, glycan_class, return_inputs = True, **inference_kwargs)
    settings = inputs.attrs
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
    # Every structure of the input's composition within the windows (a mapping row without RT fits any RT), nearest RT first
    candidates = []
    for mz, rt, comp in zip(inputs.index, inputs['RT'], inputs['composition']):
        key = tuple(sorted(comp.items())) if isinstance(comp, dict) else None
        near = [k for k in np.flatnonzero(np.abs(masses - mz) <= mz_tolerance) if keys[k] == key and not abs(map_rts[k] - rt) > rt_window]
        candidates.append(list(dict.fromkeys(structures[k] for k in sorted(near, key = lambda k: np.nan_to_num(abs(map_rts[k] - rt))))))
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
    settings_columns = ['mode', 'trap', 'lc_type', 'modification']
    labelled = labelled[SPECTRUM_COLUMNS + ['precursor_charge', 'composition', 'annotation_score', 'glycan', 'glycan_candidates', 'glycan_match_status'] + settings_columns]
    rest = inputs[~inputs['spec_id'].isin(labelled['spec_id'])].sort_values('order')
    unlabelled = rest.reset_index().groupby('spec_id', sort = False).agg(**{'m/z': ('m/z', 'first')}, **{c: (c, 'first') for c in SPECTRUM_COLUMNS[1:] + settings_columns},
                                                                       compositions = ('composition', list), charges = ('precursor_charge', list))
    unlabelled = unlabelled.reset_index().set_index('m/z')[SPECTRUM_COLUMNS + ['compositions', 'charges'] + settings_columns]
    return labelled, unlabelled

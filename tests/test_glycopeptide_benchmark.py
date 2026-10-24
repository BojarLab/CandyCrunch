import gzip
import json
import os
import re
from functools import lru_cache
import numpy as np
import pytest
from candycrunch.analysis import CandyCrumbs, rank_glycopeptide_compositions, rank_glycopeptide_sites, deisotope_ms2

# Glycopeptide regression benchmark: 270 MS2 spectra of O-glycoprotease and tryptic glycoprotein digests (Orbitrap tribrid: HCD, EThcD and ion-trap ETD)
# with their search-engine identifications, 90 per activation in three strata (N-glycopeptides, one-site and multi-site O-glycopeptides), frozen with
# everything resolved that code or databases would otherwise supply (glycans and sites, each glycan's database structure, the composition candidates of
# the precursor). Each score is printed with its reference, its value as of 2026-10-24, and fails below it like the CandyCrumbs scores; a change that
# moves one deliberately updates it
BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'glycopeptides', 'glycopeptide_benchmark.json.gz')
ACTIVATIONS = ('HCD', 'EThcD', 'ETD')
# Decoys: the same spectra shifted out of tolerance (+-50 ppm for orbitrap spectra, +-7.42 Da for ion-trap ones), whose annotated share is chance
DECOY_SHIFTS = {True: (50e-6, -50e-6), False: (7.42, -7.42)}
LETTERS = {'H': 'Hex', 'N': 'HexNAc', 'F': 'dHex', 'A': 'Neu5Ac', 'G': 'Neu5Gc'}
REFERENCE = {'composition_HCD_precision': 0.978, 'composition_HCD_signal': 0.585, 'composition_EThcD_precision': 0.955, 'composition_EThcD_signal': 0.424,
             'composition_ETD_precision': 0.708, 'composition_ETD_signal': 0.299, 'structure_HCD_precision': 0.987, 'structure_HCD_signal': 0.643,
             'structure_EThcD_precision': 0.94, 'structure_EThcD_signal': 0.482, 'structure_ETD_precision': 0.723, 'structure_ETD_signal': 0.344,
             'localization_EThcD': 0.769, 'localization_ETD': 0.683, 'localization_prior_EThcD': 0.824, 'localization_prior_ETD': 0.809,
             'composition_ranking_HCD': 1.0, 'composition_ranking_EThcD': 0.929, 'composition_ranking_ETD': 0.92}
# floating-point sums differ in their last digits between Python versions, which can flip a tie
TOLERANCE = 0.005


@lru_cache(maxsize = 1)
def benchmark_spectra():
    with gzip.open(BENCHMARK, 'rt', encoding = 'utf-8') as f:
        return json.load(f)['spectra']


def spectra_of(activation):
    return [s for s in benchmark_spectra() if s['activation'] == activation]


def annotated_share(s, glycans, glycosites, shift = 0.0, **kwargs):
    # the share of square-root intensity CandyCrumbs annotates among the 100 most intense peaks, deisotoped for orbitrap spectra as the app annotates them
    accurate = s['analyzer'] != 'linear'
    peaks = dict(map(tuple, s['peaks']))
    if accurate:
        peaks = deisotope_ms2(peaks, max(s['charge'], 1), 0.05)
    peaks = sorted(peaks.items(), key = lambda x: -x[1])[:100]
    mz = [m * (1 + shift) if accurate else m + shift for m, _ in peaks]
    hits = CandyCrumbs({'peptide': s['peptide'], 'glycans': glycans, 'glycosites': glycosites}, mz, 0.5 if accurate else 0.2, charge = s['charge'],
                       intensities = [x for _, x in peaks], fragmentation_method = s['activation'], mass_threshold_ppm = 10 if accurate else None, **kwargs)
    weights = np.sqrt([x for _, x in peaks])
    return float(weights[[hits.get(m) is not None for m in mz]].sum() / weights.sum())


def check(name, score, n):
    reference = REFERENCE.get(name)
    print(f"{name}: {score:.3f} ({n} spectra), reference: {reference}")
    assert reference is None or score >= reference - TOLERANCE, f"{name} {score:.3f} is below its reference {reference}"


@pytest.mark.parametrize('activation', ACTIVATIONS)
@pytest.mark.parametrize('path', ['composition', 'structure'])
def test_glycopeptide_annotation_precision(path, activation):
    # Annotated share of the true spectra against their decoys, for each glycan as its composition or as a database structure: precision is the share
    # of the annotated intensity that is not chance, signal the annotated share beyond chance
    spectra = [s for s in spectra_of(activation) if path == 'composition' or 'structures' in s]
    true, decoy = [], []
    for s in spectra:
        glycans = s['glycans'] if path == 'composition' else s['structures']
        true.append(annotated_share(s, glycans, s['glycosites']))
        decoy.append(np.mean([annotated_share(s, glycans, s['glycosites'], shift = shift) for shift in DECOY_SHIFTS[s['analyzer'] != 'linear']]))
    check(f'{path}_{activation}_precision', 1 - np.mean(decoy) / np.mean(true), len(spectra))
    check(f'{path}_{activation}_signal', np.mean(true) - np.mean(decoy), len(spectra))


@pytest.mark.parametrize('activation', ('EThcD', 'ETD'))
def test_glycopeptide_localization(activation):
    # rank_glycopeptide_sites over every placement of an O-glycopeptide's identified glycans on its S/T (at least 2, at most 40): the share of spectra whose
    # identified placement ranks first, ties shared, without a protease prior and, for the peptides of an O-glycoprotease cutting before the glycosite,
    # with its prior
    scores = {'': [], 'prior_': []}
    for s in spectra_of(activation):
        if not s['localized'] or 'N' in [s['peptide'][p] for p in s['glycosites']]:
            continue
        protease = next((p for k, p in (('IMPa', 'IMPa'), ('SmE', 'SmE'), ('Ogp', 'OgpA'), ('Opr', 'OgpA')) if k in s['book']), None)
        truth = sorted(zip(s['glycosites'], [json.dumps(g, sort_keys = True) for g in s['glycans']]))
        for key, prior in (('', None), ('prior_', protease)):
            if key and prior is None:
                continue
            try:
                df = rank_glycopeptide_sites(s['peptide'], s['glycans'], dict(map(tuple, s['peaks'])), charge = s['charge'], glycan_class = 'O',
                                             fragmentation_method = activation, mass_threshold_ppm = 10 if s['analyzer'] != 'linear' else None, protease = prior,
                                             max_placements = 40)
            except ValueError:
                break
            if len(df) < 2:
                break
            first = df[df['rank'] == 1]
            scores[key].append(sum(sorted(zip(sites, [json.dumps(g, sort_keys = True) for g in glycans])) == truth
                                   for sites, glycans in zip(first['glycosites'], first['glycans'])) / len(first))
    for key, values in scores.items():
        check(f'localization_{key}{activation}', np.mean(values), len(values))


@pytest.mark.parametrize('activation', ACTIVATIONS)
def test_glycopeptide_composition_ranking(activation):
    # rank_glycopeptide_compositions over the compositions that fit the precursor within 10 ppm at isotope errors -1 to +2: the share of spectra whose
    # identified composition ranks first, ties shared
    scores = []
    for s in spectra_of(activation):
        if 'composition_candidates' not in s:
            continue
        truth = {LETTERS[a]: int(b) for a, b in re.findall(r'([A-Z])(\d+)', s['byonic_total'])}
        df = rank_glycopeptide_compositions(s['peptide'], [c for c, _ in s['composition_candidates']], dict(map(tuple, s['peaks'])), charge = s['charge'],
                                            glycosite = s['glycosites'][0], isotope_errors = [i for _, i in s['composition_candidates']],
                                            fragmentation_method = activation, mass_threshold_ppm = 10 if s['analyzer'] != 'linear' else None)
        first = df[df['rank'] == 1]
        scores.append(sum(c == truth for c in first['composition']) / len(first))
    check(f'composition_ranking_{activation}', np.mean(scores), len(scores))

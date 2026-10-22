import ast
import bisect
import copy
import hashlib
import inspect
import os
import random
import re
import pickle
import pathlib
import sqlite3
import struct
import tempfile
import zlib
from collections import Counter, defaultdict
from itertools import combinations_with_replacement, groupby
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict
import numpy as np
import pandas as pd
from networkx import descendants
try:
    import torch
    import torch.nn.functional as F
    from candycrunch.model import CandyCrunch_CNN
except ImportError:
    # The built app ships without PyTorch and runs the model on ONNX Runtime (inference.py)
    torch = None
from glycowork.glycan_data.loader import df_glycan, stringify_dict, unwrap
from glycowork.motif.graph import subgraph_isomorphism, glycan_to_nxGraph, compare_glycans, graph_to_string
from glycowork.motif.processing import enforce_class, canonicalize_iupac, get_class, is_composition
from glycowork.motif.annotate import get_molecular_properties
from glycowork.motif.tokenization import (composition_to_mass, get_ion_mzs,
                                          glycan_to_composition, PROTON_MASS,
                                          glycan_to_mass, modification_formula_dict, calculate_adduct_mass, get_formula,
                                          mz_to_composition, structure_to_basic, mass_dict, SIALIC_KEYS,
                                          sialic_linkage_splits, resolve_sialic_linkages, _carboxyl_sites)
from glycowork.network.biosynthesis import construct_network, evoprune_network
from candycrunch.inference import CandyCrunchCNN, load_weights, set_num_threads, transform_mz, transform_rt
from candycrunch.analysis import CandyCrumbs, PEPTIDE_ION_TYPES, supporting_ions, run_jobs, deisotope_ms2
from candycrunch.utils import read_mzml, read_mzxml, read_mgf, read_raw_scan_table, read_raw_packet, centroid_ion_trap, centroid_tof, mz_order, TIMS_MS1_PEAKS, TIMS_MS2_PEAKS

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, 'glycans.pkl'), 'rb') as f:
    glycans = pickle.load(f)
with open(os.path.join(this_dir, 'glytoucan_mapping.pkl'), 'rb') as f:
    glytoucan_mapping = pickle.load(f)
sdict = os.path.join(this_dir, 'candycrunch.pt')
if torch is not None:
    # Choose the correct computing architecture
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    sdict = torch.load(sdict, map_location = device, weights_only = True)
    sdict = {k.replace('module.', ''): v for k, v in sdict.items()}
    candycrunch = CandyCrunch_CNN(2048, num_classes = len(glycans), input_precursor_dim = 12).to(device)
    candycrunch.load_state_dict(sdict)
    candycrunch = candycrunch.eval()
else:
    candycrunch = CandyCrunchCNN(load_weights(sdict))
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

NEGATIVE_ADDUCTS = ['Acetate', 'Formate', 'HCO3-']
# Trap setting for the PSI-MS instrument model (Bruker amaZon) and mass analyzer terms of an mzML instrument configuration, checked in this order
ANALYZER_TRAPS = {'MS:1001542': 'amazon', 'MS:1001546': 'amazon', 'MS:1002300': 'amazon', 'MS:1002301': 'amazon', 'MS:1003466': 'amazon',
                  'MS:1001545': 'amazon', 'MS:1000484': 'orbitrap', 'MS:1000079': 'orbitrap', 'MS:1000078': 'linear', 'MS:1000083': 'linear',
                  'MS:1000082': 'linear', 'MS:1000264': 'linear', 'MS:1000084': 'other', 'MS:1000126': 'other'}
# Time-of-flight analyzers (Q-TOF, IT-TOF); msconvert writes Waters instruments as the generic Waters model without analyzers, and the Waters instruments
# acquiring MS/MS with survey scans are Q-TOFs
TOF_TERMS = {'MS:1000084', 'MS:1000126'}
# The same for the msModel and msMassAnalyzer names of an mzXML msInstrument (lower-case), checked in this order
MZXML_ANALYZER_TRAPS = {'amazon': 'amazon', 'orbitrap': 'orbitrap', 'fourier transform': 'orbitrap',
                        'ft-icr': 'orbitrap', 'ion trap': 'linear',
                        'time-of-flight': 'other', 'tof': 'other'}
POSITIVE_ADDUCTS = ['Na+', 'K+', 'NH4+']
temperature = torch.Tensor([1.15]).to(device) if torch is not None else 1.15
comp_vector_order = ['dHex', 'Hex', 'HexA', 'HexN', 'HexNAc', 'Kdn', 'Me', 'Neu5Ac', 'Neu5Gc', 'P', 'Pen', 'S']
MZ_REF = 1600  # reference m/z used internally to turn the user's single ppm tolerance into the flat-Da window for binning/fragments; corresponds to ~0.5 Da at 300ppm
ISOTOPE_SPACING = 1.003355  # 13C - 12C
# Whether Python's sum adds floats with compensation (from 3.12), which numpy versions of it have to follow to be bitwise identical
COMPENSATED_SUM = sum([1e16, 1.0, -1e16]) == 1.0
# Label mass a reducing-end modification adds (H2 for reduction, label minus O for reductive amination)
modification_mass_dict = {k: calculate_adduct_mass(v) for k, v in modification_formula_dict.items()}
# Natural abundances of each element's isotopes at +0, +1, +2, ... Da, for the share of a glycan's molecules within the integrated isotope peaks
ISOTOPE_ABUNDANCES = {'C': [0.9893, 0.0107], 'H': [0.999885, 0.000115], 'N': [0.99636, 0.00364], 'O': [0.99757, 0.00038, 0.00205],
                      'S': [0.9499, 0.0075, 0.0425, 0, 0.0001], 'P': [1.0]}


def get_adduct_list(mode):
    return NEGATIVE_ADDUCTS if mode == 'negative' else POSITIVE_ADDUCTS


def refine_precursor_mz(ms1, mzs, rts, scans, refine, mz_tolerance = 0.3, isotope_tolerance = 0.25, min_explained = 0.7, charges = None, isolation_mz = False,
                        max_lighter = None):
    """moves precursor m/z values the instrument never refined onto the monoisotopic centroid of their survey scan\n
   | Arguments:
   | :-
   | ms1 (tuple): flat MS1 store (rts, mzs, intensities, offsets), as built by process_mzML_stack
   | mzs (list): precursor m/z values from the MS2 headers
   | rts (list): retention time of each MS2 spectrum
   | scans (list): index of the survey (MS1) scan preceding each MS2 spectrum, -1 if there was none
   | refine (list): whether to refine each precursor m/z; True where the header has no charge state, as the instrument then never determined the monoisotopic peak
   | mz_tolerance (float): maximum distance between the trigger m/z and a survey-scan centroid; default:0.3
   | isotope_tolerance (float): tolerance for locating lighter isotope peaks; default:0.25
   | min_explained (float): share of a peak's intensity that the isotope envelope of a lighter peak has to explain for the peak to count as its isotope; default:0.7
   | charges (list): header charge state of each spectrum (None if undetermined); with isolation_mz its isotope spacing alone is walked, else the header m/z is the
   |                 instrument's own monoisotopic pick, which only moves one isotope spacing down if the survey scan shows it was the 13C peak; default:None (z = 1,
   |                 then z = 2 spacing)
   | isolation_mz (bool): whether the header m/z is the isolation window center (Bruker) instead of a peak; the strongest centroid within mz_tolerance is then
   |                      the trigger (with a TOF tolerance, Bruker's timsTOF monoisotopic pick itself), and no spectrum is dropped, as every isolation window
   |                      holds the whole envelope; default:False
   | max_lighter (float): a lighter peak only counts as the monoisotopic peak if it is at most this many times as intense as the trigger peak; default:None (any)\n
   | Returns:
   | :-
   | (1) a list of refined precursor m/z values
   | (2) a boolean array marking spectra triggered on an isotope peak whose monoisotopic precursor was fragmented itself within a minute
   """
    ms1_rts, ms1_mzs, ms1_ints, ms1_offsets = ms1
    out, walked = list(mzs), [False] * len(mzs)
    # Isotope envelope relative to the monoisotopic peak, for an average glycan of a given neutral mass (elemental composition per Da of a
    # complex glycan); cached per 10 Da
    props = get_molecular_properties(['Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'])
    per_da = {el: int(n or 1) / props['exact_mass'].iloc[0] for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', props['molecular_formula'].iloc[0])}
    envelopes = {}
    def envelope(mass):
        if int(mass // 10) not in envelopes:
            dist = np.array([1.0])
            for el, per in per_da.items():
                for _ in range(round(per * (mass // 10) * 10)):
                    dist = np.convolve(dist, ISOTOPE_ABUNDANCES.get(el, [1.0]))[:4]
            envelopes[int(mass // 10)] = dist / dist[0]
        return envelopes[int(mass // 10)]
    for i, (mz, s, r) in enumerate(zip(mzs, scans, refine)):
        if not r or s < 0:
            continue
        scan_mzs, scan_ints = ms1_mzs[ms1_offsets[s]:ms1_offsets[s + 1]], ms1_ints[ms1_offsets[s]:ms1_offsets[s + 1]]
        near = np.where(np.abs(scan_mzs - mz) <= mz_tolerance)[0]
        if not len(near):
            continue
        if charges is not None and charges[i] and not isolation_mz:
            # A Thermo ion trap's own monoisotopic pick of a multiply charged precursor is often its 13C peak: the survey scan then merges the
            # monoisotopic and 13C peaks into one centroid below the header m/z, or resolves the monoisotopic peak one isotope spacing below it
            c = near[np.argmax(scan_ints[near])]
            lighter = np.where((np.abs(scan_mzs - (mz - ISOTOPE_SPACING / charges[i])) <= 0.2) & (np.arange(len(scan_mzs)) != c))[0]
            if scan_mzs[c] - mz <= -0.15 or (len(lighter) and scan_ints[lighter].max() >= 0.25 * scan_ints[c]):
                out[i], walked[i] = mz - ISOTOPE_SPACING / charges[i], True
            continue
        # Bruker isolation window centers sit up to 0.5 m/z above the monoisotopic peak, so the nearest centroid there was often a noise shoulder
        k = near[np.argmax(scan_ints[near])] if isolation_mz else np.argmin(np.abs(scan_mzs - mz))
        mono = k
        # DDA often triggers on a heavier isotope once the monoisotopic peak is excluded, so step down to the lightest peak (z = 1, then
        # z = 2 spacing) whose isotope envelope explains this peak, unless that peak itself is explained as an isotope of an even lighter one
        for z in ((charges[i],) if charges is not None and charges[i] else (1, 2)):
            chain = [k]
            for j in range(1, 4):
                near = np.where(np.abs(scan_mzs - (scan_mzs[k] - j * ISOTOPE_SPACING / z)) <= isotope_tolerance)[0]
                if not len(near):
                    break
                chain.append(near[np.argmax(scan_ints[near])])
            for j in range(len(chain) - 1, 0, -1):
                below = np.where(np.abs(scan_mzs - (scan_mzs[chain[j]] - ISOTOPE_SPACING / z)) <= isotope_tolerance)[0]
                below = below[np.argmax(scan_ints[below])] if len(below) else None
                if (scan_ints[chain[j]] * envelope(scan_mzs[chain[j]] * z)[j] >= min_explained * scan_ints[k] and (max_lighter is None or scan_ints[chain[j]] <= max_lighter * scan_ints[k]) and
                        (below is None or scan_ints[below] * envelope(scan_mzs[below] * z)[1] < min_explained * scan_ints[chain[j]])):
                    mono = chain[j]
                    break
            if mono != k:
                break
        walked[i] = mono != k
        # Ion-trap centroids jitter between scans, so average the monoisotopic centroid over the neighboring survey scans, within the isotope tolerance (an
        # orbitrap or TOF neighbour's nearest centroid up to 0.2 away is often another ion: walked glycopeptide precursors moved by up to 120 ppm)
        pos, wts = [], []
        for t in range(max(s - 1, 0), min(s + 2, len(ms1_rts))):
            t_mzs, t_ints = ms1_mzs[ms1_offsets[t]:ms1_offsets[t + 1]], ms1_ints[ms1_offsets[t]:ms1_offsets[t + 1]]
            dist = np.abs(t_mzs - scan_mzs[mono])
            if len(t_mzs) and dist.min() <= min(0.2, isotope_tolerance):
                c = dist.argmin()
                pos.append(t_mzs[c])
                wts.append(t_ints[c])
        out[i] = float(np.average(pos, weights = wts)) if sum(wts) > 0 else float(scan_mzs[mono])
    # An isotope-triggered MS2 repeats its monoisotopic precursor's fragmentation with partly 13C-shifted fragments; if that precursor was
    # fragmented itself within a minute, the isotope spectrum only blurs the cluster (and can become its apex), so it is dropped
    out_arr, walked, rts = np.array(out), np.array(walked), np.asarray(rts, dtype = float)
    drop = np.array([w and not isolation_mz and bool(np.any(~walked & (np.abs(out_arr - m) < mz_tolerance) & (np.abs(rts - rt) <= 1.0))) for m, rt, w in
                     zip(out_arr, rts, walked)], dtype = bool)
    return out, drop


def refine_tof_precursors(ms1, mzs, rts, scans, charges):
    """precursor m/z and intensities of timsTOF spectra (.d folders and their mzML), from their TOF survey frames\n
   | Arguments:
   | :-
   | ms1 (tuple): flat MS1 store (rts, mzs, intensities, offsets), as built by process_mzML_stack
   | mzs (list): precursor m/z values (Bruker's monoisotopic m/z where it has a charge state)
   | rts (list): retention time of each MS2 spectrum
   | scans (list): index of each MS2 spectrum's survey frame, -1 if there was none
   | charges (list): Bruker's charge state of each precursor (None if undetermined)\n
   | Returns:
   | :-
   | (1) a list of precursor m/z values
   | (2) a boolean array marking spectra triggered on an isotope peak whose monoisotopic precursor was fragmented itself within a minute
   | (3) a list of precursor intensities
   """
    # Bruker's monoisotopic m/z is a 13C peak for about one precursor in eight, one to three spacings too high, which the survey frame shows as a lighter
    # peak whose isotope envelope explains it: the walk of refine_precursor_mz at the header charge with TOF tolerances (as for Bruker HCT headers, as the
    # 3-4 Da PASEF isolation window holds the whole envelope, so no spectrum is dropped), and every other precursor keeps Bruker's m/z
    refined, drop = refine_precursor_mz(ms1, mzs, rts, scans, [True] * len(mzs), mz_tolerance = 0.02, isotope_tolerance = 0.02, charges = charges,
                                        isolation_mz = True)
    out = [r if not z or abs(r - m) > ISOTOPE_SPACING / z / 2 else m for r, m, z in zip(refined, mzs, charges)]
    # The intensity is the precursor's isotope envelope (M to M+3, within 20 ppm) in its survey frame: Bruker's per-precursor intensity belongs to one
    # ion mobility feature, so precursors of one peak picked at different mobilities from one frame differ up to 20-fold, which split chromatographic peaks
    intensities = []
    for k, (mz, z, s) in enumerate(zip(out, charges, scans)):
        scan_mzs, scan_ints = ms1[1][ms1[3][s]:ms1[3][s + 1]], ms1[2][ms1[3][s]:ms1[3][s + 1]]
        if s < 0:
            intensities.append(np.nan)
        elif z:
            # Bruker's m/z is that of one ion mobility feature (robust SD 2.4 ppm against the compositions of the timsTOF serum ground truth), the survey
            # frame's centroid, summed over mobility, that of the whole precursor (1.6 ppm), as finalise_predictions' accurate-MS1 window expects
            near = np.flatnonzero(np.abs(scan_mzs - mz) <= 2e-5 * mz)
            if len(near):
                out[k] = mz = float(scan_mzs[near[np.argmax(scan_ints[near])]])
            targets = mz + np.arange(4) * ISOTOPE_SPACING / z
            lo, hi = np.searchsorted(scan_mzs, targets * (1 - 2e-5)), np.searchsorted(scan_mzs, targets * (1 + 2e-5))
            intensities.append(float(sum(scan_ints[a:b].sum() for a, b in zip(lo, hi))))
        else:
            intensities.append(float(scan_ints[(scan_mzs >= mz - 1.5) & (scan_mzs < mz + 1.5)].sum()))
    return out, drop, intensities


def process_mzML_stack(filepath, num_peaks = 1000,
                       ms_level = 2, intensity = False, extract_ms1 = False):
    """function extracting all MS/MS spectra from .mzML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzML file
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | ms_level (int): which MS^n level to extract; default:2
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
   | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, intensity if True, scan (the scan= number of the native
   | spectrum ID, else the whole ID), and activation ('CID', 'HCD', 'ETD', 'ECD', 'EThcD', 'ETciD', the fragmentation_method values of CandyCrumbs,
   | with EAD as 'ECD' for its c/z ions; None if not stated), and, if the file has MS3 spectra, ms3 (a list of (MS3 precursor m/z, peak dictionary)
   | tuples per MS2 spectrum); attrs detected_mode, detected_trap (MS2 analyzer), and detected_ms1_trap (survey scan analyzer)
   """
    highest_i_dict = {}
    rts, intensities, mzs, charges, scans, activations = [], [], [], [], [], []
    detected_mode, detected_trap, detected_ms1_trap = None, None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine, thermo = [], [], [], [], [], []
    ms3s, row_of, tims = [], {}, False
    # Survey scans are centroided too: precursor refinement and XIC quantification work on centroids, and profile points of msconvert files (zero
    # intensities included) crashed the refinement or walked it along a peak's flank
    for spectrum in read_mzml(filepath, centroid_levels = (1, ms_level, ms_level + 1)):
        if spectrum['ms_level'] == ms_level + 1 and spectrum['precursor'] and mzs:
            # An MS3 spectrum fragments one fragment of an MS2 spectrum; its precursors are listed latest stage first, each referencing its
            # parent scan by native ID (else, as in old converters, it belongs to the latest MS2 spectrum)
            ref = next(spectrum['element'].iter('{http://psi.hupo.org/ms/mzml}precursor')).get('spectrumRef')
            row = row_of.get(ref, len(mzs) - 1 if ref is None else None)
            peaks = spectrum['peaks']
            if row is not None and len(peaks):
                ms3s[row].append((spectrum['precursor']['mz'], dict(sorted(((float(m), float(i)) for m, i in peaks[peaks[:, 1].argsort()][-num_peaks:]),
                                                                           key = lambda x: x[1], reverse = True))))
        if spectrum['ms_level'] == 1:
            # The survey scan analyzer sets how far the precursor m/z can be trusted (an orbitrap MS1 pins the composition by accurate mass); the
            # Thermo filter string names it, else the analyzer of the scan's instrument configuration
            if detected_ms1_trap is None:
                filt = next((cv.get('value', '') for cv in spectrum['element'].iter('{http://psi.hupo.org/ms/mzml}cvParam') if cv.get('accession') == 'MS:1000512'), '')
                detected_ms1_trap = {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(filt[:4]) or ('tof' if spectrum['ion_mobility'] or spectrum['instrument'] & TOF_TERMS else
                                                                                       next((t for a, t in ANALYZER_TRAPS.items() if a in spectrum['instrument']), None))
                tims = spectrum['ion_mobility']
            # Waters runs interleave their lockmass function (the last one) with the survey scans (function 1), and msconvert writes its reference scans as
            # MS1 spectra, which then gave the MS2 spectra after them their survey scan
            if (waters := re.search(r'\bfunction=(\d+)', spectrum['element'].get('id', ''))) and waters.group(1) != '1':
                continue
            peaks_raw = spectrum['peaks']
            if len(peaks_raw) > 0:
                ms1_rts.append(spectrum['rt'])
                # float32 halves the memory of MS1 data and still resolves ~0.2 mDa at m/z 2000
                ms1_mzs.append(peaks_raw[:, 0].astype(np.float32))
                ms1_ints.append(peaks_raw[:, 1].astype(np.float32))
        if spectrum['ms_level'] == ms_level:
            if detected_mode is None or detected_trap is None:
                for cv in spectrum['element'].iter('{http://psi.hupo.org/ms/mzml}cvParam'):
                    acc = cv.get('accession', '')
                    if acc == 'MS:1000129':
                        detected_mode = 'negative'
                    elif acc == 'MS:1000130':
                        detected_mode = 'positive'
                    elif acc == 'MS:1000512':
                        filt = cv.get('value', '')
                        if filt.startswith('ITMS'):
                            detected_trap = 'linear'
                        elif filt.startswith('FTMS'):
                            detected_trap = 'orbitrap'
                # Without a Thermo filter string, the analyzer (or Bruker amaZon model) of the scan's instrument configuration; analyzer terms are
                # never params of the spectrum itself
                if detected_trap is None:
                    detected_trap = next((t for a, t in ANALYZER_TRAPS.items() if a in spectrum['instrument']), None)
            peaks = spectrum['peaks']
            tof = spectrum['ion_mobility'] or bool(spectrum['instrument'] & TOF_TERMS)
            if tof and spectrum['precursor']:
                # As in process_tdf_stack: the unfragmented precursor envelope of a beam-type CID spectrum (PASEF or any other Q-TOF) is not a fragment
                window = {cv.get('accession'): float(cv.get('value')) for cv in spectrum['element'].iter('{http://psi.hupo.org/ms/mzml}cvParam')
                          if cv.get('accession') in ('MS:1000827', 'MS:1000828', 'MS:1000829')}
                iso_mz, charge = window.get('MS:1000827', spectrum['precursor']['mz']), spectrum['precursor'].get('charge')
                upper = max(iso_mz + window.get('MS:1000829', 0), spectrum['precursor']['mz'] + 6 * ISOTOPE_SPACING / abs(charge) + 0.05 if charge else 0)
                peaks = peaks[(peaks[:, 0] < iso_mz - window.get('MS:1000828', 0)) | (peaks[:, 0] > upper)]
            # dict(zip()) over the two columns is the same dict as dict() over the rows, without a row view per peak (~2.5x faster)
            # TOF MS2 spectra hold hundreds to thousands of centroids, mostly single ions, against the 10-50 peaks of the ion trap spectra the model was
            # trained on; their 100 most intense peaks gave more glycans, with fewer false ones, on timsTOF data
            mz_i_dict = dict(zip(*peaks[peaks[:, 1].argsort()][-(min(num_peaks, TIMS_MS2_PEAKS) if tof else num_peaks):].T))
            if mz_i_dict:
                if not spectrum['precursor']:
                    continue
                # The native ID keys the spectra, as scan numbers are not unique for non-Thermo native IDs (SCIEX cycle=1 experiment=2 and
                # cycle=2 experiment=2 both give 2)
                native_id = spectrum['element'].get('id', '')
                key = f"{native_id}_{spectrum['precursor']['mz']}"
                highest_i_dict[key] = mz_i_dict
                row_of[native_id] = len(mzs)
                ms3s.append([])
                mzs.append(float(key.split('_')[-1]))
                rts.append(spectrum['rt'])
                # Search engines identify glycopeptide spectra by scan number, and HCD and EThcD scans of one precursor alternate in
                # glycoproteomics runs; the scan= number of the native ID, else the native ID itself
                scans.append(int(m.group(1)) if (m := re.search(r'\bscan=(\d+)', native_id)) else native_id)
                # EThcD/ETciD are written as their own terms or as ETD plus a (supplemental) beam-type or resonance collisional activation
                terms = {cv.get('accession', '') for cv in spectrum['element'].iter('{http://psi.hupo.org/ms/mzml}cvParam')}
                etd, hcd, cid = 'MS:1000598' in terms, {'MS:1000422', 'MS:1002481', 'MS:1002678'}, {'MS:1000133', 'MS:1000433', 'MS:1002679'}
                activations.append('EThcD' if 'MS:1002631' in terms or (etd and terms & hcd) else
                                   'ETciD' if 'MS:1003182' in terms or (etd and terms & cid) else 'ETD' if etd else
                                   'ECD' if terms & {'MS:1000250', 'MS:1003294'} else 'HCD' if terms & hcd else 'CID' if terms & cid else None)
                raw_charge = spectrum['precursor'].get('charge', None)
                # Some vendor software defaults to charge=1 when undetermined, but Thermo files (filter string) only carry a charge state the
                # instrument determined from resolved isotopes (orbitrap survey or zoom scans), so their charge 1 is kept, as is the one of timsTOF
                # files (Bruker only writes a charge with a monoisotopic m/z it determined from the TOF survey frame)
                if raw_charge is not None and abs(int(raw_charge)) == 1 and 'MS:1000512' not in terms and not spectrum['ion_mobility']:
                    raw_charge = None
                # Without a determined charge state the instrument never determined the monoisotopic peak (Bruker's charge 1 precursors are the
                # isolation m/z, e.g., 384.39 for the T antigen at 384.16), so its trigger m/z gets refined from MS1
                ms1_scans.append(len(ms1_rts) - 1)
                refine.append(raw_charge is None)
                charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
                thermo.append('MS:1000512' in terms)
                if intensity:
                    inty = spectrum['precursor'].get('i', np.nan)
                    # Files without precursor intensities (Bruker) get the summed centroids within 1.5 m/z of the precursor in the preceding survey
                    # scan, as ThermoRawFileParser computes it; without them, a peak's representative spectrum and abundances were arbitrary
                    if not inty > 0 and ms1_mzs:
                        inty = float(ms1_ints[-1][(ms1_mzs[-1] >= mzs[-1] - 1.5) & (ms1_mzs[-1] < mzs[-1] + 1.5)].sum())
                    intensities.append(inty)
    # Sort the highest_i_dict by values
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key = lambda x: x[1], reverse = True))
    df_out = pd.DataFrame({
        'm/z': mzs,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
        'precursor_charge': charges,
    })
    if intensity:
        df_out['intensity'] = intensities
    df_out['scan'], df_out['activation'] = scans, activations
    # Each MS2 spectrum's MS3 spectra as (MS3 precursor m/z, peak dictionary) tuples; files without MS3 get no column
    if any(ms3s):
        df_out['ms3'] = ms3s
    # Flat MS1 store (rts, mzs, intensities, offsets): the peaks of scan i are mzs[offsets[i]:offsets[i + 1]]
    ms1 = (np.array(ms1_rts), np.concatenate(ms1_mzs) if ms1_mzs else np.zeros(0, np.float32),
           np.concatenate(ms1_ints) if ms1_ints else np.zeros(0, np.float32),
           np.concatenate([[0], np.cumsum([len(m) for m in ms1_mzs], dtype = np.int64)]))
    if detected_ms1_trap == 'tof' and tims:
        df_out['m/z'], drop, tof_intensities = refine_tof_precursors(ms1, mzs, rts, ms1_scans, charges)
        if intensity:
            df_out['intensity'] = tof_intensities
    elif detected_ms1_trap == 'tof':
        # Other TOF headers give the isolation target (Agilent, Bruker) or the survey peak (SCIEX, Waters), off its recalibrated survey centroid by up to ~30 ppm
        # (Waters' set mass 8 ppm in pwiz's test run), so every precursor moves onto the strongest centroid within 0.05 m/z and walks its isotopes at the
        # header charge with TOF tolerances
        df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, [True] * len(mzs), mz_tolerance = 0.05, isotope_tolerance = 0.02, charges = charges,
                                                  isolation_mz = True)
    elif not any(thermo) and detected_ms1_trap in ('linear', 'amazon'):
        # Bruker and Agilent ion traps give the isolation target as precursor m/z, also with a charge state, so every precursor is refined as from their
        # mzXML and .d files
        df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, [True] * len(mzs), mz_tolerance = 0.6, min_explained = 0.3, charges = charges,
                                                  isolation_mz = True)
    else:
        # The header charge of a multiply charged precursor comes with a Thermo ion trap's own monoisotopic pick, which is checked against the survey scan
        walk = [c in (2, 3) and t and detected_ms1_trap == 'linear' for c, t in zip(charges, thermo)]
        df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, [r or w for r, w in zip(refine, walk)],
                                                  charges = [c if w else None for c, w in zip(charges, walk)])
        if detected_ms1_trap == 'orbitrap':
            # A Thermo orbitrap's own monoisotopic pick of a glycan above ~2000 Da, whose M+1 rivals M, is often its 13C peak (GPST000651: 303 of 1,680
            # precursors, e.g., A2G2S2F 1243.95 for 1243.45, A4G4S4 1827.65 for 1826.65), which then reads as a composition 1-2 Da heavier, sometimes an isobar
            # within a few ppm (Hex7HexNAc6Neu5Ac3dHex2 for the M+1 of Hex7HexNAc6Neu5Ac4), so those precursors walk their isotopes in the survey scan at the
            # header charge with orbitrap tolerances, as timsTOF precursors do, onto a lighter peak at most twice as intense (a much stronger peak 1-2 Da lower is
            # another glycan, whose isotopes the walk took for the trigger in GPST000461); every other precursor keeps the instrument's m/z. Spectra walked onto vs away
            # from a ground-truth composition: GPST000651 75/2, GPST000403 317/5, GPST000548 40/9)
            walked, _ = refine_precursor_mz(ms1, list(df_out['m/z']), rts, ms1_scans, [bool(c) and m * c >= 2000 for m, c in zip(df_out['m/z'], charges)],
                                            mz_tolerance = 0.02, isotope_tolerance = 0.02, charges = charges, isolation_mz = True, max_lighter = 2)
            df_out['m/z'] = [w if c and abs(w - m) > ISOTOPE_SPACING / c / 2 else m for w, m, c in zip(walked, df_out['m/z'], charges)]
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    df_out.attrs['detected_ms1_trap'] = detected_ms1_trap
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def process_mzXML_stack(filepath, num_peaks = 1000, ms_level = 2, intensity = False, extract_ms1 = False):
    """function extracting all MS/MS spectra from .mzXML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzXML file
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | ms_level (int): which MS^n level to extract; default:2
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
   | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, intensity if True, scan number, and activation as in
   | process_mzML_stack
    """
    highest_i_dict = {}
    rts, intensities, mzs, charges, scans, activations = [], [], [], [], [], []
    detected_mode, detected_trap, detected_ms1_trap = None, None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine = [], [], [], [], []
    ms3s, row_of, isolation_mz = [], {}, False
    # Survey and fragment scans are centroided as in process_mzML_stack (Bruker mzXML from msconvert are profiles)
    for spectrum in read_mzxml(filepath, centroid_levels = (1, ms_level, ms_level + 1)):
        if spectrum['msLevel'] == ms_level + 1 and spectrum.get('precursorMz') and len(spectrum['m/z array']) and mzs:
            # As in process_mzML_stack; mzXML names the parent scan as precursorScanNum (nested scans are yielded one by one)
            prec = next((p for p in spectrum['precursorMz'] if int(p.get('precursorScanNum', -1)) in row_of), spectrum['precursorMz'][0])
            row = row_of.get(int(prec.get('precursorScanNum', -1)), len(mzs) - 1 if 'precursorScanNum' not in prec else None)
            if row is not None:
                top_idx = np.argsort(spectrum['intensity array'])[::-1][:num_peaks]
                ms3s[row].append((float(prec['precursorMz']), {float(m): float(i) for m, i in zip(spectrum['m/z array'][top_idx],
                                                                                                spectrum['intensity array'][top_idx])}))
        if spectrum['msLevel'] == 1 and len(spectrum['m/z array']):
            # The Thermo filter line names the analyzer, else the scan's msInstrument
            if detected_ms1_trap is None:
                filt = str(spectrum.get('filterLine', ''))
                detected_ms1_trap = {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(filt[:4]) or next((t for a, t in MZXML_ANALYZER_TRAPS.items() if a in spectrum['instrument']), None)
            order = np.argsort(spectrum['m/z array'], kind = 'stable')
            ms1_rts.append(float(spectrum['retentionTime']))
            ms1_mzs.append(spectrum['m/z array'][order].astype(np.float32))
            ms1_ints.append(spectrum['intensity array'][order].astype(np.float32))
        if spectrum['msLevel'] == ms_level:
            # mzXML scans carry their polarity and, if converted from Thermo files, the filter line naming the analyzer
            if detected_mode is None and spectrum.get('polarity') in ('+', '-'):
                detected_mode = 'negative' if spectrum['polarity'] == '-' else 'positive'
            if detected_trap is None:
                filt = str(spectrum.get('filterLine', ''))
                detected_trap = {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(filt[:4]) or next((t for a, t in MZXML_ANALYZER_TRAPS.items() if a in spectrum['instrument']), None)
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            top_idx = np.argsort(intensity_array)[::-1][:num_peaks]
            mz_i_dict = dict(zip(mz_array[top_idx], intensity_array[top_idx]))
            if mz_i_dict and spectrum['precursorMz']:
                precursor_mz = spectrum['precursorMz'][0]['precursorMz']
                key = f"{spectrum['id']}_{precursor_mz}"
                highest_i_dict[key] = mz_i_dict
                row_of[int(spectrum['num'])] = len(mzs)
                ms3s.append([])
                mzs.append(float(precursor_mz))
                rts.append(spectrum['retentionTime'])
                scans.append(int(spectrum['num']))
                # mzXML names the method itself; ambiguous ones (ETD+SA) stay None
                activations.append({k.upper(): k for k in PEPTIDE_ION_TYPES if k}.get(
                    str(spectrum['precursorMz'][0].get('activationMethod')).upper()))
                raw_charge = spectrum['precursorMz'][0].get('precursorCharge', None)
                # As in process_mzML_stack, a charge 1 is only kept from Thermo files (filter line), and precursors without one are refined
                if raw_charge is not None and abs(int(raw_charge)) == 1 and 'filterLine' not in spectrum:
                    raw_charge = None
                # Bruker headers give the isolation window center also for the charge states they determine (966.2 for a [M-2H]2- at 965.84), so
                # every precursor of a file without filter lines is refined, walking the isotopes at its header charge
                isolation_mz = isolation_mz or 'filterLine' not in spectrum
                ms1_scans.append(len(ms1_rts) - 1)
                # Thermo headers of multiply charged precursors are checked against an ion-trap survey scan as in process_raw_stack
                refine.append(raw_charge is None or 'filterLine' not in spectrum or (abs(int(raw_charge)) in (2, 3) and detected_ms1_trap == 'linear'))
                charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
                if intensity:
                    inty = spectrum['precursorMz'][0].get('precursorIntensity', np.nan)
                    # As in process_mzML_stack, missing precursor intensities come from the preceding survey scan
                    if not inty > 0 and ms1_mzs:
                        inty = float(ms1_ints[-1][(ms1_mzs[-1] >= mzs[-1] - 1.5) & (ms1_mzs[-1] < mzs[-1] + 1.5)].sum())
                    intensities.append(inty)
    # Sort the highest_i_dict by values
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key = lambda x: x[1], reverse = True))
    df_out = pd.DataFrame({
        'm/z': mzs,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
        'precursor_charge': charges,
    })
    if intensity:
        df_out['intensity'] = intensities
    df_out['scan'], df_out['activation'] = scans, activations
    if any(ms3s):
        df_out['ms3'] = ms3s
    # Same flat MS1 store and precursor refinement as process_mzML_stack
    ms1 = (np.array(ms1_rts), np.concatenate(ms1_mzs) if ms1_mzs else np.zeros(0, np.float32),
           np.concatenate(ms1_ints) if ms1_ints else np.zeros(0, np.float32),
           np.concatenate([[0], np.cumsum([len(m) for m in ms1_mzs], dtype = np.int64)]))
    # Bruker survey scans often show M+1 above the envelope of an average glycan (M+1/M 0.74-1.1 for a 1933 Da [M-2H]2-), so a lighter isotope explaining
    # 30% of the trigger peak counts (precursors within 0.1 m/z of their GT glycan: 84% at 70%, 93% at 50%, 97% at 30%)
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, refine, mz_tolerance = 0.6 if isolation_mz else 0.3,
                                              min_explained = 0.3 if isolation_mz else 0.7, charges = charges, isolation_mz = isolation_mz)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    df_out.attrs['detected_ms1_trap'] = detected_ms1_trap
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def process_raw_stack(filepath, num_peaks = 1000, ms_level = 2, intensity = False, extract_ms1 = False):
    """function extracting all MS/MS spectra from a Thermo .raw file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .raw file
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | ms_level (int): which MS^n level to extract; default:2
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
   | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, intensity if True, scan number, and activation as in
   | process_mzML_stack; precursor m/z, charge and intensity follow ThermoRawFileParser, which converted the mzML files CandyCrunch was validated on
   """
    table = read_raw_scan_table(filepath)
    peak_ds, rts, intensities, mzs, charges, scans, activations = [], [], [], [], [], [], []
    detected_mode, detected_trap, detected_ms1_trap = None, None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine, ms1_index = [], [], [], [], [], {}
    # One handle for the packets of every scan (opening the file per scan took a quarter of the reading time)
    ms3s, isolations, packets = [], [], open(filepath, 'rb')
    ms2_masters = {scan: master for scan, level, master in zip(table['scan'], table['ms_level'], table['master_scan']) if level == ms_level}
    for i, scan in enumerate(table['scan']):
        if table['ms_level'][i] not in (1, ms_level, ms_level + 1):
            continue
        # Electron-transfer fragment spectra lose the centroids Thermo's own centroid stream leaves out (ThermoRawFileParser's query too): FT side lobes beside
        # intense peaks and recurring ghosts, such as m/z 181.56 at up to the base peak of many EThcD scans, which CandyCrumbs read
        # as a Neu5Gc oxonium ion 2+ (the Byonic glycan told from its isobar Neu5Gc + Fuc in deisotoped EThcD spectra 0.285 -> 0.320). Leaving them out of every
        # MS2 cost GPST000403 and GPST000651 F1 0.018 and 0.016, and no glycan .raw file has electron-transfer scans
        peak_mzs, peak_ints, profile = read_raw_packet(packets, table['packet'][i], drop_flagged = '@etd' in (table['filter_string'][i] or ''))
        # Ion-trap profile scans come without centroids. Like Thermo's centroider, which made the centroids of mzML files from such runs, profile
        # points below 1 are dropped, the profile is split at the lowest point between maxima and neighboring centroids closer than 0.5 m/z are
        # merged, each carrying the summed signal of its share of the profile at its intensity-weighted m/z. The stored profile axis sits one bin
        # below Thermo's, which centroid_ion_trap expects; orbitrap scans always come with centroids
        if profile:
            if table['scan_mode'][i] != 'profile' or table['analyzer'][i] == 'FTMS':
                continue
            peak_mzs, peak_ints = centroid_ion_trap(peak_mzs + (np.median(np.diff(peak_mzs)) if len(peak_mzs) > 1 else 0), peak_ints).T
        if not len(peak_mzs):
            continue
        if table['ms_level'][i] == 1:
            detected_ms1_trap = detected_ms1_trap or {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(table['analyzer'][i])
            order = np.argsort(peak_mzs, kind = 'stable')
            ms1_index[scan] = len(ms1_rts)
            ms1_rts.append(table['retention_time'][i])
            ms1_mzs.append(peak_mzs[order].astype(np.float32))
            ms1_ints.append(peak_ints[order].astype(np.float32))
            continue
        filt = table['filter_string'][i] or ''
        # The filter string ends with this stage's isolation m/z and activation(s), e.g. 1022.47@etd50.00@hcd25.00
        stage = re.findall(r'([\d.]+)((?:@[a-z]+[\d.]+)+)', filt)
        if table['ms_level'][i] == ms_level + 1:
            # An MS3 scan (e.g., ms3 492.99@cid35.00 394.92@cid35.00) fragments a fragment of the latest MS2 scan isolating its first m/z
            row = next((r for r, iso in reversed(isolations) if len(stage) > 1 and abs(iso - float(stage[-2][0])) < 0.011), None)
            if row is not None:
                top = np.argsort(-peak_ints, kind = 'stable')[:num_peaks]
                ms3s[row].append((float(stage[-1][0]), dict(zip(peak_mzs[top].tolist(), peak_ints[top].tolist()))))
            continue
        precursor_mz = table['precursor_mz'][i]
        if not precursor_mz:
            continue
        # precursor_mz is the instrument's monoisotopic m/z where it determined one; like ThermoRawFileParser, the isolation m/z replaces it if a
        # firmware bug put it outside the isolation window
        iso, half = float(stage[-1][0]) if stage else precursor_mz, (table['isolation_width'][i] or 0) / 2
        if not ((iso - 3 <= precursor_mz <= iso + 2.5) if half <= 2 else (iso - half <= precursor_mz <= iso + half)):
            precursor_mz = iso
        if detected_mode is None:
            detected_mode = 'negative' if table['polarity'][i] == '-' else 'positive'
            detected_trap = {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(table['analyzer'][i])
        top = np.argsort(-peak_ints, kind = 'stable')[:num_peaks]
        peak_ds.append(dict(zip(peak_mzs[top].tolist(), peak_ints[top].tolist())))
        isolations.append((len(mzs), iso))
        ms3s.append([])
        mzs.append(precursor_mz)
        rts.append(table['retention_time'][i])
        scans.append(scan)
        methods = re.findall(r'@([a-z]+)', stage[-1][1]) if stage else []
        etd = 'etd' in methods
        # Supplemental activation without a named method (ETD + sa) is ambiguous, as in process_mzXML_stack
        activations.append('EThcD' if etd and 'hcd' in methods else 'ETciD' if etd and 'cid' in methods else None if etd and ' sa ' in filt else
                           'ETD' if etd else 'ECD' if 'ecd' in methods else 'HCD' if 'hcd' in methods else 'CID' if 'cid' in methods else None)
        raw_charge = table['charge'][i]
        ms1_scans.append(len(ms1_rts) - 1)
        refine.append(raw_charge is None)
        # The trailer only carries a charge state the instrument determined (else 0, read as None), so a charge 1 is kept as in process_mzML_stack
        charges.append(raw_charge)
        if intensity:
            # ThermoRawFileParser's precursor intensity: summed centroids within 1.5 m/z of the isolation m/z in the trailer's master scan (tribrids
            # acquire MS2 in parallel with the next survey scan), else in the preceding survey scan; LTQ trailers only carry a 'Master Index:', no scan
            # a product-triggered scan (EThcD of a glycopeptide whose HCD scan showed oxonium ions) has that MS2 scan as its master, whose master is the survey scan
            master = table['master_scan'][i]
            while master not in ms1_index and master in ms2_masters:
                master = ms2_masters[master]
            s = ms1_index.get(master, len(ms1_rts) - 1)
            intensities.append(
                float(ms1_ints[s][(ms1_mzs[s] >= iso - 1.5) & (ms1_mzs[s] < iso + 1.5)].sum()) if s >= 0 else np.nan)
    packets.close()
    df_out = pd.DataFrame({
        'm/z': mzs,
        'peak_d': peak_ds,
        'RT': rts,
        'precursor_charge': charges,
    })
    if intensity:
        df_out['intensity'] = intensities
    df_out['scan'], df_out['activation'] = scans, activations
    if any(ms3s):
        df_out['ms3'] = ms3s
    # Same flat MS1 store and precursor refinement as process_mzML_stack
    ms1 = (np.array(ms1_rts), np.concatenate(ms1_mzs) if ms1_mzs else np.zeros(0, np.float32),
           np.concatenate(ms1_ints) if ms1_ints else np.zeros(0, np.float32),
           np.concatenate([[0], np.cumsum([len(m) for m in ms1_mzs], dtype = np.int64)]))
    # The header charge of a multiply charged precursor comes with an ion trap's own monoisotopic pick, which is checked against the survey scan
    walk = [c in (2, 3) and detected_ms1_trap == 'linear' for c in charges]
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, [r or w for r, w in zip(refine, walk)],
                                              charges = [c if w else None for c, w in zip(charges, walk)])
    if detected_ms1_trap == 'orbitrap':
        # Thermo orbitrap monoisotopic picks of glycans above ~2000 Da walk their isotopes as in process_mzML_stack
        walked, _ = refine_precursor_mz(ms1, list(df_out['m/z']), rts, ms1_scans, [bool(c) and m * c >= 2000 for m, c in zip(df_out['m/z'], charges)],
                                        mz_tolerance = 0.02, isotope_tolerance = 0.02, charges = charges, isolation_mz = True, max_lighter = 2)
        df_out['m/z'] = [w if c and abs(w - m) > ISOTOPE_SPACING / c / 2 else m for w, m, c in zip(walked, df_out['m/z'], charges)]
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    df_out.attrs['detected_ms1_trap'] = detected_ms1_trap
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def process_tdf_stack(filepath, num_peaks = 1000, intensity = False, extract_ms1 = False):
    """function extracting all MS/MS spectra from a Bruker timsTOF .d folder (DDA-PASEF, with an analysis.tdf)\n
   | Arguments:
   | :-
   | filepath (string): absolute path to the .d folder
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
   | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, intensity if True, scan (the precursor's Id), and
   | activation as in process_mzML_stack, one spectrum per precursor (its PASEF scans summed); survey frames are summed over their ion mobility
   | scans; both are centroided with centroid_tof, on m/z from the file's own TOF calibration (MzCalibration)
   """
    try:
        from compression.zstd import decompress as zstd_decompress
    except ImportError:
        from zstandard import decompress as zstd_decompress
    tdf = os.path.join(filepath, 'analysis.tdf')
    if not os.path.exists(tdf):
        raise ValueError(f"{os.path.basename(os.path.normpath(filepath))} has no analysis.tdf. CandyCrunch reads timsTOF .d folders directly; other Bruker "
                         f"data (.tsf without ion mobility, .baf of amaZon, maXis, compact) has to be converted to mzML or mzXML with msconvert first.")
    # Read-only, so nothing is written next to the data (journal files) and a file the acquisition software still holds open can be read
    db = sqlite3.connect(pathlib.Path(tdf).resolve().as_uri() + '?mode=ro', uri = True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if 'Precursors' not in tables or not db.execute('SELECT COUNT(*) FROM Precursors').fetchone()[0]:
            raise ValueError(f"{os.path.basename(os.path.normpath(filepath))} has no DDA-PASEF precursors; CandyCrunch needs data-dependent MS2 spectra "
                             f"(data-independent acquisition such as diaPASEF has none).")
        compression = dict(db.execute('SELECT Key, Value FROM GlobalMetadata').fetchall()).get('TimsCompressionType')
        if compression not in ('1', '2'):
            raise ValueError(f"{os.path.basename(os.path.normpath(filepath))} stores its frames in a format CandyCrunch does not know (compression type "
                             f"{compression}); convert it to mzML with msconvert first.")
        tims_ids = dict(db.execute('SELECT Id, TimsId FROM Frames').fetchall())
        frames = {i: (t, p, k, c, t1, t2, n) for i, t, p, k, c, t1, t2, n in db.execute('SELECT Id, Time, Polarity, MsMsType, MzCalibration, T1, T2, NumPeaks FROM Frames')}
        calibrations = {row[0]: row[1:] for row in db.execute('SELECT Id, ModelType, DigitizerTimebase, DigitizerDelay, T1, T2, dC1, dC2, C0, C1, C2, C3, C4 FROM MzCalibration')}
        precursors = db.execute('SELECT Id, LargestPeakMz, MonoisotopicMz, Charge, Intensity, Parent FROM Precursors').fetchall()
        pasef = db.execute('SELECT Frame, ScanNumBegin, ScanNumEnd, Precursor, IsolationMz, IsolationWidth FROM PasefFrameMsMsInfo ORDER BY Frame').fetchall()
    finally:
        db.close()
    if any(cal[0] != 1 or cal[10] for cal in calibrations.values()):
        raise ValueError(f"{os.path.basename(os.path.normpath(filepath))} uses a TOF calibration model CandyCrunch does not know; convert it to mzML with msconvert first.")
    def tof_to_mz(tof, frame_id):
        # Bruker's TOF calibration (model 1, reproduces timsdata to 0.03 ppm): flight time t = C0 + sqrt(1e12 / C1 * (m/z + C4)) + C2 * m/z, in ns from the
        # digitizer index, then a linear temperature correction. Without it (an evenly spaced sqrt(m/z) axis over the acquisition range) m/z were up to 130 ppm low
        _, _, _, cal_id, t1, t2, _ = frames[frame_id]
        _, timebase, delay, t1_ref, t2_ref, dc1, dc2, c0, c1, c2, _, c4 = calibrations[cal_id]
        a, q = np.sqrt(1e12 / c1), np.asarray(tof, dtype = np.float64) * timebase + delay - c0 + c2 * c4
        x = 2 * q / (a + np.sqrt(a * a + 4 * c2 * q))
        return (x * x - c4) / (1 + dc1 * 1e-6 * (t1 - t1_ref) + dc2 * 1e-6 * (t2 - t2_ref))
    polarity = next((p for _, p, _, _, _, _, _ in frames.values() if p), None)
    detected_mode = {'+': 'positive', '-': 'negative'}.get(polarity)

    def read_frame(frame_id):
        # A frame in analysis.tdf_bin is its byte size and number of scans, then (compression type 2) a zstd block of 32-bit values stored as four byte
        # planes: the number of scans, twice the peak count of every scan but the last, and per peak its TOF index step from the previous peak of its scan
        # (the first from -1) and its intensity; or (type 1) the byte offsets of the scans and per scan an LZF block of 32-bit values, each an intensity
        # at the current TOF index (then the next one) if >= 0, else a skip of that many indices. Returns the scans' offsets into the peaks, the peaks'
        # TOF indices and intensities
        tdf_bin.seek(tims_ids[frame_id])
        size, num_scans = struct.unpack('<2I', tdf_bin.read(8))
        blob = tdf_bin.read(size - 8)
        if compression == '2':
            values = np.ascontiguousarray(np.frombuffer(zstd_decompress(blob), np.uint8).reshape(4, -1).T).view('<u4').ravel()
            pairs = values[num_scans:].reshape(-1, 2).astype(np.int64)
            offsets = np.concatenate([[0], np.cumsum(values[1:num_scans] // 2, dtype = np.int64), [len(pairs)]])
            steps = np.concatenate([[0], np.cumsum(pairs[:, 0])])
            return offsets, steps[1:] - np.repeat(steps[offsets[:-1]], np.diff(offsets)) - 1, pairs[:, 1].astype(np.float64)
        tofs, ints = [], []
        scan_bytes = np.frombuffer(blob, '<u4', num_scans + 1).astype(np.int64) - 8
        for begin, end in zip(scan_bytes[:-1].tolist(), scan_bytes[1:].tolist()):
            # LZF: a control byte below 32 is followed by that many literal bytes plus one, else it copies (control >> 5) + 2 bytes (7: plus the next
            # byte) from (control & 31) * 256 + the next byte + 1 bytes back, which may overlap the bytes it writes
            out, i = bytearray(), begin
            while i < end:
                c = blob[i]
                if c < 32:
                    out += blob[i + 1:i + c + 2]
                    i += c + 2
                    continue
                length = c >> 5
                if length == 7:
                    i += 1
                    length += blob[i]
                start = len(out) - ((c & 31) << 8) - blob[i + 1] - 1
                i += 2
                if start + length + 2 <= len(out):
                    out += out[start:start + length + 2]
                else:
                    for k in range(start, start + length + 2):
                        out.append(out[k])
            values = np.frombuffer(out, '<i4').astype(np.int64)
            steps = np.where(values < 0, -values, 1)
            tofs.append((np.cumsum(steps) - steps)[values >= 0])
            ints.append(values[values >= 0].astype(np.float64))
        offsets = np.concatenate([[0], np.cumsum([len(t) for t in tofs], dtype = np.int64)])
        return offsets, np.concatenate(tofs) if tofs else np.zeros(0, np.int64), np.concatenate(ints) if ints else np.zeros(0)

    # Frames without peaks are never read
    tdf_bin = open(os.path.join(filepath, 'analysis.tdf_bin'), 'rb')
    ms1_ids = sorted(i for i, f in frames.items() if f[2] == 0)
    ms1_rts, ms1_mzs, ms1_ints = [], [], []
    for i in ms1_ids:
        peaks = np.zeros((0, 2))
        if frames[i][6]:
            _, tof, ints = read_frame(i)
            # Summed per TOF index over the frame's mobility scans (np.unique is 4x faster than a bincount over all ~440k indices, with the same sums)
            tof, inverse = np.unique(tof, return_inverse = True)
            peaks = centroid_tof(tof_to_mz(tof, i), np.bincount(inverse, weights = ints), max_peaks = TIMS_MS1_PEAKS)
        ms1_rts.append(frames[i][0] / 60)
        ms1_mzs.append(peaks[:, 0].astype(np.float32))
        ms1_ints.append(peaks[:, 1].astype(np.float32))
    ms1 = (np.array(ms1_rts), np.concatenate(ms1_mzs) if ms1_mzs else np.zeros(0, np.float32),
           np.concatenate(ms1_ints) if ms1_ints else np.zeros(0, np.float32),
           np.concatenate([[0], np.cumsum([len(m) for m in ms1_mzs], dtype = np.int64)]))
    # Each precursor's MS2 spectrum is the sum of its ion mobility scan ranges in all PASEF frames that selected it
    ms2_tofs, ms2_ints, ms2_frames, isolation = defaultdict(list), defaultdict(list), {}, {}
    for frame_id, group in groupby(pasef, key = lambda x: x[0]):
        if frame_id not in frames or not frames[frame_id][6]:
            continue
        offsets, tofs, ints = read_frame(frame_id)
        for _, begin, end, pid, iso_mz, iso_width in group:
            isolation.setdefault(pid, (iso_mz, iso_width))
            begin, end = offsets[min(begin, len(offsets) - 1)], offsets[min(end, len(offsets) - 1)]
            ms2_tofs[pid].append(tofs[begin:end])
            ms2_ints[pid].append(ints[begin:end])
            ms2_frames.setdefault(pid, frame_id)
    tdf_bin.close()
    peak_ds, rts, mzs, charges, scans, ms1_scans = [], [], [], [], [], []
    for pid, largest, mono, charge, _, parent in precursors:
        if pid not in ms2_tofs:
            continue
        tof, inverse = np.unique(np.concatenate(ms2_tofs[pid]), return_inverse = True)
        peaks = centroid_tof(tof_to_mz(tof, ms2_frames[pid]), np.bincount(inverse, weights = np.concatenate(ms2_ints[pid])))
        # The unfragmented precursor envelope (and anything else co-isolated) is often the base peak of a PASEF spectrum, while ion trap CID, which
        # the model was trained on, leaves none: peaks inside the isolation window or on the precursor's isotopes above it are not fragments
        iso_mz, iso_width = isolation[pid]
        upper = max(iso_mz + iso_width / 2, (mono + 6 * ISOTOPE_SPACING / charge + 0.05) if mono and charge else 0)
        peaks = peaks[(peaks[:, 0] < iso_mz - iso_width / 2) | (peaks[:, 0] > upper)]
        if not len(peaks):
            continue
        # Bruker only writes a monoisotopic m/z and charge it determined; without them the precursor is its largest isotope peak, which gets refined
        # from the survey frame like other undetermined precursors
        frame_id = parent if parent in frames else ms2_frames[pid]
        order = np.argsort(peaks[:, 1], kind = 'stable')[::-1][:min(num_peaks, TIMS_MS2_PEAKS)]
        peak_ds.append(dict(zip(peaks[order, 0].tolist(), peaks[order, 1].tolist())))
        mzs.append(float(mono) if mono and charge else float(largest))
        charges.append(int(charge) if mono and charge else None)
        rts.append(frames[frame_id][0] / 60)
        scans.append(pid)
        ms1_scans.append(bisect.bisect_right(ms1_ids, frame_id) - 1)
    df_out = pd.DataFrame({'m/z': mzs, 'peak_d': peak_ds, 'RT': rts, 'precursor_charge': charges})
    df_out['m/z'], drop, intensities = refine_tof_precursors(ms1, mzs, rts, ms1_scans, charges)
    if intensity:
        df_out['intensity'] = intensities
    df_out['scan'], df_out['activation'] = scans, ['CID'] * len(df_out)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = 'other'
    df_out.attrs['detected_ms1_trap'] = 'tof'
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def process_yep_stack(filepath, num_peaks = 1000, intensity = False, extract_ms1 = False):
    """function extracting all MS/MS spectra from a Bruker ion trap .d folder (esquire, HCT, amaZon; with an Analysis.yep)\n
   | Arguments:
   | :-
   | filepath (string): absolute path to the .d folder
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
   | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, intensity if True, scan number, and activation as in
   | process_mzML_stack, from the line (centroided) spectra the instrument stores next to its profile spectra
   """
    with open(os.path.join(filepath, 'Analysis.yep'), 'rb') as f:
        data = f.read()
    if not data.startswith(b'Yellop'):
        raise ValueError(f"{os.path.basename(os.path.normpath(filepath))} has no readable Analysis.yep; convert it to mzML or mzXML with msconvert first.")
    # Analysis.yep is a sequence of pages (FA 0B 00 EE, length, 20 more header bytes) of records: FA 0B DB DB (zlib body) or DA DA, body length,
    # padding after the 16-byte trailer, record type, record number, then a 24-byte key whose 4th and 5th fields are the spectrum number and its kind
    # (2000 the line spectrum, 2010 the profile). Type 316 bodies are spectrum headers, type 315 bodies the spectra, and the other type 315 bodies hold
    # the definitions of the acquisition parameters every spectrum lists by number
    headers, bodies, definitions = {}, {}, b''
    p = data.find(b'\xfa\x0b\x00\xee')
    while p >= 0 and data[p:p + 4] == b'\xfa\x0b\x00\xee':
        end, j = p + 4 + struct.unpack_from('<I', data, p + 4)[0], p + 28
        while j + 48 <= end and data[j:j + 2] == b'\xfa\x0b' and data[j + 2:j + 4] in (b'\xdb\xdb', b'\xda\xda'):
            length, padding, record = struct.unpack_from('<3I', data, j + 4)
            key = struct.unpack_from('<6I', data, j + 24)
            if data[j + 2] == 0xdb and record in (315, 316) and key[4] == 2000:
                (headers if record == 316 else bodies)[key[3]] = zlib.decompress(data[j + 48:j + 48 + length])
            elif data[j + 2] == 0xdb and record == 315 and key[4] != 2010:
                definitions += zlib.decompress(data[j + 48:j + 48 + length])
            j += 64 + length + padding
        p = end
    # A parameter definition is its number and type, then its short and long name as strings (marker 1D 27 AB BF, length incl. the final 0)
    param_ids = {}
    for name in (b'iIonNeg', b'iTrgChrgAto02'):
        i = definitions.find(name + b'\x00')
        param_ids[name] = struct.unpack_from('<I', definitions, i - 16)[0] if i >= 16 else None
    ms1_rts, ms1_mzs, ms1_ints = [], [], []
    peak_ds, rts, mzs, charges, scans, ms1_scans, intensities = [], [], [], [], [], [], []
    detected_mode = None
    for number in sorted(set(headers) & set(bodies)):
        header, body = headers[number], bodies[number]
        # Header: RT in 64 us ticks, TIC, spectrum number, scan range, ..., the size of the precursor list (4 bytes per MS level above 1), then the
        # precursor m/z (Bruker's isolation target, not the monoisotopic peak)
        ms_level = struct.unpack_from('<I', header, 44)[0] // 4 + 1
        rt = struct.unpack_from('<i', header, 0)[0] * 64e-6 / 60
        # The body is a chain of blocks (16-bit length, 32-bit type, 16-bit signature A0-A2 BF; type 8 lists the parameters as number and 4-byte value),
        # and the line spectrum's (m/z, intensity) float32 pairs follow the last block
        # The chain starts with a type 2 block, signed A2 BF in amaZon files and A1 BF in HCT files, whose bodies put 80 bytes with an A0 BF signature before it;
        # Agilent's LC/MSD Trap XCT (a Bruker-built ion trap, GPST000153) starts it with a type 10 block signed A1 BF and has no type 2 block, so it read no spectra
        start, values = re.search(rb'\x02\x00\x00\x00[\xa1\xa2]\xbf', body) or re.search(rb'\x0a\x00\x00\x00\xa1\xbf', body), {}
        j = start.start() - 2 if start else len(body)
        while j + 8 <= len(body):
            size, block, signature = struct.unpack_from('<HIH', body, j)
            if signature not in (0xbfa0, 0xbfa1, 0xbfa2) or size < 8:
                break
            if block == 8:
                values = dict(struct.iter_unpack('<I4s', body[j + 12:j + 12 + 8 * struct.unpack_from('<I', body, j + 8)[0]]))
            j += size
        peaks = np.frombuffer(body[j:j + (len(body) - j) // 8 * 8], dtype = '<f4').reshape(-1, 2)
        if detected_mode is None and param_ids[b'iIonNeg'] in values:
            detected_mode = 'negative' if struct.unpack('<i', values[param_ids[b'iIonNeg']])[0] else 'positive'
        if ms_level == 1:
            ms1_rts.append(rt)
            ms1_mzs.append(peaks[:, 0].astype(np.float32))
            ms1_ints.append(peaks[:, 1].astype(np.float32))
        elif ms_level == 2 and len(peaks):
            order = np.argsort(peaks[:, 1], kind = 'stable')[::-1][:num_peaks]
            peak_ds.append(dict(zip(peaks[order, 0].astype(float).tolist(), peaks[order, 1].astype(float).tolist())))
            mzs.append(float(struct.unpack_from('<f', header, 48)[0]))
            rts.append(rt)
            # Bruker's trigger charge (0: none determined); 1 counts as undetermined as for Bruker mzXML (GPST000316's [M-2H]2- disialyl core 2 had 1)
            charge = struct.unpack('<i', values[param_ids[b'iTrgChrgAto02']])[0] if param_ids[b'iTrgChrgAto02'] in values else 0
            charges.append(charge if charge > 1 else None)
            scans.append(number)
            ms1_scans.append(len(ms1_rts) - 1)
            # As for Bruker mzXML, the precursor intensity is the survey scan's centroids around the precursor
            intensities.append(float(ms1_ints[-1][(ms1_mzs[-1] >= mzs[-1] - 1.5) & (ms1_mzs[-1] < mzs[-1] + 1.5)].sum()) if ms1_mzs else np.nan)
    df_out = pd.DataFrame({'m/z': mzs, 'peak_d': peak_ds, 'RT': rts, 'precursor_charge': charges})
    if intensity:
        df_out['intensity'] = intensities
    df_out['scan'], df_out['activation'] = scans, ['CID'] * len(df_out)
    ms1 = (np.array(ms1_rts), np.concatenate(ms1_mzs) if ms1_mzs else np.zeros(0, np.float32),
           np.concatenate(ms1_ints) if ms1_ints else np.zeros(0, np.float32),
           np.concatenate([[0], np.cumsum([len(m) for m in ms1_mzs], dtype = np.int64)]))
    # The precursor m/z is the isolation target, so every precursor is refined from its survey scan as for Bruker HCT mzXML
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, [True] * len(mzs), mz_tolerance = 0.6, min_explained = 0.3, charges = charges,
                                              isolation_mz = True)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    # The definitions name the instrument; HCT and esquire count as linear ion traps, as for their mzXML
    df_out.attrs['detected_trap'] = df_out.attrs['detected_ms1_trap'] = 'amazon' if b'amaZon' in definitions else 'linear'
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def average_dicts(dicts, mode = 'mean', round_dp = False):
    """averages a list of dictionaries containing spectra\n
   | Arguments:
   | :-
   | dicts (list): list of dictionaries of form (fragment) m/z : intensity
   | mode (string): whether to average by mean or by max; default:'mean'
   | round_dp (int): if given, m/z values are rounded to this many decimals before averaging, so close peaks pool; default:False\n
   | Returns:
   | :-
   | Returns a single dictionary of form (fragment) m/z : intensity
   """
    result = defaultdict(list)
    for d in dicts:
        for mass, intensity in d.items():
            result[np.round(mass, round_dp) if round_dp else mass].append(intensity)
    return {mass: np.mean(intensities) if mode == 'mean' else max(intensities) for mass, intensities in result.items()}


def bin_intensities(peak_d, frames):
    """sums up intensities for each bin across a spectrum\n
   | Arguments:
   | :-
   | peak_d (dict): dictionary of form (fragment) m/z : intensity
   | frames (list): m/z boundaries separating each bin\n
   | Returns:
   | :-
   | (1) an array of binned intensities
   | (2) an array of the m/z of the most intense peak in each bin minus the bin's lower edge (0 for empty bins)
   """
    num_frames = len(frames)
    binned_intensities = np.zeros(num_frames)
    mz_diff = np.zeros(num_frames)
    mzs = np.array(list(peak_d.keys()), dtype = 'float32')
    intensities = np.array(list(peak_d.values()))
    # Peaks outside the binned range would wrap into the last bin (index -1) with a remainder of about -2960
    in_range = (mzs > frames[0]) & (mzs <= frames[-1])
    mzs, intensities = mzs[in_range], intensities[in_range]
    if not len(mzs):
        return binned_intensities, mz_diff
    bin_indices = np.digitize(mzs, frames, right = True)
    mz_remainder = mzs - frames[bin_indices - 1]
    order = np.argsort(bin_indices, kind = 'stable')
    unique_bins, starts = np.unique(bin_indices[order], return_index = True)
    max_intensities = np.maximum.reduceat(intensities[order], starts)
    mz_remainder = mz_remainder * (intensities == max_intensities[np.searchsorted(unique_bins, bin_indices)])
    summed_intensities = np.add.reduceat(intensities[order], starts)
    max_mz_remainder = np.maximum.reduceat(mz_remainder[order], starts)
    binned_intensities[unique_bins - 1] = summed_intensities
    mz_diff[unique_bins - 1] = max_mz_remainder
    return binned_intensities, mz_diff


def ms1_mz_calibration(ms1, target_mzs, rt_centers, charges, mz_tolerance = 0.5, accurate = False):
    """estimates the MS1 m/z offset and spread from the most intense centroid near each singly charged precursor\n
    | Arguments:
    | :-
    | ms1 (tuple): flat MS1 data from process_mzML_stack, (rts sorted ascending, mzs, intensities, scan offsets)
    | target_mzs (array-like): (theoretical) precursor m/z values
    | rt_centers (array-like): retention time of each precursor
    | charges (array-like): charge of each precursor; only singly charged ones are used, as low-resolution MS1 merges the isotopes of the others
    | mz_tolerance (float): m/z tolerance within which to look for the centroid; default:0.5
    | accurate (bool): whether survey scans resolve the isotopes of every charge state (orbitrap, timsTOF); with fewer than 5 usable singly charged precursors,
    |                  all precursors then count, each by its most intense centroid within 20 ppm; default:False\n
    | Returns:
    | :-
    | Returns a tuple of (median m/z offset, half-width covering the m/z spread), or (0, mz_tolerance) with fewer than 5 usable precursors
    """
    ms1_rts, ms1_mzs, ms1_ints, ms1_offsets = ms1
    target_mzs, charges = np.asarray(target_mzs, dtype = np.float64), np.abs(np.asarray(charges, dtype = np.float64))
    nearest = np.clip(np.searchsorted(ms1_rts, np.asarray(rt_centers, dtype = np.float64)), 0, len(ms1_rts) - 1)
    # Runs of mostly multiply charged precursors (N-glycans, permethylated glycans in positive mode) fell back to (0, mz_tolerance), whose contiguous isotope
    # windows sum every ion within two m/z above an accurate survey scan's precursor; there, every charge state calibrates, within 20 ppm so no 13C peak counts
    for narrow in ((False, True) if accurate else (False,)):
        offsets = []
        for t in (range(len(target_mzs)) if narrow else np.where(charges <= 1)[0]):
            s0, s1 = ms1_offsets[nearest[t]], ms1_offsets[nearest[t] + 1]
            tol = 2e-5 * target_mzs[t] if narrow else mz_tolerance
            lo, hi = np.searchsorted(ms1_mzs[s0:s1], target_mzs[t] - tol), np.searchsorted(ms1_mzs[s0:s1], target_mzs[t] + tol)
            if hi > lo:
                offsets.append(ms1_mzs[s0 + lo + np.argmax(ms1_ints[s0 + lo:s0 + hi])] - target_mzs[t])
        if len(offsets) >= 5:
            break
    if len(offsets) < 5:
        return 0.0, mz_tolerance
    shift = float(np.median(offsets))
    return shift, min(mz_tolerance, 4 * 1.4826 * float(np.median(np.abs(np.array(offsets) - shift))))


def extract_xic_areas(ms1, target_mzs, rt_centers, charges = None, rt_window = 1.0, mz_tolerance = 0.5, search_window = 0,
                      isotopes = 3, weights = None, mz_calibration = None, glycans = None, sample_prep = 'underivatized', other_charges = None, mode = 'negative',
                      details = None):
    """integrates the MS1 isotope envelope of each precursor over its own chromatographic peak\n
    | Arguments:
    | :-
    | ms1 (tuple): flat MS1 data from process_mzML_stack, (rts sorted ascending, mzs, intensities, scan offsets)
    | target_mzs (array-like): monoisotopic precursor m/z values to extract XICs for
    | rt_centers (array-like): retention time of each precursor (MS2 or consensus RT)
    | charges (array-like): charge of each precursor, which sets the isotope spacing; default:None (all singly charged)
    | rt_window (float): maximal distance (in minutes) of a peak boundary from the apex; default:1.0
    | mz_tolerance (float): largest m/z half-width of an isotope window; default:0.5
    | search_window (float): if > 0, the apex is the highest point within this many minutes of rt_center and its genuineness is checked (MS1 gap filling), else the apex is reached by climbing uphill from rt_center; default:0
    | isotopes (int): number of isotope peaks summed on top of the monoisotopic one; default:3
    | weights (array-like): if given, precursors that land on the same peak (same apex and m/z) split its area in proportion to these instead of each claiming all of it; default:None
    | mz_calibration (tuple): (m/z offset, half-width) from ms1_mz_calibration; default:None (estimated from the targets)
    | glycans (list): if given, each area is divided by the share of the glycan's molecules within the summed isotope peaks (computed from its elemental formula; None entries are interpolated by mass); a composition or an element formula counts as the whole molecule, e.g., a glycopeptide as get_formula('H5N4', peptide = 'EEQYNSTYR'); default:None
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated, for the elemental formula; default:'underivatized'
    | other_charges (list): if given, per target the other charge states of its molecule whose envelopes are added if they co-elute with it (over its peak); default:None
    | mode (string): ion mode, which sets the charge carrier of other_charges; default:'negative'
    | details (list): if given, one dict per target is appended that shows how its area was integrated: the summed envelope trace ('times', 'trace', from
    |                 rt_window before to rt_window after the scans searched), 'left'/'apex'/'right' as indices into it (-1 if not integrated), 'baseline',
    |                 'area' (before 'scale', the area split and isotope share applied after), the m/z 'windows', the survey scan at the apex ('scan'), and
    |                 'others' (other_charges with their traces, r and whether they were added); default:None\n
    | Returns:
    | :-
    | Returns an array of integrated XIC areas, one per target m/z; with search_window, a tuple of (areas, apex RTs, apex intensities, whether the apex is a genuine peak)
    """
    ms1_rts, ms1_mzs, ms1_ints, ms1_offsets = ms1
    target_mzs = np.asarray(target_mzs, dtype = np.float64)
    rt_centers = np.asarray(rt_centers, dtype = np.float64)
    charges = np.ones(len(target_mzs)) if charges is None else np.maximum(np.abs(np.asarray(charges, dtype = np.float64)), 1)
    shift, half_width = mz_calibration if mz_calibration is not None else ms1_mz_calibration(ms1, target_mzs, rt_centers, charges, mz_tolerance = mz_tolerance)
    spacing = ISOTOPE_SPACING / charges
    # Where the MS1 m/z spread is a sizeable part of the isotope spacing (e.g., ion traps, which also merge the isotopes of multiply charged ions), the isotope windows touch; otherwise each isotope gets its own narrow window
    widths = np.where(half_width >= 0.2 * spacing, 0.5 * spacing, np.maximum(half_width, 5e-6 * target_mzs))
    # Binary search for RT window boundaries instead of boolean masking all scans per target
    scan_starts = np.searchsorted(ms1_rts, rt_centers - search_window - rt_window)
    scan_ends = np.searchsorted(ms1_rts, rt_centers + search_window + rt_window, side = 'right')
    areas, apex_rts, apex_ints = np.zeros(len(target_mzs)), np.full(len(target_mzs), np.nan), np.zeros(len(target_mzs))
    is_peak, apex_scans = np.zeros(len(target_mzs), dtype = bool), np.full(len(target_mzs), -1)
    # Each survey scan as its m/z values and cumulative intensities, so the intensity within an m/z window is one difference
    cumulated = lambda first, last: [(ms1_mzs[ms1_offsets[si]:ms1_offsets[si + 1]], np.concatenate([[0], np.cumsum(ms1_ints[ms1_offsets[si]:ms1_offsets[si + 1]], dtype = np.float64)]))
                                     for si in range(first, last)]
    records = [{'mz': float(target_mzs[t]), 'charge': int(charges[t]), 'rt': float(rt_centers[t]), 'times': np.zeros(0, dtype = np.float32), 'trace': np.zeros(0, dtype = np.float32),
                'left': -1, 'apex': -1, 'right': -1, 'baseline': 0.0, 'area': 0.0, 'windows': [], 'scan': None, 'others': []} for t in range(len(target_mzs))] if details is not None else None
    for t in range(len(target_mzs)):
        s0, s1 = scan_starts[t], scan_ends[t]
        if s1 - s0 < 3:
            continue
        rts = ms1_rts[s0:s1]
        scans = cumulated(s0, s1)
        xic = lambda centers, width, scans = scans: np.array([np.sum(cum[np.searchsorted(mzs, centers + width)] - cum[np.searchsorted(mzs, centers - width)]) for mzs, cum in scans])
        ints_arr = xic(target_mzs[t] + shift + spacing[t] * np.arange(isotopes + 1), widths[t])
        # Smoothing (sigma of 0.05 min, at least 0.7 scans) only steers apex and boundary finding; the area comes from the raw trace. Survey scans
        # sharing a retention time (one per ion mobility scan) gave a zero median step and an infinite sigma, so only real steps count
        steps = np.diff(rts)
        sigma = max(0.7, 0.05 / np.median(steps[steps > 0])) if (steps > 0).any() else 0.7
        kernel = np.exp(-0.5 * (np.arange(-int(3 * sigma), int(3 * sigma) + 1) / sigma) ** 2)
        smooth = np.convolve(np.pad(ints_arr, len(kernel) // 2, mode = 'edge'), kernel / kernel.sum(), mode = 'valid')
        if search_window:
            candidates = np.where(np.abs(rts - rt_centers[t]) <= search_window)[0]
            if not len(candidates):
                continue
            apex = candidates[np.argmax(ints_arr[candidates])]
            # A genuine peak has its apex inside the search window (not the flank of a neighbor), signal in at least 3 consecutive scans, and 3x the median within rt_window of it
            first, last = apex, apex
            while first > 0 and ints_arr[first - 1] > 0:
                first -= 1
            while last < len(ints_arr) - 1 and ints_arr[last + 1] > 0:
                last += 1
            is_peak[t] = (candidates[0] < apex < candidates[-1] and last - first >= 2 and
                          ints_arr[apex] >= 3 * np.median(ints_arr[np.abs(rts - rts[apex]) <= rt_window]))
            apex_rts[t], apex_ints[t] = rts[apex], ints_arr[apex]
        else:
            apex = np.argmin(np.abs(rts - rt_centers[t]))
        # Climb to the top of the peak the precursor sits on, then walk down both flanks to a valley (splitting co-eluting isomers), 1% of the peak height, or rt_window
        while 0 < apex < len(smooth) - 1 and max(smooth[apex - 1], smooth[apex + 1]) > smooth[apex]:
            apex += 1 if smooth[apex + 1] > smooth[apex - 1] else -1
        baseline = np.percentile(ints_arr, 10)
        cut = baseline + 0.01 * (smooth[apex] - baseline)
        left, right = apex, apex
        while left > 0 and rts[apex] - rts[left - 1] <= rt_window and smooth[left - 1] <= smooth[left] and smooth[left] > cut:
            left -= 1
        while right < len(smooth) - 1 and rts[right + 1] - rts[apex] <= rt_window and smooth[right + 1] <= smooth[right] and smooth[right] > cut:
            right += 1
        areas[t], apex_scans[t] = _trapezoid(np.clip(ints_arr[left:right + 1] - baseline, 0, None), rts[left:right + 1]), s0 + apex
        if records is not None:
            # The trace again from rt_window before to rt_window after the scans searched, which shows the neighbouring peaks, and the survey scan at the
            # apex around the envelope; indices point into this view
            v0, v1 = np.searchsorted(ms1_rts, rts[0] - rt_window), np.searchsorted(ms1_rts, rts[-1] + rt_window, side = 'right')
            centers, wide = target_mzs[t] + shift + spacing[t] * np.arange(isotopes + 1), cumulated(v0, v1)
            a0, a1 = ms1_offsets[s0 + apex], ms1_offsets[s0 + apex + 1]
            near = np.abs(ms1_mzs[a0:a1] - centers.mean()) <= (isotopes / 2 + 2) * spacing[t]
            records[t].update(times = ms1_rts[v0:v1].astype(np.float32), trace = xic(centers, widths[t], wide).astype(np.float32), left = int(s0 - v0 + left),
                              apex = int(s0 - v0 + apex), right = int(s0 - v0 + right), baseline = float(baseline), area = float(areas[t]),
                              windows = [(float(c - widths[t]), float(c + widths[t])) for c in centers], scan = (ms1_mzs[a0:a1][near], ms1_ints[a0:a1][near]))
        if other_charges is not None and right - left >= 2:
            # The other charge states of the molecule (those no row of the run reports) elute with it: each adds its envelope over the same peak if its
            # trace follows this one (r >= 0.9), as a glycan's charge states share its abundance (N-glycans ionize mostly as [M-2H]2-, also as [M-H]-)
            neutral = (target_mzs[t] - (-1 if mode == 'negative' else 1) * PROTON_MASS) * charges[t]
            for z in other_charges[t]:
                mz, sp = neutral / z + (-1 if mode == 'negative' else 1) * PROTON_MASS, ISOTOPE_SPACING / z
                width = 0.5 * sp if half_width >= 0.2 * sp else max(half_width, 5e-6 * mz)
                trace = xic(mz + shift + sp * np.arange(isotopes + 1), width)
                smooth_z = np.convolve(np.pad(trace, len(kernel) // 2, mode = 'edge'), kernel / kernel.sum(), mode = 'valid')
                r = np.corrcoef(smooth_z[left:right + 1], smooth[left:right + 1])[0, 1] if np.ptp(smooth_z[left:right + 1]) > 0 and np.ptp(smooth[left:right + 1]) > 0 else None
                area = _trapezoid(np.clip(trace[left:right + 1] - np.percentile(trace, 10), 0, None), rts[left:right + 1]) if r is not None and r >= 0.9 else 0.0
                areas[t] += area
                if records is not None:
                    records[t]['others'].append({'charge': z, 'mz': float(mz), 'trace': xic(mz + shift + sp * np.arange(isotopes + 1), width, wide).astype(np.float32),
                                                 'baseline': float(np.percentile(trace, 10)), 'r': None if r is None else float(r), 'added': bool(r is not None and r >= 0.9),
                                                 'area': float(area)})
    # What each raw area is multiplied with: its share of a peak that several targets landed on, then 1 / its isotope share
    scale = np.ones(len(target_mzs))
    if weights is not None:
        weights = np.nan_to_num(np.asarray(weights, dtype = np.float64))
        shared = [np.where((apex_scans == apex_scans[t]) & (charges == charges[t]) & (np.abs(target_mzs - target_mzs[t]) <= widths[t]))[0] if apex_scans[t] >= 0 else [t] for t in range(len(target_mzs))]
        scale = np.array([weights[t] / weights[same].sum() if weights[same].sum() > 0 else 1 / len(same) for t, same in enumerate(shared)])
        areas = areas * scale
    if glycans is not None:
        # Divide by the share of each glycan's molecules within the summed isotope peaks, from its elemental formula plus any derivatization groups; glycans without a formula are interpolated by mass
        # A composition or element formula is weighed whole by glycowork's get_formula (derivatization included), e.g., a glycopeptide as
        # get_formula('H5N4F1', peptide = 'EEQYNSTYR'); structures go through get_molecular_properties and get their derivatization groups below
        whole = {g: get_formula(g, sample_prep = sample_prep) for g in set(glycans) if isinstance(g, str) and (is_composition(g) or re.fullmatch(r'(?:(?:Cl|Br|Na|[CHNOPSKFI])\d*)+', g))}
        props = get_molecular_properties([g for g in set(glycans) if isinstance(g, str) and g not in whole]) if any(isinstance(g, str) and g not in whole for g in glycans) else pd.DataFrame()
        formulas = (props['molecular_formula'].to_dict() if 'molecular_formula' in props.columns else {}) | whole
        unit = {'permethylated': 'CH2', 'peracetylated': 'C2H2O'}.get(sample_prep)
        share = np.full(len(target_mzs), np.nan)
        for t, g in enumerate(glycans):
            if g not in formulas:
                continue
            counts = defaultdict(int, {el: int(k or 1) for el, k in re.findall(r'([A-Z][a-z]?)(\d*)', formulas[g])})
            if unit and g not in whole:
                comp = get_comp(g)
                n_units = round((composition_to_mass(comp, sample_prep = sample_prep) - composition_to_mass(comp)) / calculate_adduct_mass(unit))
                for el, k in re.findall(r'([A-Z][a-z]?)(\d*)', unit):
                    counts[el] += int(k or 1) * n_units
            dist = np.array([1.0])
            for el, k in counts.items():
                for _ in range(k):
                    dist = np.convolve(dist, ISOTOPE_ABUNDANCES.get(el, [1.0]))[:isotopes + 1]
            share[t] = dist.sum()
        known = ~np.isnan(share)
        if known.any():
            order = np.argsort(target_mzs[known] * charges[known])
            share[~known] = np.interp(target_mzs[~known] * charges[~known], (target_mzs[known] * charges[known])[order], share[known][order])
            areas = areas / share
            scale = scale / share
    if details is not None:
        details.extend({**record, 'scale': float(s)} for record, s in zip(records, scale))
    return (areas, apex_rts, apex_ints, is_peak) if search_window else areas


def process_for_inference(df, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear', rt_max_default = 30.0, tta_thresh = None, keep_inputs = False):
    """processes averaged spectra for them being inputs to CandyCrunch\n
   | Arguments:
   | :-
   | df (dataframe): condensed dataframe from condense_dataframe
   | glycan_class (int or list): 0 = O-linked, 1 = N-linked, 2 = lipid/free, or one such code per row
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | tta_thresh (float): inputs of spectra whose best annotation_score exceeds this get 5 augmented copies, the others one plain copy (column 'tta'); default: None (all get 5)
   | keep_inputs (bool): whether the returned dataframe keeps each row's encoded spectrum, RT and glycan class (binned_intensities, mz_remainder, RT2,
   |                     glycan_type) as the model gets them; default:False\n
   | Returns:
   | :-
   | (1) the model inputs in batches of 256 (5-copy inputs first, then 1-copy inputs, each in input_id order)
   | (2) a preliminary df_out dataframe
   """
    df = df.assign(glycan_type = glycan_class,
                   mode = int(mode == 'negative'),
                   lc = np.select([lc == 'PGC', lc == 'C18'], [0, 1], 2),
                   modification = np.select([modification == 'reduced', modification == 'permethylated'], [0, 1], 2),
                   trap = np.select([trap == 'linear', trap == 'orbitrap', trap == 'amazon'], [0, 1, 2], 3))
    # Retention time normalization
    max_rt = max(max(df['RT'], default = 0), rt_max_default)
    df['RT2'] = df['RT'] / max_rt
    # Candidate structures of one composition share spectrum and composition vector, i.e., the model input, so every distinct input is
    # predicted once (input_id points each row to its input)
    input_ids = {}
    df['input_id'] = [input_ids.setdefault((s, tuple(v), t), len(input_ids)) for s, v, t in zip(df['spec_id'], df['compositional_vector'], df['glycan_type'])]
    df['tta'] = True if tta_thresh is None else (df.groupby('spec_id')['annotation_score'].transform('max') > tta_thresh).values
    df_in = df.drop_duplicates('input_id')
    # Dataloader generation
    X = list(zip(df_in.binned_intensities.values.tolist(), df_in.mz_remainder.values.tolist(),
                 df_in.compositional_vector.values.tolist(), df_in.glycan_type.values.tolist(),
                 df_in.RT2.values.tolist(), df_in['mode'].values.tolist(), df_in.lc.values.tolist(),
                 df_in.modification.values.tolist(), df_in.trap.values.tolist()))
    # Test-time augmentation (5 augmented copies, aggregated by max) only where the prediction can reach the output
    tta = df_in['tta'].values
    # Each input's 5 copies are augmented from a seed taken from the input itself (spectrum, remainders, composition vector), so the augmentation of an
    # input no longer depends on how many inputs came before it: with one global random stream, a change adding or dropping a single candidate row
    # reshuffled the augmentation of every later input (up to 0.02 F1 per dataset), and comparisons needed several seeds. The global random states are
    # left as they were
    state = (random.getstate(), np.random.get_state())
    X_tta = []
    for k in [k for k, t in zip(X, tta) if t]:
        seed = zlib.crc32(b''.join(np.asarray(v, dtype = np.float32).tobytes() for v in k[:3]))
        random.seed(seed)
        np.random.seed(seed)
        X_tta.extend((transform_mz(np.asarray(k[0])), *k[1:4], transform_rt(k[4]), *k[5:]) for _ in range(5))
    random.setstate(state[0])
    np.random.set_state(state[1])
    X = X_tta + [k for k, t in zip(X, tta) if not t]
    # Batches as the model takes them: binned intensities and m/z remainders as two channels, composition vector, glycan type, RT, mode, LC, modification, trap
    batches = [(np.stack([np.asarray([x[0] for x in b], dtype = np.float32), np.asarray([x[1] for x in b], dtype = np.float32)], axis = 1),
                np.asarray([x[2] for x in b], dtype = np.float32), np.asarray([x[3] for x in b], dtype = np.int64), np.asarray([[x[4]] for x in b], dtype = np.float32),
                *[np.asarray([x[c] for x in b], dtype = np.int64) for c in range(5, 9)]) for b in (X[i:i + 256] for i in range(0, len(X), 256))]
    idx_col = 'm/z' if 'm/z' in df.columns else 'reducing_mass'
    df.set_index(idx_col, inplace = True)
    drop_cols = ['mode', 'modification', 'trap', 'lc'] + ([] if keep_inputs else ['binned_intensities', 'mz_remainder', 'RT2', 'glycan_type'])
    df.drop(drop_cols, axis = 1, inplace = True)
    return batches, df


def get_topk(batches, model, glycans, k = 25, temp = False, temperature = temperature):
    """yields topk CandyCrunch predictions for the model inputs in batches\n
   | Arguments:
   | :-
   | batches (list): model inputs in batches, from process_for_inference
   | model (PyTorch or CandyCrunchCNN): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch
   | k (int): how many top predictions to provide for each spectrum; default:25
   | temp (bool): whether to calibrate logits by temperature factor; default:False
   | temperature (tensor): the temperature factor used to calibrate logits; default:1.15\n
   | Returns:
   | :-
   | (1) a nested list of topk glycans for each spectrum
   | (2) a nested list of associated prediction confidences, for each spectrum
   """
    preds, conf = [], []
    # PyTorch runs every model but the ONNX one (also the app's progress counter around a PyTorch model), on the GPU if there is one
    on_torch = torch is not None and not isinstance(model, CandyCrunchCNN)
    for batch in batches:
        if on_torch:
            with torch.inference_mode():
                pred = model(*[torch.from_numpy(x if x.ndim > 1 else x[:, None]).to(device, non_blocking = True) for x in batch])
                if temp:
                    pred = pred / temperature
                conf_topk, idx_topk = torch.topk(F.softmax(pred, dim = 1), k, dim = 1)
            idx, top = idx_topk.cpu().numpy(), conf_topk.cpu().numpy()
        else:
            pred = model(*batch)
            if temp:
                pred = pred / float(temperature)
            pred = np.exp(pred - pred.max(1, keepdims = True))
            pred /= pred.sum(1, keepdims = True)
            # The k most likely structures, most likely first
            idx = np.argpartition(-pred, k - 1, axis = 1)[:, :k]
            idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(pred, idx, 1), axis = 1, kind = 'stable'), 1)
            top = np.take_along_axis(pred, idx, 1)
        preds.extend([glycans[i] for i in j] for j in idx)
        conf.extend(top.astype(float).tolist())
    return preds, conf


comp_cache = {}


def get_comp(glycan):
    if glycan not in comp_cache:
        try:
            comp_cache[glycan] = glycan_to_composition(glycan)
        except Exception:
            comp_cache[glycan] = {}
    return comp_cache[glycan]


def plain_composition(comp):
    """Counts the linkage-resolved sialic acids of a composition (Neu5Ac_a2-6, from a linkage-specific derivatization) as plain ones, as the model and the
    structure maps know compositions"""
    out = {}
    for k, v in comp.items():
        base = next((b for b, keys in SIALIC_KEYS.items() if k in keys), k)
        out[base] = out.get(base, 0) + v
    return out


def fit_sialic_linkages(glycan, comp, scoring = False):
    """A structure with the sialic acid linkages of a linkage-resolved composition, as a linkage-specific derivatization reads them off the precursor mass:
    unknown ones are set where the composition decides them (glycowork's resolve_sialic_linkages), and where the structure states other linkages, the
    composition's replace them, all of them if its sialic acids are of one kind, while a mix of a2-6 and a2-3 leaves them open (a2-3/6), as the mass does
    not say which sits where. With scoring (for CandyCrumbs, whose glycosidic fragment masses do not depend on that), every sialic acid gets one: the stated
    ones as far as the composition has them, the rest its a2-6, then its a2-3 ones in written order. None if its sialic acids cannot have these linkages"""
    resolved = resolve_sialic_linkages(glycan, comp)
    sites = [s for s in _carboxyl_sites(resolved or glycan) if s[0] in SIALIC_KEYS]
    if resolved is not None and (not scoring or all(i is not None for _, i, _, _ in sites)):
        return resolved
    glycan, want = resolved or glycan, {base: [comp.get(k, 0) for k in keys] for base, keys in SIALIC_KEYS.items()}
    if any(len([s for s in sites if s[0] == base]) != sum(n) for base, n in want.items()) or any(span is None for *_, span in sites):
        return None
    new = []
    for base, i, _, (start, end) in sites:
        donor, _, acceptor = glycan[start:end].partition('-')
        lactone = next((o for o in acceptor.split('/') if o and o in '389'), '3')
        if not scoring:
            new.append((start, end, f"{donor}-{lactone if not want[base][0] else '6' if not want[base][1] else '3/6'}"))
        elif i is not None and want[base][i]:
            want[base][i] -= 1
        else:
            new.append((start, end, None, base, lactone, donor))
    if scoring:
        filled = []
        for start, end, _, base, lactone, donor in new:
            kind = not want[base][0]
            want[base][kind] -= 1
            filled.append((start, end, f"{donor}-{lactone if kind else '6'}"))
        new = filled
    for start, end, link in sorted(new, reverse = True):
        glycan = glycan[:start] + link + glycan[end:]
    return glycan


_IONS = {}


def mass_check(mass, glycan, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized', mass_tag = None, double_thresh = 900,
               triple_thresh = 1500, quadruple_thresh = 3500, mass_tolerance = 0.5, permitted_charges = [1, 2, 3, 4], sialic_derivatization = None):
    """determine whether glycan could explain m/z\n
   | Arguments:
   | :-
   | mass (float): observed m/z
   | glycan (string): glycan in IUPAC-condensed nomenclature
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
   | mass_tag (float): label mass to add when calculating possible m/z (the mass of a 'custom' label); default:None
   | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
   | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1500
   | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500
   | mass_tolerance (float): maximum allowed mass difference to return True; default:0.5
   | permitted_charges (list): charges of ions used to check mass against; default:[1,2,3,4]
   | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS); a sialic acid of unknown linkage
   |                                          in glycan may then be either product; default:None\n
   | Returns:
   | :-
   | Returns a list of the ion m/z values of glycan (or of a composition mass) within mass_tolerance of mass (empty if none), or False if its mass cannot be computed
   """
    # The ions of a glycan are the same for every row that asks (a run asks about a few hundred structures thousands of times), so they are kept in _IONS
    try:
        key = (glycan, mode, modification, sample_prep, mass_tag, double_thresh, triple_thresh, quadruple_thresh, max(permitted_charges, default = None),
               sialic_derivatization)
        ions = _IONS.get(key)
    except TypeError:
        key, ions = None, None
    if ions is None:
        try:
            if isinstance(glycan, str):
                mz = glycan_to_mass(glycan, sample_prep = sample_prep, modification = modification)
                if sialic_derivatization is not None:
                    # every product its carboxyls can carry, a sialic acid of unknown linkage being either
                    sites = Counter(base if base == 'HexA' or i is None else SIALIC_KEYS[base][i] for base, i, _, _ in _carboxyl_sites(glycan))
                    mz = np.array([mz + composition_to_mass(c, sialic_derivatization = sialic_derivatization) - composition_to_mass(c) for c in sialic_linkage_splits(sites)])
            else:
                mz = glycan + composition_to_mass({}, sample_prep = sample_prep,
                                                  modification = modification) - composition_to_mass({},
                                                                                                     sample_prep = sample_prep)
        except Exception:
            ions = False
        else:
            ions = get_ion_mzs(mz + (mass_tag or 0), max_charge = int(max(permitted_charges)) * (1 if mode == 'positive' else -1),
                               adducts = get_adduct_list(mode), min_mass = {2: double_thresh, 3: triple_thresh, 4: quadruple_thresh})
        if key is not None:
            if len(_IONS) > 50000:
                _IONS.clear()
            _IONS[key] = ions
    if ions is False:
        return False
    # ion names end in their charge, e.g., '[M-H]-' or '[M+Acetate-H]2-'
    return [m for ion, ms in ions.items() if int(ion.rsplit(']', 1)[1][:-1] or 1) in permitted_charges for m in np.atleast_1d(ms) if abs(mass - m) < mass_tolerance]


# Smallest half-width (ppm) of accurate_ms1_window per survey scan analyzer: TOF precursors are less accurate than orbitrap ones (Agilent Q-TOF mgf of GPST000570
# without survey scans to refine them: +-7 ppm; timsTOF after refine_tof_precursors: robust SD 1.6 ppm)
MS1_PPM_FLOORS = {'orbitrap': 3.0, 'tof': 5.0}


def accurate_ms1_window(ppm, scores, ppm_thresh, floor = 3.0, groups = None):
    """center and half-width (ppm) of the composition errors of a run with accurate (orbitrap, timsTOF) survey scans: real glycans pile up within a few ppm,
    while matches of other compositions spread over the whole ppm_thresh and swamp a plain median and MAD (permethylated serum on a Q Exactive: 3 robust SD
    of all rows 116 ppm). The center is the densest 4 ppm window, refined to the median of the rows within 5 ppm of it whose fragments back them up best
    (annotation score in their top 10%, at least 20 rows), and the half-width 3 robust SD of those rows, at least floor; None with fewer than 5 rows. With groups (e.g., the
    rows' spectra and compositions), the densest window counts each group once, at its median error: one junk spectrum of GPST000461 gave 8 rows (one
    per candidate structure) at -41.9 ppm, which made it the densest window of its free oligosaccharide run"""
    ok = np.isfinite(ppm) & (np.abs(ppm) <= ppm_thresh)
    ppm, scores = ppm[ok], np.asarray(scores, dtype = float)[ok]
    if len(ppm) < 5:
        return None
    if groups is None:
        s = np.sort(ppm)
    else:
        groups = np.asarray(groups, dtype = object)[ok]
        s = np.sort([np.median(ppm[groups == g]) for g in pd.unique(groups)])
    grid = np.arange(s[0], s[-1] + 0.25, 0.25)
    center = grid[np.argmax(np.searchsorted(s, grid + 2, side = 'right') - np.searchsorted(s, grid - 2))]
    near = np.abs(ppm - center) <= 5
    best = near & (scores >= np.quantile(scores[near], 0.9))
    # From the top 10% of a few dozen rows, center and spread scatter: GPST000578 file 2 got 1.5 +- 3.0 ppm while its glycans lie at -0.1 +- 5.8 ppm, so the
    # database fallback lost every glycan below -1.5 ppm
    if best.sum() < 20:
        best = near & (scores >= np.sort(scores[near])[::-1][min(20, near.sum()) - 1])
    best = best if best.sum() >= 5 else near
    if best.sum() < 5:
        return None
    center = float(np.median(ppm[best]))
    return center, max(3 * 1.4826 * float(np.median(np.abs(ppm[best] - center))), floor)


def median3(x):
    """np.median of each value with its two neighbors (one at the ends), vectorized and bitwise identical to calling np.median on each window"""
    if len(x) < 3:
        return np.full(len(x), x[0] if len(x) == 1 else (x[0] + x[1]) / 2, dtype = float)
    a, b, c = x[:-2], x[1:-1], x[2:]
    return np.concatenate([[(x[0] + x[1]) / 2], np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c)), [(x[-2] + x[-1]) / 2]])


def ms2_vector(peak_ds, max_mz):
    # Square-root intensities of pooled MS2 spectra in 1 Da bins, scaled to unit length; np.add.at adds the peaks in their order, as a loop over them does
    v = np.zeros(int(max_mz) + 1)
    mzs = np.array([m for spec in peak_ds for m in spec], dtype = float)
    np.add.at(v, np.rint(np.clip(mzs, 0, max_mz)).astype(int), np.array([i for spec in peak_ds for i in spec.values()], dtype = float))
    return np.sqrt(v) / (np.linalg.norm(np.sqrt(v)) + 1e-12)


def condense_dataframe(df, mz_diff = 0.5, rt_diff = 1.0, min_mz = 39.714, max_mz = 3000, bin_num = 2048, mz_ppm = None, merged = None, ms1 = None):
    """groups spectra and combines the clusters into averaged and binned spectra\n
    | Arguments:
    | :-
    | df (dataframe): dataframe from load_spectra_filepath
    | mz_diff (float): mass tolerance for assigning spectra to the same peak; default:0.5
    | rt_diff (float): retention time tolerance (in minutes) for assigning spectra to the same peak; default:1.0
    | min_mz (float): minimal m/z used for binning; don't change; default:39.714
    | max_mz (float): maximal m/z used for binning; don't change; default:3000
    | bin_num (int): number of bins for binning; don't change; default: 2048
    | mz_ppm (float): relative mass tolerance (ppm) for assigning spectra to the same peak, used instead of mz_diff; default:None
    | merged (dict): pooled spectra of clusters and binned spectra, kept by the caller across calls on the same spectra (the reducing end and glycan class
    |                checks condense a file up to 5 times, nearly always into the same clusters); default:None
    | ms1 (tuple): accurate survey scans (rts, mzs, intensities, scan offsets), along whose precursor traces clusters are split and placed; default:None\n
    | Returns:
    | :-
    | Returns a dataframe that has one row per RT-Mass cluster
    """
    # Intensity binning
    step = (max_mz - min_mz) / (bin_num - 1)
    frames = np.array([min_mz + step * i for i in range(bin_num)])
    if 'precursor_charge' not in df.columns:
        df['precursor_charge'] = None
    clusters = []
    # Sort the dataframe by 'reducing_mass'/'m/z' and 'RT'
    idx_col = 'm/z' if 'm/z' in df.columns else 'reducing_mass'
    df['rounded_mz'] = np.round(df[idx_col] * 2) / 2
    df['rounded_RT'] = np.round(df['RT'], 1)
    df.sort_values(by = ['rounded_mz', 'rounded_RT'], inplace = True)
    rounded_mz = df['rounded_mz'].to_numpy()
    df.drop(['rounded_mz', 'rounded_RT'], axis = 1, inplace = True)
    # Initialize the first cluster
    mz_arr = df[idx_col].to_numpy()
    rt_arr = df['RT'].to_numpy()
    int_arr = df['intensity'].to_numpy()
    peak_arr = df['peak_d'].to_numpy(dtype = object)
    chg_arr = df['precursor_charge'].to_numpy(dtype = object)
    ms3_arr = df['ms3'].to_numpy(dtype = object) if 'ms3' in df.columns else [[]] * len(df)
    clusters.append({
        'm/z': [mz_arr[0]],
        'RT': [rt_arr[0]],
        'intensity': [int_arr[0]],
        'peak_d': [peak_arr[0]],
        'precursor_charge': [chg_arr[0]],
        'ms3': list(ms3_arr[0]),
        'apex_int': int_arr[0],
        'apex_mz': mz_arr[0],
        'apex_rt': rt_arr[0]
    })
    active = list(clusters)
    # Loop through the sorted dataframe starting from the second row
    for r in range(1, len(mz_arr)):
        mz, rt, intensity, peak_d = mz_arr[r], rt_arr[r], int_arr[r], peak_arr[r]
        # Rows come sorted by m/z rounded to 0.5, so a cluster whose reference m/z lies more than mz_diff below this rounding bin can never
        # match again; dropping those keeps the scan from growing with every cluster made so far (the first matching cluster is unchanged)
        if rounded_mz[r] != rounded_mz[r - 1]:
            active = [c for c in active if (c['apex_mz'] if c['apex_int'] > 0 else c['m/z'][-1]) >= rounded_mz[r] - 0.5 - mz_diff]
        found = False
        tol = mz * mz_ppm / 1e6 if mz_ppm else mz_diff
        for cluster in active:
            last_max = cluster['apex_int']
            if last_max > 0:
                last_mz, last_rt = cluster['apex_mz'], cluster['apex_rt']
            else:
                last_mz, last_rt = cluster['m/z'][-1], cluster['RT'][-1]
            if abs(last_mz - mz) <= tol and abs(last_rt - rt) <= rt_diff:
                cluster['m/z'].append(mz)
                cluster['RT'].append(rt)
                cluster['intensity'].append(intensity)
                cluster['peak_d'].append(peak_d)
                cluster['precursor_charge'].append(chg_arr[r])
                cluster['ms3'].extend(ms3_arr[r])
                if intensity > last_max:
                    cluster['apex_int'], cluster['apex_mz'], cluster['apex_rt'] = intensity, mz, rt
                found = True
                break
        if not found:
            clusters.append({
                'm/z': [mz],
                'RT': [rt],
                'intensity': [intensity],
                'peak_d': [peak_d],
                'precursor_charge': [chg_arr[r]],
                'ms3': list(ms3_arr[r]),
                'apex_int': intensity,
                'apex_mz': mz,
                'apex_rt': rt
            })
            active.append(clusters[-1])
    # A cluster at most a tenth as intense as one at the same m/z whose spectra it continues (within 0.25 min), with a similar MS2 spectrum (cosine
    # >= 0.75 of square-root intensities in 1 Da bins), is that peak's PGC tail, which can outlast rt_diff from the apex: it joins the peak instead of
    # becoming a glycan peak of its own (co-eluting isomers fragment differently)
    mz_diff_use = mz_ppm / 1e6 if mz_ppm else mz_diff
    # Each cluster's MS2 vector is made once, and again only after it took up more spectra
    vectors = {}
    def vector(cl):
        if vectors.get(id(cl), (0, None))[0] != len(cl['peak_d']):
            vectors[id(cl)] = (len(cl['peak_d']), ms2_vector(cl['peak_d'], max_mz))
        return vectors[id(cl)][1]
    by_apex = sorted(clusters, key = lambda c: -c['apex_int'])
    # Only the clusters before the first one less than ten times as intense (by_apex order) and within mz_diff_use are looked at, found with numpy: looping over
    # every stronger cluster made this quadratic (10 million iterations, 40% of condensing a 16,000-cluster run)
    tenth, by_mz = np.array([c['apex_int'] * 0.1 for c in by_apex]), np.array([c['apex_mz'] for c in by_apex])
    # (each cluster's candidates from a window of them sorted by m/z, in by_apex order)
    tail_of, mz_order = {}, np.argsort(by_mz, kind = 'stable')
    mz_sorted = by_mz[mz_order]
    for w in by_apex[::-1]:
        if w['apex_int'] <= 0:
            continue
        below = tenth < w['apex_int']
        n = int(below.argmax()) if below.any() else len(by_apex)
        near = np.sort(mz_order[np.searchsorted(mz_sorted, w['apex_mz'] - mz_diff_use - 1e-6):np.searchsorted(mz_sorted, w['apex_mz'] + mz_diff_use + 1e-6, side = 'right')])
        for k in near[(near < n) & (np.abs(by_mz[near] - w['apex_mz']) <= mz_diff_use)]:
            c = by_apex[k]
            if np.min(np.abs(np.subtract.outer(c['RT'], w['RT']))) <= 0.25:
                if vector(c) @ vector(w) >= 0.75:
                    tail_of[id(w)] = c
                    break
    for w in by_apex[::-1]:
        if id(w) in tail_of:
            c = tail_of[id(w)]
            while id(c) in tail_of:
                c = tail_of[id(c)]
            for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge', 'ms3'):
                c[key].extend(w[key])
    clusters = [c for c in clusters if id(c) not in tail_of]
    # Two clusters at one m/z are one chromatographic peak if their precursor intensities never drop below 0.7 of the smaller apex (apexes as medians of 3
    # spectra) between their apexes, as a ground-truth peak needs such a valley: a long PGC tail that keeps triggering MS2 becomes clusters rt_diff apart,
    # each only a few times weaker than the one before, so the tenfold rule above catches only its end. Similar MS2 spectra (cosine >= 0.9) also join
    # across a valley down to 0.5, or across gaps of up to 2 min without spectra (else 0.25 min); weakest first, so a cluster that absorbed weaker ones
    # can still join a stronger one
    # Files that fragment a precursor again only after a long dynamic exclusion (every ~1.5 min in GPST000461 and GPST000548) have gaps of that length inside
    # every peak, so there the gap allowed without spectra is 1.2x the file's re-trigger interval (lower quartile of the time between successive spectra of
    # one precursor m/z) if that is a minute or more (0.13-0.5 min in the other test files; GPST000596 at 0.42 lost 0.006 with it)
    order = np.lexsort((rt_arr, np.round(mz_arr / 0.02)))
    gaps = np.diff(rt_arr[order])[np.abs(np.diff(mz_arr[order])) <= 0.02]
    gaps = gaps[gaps > 0.1]
    retrigger = float(np.quantile(gaps, 0.25)) if len(gaps) >= 50 else 0.0
    # rounded up to 0.05 min: the quartile lies in the sparse stretch before the re-trigger peak, where one spectrum more or less moves it by up to 0.006 min, and
    # 1.2x it falls among the re-trigger gaps (GPST000461_O 1.6244 min, 1.6178 with one precursor m/z refined differently, which split clusters all over the file)
    max_gap = float(np.ceil(round(1.2 * retrigger / 0.05, 6))) * 0.05 if retrigger >= 1.0 else 0.25
    # Python's min, max, sorted and division agree with numpy's unless NaN, negative values, or negative zeros are involved
    try:
        ints_f = np.asarray(int_arr, dtype = float)
        plain = not (np.isnan(np.asarray(rt_arr, dtype = float)).any() or np.isnan(ints_f).any() or np.signbit(ints_f).any())
    except (TypeError, ValueError):
        plain = False
    apex_mzs, apex_ints = np.array([c['apex_mz'] for c in clusters]), np.array([max(c['intensity']) for c in clusters])
    # (each cluster's candidates from a window of them sorted by m/z, in cluster order)
    joined, mz_order = set(), np.argsort(apex_mzs, kind = 'stable')
    mz_sorted = apex_mzs[mz_order]
    for i in np.argsort(apex_ints, kind = 'stable'):
        w = clusters[i]
        if apex_ints[i] <= 0:
            continue
        tol = apex_mzs[i] * mz_ppm / 1e6 if mz_ppm else mz_diff
        near = np.sort(mz_order[np.searchsorted(mz_sorted, apex_mzs[i] - tol - 1e-6):np.searchsorted(mz_sorted, apex_mzs[i] + tol + 1e-6, side = 'right')])
        for j in near[(np.abs(apex_mzs[near] - apex_mzs[i]) <= tol) & (apex_ints[near] >= apex_ints[i])]:
            c = clusters[j]
            if j == i or j in joined:
                continue
            if plain:
                # the same in plain Python (faster for a few spectra), which gives the same values without NaN and negative values (plain)
                rts, ints = c['RT'] + w['RT'], c['intensity'] + w['intensity']
                order = sorted(range(len(rts)), key = rts.__getitem__)
                rts, ints = [rts[k] for k in order], [float(ints[k]) for k in order]
                sm = [ints[0] if len(ints) == 1 else (ints[0] + ints[1]) / 2] + [max(min(x, y), min(max(x, y), z)) for x, y, z in zip(ints, ints[1:], ints[2:])] + (
                    [(ints[-2] + ints[-1]) / 2] if len(ints) > 1 else [])
                a, b = sorted([max((k for k in range(len(order)) if order[k] < len(c['RT'])), key = sm.__getitem__),
                               max((k for k in range(len(order)) if order[k] >= len(c['RT'])), key = sm.__getitem__)])
                gap = max(rts[k + 1] - rts[k] for k in range(a, b)) if b > a else 0.0
                low, floor = min(ints[a + 1:b]) if b > a + 1 else np.inf, min(sm[a], sm[b])
                # (numpy's division by zero)
                depth = low / floor if floor else np.inf if low else np.nan
            else:
                order = np.argsort(c['RT'] + w['RT'], kind = 'stable')
                rts, ints = np.array(c['RT'] + w['RT'])[order], np.array(c['intensity'] + w['intensity'], dtype = float)[order]
                own = (np.arange(len(order)) < len(c['RT']))[order]
                sm = median3(ints)
                a, b = sorted([np.flatnonzero(own)[np.argmax(sm[own])], np.flatnonzero(~own)[np.argmax(sm[~own])]])
                gap = np.diff(rts[a:b + 1]).max() if b > a else 0.0
                depth = (ints[a + 1:b].min() if b > a + 1 else np.inf) / min(sm[a], sm[b])
            if gap > max(2, max_gap) or depth < 0.5 or (gap > max_gap and depth < 0.7):
                continue
            if depth < 0.7 or gap > max_gap:
                if vector(c) @ vector(w) < 0.9:
                    continue
            for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge', 'ms3'):
                c[key].extend(w[key])
            joined.add(i)
            break
    clusters = [c for k, c in enumerate(clusters) if k not in joined]
    # The other way round, a cluster whose precursor intensities fall below half the smaller of two maxima (medians of 3 spectra) between them holds two
    # peaks (isomers within rt_diff of each other), each with at least 2 spectra; the MS3 spectra stay with the half holding the apex
    stack, clusters = clusters, []
    while stack:
        c = stack.pop()
        order = np.argsort(c['RT'], kind = 'stable')
        ints = np.array(c['intensity'], dtype = float)[order]
        rts = np.array(c['RT'], dtype = float)[order]
        cut = None
        if len(ints) >= 4 and ints.max() > 0:
            sm = median3(ints)
            top = int(np.argmax(sm))
            for m in sorted(range(len(sm)), key = lambda k: -sm[k]):
                if abs(m - top) < 2 or sm[m] <= 0:
                    continue
                lo, hi = sorted([top, m])
                v = lo + 1 + int(np.argmin(ints[lo + 1:hi]))
                # A valley lasts: the spectra within 0.03 min of it (MS2 every 1-2 s) stay below 0.75 of the smaller maximum, as a single low precursor
                # intensity between high ones is noise
                if ints[v] < 0.5 * min(sm[top], sm[m]) and min(v, len(ints) - v) >= 2 and (ints[np.abs(rts - rts[v]) <= 0.03] < 0.75 * min(sm[top], sm[m])).all():
                    cut = v
                    break
        if cut is None:
            clusters.append(c)
            continue
        for ix in (order[:cut], order[cut:]):
            h = {key: [c[key][x] for x in ix] for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge')}
            k = int(np.argmax(h['intensity']))
            h['apex_int'], h['apex_mz'], h['apex_rt'] = h['intensity'][k], h['m/z'][k], h['RT'][k]
            h['ms3'] = c['ms3'] if c['apex_rt'] in h['RT'] else []
            stack.append(h)
    # Without precursor intensities there is no valley to split at, and spectra chain into one cluster as each one only has to lie within rt_diff of the
    # last: a stretch without spectra longer than 0.55 min splits it
    split_clusters = []
    for c in clusters:
        if any(x > 0 for x in c['intensity']):
            split_clusters.append(c)
            continue
        order = np.argsort(c['RT'], kind = 'stable')
        parts = np.split(order, np.flatnonzero(np.diff(np.array(c['RT'], dtype = float)[order]) > 0.55) + 1)
        if len(parts) == 1:
            split_clusters.append(c)
            continue
        largest = max(range(len(parts)), key = lambda k: len(parts[k]))
        for k, ix in enumerate(parts):
            h = {key: [c[key][x] for x in ix] for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge')}
            h['apex_int'], h['apex_mz'], h['apex_rt'] = np.nan, h['m/z'][0], h['RT'][0]
            # The MS3 spectra stay with the part holding most MS2 spectra
            h['ms3'] = c['ms3'] if k == largest else []
            split_clusters.append(h)
    clusters = split_clusters
    # With accurate survey scans, the clusters of one precursor m/z (within mz_ppm, else mz_diff) are regrouped by the peaks of its MS1 trace (10 ppm,
    # smoothed with a Gaussian of 0.05 min, at least 0.7 cycles; maxima without a valley below 0.7 of the lower one between them are one peak), and every
    # group is reported at the apex of its MS1 peak: DDA fragments a peak on its flanks (dynamic exclusion), so anomers and isomers 0.3-1.5 min apart
    # chained into one cluster reported between them, while a broad peak that kept triggering MS2 for minutes became a cluster every rt_diff
    if ms1 is not None and len(ms1[0]) > 5:
        rts1, mzs1, ints1, offs1 = ms1
        keys1 = np.repeat(np.arange(len(rts1)) * 1e4, np.diff(offs1)) + mzs1.astype(np.float64)
        order1 = np.argsort(keys1, kind = 'stable')
        keys1, cum1, base1 = keys1[order1], np.concatenate([[0], np.cumsum(ints1[order1], dtype = np.float64)]), np.arange(len(rts1)) * 1e4
        clusters.sort(key = lambda c: c['apex_mz'])
        groups, placed = [], []
        for c in clusters:
            if groups and c['apex_mz'] - groups[-1][-1]['apex_mz'] <= (c['apex_mz'] * mz_ppm / 1e6 if mz_ppm else mz_diff):
                groups[-1].append(c)
            else:
                groups.append([c])
        for group in groups:
            rts_g = np.concatenate([c['RT'] for c in group])
            lo, hi = np.searchsorted(rts1, rts_g.min() - 2), np.searchsorted(rts1, rts_g.max() + 2, side = 'right')
            mz0 = float(np.median([c['apex_mz'] for c in group]))
            b = base1[lo:hi]
            trace = cum1[np.searchsorted(keys1, b + mz0 * (1 + 1e-5), side = 'right')] - cum1[np.searchsorted(keys1, b + mz0 * (1 - 1e-5))]
            if hi - lo < 5 or trace.max() <= 0:
                placed += group
                continue
            # Smoothed in time, not in scans: survey scans came every 0.4 s overall but every 2.4 s around the RNase B high-mannose peaks of GPST000134, where
            # 7 scans (0.05 min at the median cycle) blurred anomers 1 min apart into one peak
            t = rts1[lo:hi]
            steps = np.diff(t)
            sigma = max(0.05, 0.7 * float(np.median(steps[steps > 0]))) if (steps > 0).any() else 0.05
            half = int(min(200, np.max(np.searchsorted(t, t + 3 * sigma, side = 'right') - np.arange(len(t)))))
            idx = np.arange(len(t))[:, None] + np.arange(-half, half + 1)[None, :]
            w = np.where((idx >= 0) & (idx < len(t)), np.exp(-0.5 * ((t[np.clip(idx, 0, len(t) - 1)] - t[:, None]) / sigma) ** 2), 0)
            sm = (trace[np.clip(idx, 0, len(t) - 1)] * w).sum(1) / w.sum(1)
            # Maxima inside the window (one at its edge is the flank of a peak outside it)
            tops = [k for k in range(1, len(sm) - 1) if sm[k] > 0 and sm[k] > sm[k - 1] and sm[k] >= sm[k + 1]] or [int(np.argmax(sm))]
            merged_any = True
            while merged_any and len(tops) > 1:
                merged_any = False
                for k in range(len(tops) - 1):
                    if sm[tops[k]:tops[k + 1] + 1].min() >= 0.7 * min(sm[tops[k]], sm[tops[k + 1]]):
                        tops.pop(k if sm[tops[k]] < sm[tops[k + 1]] else k + 1)
                        merged_any = True
                        break
            bounds = rts1[[lo + tops[k] + int(np.argmin(sm[tops[k]:tops[k + 1] + 1])) for k in range(len(tops) - 1)]]
            parts = {}
            for c in group:
                part = np.searchsorted(bounds, c['RT'])
                for x, k in enumerate(part):
                    h = parts.setdefault(int(k), {key: [] for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge', 'ms3')})
                    for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge'):
                        h[key].append(c[key][x])
                    if c['RT'][x] == c['apex_rt']:
                        h['ms3'] += c['ms3']
            # A part needs 2 spectra and 5% of the highest MS1 apex of the group's parts, else its spectra join the part whose apex is nearest: on
            # GPST000134, whose high-mannose glycans fly as [M+K+H]2+ with in-source fragments at every apex, single spectra on small maxima were rows the
            # ground truth does not hold
            keep = [k for k in parts if len(parts[k]['RT']) >= 2 and sm[tops[k]] >= 0.05 * max(sm[tops[j]] for j in parts)] or [max(parts, key = lambda k: sm[tops[k]])]
            for k in [k for k in parts if k not in keep]:
                into = parts[min(keep, key = lambda j: abs(rts1[lo + tops[j]] - rts1[lo + tops[k]]))]
                for key in ('m/z', 'RT', 'intensity', 'peak_d', 'precursor_charge', 'ms3'):
                    into[key] += parts[k][key]
            parts = {k: parts[k] for k in keep}
            for k, h in parts.items():
                j = int(np.argmax(h['intensity']))
                h['apex_int'], h['apex_mz'], h['apex_rt'], h['xic_rt'] = h['intensity'][j], h['m/z'][j], h['RT'][j], float(rts1[lo + tops[k]])
                placed.append(h)
        clusters = placed
    # Create a condensed dataframe
    condensed_data = []
    merged = {} if merged is None else merged
    # The spectra still to bin, all at once: one stable argsort and reduceat over their peaks by spectrum and bin gives each spectrum the arrays
    # bin_intensities gives it, bitwise (one call per spectrum was ~15% of condensing a large run); peaks of other types than Python floats go through it
    todo = list({id(s): s for c in clusters if (tuple(map(id, c['peak_d'])), mz_diff, min_mz, max_mz, bin_num) not in merged for s in c['peak_d']
                 if ('bin', id(s), min_mz, max_mz, bin_num) not in merged and set(map(type, s)).union(map(type, s.values())) <= {float}}.values())
    if todo:
        sizes = [len(s) for s in todo]
        mzs = np.fromiter((m for s in todo for m in s), np.float64, sum(sizes)).astype(np.float32)
        ints = np.fromiter((i for s in todo for i in s.values()), np.float64, sum(sizes))
        owner = np.repeat(np.arange(len(todo)), sizes)
        in_range = (mzs > frames[0]) & (mzs <= frames[-1])
        mzs, ints, owner = mzs[in_range], ints[in_range], owner[in_range]
        binned_all, remainder_all = np.zeros((len(todo), bin_num)), np.zeros((len(todo), bin_num))
        if len(mzs):
            bins = np.digitize(mzs, frames, right = True)
            keys = owner * (bin_num + 1) + bins
            order = np.argsort(keys, kind = 'stable')
            unique_keys, starts = np.unique(keys[order], return_index = True)
            remainders = (mzs - frames[bins - 1]) * (ints == np.maximum.reduceat(ints[order], starts)[np.searchsorted(unique_keys, keys)])
            binned_all[unique_keys // (bin_num + 1), unique_keys % (bin_num + 1) - 1] = np.add.reduceat(ints[order], starts)
            remainder_all[unique_keys // (bin_num + 1), unique_keys % (bin_num + 1) - 1] = np.maximum.reduceat(remainders[order], starts)
        for k, s in enumerate(todo):
            merged[('bin', id(s), min_mz, max_mz, bin_num)] = (s, binned_all[k], remainder_all[k])
    for cluster in clusters:
        highest_intensity_index = np.argmax(cluster['intensity'])
        highest_intensity = cluster['intensity'][highest_intensity_index]
        if highest_intensity > 0:
            rep_mz = cluster['m/z'][highest_intensity_index]
            mean_rt = cluster['RT'][highest_intensity_index]
            # DDA re-triggers on the 13C peak of a multiply charged precursor once its monoisotopic peak is excluded, and where an ion-trap survey scan
            # merges the isotopes into one centroid as intense as the monoisotopic one, the loader cannot walk it down; such spectra chain into their
            # precursor's cluster, whose apex they can be. A cluster whose spectra 0.3-0.7 m/z below its apex hold at least half the intensity of those above is reported at the strongest of them
            mzs_c, ints_c = np.array(cluster['m/z'], dtype = float), np.array(cluster['intensity'], dtype = float)
            lower = (mzs_c >= rep_mz - 0.7) & (mzs_c <= rep_mz - 0.3)
            if lower.any() and ints_c[lower].sum() >= 0.5 * ints_c[mzs_c > rep_mz - 0.3].sum():
                rep_mz = float(mzs_c[lower][np.argmax(ints_c[lower])])
        else:
            rep_mz = min(cluster['m/z'])
            mean_rt = np.mean(cluster['RT'])
        mean_rt = cluster.get('xic_rt', mean_rt)
        # The same spectra in the same order pool to the same spectrum (the cached entry keeps them, so their ids stay theirs)
        key = (tuple(map(id, cluster['peak_d'])), mz_diff, min_mz, max_mz, bin_num)
        if key in merged:
            _, peaks, binned_intensities, mz_remainder = merged[key]
            condensed_data.append([rep_mz, mean_rt, highest_intensity, peaks, binned_intensities.copy(), mz_remainder.copy(), len(cluster['RT']),
                                   cluster['precursor_charge'][highest_intensity_index]])
            continue
        # Cluster fragment peaks across spectra by mass proximity, then weight-average mass and sum intensity
        ms = np.fromiter((m for spec in cluster['peak_d'] for m in spec), float)
        if all(set(map(type, spec)).union(map(type, spec.values())) <= {float} for spec in cluster['peak_d']) and not np.isnan(ms).any():
            # Python floats (as every loader gives them) in numpy, bitwise identical to the loop below: a stable argsort is sorted's order, a group starts at
            # the first peak further than mz_diff / 2 from the group's first one, a group below 8 peaks adds up its weighted masses left to right and its
            # intensities as sum does (compensated from Python 3.12), a larger one as np.average does. The loop was ~40% of condensing a large run
            ints = np.fromiter((i for spec in cluster['peak_d'] for i in spec.values()), float, len(ms))
            order = np.argsort(ms, kind = 'stable')
            ms, ints = ms[order], ints[order]
            # For every peak, the first one further than mz_diff / 2 (found by searchsorted, then corrected to the loop's own subtraction); the groups then
            # chain from the first peak
            nxt = np.searchsorted(ms, ms + mz_diff / 2, side = 'right')
            while (fix := (nxt < len(ms)) & (ms[np.minimum(nxt, len(ms) - 1)] - ms <= mz_diff / 2)).any():
                nxt[fix] += 1
            while (fix := ~(ms[np.maximum(nxt - 1, 0)] - ms <= mz_diff / 2)).any():
                nxt[fix] -= 1
            nxt, starts, k = nxt.tolist(), [], 0
            while k < len(nxt):
                starts.append(k)
                k = nxt[k]
            bounds = np.array(starts + [len(nxt)])
            sizes = np.diff(bounds)
            small = sizes < 8
            idx = np.minimum(bounds[:-1][small, None] + np.arange(7), len(ms) - 1)
            valid = np.arange(7) < sizes[small, None]
            prods, weights = np.where(valid, ms[idx] * ints[idx], 0.0), np.where(valid, ints[idx], 0.0)
            num, den, total, comp = (np.zeros(len(idx)) for _ in range(4))
            for c in range(7):
                num = num + prods[:, c]
                den = den + weights[:, c]
                if COMPENSATED_SUM:
                    t = total + weights[:, c]
                    comp += np.where(np.abs(total) >= np.abs(weights[:, c]), (total - t) + weights[:, c], (weights[:, c] - t) + total)
                    total = t
            # A group of zero-intensity peaks is no peak (np.average raised for it)
            keep = np.ones(len(starts), dtype = bool)
            keep[np.flatnonzero(small)[den == 0]] = False
            keys, values = np.empty(len(starts)), np.empty(len(starts))
            keys[small] = num / np.where(den == 0, 1.0, den)
            values[small] = np.where((comp != 0) & np.isfinite(comp), total + comp, total) if COMPENSATED_SUM else den
            ints_list = ints.tolist()
            for g in np.flatnonzero(~small):
                a, b = bounds[g], bounds[g + 1]
                den = ints[a:b].sum()
                if den == 0:
                    keep[g] = False
                    continue
                keys[g] = (ms[a:b] * ints[a:b]).sum() / den
                values[g] = sum(ints_list[a:b])
            pk = dict(zip(list(keys[keep]), values[keep].tolist()))
        else:
            mi_pairs = sorted([(m, i) for spec in cluster['peak_d'] for m, i in spec.items()], key = lambda x: x[0])
            starts, grp_start = [], None
            for k, (m, _) in enumerate(mi_pairs):
                if grp_start is None or not m - grp_start <= mz_diff / 2:
                    starts.append(k)
                    grp_start = m
            pk = {}
            for a, b in zip(starts, starts[1:] + [len(mi_pairs)]):
                ms, ints = zip(*mi_pairs[a:b])
                # np.average written out for groups below 8 float peaks, which numpy also adds up one by one (bitwise identical)
                if b - a < 8 and all(isinstance(x, float) for x in ms + ints):
                    num = den = 0.0
                    for m, i in zip(ms, ints):
                        num += m * i
                        den += i
                    if den == 0:
                        continue
                    pk[np.float64(num / den)] = sum(ints)
                elif sum(ints) != 0:
                    pk[np.average(ms, weights = ints)] = sum(ints)
        peaks = dict(sorted(pk.items(), key = lambda x: x[1], reverse = True))
        binned = []
        for spec in cluster['peak_d']:
            k = ('bin', id(spec), min_mz, max_mz, bin_num)
            if k not in merged:
                merged[k] = (spec, *bin_intensities(spec, frames))
            binned.append(merged[k][1:])
        binned_intensities, mz_remainder = zip(*binned)
        binned_intensities = np.mean(np.array(binned_intensities), axis = 0)
        mz_remainder = np.mean(np.array(mz_remainder), axis = 0)
        # Bin intensity normalization
        binned_intensities = binned_intensities / binned_intensities.sum()
        merged[key] = (cluster['peak_d'], peaks, binned_intensities.copy(), mz_remainder.copy())
        num_spectra = len(cluster['RT'])
        rep_charge = cluster['precursor_charge'][highest_intensity_index]
        # Without MS1, the apex precursor intensity is the abundance: it tracks MS1 peak areas far better than summing or integrating
        # the precursor intensities of however many MS2 spectra dynamic exclusion happened to allow
        condensed_data.append(
            [rep_mz, mean_rt, highest_intensity, peaks, binned_intensities, mz_remainder, num_spectra, rep_charge])
    df_out = pd.DataFrame(condensed_data,
                          columns = ['m/z', 'RT', 'intensity', 'peak_d', 'binned_intensities', 'mz_remainder',
                                     'num_spectra', 'precursor_charge'])
    # First and last retention time of each cluster's spectra, to tell whether two clusters are one chromatographic peak
    df_out['rt_range'] = [(min(c['RT']), max(c['RT'])) for c in clusters]
    # The MS3 spectra of all MS2 spectra in a cluster
    if 'ms3' in df.columns:
        df_out['ms3'] = [c['ms3'] for c in clusters]
    return df_out


_TOPOLOGIES = {}
CLASS_CODES = {'O': 0, 'N': 1, 'free': 2, 'lipid': 2}


def glycan_classes(glycan_class):
    # A run holds one glycan class (a string) or several measured together (a list, e.g., ['N', 'O'] for N- and O-glycans released from one sample)
    return [glycan_class] if isinstance(glycan_class, str) else list(dict.fromkeys(glycan_class))


def in_classes(glycan, classes, conf = None, extra_thresh = 0.3):
    # enforce_class for several classes: a structure of any of them, or (with conf) one more likely than extra_thresh
    return any(enforce_class(glycan, c) for c in classes) or (conf is not None and conf > extra_thresh)


def create_struct_maps(df_glycan, classes, filter_out = None, phylo_level = 'Kingdom', phylo_filter = 'Animalia'):
    # create_struct_map of several classes as one: a composition's candidate structures are those of every class whose database has it (its topologies,
    # else its most common structure), so a spectrum is scored, and goes to the model, as a glycan of each class its composition belongs to
    maps = [create_struct_map(df_glycan, c, filter_out = filter_out, phylo_level = phylo_level, phylo_filter = phylo_filter) for c in classes]
    if len(maps) == 1:
        return maps[0]
    comp_map = {k: v for m in maps[::-1] for k, v in m[0].items()}
    topo_map = {k: list(dict.fromkeys(s for m in maps for s in (m[2].get(k) or ([m[0][k]] if k in m[0] else [])))) for k in comp_map}
    return comp_map, pd.concat([m[1] for m in maps]), topo_map


def create_struct_map(df_glycan, glycan_class, filter_out = None, phylo_level = 'Kingdom', phylo_filter = 'Animalia'):
    processed_df_use = df_glycan[df_glycan[phylo_level].apply(lambda x: phylo_filter in x) & (df_glycan['glycan_type'] == glycan_class)]
    if filter_out:
        processed_df_use = processed_df_use.iloc[
            [i for i, x in enumerate(processed_df_use.Composition) if not filter_out.intersection(x)]]
    processed_df_use = processed_df_use.assign(comp_str = [stringify_dict(x) for x in processed_df_use.Composition])
    processed_df_use = processed_df_use.assign(
        ref_counts = processed_df_use.loc[:, 'ref'].map(len) + processed_df_use.loc[:, 'tissue_ref'].map(len) +
                     processed_df_use.loc[:, 'disease_ref'].map(len))
    processed_df_use = processed_df_use[~(processed_df_use['glycan'].str.contains("}"))]
    processed_df_use = processed_df_use.sort_values('ref_counts', ascending = False)
    small = np.array([0 < sum(x.values()) < 6 for x in processed_df_use.Composition], dtype = bool)
    # Only the topology map of compositions below 6 residues needs topologies; every run builds these maps (also for the other classes with accurate
    # MS1), so they are kept across calls
    topologies = [_TOPOLOGIES[x] if x in _TOPOLOGIES else _TOPOLOGIES.setdefault(x, structure_to_basic(x)) for x in processed_df_use['glycan'][small]]
    processed_df_use = processed_df_use.assign(
        glycan = [x.replace('-ol', '').replace('1Cer', '') for x in processed_df_use.glycan])
    small_comps = processed_df_use[small].assign(topology = topologies)
    df_use_unq_topos = small_comps.groupby('topology').first().groupby('comp_str').agg(list)
    topology_map = dict(zip(df_use_unq_topos.index, df_use_unq_topos.glycan))
    df_use_unq_comps = processed_df_use.groupby('comp_str').first()
    common_struct_map = dict(zip(df_use_unq_comps.index, df_use_unq_comps.glycan))
    return common_struct_map, processed_df_use, topology_map


def complementary_pairs(mz, peaks, charge, tolerance, n = 15):
    """count pairs among the n most intense fragments that add up to the precursor at a charge state: a singly charged B/C ion and the
    (charge - 1)-fold charged rest of the glycan (b + (charge - 1) * y = charge * mz, the charge carriers cancel)"""
    top = [m for m, _ in sorted(peaks.items(), key = lambda x: x[1], reverse = True)[:n]]
    return sum(abs(b + (charge - 1) * y - charge * mz) < charge * tolerance for b in top for y in top if y > b)


def assign_candidate_structures(df_in, df_glycan_in, comp_struct_map, topo_struct_map, mass_tolerance, mode, mass_tag,
                                modification = 'reduced', sample_prep = 'underivatized', max_charge = -3, cross_class = None, ms1_ppm = None,
                                sialic_derivatization = None):
    idx_col = 'm/z' if 'm/z' in df_in.columns else 'reducing_mass'
    red_masses = np.array(df_in[idx_col])
    known_charges = [None if pd.isna(c) else int(c) for c in df_in['precursor_charge'].values] if 'precursor_charge' in df_in.columns else [None] * len(red_masses)
    tag = mass_tag if mass_tag else 0
    comps_out = [(None, 0)] * len(red_masses)
    # A top 5 fragment heavier than 1.1x the precursor m/z rules out z = 1 (as in domain_filter)
    heavy_frags = [any(m > 1.1 * mz for m in list(peaks)[:5]) for mz, peaks in zip(red_masses, df_in['peak_d'])]
    # The charge states domain_filter rules out, which would empty any prediction: z = 1 with a top 5 fragment above 1.1x the precursor m/z and
    # >= 20% of the base peak, z >= 2 without a top 15 fragment above 1.2x the precursor m/z
    ruled_out = [(any(m > 1.1 * mz and y >= 0.2 * max(peaks.values(), default = 0) for m, y in list(peaks.items())[:5]),
                  not any(m > 1.2 * mz for m in list(peaks)[:15])) for mz, peaks in zip(red_masses, df_in['peak_d'])]
    if ms1_ppm:
        # ... except at an instrument charge state with accurate survey scans, which domain_filter (trust_header_charge) takes as it is
        ruled_out = [(False, False) if kc is not None else out for out, kc in zip(ruled_out, known_charges)]
    # Compositions of the run's glycan class come first. Glycans of other classes do occur (high-mannose and complex N-glycans in O-glycan
    # preparations, O-glycans and glucose oligomers in N-glycan preparations), but their compositions (cross_class: database, composition map,
    # topology map) only fill spectra that no in-class composition explains at a charge state the spectrum allows or, with accurate survey scans
    # (ms1_ppm), within ms1_ppm, so they never compete with an in-class explanation
    pools = [(df_glycan_in, comp_struct_map, topo_struct_map, False)] + ([cross_class + (True,)] if cross_class else [])
    multi = [None] * len(red_masses)
    multi_all = [{} for _ in red_masses]
    for pool_df, pool_comp_map, pool_topo_map, fallback in pools:
        pool_comps = pool_df.groupby('comp_str').first()['Composition']
        comps_in = [x for c, x in pool_comps.items() if not fallback or c not in comp_struct_map]
        if sialic_derivatization is not None:  # a linkage-specific derivatization gives each way of linking the sialic acids a mass of its own
            comps_in = [split for x in comps_in for split in sialic_linkage_splits(x)]
        comp_masses = np.array([composition_to_mass(x, mass_value = 'monoisotopic', sample_prep = sample_prep, modification = modification,
                                                    sialic_derivatization = sialic_derivatization) + tag for x in comps_in])
        comps_with_none = [(x, pool_comp_map, pool_topo_map) for x in comps_in] + [None]
        # With accurate survey scans, a cross-class composition has to match the precursor m/z within ms1_ppm
        tolerances = ms1_ppm * red_masses / 1e6 if fallback and ms1_ppm else np.full(len(red_masses), mass_tolerance)

        def _match_chunked(candidate_masses):
            # The compositions within tolerance of each precursor, in composition order: the test of every precursor against every composition, but only
            # on the compositions in a window (slightly wider than the tolerance) of the sorted masses
            by_mass = np.argsort(candidate_masses, kind = 'stable')
            lo = np.searchsorted(candidate_masses[by_mass], red_masses - tolerances - 1e-6)
            n = np.searchsorted(candidate_masses[by_mass], red_masses + tolerances + 1e-6, side = 'right') - lo
            row_idx = np.repeat(np.arange(len(red_masses)), n)
            comp_idx = by_mass[np.repeat(lo - np.cumsum(n) + n, n) + np.arange(n.sum())]
            hit = np.abs(candidate_masses[comp_idx] - red_masses[row_idx]) < tolerances[row_idx]
            row_idx, comp_idx = row_idx[hit], comp_idx[hit]
            order = np.lexsort((comp_idx, row_idx))
            row_idx, comp_idx = row_idx[order], comp_idx[order]
            values, indices, _ = np.unique(row_idx, return_counts = True, return_index = True)
            subarrays = np.split(comp_idx, indices)[1:]
            comps_all = [None] * len(red_masses)
            for x, y in zip(values, subarrays):
                comps_all[x] = y
            # Each match keeps its smallest m/z error, which accurate survey scans can hold against it
            return [([comps_with_none[mc] for mc in x], np.min(np.abs(candidate_masses[x] - mz))) if x is not None else x
                    for x, mz in zip(comps_all, red_masses)]

        def _update(chunked, charge, replace = False):
            # Only overwrite where this scenario found a match, the slot is still empty (or, with replace, holds z = 1 matches that the
            # spectrum rules out and domain_filter would empty; for the fallback, a match that the spectrum rules out or, with accurate survey
            # scans, that misses the precursor m/z by more than ms1_ppm, by an allowed one), and any known charge agrees
            return [(y, charge) if ((not x[0] or (replace and heavy and x[1] == 1) or (fallback and (out[min(x[1], 2) - 1] or (
                    ms1_ppm and x[0][1] > ms1_ppm * mz / 1e6)) and not out[min(charge, 2) - 1])) and y and (kc is None or kc == charge)) else x
                    for x, y, kc, heavy, out, mz in zip(comps_out, chunked, known_charges, heavy_frags, ruled_out, red_masses)]

        # Try each charge state separately; higher charges produce lower observed m/z for the same neutral mass, so a ruled-out singly charged
        # match must not hide a multiply charged one at the same m/z (e.g., HexNAc1Neu5Ac2 [M-H]- vs. Hex3HexNAc4dHex2 [M-2H]2- at 804.3)
        for charge in range(1, abs(max_charge) + 1):
            if mode == 'negative':
                charged_comp_masses = (comp_masses - charge * PROTON_MASS) / charge
            else:
                charged_comp_masses = (comp_masses + charge * PROTON_MASS) / charge
            matches = _match_chunked(charged_comp_masses)
            comps_out = _update(matches, charge, replace = True)
            if charge > 1 and not fallback:
                multi = [x if x is not None else ((y, charge) if y else None) for x, y in zip(multi, matches)]
                for m_all, y in zip(multi_all, matches):
                    if y:
                        m_all[charge] = y
        valid_adducts = [(a, mass_dict[a]) for a in get_adduct_list(mode) if mass_dict.get(a, 999) != 999]
        for _, adduct_mass in valid_adducts:
            comps_out = _update(_match_chunked(comp_masses + adduct_mass), 1)
        # Multiply-charged adduct ions: only fill gaps not explained by protonated or singly-charged adduct matches
        threshold_dict = {2: 900, 3: 1500, 4: 3500}
        for _, adduct_mass in valid_adducts:
            for charge in range(2, abs(max_charge) + 1):
                threshold = threshold_dict.get(charge, 9999)
                if mode == 'negative':
                    charged_adduct_masses = (comp_masses + adduct_mass - (charge - 1) * PROTON_MASS) / charge
                else:
                    charged_adduct_masses = (comp_masses + adduct_mass + (charge - 1) * PROTON_MASS) / charge
                # Mask out compositions too small to realistically form multiply-charged adducts
                charged_adduct_masses = np.where(comp_masses + adduct_mass > threshold, charged_adduct_masses, 9999)
                comps_out = _update(_match_chunked(charged_adduct_masses), charge)
        # Pure multi-adduct ions: e.g., [M + 2Na]_2+, [M + 3Na]_3+ ; only fill remaining gaps
        for _, adduct_mass in valid_adducts:
            for charge in range(2, abs(max_charge) + 1):
                threshold = threshold_dict.get(charge, 9999)
                # All charge carriers are the adduct (no protons): m/z = (M + z×adduct) / z
                multi_adduct_masses = (comp_masses + charge * adduct_mass) / charge
                multi_adduct_masses = np.where(comp_masses + charge * adduct_mass > threshold, multi_adduct_masses, 9999)
                comps_out = _update(_match_chunked(multi_adduct_masses), charge)
    # A spectrum matched as singly charged whose most intense fragments come in pairs half a mass unit apart (domain_filter's sign of a multiply charged
    # precursor) also gets the multiply charged compositions of glycan_class at its m/z, and the annotation scores decide: ion traps read the 13C peak
    # of large [M-2H]2- next to a lighter [M-H]- composition (Hex5HexNAc4Neu5Ac1 965.8 as Hex1HexNAc1Neu5Ac2, Hex6HexNAc5Neu5Ac4 1585.1 as
    # Hex5HexNAc3dHex1), whose singly charged fragments lie above the scan range
    doublets = [any(0.4 <= q - j <= 0.62 and 0.3 * y <= w <= 1.5 * y for j, y in top[:5] for q, w in top)
                for top in (sorted(peaks.items(), key = lambda x: x[1], reverse = True)[:10] for peaks in df_in['peak_d'])]
    extra = [[m] if m and x[0] and x[1] == 1 and d and kc is None else [] for m, x, d, kc in zip(multi, comps_out, doublets, known_charges)]
    # Without an instrument charge, a multiply charged match also gets the in-class compositions of another multiply charged state at its m/z if more of
    # its 15 most intense fragments pair up into a B/C ion and its complement at that charge, and the annotation scores decide: ion traps do not resolve
    # z = 3 isotopes, so the lower charge, tried first, took every [M+3H]3+ of the trisialylated N-glycans of GPST000237 (Hex7HexNAc6Neu5Ac3dHex1 1157.09 as
    # Hex4HexNAc7dHex1 [M+2H]2+, its 657.2 B ion and the [M+3H-657]2+ complement 1407.0 aside). Such pairs favor z = 3 in 77% of its GT spectra and in
    # 0-2.8% of the z = 2 GT spectra of GPST000153 N, 374 N, 503, 596 and 134; without that condition GPST000153 N lost 0.064, GPST000487 0.026
    extra = [e + [(y, z) for z, y in m_all.items() if z != x[1] and complementary_pairs(mz, peaks, z, 0.75 * mass_tolerance) > complementary_pairs(
        mz, peaks, x[1], 0.75 * mass_tolerance)] if x[0] and x[1] >= 2 and kc is None and x[0][0][0][1] is comp_struct_map else e
             for e, x, m_all, kc, mz, peaks in zip(extra, comps_out, multi_all, known_charges, red_masses, df_in['peak_d'])]
    candidate_data = []
    for x, e in zip(comps_out, extra):
        matched = ([(c, x[1]) for c in x[0][0]] + [(c, z) for y, z in e for c in y[0]]) if x[0] else None
        if not matched:
            candidate_data.append(([None], [None], [None]))
        else:
            # Prefer topology-level structures when available; fall back to the most common structure for that composition. With a linkage-specific
            # derivatization, compositions carry their sialic linkages, which the structure is fitted to for CandyCrumbs
            pairs = [(s, comp, z) for (comp, pool_comp_map, pool_topo_map), z in matched for s in
                     pool_topo_map.get(stringify_dict(plain_composition(comp)), [pool_comp_map[stringify_dict(plain_composition(comp))]])]
            candidate_data.append(([s if sialic_derivatization is None else fit_sialic_linkages(s, c, scoring = True) for s, c, _ in pairs], [c for _, c, _ in pairs],
                                   [z for _, _, z in pairs]))
    df_in['candidate_structure'], df_in['composition'], df_in['charge'] = zip(*candidate_data)
    df_in['cross_class'] = [bool(x[0]) and x[0][0][0][1] is not comp_struct_map for x in comps_out]
    df_in = df_in.explode(['composition', 'candidate_structure', 'charge']).reset_index(names = 'spec_id')
    df_in['charge'] = df_in['charge'].astype(float)
    # Build a fixed-length composition vector aligned to comp_vector_order for model input
    df_in['compositional_vector'] = [
        np.array([plain_composition(x).get(m, 0) for m in comp_vector_order]) if x else None
        for x in df_in.composition
    ]
    return df_in


def assign_annotation_scores_pooled(df_in, multiplier, mass_tag, mass_tolerance, modification = 'reduced',
                                    sample_prep = 'underivatized', sialic_derivatization = None, n_jobs = 1, fragment_tolerance = None, fragment_ppm = None, deisotoped = None):
    mode = 'negative' if multiplier == -1 else 'positive'
    # Fragments are matched within mass_tolerance (the precursor tolerance, 0.48 Da at 300 ppm) unless the MS2 is accurate (fragment_tolerance); they are
    # pooled at 0.01 m/z then, as 0.1 m/z rounding alone would be off by up to 0.05. fragment_ppm bounds each match further, pooled at 0.001 m/z (0.005 is
    # already 25 ppm at m/z 200)
    fragment_tolerance = fragment_tolerance or mass_tolerance
    decimals = 3 if fragment_ppm else 2 if fragment_tolerance < 0.1 else 1
    adduct_list = get_adduct_list(mode)
    idx_col = 'm/z' if 'm/z' in df_in.columns else 'reducing_mass'
    unq_structs = df_in[df_in['candidate_structure'].notnull()].groupby('candidate_structure').first().reset_index()
    # Candidates of another reducing end (re_offset, see wrap_inference) are scored with their own reducing end
    offsets = df_in['re_offset'].values if 're_offset' in df_in.columns else np.zeros(len(df_in))
    groups = df_in.groupby([df_in['candidate_structure'], offsets], sort = False).indices
    comp_map = dict(zip(unq_structs.candidate_structure, unq_structs.composition))
    charge_vals = np.asarray(df_in['charge'].values, dtype = float)
    mz_vals, peak_vals, spec_ids = df_in[idx_col].values, df_in['peak_d'].values, df_in['spec_id'].values
    scores_out, annotated_out, fragments_out = np.zeros(len(df_in)), np.zeros(len(df_in)), np.zeros(len(df_in))
    # A spectrum is a row of every candidate structure it matched, so its deisotoped fragments (only the 15 kept get rounded) and MS3 spectra are computed
    # once per charge state
    # The deisotoped fragments are kept in deisotoped across calls, keyed by the peak dict (alive as long as the caller's condensed clusters) and what else
    # goes into them: the reducing end check scores the same clusters again for every reducing end, and a pooled orbitrap or TOF spectrum of thousands of
    # peaks takes ~10 ms to deisotope
    rounded_masses, rounded_ms3, structs, jobs = {} if deisotoped is None else deisotoped, {}, [], []
    for (glycan, offset), rows in groups.items():
        grp_charges = charge_vals[rows]
        row_charge = np.nanmax(grp_charges) if not np.all(np.isnan(grp_charges)) else 1.0
        comp_mass = composition_to_mass(comp_map[glycan], sample_prep = sample_prep, modification = modification,
                                        sialic_derivatization = sialic_derivatization) + (mass_tag if mass_tag else 0) + offset
        charges_arr = np.abs(charge_vals[rows])
        spec_masses = mz_vals[rows] * charges_arr
        is_adduct = any(
            np.any(np.abs(comp_mass + mass_dict.get(adduct, 999) - spec_masses) < mass_tolerance) or
            np.any(np.abs(comp_mass + mass_dict.get(adduct, 999) * charges_arr - spec_masses) < mass_tolerance)
            for adduct in adduct_list)
        # Clusters of at least 20 MS2 spectra name a deisotoped pattern by its most intense peak (relabel), as their pooled noise puts a weak peak 1-3 Da
        # below most fragments, which deisotope_ms2 otherwise takes as their monoisotopic peak (10 or 40 spectra were worse)
        # Peaks above the heaviest fragment a precursor of this charge can give (its singly charged ion, plus a charge-reduced one) are noise or another
        # ion's, and in pooled spectra would otherwise take places among the 15 most intense fragments (Bruker ion-trap MS2 scans up to 2500 m/z carry many).
        # A single spectrum keeps them: there the freed places let chance annotations pass crumbs_thresh (5 of the 6 single spectra it newly passed were
        # in no ground truth)
        # A sparse spectrum is judged by its fragments above 2% of its most intense one: pooled from several spectra, a singly charged precursor carries noise
        # peaks among its 15 most intense fragments, which hid that GM3 (634.22 in GPST000239) or sialyl-Tn (513.31 in GPST000350) show little but the Neu5Ac
        # B ion. Multiply charged ones count all 15, as the floor let junk z = 2 rows through (GPST000487 617.32, GPST000029 1463.81)
        rounded_mass_rows, strong_counts = [], []
        for x, n, mz in zip(peak_vals[rows], df_in['num_spectra'].values[rows], mz_vals[rows]):
            if (key := (id(x), n == 1, n >= 20, row_charge, mz, decimals)) not in rounded_masses:
                # Nor is anything within 1.5 m/z of the precursor a fragment of any charge: the unfragmented precursor and co-isolated ions, which CandyCrumbs
                # explains as the intact M of every candidate (junk and tail rows passed crumbs_thresh on it)
                deiso = list(deisotope_ms2({m: i for m, i in x.items() if (n == 1 or m <= abs(row_charge) * mz + 1.5) and abs(m - mz) > 1.5}, int(abs(row_charge)), 0.05, relabel = n >= 20).items())[:15]
                # (rounded all at once, as np.round does each one)
                rounded = list(np.round(ys, decimals)) if (ys := np.array([y for y, _ in deiso])).dtype == np.float64 else [np.round(y, decimals) for y, _ in deiso]
                rounded_masses[key] = (rounded, len({y for y, (_, i) in zip(rounded, deiso) if abs(row_charge) > 1 or i >= 0.02 * deiso[0][1]}))
            rounded_mass_rows.append(rounded_masses[key][0])
            strong_counts.append(rounded_masses[key][1])
        # MS3 spectra (if the file has them) are pooled per isolated MS2 fragment, as CandyCrumbs only explains their peaks with fragments of
        # what that fragment is in this structure
        ms3_rows = [rounded_ms3[key] if (key := (spec, row_charge)) in rounded_ms3 else
                    rounded_ms3.setdefault(key,
                                           [(round(p), p, [np.round(y, decimals) for y in list(deisotope_ms2(x, int(abs(row_charge)), 0.05))[:15]]) for p, x
                                            in ms3])
                    for spec, ms3 in zip(spec_ids[rows], df_in['ms3'].values[rows])] if 'ms3' in df_in.columns else []
        pools = {None: (None, {x for y in rounded_mass_rows for x in y})}
        for key, p, masses in [x for y in ms3_rows for x in y]:
            pools.setdefault(key, (p, set()))[1].update(masses)
        structs.append((rows, rounded_mass_rows, ms3_rows, list(pools), strong_counts))
        jobs += [(CandyCrumbs, (glycan, unq_rounded_masses, fragment_tolerance),
                  dict(simplify = False, charge = int(multiplier * abs(row_charge)), disable_global_mods = (not is_adduct or mode == "negative"),
                       disable_X_cross_rings = True, disable_A_cross_rings = multiplier > 0, max_cleavages = 2, mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0) + offset,
                       sample_prep = sample_prep, ms3_precursor = ms3_precursor, sialic_derivatization = sialic_derivatization, mass_threshold_ppm = fragment_ppm))
                 for ms3_precursor, unq_rounded_masses in pools.values()]
    # Every CandyCrumbs call first, so that n_jobs processes can share them
    results = iter(run_jobs(jobs, n_jobs))
    for rows, rounded_mass_rows, ms3_rows, keys, strong_counts in structs:
        tester_mass_scores = {}
        for key in keys:
            cc_out = next(results)
            # Score each fragment mass by how many non-redundant Domon-Costello annotations it receives;
            # cross-ring (A/X) and internal (M) fragments are only counted when they appear alone or in small combinations
            for _k, _v in cc_out.items():
                if _v:
                    _filtered = []
                    for ant in _v['Domon-Costello nomenclatures']:
                        prefs = [a.split('_')[0][-1] for a in ant]
                        if ('A' in prefs or 'X' in prefs) and len(prefs) > 1:
                            continue
                        if 'M' in prefs and len(prefs) > 2:
                            continue
                        _filtered.append(ant)
                    tester_mass_scores[(key, _k)] = len(_filtered)
                else:
                    tester_mass_scores[(key, _k)] = 0
        row_scores = [sum([tester_mass_scores[(None, x)] for x in y]) for y in rounded_mass_rows]
        # A sparse spectrum (a few fragments above the noise) cannot reach crumbs_thresh, so how many of its fragments are annotated counts too
        annotated_out[rows] = [sum(tester_mass_scores[(None, x)] > 0 for x in set(y)) for y in rounded_mass_rows]
        fragments_out[rows] = strong_counts
        # A row's MS3 evidence is the mean score of its MS3 spectra, so it does not grow with how often the instrument repeated them
        if ms3_rows:
            row_scores = [score + (np.mean([sum(tester_mass_scores[(key, x)] for x in masses) for key, _, masses in ms3]) if ms3 else 0)
                          for score, ms3 in zip(row_scores, ms3_rows)]
        scores_out[rows] = row_scores
    df_in['annotation_score'] = scores_out
    df_in['annotated'], df_in['fragments'] = annotated_out, fragments_out
    return df_in


def deduplicate_predictions(df, mz_diff = 0.5, rt_diff = 1.0, mz_ppm = None):
    """removes/unifies duplicate predictions\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference
   | mz_diff (float): mass tolerance for assigning spectra to the same peak; default:0.5
   | rt_diff (float): retention time tolerance (in minutes) for assigning spectra to the same peak; default:1.0
   | mz_ppm (float): relative mass tolerance (ppm) for assigning spectra to the same peak, used instead of mz_diff; default:None\n
   | Returns:
   | :-
   | Returns a deduplicated dataframe
   """
    # Sort by index and 'RT' (the index sort has to be stable to keep the RT order within one m/z)
    df.sort_values(by = 'RT', inplace = True)
    df.sort_index(kind = 'stable', inplace = True)
    idx_vals = df.index.values
    rt_vals = df['RT'].values
    preds_col = df['predictions'].values
    first_preds = np.array([p[0][0] if p else None for p in preds_col], dtype = object)
    conf_first = np.array([p[0][1] if p else -np.inf for p in preds_col])
    abund = df['rel_abundance'].values if 'rel_abundance' in df.columns else None
    window = idx_vals * mz_ppm / 1e6 if mz_ppm else mz_diff
    lo = np.searchsorted(idx_vals, idx_vals - window, side = 'right')
    hi = np.searchsorted(idx_vals, idx_vals + window, side = 'left')
    # Each row is represented by the row with its top prediction within mz_diff and rt_diff whose confidence times abundance is highest (rows without a
    # prediction by themselves): by confidence alone, a peak 30x weaker but more confident 0.9 min away carried the main peak's abundance at its own RT
    # (timsTOF, A2G2S2 at 12.61 instead of 13.55)
    rank = conf_first.copy()
    if abund is not None:
        has_pred = np.isfinite(conf_first)
        rank[has_pred] = conf_first[has_pred] * (np.nan_to_num(abund[has_pred].astype(float)) + 1e-12)
    reps = np.arange(len(df))
    ranges = df['rt_range'].values if 'rt_range' in df.columns else None
    for k in range(len(df)):
        if first_preds[k] is not None:
            window = np.arange(lo[k], hi[k])
            sel = window[(np.abs(rt_vals[window] - rt_vals[k]) < rt_diff) & (first_preds[window] == first_preds[k])]
            if ranges is not None:
                # Clusters that condense_dataframe kept apart are separate chromatographic peaks (isomers, or anomers of a free reducing end), unless their
                # spectra overlap in RT or their apexes lie within 0.1 min (one peak split by a single dip in precursor intensity)
                sel = sel[[j == k or max(ranges[j][0] - ranges[k][1], ranges[k][0] - ranges[j][1]) <= 0 or abs(rt_vals[j] - rt_vals[k]) <= 0.1 for j in sel]]
            reps[k] = sel[np.argmax(rank[sel])]
    # A row without a prediction one or two isotope spacings above a co-eluting predicted row of the same charge is the 13C peak of that glycan,
    # fragmented on its own (e.g., a header charge kept its precursor from being refined), so it joins that row instead of claiming the peak
    charges = df['charge'].values
    for k in np.flatnonzero([p is None for p in first_preds]):
        iso = [j for n in (1, 2) for j in np.flatnonzero(np.abs(idx_vals[k] - n * ISOTOPE_SPACING / abs(charges[k]) - idx_vals) <= 0.25)
               if first_preds[j] is not None and charges[j] == charges[k] and abs(rt_vals[j] - rt_vals[k]) < rt_diff]
        if iso:
            reps[k] = reps[max(iso, key = lambda j: conf_first[j])]
    # A predicted row one or two isotope spacings above a co-eluting, more abundant predicted row of the same charge, whose MS2 spectrum is the same (cosine
    # >= 0.8 of square-root intensities in 1 Da bins), is that glycan's 13C peak fragmented on its own: an ion trap isolates both, so its fragments are the
    # lighter glycan's, which other rules cannot tell from a glycan 1 Da heavier (Neu5Ac vs dHex2, Neu5Gc vs Hex + dHex); it joins that row
    # (the vectors only of the rows compared, as they come up: orbitrap and TOF spectra hold thousands of peaks)
    vectors, peak_col = {}, df['peak_d'].values if 'peak_d' in df.columns else None
    for k in np.flatnonzero([p is not None for p in first_preds]) if peak_col is not None else []:
        iso = [j for n in (1, 2) for j in np.flatnonzero(np.abs(idx_vals[k] - n * ISOTOPE_SPACING / abs(charges[k]) - idx_vals) <= 0.25)
               if first_preds[j] is not None and charges[j] == charges[k] and abs(rt_vals[j] - rt_vals[k]) < rt_diff and (abund is None or abund[j] > abund[k])]
        for x in [k] + iso if iso else []:
            if x not in vectors:
                v = defaultdict(float)
                for m, i in (peak_col[x].items() if isinstance(peak_col[x], dict) else []):
                    v[int(round(m))] += i
                # the total once per spectrum, not once per bin (quadratic: ~1 s per call for orbitrap or TOF spectra of thousands of peaks)
                total = sum(v.values()) or 1
                vectors[x] = {b: np.sqrt(i / total) for b, i in v.items()}
        cos = {j: sum(y * vectors[k].get(b, 0) for b, y in vectors[j].items()) for j in iso}
        if cos and max(cos.values()) >= 0.8:
            reps[k] = reps[max(cos, key = cos.get)]
    # A predicted row one or two isotope spacings above a co-eluting, more abundant row of the same charge, whose composition is that row's with a Neu5Ac read as
    # two dHex or a Neu5Gc as Hex + dHex (1.02 Da heavier), is that row's 13C peak read as the heavier isobar, whatever its spectrum (ion-trap MS2 of such z = 2
    # pairs had cosines of 0.65-0.79) or the lighter row's prediction; it joins that row
    comps = df['composition'].values if 'composition' in df.columns else [None] * len(df)
    isobars = ({'dHex': 2, 'Neu5Ac': -1}, {'Hex': 1, 'dHex': 1, 'Neu5Gc': -1})
    for k in np.flatnonzero([p is not None and isinstance(c, dict) for p, c in zip(first_preds, comps)]):
        iso = [j for n in (1, 2) for j in np.flatnonzero(np.abs(idx_vals[k] - n * ISOTOPE_SPACING / abs(charges[k]) - idx_vals) <= 0.25)
               if isinstance(comps[j], dict) and charges[j] == charges[k] and abs(rt_vals[j] - rt_vals[k]) < rt_diff and (abund is None or abund[j] > abund[k])
               and any(all(comps[k].get(r, 0) - comps[j].get(r, 0) == d.get(r, 0) for r in set(comps[k]) | set(comps[j]) | set(d)) for d in isobars)]
        if iso:
            reps[k] = reps[iso[0]]
    dedup_df = df.iloc[pd.unique(reps)].copy()
    if abund is not None:
        # A representative carries the abundance of the rows it represents, so a row linking two representatives is not counted twice
        dedup_df['rel_abundance'] = [np.nansum(abund[reps == r]) for r in pd.unique(reps)]
        dedup_df = dedup_df.astype({'rel_abundance': df['rel_abundance'].dtype})
    return dedup_df


def domain_filter(df_out, glycan_class, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized',
                  max_charge = -3, mass_tolerance = 0.5, filter_out = set(), df_use = None, mass_tag = None, sialic_derivatization = None,
                  trust_header_charge = False, fragment_tolerance = None, fragment_ppm = None):
    """filters out false-positive predictions\n
   | Arguments:
   | :-
   | df_out (dataframe): df_out generated within wrap_inference
   | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:set()
   | df_use (dataframe): glycan database used to check whether compositions are valid; default: df_glycan
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS), whose products the sialic
   |                                          acid ions and losses carry; default:None
   | trust_header_charge (bool): whether a row's charge is beyond doubt where it equals its precursor_charge column (the instrument's charge state, from
   |                             accurate survey scans), so the fragment size checks of the charge state are skipped; default:False
   | fragment_tolerance (float): fragment mass tolerance of accurate MS2 (orbitrap, TOF), with which the sialic acid ions are weighed by intensity; default:None
   | fragment_ppm (float): relative fragment mass tolerance (ppm), used instead of fragment_tolerance; default:None\n
   | Returns:
   | :-
   | Returns a filtered prediction dataframe
   """
    if df_use is None:
        df_use = df_glycan
    multiplier = -1 if mode == 'negative' else 1
    # Identify which spectra carry common adducts so downstream checks can account for the mass offset
    adduct_list = get_adduct_list(mode)
    tag = mass_tag if mass_tag else 0
    computed_masses = np.array(
        [composition_to_mass(comp, sample_prep = sample_prep, modification = modification, sialic_derivatization = sialic_derivatization) + tag for comp in
         df_out['composition'].values]) + (df_out['re_offset'].values if 're_offset' in df_out.columns else 0)
    # Residue masses of the sample's derivatization; a residue that leaves the glycan (lost as a neutral, or as its ion less the charge carrier) also takes the
    # group capping the bond it left by (CH2 when permethylated, C2H2O when peracetylated), e.g., permethylated Neu5Ac 375.19 (ion 376.20), not 291.10 (292.10).
    # With underivatized masses, sialylated permethylated structures were only kept when some fragment happened to lie within 1 Da of one of them, which held
    # for one charge state of a glycan and not for another (A2G2S2 [M+3H]3+ 929.48 lost, [M+2H]2+ 1393.72 kept)
    if sample_prep == 'underivatized':
        res_mass, cap = mass_dict, 0.0
    else:
        base = composition_to_mass({}, sample_prep = sample_prep)
        res_mass = {r: composition_to_mass({r: 1}, sample_prep = sample_prep) - base for r in ('Hex', 'HexNAc', 'dHex', 'Pen', 'Neu5Ac', 'Neu5Gc', 'Kdn', 'HexA', 'HexN')}
        cap = (base - composition_to_mass({})) / 2
    # A derivatized sialic acid leaves its ion, or is lost, as its a2-6 or its lactone-derived product
    sia_masses = {sia: [res_mass[sia] + cap] if sialic_derivatization is None else [composition_to_mass({k: 1}, sialic_derivatization = sialic_derivatization) - composition_to_mass({})
                                                                                   for k in SIALIC_KEYS[sia]] for sia in SIALIC_KEYS}
    charges_abs = np.abs(df_out['charge'].values)
    raw_masses = df_out.index.values * charges_abs
    df_out['adduct'] = None
    for adduct in adduct_list:
        adduct_mass = mass_dict.get(adduct, 999)
        if adduct_mass == 999:
            continue
        # Singly-charged adduct: mz × |z| = M + adduct
        df_out.loc[np.abs(computed_masses + adduct_mass - raw_masses) < mass_tolerance, 'adduct'] = adduct
        # Multiply-charged adduct: mz × |z| = M + adduct − (|z|−1)×H in negative mode, + (|z|−1)×H in positive mode
        proton_offset = (charges_abs - 1) * PROTON_MASS
        df_out.loc[(charges_abs > 1) & (
                np.abs(computed_masses + adduct_mass + multiplier * proton_offset - raw_masses) < mass_tolerance), 'adduct'] = adduct
        # A multiply charged precursor may carry only cations, e.g., [M+2Na]2+ (assign_candidate_structures matches these too)
        df_out.loc[(charges_abs > 1) & (np.abs(
            computed_masses + charges_abs * adduct_mass - raw_masses) < mass_tolerance), 'adduct'] = adduct
    # In ion-trap CID an anion adduct ([M+Formate]-, [M+HCO3-H]2-, ...) mostly fragments as the glycan, without the (de)protonated ion its neutral loss
    # would give (GPST000578: none of 91 such rows with a model prediction had it among its 10 most intense fragments), so an adduct reading also holds if
    # the (de)protonated ion of its composition co-elutes (within 1 min), which confirms the composition behind it
    keys = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
    plain_rts = defaultdict(list)
    for key, a, rt, preds in zip(keys, df_out['adduct'], df_out['RT'], df_out['predictions']):
        if key and not isinstance(a, str) and len(preds):
            plain_rts[key].append(rt)
    coeluting = [mode == 'negative' and isinstance(a, str) and any(abs(rt - r) <= 1.0 for r in plain_rts.get(key, [])) for key, a, rt in zip(keys, df_out['adduct'], df_out['RT'])]
    new_preds = []
    top_fragments_col = df_out['top_fragments'].tolist()
    peak_col = df_out['peak_d'].tolist()
    predictions_col = df_out['predictions'].tolist()
    adduct_col = df_out['adduct'].tolist()
    charge_col = df_out['charge'].tolist()
    index_vals = df_out.index.tolist()
    header_col = df_out['precursor_charge'].tolist() if trust_header_charge and 'precursor_charge' in df_out.columns else [None] * len(df_out)
    double_mass_tolerance = 2 * mass_tolerance
    has_sulfate = {}
    # Sialic acid ions as a share of the base peak per spectrum (None where the scan range starts above all of them), for accurate MS2
    accurate_sia = None
    if fragment_tolerance is not None and sialic_derivatization is None:
        water = {'permethylated': calculate_adduct_mass('CH4O'), 'peracetylated': calculate_adduct_mass('C2H4O2')}.get(sample_prep, mass_dict['H2O'])
        ions = {sia: [sm + a + PROTON_MASS * multiplier - w for sm in sia_masses[sia] for a in (0, res_mass['Hex'] + res_mass['HexNAc']) for w in (0, water)]
                for sia in ('Neu5Ac', 'Neu5Gc')}
        accurate_sia = []
        for pk in peak_col:
            if not isinstance(pk, dict) or not pk:
                accurate_sia.append({})
                continue
            mz_arr, int_arr = np.fromiter(pk.keys(), float, len(pk)), np.fromiter(pk.values(), float, len(pk))
            levels = {}
            for sia, targets in ions.items():
                seen = [t for t in targets if t > mz_arr.min()]
                tol = np.array(seen)[None, :] * fragment_ppm / 1e6 if fragment_ppm else fragment_tolerance
                hits = int_arr[np.any(np.abs(mz_arr[:, None] - np.array(seen)[None, :]) < tol, axis = 1)] if seen else []
                levels[sia] = (hits.max() / int_arr.max() if len(hits) else 0.0) if seen else None
            accurate_sia.append(levels)
    for k in range(len(df_out)):
        keep = []
        c = abs(charge_col[k])
        addy = (c - 1) * -multiplier * PROTON_MASS
        precursor_mz = index_vals[k]
        assumed_mass = precursor_mz * c + addy
        current_preds = predictions_col[k]
        if len(current_preds) == 0:
            new_preds.append(keep)
            continue
        top_frags = top_fragments_col[k]
        # 15% of the base peak, for the sialylated antenna ions below (once per row, not once per peak, ion, and candidate)
        base_floor = 0.15 * max(peak_col[k].values()) if isinstance(peak_col[k], dict) and peak_col[k] else None
        float_frags = [j for j in top_frags if isinstance(j, float)]
        adduct_name = adduct_col[k]
        for i, m in enumerate(current_preds):
            m = m[0]
            truth = [True]
            # Diagnostic ions: a sialic acid in the structure must leave its diagnostic fragment (its ion, or the precursor without one or more of its sialic acids)
            for sia in ('Neu5Ac', 'Neu5Gc', 'Kdn'):
                if sia in m:
                    losses = [n * sia_masses[sia][0] for n in range(1, m.count(sia) + 1)] if sialic_derivatization is None else [
                        sum(x) for n in range(1, m.count(sia) + 1) for x in
                        combinations_with_replacement(sia_masses[sia], n)]
                    # A sialylated LacNAc antenna also shows as its B and C ions (Neu5Ac-Hex-HexNAc 655.2 / 673.2), e.g., in ion traps whose low-mass cutoff
                    # hides the sialic acid ion of a large precursor: among the 5 most intense fragments, or anywhere with at least 15% of the base peak
                    # (z = 2 sialylated bi- and triantennary N-glycans, whose doubly charged losses outrank it, carried it at 16-39% of the base peak, ranks 10-33)
                    # A floating sialic acid ({Neu5Ac(a2-6)}Gal(b1-4)GlcNAc...) sits on one of the structure's Gal(b1-4)GlcNAc antennas, which the attached form
                    # cannot match: every floating monosialylated tri- and tetraantennary N-glycan was emptied (GPST000503 1148.41, base peak 835.3, 655.2 at 87%)
                    floating = re.search(r'\{' + sia + r'\(a2-', m) is not None
                    lacnac = re.search(sia + r'[^()]*\(a2-[^)]*\)Gal\(b1-4\)GlcNAc', m) is not None or (floating and 'Gal(b1-4)GlcNAc' in m)
                    # ... and, on an N-glycan antenna, together with the mannose it sits on (Neu5Ac-Hex-HexNAc-Hex C ion 835.3, the base peak of many z = 2
                    # monosialylated N-glycans in ion traps, whose 290 ion lies below the low-mass cutoff and which lose no sialic acid as a charged fragment)
                    antenna = [res_mass['Hex'] + res_mass['HexNAc']] + ([2 * res_mass['Hex'] + res_mass['HexNAc']] if re.search(
                        sia + r'[^()]*\(a2-[^)]*\)Gal\(b1-4\)GlcNAc\(b1-[^)]*\)Man', m) or (floating and re.search(r'Gal\(b1-4\)GlcNAc\(b1-[^)]*\)Man', m)) else [])
                    truth.append(any(
                        any(abs(sm + PROTON_MASS * multiplier - j) < double_mass_tolerance for sm in sia_masses[sia]) or
                        (lacnac and j in float_frags[:5] and any(
                            abs(sm + a + PROTON_MASS * multiplier + w - j) <
                            mass_tolerance / 2 for sm in sia_masses[sia] for a in antenna for w in (0, mass_dict['H2O']))) or
                                     any(abs(assumed_mass - loss - j) < double_mass_tolerance for loss in losses) or
                                     any(abs(precursor_mz - ((sm - addy) / c) - j) < double_mass_tolerance for sm in sia_masses[sia]) for j
                                     in float_frags) or (lacnac and any(
                        y >= base_floor and abs(sm + a + PROTON_MASS * multiplier + w - j) <
                        mass_tolerance / 2 for j, y in peak_col[k].items() for sm in sia_masses[sia] for a in antenna for w in (0, mass_dict['H2O']))))
            if accurate_sia is not None:
                # Accurate MS2 tells Neu5Ac from Neu5Gc by their own ions (each also less MeOH / H2O, and with Hex + HexNAc of its antenna, observable above the
                # scan's lowest m/z), which the checks above match anywhere within 1 Da: each sialic acid of the structure needs at least 5% of the base peak,
                # and one it lacks must stay below half of it. The exact isobars Neu5Gc + dHex and Neu5Ac + Hex were both kept and the 1 Da window found some
                # fragment for either
                for sia, level in accurate_sia[k].items():
                    if level is not None and ((sia in m and level < 0.05) or (sia not in m and level >= 0.5)):
                        truth.append(False)
            if 'Neu5Gc' not in m:
                truth.append(not any(abs(sm + PROTON_MASS * multiplier - j) < mass_tolerance
                                     for j in top_frags[:5] if isinstance(j, float) for sm in sia_masses['Neu5Gc']))
            if 'Neu5Ac' not in m and 'Neu5Gc' not in m:
                truth.append(not any(abs(sm + PROTON_MASS * multiplier - j) < mass_tolerance
                                     for j in top_frags[:5] if isinstance(j, float) for sm in sia_masses['Neu5Ac']))
            if 'Neu5Ac' not in m and (m.count('Fuc') + m.count('dHex') > 1):
                # Neu5Ac (291.095) and two dHex (292.116) differ by 1.02 Da, inside 2x mass_tolerance, and the 13C peak of a loss of two dHex sits 0.02 Da from a
                # loss of Neu5Ac: a Neu5Ac ion or loss only counts within half the mass tolerance and if it is not the 13C peak of a more intense peak below it
                pk = peak_col[k]
                truth.append(not any((abs(sm + PROTON_MASS * multiplier - j) < mass_tolerance / 2 or abs(precursor_mz - sm - j) < mass_tolerance / 2) and not any(
                    abs(j - ISOTOPE_SPACING - q) < 0.15 and y > max([v for p, v in pk.items() if abs(p - j) < 0.01], default = 0) for q, y in pk.items())
                                     for j in top_frags[:10] if isinstance(j, float) for sm in sia_masses['Neu5Ac']))
            if 'S' in m and len(current_preds) == 1:
                # Rows exploded from one spectrum share top_frags, so each fragment's composition lookup is cached
                for t in top_frags[:20]:
                    if t not in has_sulfate:
                        has_sulfate[t] = 'S' in (mz_to_composition(t, max_charge = max_charge, mass_tolerance = mass_tolerance,
                                                                   glycan_class = glycan_class if isinstance(glycan_class, str) else 'all', df_use = df_use,
                                                                   filter_out = filter_out, modification = modification,
                                                                   sample_prep = sample_prep)[0:1] or ({},))[0]
                    if has_sulfate[t]:
                        break
                truth.append(any(has_sulfate[t] for t in top_frags[:20]))
            # Check fragment size distribution: a multiply charged precursor shows singly charged fragments above its m/z, or loses one of its
            # residues at its own charge, which a singly charged ion cannot (large N-glycans in ion traps, e.g. Man9 [M-2H]2- 941.3 -> 860.3, -Hex/2),
            # or shows one of its 5 most intense fragments with a 13C peak half a mass unit above it among the 10 most intense.
            # A charge state the instrument read off resolved isotopes of an accurate survey scan needs no such check: timsTOF beam-type CID breaks
            # [M-2H]2- sialylated N-glycans into small B/C ions, so the check emptied A2G2S2 at 1111.39 and every other z = 2 glycan of a serum run
            trusted = header_col[k] is not None and not pd.isna(header_col[k]) and abs(int(header_col[k])) == c
            if c > 1 and not trusted:
                top10 = sorted(peak_col[k].items(), key = lambda x: x[1], reverse = True)[:10]
                multiply_charged = any(j > precursor_mz * 1.2 for j in top_frags[:15]) or any(
                    abs(precursor_mz - rm / c - j) < mass_tolerance / 2 for j in float_frags[:3] for r in
                    get_comp(m) if r in res_mass for rm in sia_masses.get(r, [res_mass[r] + cap])) or any(
                    0.4 <= q - j <= 0.62 and 0.3 * y <= w <= 1.5 * y for j, y in top10[:5] for q, w in top10)
                # Without these signs, the charge state still holds if CandyCrumbs explains at least 3 of the 20 most intense peaks (one per mass_tolerance)
                # only as fragments of a single glycosidic cleavage above charge 1, with no singly charged fragment of the structure at that peak (z = 2
                # N-glycans in ion traps whose doubly charged Y ions dominate)
                if not multiply_charged and all(truth) and sialic_derivatization is None:
                    peaks = []
                    for j in sorted(peak_col[k], key = peak_col[k].get, reverse = True)[:20]:
                        if all(abs(j - q) > mass_tolerance for q in peaks):
                            peaks.append(j)
                    try:
                        cc_out = CandyCrumbs(m.replace('-ol', '').replace('1Cer', ''), peaks, mass_tolerance, simplify = False, charge = int(charge_col[k]),
                                             disable_global_mods = True, disable_X_cross_rings = True, max_cleavages = 2, sample_prep = sample_prep,
                                             mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0) + (
                                                 df_out['re_offset'].values[k] if 're_offset' in df_out.columns else 0))
                    except Exception:
                        cc_out = {}
                    multiply_charged = sum(bool(v) and all(abs(q) > 1 for q in v['Fragment charges']) and any(
                        len(d) == 1 and d[0].split('_')[0] in ('B', 'C', 'Y', 'Z') for d in v['Domon-Costello nomenclatures']) for v in cc_out.values()) >= 3
                truth.append(multiply_charged)
            if c == 1 and not trusted:
                # A singly charged precursor cannot have fragments above its m/z; a weak one (below 20% of the base peak) is a co-isolated ion, not a wrong charge
                top_peaks = sorted(peak_col[k].items(), key = lambda x: x[1], reverse = True)[:5]
                truth.append(all(j < precursor_mz * 1.1 or y < 0.2 * top_peaks[0][1] for j, y in top_peaks))
            if len(top_frags) < 2:
                truth.append(False)
            # Check neutral loss of adduct for adducts
            # An anion adduct (or NH4+, as NH3) leaves as a neutral and gives the (de)protonated ion; a metal cation stays on the fragments
            if isinstance(adduct_name, str) and (mode == 'negative' or adduct_name == 'NH4+'):
                neutral_loss = mass_dict.get(adduct_name, 999) - (PROTON_MASS * multiplier)
                expected_frag = precursor_mz - neutral_loss / c
                truth.append(coeluting[k] or any(abs(expected_frag - j) < mass_tolerance for j in top_frags[:10]))
            if all(truth):
                keep.append(current_preds[i])
        new_preds.append(keep)
    df_out['predictions'] = new_preds
    return df_out


def impute(df_out, pred_thresh, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized', mass_tag = 0.0, glycan_class = "O",
           sialic_derivatization = None):
    """searches for specific isomers that could be added to the prediction dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | pred_thresh (float): confidence given to an imputed prediction
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default:'negative'
    | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:0.0
    | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together; default:"O"
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS); variants with a sialic acid of
    |                                          unknown linkage then have no mass and are not imputed; default:None\n
    | Returns:
    | :-
    | Returns prediction dataframe with imputed predictions (if possible)
    """

    def _get_all_variants(iupac_string, pattern1, pattern2):
        # Generate every combination of forward and reverse substitutions between two interchangeable monosaccharides
        p1_count = iupac_string.count(pattern1)
        all_variants = {iupac_string}
        for fwd in range(p1_count + 1):
            variant = iupac_string.replace(pattern1, pattern2, fwd)
            all_variants.add(variant)
            for rev in range(1, p1_count - fwd + 1):
                all_variants.add(pattern2.join(variant.rsplit(pattern1, rev)))
        final_strings = all_variants.copy()
        for current_variant in all_variants:
            p2_count = current_variant.count(pattern2)
            for fwd in range(p2_count + 1):
                variant = current_variant.replace(pattern2, pattern1, fwd)
                final_strings.add(variant)
                for rev in range(1, p2_count - fwd + 1):
                    final_strings.add(pattern1.join(variant.rsplit(pattern2, rev)))
        return sorted(final_strings)

    predictions_list = df_out.predictions.values.tolist()
    index_list = df_out.index.tolist()
    charge_list = df_out.charge.tolist()
    seqs = [p[0][0] for p in predictions_list if p and ("Neu5Ac" in p[0][0] or "Neu5Gc" in p[0][0])]
    variants = set(unwrap([_get_all_variants(s, 'Neu5Ac', 'Neu5Gc') for s in seqs]))
    if "O" in glycan_classes(glycan_class):
        # O-glycans may carry a sulfated GlcNAc branch that is interchangeable with the unsulfated form
        seqs = [p[0][0] for p in predictions_list if p and ("GlcNAc6S(b1-6)" in p[0][0] or "GlcNAc(b1-6)" in p[0][0])]
        variants.update(unwrap([_get_all_variants(s, 'GlcNAc6S(b1-6)', 'GlcNAc(b1-6)') for s in seqs]))
    # Every empty row tries the same variants (first match in sorted order, as mass_check would find it), so their masses and ion m/z are
    # computed once per charge state instead of once per row and variant
    variants = sorted(variants)
    variant_masses, ion_mzs = [], {}
    for v in variants:
        try:
            variant_masses.append(glycan_to_mass(v, sample_prep = sample_prep, modification = modification, sialic_derivatization = sialic_derivatization))
        except Exception:
            variant_masses.append(np.nan)
    variant_masses = np.array(variant_masses) + (mass_tag or 0)
    for i, k in enumerate(predictions_list):
        if len(k) < 1 and variants:
            z = abs(charge_list[i])
            if z not in ion_mzs:
                ions = get_ion_mzs(variant_masses, max_charge = int(z) * (1 if mode == 'positive' else -1), adducts = get_adduct_list(mode),
                                   min_mass = {2: 900, 3: 1500, 4: 3500})
                ion_mzs[z] = np.array([m for ion, m in ions.items() if int(ion.rsplit(']', 1)[1][:-1] or 1) == z])
            hits = np.flatnonzero((np.abs(index_list[i] - ion_mzs[z]) < 0.5).any(axis = 0))
            if hits.size:
                df_out.iat[i, 0] = [(variants[hits[0]], pred_thresh)]
    return df_out


def possibles(df_out, mass_dic, mass_offset):
    """searches for known glycans that could explain the observed m/z value if we don't have a prediction there\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference
   | mass_dic (dict): dictionary of form mass : list of glycans
   | mass_offset (float): mass from reducing end modification\n
   | Returns:
   | :-
   | Returns prediction dataframe with imputed predictions (if possible)
   """
    predictions_list = df_out.predictions.values.tolist()
    top1_preds = {k[0][0] for k in predictions_list if k and k[0]}
    index_list = df_out.index.tolist()
    mass_keys = np.array(list(mass_dic.keys()))
    for k in range(len(df_out)):
        if len(predictions_list[k]) < 1:
            check_mass = index_list[k] - mass_offset
            diffs = np.abs(mass_keys - check_mass)
            min_diff_index = np.argmin(diffs)
            if diffs[min_diff_index] < 0.5:
                possible = mass_dic[mass_keys[min_diff_index]]
                df_out.iat[k, 0] = [(m,) for m in possible if m not in top1_preds]
    return df_out


_DIC_MASSES = {}


def make_mass_dic(glycans, glycan_class, filter_out, df_use, sample_prep = 'underivatized', sialic_derivatization = None):
    """generates a mass dict that can be used in the possibles() function\n
   | Arguments:
   | :-
   | glycans (list): glycans used for training CandyCrunch
   | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
   | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS); a glycan is then listed under
   |                                          the mass of every way its sialic acids can be linked, with the linkages that mass decides filled in; default:None\n
   | Returns:
   | :-
   | Returns a dictionary of form mass : list of glycans
   """
    exp_glycans = set(df_use.glycan.values.tolist())
    class_glycans = [k for k in glycans if in_classes(k, glycan_classes(glycan_class))]
    exp_glycans.update(class_glycans)
    mass_dic = {9999: []}
    for k in exp_glycans:
        try:
            composition = get_comp(k)
            if filter_out.intersection(composition):
                mass_dic[9999].append(k)
            elif sialic_derivatization is not None:
                for split in sialic_linkage_splits(composition):
                    if (resolved := resolve_sialic_linkages(k, split)) is not None:
                        mass_dic.setdefault(composition_to_mass(split, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization), []).append(resolved)
            else:
                # (kept across calls: every run weighs the same few thousand database and vocabulary glycans)
                mass = _DIC_MASSES.get((k, sample_prep))
                if mass is None:
                    mass = _DIC_MASSES[(k, sample_prep)] = glycan_to_mass(k, sample_prep = sample_prep)
                mass_dic.setdefault(mass, []).append(k)
        except Exception:
            mass_dic[9999].append(k)
    return mass_dic


def canonicalize_biosynthesis(df_out):
    """regularize predictions by incentivizing biosynthetic feasibility\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference\n
   | Returns:
   | :-
   | Returns prediction dataframe with re-ordered predictions, based on observed biosynthetic activities
   """
    df_out = df_out.assign(
        true_mass = df_out.index * abs(df_out['charge']) - (df_out['charge'] + np.sign(df_out['charge'].sum())))
    df_out.sort_values(by = 'true_mass', inplace = True)
    strong_evidence_preds = [pred[0][0] for pred, evidence in zip(df_out['predictions'], df_out['evidence']) if
                             len(pred) > 0 and evidence == 'strong']
    rest_top1 = set(strong_evidence_preds)
    # How many of those each candidate is part of, once per candidate (rows share most of their candidates)
    sub_counts = {}
    prediction_column = []
    for _, row in df_out[::-1].iterrows():
        new_preds = []
        preds = row['predictions']
        if len(preds) == 0:
            prediction_column.append(new_preds)
            continue
        for p in [x for x in preds if len(x) == 2 if x[1]]:
            p_list = list(p)
            if p_list[0] not in sub_counts:
                sub_counts[p_list[0]] = sum(subgraph_isomorphism(p_list[0], t) for t in rest_top1 if t != p_list[0])
            p_list[1] += 0.1 * sub_counts[p_list[0]]
            new_preds.append(tuple(p_list))
        new_preds.sort(key = lambda x: x[1], reverse = True)
        total = sum(p[1] for p in new_preds)
        if total > 1:
            new_preds = [(p[0], p[1] / total) for p in new_preds][:5]
        else:
            new_preds = new_preds[:5]
        prediction_column.append(new_preds)
    df_out['predictions'] = prediction_column[::-1]
    df_out.drop(['true_mass'], axis = 1, inplace = True)
    return df_out.sort_index()


class DictStorage:
    @staticmethod
    def serialize_dict(d: Dict) -> str:
        """Convert float dictionary to JSON string with string keys"""
        return json.dumps({str(k): v for k, v in d.items()})

    @staticmethod
    def deserialize_dict(s: str) -> Dict[float, float]:
        """Convert JSON string back to dictionary with float types"""
        return {float(k): v for k, v in json.loads(s).items()}

    def store(self, df: pd.DataFrame, dict_col: str, filepath: str) -> None:
        """Store DataFrame with dictionary column as CSV"""
        df = df.copy()
        df[dict_col] = df[dict_col].apply(self.serialize_dict)
        df.to_csv(filepath, index = False)

    def read(self, filepath: str, dict_col: str) -> pd.DataFrame:
        """Read CSV and convert dictionary column back to dicts"""
        df = pd.read_csv(filepath)
        df[dict_col] = df[dict_col].apply(self.deserialize_dict)
        return df


def load_spectra_filepath(spectra_filepath, extract_ms1 = False):
    # A timsTOF .d is a folder, which may come with a trailing separator
    ext = os.path.splitext(os.path.normpath(spectra_filepath))[1].lower()
    if ext == ".d":
        # timsTOF folders hold an analysis.tdf, the ion traps (esquire, HCT, amaZon) an Analysis.yep
        stack = process_yep_stack if os.path.exists(os.path.join(spectra_filepath, 'Analysis.yep')) else process_tdf_stack
        return stack(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
    if ext in (".mzml", ".mzxml"):
        try:
            return (process_mzML_stack if ext == ".mzml" else process_mzXML_stack)(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
        except SyntaxError as e:
            # The XML parser's ParseError (a SyntaxError) does not say that the file is incomplete or no mzML/mzXML at all
            raise ValueError(f"{os.path.basename(spectra_filepath)} is not a complete {'mzML' if ext == '.mzml' else 'mzXML'} file ({e}); it may have been cut "
                             f"short while being copied or converted") from e
    if ext == ".raw":
        return process_raw_stack(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
    if ext == ".mgf":
        rows, agilent = [], []
        for spectrum in read_mgf(spectra_filepath):
            params = spectrum['params']
            # msconvert writes the native ID into TITLE; scanId= is Agilent MassHunter's format (PSI-MS MS:1001508), whose MS/MS instruments are Q-TOFs
            agilent.append('NativeID:"scanId=' in str(params.get('title', '')))
            # Like MS1 scans of mzML files, spectra without a precursor m/z (no or an empty PEPMASS) are not predicted
            if not len(spectrum['m/z array']) or params.get('pepmass', (None,))[0] is None:
                continue
            if 'rtinseconds' not in params:
                raise ValueError(f"MGF spectrum '{params.get('title', '')}' has no RTINSECONDS entry; CandyCrunch needs retention times")
            # Same conventions as process_mzML_stack: top 1000 peaks sorted by intensity, charge 1 treated as undetermined; so is CHARGE=0, which some
            # writers use for an unknown charge and which as a known charge left no candidate composition for any spectrum
            charge = (abs(int(params['charge'][0])) or None) if params.get('charge') else None
            peak_d = dict(sorted(zip(spectrum['m/z array'].tolist(), spectrum['intensity array'].tolist()), key = lambda x: x[1], reverse = True)[:1000])
            rows.append([float(params['pepmass'][0]), peak_d, float(params['rtinseconds']) / 60,
                         charge if charge != 1 else None, params['pepmass'][1] if params['pepmass'][1] is not None else np.nan,
                         int(params['scans']) if str(params.get('scans')).isdigit() else params.get('scans'), None])
        df_out = pd.DataFrame(rows, columns = ['m/z', 'peak_d', 'RT', 'precursor_charge', 'intensity', 'scan', 'activation'])
        # An mgf names no analyzer; Agilent Q-TOF precursors and fragments are accurate to a few ppm (MS1 'tof', MS2 'other', as for timsTOF). Its precursor
        # intensities are the survey abundance when DDA triggered, not a trace (Spearman 0.15/0.18 with MassHunter feature heights in GPST000570, spectrum
        # counts 0.70/0.56), so they are dropped, as no valley or apex can be read from them
        if agilent and all(agilent):
            df_out.attrs.update(detected_trap = 'other', detected_ms1_trap = 'tof')
            df_out['intensity'] = np.nan
        return df_out
    if ext == ".cc":
        # extract_spectra's binary file: numpy arrays stored as byte planes, numbers that are not peaks as loaded, a JSON header with the rest
        with np.load(spectra_filepath) as f:
            meta = json.loads(f['meta'].tobytes())
            arrays = {k: np.ascontiguousarray(f[k].T).view(meta['dtypes'][k]).ravel() for k in f.files if k != 'meta'}
        # A .cc holds what the loader of its time made of the raw file (refined precursors, centroids, charge states, dropped isotope triggers), so later loader
        # changes only reach it by extracting the raw file again
        if meta.get('loader') != SPECTRA_LOADER:
            print(f"WARNING: {os.path.basename(spectra_filepath)} was written by another version of CandyCrunch's spectra loader; extract it again from its raw file "
                  f"({meta.get('source')}) to get the current precursor refinement, centroiding and charge states.")
        dicts = lambda n, mzs, ints: [dict(zip(m.tolist(), i.tolist())) for m, i in zip(np.split(mzs.astype(float), np.cumsum(n)[:-1]), np.split(ints.astype(float), np.cumsum(n)[:-1]))]
        spectra = {'peak_d': dicts(arrays['peak_n'], arrays['peak_mz'], arrays['peak_int'])}
        if 'ms3_n' in arrays:
            ms3 = list(zip(arrays['ms3_precursor'].tolist(), dicts(arrays['ms3_peak_n'], arrays['ms3_peak_mz'], arrays['ms3_peak_int'])))
            ends = np.cumsum(arrays['ms3_n']).tolist()
            spectra['ms3'] = [ms3[start:end] for start, end in zip([0] + ends[:-1], ends)]
        loaded_file = pd.DataFrame({c: spectra[c] if c in spectra else arrays[f'col{i}'] if f'col{i}' in arrays else meta['json_columns'][c] for i, c in enumerate(meta['columns'])})
        loaded_file.attrs.update(meta['attrs'])
        if extract_ms1 and 'ms1_rt' in arrays:
            loaded_file.attrs['ms1'] = (arrays['ms1_rt'], arrays['ms1_mz'], arrays['ms1_int'], np.concatenate([[0], np.cumsum(arrays['ms1_n'], dtype = np.int64)]))
        return loaded_file
    if ext == ".pkl":
        loaded_file = pd.read_pickle(spectra_filepath)
        return loaded_file
    if ext == ".xlsx":
        loaded_file = pd.read_excel(spectra_filepath)
        if 'peak_d' not in loaded_file.columns:
            raise ValueError(f"{os.path.basename(spectra_filepath)} has no peak_d column; CandyCrunch reads .xlsx files of spectra made by extract_spectra, not "
                             f"result tables or other spreadsheets")

        def parse_peak_dict(value):

            def convert_dict(d):
                converted = {}
                for k, v in d.items():
                    try:
                        key = float(k)
                        val = float(v)
                    except (TypeError, ValueError):
                        continue
                    converted[key] = val
                return converted if converted else None

            if isinstance(value, dict):
                return convert_dict(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                if 'np.float64' in text:
                    text = re.sub(r'np\.float64\(([^)]+)\)', r'\1', text)
                # A plain {m/z: intensity, ...} (as extract_spectra writes it) is split and read with float, which reads numbers as literal_eval does but
                # ~4x faster; anything else, or anything float rejects, goes through literal_eval
                if text[0] == '{' and text[-1] == '}' and set('0123456789.eE+-:, {}').issuperset(text):
                    try:
                        return dict(map(float, kv.split(':')) for kv in text[1:-1].split(','))
                    except ValueError:
                        pass
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    return None
                if isinstance(parsed, dict):
                    return convert_dict(parsed)
            return None

        loaded_file['peak_d'] = loaded_file['peak_d'].apply(parse_peak_dict)
        loaded_file = loaded_file[loaded_file['peak_d'].notnull()].reset_index(drop = True)
        # MS3 spectra written by extract_spectra, as a list of (MS3 precursor m/z, peak dictionary) per MS2 spectrum
        if 'ms3' in loaded_file.columns:
            loaded_file['ms3'] = [ast.literal_eval(x) if isinstance(x, str) else [] for x in loaded_file['ms3']]
        # Files written by extract_spectra carry the ion mode and the MS2 and survey scan analyzers detected from their raw file
        for k in ('mode', 'trap', 'ms1_trap'):
            if k in loaded_file.columns:
                detected = loaded_file.pop(k).dropna()
                loaded_file.attrs[f'detected_{k}'] = detected.iloc[0] if len(detected) else None
        return loaded_file
    if ext == ".csv":
        storage = DictStorage()
        loaded_file = storage.read(spectra_filepath, 'peak_d')
        return loaded_file
    raise FileNotFoundError(f"{os.path.basename(os.path.normpath(spectra_filepath))} is not a file type CandyCrunch reads (Thermo .raw, Bruker timsTOF .d folder, "
                            f".mzML, .mzXML, .mgf, or .cc/.xlsx made by extract_spectra)")


# Hash of the code that turns spectra files into spectra (the readers in utils.py and the loader functions above), taken at import, before the app or
# the test harness wrap any of them; .cc files carry it, so that one written by another loader version can be told apart (None without readable source)
try:
    SPECTRA_LOADER = hashlib.sha1(''.join([inspect.getsource(inspect.getmodule(read_mzml))] + [inspect.getsource(f) for f in (
        refine_precursor_mz, refine_tof_precursors, process_mzML_stack, process_mzXML_stack, process_raw_stack, process_tdf_stack, process_yep_stack,
        load_spectra_filepath)]).encode()).hexdigest()[:12]
except (OSError, TypeError):
    SPECTRA_LOADER = None
# Minutes around a peak that extract_spectra(region = ...) keeps, for misprediction reports: tails and neighboring isomers of PGC peaks lie within it
REGION_RT = 3.0


def extract_spectra(spectra_filepath, output_filepath = None, file_format = 'xlsx', region = None):
    """extracts the MS/MS spectra of a .raw/.d/.mzML/.mzXML/.mgf file into a lightweight .xlsx or .cc file that wrap_inference reads directly\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".raw", ".d" (Bruker timsTOF or ion trap folder), ".mzML", ".mzXML", or ".mgf"
   | output_filepath (string): .xlsx or .cc filepath to write; default:None (spectra_filepath with the extension of file_format)
   | file_format (string): 'xlsx' (opens in Excel; MS2 and MS3 spectra) or 'cc' (compact binary file that also keeps the survey scans, so predictions
   |                       from it are quantified by XIC areas, as from the original file); an output_filepath ending in .cc writes 'cc'; default:'xlsx'
   | region (tuple): (m/z, signed charge, RT) of a peak, to keep only its surroundings: the survey scans within REGION_RT min of RT and the MS2 (with
   |                 their MS3) spectra there of the same neutral mass at any charge up to 4, within 1.5 Da (isotope picks included); a .cc then also
   |                 records the peak, how many MS2 spectra the whole run has and its last MS2 RT, and names the source only by its file type; default:None\n
   | Returns:
   | :-
   | Returns the filepath of the written file; precursor m/z values are already refined from MS1 and isotope-triggered repeat spectra removed,
   | MS3 spectra are kept (column ms3), and MS1 only in a .cc, so predictions from an .xlsx are quantified by precursor intensity instead of XIC areas
   """
    cc = (output_filepath or '.' + file_format).lower().endswith('.cc')
    df = load_spectra_filepath(spectra_filepath, extract_ms1 = cc)
    if df.empty:
        # An empty .xlsx would only fail later as a run without glycan peaks
        raise ValueError(f"{os.path.basename(os.path.normpath(spectra_filepath))} has no MS2 spectra with a precursor m/z, so there is nothing to extract (some conversions "
                         f"keep only the MS1 scans; data-independent acquisition such as Waters MSE has no precursors)")
    region_meta = None
    if region is not None:
        # The surroundings of one peak, as a misprediction report sends them: its ion at every charge state, in the survey scans and in the MS2/MS3 spectra
        mz, charge, rt = region
        z, sign = abs(int(charge)) or 1, -1 if charge < 0 else 1
        neutral, attrs = z * mz - sign * z * PROTON_MASS, df.attrs
        region_meta = {'mz': float(mz), 'charge': int(charge), 'rt': float(rt), 'rt_window': REGION_RT, 'run_ms2': len(df), 'run_rt_max': float(df['RT'].max())}
        keep = (np.abs(df['RT'].to_numpy(float) - rt) <= REGION_RT) & np.any(
            [np.abs(df['m/z'].to_numpy(float) - (neutral + sign * c * PROTON_MASS) / c) <= 1.5 / c for c in range(1, 5)], axis = 0)
        df = df[keep].reset_index(drop = True)
        df.attrs = attrs
        if attrs.get('ms1') is not None:
            rts, mzs, ints, offsets = attrs['ms1']
            lo, hi = np.searchsorted(rts, rt - REGION_RT, side = 'left'), np.searchsorted(rts, rt + REGION_RT, side = 'right')
            df.attrs['ms1'] = (rts[lo:hi], mzs[offsets[lo]:offsets[hi]], ints[offsets[lo]:offsets[hi]], offsets[lo:hi + 1] - offsets[lo])
    if cc:
        output_filepath = output_filepath or os.path.splitext(os.path.normpath(spectra_filepath))[0] + '.cc'
        # A .cc is a zip of numpy arrays (np.load reads it), each stored as its byte planes (all first bytes, then all second bytes, ...), which deflate packs
        # far better than interleaved numbers. Everything is kept exactly as loaded (MS2/MS3 peaks as float32 where that is exact), as rounding MS2 m/z to
        # 0.1 mDa or intensities to 0.02% already changed rows and F1 by up to 0.01; only survey scan intensities are rounded to 12 significant bits (0.02%),
        # which left every prediction identical and moved abundances by < 0.001%. Columns that are not numbers (scan IDs that are text, activation) and the
        # detected ion mode and analyzers travel in a JSON header
        exact32 = lambda x: x.astype(np.float32) if np.array_equal(x.astype(np.float32), x) else x
        columns = list(df.columns)
        meta = {'format': 'CandyCrunch spectra', 'version': 1, 'source': os.path.basename(os.path.normpath(spectra_filepath)), 'loader': SPECTRA_LOADER, 'columns': columns,
                'attrs': {k: v for k, v in df.attrs.items() if k != 'ms1'}, 'json_columns': {c: df[c].tolist() for c in columns if c not in ('peak_d', 'ms3') and
                                                                                             not pd.api.types.is_numeric_dtype(df[c])}}
        if region_meta:
            meta.update(region = region_meta, source = 'run' + os.path.splitext(os.path.normpath(spectra_filepath))[1].lower())
        arrays = {f'col{i}': df[c].to_numpy() for i, c in enumerate(columns) if c not in ('peak_d', 'ms3') and c not in meta['json_columns']}
        spectra = [('', df['peak_d'])] + ([('ms3_', [d for x in df['ms3'] for _, d in x])] if 'ms3' in columns else [])
        for prefix, dicts in spectra:
            peaks = np.concatenate([np.zeros((0, 2))] + [np.array(list(d.items()), dtype = float).reshape(-1, 2) for d in dicts])
            arrays.update({f'{prefix}peak_n': np.array([len(d) for d in dicts], dtype = np.uint32), f'{prefix}peak_mz': exact32(peaks[:, 0]), f'{prefix}peak_int': exact32(peaks[:, 1])})
        if 'ms3' in columns:
            arrays.update(ms3_n = np.array([len(x) for x in df['ms3']], dtype = np.uint32), ms3_precursor = np.array([p for x in df['ms3'] for p, _ in x], dtype = float))
        if df.attrs.get('ms1') is not None and len(df.attrs['ms1'][0]):
            rts, mzs, ints, offsets = df.attrs['ms1']
            arrays.update(ms1_rt = np.asarray(rts, dtype = float), ms1_mz = mzs, ms1_int = ((np.asarray(ints, dtype = np.float32).view(np.uint32) + np.uint32(0x400)) & np.uint32(0xFFFFF800)).view(np.float32),
                          ms1_n = np.diff(offsets).astype(np.uint32))
        meta['dtypes'] = {k: v.dtype.str for k, v in arrays.items()}
        with open(output_filepath, 'wb') as f:
            np.savez_compressed(f, meta = np.frombuffer(json.dumps(meta, default = lambda x: x.item()).encode(), dtype = np.uint8),
                                **{k: np.ascontiguousarray(np.ascontiguousarray(v).view(np.uint8).reshape(-1, v.dtype.itemsize).T) for k, v in arrays.items()})
        return output_filepath
    # Rounding keeps the file small and every peak dictionary below Excel's 32,767-character cell limit; 4 decimals and 4 significant
    # digits sit far below the binning and fragment-annotation tolerances
    df['peak_d'] = [str({round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for d in df['peak_d']]
    if 'ms3' in df.columns:
        df['ms3'] = [str([(round(p, 4), {round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for p, d in x]) for x in df['ms3']]
    # Ion mode and analyzers detected from the raw file would not survive the export, so they travel as columns that load_spectra_filepath
    # turns back into the attrs wrap_inference checks
    for k in ('mode', 'trap', 'ms1_trap'):
        df[k] = df.attrs.get(f'detected_{k}')
    output_filepath = output_filepath or os.path.splitext(os.path.normpath(spectra_filepath))[0] + '.xlsx'
    df.to_excel(output_filepath, index = False)
    return output_filepath


def combine_charge_states(df_out):
    """looks for several charges at the same RT with the same top prediction composition and combines their relative abundances\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe where one row per co-eluting charge-state group (a reported structure, then the most abundant) carries the sum of abundances
    """
    # Charge states are grouped by the composition of their top prediction, as the isomer chosen from a z = 1 and a z = 2 spectrum of one glycan can differ
    df_out['top_pred'] = [stringify_dict(get_comp(k[0][0])) if len(k) > 0 else np.nan for k in df_out.predictions]
    repeated_top_pred = df_out['top_pred'].value_counts()
    repeated_top_pred = repeated_top_pred[repeated_top_pred > 1].index.tolist()
    filtered_top_pred = []
    for pred in repeated_top_pred:
        charge_values = df_out[df_out['top_pred'] == pred]['charge']
        if abs(charge_values.max() - charge_values.min()) >= 1:
            filtered_top_pred.append(pred)
    df_filtered = df_out[df_out['top_pred'].isin(filtered_top_pred)].copy()
    for pred in filtered_top_pred:
        pred_rows = df_filtered[df_filtered['top_pred'] == pred]
        lowest = pred_rows['charge'].abs().min()
        # Every lowest-charge row and the higher charge states of this glycan that co-elute with it are one species, reported by its best-supported
        # ion (a reported structure, then the most abundant), as the lowest charge alone let a weak row (e.g. imputed with this structure, without a
        # top1_pred) stand for a reported one, and a z = 1 row cannot match a peak curated at its z = 2 m/z while a z = 2 row matches both
        absorbed = set()
        for idx, row in pred_rows[pred_rows['charge'].abs() == lowest].iterrows():
            partners = pred_rows[(pred_rows['charge'].abs() > lowest) & ((pred_rows['RT'] - row['RT']).abs() < 1) & ~pred_rows.index.isin(absorbed)]
            absorbed.update(partners.index)
            group = pd.concat([pred_rows.loc[[idx]], partners])
            rep = group.assign(_weak = group['top1_pred'].isna()).sort_values(['_weak', 'rel_abundance'], ascending = [True, False], kind = 'stable').index[0]
            df_filtered.at[rep, 'rel_abundance'] = group['rel_abundance'].sum()
            df_filtered.at[rep, 'num_spectra'] = group['num_spectra'].sum()
            if 'xic' in df_filtered.columns:
                df_filtered.at[rep, 'xic'] = sum(group['xic'], [])
            df_filtered = df_filtered.drop(group.index.drop(rep))
    df_out = pd.concat([df_out[~df_out['top_pred'].isin(filtered_top_pred)], df_filtered]).sort_index()
    df_out.drop(['top_pred'], axis = 1, inplace = True)
    return df_out


def combine_adduct_species(df_out, rt_diff = 1.0, mode = 'negative'):
    """combines abundances when the same glycan appears as different adduct forms (e.g., [M-H]⁻ and [M+Acetate]⁻)\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | rt_diff (float): retention time tolerance (in minutes) for considering two rows the same species; default:1.0
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default:'negative'\n
    | Returns:
    | :-
    | Returns prediction dataframe with adduct abundances consolidated into the primary ion form
    """
    # In negative mode ion forms are grouped by the composition of their top prediction, as charge states are in combine_charge_states: the isomer chosen
    # from an anion adduct's spectrum can differ from the one of its (de)protonated ion, which then left the adduct as a species of its own (positive-mode
    # metal adducts are folded by composition in finalise_predictions, after the permethylated NH4+ check, which compares the rows' own abundances)
    df_out['top_pred'] = [(stringify_dict(get_comp(k[0][0])) if mode == 'negative' else k[0][0]) if len(k) > 0 else np.nan for k in df_out.predictions]
    repeated = df_out['top_pred'].value_counts()
    repeated = repeated[repeated > 1].index.tolist()
    # Only act when multiple rows share a prediction but differ in m/z (i.e., different ionization forms)
    rows_to_drop = []
    # The (de)protonated ion reports the species when it was observed, as it is the m/z every other tool lists, else the stronger signal
    plain = dict(zip(df_out.index, ~df_out['adduct'].apply(lambda a: isinstance(a, str)))) if 'adduct' in df_out.columns else defaultdict(lambda: True)
    for pred in repeated:
        pred_df = df_out[df_out['top_pred'] == pred].sort_values('RT')
        mz_vals = pred_df.index.values
        if mz_vals.max() - mz_vals.min() < 1.0:
            continue
        used = set()
        for i, (idx_i, row_i) in enumerate(pred_df.iterrows()):
            if idx_i in used:
                continue
            for idx_j, row_j in pred_df.iloc[i + 1:].iterrows():
                if idx_j in used:
                    continue
                if abs(row_i['RT'] - row_j['RT']) > rt_diff:
                    continue
                if abs(idx_i - idx_j) < 1.0:
                    continue
                # Fold the weaker signal into the stronger one, with current values, as either row may already have absorbed another
                if (plain[idx_i], df_out.at[idx_i, 'rel_abundance']) >= (plain[idx_j], df_out.at[idx_j, 'rel_abundance']):
                    df_out.at[idx_i, 'rel_abundance'] = df_out.at[idx_i, 'rel_abundance'] + df_out.at[idx_j, 'rel_abundance']
                    df_out.at[idx_i, 'num_spectra'] = df_out.at[idx_i, 'num_spectra'] + df_out.at[idx_j, 'num_spectra']
                    if 'xic' in df_out.columns:
                        df_out.at[idx_i, 'xic'] = df_out.at[idx_i, 'xic'] + df_out.at[idx_j, 'xic']
                    rows_to_drop.append(idx_j)
                    used.add(idx_j)
                else:
                    df_out.at[idx_j, 'rel_abundance'] = df_out.at[idx_j, 'rel_abundance'] + df_out.at[idx_i, 'rel_abundance']
                    df_out.at[idx_j, 'num_spectra'] = df_out.at[idx_j, 'num_spectra'] + df_out.at[idx_i, 'num_spectra']
                    if 'xic' in df_out.columns:
                        df_out.at[idx_j, 'xic'] = df_out.at[idx_j, 'xic'] + df_out.at[idx_i, 'xic']
                    rows_to_drop.append(idx_i)
                    used.add(idx_i)
                    break
    df_out = df_out.drop(rows_to_drop)
    df_out.drop(['top_pred'], axis = 1, inplace = True)
    return df_out


def Ac_follows_Gc(df_out):
    """function to inform Neu5Ac isomer deduplication based on Neu5Gc isomers\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe in which, if possible, Neu5Ac isomers have been deduplicated
    """
    # Add a helper column for easier manipulation, if not already present
    if "Original_Prediction" not in df_out.columns:
        df_out["Original_Prediction"] = df_out["predictions"].apply(lambda x: x[0][0] if x else None)
    # Iterate through unique Gc predictions
    for gc_pred in df_out[df_out["Original_Prediction"].str.contains("Neu5Gc", na = False)][
        "Original_Prediction"].unique():
        ac_pred = gc_pred.replace("Neu5Gc", "Neu5Ac")
        # Filter rows for current Ac and Gc pair
        gc_rows = df_out[df_out["Original_Prediction"] == gc_pred]
        ac_rows = df_out[df_out["Original_Prediction"] == ac_pred]
        # Continue if either Ac or Gc rows are empty
        if gc_rows.empty or ac_rows.empty:
            continue
        # For simplicity, use the first Gc row's RT as reference
        gc_rt = gc_rows.iloc[0]["RT"]
        # Determine canonical Ac rows: Ac rows within ±1.0 RT of Gc
        ac_rows = ac_rows.assign(RT_diff = np.abs(ac_rows["RT"] - gc_rt))
        canonical_ac_indices = ac_rows[(ac_rows["RT_diff"] <= 1.0)].index
        # Update non-canonical Ac rows
        for idx, row in ac_rows.iterrows():
            if idx not in canonical_ac_indices and len(row["predictions"]) > 1:
                df_out.at[idx, "predictions"] = row["predictions"][1:]
    # Remove the helper column
    df_out.drop(columns = ["Original_Prediction"], inplace = True, errors = "ignore")
    return df_out


def filter_delayed_rts(df_out, mass_tolerance):
    """drops rows more than 10 min after the first row with the same top1 prediction and m/z, unless they have the group's highest confidence x precursor intensity (x spectra without intensities), or at least 3 spectra and a tenth of it\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | mass_tolerance (float): m/z window (in Da) within which rows with the same top1 prediction count as one group\n
    | Returns:
    | :-
    | Returns prediction dataframe in which, if possible, fake isomers have been deduplicated
    """
    # Create an empty list to store indices of rows to discard
    rows_to_discard = []
    # Group by m/z within mass_tolerance and identical top1 predictions
    # A flat background ion that triggers MS2 for minutes becomes one cluster of hundreds of spectra, so spectra would make it outweigh the real peak
    weight = df_out['rel_abundance'].to_numpy() if 'rel_abundance' in df_out.columns and (df_out['rel_abundance'] > 0).any() else df_out['num_spectra'].to_numpy()
    cxn = np.array([p[0][1] if p else 0.0 for p in df_out['predictions']]) * weight
    # numpy and a dict per window instead of a pandas groupby (rows without a top1 prediction belong to no group, as groupby leaves them out)
    mzs, rts, n_spectra, top1 = df_out.index.values, df_out['RT'].to_numpy(dtype = float), df_out['num_spectra'].to_numpy(), df_out['top1_pred'].tolist()
    for mz_val in df_out.index.unique():
        groups = {}
        for r in np.flatnonzero((mzs >= mz_val - mass_tolerance) & (mzs <= mz_val + mass_tolerance)):
            if not pd.isna(top1[r]):
                groups.setdefault(top1[r], []).append(r)
        for group in groups.values():
            # the first instance (NaN RTs sort last) and the group's highest confidence x intensity (NaN skipped)
            first_rt = np.nanmin(rts[group]) if not np.isnan(rts[group]).all() else np.nan
            max_cxn = np.nanmax(cxn[group]) if not np.isnan(cxn[group]).all() else np.nan
            # Delayed rows that lose to the group maximum, unless at least 3 spectra and a tenth of it back them: long gradients elute isomers the model names
            # alike more than 10 min apart (GPST000054 893.33 at 27.1 and 34.15, GPST000461 free 859.46, GPST000403 929.48 at 49.68)
            rows_to_discard.extend(mzs[np.array(group)[(rts[group] > first_rt + 10) & (cxn[group] < max_cxn) & ~(
                (cxn[group] >= 0.1 * max_cxn) & (n_spectra[group] >= 3))]])
    # Discard the marked rows
    df_out_filtered = df_out.drop(rows_to_discard)
    return df_out_filtered


def filter_rts(loaded_file, rt_min, rt_max):
    if rt_min:
        loaded_file = loaded_file[loaded_file['RT'] >= rt_min].reset_index(drop = True)
    elif loaded_file['RT'].max() > 20:
        loaded_file = loaded_file[loaded_file['RT'] >= 2].reset_index(drop = True)
    if rt_max:
        loaded_file = loaded_file[loaded_file['RT'] <= rt_max].reset_index(drop = True)
    elif loaded_file['RT'].max() > 40:
        loaded_file = loaded_file[loaded_file['RT'] < 0.9 * loaded_file['RT'].max()].reset_index(drop = True)
    return loaded_file


def augment_predictions(df_out, pred_thresh, supplement, experimental, glycan_class, df_use, mode, modification,
                        mass_tag, filter_out, mass_tolerance, mass_dic, sample_prep = 'underivatized',
                        max_charge = -3, glycans = glycans, sialic_derivatization = None, trust_header_charge = False, accurate_ms2 = False):
    """adds and reorders predictions based on possible structures\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of filtered predictions
    | pred_thresh (float): prediction confidence threshold used for filtering
    | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks
    | experimental (bool): whether to impute missing predictions via database searches etc.
    | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together
    | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
    | mass_tolerance (float): the general mass tolerance (in Da) that is used for composition matching
    | mass_dic (dict): dictionary of form mass : list of glycans; generated here if None
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
    | glycans (list): CandyCrunch vocabulary whose structures of glycan_class join df_use in a mass_dic built here; default: the shipped vocabulary
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS); default:None
    | trust_header_charge (bool): whether instrument charge states (precursor_charge column) are beyond doubt, as passed to domain_filter; default:False
    | accurate_ms2 (bool): whether the MS2 spectra are orbitrap spectra, whose rows supported by neither the model nor the fragment-backed database structure
    |                      are no glycan peaks; default:False\n
    | Returns:
    | :-
    | Returns a dataframe of predictions
    """
    # Rows of another reducing end (re_offset, from wrap_inference) passed domain_filter at their own mass there, which the precursor-based checks here would repeat at the wrong one
    variant = df_out['re_offset'].values != 0 if 're_offset' in df_out.columns else np.zeros(len(df_out), dtype = bool)
    df_variant, df_out = df_out[variant], df_out[~variant]
    # Construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
    if supplement:
        try:
            df_out = supplement_prediction(df_out, glycan_class, mode = mode, modification = modification,
                                           mass_tag = mass_tag, sample_prep = sample_prep)
            df_out['evidence'] = [
                'medium' if pd.isna(evidence) and preds else evidence
                for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
            ]
        except Exception:
            pass
    # Check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
    if experimental:
        df_out = impute(df_out, pred_thresh, mode = mode, modification = modification, mass_tag = mass_tag,
                        glycan_class = glycan_class, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization)
        try:
            df_out = filter_delayed_rts(df_out, mass_tolerance)
            df_out = Ac_follows_Gc(df_out)
        except ValueError:
            pass
        ionization = -PROTON_MASS if mode == 'negative' else PROTON_MASS
        # Reducing-end modification as glycowork adds it, including the group a derivatized alditol gains on its extra free OH
        mass_offset = composition_to_mass({}, sample_prep = sample_prep, modification = modification) - composition_to_mass(
            {}, sample_prep = sample_prep) + (mass_tag or 0) + ionization
        mass_dic = mass_dic if mass_dic else make_mass_dic(glycans, glycan_class, filter_out, df_use, sample_prep = sample_prep,
                                                           sialic_derivatization = sialic_derivatization)
        df_out = possibles(df_out, mass_dic, mass_offset)
        df_out['evidence'] = [
            'weak' if pd.isna(evidence) and preds else evidence
            for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
        ]
        if accurate_ms2:
            # Orbitrap MS2 annotate a real glycan well enough for the model or the database fallback, so a row filled in only here is a tail or an artifact
            # of another ion
            df_out['predictions'] = [[] if e == 'weak' else preds for e, preds in zip(df_out['evidence'], df_out['predictions'])]
    # Filter out wrong predictions via diagnostic ions etc.
    if supplement or experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, modification = modification,
                               mass_tolerance = mass_tolerance, df_use = df_use, sample_prep = sample_prep,
                               mass_tag = mass_tag, max_charge = max_charge, sialic_derivatization = sialic_derivatization,
                               trust_header_charge = trust_header_charge)
        df_out['predictions'] = [[(k[0].replace('-ol', '').replace('1Cer', ''), k[1]) if len(k) > 1 else (
            k[0].replace('-ol', '').replace('1Cer', ''),) for k in j] if j else j for j in df_out['predictions']]
    return pd.concat([df_out, df_variant]) if len(df_variant) else df_out


def finalise_predictions(df_out, get_missing, mode, modification, mass_tag, ppm_thresh, rt_diff,
                         sample_prep = 'underivatized', glycan_class = 'O', mass_tolerance = 0.5, ms1 = None, accurate_ms1 = False,
                         sialic_derivatization = None, n_jobs = 1, ms1_ppm_floor = 3.0, ion_trap = False, xic_details = False):
    """Cleans up incorrect structure predictions, quantifies, and formats dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of augmented predictions
    | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | ppm_thresh (float): ppm error threshold; the file's own robust ppm error spread can only make it stricter
    | rt_diff (float): maximum retention time difference (in minutes) for combining adduct species of one glycan
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together
    | mass_tolerance (float): flat-Da mass tolerance used for composition matching and XIC extraction; default:0.5
    | ms1 (tuple): flat MS1 data from process_mzML_stack; if given, abundances are XIC areas; default:None
    | accurate_ms1 (bool): whether precursor m/z come from accurate (orbitrap or timsTOF) survey scans; if not, mass errors are judged against the file's own m/z calibration; default:False
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization (glycowork's SIALIC_DERIVATIZATIONS); compositions then give each
    |                                          sialic acid's linkage, as the precursor mass tells it; default:None
    | n_jobs (int): number of processes for the supporting peaks of the rows (see run_jobs); default:1
    | ms1_ppm_floor (float): smallest half-width (ppm) of the composition-error window with accurate survey scans (MS1_PPM_FLOORS); default:3.0
    | ion_trap (bool): whether the MS2 spectra are ion-trap CID, the only data supporting_ions' linkage reading of a lone Neu5Ac was measured on, which
    |                  then goes into supported_by; default:False
    | xic_details (bool): whether to add the column xic: per row the MS1 integrations its abundance sums (extract_xic_areas' details of its own ion and
    |                     of the rows folded into it; empty without MS1); default:False\n
    | Returns:
    | :-
    | Returns a tuple of (dataframe of corrected predictions with unnormalized rel_abundance, list of spectra)
    """
    df_out = df_out.drop(columns = ['precursor_charge'], errors = 'ignore')
    # Reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
    try:
        df_out = canonicalize_biosynthesis(df_out)
    except ValueError:
        pass
    # Keep or remove spectra that still lack a prediction after all this (canonicalize_biosynthesis drops confidence-less database guesses)
    if not get_missing:
        df_out = df_out[df_out['predictions'].str.len() > 0]
    # Calculate ppm error
    valid_indices, ppm_errors, theo_mzs = [], [], []
    for preds, obs_mass in zip(df_out['predictions'], df_out.index):
        theo_mass = mass_check(obs_mass, preds[0][0], modification = modification, mass_tag = mass_tag, mode = mode,
                               sample_prep = sample_prep, mass_tolerance = mass_tolerance, sialic_derivatization = sialic_derivatization) if preds else None
        valid_indices.append(bool(theo_mass) or not preds)
        if theo_mass:
            ppm_errors.append(abs(((theo_mass[0] - obs_mass) / theo_mass[0]) * 1e6))
            theo_mzs.append(theo_mass[0])
        elif not preds:
            ppm_errors.append(np.nan)
            theo_mzs.append(obs_mass)
    df_out = df_out[np.array(valid_indices, dtype = bool)]
    df_out['ppm_error'] = ppm_errors
    df_out['theo_mz'] = theo_mzs
    # Drop ppm outliers: robust 3-sigma above the file's median ppm error, never looser than ppm_thresh
    # Unreduced (unlabeled) rows (wrap_inference's reducing_end_variants, folded into their glycan below) are cut like every other row, but the file's error
    # and RT statistics come from the rows it reports
    if 're_offset' in df_out.columns:
        # ... and so are glycosylamines of PA glycans, a labeling side product as well
        df_out['unlabelled'] = ((np.abs(df_out['re_offset'].values + modification_mass_dict.get(modification, 0) + (mass_tag or 0)) < 0.01) | (
            (df_out['reducing_end'].values == 'glycosylamine') if 'reducing_end' in df_out.columns else False)) & (df_out['re_offset'].values != 0)
    regular = ~df_out['unlabelled'].values if 're_offset' in df_out.columns else np.ones(len(df_out), dtype = bool)
    known = regular & df_out['ppm_error'].notna().values
    known_ppm = df_out['ppm_error'].values[known]
    signed = (df_out.index.values - df_out['theo_mz'].values) / df_out['theo_mz'].values * 1e6
    # With accurate survey scans the errors center on the run's calibration, which the composition matches of junk spectra hide from a median and MAD
    window = accurate_ms1_window(signed[known], df_out['annotation_score'].values[known], ppm_thresh, floor = ms1_ppm_floor) if accurate_ms1 else None
    if window is not None:
        keep = (df_out['ppm_error'].isna() | (np.abs(signed - window[0]) <= window[1])).values
    else:
        if len(known_ppm) >= 5:
            med = np.median(known_ppm)
            # ... but not below 100 ppm without accurate survey scans, which no few rows can show to read better than that
            ppm_thresh = min(ppm_thresh, max(med + 3.0 * np.median(np.abs(known_ppm - med)) * 1.4826, 0 if accurate_ms1 else 100))
        keep = (~(df_out['ppm_error'] >= ppm_thresh)).values
    errors, singly = df_out.index.values - df_out['theo_mz'].values, df_out['ppm_error'].notna().values & (
                np.abs(df_out['charge'].values) == 1) & regular
    if not accurate_ms1 and singly.sum() >= 5:
        # Ion-trap survey scans read every precursor off by the file's m/z calibration (-0.15 to +0.15 Da in our test files), which the ppm cut takes for a
        # wrong composition at low m/z (-0.12 Da is -320 ppm at 384): singly charged rows count their error from the file's median z = 1 error instead, in Da,
        # within 4 robust SD (as ms1_mz_calibration). Multiply charged rows keep the ppm cut, as the ion trap merges their isotopes into one centroid that sits a
        # varying fraction of a spacing too high
        offset = np.median(errors[singly])
        width = max(4 * 1.4826 * np.median(np.abs(errors[singly] - offset)), 0.2)
        keep = np.where(np.abs(df_out['charge'].values) == 1,
                        df_out['ppm_error'].isna().values | (np.abs(errors - offset) <= width), keep)
        # ... unless fragments back a multiply charged row that the model or the database fallback named (annotation score of at least 10): its merged centroid
        # then counts anywhere between the monoisotopic and the M+2 position above the file's offset, as large glycans, whose 13C peaks outgrow the
        # monoisotopic one, read half a spacing high or more (GPST000556 [M-2H]2- milk oligosaccharides 0.23-0.38 m/z high, most of them database structures)
        z = np.abs(df_out['charge'].values)
        backed = np.isin(df_out['evidence'].values, ['strong', 'composition']) & (df_out['annotation_score'].values >= 10)
        keep |= backed & (z > 1) & (np.abs(errors - offset - ISOTOPE_SPACING / z) <= ISOTOPE_SPACING / z)
    df_out = df_out[keep]
    # Retention-time outlier removal: drop predictions whose RT lies far outside the file's overall
    # elution distribution, scaled to the observed spread (tight cluster => strict, wide spread => permissive)
    has_pred = df_out['predictions'].apply(len) > 0
    # Negative-mode adduct rows (folded into their (de)protonated ion below, or dropped) do not count either: anion adducts of one glycan family narrowed
    # the window and cut early glycans (GPST000556 546.16 at 15.44)
    side = (df_out['unlabelled'].values.astype(bool) if 're_offset' in df_out.columns else np.zeros(len(df_out), dtype = bool)) | (
        df_out['adduct'].apply(lambda a: isinstance(a, str)).values.astype(bool) if mode == 'negative' and 'adduct' in df_out.columns else False)
    rts_pred = df_out.loc[has_pred.values & ~side, 'RT'].values
    if len(rts_pred) >= 5:
        rt_med = np.median(rts_pred)
        rt_mad = np.median(np.abs(rts_pred - rt_med))
        if rt_mad > 0:
            rt_cut = 3.0 * rt_mad * 1.4826  # ~3-sigma on robust spread
            # ... except for a major peak of the run (at least a tenth of the most abundant one), which is no stray elution
            top = df_out.loc[has_pred, 'rel_abundance'].max() if 'rel_abundance' in df_out.columns else 0
            major = (df_out['rel_abundance'] >= 0.1 * top).values if top > 0 else False
            df_out = df_out[(~has_pred) | (np.abs(df_out['RT'] - rt_med) <= rt_cut) | major]
    # Clean-up
    if sialic_derivatization is None:
        df_out['composition'] = [get_comp(k[0][0]) if k else val for k, val in
                                 zip(df_out['predictions'], df_out['composition'])]
    else:
        # the sialic linkages of the composition are those whose derivatized mass is the precursor's, of the ones the structure allows
        df_out['composition'] = [val if not k else next((c for c in sialic_linkage_splits(get_comp(k[0][0])) if resolve_sialic_linkages(k[0][0], c) is not None and mass_check(
            obs, composition_to_mass(c, sialic_derivatization = sialic_derivatization), modification = modification, mass_tag = mass_tag, mode = mode,
            sample_prep = sample_prep, mass_tolerance = mass_tolerance)), val) for k, val, obs in zip(df_out['predictions'], df_out['composition'], df_out.index)]
    df_out['charge'] = round(df_out['composition'].apply(lambda x: composition_to_mass(x, sample_prep = sample_prep, modification = modification,
                                                                                       sialic_derivatization = sialic_derivatization)) /
                             df_out.index) * (-1 if mode == 'negative' else 1)
    df_out = df_out.astype({'num_spectra': 'int', 'charge': 'int'})
    # Rows of another reducing end (wrap_inference's reducing_end_variants) are carried at the m/z of the run's reducing end; their ions sit re_offset / z away
    if 're_offset' in df_out.columns:
        df_out['theo_mz'] += df_out['re_offset'].values / np.abs(df_out['charge'].values)
    # Quantify via MS1 where available, else keep precursor intensities: each row's isotope envelope at its theoretical m/z, over its own chromatographic peak, before charge states and adducts of a glycan are summed
    # (xic holds the integrations a row's abundance sums, which every fold below hands on with the abundance)
    df_out['xic'] = [[] for _ in range(len(df_out))]
    if ms1 is not None and len(ms1[0]) > 0 and len(df_out) > 0:
        # Charge states of a row's composition that the neutral mass allows (as mass_check), up to the run's highest, except those another row of the
        # composition reports within rt_diff (combine_charge_states adds those) and for adduct ions
        comps = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
        zs, rts = np.abs(df_out['charge'].values), df_out['RT'].values
        adducts = df_out['adduct'].apply(lambda a: isinstance(a, str)).values if 'adduct' in df_out.columns else np.zeros(len(df_out), dtype = bool)
        neutral = (df_out['theo_mz'].values - (-1 if mode == 'negative' else 1) * PROTON_MASS) * zs
        other_charges = [[] if comps[i] is None or adducts[i] else [z for z in range(1, zs.max() + 1) if z != zs[i] and neutral[i] >= {1: 0, 2: 900, 3: 1500}.get(z, 3500)
                          and not any(comps[j] == comps[i] and zs[j] == z and abs(rts[j] - rts[i]) < rt_diff for j in range(len(df_out)))] for i in range(len(df_out))]
        calibration = ms1_mz_calibration(ms1, df_out['theo_mz'].values, df_out['RT'].values, df_out['charge'].values,
                                         mz_tolerance = mass_tolerance, accurate = accurate_ms1)
        details = [] if xic_details else None
        xic_areas = extract_xic_areas(ms1, df_out['theo_mz'].values, df_out['RT'].values, charges = df_out['charge'].values,
                                      mz_tolerance = mass_tolerance, weights = df_out['rel_abundance'].values,
                                      mz_calibration = calibration, sample_prep = sample_prep, mode = mode, other_charges = other_charges,
                                      glycans = [k[0][0] if k else None for k in df_out['predictions']], details = details)
        if (xic_areas > 0).any():
            df_out['rel_abundance'] = xic_areas
            df_out.attrs['ms1_calibration'] = calibration
            if details is not None:
                df_out['xic'] = [[d] for d in details]
    df_out = df_out.drop(columns = ['theo_mz'])
    if mode == 'positive' and 'adduct' in df_out.columns and 'rt_range' in df_out.columns:
        # In positive mode a glycan's metal adduct ([M+K+H]2+, [M+Na+H]2+) can dominate its (de)protonated ion, whose weak, noisy precursor intensities then
        # split one chromatographic peak into several clusters; (de)protonated rows of a composition inside the RT span of one of its adduct clusters are
        # one peak, reported by the row closest to that cluster's apex, if the adduct dominates them
        # (as bool: an empty run's object mask selected columns instead of rows, and the result lost every column)
        plain = ~df_out['adduct'].apply(lambda a: isinstance(a, str)).values.astype(bool)
        comps = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
        rts, ranges = df_out['RT'].values, df_out['rt_range'].values
        ab, ns, xc = df_out.columns.get_loc('rel_abundance'), df_out.columns.get_loc('num_spectra'), df_out.columns.get_loc('xic')
        merged = set()
        for i in sorted(np.flatnonzero(~plain), key = lambda i: -df_out.iat[i, ab]):
            if comps[i] is None or not isinstance(ranges[i], tuple):
                continue
            inside = [j for j in np.flatnonzero(plain) if j not in merged and comps[j] == comps[i] and ranges[i][0] <= rts[j] <= ranges[i][1]]
            if len(inside) > 1 and df_out.iat[i, ab] >= sum(df_out.iat[j, ab] for j in inside):
                keep = min(inside, key = lambda j: abs(rts[j] - rts[i]))
                for j in inside:
                    if j != keep:
                        df_out.iat[keep, ab] += df_out.iat[j, ab]
                        df_out.iat[keep, ns] += df_out.iat[j, ns]
                        df_out.iat[keep, xc] = df_out.iat[keep, xc] + df_out.iat[j, xc]
                        merged.add(j)
        df_out = df_out.iloc[[k for k in range(len(df_out)) if k not in merged]]
    df_out = df_out.drop(columns = ['rt_range'], errors = 'ignore')
    if accurate_ms1 and len(df_out) > 1:
        # The instrument's monoisotopic pick of a large multiply charged glycan is sometimes its 13C peak, which then reads as a composition one or two isotope
        # spacings heavier (HexNAc7Hex8Neu5Ac2 for the M+2 of HexNAc5Hex6dHex1Neu5Ac4, 1.3 mDa apart), while a co-eluting charge state of the glycan shows its
        # true mass: such a row joins the row of that charge state (within 20 ppm of its neutral mass plus the spacings), whose abundance and spectra it adds
        sign = -1 if mode == 'negative' else 1
        zs = np.abs(df_out['charge'].values)
        neutral = (df_out.index.values - sign * PROTON_MASS) * zs
        rts = df_out['RT'].values
        ab, ns, xc = df_out.columns.get_loc('rel_abundance'), df_out.columns.get_loc('num_spectra'), df_out.columns.get_loc('xic')
        folded = set()
        for i in range(len(df_out)):
            partners = [j for j in range(len(df_out)) if j != i and j not in folded and zs[j] != zs[i] and abs(rts[j] - rts[i]) < rt_diff and any(
                abs(neutral[i] - n * ISOTOPE_SPACING - neutral[j]) <= 2e-5 * neutral[i] for n in (1, 2))]
            if partners:
                j = min(partners, key = lambda j: abs(rts[j] - rts[i]))
                df_out.iat[j, ab] += df_out.iat[i, ab]
                df_out.iat[j, ns] += df_out.iat[i, ns]
                df_out.iat[j, xc] = df_out.iat[j, xc] + df_out.iat[i, xc]
                folded.add(i)
        df_out = df_out.iloc[[k for k in range(len(df_out)) if k not in folded]]
    df_out = combine_charge_states(df_out)
    df_out = combine_adduct_species(df_out, rt_diff = rt_diff, mode = mode)
    if mode == 'negative' and 'adduct' in df_out.columns:
        # An anion adduct without a co-eluting (de)protonated ion of its composition is not reported: adduct readings only fill precursors that no
        # (de)protonated composition explains, formate and bicarbonate are exact isobars with Hex and dHex (or Neu5Ac and Neu5Gc) swapped, and [M+2Acetate]2-
        # sits on the bicarbonate reading of a composition one Hex + HexNAc heavier; each such row of the test sets was a false positive (GPST000054 832.84,
        # GPST000302 617.31 and 860.26, GPST000374 N 779.27, GPST000596 941.38)
        df_out = df_out[~df_out['adduct'].apply(lambda a: isinstance(a, str)).values.astype(bool)]
    if mode == 'positive' and sample_prep == 'permethylated' and len(df_out):
        # Permethylated, an NH4+ in place of a proton weighs exactly what Hex + dHex add in place of a Neu5Ac (NH3, 17.0265 Da, either way), so the
        # [M+NH4+2H]3+ of A2G2S2 (935.16, 60% of its [M+3H]3+ in serum) reads as Hex6HexNAc4dHex1Neu5Ac1 [M+3H]3+: a row whose composition is that of a
        # co-eluting, at least as abundant row with k Hex + dHex in place of k Neu5Ac (k up to its charge) is that row's NH4+ adduct, and counts for it
        comps = [c if isinstance(c, dict) else {} for c in df_out['composition']]
        keys = [stringify_dict(c) for c in comps]
        rts, zs = df_out['RT'].values, np.abs(df_out['charge'].values)
        ab, ns, xc = df_out.columns.get_loc('rel_abundance'), df_out.columns.get_loc('num_spectra'), df_out.columns.get_loc('xic')
        folded = set()
        for i in sorted(range(len(df_out)), key = lambda i: df_out.iat[i, ab]):
            c = comps[i]
            for k in range(1, min(c.get('Hex', 0), c.get('dHex', 0), zs[i]) + 1):
                target = stringify_dict({m: n for m, n in {**c, 'Hex': c['Hex'] - k, 'dHex': c['dHex'] - k, 'Neu5Ac': c.get('Neu5Ac', 0) + k}.items() if n})
                partners = [j for j in range(len(df_out)) if j != i and j not in folded and keys[j] == target and abs(rts[j] - rts[i]) <= rt_diff
                            and df_out.iat[j, ab] >= df_out.iat[i, ab]]
                if partners:
                    j = min(partners, key = lambda j: abs(rts[j] - rts[i]))
                    df_out.iat[j, ab] += df_out.iat[i, ab]
                    df_out.iat[j, ns] += df_out.iat[i, ns]
                    df_out.iat[j, xc] = df_out.iat[j, xc] + df_out.iat[i, xc]
                    folded.add(i)
                    break
        df_out = df_out.iloc[[k for k in range(len(df_out)) if k not in folded]]
    if sample_prep == 'permethylated' and len(df_out):
        # Permethylated glycans on C18 tail for minutes and DDA keeps sampling the tail, so a row within 3 min of an at least 5x more abundant row of its
        # composition is that peak's tail and counts for it; its isomers on C18 (sialic acid linkages) elute 6-7 min apart
        comps = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
        rts = df_out['RT'].values
        ab, ns, xc = df_out.columns.get_loc('rel_abundance'), df_out.columns.get_loc('num_spectra'), df_out.columns.get_loc('xic')
        folded = set()
        for i in sorted(range(len(df_out)), key = lambda i: df_out.iat[i, ab]):
            partners = [j for j in range(len(df_out)) if j != i and j not in folded and comps[i] is not None and comps[j] == comps[i] and abs(rts[j] - rts[i]) <= 3
                        and df_out.iat[j, ab] >= 5 * df_out.iat[i, ab]]
            if partners:
                j = max(partners, key = lambda j: df_out.iat[j, ab])
                df_out.iat[j, ab] += df_out.iat[i, ab]
                df_out.iat[j, ns] += df_out.iat[i, ns]
                df_out.iat[j, xc] = df_out.iat[j, xc] + df_out.iat[i, xc]
                folded.add(i)
        df_out = df_out.iloc[[k for k in range(len(df_out)) if k not in folded]]
    if mode == 'positive' and 'adduct' in df_out.columns:
        # In positive mode a metal adduct ion ([M+K+H]2+, [M+Na+H]2+, [M+2Na]2+, ...) is reported by the co-eluting (de)protonated ion of its composition,
        # which takes its abundance and spectra count, whichever isomer each was given; without one it is no glycan peak
        # (as bool: an empty run's object mask selected columns instead of rows, and the result lost every column)
        plain = ~df_out['adduct'].apply(lambda a: isinstance(a, str)).values.astype(bool)
        comps = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
        rts, ab, ns, xc = df_out['RT'].values, df_out.columns.get_loc('rel_abundance'), df_out.columns.get_loc('num_spectra'), df_out.columns.get_loc('xic')
        for i in np.flatnonzero(~plain):
            partners = [j for j in np.flatnonzero(plain) if comps[i] is not None and comps[j] == comps[i] and abs(rts[j] - rts[i]) <= rt_diff]
            if partners:
                j = min(partners, key = lambda j: abs(rts[j] - rts[i]))
                df_out.iat[j, ab] += df_out.iat[i, ab]
                df_out.iat[j, ns] += df_out.iat[i, ns]
                df_out.iat[j, xc] = df_out.iat[j, xc] + df_out.iat[i, xc]
        df_out = df_out[plain]
    if 'adduct' in df_out.columns and sialic_derivatization is None:
        # A species seen only as an adduct is reported at the m/z of the (de)protonated ion of its charge, the m/z it is listed by elsewhere (and that its
        # (de)protonated row would have had); 'adduct' then names the observed ion
        new_index, observed = [], []
        for obs, top1, z, add in zip(df_out.index, df_out['top1_pred'], df_out['charge'], df_out['adduct']):
            ions = {}
            if isinstance(add, str) and isinstance(top1, str):
                try:
                    mass = glycan_to_mass(top1, sample_prep = sample_prep, modification = modification) + (mass_tag or 0)
                    ions = {k: float(v) for k, v in get_ion_mzs(mass, max_charge = int(z), adducts = get_adduct_list(mode)).items() if int(k.rsplit(']', 1)[1][:-1] or 1) == abs(int(z))}
                except Exception:
                    ions = {}
            z_str = str(abs(int(z))) if abs(int(z)) > 1 else ''
            plain = f"[M{'+' if z > 0 else '-'}{z_str}H]{z_str}{'+' if z > 0 else '-'}"
            if ions and plain in ions:
                name = min(ions, key = lambda k: abs(ions[k] - obs))
                new_index.append(obs - ions[name] + ions[plain])
                observed.append(name)
            else:
                new_index.append(obs)
                observed.append(add)
        df_out.index = pd.Index(new_index, name = df_out.index.name)
        df_out['adduct'] = observed
    if 're_offset' in df_out.columns:
        # An unreduced (unlabeled) glycan is a glycan of the run that escaped reduction (labeling), a glycosylamine one whose PA label was not reduced: its
        # abundance and spectra count for that glycan's most abundant row, and it is never a row of its own. Every other reducing end (e.g., ethyl glycosides) is
        # a species of its own, reported at its observed m/z
        unlabelled = df_out['unlabelled'].values
        top1, abundance = df_out['top1_pred'].values, df_out['rel_abundance'].values
        target = {}
        for i in np.flatnonzero(df_out['re_offset'].values == 0):
            if isinstance(top1[i], str) and (top1[i] not in target or abundance[i] > abundance[target[top1[i]]]):
                target[top1[i]] = i
        for i in np.flatnonzero(unlabelled & np.isin(top1, list(target))):
            for col in ('rel_abundance', 'num_spectra'):
                df_out.iloc[target[top1[i]], df_out.columns.get_loc(col)] += df_out[col].iloc[i]
            df_out.iat[target[top1[i]], df_out.columns.get_loc('xic')] = df_out['xic'].iloc[target[top1[i]]] + df_out['xic'].iloc[i]
        df_out.index = pd.Index(df_out.index.values + df_out['re_offset'].values / np.abs(df_out['charge'].values), name = df_out.index.name)
        df_out = df_out[~unlabelled]
    df_out = flag_coeluting_substructures(df_out, glycan_class, sample_prep = sample_prep, unreduced = modification is None and not mass_tag,
                                          ms1 = ms1 if ms1 is not None and len(ms1[0]) > 0 else None, mz_tolerance = mass_tolerance)
    # With accurate survey scans, a row noted as an in-source fragment (a Y-type substructure whose trace is a scaled copy of its parent's, at the parent's
    # apex, at most a third of it) is that fragment, not a glycan of the sample. Only if the scans were read: an xlsx is marked orbitrap without them, and
    # its abundance-only note dropped real glycans (GPST000578 0.938 -> 0.885, GPST000651 0.761 -> 0.753)
    if accurate_ms1 and ms1 is not None and len(ms1[0]) > 0 and 'notes' in df_out.columns:
        df_out = df_out[~df_out['notes'].fillna('').str.contains('Possible in-source fragment', regex = False).values]
    elif mode == 'positive' and 'notes' in df_out.columns and 'top_fragments' in df_out.columns and len(df_out):
        # Without accurate survey scans the note alone dropped real glycans, but in positive mode a noted row that its parent's own MS2 makes as one of its
        # three most intense fragments is that fragment (CID of a sialylated N-glycan loses a whole antenna there, as in the source): GPST000750 Deriv_A1-1_8
        # FA2G2 1865.8 [M+H]+ under 6N-TRF(6) 1281.8 [M+2H]2+, whose MS2 base peak it is. In negative mode it dropped real glycans
        z1 = df_out.index.values * np.abs(df_out['charge'].values) - (np.abs(df_out['charge'].values) - 1) * PROTON_MASS
        keep = np.ones(len(df_out), dtype = bool)
        for a, note in enumerate(df_out['notes'].fillna('').values):
            if 'Possible in-source fragment of ' in note:
                parent = note.split('Possible in-source fragment of ')[1].split(' (co-elutes')[0]
                rows = np.flatnonzero((df_out['top1_pred'].values == parent) & (np.abs(df_out['RT'].values - df_out['RT'].values[a]) <= 0.1))
                keep[a] = not any(abs(f - z1[a]) <= 0.5 or abs(f - df_out.index.values[a]) <= 0.5 for b in rows for f in list(df_out['top_fragments'].iloc[b] or [])[:3])
        df_out = df_out[keep]
    # Map GlyTouCan IDs of the structure reported, which can differ from the first candidate
    df_out["GlyTouCan_ID"] = [glytoucan_mapping.get(g, '') if isinstance(g, str) else '' for g in df_out['top1_pred']]
    if 'reducing_end' in df_out.columns:
        # the ID of the glycan, not of its ethyl glycoside or other reducing-end variant
        df_out.loc[df_out['reducing_end'].notna(), 'GlyTouCan_ID'] = ''
    # MS2 peaks that support the reported structure, per residue: intense single glycosidic cleavages that no other placement of that residue could form
    # (residues without any other placement are not counted), and peaks that only another placement explains
    supported_by = []
    for support in run_jobs([(supporting_ions, (top1, peaks), dict(charge = int(charge), candidates = [p[0] for p in preds if p[0] != top1],
                                                                    mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0) + offset,
                                                                    sample_prep = sample_prep, sialic_derivatization = sialic_derivatization))
                             if isinstance(top1, str) and isinstance(peaks, dict) else (dict, (), dict(residues = []))
                             for top1, preds, peaks, charge, offset in zip(df_out['top1_pred'], df_out['predictions'], df_out['peak_d'], df_out['charge'],
                                                                           df_out['re_offset'] if 're_offset' in df_out.columns else [0] * len(df_out))], n_jobs):
        tested, texts = [e for e in support['residues'] if e['support'] or e['open']], []
        for e in tested:
            names = [re.sub(r'_(\d+)_(.).*', lambda m: m[1] + m[2].lower(), s[2][0]) for s in e['support'][:2]]
            texts += [e['name'] + ' (' + ', '.join(f'{s[0]:.1f} {n}' for s, n in zip(e['support'], names)) + ')'] if names else []
            texts += [f"against {e['name']} ({a[0]:.1f} fits {a[4]})" for a in e['against'][:1]]
        # the linkage of a lone Neu5Ac read from how intense [M-H-Neu5Ac]- is, after the residue counts, which it is not part of
        readings = [f"Neu5Ac linkage reads {e['linkage'][2]}" + ('' if e['linkage'][3] else ', not as shown') + f" ([M-H-Neu5Ac]- {e['linkage'][0]:.1f} at {e['linkage'][1]:.0%})"
                    for e in support['residues'] if ion_trap and e.get('linkage')]
        supported_by.append('; '.join(([f"{sum(bool(e['support']) for e in tested)}/{len(tested)} residues supported" + (': ' + '; '.join(texts) if texts else '')] if tested else []) + readings))
    df_out['supported_by'] = supported_by
    if 're_offset' in df_out.columns:
        df_out = df_out.drop(columns = ['re_offset', 'unlabelled'] + (['reducing_end'] if df_out['reducing_end'].isna().all() else []))
    if (df_out['rel_abundance'] == 0).all():
        df_out = df_out.drop(columns = ['rel_abundance'])
    # By m/z, but the rows of one ion (a composition and charge, whose m/z differ by noise) by RT; before the spectra are split off, which keep the row order
    df_out = df_out.iloc[np.argsort(mz_order(df_out.index, df_out['RT'], df_out['charge'].values, df_out['composition'].tolist()))]
    spectra_out = df_out.pop('peak_d').values.tolist()
    df_out = df_out[['top1_pred', 'predictions'] + [c for c in df_out.columns if c not in ('top1_pred', 'predictions', 'xic')] + (['xic'] if xic_details else [])]
    df_out.index.name = "m/z"
    return df_out, spectra_out


def flag_coeluting_substructures(df_out, glycan_class, sample_prep = 'underivatized', unreduced = False, ms1 = None, mz_tolerance = 0.5,
                                 rt_tolerance = 0.1, min_ratio = 3.0):
    """Notes on rows to check before trusting them: possible in-source fragments and O-glycan peeling products, structures reported at more than one
    peak, and structures outside the run's glycan class\n
    | Arguments:
    | :-
    | df_out (dataframe): finalised predictions (index m/z; top1_pred, RT, charge, rel_abundance)
    | glycan_class (string or list): glycan class of the run, options are "O", "N", "lipid", "free", or a list of several measured together
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
    | unreduced (bool): whether the glycans have a free reducing end, whose anomers make two peaks of one structure; default:False
    | ms1 (tuple): flat MS1 data from process_mzML_stack; with it, an in-source fragment also needs the apex and peak shape of its parent; default:None
    | mz_tolerance (float): largest m/z half-width of an isotope window; default:0.5
    | rt_tolerance (float): largest retention time difference (min) of an in-source fragment and its parent; default:0.1
    | min_ratio (float): how many times more abundant than the row its parent (in-source fragment) or source (peeling product) must be; default:3.0\n
    | Returns:
    | :-
    | Returns df_out with a 'notes' column
    """
    notes = [[] for _ in range(len(df_out))]
    tops, rts, zs, mzs = df_out['top1_pred'].values, df_out['RT'].values, df_out['charge'].values, df_out.index.values
    # Without precursor intensities nothing tells a fragment from its parent
    abundance = df_out['rel_abundance'].values.astype(float) if 'rel_abundance' in df_out.columns and (df_out['rel_abundance'] > 0).any() else None
    graphs = {}
    for g in {t for t in tops if isinstance(t, str) and '{' not in t}:
        try:
            graphs[g] = glycan_to_nxGraph(g)
        except Exception:
            pass
    # In-source fragments: only a fragment that keeps the reducing end (Y-type: the parent minus a non-reducing part) has the mass of a reducing-end-
    # intact glycan, and only without derivatization (a permethylated Y ion has a free hydroxyl where it lost a residue, 14 Da below the permethylated
    # smaller glycan). It co-elutes with its parent and is a small share of it; with MS1 it has the parent's apex and peak shape too. The old rule (any
    # substructure of the first prediction within 0.25 min, whatever its abundance) put the note on curated ground-truth peaks in four of five cases
    if sample_prep == 'underivatized' and abundance is not None:
        half_width = None
        for a in range(len(df_out)):
            parents = {}
            for b in range(len(df_out)):
                # (with MS1, the trace heights decide the ratio, as a row's abundance also holds its adducts and charge states)
                if (a == b or abundance[a] <= 0 or (ms1 is None and abundance[b] < min_ratio * abundance[a]) or abs(rts[a] - rts[b]) > (0.25 if ms1 is not None else rt_tolerance)
                        or tops[a] not in graphs or tops[b] not in graphs or len(graphs[tops[a]]) >= len(graphs[tops[b]])):
                    continue
                # ... from one cleavage: the parent's reducing end and all but one of its non-reducing parts, as two cleavages in the source are rare
                n, matches = subgraph_isomorphism(graphs[tops[b]], graphs[tops[a]], return_matches = True)
                if not n or not any(max(graphs[tops[b]].nodes()) in m and sum(v not in m for u, v in graphs[tops[b]].edges(m)) == 1 for m in matches):
                    continue
                if ms1 is not None:
                    # Both isotope envelopes within 0.5 min of the parent: the fragment's highest point must be the parent's apex (within a scan), its
                    # trace a scaled copy of the parent's (r >= 0.9; ion-trap survey scans give only 10-12 points there), and the parent min_ratio times higher
                    ms1_rts, ms1_mzs, ms1_ints, ms1_offsets = ms1
                    half_width = ms1_mz_calibration(ms1, mzs, rts, zs, mz_tolerance = mz_tolerance)[1] if half_width is None else half_width
                    s0, s1 = np.searchsorted(ms1_rts, rts[b] - 0.5), np.searchsorted(ms1_rts, rts[b] + 0.5, side = 'right')
                    traces = np.zeros((2, max(s1 - s0, 0)))
                    for k, t in enumerate((a, b)):
                        spacing = ISOTOPE_SPACING / max(abs(zs[t]), 1)
                        width = 0.5 * spacing if half_width >= 0.2 * spacing else max(half_width, 5e-6 * mzs[t])
                        centers = mzs[t] + spacing * np.arange(3)
                        for j, si in enumerate(range(s0, s1)):
                            scan_mzs, cum = ms1_mzs[ms1_offsets[si]:ms1_offsets[si + 1]], np.concatenate([[0], np.cumsum(ms1_ints[ms1_offsets[si]:ms1_offsets[si + 1]], dtype = np.float64)])
                            traces[k, j] = np.sum(cum[np.searchsorted(scan_mzs, centers + width)] - cum[np.searchsorted(scan_mzs, centers - width)])
                    if (s1 - s0 < 5 or traces[0].std() == 0 or traces[1].std() == 0 or abs(np.argmax(traces[0]) - np.argmax(traces[1])) > 1
                            or np.corrcoef(traces[0], traces[1])[0, 1] < 0.9 or traces[1].max() < min_ratio * traces[0].max()):
                        continue
                parents[b] = traces[0].max() / traces[1].max() if ms1 is not None else abundance[a] / abundance[b]
            if parents:
                b = min(parents, key = parents.get)
                notes[a].append(f"Possible in-source fragment of {tops[b]} (co-elutes at {100 * parents[b]:.0f}% of its intensity" + (', same peak shape)' if ms1 is not None else ')'))
    # Peeling products: during release by beta-elimination, the reducing-end GalNAc can lose its 3-linked arm, which then elutes as a glycan of its own
    # (6-linked arms stay); a product is a small share of its source
    classes = glycan_classes(glycan_class)
    if 'O' in classes and abundance is not None:
        arms = {}
        for g, graph in graphs.items():
            root = max(graph.nodes())
            link = next((n for n in graph.successors(root) if graph.nodes[n]['string_labels'].endswith('-3')), None)
            if graph.nodes[root]['string_labels'] == 'GalNAc' and link is not None:
                child = next(graph.successors(link))
                arms[g] = graph_to_string(graph.subgraph(descendants(graph, child) | {child}))
        for a in range(len(df_out)):
            sources = [b for b in range(len(df_out)) if b != a and abundance[a] > 0 and abundance[b] >= min_ratio * abundance[a] and tops[b] in arms and isinstance(tops[a], str)
                       and tops[a].count('(') == arms[tops[b]].count('(') and compare_glycans(tops[a], arms[tops[b]])]
            if sources:
                b = max(sources, key = lambda b: abundance[b])
                notes[a].append(f"Possible peeling product of {tops[b]} (its 3-linked arm, at {100 * abundance[a] / abundance[b]:.0f}% of its abundance)")
    # A fully specified structure elutes as one peak (two anomers if unreduced), so of several peaks with it at most one is likely that isomer (on the
    # test sets, 45% of such rows had the ground-truth structure, against 72% of the other rows)
    if not unreduced:
        same = defaultdict(list)
        for i, t in enumerate(tops):
            if isinstance(t, str) and not re.search(r'\?|/|\{|\b(Hex|HexNAc|HexN|HexA|dHex|Pen|Sia)\b', t):
                same[t].append(i)
        for rows in same.values():
            for i in rows:
                others = ', '.join(f'{rts[j]:.2f}' for j in sorted(rows, key = lambda j: rts[j]) if j != i)
                if others:
                    notes[i].append(f"Same structure as the peak at {others} min; one of the two is likely another isomer" if len(rows) == 2 else
                                    f"Same structure as the peaks at {others} min; all but one of these {len(rows)} are likely other isomers")
    # Structures of another class (an N-glycan in an O-glycan run) are real glycans in most cases, but outside what the run was set up for
    names = {'N': 'an N-glycan', 'O': 'an O-glycan', 'free': 'a free glycan', 'lipid': 'a glycolipid glycan', 'lipid/free': 'a free or glycolipid glycan'}
    run = ' or '.join(names[c] for c in classes)
    for i, t in enumerate(tops):
        if isinstance(t, str) and not in_classes(t, classes):
            name = names.get(get_class(t), 'a glycan of another class')
            notes[i].append(f"{name[0].upper() + name[1:]} by structure, in {run} run")
    df_out['notes'] = ['; '.join(n) for n in notes]
    return df_out


def check_reducing_end(spectra_filepath, glycan_class, run_args, loaded = None, n_top = None, n_chosen = None, condensed = None):
    """checks whether another reducing end explains a file better than the chosen one (a forgotten label, or the default 'reduced' for unreduced glycans)\n
   | Arguments:
   | :-
   | spectra_filepath (string): spectra file, as for wrap_inference
   | glycan_class (string or list): glycan class, as for wrap_inference
   | run_args (dict): every other wrap_inference argument, with the chosen reducing end as modification and mass_tag
   | loaded (dataframe): the file as load_spectra_filepath returned it, so it is not read again; default:None
   | n_top (int): model predictions of the chosen reducing end among the file's 100 most intense clusters; default:None (processes those clusters)
   | n_chosen (int): model predictions of the chosen reducing end in the whole file; default:None (processes the file)
   | condensed (dict): wrap_inference's condensed clusters of loaded, so they are not condensed again for every reducing end; default:None\n
   | Returns:
   | :-
   | Returns the better reducing end as (modification, mass_tag) and the intermediate wrap_inference result of the file with it, or (None, None)
   """
    # A wrong reducing end shifts every precursor and Y/Z ion, so the model finds few of its structures in the compositions and fragment annotations it gets:
    # under 20 among the 100 most intense clusters, against 24-45 with the right one on the test sets (and up to 14 with a wrong one). Those clusters are then
    # processed with every other reducing end (unreduced, 2AA, 2AB, procainamide, PA); if one gives at least 1.5x as many model predictions (and at least 2), the
    # whole file is processed with it, which takes over if it gives at least 1.5x as many (and at least 5) as the chosen reducing end did. The random state is
    # restored, so a run the check leaves alone stays as it was
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state() if torch is not None else None)
    args = {k: v for k, v in run_args.items() if k != '_return_intermediate'}
    chosen = ('custom' if args['modification'] in (None, 'custom') else args['modification'], args['mass_tag'])
    ends = {'reduced': ('reduced', None), 'unreduced': ('custom', None), **{k: (k, None) for k in ('2AA', '2AB', 'procainamide', 'PA')}}
    names = {v: k for k, v in ends.items()}
    found, out = ({} if n_top is None else {chosen: n_top}), (None, None)
    # Condensing the whole file took as long as the rest of each run on its 100 most intense clusters, and it is the same for every reducing end (but for
    # orbitrap isotope corrections, which the cache key catches)
    condensed = {} if condensed is None else condensed
    for end in [chosen] + [e for e in ends.values() if e != chosen]:
        if found.get(chosen, 0) >= 20:
            break
        if end not in found:
            df_end = wrap_inference(spectra_filepath, glycan_class, **{**args, 'modification': end[0], 'mass_tag': end[1]}, _return_intermediate = True,
                                    _loaded = loaded, _top_clusters = 100, _condensed = condensed)
            found[end] = (df_end['evidence'] == 'strong').sum() if 'evidence' in df_end.columns else 0
    best = max(found, key = found.get)
    if found[chosen] < 20 and found[best] >= max(2, 1.5 * found[chosen]):
        if n_chosen is None:
            df_end = wrap_inference(spectra_filepath, glycan_class, **args, _return_intermediate = True, _loaded = loaded, _condensed = condensed)
            n_chosen = (df_end['evidence'] == 'strong').sum() if 'evidence' in df_end.columns else 0
        df_end = wrap_inference(spectra_filepath, glycan_class, **{**args, 'modification': best[0], 'mass_tag': best[1]}, _return_intermediate = True,
                                _loaded = loaded, _condensed = condensed)
        n_best = (df_end['evidence'] == 'strong').sum() if 'evidence' in df_end.columns else 0
        chosen_name = names.get(chosen, f'custom (mass_tag {chosen[1]})')
        # A wrong reducing end gave at most 25 model predictions on the test sets (5-187 with the right one), so a file with 50 or more under the chosen one
        # holds glycans of that reducing end: GPST000461's partly unreduced permethylated milk oligosaccharides gave 127 reduced against 158 unreduced
        switch = n_best >= max(5, 1.5 * n_chosen) and n_chosen < 50
        print(f"WARNING: With {chosen_name} glycans, {os.path.basename(spectra_filepath)} gave {n_chosen} predicted structures; with {names[best]} glycans, it gives "
              f"{n_best}. " + (f"Overriding the reducing end to '{names[best]}'." if switch else f"Keeping '{chosen_name}', but check the reducing end."))
        if switch:
            out = (best, df_end)
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    if torch is not None:
        torch.set_rng_state(rng[2])
    return out


def wrap_inference(spectra_filepath, glycan_class, model = candycrunch, glycans = glycans, bin_num = 2048,
                   max_charge = -3, frag_num = 100, modification = 'reduced', mass_tag = None, lc = 'PGC', trap = 'linear',
                   rt_min = 0, rt_max = 0, rt_diff = 1.0, rt_max_default = 30.0,
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False,
                   extra_thresh = 0.2, crumbs_thresh = 3, ppm_thresh = 300,
                   filter_out = {'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True,
                   experimental = True, mass_dic = None, sample_prep = 'underivatized',
                   taxonomy_level = 'Class', taxonomy_filter = 'Mammalia', df_use = None, plot_glycans = False,
                   sialic_derivatization = None, reducing_end_variants = None, n_jobs = 1, return_inputs = False, _return_intermediate = False, _loaded = None, _top_clusters = None,
                   _condensed = None, _pair = None):
    """wrapper function to get & curate CandyCrunch predictions\n
   | Arguments:
   | :-
   | spectra_filepath (string or dataframe): absolute filepath ending in ".raw" (Thermo), ".d" (Bruker timsTOF or ion trap folder), ".mzML", ".mzXML", ".mgf", ".cc", or ".xlsx" pointing to a file containing
   |                            spectra or preprocessed spectra; MS3 spectra in it (.raw/.mzML/.mzXML, or .xlsx from extract_spectra) are used automatically, as fragment
   |                            evidence and to rank isomers; or a dataframe of spectra as load_spectra_filepath returns them (survey scans in attrs['ms1'], named by
   |                            attrs['name']), such as the pseudo-spectra of a DIA run
   | glycan_class (string or list): glycan class as string, options are "O", "N", "lipid", "free", or a list of the classes measured together in one
   |                               run (e.g., ["N", "O"]): candidate compositions and database structures then come from all of them, each candidate goes to
   |                               the model as a glycan of its own class, and only structures outside all of them need extra_thresh.
   |                               An "N" or "O" run with at least 2 rows of a structure only the other of the two has also looks for glycans of that class
   |                               among the peaks it did not report (e.g., O-glycans carried into an N-glycan release) and adds them
   | model (PyTorch or callable): trained CandyCrunch model, or any other structure predictor as a callable that takes a dataframe with one row per
   |                              (pooled spectrum, candidate composition) input, in input order (index m/z; columns spec_id, RT, intensity, peak_d,
   |                              num_spectra, precursor_charge, charge (unsigned), composition, compositional_vector, annotation_score; run settings
   |                              glycan_class, mode, modification, mass_tag, lc, trap, sample_prep in .attrs) and returns, per row, a list of
   |                              (IUPAC structure, probability) tuples; temperature and test-time augmentation only apply to a CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch, and the vocabulary a callable model's structures are written like; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): spectra before this retention time (in minutes) are ignored; default:0 (runs longer than 20 min then skip their first 2 min)
   | rt_max (float): spectra after this retention time (in minutes) are ignored; default:0 (runs longer than 40 min then skip their last 10%)
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
   | temperature (float): the temperature factor used to calibrate logits; default:1.15
   | spectra (bool): whether to also output the actual spectra used for prediction; default:False
   | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
   | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); below it, they are only allowed if more likely than every in-class structure of that composition; default:0.2
   | crumbs_thresh (float): threshold for annotation score to keep predictions; 2 lower for files with orbitrap or timsTOF survey scans, whose accurate precursor mass already confirms the composition; default:3
   | ppm_thresh (float): ppm error threshold for filtering; default:300
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:{'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}
   | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
   | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
   | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
   | taxonomy_level (string): taxonomy level of taxonomy_filter; default:'Class'
   | taxonomy_filter (string): only glycans of this taxon in df_use supply candidate compositions and database structures; default:'Mammalia'
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | plot_glycans (bool): whether to save an .xlsx file with SNFG images of all top1 predictions next to spectra_filepath, named like it plus _output; default:False
   | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization of the (underivatized) sample: 'EE' (ethyl esterification), 'EEA'
   |                                          (ethyl esterification and amidation), 'ME' (methyl esterification), 'DMA' (dimethylamidation), 'SALSA', 'GQ' (glycoqueuing:
   |                                          aniline on every sialic acid, which tells no linkage; with modification 'GirardP'), 'GQ-d5' (d5-aniline on a2-6), or a pair of
   |                                          glycowork's SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids. Precursor masses then tell how many sialic acids are a2-6
   |                                          and how many a2-3/8/9 (compositions as H5N4E1L1), structures must have these linkages (unknown ones are filled in where
   |                                          the mass decides them), and fragments carry the derivatized sialic acids. The model was trained on underivatized
   |                                          spectra; default:None
   | reducing_end_variants (dict): other reducing ends that glycans of the (underivatized) sample may carry, as name: group on the free glycan's reducing end
   |                               (formula or mass; 0 for a free reducing end), e.g. {'free': 0} for glycans that escaped reduction or labelling, whose
   |                               abundance and spectra then count for their glycan's row, or {'ethyl glycoside': 'C2H4'}, reported as rows of their own at
   |                               their observed m/z and named in a column reducing_end. Only structures the run reports with its own reducing end are searched
   |                               for; default:None (free reducing ends of reduced or labeled glycans, plus ethyl glycosides for 'lipid', as released
   |                               glycosphingolipid glycans form them, and for 'PA' glycosylamines, the PA glycan 2 H lighter, whose abundance and spectra
   |                               count for their glycan's row like those of unlabeled glycans)
   | n_jobs (int): number of processes for fragment scoring (CandyCrumbs) and supporting peaks, which are pure Python; the model already uses every core.
   |               The processes stay up for later runs; a script using n_jobs > 1 needs if __name__ == '__main__' on Windows and macOS; default:1
   | return_inputs (bool): whether to return the model's inputs instead of predictions, as a callable model receives them (e.g., to build training data from
   |                       runs with known structures); the model and everything after it, including the reducing-end and glycan-class checks, are
   |                       skipped, so modification and glycan_class have to be right; default:False\n
   | Returns:
   | :-
   | Returns dataframe of predictions for spectra in file; if spectra=True, a tuple of (dataframe, list of its MS2 spectra) with the column xic (the MS1
   | integrations each row's abundance sums, see finalise_predictions), and files with MS3 spectra
   | then also have a column ms3 (list of (isolated MS2 fragment m/z, MS3 peak dictionary) per row); with return_inputs, the dataframe of model inputs,
   | each with the binned_intensities, mz_remainder, RT2 and glycan_type it was encoded with
   """
    # For processing the file as another glycan class, if its predictions say that it holds that class rather than glycan_class
    run_args = {k: v for k, v in locals().items() if k not in ('spectra_filepath', 'glycan_class', '_return_intermediate', '_loaded', '_top_clusters', '_condensed', '_pair')}
    # Spectra already in memory are processed as if loaded from a file named after attrs['name']
    if isinstance(spectra_filepath, pd.DataFrame):
        spectra_filepath, _loaded = spectra_filepath.attrs.get('name', 'spectra'), spectra_filepath
    # A timsTOF .d is a folder, which a shell completes with a trailing separator
    spectra_filepath = os.path.normpath(spectra_filepath)
    # ppm_thresh is the only tolerance the user sets; derive the flat-Da window everything downstream needs from it here
    mass_tolerance = ppm_thresh * MZ_REF / 1e6
    # 'custom' labels are described by mass_tag alone; glycowork's mass functions only know the named modifications
    modification = None if modification == 'custom' else modification
    mode = "negative" if max_charge < 0 else "positive"
    # Several glycan classes measured in one run (e.g., N- and O-glycans of one sample) are a list; a list of one class is that class
    classes = glycan_classes(glycan_class)
    glycan_class = classes[0] if len(classes) == 1 else classes
    if sialic_derivatization is not None and sample_prep != 'underivatized':
        raise ValueError(f"sialic_derivatization is a derivatization of underivatized glycans, not of {sample_prep} ones.")
    if not _return_intermediate:
        print(f"Your chosen settings are: {'+'.join(classes)} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type.isin(classes)])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
        if df_use.empty:
            raise ValueError(f"No {'+'.join(classes)} glycans of {taxonomy_level} '{taxonomy_filter}' in df_glycan; check taxonomy_level and taxonomy_filter")
    multiplier = -1 if mode == 'negative' else 1
    loaded_file = _loaded.copy() if _loaded is not None else load_spectra_filepath(spectra_filepath, extract_ms1 = spectra_filepath.lower().endswith(('.mzml', '.mzxml', '.raw', '.d', '.cc')))
    ms1 = loaded_file.attrs.pop('ms1', None)
    # The spectra as loaded, for processing the file again with another reducing end without reading it again
    loaded_spectra = None if _return_intermediate else loaded_file.copy()
    detected_mode = loaded_file.attrs.get('detected_mode')
    detected_trap = loaded_file.attrs.get('detected_trap')
    if detected_mode and detected_mode != mode:
        print(f"WARNING: File contains {detected_mode}-mode spectra but mode='{mode}' was specified. Overriding to '{detected_mode}'.")
        mode = detected_mode
        multiplier = -1 if mode == 'negative' else 1
        # The sign of max_charge is the ion mode for everything that takes it (e.g., mz_to_composition in domain_filter)
        max_charge = abs(max_charge) * multiplier
    if detected_trap and detected_trap != trap:
        print(f"WARNING: File was acquired on {detected_trap} but trap='{trap}' was specified. Overriding to '{detected_trap}'.")
        trap = detected_trap
    # An orbitrap or timsTOF survey scan pins a precursor's composition to a few ppm, so fewer annotated fragments need to back it up; a lower threshold loses on ion-trap MS1 files
    ms1_ppm_floor = MS1_PPM_FLOORS.get(loaded_file.attrs.get('detected_ms1_trap'), 0.0)
    accurate_ms1 = ms1_ppm_floor > 0
    # The accurate mass only makes up for fragment evidence within glycan_class; a composition of another class keeps the full threshold
    cross_crumbs_thresh = crumbs_thresh
    if accurate_ms1:
        crumbs_thresh = max(crumbs_thresh - 2, 0)
    loaded_file = filter_rts(loaded_file, rt_min, rt_max)
    if loaded_file.empty:
        # Without MS2 spectra there are no glycan peaks, so this is the empty result of any other file without them
        print(f"WARNING: {os.path.basename(spectra_filepath)} has no MS2 spectra with a precursor m/z in its RT range, so it has no glycan peaks "
              f"(data-independent acquisition such as Waters MSE has none).")
        df_out = pd.DataFrame(columns = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d', 'annotation_score', 'rel_abundance',
                                         'top_fragments'], index = pd.Index([], name = 'm/z'))
        if _return_intermediate:
            df_out.attrs.update(ms1 = ms1, mode = mode, accurate_ms1 = accurate_ms1)
            return df_out
        df_out.insert(0, 'top1_pred', [])
        df_out['ppm_error'] = []
        return (df_out, []) if spectra else df_out
    intensity = 'intensity' in loaded_file.columns and not (loaded_file['intensity'] == 0).all() and not loaded_file['intensity'].isnull().all()
    # Prepare file for processing
    loaded_file.dropna(subset = ['peak_d'], inplace = True)
    idx_col = 'm/z' if 'm/z' in loaded_file.columns else 'reducing_mass'
    # A targeted run (PRM, tMS2) isolates a list of precursor m/z over and over, without survey scans and so without precursor intensities; there each
    # MS2's own signal (PRM quantifies by fragments) traces its target's elution, so it finds the peaks and abundances that precursor intensities give
    targeted = not intensity and len(loaded_file) >= 100 and (loaded_file[idx_col].round(4).map(loaded_file[idx_col].round(4).value_counts()) >= 20).mean() >= 0.9
    if targeted:
        loaded_file['intensity'] = [float(sum(d.values())) for d in loaded_file['peak_d']]
        intensity = True
    if intensity:
        loaded_file.loc[loaded_file['intensity'].isnull(), 'intensity'] = 0
    else:
        loaded_file['intensity'] = [0] * len(loaded_file)
    # A fixed tie-break jitter, so the output does not depend on the global random state
    loaded_file[idx_col] += np.random.RandomState(0).uniform(10 ** (-20), 0.00001, size = len(loaded_file))
    common_structure_map, df_use, topo_struct_map = create_struct_maps(df_use, classes, filter_out = filter_out,
                                                                       phylo_level = taxonomy_level,
                                                                       phylo_filter = taxonomy_filter)
    if (loaded_file.attrs.get('detected_ms1_trap') == 'orbitrap' or targeted) and 'precursor_charge' in loaded_file.columns and loaded_file['precursor_charge'].notna().any():
        # With a header charge, the precursor m/z is the instrument's own monoisotopic pick in the orbitrap survey scan, which for large multiply charged
        # glycans sometimes lands on the 13C isotope. Such a precursor sits one or two isotope spacings / z above a
        # composition, and only the accurate m/z tells this from a composition at the m/z itself (isobaric combinations, e.g., Neu5Gc2 vs Hex2Neu5Ac, differ from
        # 13C by several ppm), so it moves down when only an isotope fits within the file's own calibration (median and spread of the ppm errors within 20 ppm).
        # A targeted run's precursors are its targets as set, computed m/z that are often of the 13C peak (the most intense isotope of large glycans: all 30
        # PRM targets of GPST000403, at 0.6-5 ppm), so they move down alike, within 10 ppm of the computed m/z instead of a calibration
        known = loaded_file['precursor_charge'].notna().values
        charges = np.abs(loaded_file['precursor_charge'].values[known].astype(float)).reshape(-1, 1)
        neutral = np.array([composition_to_mass(c, sample_prep = sample_prep, modification = modification, sialic_derivatization = sialic_derivatization)
                            for comp in df_use.groupby('comp_str').first()['Composition'] for c in ([comp] if sialic_derivatization is None else sialic_linkage_splits(comp))]) + (mass_tag or 0)
        mzs, z_all = loaded_file[idx_col].values[known], charges.ravel()
        def isotope_ppm(k, width):
            # The ppm errors of precursors (less k isotope spacings) to the composition ions of their charge, only for the compositions in a window of a
            # little more than width ppm around each, found in their sorted m/z (comparing every precursor with every composition was 4% of an orbitrap run)
            rows, errors = [np.zeros(0, dtype = int)], [np.zeros(0)]
            for z in np.unique(z_all):
                at = np.flatnonzero(z_all == z)
                theo = np.sort((neutral + multiplier * z * PROTON_MASS) / z)
                x = mzs[at] - k * ISOTOPE_SPACING / z_all[at]
                lo = np.searchsorted(theo, x / (1 + (width + 1) * 1e-6))
                n = np.searchsorted(theo, x / (1 - (width + 1) * 1e-6), side = 'right') - lo
                rows.append(np.repeat(at, n))
                t = theo[np.repeat(lo - np.cumsum(n) + n, n) + np.arange(n.sum())]
                errors.append((mzs[rows[-1]] - k * ISOTOPE_SPACING / z_all[rows[-1]] - t) / t * 1e6)
            return np.concatenate(rows), np.concatenate(errors)
        exact = isotope_ppm(0, 20)[1]
        exact = exact[np.abs(exact) < 20]
        if targeted or len(exact) >= 5:
            cal = 0 if targeted else np.median(exact)
            tol = 10 if targeted else max(3 * 1.4826 * np.median(np.abs(exact - cal)), 2)
            fits = []
            for k in range(3):
                rows, errors = isotope_ppm(k, abs(cal) + tol)
                fits.append(np.isin(np.arange(len(mzs)), rows[np.abs(errors - cal) < tol]))
            loaded_file.loc[known, idx_col] -= np.where(fits[0], 0, np.where(fits[1], 1, np.where(fits[2], 2, 0))) * ISOTOPE_SPACING / charges.ravel()
            if targeted and (fits[0] | fits[1] | fits[2]).mean() >= 0.5:
                # Targets set from computed m/z then pin their compositions like orbitrap survey scans, within the 10 ppm they were matched to (the z = 3
                # targets of GPST000403 sit 3-5 ppm below their compositions, its z = 2 ones 0.7 ppm)
                ms1_ppm_floor, accurate_ms1, crumbs_thresh = 10.0, True, max(crumbs_thresh - 2, 0)
    # Group spectra by mass/retention isomers and process them for being inputs to CandyCrunch. The reducing end and glycan class checks process the same
    # spectra again, so the clusters are kept for them, keyed by the precursor m/z and RT (only the isotope correction above can differ between their runs)
    # With accurate survey scans, spectra join a peak within 20 ppm instead of the flat-Da window: at 0.48 Da, z = 3/4 peaks pooled the MS2 triggered on their
    # 13C peaks 0.25-0.33 m/z higher (GPST000403 978.83 with 979.16, 1199.62 with 1199.95/1200.29). GPST000403 +0.033, GPST000548 +0.005, GPST000134 +0.008,
    # timsTOF serum -0.006 (an open isomer peak), 10 and 50 ppm alike; matching compositions within 20 ppm as well lost (GPST000548 -0.047, 461_O -0.041)
    mz_ppm = 20 if accurate_ms1 else None
    condensed = {} if _condensed is None else _condensed
    # Clusters follow the MS1 peaks of their precursor only for free reducing ends, whose anomers make two peaks 0.5-2 min apart
    regroup = accurate_ms1 and modification is None and not mass_tag
    key = (loaded_file[idx_col].values.tobytes(), loaded_file['RT'].values.tobytes(), mass_tolerance, rt_diff, bin_num, regroup)
    if key not in condensed:
        condensed[key] = condense_dataframe(loaded_file, mz_diff = mass_tolerance, rt_diff = rt_diff, bin_num = bin_num, mz_ppm = mz_ppm,
                                            merged = condensed.setdefault('merged', {}), ms1 = ms1 if regroup else None)
    df_out = condensed[key].copy()
    # The most intense clusters, which the reducing end check below looks at
    top_clusters = df_out.sort_values(['intensity', 'num_spectra'], ascending = False, kind = 'stable').iloc[:_top_clusters or 100]
    if _top_clusters:
        df_out = top_clusters.sort_index()
    if _pair is not None:
        # The other-class pass below: clusters at a row the run already reports, or of one spectrum (it keeps rows from >= 2), cannot add a row
        mzs, rts = np.array(_pair[1], dtype = float).reshape(-1, 2).T
        df_out = df_out[(df_out['num_spectra'].values >= 2) & ~((np.abs(df_out['m/z'].values[:, None] - mzs) < mass_tolerance) & (np.abs(df_out['RT'].values[:, None] - rts) < rt_diff)).any(axis = 1)]
    # Other reducing ends (unreduced glycans that escaped reduction, ethyl glycosides formed on ethanol during glycosphingolipid glycan release): neutral mass
    # difference to the run's reducing end -> name; the condensed spectra are searched for them once the run's own glycans are known (not in the reducing end
    # check's runs on the most intense clusters)
    # PA labeling leaves part of the glycans as their glycosylamine (the Schiff base not reduced, 2 H lighter; GPST000750 lists 162 such rows of 1536 and its
    # Deriv_A1-1_8 holds 6N-TRF(6) as glycosylamine at 39.17 min, twice the apex of the PA glycan at 42.72), a side product like an unlabeled glycan
    variants = ({'free': 0} | ({'ethyl glycoside': 'C2H4'} if 'lipid' in classes else {}) | ({'glycosylamine': 'C5H4N2-O'} if modification == 'PA' else {})
                if reducing_end_variants is None else reducing_end_variants)
    label = modification_mass_dict.get(modification, 0) + (mass_tag or 0)
    re_names = {(calculate_adduct_mass(g) if isinstance(g, str) else g) - label: name for name, g in variants.items()} if (
        sample_prep == 'underivatized' and not _top_clusters) else {}
    re_names = {offset: name for offset, name in re_names.items() if abs(offset) >= 0.01}
    df_cond = df_out.copy() if re_names else None
    # Compositions of the other glycan classes, for spectra that no composition of glycan_class explains. Only accurate survey scans tell them
    # apart from an in-class composition at the same nominal mass: with ion-trap MS1 they added as many glycans outside the ground truth as
    # inside it. The 20 ppm allow for the few ppm of orbitrap calibration offset seen in practice (up to -9 ppm)
    cross_class = None
    if accurate_ms1 and any(c not in classes for c in ['N', 'O', 'free']):
        cross_maps = [create_struct_map(df_glycan, c, filter_out = filter_out, phylo_level = taxonomy_level, phylo_filter = taxonomy_filter)
                      for c in ['N', 'O', 'free'] if c not in classes]
        cross_class = (pd.concat([m[1] for m in cross_maps]), {k: v for m in cross_maps[::-1] for k, v in m[0].items()},
                       {k: v for m in cross_maps[::-1] for k, v in m[2].items()})
    df_out = assign_candidate_structures(df_out, df_use, common_structure_map, topo_struct_map, mass_tolerance, mode,
                                         mass_tag, modification = modification, max_charge = max_charge,
                                         sample_prep = sample_prep, cross_class = cross_class, ms1_ppm = 20 if accurate_ms1 else None,
                                         sialic_derivatization = sialic_derivatization)
    if _pair is not None:
        # and only compositions of the other class can give one of its structures, so only they go to the model
        pair_comps = set(df_use.loc[df_use['glycan_type'] == _pair[0], 'comp_str'])
        df_out = df_out[[isinstance(c, dict) and stringify_dict(c) in pair_comps for c in df_out['composition']]]
    # Orbitrap and TOF MS2 hold hundreds of fragments, which the 0.48 Da window annotated by chance: junk spectra of a permethylated serum run (Q Exactive
    # HCD) reached annotation scores of 4-10 and filled 400+ rows through the accurate-MS1 database fallback, while its real glycans scored 9-46
    # TOF MS2 fragments sit 10-40 ppm off, 0.02 Da lost glycans there
    fragment_tolerance = {'orbitrap': 0.02, 'other': 0.05}.get(trap)
    # Orbitrap fragment errors scale with m/z (0.02 Da is 100 ppm at m/z 200), so they are matched within 15 ppm, as the app annotates them: GPST000548 +0.010,
    # GPST000403 +0.001 (10 ppm +0.010/+0.006, 20 ppm as 15). TOF MS2 keep 0.05 Da: 50 ppm lost GPST000570 N 0.032 and O 0.015
    fragment_ppm = 15 if trap == 'orbitrap' else None
    df_out = assign_annotation_scores_pooled(df_out, multiplier, mass_tag, mass_tolerance, modification = modification,
                                             sample_prep = sample_prep, sialic_derivatization = sialic_derivatization, n_jobs = n_jobs,
                                             fragment_tolerance = fragment_tolerance, fragment_ppm = fragment_ppm, deisotoped = condensed.setdefault('deisotoped', {}))
    own_best = df_out.groupby('spec_id')['annotation_score'].max() if 'glycosylamine' in re_names.values() else None
    df_out = df_out[df_out['compositional_vector'].notnull()].reset_index(drop = True)
    # Spectra without an annotation above crumbs_thresh are dropped unless rescued, so their inputs get one plain pass instead of 5 augmented ones
    # The model knows permethylated spectra by their own modification code (as build_training_pickles encodes them), whatever the reducing end. A free
    # reducing end (custom without mass_tag) is fed as reduced: build_training_pickles codes every reducing end but reduced and permethylated as 2, with
    # the labels, while free and reduced glycans fragment alike (reducing-end ions 2 Da apart); GPST000578 0.887 -> 0.932, GPST000570_N 0.651 -> 0.667
    # In a run of several classes, each candidate goes to the model as a glycan of the class whose database structure it is
    if len(classes) == 1:
        coded_class = CLASS_CODES[glycan_class]
    else:
        struct_class = {}
        for c in classes:
            for g in df_use.loc[df_use['glycan_type'] == c, 'glycan']:
                struct_class.setdefault(g, c)
        coded_class = [CLASS_CODES[struct_class.get(g, classes[0])] for g in df_out['candidate_structure']]
    loader, df_out = process_for_inference(df_out, coded_class, mode = mode, modification = 'permethylated' if sample_prep == 'permethylated' else (
        'reduced' if modification is None and not mass_tag else modification), lc = lc, trap = trap,
                                           rt_max_default = rt_max_default, tta_thresh = crumbs_thresh, keep_inputs = return_inputs and not _return_intermediate)
    if return_inputs and not _return_intermediate:
        # One row per input (pooled spectrum and candidate composition; index its m/z), as a callable model gets them, with the settings it was encoded with
        df_in = df_out.drop_duplicates('input_id').drop(columns = ['input_id', 'tta'])
        df_in.attrs.update(glycan_class = glycan_class, mode = mode, modification = modification, mass_tag = mass_tag, lc = lc, trap = trap,
                           sample_prep = sample_prep, accurate_ms1 = accurate_ms1)
        return df_in
    if isinstance(model, CandyCrunchCNN) or (torch is not None and isinstance(model, torch.nn.Module)):
        # Predict glycans from spectra
        preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
        # Max over the 5 augmented copies of each test-time augmented input (they come first in the loader), a plain input has one copy
        tta = df_out.drop_duplicates('input_id')['tta'].values
        order = np.concatenate([np.flatnonzero(tta), np.flatnonzero(~tta)])
        sizes = np.where(tta[order], 5, 1)
        preds_in, conf_in = [None] * len(tta), [None] * len(tta)
        for i, start, n in zip(order, np.cumsum(sizes) - sizes, sizes):
            combs = [dict(zip(cp, cc)) for cp, cc in zip(preds[start:start + n], pred_conf[start:start + n])]
            combs = dict(sorted(average_dicts(combs, mode = 'max').items(), key = lambda x: x[1], reverse = True))
            preds_in[i], conf_in[i] = list(combs.keys()), list(combs.values())
        preds, pred_conf = preds_in, conf_in
    else:
        # Any other model is a callable from the distinct inputs (one row per input_id, in order; run settings in attrs) to (structure, probability)
        # lists. Its structures are written like the vocabulary in glycans (same glycan, same string; no -ol, the reducing end is `modification`),
        # so GlyTouCan IDs, deduplication, biosynthesis and supporting peaks treat them alike, and repeats of one structure (e.g. beams) add up
        df_in = df_out.drop_duplicates('input_id').drop(columns = ['input_id', 'tta'])
        df_in.attrs.update(glycan_class = glycan_class, mode = mode, modification = modification, mass_tag = mass_tag, lc = lc, trap = trap,
                           sample_prep = sample_prep)
        vocab = {canonicalize_iupac(g): g for g in glycans}
        written = {}
        preds, pred_conf = [], []
        for ranked in model(df_in):
            combs = defaultdict(float)
            for g, c in ranked:
                if g not in written:
                    try:
                        written[g] = canonicalize_iupac(g[:-3] if g.endswith('-ol') else g)
                        written[g] = vocab.get(written[g], written[g])
                    except Exception:
                        # A structure glycowork cannot parse is no candidate: glycan_to_composition still reads some (e.g., unbalanced branches in
                        # GlycoBERT's class list), and later steps that parse the structure then fail
                        written[g] = None
                if written[g]:
                    combs[written[g]] += float(c)
            combs = sorted(combs.items(), key = lambda x: x[1], reverse = True)
            preds.append([g for g, _ in combs])
            pred_conf.append([c for _, c in combs])
    df_out['rel_abundance'] = df_out['intensity']
    df_out['predictions'] = [list(zip(preds[i], pred_conf[i])) for i in df_out['input_id']]
    _raw_predictions = df_out['predictions'].tolist()
    # Check correctness of glycan class & mass
    df_out['isomers'] = [[g for g in preds if g[1] > pred_thresh / 10 and get_comp(g[0]) == comp] for preds, comp in
                         zip(df_out.predictions, df_out.composition)]
    if sialic_derivatization is not None:
        # The composition gives each sialic acid's linkage, which replaces the linkages the model learned from underivatized spectra (it gives the topology)
        isomers = []
        for preds, comp in zip(df_out.predictions, df_out.composition):
            resolved = [(fit_sialic_linkages(g, comp), c) for g, c in preds if c > pred_thresh / 10 and get_comp(g) == plain_composition(comp)]
            isomers.append([(g, c) for k, (g, c) in enumerate(resolved) if g is not None and g not in [r[0] for r in resolved[:k]]])
        df_out['isomers'] = isomers
    # A cross-class structure needs extra_thresh, unless it is more likely than every in-class isomer of that composition:
    # the class prior may veto a composition's cross-class structures, but should not make the row pick a less likely isomer.
    # A composition of another class only reached the model because no composition of glycan_class explains the spectrum, so it needs no veto
    best_in_class = [max([g[1] for g in preds if g[1] > pred_thresh and in_classes(g[0], classes)], default = 0) for preds in df_out.isomers]
    df_out['predictions'] = [
        [(g[0], round(g[1], 4)) for g in preds if
         g[1] > pred_thresh and (in_classes(g[0], classes, g[1], extra_thresh = extra_thresh) or 0 < best_in <= g[1] or xc)][:5]
        for preds, best_in, xc in zip(df_out.isomers, best_in_class, df_out['cross_class'])
    ]
    # Cross-validate: for spectra where exact composition equality eliminated all model
    # predictions, run CandyCrumbs on the model's top mass-compatible prediction.
    # Accepted only when fragment annotation independently passes crumbs_thresh.
    _no_pred_indices = [i for i in range(len(df_out)) if not df_out['predictions'].iloc[i]]
    if _no_pred_indices:
        _tag = mass_tag if mass_tag else 0
        _adduct_list = get_adduct_list(mode)
        _model_tops = {}
        for i in _no_pred_indices:
            for g_name, g_conf in _raw_predictions[i]:
                if g_conf <= pred_thresh:
                    break
                if (in_classes(g_name, classes, g_conf, extra_thresh = extra_thresh)
                        and mass_check(df_out.index[i], g_name, mode = mode, modification = modification,
                                       mass_tag = mass_tag, sample_prep = sample_prep,
                                       permitted_charges = [abs(int(df_out['charge'].iloc[i]))],
                                       mass_tolerance = mass_tolerance, sialic_derivatization = sialic_derivatization)):
                    _model_tops[i] = (g_name, g_conf)
                    break
        _struct_to_indices = defaultdict(list)
        for i, (g_name, _) in _model_tops.items():
            _struct_to_indices[g_name].append(i)
        for _struct, _indices in _struct_to_indices.items():
            _pred_comp = get_comp(_struct)
            if _pred_comp and sialic_derivatization is not None:
                # the sialic linkages whose derivatized mass is the precursor's, which the structure then gets for CandyCrumbs
                _pred_comp = next((c for c in sialic_linkage_splits(_pred_comp) if mass_check(
                    df_out.index[_indices[0]], composition_to_mass(c, sialic_derivatization = sialic_derivatization), mode = mode, modification = modification,
                    mass_tag = mass_tag, permitted_charges = [abs(int(df_out['charge'].iloc[_indices[0]]))], mass_tolerance = mass_tolerance)), None)
                _fitted = fit_sialic_linkages(_struct, _pred_comp, scoring = True) if _pred_comp else None
                if _fitted is None:
                    continue
            if not _pred_comp:
                continue
            _pred_comp_mass = composition_to_mass(_pred_comp, sample_prep = sample_prep,
                                                  modification = modification, sialic_derivatization = sialic_derivatization) + _tag
            _charges_arr = np.array([abs(int(df_out['charge'].iloc[i])) for i in _indices])
            _spec_masses = np.array([df_out.index[i] * _charges_arr[j] for j, i in enumerate(_indices)])
            _is_adduct = any(
                np.any(np.abs(_pred_comp_mass + mass_dict.get(adduct, 999) - _spec_masses) < mass_tolerance) or
                np.any(
                    np.abs(_pred_comp_mass + mass_dict.get(adduct, 999) * _charges_arr - _spec_masses) < mass_tolerance)
                for adduct in _adduct_list)
            _row_charge = int(max(_charges_arr))
            _peak_dicts = [df_out['peak_d'].iloc[i] for i in _indices]
            _rounded_mass_rows = [[np.round(y, 1) for y in deisotope_ms2(d, _row_charge, 0.05, relabel = df_out['num_spectra'].iloc[i] >= 20)][:15] for d, i in zip(_peak_dicts, _indices)]
            _unq_rounded = {m for row in _rounded_mass_rows for m in row}
            try:
                _cc_out = CandyCrumbs(_struct if sialic_derivatization is None else _fitted, _unq_rounded, mass_tolerance, simplify = False,
                                      charge = int(multiplier * _row_charge),
                                      disable_global_mods = (not _is_adduct or mode == "negative"),
                                      disable_X_cross_rings = True, disable_A_cross_rings = multiplier > 0, max_cleavages = 2,
                                      mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0),
                                      sample_prep = sample_prep, sialic_derivatization = sialic_derivatization)
            except Exception:
                continue
            _tms = {}
            for _k, _v in _cc_out.items():
                if _v:
                    _filtered = []
                    for ant in _v['Domon-Costello nomenclatures']:
                        prefs = [a.split('_')[0][-1] for a in ant]
                        if ('A' in prefs or 'X' in prefs) and len(prefs) > 1:
                            continue
                        if 'M' in prefs and len(prefs) > 2:
                            continue
                        _filtered.append(ant)
                    _tms[_k] = len(_filtered)
                else:
                    _tms[_k] = 0
            for _idx, _rmasses in zip(_indices, _rounded_mass_rows):
                _score = sum(_tms.get(m, 0) for m in _rmasses)
                if _score > crumbs_thresh:
                    _gn, _gc = _model_tops[_idx]
                    _gn = _gn if sialic_derivatization is None else fit_sialic_linkages(_gn, _pred_comp) or _gn
                    df_out.iat[_idx, df_out.columns.get_loc('predictions')] = [(_gn, round(_gc, 4))]
                    df_out.iat[_idx, df_out.columns.get_loc('isomers')] = [(_gn, round(_gc, 4))]
                    df_out.iat[_idx, df_out.columns.get_loc('composition')] = get_comp(_gn) if sialic_derivatization is None else _pred_comp
                    df_out.iat[_idx, df_out.columns.get_loc('annotation_score')] = float(_score)
    database_rows = set()
    has_pred = np.array([bool(p) for p in df_out['predictions']], dtype = bool)
    if accurate_ms1:
        # Accurate survey scans pin a precursor's composition, so a spectrum without any model prediction whose in-class composition matches within the
        # file's own calibration (median and 3 robust SD, at least 3 ppm, of the rows with a prediction) and whose fragments back up a database structure of
        # it as much as a model prediction without accurate survey scans needs (crumbs_thresh before lowering) takes that structure. The model knows few large
        # glycans (tri- and tetra-antennary sialylated N-glycans), which were lost although their peaks are plain: timsTOF serum N-glycans 45 -> 75 of 113
        # MS1 glycan peaks; together with the trusted charge states, GlycoGauntlet 7_N 0.695 -> 0.796, 8_O 0.698 -> 0.806, 2_O 0.890 -> 0.864 (seeds 0-2;
        # 2_O gains a large O-glycan its ground truth lacks)
        theo = np.array([(composition_to_mass(c, sample_prep = sample_prep, modification = modification,
                                              sialic_derivatization = sialic_derivatization) + (mass_tag or 0)
                          + multiplier * z * PROTON_MASS) / z if isinstance(c, dict) and z else np.nan for c, z in
                         zip(df_out['composition'], df_out['charge'])])
        ppm = (df_out.index.values - theo) / theo * 1e6
        # ppm is to the (de)protonated ion, so rows matched as adducts sit thousands of ppm off; they never pass and must not set the calibration either
        window = accurate_ms1_window(ppm[has_pred], df_out['annotation_score'].values[has_pred], ppm_thresh, floor = ms1_ppm_floor,
                                     groups = [f'{i} {c}' for i, c in zip(df_out['spec_id'].values[has_pred], df_out['composition'].values[has_pred])])
        if window is not None:
            center, width = window
            predicted_specs = set(df_out['spec_id'][has_pred])
            # Orbitrap MS2 hold hundreds of fragments, so there a database structure needs 14 annotations; permethylated orbitrap DDA (GPST000403/548)
            # samples a real chromatographic peak repeatedly, so there it also needs 3 spectra (one-spectrum rows: 56 FP vs 5 TP in GPST000403), while with
            # dynamic exclusion most glycans get one or two (GPST000651, 2AB HILIC: 0.516 with 3 spectra, 0.589 with 2, 0.761 with 1)
            ok = (~has_pred & ~df_out['cross_class'].values & (np.abs(ppm - center) <= width) & (
                        df_out['annotation_score'].values > cross_crumbs_thresh) & (
                    (((df_out['num_spectra'].values >= 3) | (sample_prep != 'permethylated')) & (df_out['annotation_score'].values >= 14)) if trap == 'orbitrap' else True)
                  & ~df_out['spec_id'].isin(predicted_specs).values & np.array(
                        [isinstance(g, str) for g in df_out['candidate_structure']]))
            for i in df_out.reset_index(drop = True)[ok].groupby('spec_id')['annotation_score'].idxmax().values if ok.any() else []:
                g = df_out['candidate_structure'].iloc[i]
                df_out.iat[i, df_out.columns.get_loc('predictions')] = [(g, round(pred_thresh, 4))]
                df_out.iat[i, df_out.columns.get_loc('isomers')] = [(g, round(pred_thresh, 4))]
                database_rows.add(df_out.index[i])
    else:
        # The model knows few Neu5Gc glycans: in ion-trap runs it mostly has no structure of their composition in its top 25 and prefers the composition
        # with Hex + dHex instead (1 Da heavier, inside the precursor tolerance), so a spectrum without any model prediction whose candidate composition with
        # Neu5Gc, at a charge of at most 2, is backed up by more than 10 fragment annotations takes that composition's database structure. The same holds for
        # free glycans, as the model knows few of the large milk oligosaccharides, and for [M+3H]3+ and higher with at least 20 annotations whose fragments
        # pair up into B/C ions and their complements at that charge more often than at z = 2 (complementary_pairs), as the model rarely has a structure of
        # the composition of a large multiply sialylated N-glycan (GPST000237: 10 of its 12 trisialylated compositions); without the pairs it lost GPST000302
        # 0.030, GPST000487 0.028,  at 15 annotations GPST000237 0.515 instead of 0.530. Not for other compositions: there it filled O-glycans into
        # N runs and junk into O runs
        z3 = (df_out['charge'].values >= 3) & (df_out['annotation_score'].values >= 20) & np.array([
            c >= 3 and complementary_pairs(mz, peaks, int(c), 0.75 * mass_tolerance) > complementary_pairs(mz, peaks, 2, 0.75 * mass_tolerance)
            for mz, peaks, c in zip(df_out.index, df_out['peak_d'], df_out['charge'])], dtype = bool)
        # Positive-mode and sialic-derivatized runs lie outside the model's training data (underivatized spectra, mostly negative mode; it gets the plain
        # composition), so there a z <= 2 composition whose fragments clearly out-explain every other candidate composition of its spectrum (annotation score
        # of at least 20 and 1.5 times the next composition's) takes its database structure too: GPST000750 Deriv_A1-1_8 (PA, SALSA, LTQ) 6N-TRF(6) 1281.8
        # [M+2H]2+, 26 against 17. In negative-mode ion-trap runs the same rule filled more wrong large compositions than right ones
        clear = np.zeros(len(df_out), dtype = bool)
        if mode == 'positive' or sialic_derivatization is not None:
            best = defaultdict(dict)
            for spec, c, score in zip(df_out['spec_id'].values, df_out['composition'].values, df_out['annotation_score'].values):
                if isinstance(c, dict):
                    best[spec][stringify_dict(c)] = max(best[spec].get(stringify_dict(c), 0), score)
            clear = np.array([isinstance(c, dict) and score >= 20 and score == best[spec][stringify_dict(c)] and score >= 1.5 * max(
                [v for k, v in best[spec].items() if k != stringify_dict(c)], default = 0) for spec, c, score in zip(
                df_out['spec_id'].values, df_out['composition'].values, df_out['annotation_score'].values)], dtype = bool)
        ok = (~has_pred & ~df_out['cross_class'].values & (df_out['annotation_score'].values >= 10) & ((df_out['charge'].values <= 2) | z3)
              & ~df_out['spec_id'].isin(set(df_out['spec_id'][has_pred])).values & np.array([isinstance(g, str) for g in df_out['candidate_structure']], dtype = bool)
              & (np.array([isinstance(c, dict) and ('Neu5Gc' in c or 'free' in classes) for c in df_out['composition']], dtype = bool) | z3 | clear))
        for i in df_out.reset_index(drop = True)[ok].groupby('spec_id')['annotation_score'].idxmax().values if ok.any() else []:
            g = df_out['candidate_structure'].iloc[i]
            df_out.iat[i, df_out.columns.get_loc('predictions')] = [(g, round(pred_thresh, 4))]
            df_out.iat[i, df_out.columns.get_loc('isomers')] = [(g, round(pred_thresh, 4))]
            database_rows.add(df_out.index[i])
    df_out['charge'] = [c * multiplier for c in df_out['charge']]
    df_out['RT'] = df_out['RT'].round(2)
    # Extract & sort the top frag_num fragments by intensity
    df_out['top_fragments'] = [
        [round(float(frag[0]), 4) for frag in sorted(peak_d.items(), key = lambda x: x[1], reverse = True)[:frag_num]]
        for peak_d in df_out['peak_d']
    ]
    # Fragment evidence overrules the model's top isomer only when it is decisive: an isomer (down to a tenth of pred_thresh, of any class) that explains at
    # least two of the 20 most intense peaks by a single glycosidic cleavage which the top isomer cannot explain at all, beyond any such peaks of the top
    # isomer; before domain_filter, which vets the new order (supporting_ions without reference glycans only compares the candidates, not every
    # placement of each residue), and only for rows that can pass crumbs_thresh
    rows = [(preds, [g for g in isomers if g[0] != preds[0][0]] if preds and score > crumbs_thresh and isinstance(peaks, dict) else [], peaks, charge)
            for preds, isomers, peaks, charge, score in zip(df_out['predictions'], df_out['isomers'], df_out['peak_d'], df_out['charge'], df_out['annotation_score'])]
    # Accurate fragment masses (orbitrap, TOF) resolve isotope envelopes, whose 13C peaks would otherwise pass for fragments one isotope spacing heavier
    supports = iter(run_jobs([(supporting_ions, (preds[0][0], deisotope_ms2(peaks, int(abs(charge)), 0.02) if trap in ('orbitrap', 'other') else peaks),
                               dict(charge = int(charge), candidates = [g[0] for g in others], mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0),
                                    sample_prep = sample_prep, reference_glycans = [], sialic_derivatization = sialic_derivatization))
                              for preds, others, peaks, charge in rows if others], n_jobs))
    reordered = []
    for preds, others, _, _ in rows:
        compared = {c['structure']: len(c['against']) - len(c['support']) for c in next(supports)['candidates']} if others else {}
        best = next((g for g in others if compared.get(g[0], 0) >= 2 and compared[g[0]] == max(compared.values())), None)
        reordered.append([(best[0], round(best[1], 4))] + [g for g in preds if g[0] != best[0]][:4] if best else preds)
    df_out['predictions'] = reordered
    # Filter out wrong predictions via diagnostic ions etc.
    if experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, sample_prep = sample_prep,
                               mass_tag = mass_tag, max_charge = max_charge,
                               modification = modification, mass_tolerance = mass_tolerance,
                               df_use = df_use, sialic_derivatization = sialic_derivatization, trust_header_charge = accurate_ms1,
                               fragment_tolerance = fragment_tolerance, fragment_ppm = fragment_ppm).reset_index()
    else:
        df_out = df_out.reset_index()
    # A spectrum's row is the one with the best fragment evidence, unless its top structure is at least ppm_thresh off the precursor where
    # finalise_predictions cuts by ppm alone (multiply charged rows, or accurate survey scans), so it could never be reported, while another composition of
    # the spectrum has a prediction: at z = 2, Neu5Ac and two dHex (or Neu5Gc and Hex + dHex) lie 0.51 m/z apart, so with an ion-trap precursor both are
    # candidates, and the one 0.4 m/z off often explains a few more fragments by chance
    theo = [mass_check(mz, preds[0][0], modification = modification, mass_tag = mass_tag, mode = mode, sample_prep = sample_prep, mass_tolerance = mass_tolerance,
                       sialic_derivatization = sialic_derivatization) if preds and (accurate_ms1 or abs(charge) > 1) else None
            for mz, preds, charge in zip(df_out['m/z'], df_out['predictions'], df_out['charge'])]
    df_out['doomed'] = [t is not None and (not t or abs(t[0] - mz) / t[0] * 1e6 >= ppm_thresh) for t, mz in zip(theo, df_out['m/z'])]
    # The same for a multiply charged composition without a model structure that joined a singly charged one for its doubly charged fragment pairs
    # (assign_candidate_structures)
    zabs = np.abs(df_out['charge'].values.astype(float))
    df_out['doomed'] |= (zabs >= 2) & np.array([not p for p in df_out['predictions']], dtype = bool) & pd.Series(zabs == 1).groupby(df_out['spec_id'].values).transform('max').values
    # Of the rest, a composition with a model structure counts one annotation more (ties going to it too): an ion-trap precursor fits isobars 0.5 m/z apart at
    # z = 2, and chance annotations of the one without a structure won (Hex4HexNAc4S1 over Man7 779.27, Hex4HexNAc4Neu5Gc1dHex2 over A2G2S1F 1038.87 in GPST000054)
    df_out['_pred'] = np.array([bool(p) for p in df_out['predictions']], dtype = float)
    df_out['_rank'] = df_out['annotation_score'] + df_out['_pred']
    # Neu5Ac + Hex and Neu5Gc + dHex are one formula, so a spectrum is a row of both compositions, and the one with more residue kinds explains more
    # fragments by chance. Of such a pair, the row whose sialic acid leaves the weaker specific ions (its B1 ion, less water, its LacNAc antenna as B and
    # C ion, also with the mannose below, and the precursor less that acid at its charge or one lower) ranks lower, if its rival's reach at least 5% of the
    # base peak and twice its own (an acid's own ions outweighed the other's at least 2.5-fold wherever the acid was known)
    df_out['_sia'] = 0
    if sample_prep == 'underivatized' and sialic_derivatization is None:
        keys = [stringify_dict(c) if isinstance(c, dict) else None for c in df_out['composition']]
        spec_keys = defaultdict(set)
        for spec, key in zip(df_out['spec_id'].values, keys):
            spec_keys[spec].add(key)
        sia_tol = fragment_tolerance if fragment_tolerance else mass_tolerance
        evidence, promoted = {}, set()
        for i, (spec, c, mz, z, peaks) in enumerate(zip(df_out['spec_id'].values, df_out['composition'].values, df_out['m/z'].values, df_out['charge'].values, df_out['peak_d'].values)):
            if not isinstance(c, dict) or not isinstance(peaks, dict) or not peaks:
                continue
            for own, other, swap in (('Neu5Ac', 'Neu5Gc', {'Neu5Ac': -1, 'Hex': -1, 'Neu5Gc': 1, 'dHex': 1}), ('Neu5Gc', 'Neu5Ac', {'Neu5Gc': -1, 'dHex': -1, 'Neu5Ac': 1, 'Hex': 1})):
                rival = {k: c.get(k, 0) + swap.get(k, 0) for k in set(c) | set(swap)}
                rival_key = stringify_dict({k: v for k, v in rival.items() if v})
                if min(rival.values()) < 0 or rival_key not in spec_keys[spec]:
                    continue
                z_abs, base = int(abs(z)), max(peaks.values())
                neutral = mz * z_abs - multiplier * z_abs * PROTON_MASS
                for acid in (own, other):
                    if (spec, z_abs, acid) not in evidence:
                        a = mass_dict[acid]
                        ions = [a + multiplier * PROTON_MASS + w for w in (0, -mass_dict['H2O'])] + [
                            a + mass_dict['Hex'] + mass_dict['HexNAc'] + multiplier * PROTON_MASS + w for w in (0, mass_dict['H2O'])] + [
                            a + 2 * mass_dict['Hex'] + mass_dict['HexNAc'] + mass_dict['H2O'] + multiplier * PROTON_MASS] + [
                            (neutral - a + multiplier * q * PROTON_MASS) / q for q in (z_abs, z_abs - 1) if q > 0]
                        evidence[spec, z_abs, acid] = max(y for t in ions for y in [0] + [y for m, y in peaks.items() if abs(m - t) < sia_tol]) / base
                if evidence[spec, z_abs, other] >= max(0.05, 2 * evidence[spec, z_abs, own]):
                    df_out.iat[i, df_out.columns.get_loc('_sia')] = 1
                    if df_out['predictions'].iloc[i]:
                        promoted.add((spec, rival_key))
        # A row promoted this way without a model structure takes its composition's database structure, as in the Neu5Gc fallback above, if the row it
        # outranks had one: the model often has a structure only for the wrong one of the pair, and the peak, reported before, would go unreported
        for i, (spec, key, preds, g) in enumerate(zip(df_out['spec_id'].values, keys, df_out['predictions'].values, df_out['candidate_structure'].values)):
            if (spec, key) in promoted and not preds and isinstance(g, str):
                df_out.iat[i, df_out.columns.get_loc('predictions')] = [(g, round(pred_thresh, 4))]
                df_out.iat[i, df_out.columns.get_loc('isomers')] = [(g, round(pred_thresh, 4))]
                df_out.iat[i, df_out.columns.get_loc('_pred')] = 1.0
                database_rows.add(df_out['m/z'].iloc[i])
    df_out = df_out.sort_values(['spec_id', 'doomed', '_sia', '_rank', '_pred'], ascending = [False, True, True, False, False]).groupby('spec_id').first().drop(columns = ['doomed', '_sia', '_rank', '_pred'])
    # A sparse spectrum (at most 6 fragments, as an ion trap near its detection limit gives) can never pass crumbs_thresh, so it passes if at least two and 60%
    # of its fragments are annotated and the model is confident in its in-class top structure; a multiply charged one already if one fragment is annotated and
    # the model gives its top structure at least 0.75
    sparse = (df_out['fragments'] <= 6) & ~df_out['cross_class']
    top_conf = np.array([p[0][1] if p else 0 for p in df_out['predictions']])
    passes = (df_out['annotation_score'] > np.where(df_out['cross_class'], cross_crumbs_thresh, crumbs_thresh)) | (
        sparse & (df_out['annotated'] >= np.maximum(2, 0.6 * df_out['fragments'])) & (top_conf >= 0.5)) | (
        sparse & (np.abs(df_out['charge']) > 1) & (df_out['annotated'] >= 1) & (top_conf >= 0.75))
    # One chromatographic peak is one glycan: when a cluster's fragments fail crumbs_thresh while a cluster at least ten times weaker at the same m/z,
    # whose spectra continue it (within 0.25 min), passes with the same top prediction, the tail's evidence counts for the peak, reported at its apex
    tails = []
    for spec, row in df_out[passes & np.array([bool(p) for p in df_out['predictions']], dtype = bool)].iterrows():
        apex = df_out[~passes & ((df_out['m/z'] - row['m/z']).abs() <= mass_tolerance) & (df_out['rel_abundance'] >= 10 * row['rel_abundance'])]
        apex = apex[[bool(p) and p[0][0] == row['predictions'][0][0] and max(r[0] - row['rt_range'][1], row['rt_range'][0] - r[1]) <= 0.25
                     for p, r in zip(apex['predictions'], apex['rt_range'])]]
        if len(apex):
            a = apex['rel_abundance'].idxmax()
            df_out.at[a, 'annotation_score'] = row['annotation_score']
            df_out.at[a, 'num_spectra'] += row['num_spectra']
            passes[a] = True
            tails.append(spec)
    df_out = df_out[passes].drop(tails).drop(columns = ['candidate_structure']).set_index('m/z')
    if 'ms3' in df_out.columns:
        # MS3 spectra resolve isomers whose MS2 fragments look alike: an isomer explaining a row's MS3 spectra with more fragments of the fragment
        # they isolated (scored as in assign_annotation_scores_pooled) moves up, isomers explaining them equally well keep the model's order
        reranked = []
        for preds, ms3, charge, comp in zip(df_out['predictions'], df_out['ms3'], df_out['charge'], df_out['composition']):
            pools = {}
            for p, x in ms3 if len(preds) > 1 else []:
                pools.setdefault(round(p), (p, []))[1].append([np.round(y, 1) for y in deisotope_ms2(x, int(abs(charge)), 0.05)][:15])
            scores = [0] * len(preds)
            for i, (g, _) in enumerate(preds):
                g = g if sialic_derivatization is None else fit_sialic_linkages(g, comp, scoring = True)
                for p, ms3_spectra in pools.values() if g else []:
                    cc_out = CandyCrumbs(g, {m for masses in ms3_spectra for m in masses}, mass_tolerance, simplify = False, charge = int(charge),
                                         disable_global_mods = True, disable_X_cross_rings = True, max_cleavages = 2,
                                         mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0), sample_prep = sample_prep,
                                         ms3_precursor = p, sialic_derivatization = sialic_derivatization)
                    counts = {k: sum(not (len(a) > 1 and {c.split('_')[0][-1] for c in a} & {'A', 'X'}) and not (
                        len(a) > 2 and 'M' in {c.split('_')[0][-1] for c in a}) for a in v['Domon-Costello nomenclatures']) if v else 0
                              for k, v in cc_out.items()}
                    scores[i] += np.mean([sum(counts[m] for m in masses) for masses in ms3_spectra])
            reranked.append([preds[i] for i in sorted(range(len(preds)), key = lambda i: -scores[i])])
        df_out['predictions'] = reranked
    if lc == 'PGC' and len(df_out):
        # On PGC, a sialic acid linked a2-6 elutes before the same glycan with it linked a2-3. Rows whose top structures differ only in such linkages
        # (one family) and lie at least rt_diff apart therefore get structures from their isomers with a non-decreasing number of a2-3 linkages and, if
        # the family's isomers suffice, never the same structure twice (the model tends to name every peak of a family by its most common isomer).
        # The most likely such assignment wins; a row's reported structure counts as its most likely isomer (fragment evidence or MS3 may have put it first)
        sia_link = r'((?:Neu5Ac|Neu5Gc|Kdn)[^()]*\(a2-)([36])\)'
        sialylated = {g for isomers in df_out['isomers'] for g, _ in isomers if re.search(sia_link, g)}
        canon = {g: canonicalize_iupac(g) for g in sialylated}
        family = {g: canonicalize_iupac(re.sub(sia_link, r'\1?)', g)) for g in sialylated}
        n_a23 = {g: sum(m[1] == '3' for m in re.findall(sia_link, g)) for g in sialylated}
        top1 = [p[0][0] if p and p[0][0] in sialylated else None for p in df_out['predictions']]
        for fam in {family[g] for g in top1 if g}:
            rows = sorted([i for i, g in enumerate(top1) if g and family[g] == fam],
                          key = lambda i: df_out['RT'].iloc[i])
            cands = [[(top1[i], max([p for _, p in df_out['isomers'].iloc[i]], default = 1.0))] +
                     [(g, p) for g, p in df_out['isomers'].iloc[i] if family.get(g) == fam and g != top1[i]] for i in
                     rows]
            rts = [df_out['RT'].iloc[i] for i in rows]
            pairs = [(a, b) for a in range(len(rows)) for b in range(a + 1, len(rows)) if rts[b] - rts[a] >= rt_diff]
            if not pairs or np.prod([len(c) for c in cands]) > 50000:
                continue
            best, best_score = None, (False, -np.inf)
            for combo in np.ndindex(*[len(c) for c in cands]):
                picked = [cands[k][c] for k, c in enumerate(combo)]
                if all(n_a23[picked[a][0]] <= n_a23[picked[b][0]] for a, b in pairs):
                    score = (all(canon[picked[a][0]] != canon[picked[b][0]] for a, b in pairs),
                             sum(np.log(max(p, 1e-6)) for _, p in picked))
                    if score > best_score:
                        best, best_score = picked, score
            for i, (g, p) in zip(rows, best or []):
                if g != top1[i]:
                    df_out.iat[i, df_out.columns.get_loc('predictions')] = [(g, round(p, 4))] + [x for x in df_out[
                        'predictions'].iloc[i] if x[0] != g][:4]
    df_out = df_out[
        ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d', 'annotation_score', 'rel_abundance',
         'top_fragments'] + [c for c in ('ms3', 'precursor_charge', 'rt_range') if c in df_out.columns]]
    if not df_out.empty:
        # Deduplicate identical predictions for different spectra
        df_out = deduplicate_predictions(df_out, mz_diff = mass_tolerance, rt_diff = rt_diff, mz_ppm = mz_ppm)
        # Positive-mode rows keep their clusters' RT span for finalise_predictions, which merges (de)protonated rows under one dominant adduct peak
        if _return_intermediate or mode == 'negative':
            df_out = df_out.drop(columns = ['rt_range'], errors = 'ignore')
        # A database structure of a composition that accurate survey scans and fragments confirm is weaker evidence than a model prediction
        df_out['evidence'] = ['composition' if mz in database_rows else 'strong' if preds else np.nan for mz, preds in zip(df_out.index, df_out['predictions'])]
    if re_names and not df_out.empty:
        # A glycan of another reducing end is a side product of a glycan of the run, so the structures the run reports are looked for at each variant's mass
        # (re_offset) in every spectrum, with their best confidence. A spectrum is such a variant if fragments of one of them at that reducing end pass the
        # annotation gate and, where the spectrum has a row with a prediction, outscore it by more than 2 (any lead gave rows of real glycans to compositions
        # with more chance hits, e.g. 840.94 in GPST000169 12_N); the variant then replaces that row
        known = defaultdict(dict)
        for g, conf in [p[0] for p in df_out['predictions'] if p]:
            known[stringify_dict(get_comp(g))][g] = max(conf, known[stringify_dict(get_comp(g))].get(g, 0))
        found = []
        for offset, name in re_names.items():
            cands = assign_candidate_structures(df_cond.copy(), df_use, common_structure_map, topo_struct_map, mass_tolerance, mode, (mass_tag or 0) + offset,
                                                modification = modification, max_charge = max_charge, sample_prep = sample_prep, cross_class = cross_class,
                                                ms1_ppm = 20 if accurate_ms1 else None, sialic_derivatization = sialic_derivatization)
            # with a sialic derivatization the candidates carry their sialic linkages (Neu5Ac_a2-6), the run's structures their plain composition
            cands['comp_str'] = [stringify_dict(plain_composition(c)) if isinstance(c, dict) else None for c in cands['composition']]
            cands = cands[cands['comp_str'].isin(list(known))].drop_duplicates(['spec_id', 'comp_str'])
            if len(cands):
                cands = cands.assign(reported = [list(known[c]) for c in cands['comp_str']], re_offset = offset, reducing_end = name).explode('reported').reset_index(drop = True)
                # scored with the sialic linkages the variant's precursor mass gives, reported as the run reports the structure
                cands['candidate_structure'] = cands['reported'] if sialic_derivatization is None else [
                    fit_sialic_linkages(g, c, scoring = True) for g, c in zip(cands['reported'], cands['composition'])]
                found.append(assign_annotation_scores_pooled(cands[cands['candidate_structure'].notna()].reset_index(drop = True), multiplier, mass_tag, mass_tolerance,
                                                             modification = modification, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization,
                                                             n_jobs = n_jobs, fragment_tolerance = fragment_tolerance, fragment_ppm = fragment_ppm,
                                                             deisotoped = condensed.setdefault('deisotoped', {})))
        df_var = pd.concat(found, ignore_index = True) if found else pd.DataFrame(columns = ['annotation_score'])
        df_var['conf'] = [known[c][g] for c, g in zip(df_var.get('comp_str', []), df_var.get('reported', []))]
        if len(df_var):
            sparse = (df_var['fragments'] <= 6) & (df_var['annotated'] >= np.maximum(2, 0.6 * df_var['fragments'])) & (df_var['conf'] >= 0.5)
            own = df_out[df_out['predictions'].str.len() > 0]
            own = own.groupby([np.round(own.index.values, 5), own['RT']])['annotation_score'].max()
            # A variant has to fit its precursor as a row of the run has to (the doomed rule below): with an ion-trap precursor, the unreduced form of a glycan with
            # two dHex in place of a Neu5Ac lies 0.5 m/z (z = 2) below the reduced sialylated glycan, so it matched that glycan's precursor 0.46 off, outscored it
            # by chance fragments and replaced it, and finalise_predictions then cut it by ppm (GPST000153 1038.87 [M-2H]2-, the run's main sialylated peak)
            theo = [mass_check(mz, g, modification = modification, mass_tag = (mass_tag or 0) + o, mode = mode, sample_prep = sample_prep,
                               mass_tolerance = mass_tolerance, sialic_derivatization = sialic_derivatization) if accurate_ms1 or abs(z) > 1 else None
                    for mz, g, z, o in zip(df_var['m/z'], df_var['candidate_structure'], df_var['charge'], df_var['re_offset'])]
            doomed = np.array([t is not None and (not t or abs(t[0] - mz) / t[0] * 1e6 >= ppm_thresh) for t, mz in zip(theo, df_var['m/z'])], dtype = bool)
            # A glycosylamine lies 2.016 Da below its PA glycan, so an ion trap's precursor window also fits the glycosylamine of a composition about 1 Da heavier
            # (two dHex for a Neu5Ac) at a real PA glycan: in GPST000237 the 8 spectra of Hex7HexNAc6Neu5Ac3 1108.41 [M+3H]3+ went to a fucosylated
            # glycosylamine. One has to outscore every candidate of its spectrum at the run's own reducing end, whether the model named one or not
            ga_ok = (df_var['reducing_end'].values != 'glycosylamine') | (df_var['annotation_score'].values > df_var['spec_id'].map(own_best).fillna(-np.inf).values + 2) if (
                own_best is not None) else True
            df_var = df_var[((df_var['annotation_score'] > crumbs_thresh) | sparse) & ~doomed & ga_ok & (df_var['annotation_score'] > np.array(
                [own.get(key, -np.inf) for key in zip(np.round(df_var['m/z'].values, 5), np.round(df_var['RT'].values, 2))]) + 2)]
        if len(df_var):
            df_var = df_var.sort_values(['annotation_score', 'conf'], ascending = False, kind = 'stable')
            # a spectrum's predictions are the structures of its best variant, by annotation score, then confidence
            preds = {spec: [(g, round(c, 4)) for g, c, o in zip(x['reported'], x['conf'], x['re_offset']) if o == x['re_offset'].iloc[0]][:5]
                     for spec, x in df_var.groupby('spec_id', sort = False)}
            df_var = df_var.drop_duplicates('spec_id')
            df_var = df_var.assign(predictions = [preds[k] for k in df_var['spec_id']], charge = df_var['charge'] * multiplier, RT = df_var['RT'].round(2),
                                   rel_abundance = df_var['intensity'], evidence = 'side product', top_fragments = [
                    [round(float(m), 4) for m, _ in sorted(p.items(), key = lambda x: x[1], reverse = True)[:frag_num]] for p in df_var['peak_d']]).set_index('m/z')
            if experimental:
                df_var = domain_filter(df_var, glycan_class, mode = mode, filter_out = filter_out, sample_prep = sample_prep, mass_tag = mass_tag,
                                       max_charge = max_charge, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use,
                                       sialic_derivatization = sialic_derivatization, trust_header_charge = accurate_ms1)
            df_var = df_var[df_var['predictions'].str.len() > 0][[c for c in df_out.columns if c in df_var.columns and c != 'rt_range'] + ['re_offset', 'reducing_end', 'rt_range']]
        if len(df_var):
            df_var = deduplicate_predictions(df_var, mz_diff = mass_tolerance, rt_diff = rt_diff).drop(columns = ['rt_range'])
            replaced = set(zip(np.round(df_var.index.values, 5), df_var['RT']))
            df_var.index = pd.Index(df_var.index.values - df_var['re_offset'].values / np.abs(df_var['charge'].values), name = df_out.index.name)
            df_out = pd.concat([df_out[[key not in replaced for key in zip(np.round(df_out.index.values, 5), df_out['RT'])]].assign(re_offset = 0.0), df_var])
        else:
            # A run without side products carries the columns too: wrap_inference_batch concatenates runs, which gave such a run NaN offsets (NaN m/z, and a ZeroDivisionError in the MS1 quantification)
            df_out = df_out.assign(re_offset = 0.0, reducing_end = None)
    if not _return_intermediate:
        # A wrong reducing end shows in the run itself: few model predictions among its most intense clusters
        strong = df_out[df_out['evidence'] == 'strong'] if 'evidence' in df_out.columns else df_out.iloc[:0]
        n_top = sum(np.any((np.abs(top_clusters['m/z'].values - mz) < mass_tolerance) & (np.abs(top_clusters['RT'].values - rt) < rt_diff))
                    for mz, rt in zip(strong.index, strong['RT']))
        best, df_end = check_reducing_end(spectra_filepath, glycan_class, run_args, loaded_spectra, n_top = n_top, n_chosen = len(strong), condensed = condensed)
        if best is not None:
            run_args.update(modification = best[0], mass_tag = best[1])
            modification, mass_tag, df_out, mass_dic = (None if best[0] == 'custom' else best[0]), best[1], df_end, None
            df_out.attrs.clear()
        # A wrong glycan_class shows in the run itself: about as many of its glycans get the top structure of another class (above extra_thresh, or the
        # likelier isomer) as of glycan_class. The file is then also processed as that class, which takes over if it finds at least 1.5x as many glycans of
        # its own class.
        # (rows of another reducing end repeat structures of the run's own rows, so they do not count)
        tops = [p[0][0] for p in (df_out.loc[df_out['re_offset'] == 0, 'predictions'] if 're_offset' in df_out.columns else df_out['predictions']) if p]
        # Another class only counts structures no class of the run can hold: free and lipid glycans share every reducing end, and O and free share Gal and GlcNAc
        label, own = '+'.join(classes), sum(in_classes(g, classes) for g in tops)
        counts = {c: sum(enforce_class(g, c) and not in_classes(g, classes) for g in tops) for c in ['N', 'O', 'free'] if c not in classes}
        other = max(counts, key = counts.get, default = None)
        if other is not None and counts[other] >= max(5, own):
            df_other = wrap_inference(spectra_filepath, other, **run_args, _return_intermediate = True, _loaded = loaded_spectra, _condensed = condensed)
            found = sum(bool(p) and enforce_class(p[0][0], other) for p in (
                df_other.loc[df_other['re_offset'] == 0, 'predictions'] if 're_offset' in df_other.columns else df_other['predictions']))
            print(f"WARNING: As {label} glycans, {os.path.basename(spectra_filepath)} gave {own} {label} structures and "
                  f"{counts[other]} {other} structures that no {label} glycan has; as {other} glycans, it gives {found} {other} structures. " + (
                      f"Overriding the glycan class to '{other}'." if found >= 1.5 * own else f"Keeping '{label}', but check the glycan class."))
            if found >= 1.5 * own:
                glycan_class, classes, df_out, df_use, mass_dic = other, [other], df_other, df_other.attrs['df_use'], None
            df_other.attrs.clear()
        # The clusters (with their binned spectra) are not needed any more, and MS1 quantification is next (an N or O run keeps them for its other-class pass)
        pair = {'N': 'O', 'O': 'N'}.get(classes[0]) if len(classes) == 1 else None
        if pair is None:
            condensed.clear()
    if _return_intermediate:
        df_out.attrs.update(ms1 = ms1, mode = mode, accurate_ms1 = accurate_ms1, df_use = df_use, accurate_ms2 = trap == 'orbitrap', ms1_ppm_floor = ms1_ppm_floor,
                            ion_trap = trap in ('linear', 'amazon'))
        return df_out
    if df_out.empty:
        df_out.insert(0, 'top1_pred', [])
        df_out['ppm_error'] = []
        return (df_out, []) if spectra else df_out
    df_out['top1_pred'] = [x[0][0] if x else np.nan for x in df_out.predictions]
    # Same augmentation and clean-up as every file in wrap_inference_batch
    if supplement or experimental:
        df_out = augment_predictions(df_out, pred_thresh, supplement, experimental, glycan_class, df_use, mode,
                                     modification, mass_tag, filter_out, mass_tolerance, mass_dic,
                                     sample_prep = sample_prep, max_charge = max_charge, glycans = glycans,
                                     sialic_derivatization = sialic_derivatization, trust_header_charge = accurate_ms1, accurate_ms2 = trap == 'orbitrap')
    df_out, spectra_out = finalise_predictions(df_out, get_missing, mode, modification, mass_tag,
                                               ppm_thresh, rt_diff, sample_prep = sample_prep,
                                               glycan_class = glycan_class, mass_tolerance = mass_tolerance, ms1 = ms1,
                                               accurate_ms1 = accurate_ms1, sialic_derivatization = sialic_derivatization, n_jobs = n_jobs,
                                               ms1_ppm_floor = ms1_ppm_floor, ion_trap = trap in ('linear', 'amazon'), xic_details = spectra)
    # The other of N and O (O-glycans in an N-glycan release, N-glycans in an O-glycan one): if >= 2 rows have a structure only that class has, the clusters
    # without a row are processed again as N+O glycans, with only the compositions of that class (so the model only sees inputs that can add a row) and without
    # the side-product search (none of its rows were side products), and its rows of such a structure join the run. Whole runs as N+O worsened the
    # own-class rows, and these rows added to every run lost on files without the other class
    if pair is not None:
        if sum(isinstance(g, str) and enforce_class(g, pair) and not enforce_class(g, classes[0]) for g in df_out['top1_pred']) >= 2:
            df_pair = wrap_inference(spectra_filepath, [classes[0], pair], **{**run_args, 'reducing_end_variants': {}}, _return_intermediate = True,
                                     _loaded = loaded_spectra, _condensed = condensed, _pair = (pair, list(zip(df_out.index.values, df_out['RT'].values))))
            # Its rows of a structure only the other class has, from >= 2 spectra, are augmented and finalised as any run's rows and join the run unless they are
            # another charge state of a row it reports (a confidence cut does not carry over from N+O runs: finalise sums the confidences of the rows it merges)
            only = lambda p: isinstance(p, list) and len(p) > 0 and enforce_class(p[0][0], pair) and not enforce_class(p[0][0], classes[0])
            use_pair = df_pair.attrs['df_use']
            df_pair = df_pair[[only(p) and n >= 2 for p, n in zip(df_pair['predictions'], df_pair['num_spectra'])]]
            df_pair.attrs.clear()
            if len(df_pair):
                df_pair = augment_predictions(df_pair.assign(top1_pred = [p[0][0] for p in df_pair['predictions']]), pred_thresh, supplement, experimental,
                                              [classes[0], pair], use_pair, mode, modification, mass_tag, filter_out, mass_tolerance, None, sample_prep = sample_prep,
                                              max_charge = max_charge, glycans = glycans, sialic_derivatization = sialic_derivatization,
                                              trust_header_charge = accurate_ms1, accurate_ms2 = trap == 'orbitrap')
                df_pair, spectra_pair = finalise_predictions(df_pair, False, mode, modification, mass_tag, ppm_thresh, rt_diff, sample_prep = sample_prep,
                                                             glycan_class = glycan_class, mass_tolerance = mass_tolerance, ms1 = ms1, accurate_ms1 = accurate_ms1,
                                                             sialic_derivatization = sialic_derivatization, n_jobs = n_jobs, ms1_ppm_floor = ms1_ppm_floor,
                                                             ion_trap = trap in ('linear', 'amazon'), xic_details = spectra)
                neutral = lambda mz, z: np.abs(z) * mz - (np.abs(z) - 1) * multiplier * PROTON_MASS
                own_n, own_rt = neutral(df_out.index.values, df_out['charge'].values), df_out['RT'].values
                keep = [only(p) and not ((np.abs(own_n - neutral(mz, z)) < mass_tolerance) & (np.abs(own_rt - rt) < rt_diff)).any()
                        for p, mz, z, rt in zip(df_pair['predictions'], df_pair.index.values, df_pair['charge'].values, df_pair['RT'].values)]
                df_pair, spectra_pair = df_pair[keep], [x for x, k in zip(spectra_pair, keep) if k]
                if len(df_pair):
                    print(f"{os.path.basename(spectra_filepath)} also holds {pair}-glycans: {len(df_pair)} {pair}-glycan peak(s) added to its {classes[0]}-glycans.")
                    # (the run's attrs, such as its MS1 calibration, stay those of its own rows)
                    attrs, df_pair.attrs = df_out.attrs, {}
                    df_out, spectra_out = pd.concat([df_out, df_pair]), spectra_out + spectra_pair
                    df_out.attrs = attrs
        condensed.clear()
    if 'rel_abundance' in df_out.columns:
        df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
    if plot_glycans:
        from glycowork.motif.draw import plot_glycans_excel
        plot_glycans_excel(df_out.drop(columns = ['ms3'], errors = 'ignore').reset_index(),
                           os.path.splitext(spectra_filepath)[0] + '_output.xlsx', glycan_col_num = 'top1_pred')
    # MS3 spectra are output like the MS2 spectra, only with spectra=True
    return (df_out, spectra_out) if spectra else df_out.drop(columns = ['ms3'], errors = 'ignore')


ACIDIC_RESIDUES = {'Neu5Ac', 'Neu5Gc', 'Kdn', 'S', 'P', 'HexA'}


def acidic_count(composition):
    # Charged residues of a composition (linkage-split keys of derivatized sialic acids are neutral and do not count)
    return sum(v for k, v in composition.items() if k in ACIDIC_RESIDUES) if isinstance(composition, dict) else 0


def wrap_inference_batch(spectra_filepath_list, glycan_class, intra_cat_thresh, top_n_isomers = 5, model = candycrunch,
                         glycans = glycans, bin_num = 2048, max_charge = -3,
                         frag_num = 100, modification = 'reduced', mass_tag = None, lc = 'PGC',
                         trap = 'linear', rt_min = 0, rt_max = 0, rt_diff = 1.0, rt_max_default = 30.0,
                         pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False,
                         extra_thresh = 0.2, crumbs_thresh = 3, ppm_thresh = 300,
                         filter_out = {'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True,
                         experimental = True, mass_dic = None, sample_prep = 'underivatized',
                         taxonomy_level = 'Class', taxonomy_filter = 'Mammalia', df_use = None, plot_glycans = False,
                         n_jobs = 1, sialic_derivatization = None, library_thresh = 0.9, reducing_end_variants = None):
    """wrapper function to get & curate CandyCrunch predictions, then harmonize them across multiple files\n
   | Arguments:
   | :-
   | spectra_filepath_list (list): list of absolute filepaths ending in ".raw" (Thermo), ".d" (Bruker timsTOF or ion trap folder), ".mzML", ".mzXML", ".mgf", ".cc", or ".xlsx" pointing to files containing spectra
   | glycan_class (string or list): glycan class as string, options are "O", "N", "lipid", "free", or a list of the classes measured together in one
   |                               run (e.g., ["N", "O"]): candidate compositions and database structures then come from all of them, each candidate goes to
   |                               the model as a glycan of its own class, and only structures outside all of them need extra_thresh.
   |                               An "N" or "O" file with at least 2 rows of a structure only the other of the two has also looks for glycans of
   |                               that class among the peaks it did not report, as in wrap_inference
   | intra_cat_thresh (float): minutes the RT of a structure can differ from the mean of a group, after correcting each file's consistent RT drift against the file with the most predicted peaks (over peaks both files share; with enough of them as a function of RT and of the number of sialic acids and other charged residues)
   | top_n_isomers (int): number of isomer groups at each composition to retain per file (a group stays if it is among the top_n_isomers of any file); default:5
   | model (PyTorch or callable): trained CandyCrunch model, or any other structure predictor as a callable (see wrap_inference); with n_jobs > 1 it has to be picklable (a module-level function or class instance)
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): spectra before this retention time (in minutes) are ignored; default:0 (runs longer than 20 min then skip their first 2 min)
   | rt_max (float): spectra after this retention time (in minutes) are ignored; default:0 (runs longer than 40 min then skip their last 10%)
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
   | temperature (float): the temperature factor used to calibrate logits; default:1.15
   | spectra (bool): whether to also output the actual spectra used for prediction; default:False
   | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
   | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); below it, they are only allowed if more likely than every in-class structure of that composition; default:0.2
   | crumbs_thresh (float): threshold for annotation score to keep predictions; 2 lower for files with orbitrap or timsTOF survey scans, whose accurate precursor mass already confirms the composition; default:3
   | ppm_thresh (float): ppm error threshold for filtering; default:300
   | filter_out (set): set of monosaccharide or modification types to filter out; default:{'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}
   | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
   | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
   | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
   | taxonomy_level (string): taxonomy level to filter by; default:'Class'
   | taxonomy_filter (string): which taxonomy to pull glycans for; default:'Mammalia'
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
   | plot_glycans (bool): whether to save a <file>_output.xlsx file per input file with SNFG images of all top1 predictions; default:False
   | n_jobs (int): number of files to process in parallel (separate processes), and of processes for the supporting peaks of each file's rows; default:1
   | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization of the samples, as in wrap_inference; default:None
   | library_thresh (float): MS2 cosine at which a peak without a structure takes the structure of a matching peak of another file (see below); None turns this off; default:0.9
   | reducing_end_variants (dict): other reducing ends that glycans of the samples may carry, as in wrap_inference; default:None\n
   | Returns:
   | :-
   | Returns a tuple of (feature table with one row per isomer group with a structure (also those without one if get_missing; a glycan that files report at different charge states is one row): top1_pred, consensus m/z, RT, charge, composition, GlyTouCan_ID, number of files with MS2 evidence, then per file its rel_abundance (num_spectra if no file has intensities) and evidence_<file>; dict of per-file dataframes, or of (dataframe, spectra) tuples if spectra=True, with the column xic, and ms3 for files with MS3 spectra, as in wrap_inference)
   """
    mode = "negative" if max_charge < 0 else "positive"
    mass_tolerance = ppm_thresh * MZ_REF / 1e6
    modification = None if modification == 'custom' else modification
    classes = glycan_classes(glycan_class)
    glycan_class = classes[0] if len(classes) == 1 else classes
    print(f"Your chosen settings are: {'+'.join(classes)} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type.isin(classes)])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
        if df_use.empty:
            raise ValueError(f"No {'+'.join(classes)} glycans of {taxonomy_level} '{taxonomy_filter}' in df_glycan; check taxonomy_level and taxonomy_filter")
    # Files with the same name in different folders still get distinct labels; a timsTOF .d folder may come with a trailing separator
    spectra_filepath_list = [os.path.normpath(fp) for fp in spectra_filepath_list]
    file_labels = [os.path.splitext(os.path.basename(fp))[0] for fp in spectra_filepath_list]
    file_labels = [label if file_labels.count(label) == 1 else f"{label}_{i}" for i, label in enumerate(file_labels)]
    # Core inference per file via wrap_inference intermediate return; MS1 data is parked on disk so only one file's MS1 is in memory at a time
    ms1_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
    inference_kwargs = dict(glycans = glycans, bin_num = bin_num, max_charge = max_charge,
                            frag_num = frag_num, modification = modification, mass_tag = mass_tag, lc = lc, trap = trap,
                            rt_min = rt_min, rt_max = rt_max, rt_diff = rt_diff, rt_max_default = rt_max_default,
                            pred_thresh = pred_thresh, temperature = temperature, extra_thresh = extra_thresh,
                            crumbs_thresh = crumbs_thresh, ppm_thresh = ppm_thresh, filter_out = filter_out,
                            experimental = experimental, sample_prep = sample_prep, taxonomy_level = taxonomy_level,
                            taxonomy_filter = taxonomy_filter, sialic_derivatization = sialic_derivatization,
                            reducing_end_variants = reducing_end_variants, _return_intermediate = True)
    # Worker processes load the default model themselves, so only a custom model has to be sent to them
    if model is not candycrunch:
        inference_kwargs['model'] = model
    # The reducing end check of wrap_inference, on the first file: all files are then processed with the reducing end it takes over. The file is read once,
    # with its MS1 data, which the check does not need; when files are processed one after the other, the first one's prediction then takes it as read
    first = load_spectra_filepath(spectra_filepath_list[0], extract_ms1 = spectra_filepath_list[0].lower().endswith(('.mzml', '.mzxml', '.raw', '.d', '.cc')))
    ms1 = first.attrs.pop('ms1', None)
    condensed = {}
    best, _ = check_reducing_end(spectra_filepath_list[0], glycan_class, {**inference_kwargs, 'model': model, 'df_use': df_use}, loaded = first, condensed = condensed)
    if ms1 is not None:
        first.attrs['ms1'] = ms1
    if best is not None:
        modification, mass_tag, mass_dic = (None if best[0] == 'custom' else best[0]), best[1], None
        inference_kwargs.update(modification = best[0], mass_tag = best[1])
    # ... and its clusters as the check condensed them, unless a free reducing end and survey scans let its run regroup them by their MS1 traces, which the
    # check did not read
    condensed = [condensed] if n_jobs <= 1 and (ms1 is None or not (modification is None and not mass_tag)) else []
    first, ms1 = ([first] if n_jobs <= 1 else []), None
    # The glycan class check of wrap_inference, over the predictions of all files: if they say that the files hold another class rather than
    # glycan_class, all files are also processed as that class, which takes over if it finds at least 1.5x as many glycans of its own class
    runs, key, label = {}, tuple(classes), '+'.join(classes)
    for c in [key, None]:
        use_c = df_use
        if c is None:
            tops = [p[0][0] for r in runs[key][0] for p in (r[0].loc[r[0]['re_offset'] == 0, 'predictions'] if 're_offset' in r[0].columns else r[0]['predictions']) if p]
            own = sum(in_classes(g, classes) for g in tops)
            counts = {c: sum(enforce_class(g, c) and not in_classes(g, classes) for g in tops) for c in ['N', 'O', 'free'] if c not in classes}
            c = max(counts, key = counts.get, default = None)
            if c is None or counts[c] < max(5, own):
                break
            use_c = copy.deepcopy(df_glycan[df_glycan.glycan_type == c])
            use_c = use_c[use_c[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
        tasks = [(fp, os.path.join(ms1_dir.name, f'{i}.npz'), {**inference_kwargs, 'glycan_class': glycan_class if c == key else c, 'df_use': use_c, **({'_loaded': first.pop()} if i == 0 and first else {}),
                                                                       **({'_condensed': condensed.pop()} if i == 0 and condensed else {})}) for i, fp in enumerate(spectra_filepath_list)]
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers = n_jobs, initializer = set_num_threads if torch is None else torch.set_num_threads,
                                     initargs = (max(1, (os.cpu_count() or 1) // n_jobs),)) as pool:
                runs[c] = (list(pool.map(_batch_inference_file, *zip(*tasks))), use_c)
        else:
            # each task is let go once done, so the first file's spectra and MS1 data are not kept while the others run
            runs[c] = ([_batch_inference_file(*tasks.pop(0)) for _ in range(len(tasks))], use_c)
    if len(runs) > 1:
        found = sum(bool(p) and enforce_class(p[0][0], c) for r in runs[c][0] for p in (
            r[0].loc[r[0]['re_offset'] == 0, 'predictions'] if 're_offset' in r[0].columns else r[0]['predictions']))
        print(f"WARNING: As {label} glycans, these files gave {own} {label} structures and {counts[c]} {c} structures that no "
              f"{label} glycan has; as {c} glycans, they give {found} {c} structures. " + (f"Overriding the glycan class to '{c}'." if found >= 1.5 * own else f"Keeping '{label}', but check the glycan class."))
        if found >= 1.5 * own:
            glycan_class, classes, key, mass_dic = c, [c], c, None
    results, df_use = runs[key]
    # Built once here instead of once per file inside augment_predictions
    if experimental and not mass_dic:
        mass_dic = make_mass_dic(glycans, glycan_class, filter_out, df_use, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization)
    inference_dfs, file_modes, ms1_paths, accurate_ms1s, accurate_ms2s, ms1_ppm_floors, ion_traps = {}, {}, {}, {}, {}, {}, {}
    for file_label, (df_out, file_mode, ms1_path, accurate_ms1, accurate_ms2, ms1_ppm_floor, ion_trap) in zip(file_labels, results):
        # wrap_inference overrides the ion mode if the file says otherwise, so downstream steps have to follow it
        file_modes[file_label], ms1_paths[file_label], accurate_ms1s[file_label], accurate_ms2s[file_label] = file_mode, ms1_path, accurate_ms1, accurate_ms2
        ms1_ppm_floors[file_label], ion_traps[file_label] = ms1_ppm_floor, ion_trap
        inference_dfs[file_label] = df_out.assign(condition_label = file_label) if not df_out.empty else df_out
    # The table of a file without any glycan peak
    empty_table = pd.DataFrame(columns = ['top1_pred', 'predictions', 'composition', 'num_spectra', 'charge', 'RT', 'rel_abundance'],
                               index = pd.Index([], name = 'm/z'))
    non_empty = {k: v for k, v in inference_dfs.items() if not v.empty}
    if not non_empty:
        ms1_dir.cleanup()
        return pd.DataFrame(), {k: (empty_table.copy(), []) if spectra else empty_table.copy() for k in file_labels}
    # A column without any value in one file (precursor_charge of a file without instrument charges) would set the dtype of the concatenation in future
    # pandas; it takes the dtype that the files with values give it, which is what pandas does so far
    dtypes = {c: pd.concat([v[c] for v in non_empty.values() if c in v.columns and v[c].notna().any()]).dtype for c in {c for v in non_empty.values() for c in v.columns}
              if any(c in v.columns and v[c].notna().any() for v in non_empty.values())}
    all_ms2 = pd.concat([v.astype({c: np.float64 if dtypes[c].kind in 'iu' else object if dtypes[c].kind == 'b' else dtypes[c] for c in v.columns if c in dtypes and v[c].isna().all()}) for v in non_empty.values()])

    def mass_labels(mzs):
        # Cluster precursor m/z across files: split at gaps above the mass tolerance, then split clusters wider than it at their largest gap
        mz_sorted = np.unique(mzs)
        to_split, mz_clusters = np.split(mz_sorted, np.where(np.diff(mz_sorted) > mass_tolerance)[0] + 1), []
        while to_split:
            cluster = to_split.pop()
            if cluster[-1] - cluster[0] <= mass_tolerance:
                mz_clusters.append(cluster)
            else:
                cut = np.argmax(np.diff(cluster)) + 1
                to_split.extend([cluster[:cut], cluster[cut:]])
        return {mz: round(float(np.median(cluster)), 4) for cluster in mz_clusters for mz in cluster}

    all_ms2['mass_label'] = all_ms2.index.map(mass_labels(all_ms2.index.values))
    # Retention time alignment (as LaCyTools and GlycReSoft align runs on analytes they share): a file's drift towards the file with the most predicted rows
    # comes from anchor peaks, a mass_label and first prediction found exactly once in both files. On PGC it can grow with retention time and with every
    # charged residue (sialic acids, sulfates, phosphates), so with at least 10 anchors it is a + b * RT + c * acidic residues (robust least squares; each
    # term only where the anchors span it), else the median difference over at least 3 anchors. Isomer groups, gap filling and feature merging compare RTs
    # shifted by it, while every row keeps its own RT
    rt_shift = {}
    keyed = all_ms2.assign(top = [p[0][0] if p else None for p in all_ms2['predictions']]).dropna(subset = ['top'])
    keyed = keyed[keyed.groupby(['condition_label', 'mass_label', 'top'])['RT'].transform('size') == 1].set_index(['mass_label', 'top'])
    if len(keyed):
        ref = keyed['condition_label'].value_counts().idxmax()
        ref_rts = keyed.loc[keyed['condition_label'] == ref, 'RT']
        for file_label, file_rows in keyed[keyed['condition_label'] != ref].groupby('condition_label'):
            diffs = (ref_rts - file_rows['RT']).dropna()
            if len(diffs) < 3:
                continue
            rts = file_rows.loc[diffs.index, 'RT'].values.astype(float)
            n_acid = np.array([acidic_count(c) for c in file_rows.loc[diffs.index, 'composition']], dtype = float)
            use_rt = bool(len(diffs) >= 10 and np.percentile(rts, 90) - np.percentile(rts, 10) >= 5)
            use_acid = bool(len(diffs) >= 10 and (n_acid == 0).sum() >= 3 and (n_acid > 0).sum() >= 3)
            y = diffs.values.astype(float)
            # Only a consistent drift is corrected: anchors of replicate runs scatter by at most ~0.5 min around their median difference, those of runs from
            # different LC set-ups by ~2 min (median absolute deviation)
            if np.median(np.abs(y - np.median(y))) > 1:
                continue
            beta, resid = np.array([np.median(y)]), y - np.median(y)
            if use_rt or use_acid:
                # Iteratively reweighted least squares from the median, Huber then Tukey's biweight, so a mismatched anchor (often at the end of the
                # gradient, where it pulls hardest) ends up with no weight; the model replaces the median only if its anchors scatter less around it
                X, r = np.column_stack([np.ones(len(y))] + [rts] * use_rt + [n_acid] * use_acid), resid
                for it in range(30):
                    u = np.abs(r) / (1.4826 * np.median(np.abs(r)) + 0.01)
                    w = np.sqrt(np.clip(1.345 / np.maximum(u, 1e-9), 0, 1) if it < 20 else np.clip(1 - (u / 4.685) ** 2, 0, None) ** 2)
                    fit = np.linalg.lstsq(X * w[:, None], y * w, rcond = None)[0]
                    r = y - X @ fit
                if np.median(np.abs(r)) < np.median(np.abs(resid)):
                    beta = fit
                else:
                    use_rt = use_acid = False
            coefs = iter(beta[1:])
            rt_shift[file_label] = (float(beta[0]), float(next(coefs)) if use_rt else 0.0, float(next(coefs)) if use_acid else 0.0, rts.min(), rts.max(), n_acid.max())
        if rt_shift:
            mid = np.median(keyed['RT'])
            print(f"Retention times aligned to {ref}: " + ', '.join(f"{k} {a + b * mid:+.2f} min" + (f" ({b:+.3f}/min, {c:+.2f} per acidic residue)" if b or c else '')
                                                                     for k, (a, b, c, *_) in rt_shift.items()))

    def drift(rts, labels, comps):
        # Each row's drift towards the reference at its own RT and number of acidic residues (within the ranges of its file's anchors; 0 for the reference)
        rts, labels, comps, out = np.asarray(rts, dtype = float), np.asarray(labels), np.asarray(comps, dtype = object), np.zeros(len(rts))
        for k, (a, b, c, lo, hi, n_max) in rt_shift.items():
            m = labels == k
            if m.any():
                out[m] = a + b * np.clip(rts[m], lo, hi) + c * np.minimum([acidic_count(x) for x in comps[m]], n_max)
        return out

    def file_rt(rts, label, comps):
        # A file's own RT of a reference-aligned RT (the fixed point of t = T - drift(t))
        rts = np.asarray(rts, dtype = float)
        t = rts.copy()
        for _ in range(10):
            t = rts - drift(t, [label] * len(t), comps)
        return t

    # Cross-file harmonization: resolve variant predictions, link DDA gaps
    assigned_cats = assign_categories(all_ms2.assign(RT = all_ms2['RT'] + drift(all_ms2['RT'], all_ms2['condition_label'], all_ms2['composition']), RT_own = all_ms2['RT']),
                                      intra_cat_thresh = intra_cat_thresh)
    assigned_cats['RT'] = assigned_cats.pop('RT_own')
    smoothed_category_predictions = assign_modal_category_prediction(assigned_cats)
    # Cross-run library: a row still without a structure takes the one of the most similar row of another file with a structure of glycan_class, at the
    # same m/z (within mass_tolerance) and charge, if their pooled MS2 spectra (square-root intensities in 1 Da bins, as in condense_dataframe) have a
    # cosine of at least library_thresh. It also takes that row's composition: its own is often an isobaric alternative (Hex4HexNAc4S1 for Man7)
    if library_thresh is not None:
        cats = smoothed_category_predictions
        vecs = np.zeros((len(cats), 3001), dtype = np.float32)
        for k, peaks in enumerate(cats['peak_d']):
            for m, x in peaks.items() if isinstance(peaks, dict) else []:
                vecs[k, int(round(min(max(m, 0), 3000)))] += x
        vecs = np.sqrt(vecs)
        vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True) + 1e-12
        named = np.array([bool(p) and isinstance(t, str) for p, t in zip(cats['predictions'], cats['top1_pred'])], dtype = bool)
        in_library = named & np.array([isinstance(t, str) and in_classes(t, classes) for t in cats['top1_pred']], dtype = bool)
        mzs, charges, files = cats.index.values.astype(float), cats['charge'].abs().values, cats['condition_label'].values
        top1, predictions, compositions = list(cats['top1_pred']), list(cats['predictions']), list(cats['composition'])
        for k in np.flatnonzero(~named & np.array([isinstance(p, dict) for p in cats['peak_d']], dtype = bool)):
            hits = np.flatnonzero(in_library & (np.abs(mzs - mzs[k]) <= mass_tolerance) & (charges == charges[k]) & (files != files[k]))
            if len(hits) and (vecs[hits] @ vecs[k]).max() >= library_thresh:
                j = hits[np.argmax(vecs[hits] @ vecs[k])]
                top1[k], predictions[k], compositions[k] = top1[j], [cats['predictions'].iloc[j][0]], compositions[j]
        cats['top1_pred'], cats['predictions'], cats['composition'] = top1, predictions, compositions
    prevailing_category_predictions = filter_top_n_isomers(smoothed_category_predictions, top_n = top_n_isomers)
    # Per-file augment, finalize, and quantify
    harmonized_labels = set(prevailing_category_predictions.condition_label.unique())
    for file_label in file_labels:
        file_mode = file_modes[file_label]
        df_out, spectra_out = empty_table.copy(), []
        if file_label in harmonized_labels:
            df_out = prevailing_category_predictions[prevailing_category_predictions['condition_label'] == file_label].sort_index()
            if supplement or experimental:
                df_out = augment_predictions(df_out, pred_thresh, supplement, experimental, glycan_class, df_use,
                                             file_mode, modification, mass_tag, filter_out,
                                             mass_tolerance, mass_dic, sample_prep = sample_prep,
                                             max_charge = abs(max_charge) * (-1 if file_mode == 'negative' else 1),
                                             sialic_derivatization = sialic_derivatization, trust_header_charge = accurate_ms1s[file_label],
                                             accurate_ms2 = accurate_ms2s[file_label])
            if len(df_out) > 0:
                ms1 = None
                if ms1_paths[file_label]:
                    with np.load(ms1_paths[file_label]) as ms1_file:
                        ms1 = (ms1_file['rts'], ms1_file['mzs'], ms1_file['ints'], ms1_file['offsets'])
                df_out, spectra_out = finalise_predictions(df_out, get_missing, file_mode, modification,
                                                           mass_tag, ppm_thresh, rt_diff, sample_prep = sample_prep,
                                                           glycan_class = glycan_class, mass_tolerance = mass_tolerance,
                                                           ms1 = ms1, accurate_ms1 = accurate_ms1s[file_label],
                                                           sialic_derivatization = sialic_derivatization, n_jobs = n_jobs,
                                                           ms1_ppm_floor = ms1_ppm_floors[file_label], ion_trap = ion_traps[file_label], xic_details = spectra)
        inference_dfs[file_label] = (df_out, spectra_out)
    # The other of N and O (O-glycans in N-glycan releases, N-glycans in O-glycan ones), as in wrap_inference: a file with >= 2 rows of a structure only that
    # class has is processed again as N+O glycans on its clusters without a row, and its rows of such a structure from >= 2 spectra are harmonized across these
    # files like the own rows (aligned RTs, isomer groups, modal structure, top_n_isomers), augmented and finalised per file, and join their file unless they
    # are another charge state of one of its rows; gap filling and the feature table then treat them like any other row. Running the pass on every file once
    # one file has the other class lost on the files without it (GlycoGauntlet N and O batches -0.005 each)
    pair = {'N': 'O', 'O': 'N'}.get(classes[0]) if len(classes) == 1 else None
    only = lambda p: isinstance(p, list) and len(p) > 0 and enforce_class(p[0][0], pair) and not enforce_class(p[0][0], classes[0])
    triggered = [k for k in file_labels if pair is not None and sum(isinstance(g, str) and enforce_class(g, pair) and not enforce_class(g, classes[0])
                                                                     for g in inference_dfs[k][0]['top1_pred']) >= 2]
    pair_rows = []
    if triggered:
        use_pair = copy.deepcopy(df_glycan[df_glycan.glycan_type.isin([classes[0], pair])])
        use_pair = use_pair[use_pair[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
        tasks = [(spectra_filepath_list[file_labels.index(k)], None, {**inference_kwargs, 'glycan_class': [classes[0], pair], 'df_use': use_pair, 'reducing_end_variants': {},
                                                                      '_pair': (pair, list(zip(inference_dfs[k][0].index.values, inference_dfs[k][0]['RT'].values)))}) for k in triggered]
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers = n_jobs, initializer = set_num_threads if torch is None else torch.set_num_threads,
                                     initargs = (max(1, (os.cpu_count() or 1) // n_jobs),)) as pool:
                pair_runs = list(pool.map(_batch_inference_file, *zip(*tasks)))
        else:
            pair_runs = [_batch_inference_file(*task) for task in tasks]
        pair_rows = [r[0][[only(p) and n >= 2 for p, n in zip(r[0]['predictions'], r[0]['num_spectra'])]].assign(condition_label = k) for k, r in zip(triggered, pair_runs)]
        pair_rows = [r for r in pair_rows if len(r)]
    if pair_rows:
        pair_rows = pd.concat(pair_rows)
        pair_rows['mass_label'] = pair_rows.index.map(mass_labels(pair_rows.index.values))
        pair_cats = assign_categories(pair_rows.assign(RT = pair_rows['RT'] + drift(pair_rows['RT'], pair_rows['condition_label'], pair_rows['composition']), RT_own = pair_rows['RT']),
                                      intra_cat_thresh = intra_cat_thresh)
        pair_cats['RT'] = pair_cats.pop('RT_own')
        # (their isomer groups are numbered after the own ones, so no group mixes both)
        pair_cats['category_label'] += int(assigned_cats['category_label'].max()) + 1
        pair_cats = filter_top_n_isomers(assign_modal_category_prediction(pair_cats), top_n = top_n_isomers)
        neutral = lambda mz, z, file_mode: np.abs(z) * mz - (np.abs(z) - 1) * (-1 if file_mode == 'negative' else 1) * PROTON_MASS
        for file_label, df_pair in pair_cats.groupby('condition_label', sort = False):
            file_mode = file_modes[file_label]
            df_pair = df_pair.sort_index()
            if supplement or experimental:
                df_pair = augment_predictions(df_pair, pred_thresh, supplement, experimental, [classes[0], pair], use_pair, file_mode, modification, mass_tag,
                                              filter_out, mass_tolerance, None, sample_prep = sample_prep, max_charge = abs(max_charge) * (-1 if file_mode == 'negative' else 1),
                                              glycans = glycans, sialic_derivatization = sialic_derivatization, trust_header_charge = accurate_ms1s[file_label],
                                              accurate_ms2 = accurate_ms2s[file_label])
            if not len(df_pair):
                continue
            ms1 = None
            if ms1_paths[file_label]:
                with np.load(ms1_paths[file_label]) as ms1_file:
                    ms1 = (ms1_file['rts'], ms1_file['mzs'], ms1_file['ints'], ms1_file['offsets'])
            df_pair, spectra_pair = finalise_predictions(df_pair, False, file_mode, modification, mass_tag, ppm_thresh, rt_diff, sample_prep = sample_prep,
                                                         glycan_class = glycan_class, mass_tolerance = mass_tolerance, ms1 = ms1, accurate_ms1 = accurate_ms1s[file_label],
                                                         sialic_derivatization = sialic_derivatization, n_jobs = n_jobs, ms1_ppm_floor = ms1_ppm_floors[file_label],
                                                         ion_trap = ion_traps[file_label], xic_details = spectra)
            df_out, spectra_out = inference_dfs[file_label]
            own_n, own_rt = neutral(df_out.index.values, df_out['charge'].values, file_mode), df_out['RT'].values
            keep = [only(p) and not ((np.abs(own_n - neutral(mz, z, file_mode)) < mass_tolerance) & (np.abs(own_rt - rt) < rt_diff)).any()
                    for p, mz, z, rt in zip(df_pair['predictions'], df_pair.index.values, df_pair['charge'].values, df_pair['RT'].values)]
            df_pair, spectra_pair = df_pair[keep], [x for x, k in zip(spectra_pair, keep) if k]
            if len(df_pair):
                print(f"{file_label} also holds {pair}-glycans: {len(df_pair)} {pair}-glycan peak(s) added to its {classes[0]}-glycans.")
                # (the file's attrs, such as its MS1 calibration, stay those of its own rows)
                attrs, df_pair.attrs = df_out.attrs, {}
                df_out = pd.concat([df_out, df_pair])
                df_out.attrs = attrs
                inference_dfs[file_label] = (df_out, spectra_out + spectra_pair)
    # Consensus of each isomer group that survived clean-up in at least one file, for MS1 gap filling in the others
    finalized = [v[0] for v in inference_dfs.values() if not v[0].empty]
    # Rows with a top1_pred first, so every value of a group comes from one row (groupby.first skips NaN column by column)
    master = pd.concat(finalized).reset_index().pipe(lambda x: x.assign(RT = x['RT'] + drift(x['RT'], x['condition_label'], x['composition']))).sort_values(
        'top1_pred', key = lambda s: s.isna(), kind = 'stable').groupby(
        ['mass_label', 'category_label']).agg(
        repr_mz = ('m/z', 'median'),
        repr_rt = ('RT', 'median'),
        repr_charge = ('charge', 'first'),
        repr_composition = ('composition', 'first'),
        repr_predictions = ('predictions', 'first'),
        repr_top1 = ('top1_pred', 'first'),
        present_in = ('condition_label', set)
    ).reset_index() if finalized else pd.DataFrame(columns = ['present_in'])
    for file_label, spectra_filepath in zip(file_labels, spectra_filepath_list):
        df_out, spectra_out = inference_dfs[file_label]
        file_mode = file_modes[file_label]
        # Cross-file MS1 gap filling: propagate isomer groups to files without MS2 for them when MS1 shows their peak
        missing = master[~master['present_in'].apply(lambda s: file_label in s)]
        if ms1_paths[file_label] and not missing.empty:
            with np.load(ms1_paths[file_label]) as ms1_file:
                ms1 = (ms1_file['rts'], ms1_file['mzs'], ms1_file['ints'], ms1_file['offsets'])
            # Look for the apex within intra_cat_thresh of the group's consensus RT, on the monoisotopic trace around the consensus m/z
            _, apex_rts, apex_ints, is_peak = extract_xic_areas(ms1, missing['repr_mz'].values, file_rt(missing['repr_rt'].values, file_label, missing['repr_composition'].values),
                                                                mz_tolerance = mass_tolerance, search_window = intra_cat_thresh,
                                                                isotopes = 0, mz_calibration = (0, mass_tolerance))
            # Then quantify that peak exactly like the MS2 rows of this file, so both share one abundance scale
            theo_mzs = [(mass_check(mz, top1, mode = file_mode, modification = modification, mass_tag = mass_tag,
                                    sample_prep = sample_prep, mass_tolerance = mass_tolerance, sialic_derivatization = sialic_derivatization) or [mz])[0] if isinstance(top1, str) else mz
                        for mz, top1 in zip(missing['repr_mz'], missing['repr_top1'])]
            # with the other charge states the neutral mass allows, as finalise_predictions quantifies the MS2 rows
            z_max = int(max(np.abs(missing['repr_charge']).max(), np.abs(df_out['charge']).max() if len(df_out) else 1))
            other_charges = [[z for z in range(1, z_max + 1) if z != abs(int(z0)) and (mz - (-1 if file_mode == 'negative' else 1) * PROTON_MASS) * abs(z0) >= {1: 0, 2: 900, 3: 1500}.get(z, 3500)]
                             for mz, z0 in zip(theo_mzs, missing['repr_charge'])]
            details = [] if spectra else None
            areas = extract_xic_areas(ms1, theo_mzs, np.where(np.isnan(apex_rts), file_rt(missing['repr_rt'].values, file_label, missing['repr_composition'].values), apex_rts),
                                      charges = missing['repr_charge'].values, mz_tolerance = mass_tolerance,
                                      mz_calibration = df_out.attrs.get('ms1_calibration'), sample_prep = sample_prep,
                                      glycans = missing['repr_top1'].tolist(), other_charges = other_charges, mode = file_mode, details = details)
            # Only fill with a genuine peak at least as intense as the weakest precursor that got MS2 in this file,
            # and never with the peak of a feature that already has MS2 in this file
            own = all_ms2[all_ms2['condition_label'] == file_label]
            own_apex = extract_xic_areas(ms1, own.index.values, own['RT'].values, mz_tolerance = mass_tolerance,
                                         search_window = rt_diff, isotopes = 0,
                                         mz_calibration = (0, mass_tolerance))[2] if len(own) else np.zeros(0)
            floor = own_apex[own_apex > 0].min() if (own_apex > 0).any() else 0
            own_rts = own.groupby('mass_label')['RT'].apply(np.array).to_dict()
            # A glycan this file reports at another charge state already holds the area of this one (finalise_predictions adds co-eluting charge states)
            comp_rts = defaultdict(list)
            for c, rt in zip(df_out['composition'], df_out['RT']):
                if isinstance(c, dict):
                    comp_rts[stringify_dict(c)].append(rt)
            # A gap peak k/z above a co-eluting precursor of this file at least half as intense (k = 1-3, same charge) is that precursor's 13C isotope peak:
            # replacing one Neu5Ac by two dHex adds 1.02 Da, so many compositions sit there. Of the gap rows of one composition (at any charge) on one peak
            # only the most intense stays (ion traps scatter a precursor over several m/z clusters, and each of them would fill that peak again)
            own_z = np.abs(own['charge'].values.astype(float)) if len(own) else np.zeros(0)
            keep, gap_rts = np.zeros(len(missing), dtype = bool), defaultdict(list)
            for i in np.argsort(-np.nan_to_num(np.asarray(areas, dtype = float)), kind = 'stable'):
                row, apex_rt = missing.iloc[i], apex_rts[i]
                if not is_peak[i] or apex_ints[i] < floor or (np.abs(own_rts.get(row['mass_label'], np.zeros(0)) - apex_rt) <= rt_diff).any():
                    continue
                z = abs(int(row['repr_charge'])) or 1
                near = (own_z == z) & (np.abs(own['RT'].values - apex_rt) <= rt_diff) & (own_apex >= 0.5 * apex_ints[i]) if len(own) else np.zeros(0, dtype = bool)
                if any((np.abs(row['repr_mz'] - own.index.values[near] - k * 1.00336 / z) <= mass_tolerance / 2).any() for k in (1, 2, 3)):
                    continue
                if isinstance(row['repr_composition'], dict):
                    ck = stringify_dict(row['repr_composition'])
                    if (np.abs(np.array(comp_rts.get(ck, []) + gap_rts[ck]) - apex_rt) <= rt_diff).any():
                        continue
                    gap_rts[ck].append(apex_rt)
                keep[i] = True
            gaps = [{
                'm/z': row['repr_mz'],
                'top1_pred': row['repr_top1'],
                'predictions': row['repr_predictions'],
                'composition': row['repr_composition'],
                'num_spectra': 0,
                'charge': row['repr_charge'],
                'RT': round(apex_rt, 2),
                'rel_abundance': area,
                'evidence': 'ms1_only',
                'condition_label': file_label,
                'mass_label': row['mass_label'],
                'category_label': row['category_label'],
                'GlyTouCan_ID': glytoucan_mapping.get(row['repr_top1'], ''),
                'ppm_error': np.nan,
                'notes': 'MS1 signal only; propagated from other files', **({'xic': [record]} if spectra else {})}
                for (_, row), area, apex_rt, record, k in
                zip(missing.iterrows(), areas, apex_rts, details or [None] * len(missing), keep) if k]
            if gaps:
                gap_df = pd.DataFrame(gaps).set_index('m/z')
                df_out = pd.concat([df_out, gap_df]) if not df_out.empty else gap_df
                spectra_out = spectra_out + [None] * len(gaps)
        if 'rel_abundance' in df_out.columns and df_out['rel_abundance'].sum() > 0:
            df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
        if plot_glycans and not df_out.empty:
            from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df_out.drop(columns = ['ms3'], errors = 'ignore').reset_index(),
                               os.path.splitext(spectra_filepath)[0] + '_output.xlsx',
                               glycan_col_num = 'top1_pred')
        # MS3 spectra are output like the MS2 spectra, only with spectra=True
        inference_dfs[file_label] = (df_out, spectra_out) if spectra else df_out.drop(columns = ['ms3'], errors = 'ignore')
    # Removed once the MS1 data is not needed any more, as an implicit clean-up raises a ResourceWarning
    ms1_dir.cleanup()
    all_outputs = [d for d in (v[0] if spectra else v for v in inference_dfs.values()) if not d.empty]
    group_keys = ['mass_label', 'category_label']
    all_outputs = pd.concat(all_outputs).reset_index().sort_values('top1_pred', key = lambda s: s.isna(), kind = 'stable') if all_outputs else pd.DataFrame()
    # Isomer groups without a structure in any file (kept in the per-file tables, as wrap_inference keeps them) are features only with get_missing
    if not get_missing and not all_outputs.empty:
        all_outputs = all_outputs[all_outputs.groupby(group_keys)['top1_pred'].transform('count') > 0]
    if all_outputs.empty:
        return pd.DataFrame(), inference_dfs
    # Feature table: one row per isomer group with its consensus annotation, then abundance and evidence per file
    # Without any intensities (e.g., only .mgf files), spectrum counts stand in for abundances
    value_col = 'rel_abundance' if 'rel_abundance' in all_outputs.columns else 'num_spectra'
    # One glycan stays one feature when files report it at different charge states or in neighbouring m/z clusters: groups with the same structure, RTs
    # within intra_cat_thresh and no file in common join the group seen in the most files, which supplies m/z, RT and charge
    groups = all_outputs.assign(RT = all_outputs['RT'] + drift(all_outputs['RT'], all_outputs['condition_label'], all_outputs['composition'])).groupby(group_keys).agg(
        top1 = ('top1_pred', 'first'), rt = ('RT', 'median'), files = ('condition_label', set)).reset_index()
    groups = groups.assign(n = groups['files'].map(len)).sort_values('n', ascending = False, kind = 'stable')
    target = {}
    for _, grp in groups[groups['top1'].notna()].groupby('top1'):
        kept = []
        for key, rt, files in zip(zip(grp.mass_label, grp.category_label), grp.rt, grp.files):
            hit = next((k for k in kept if abs(k[1] - rt) <= intra_cat_thresh and not (k[2] & files)), None)
            if hit:
                hit[2].update(files)
                target[key] = hit[0]
            else:
                kept.append([key, rt, set(files)])
    rep = np.array([k not in target for k in zip(all_outputs.mass_label, all_outputs.category_label)], dtype = bool)
    merged = [target.get(k, k) for k in zip(all_outputs.mass_label, all_outputs.category_label)]
    all_outputs = all_outputs.assign(mass_label = [k[0] for k in merged], category_label = [k[1] for k in merged])
    combined_batch = all_outputs[rep].groupby(group_keys).agg(top1_pred = ('top1_pred', 'first'), mz = ('m/z', 'median'),
                                                         RT = ('RT', 'median'), charge = ('charge', 'first'),
                                                         composition = ('composition', 'first'),
                                                         GlyTouCan_ID = ('GlyTouCan_ID', 'first'))
    combined_batch['n_files_ms2'] = all_outputs[all_outputs['evidence'] != 'ms1_only'].groupby(group_keys)[
        'condition_label'].nunique().reindex(combined_batch.index, fill_value = 0)
    if 'reducing_end' in all_outputs.columns:
        # an ethyl glycoside (or other reducing-end variant) is a feature of its own, at its own m/z
        combined_batch['reducing_end'] = all_outputs[rep].groupby(group_keys)['reducing_end'].first().reindex(combined_batch.index)
    abundances = all_outputs.pivot_table(index = group_keys, columns = 'condition_label', values = value_col,
                                         aggfunc = 'sum')
    evidence = all_outputs.pivot_table(index = group_keys, columns = 'condition_label', values = 'evidence',
                                       aggfunc = 'first')
    # Every input file keeps its columns, even without any feature, so they can serve as groups downstream
    combined_batch = combined_batch.join(abundances.reindex(columns = file_labels)).join(
        evidence.reindex(columns = file_labels).add_prefix('evidence_'))
    combined_batch = combined_batch.rename(columns = {'mz': 'm/z'})
    combined_batch = combined_batch.iloc[np.argsort(mz_order(combined_batch['m/z'], combined_batch['RT'], combined_batch['charge'].values,
                                                             combined_batch['composition'].tolist()))].reset_index(drop = True)
    return combined_batch, inference_dfs


def _batch_inference_file(spectra_filepath, ms1_path, inference_kwargs):
    """runs the per-file part of wrap_inference_batch and parks the file's MS1 data on disk; module-level so that it can run in a process pool\n
   | Returns:
   | :-
   | Returns a tuple of (intermediate prediction dataframe, ion mode used, path to the MS1 .npz or None, whether the survey scans are accurate (orbitrap,
   | TOF), whether the MS2 spectra are orbitrap spectra, the survey scans' ppm floor of MS1_PPM_FLOORS, whether the MS2 spectra are ion-trap spectra)
   """
    df_out = wrap_inference(spectra_filepath, **inference_kwargs)
    ms1, file_mode, accurate_ms1 = df_out.attrs.pop('ms1', None), df_out.attrs.pop('mode'), df_out.attrs.pop('accurate_ms1', False)
    accurate_ms2, ms1_ppm_floor, ion_trap = df_out.attrs.pop('accurate_ms2', False), df_out.attrs.pop('ms1_ppm_floor', 3.0), df_out.attrs.pop('ion_trap', False)
    df_out.attrs.clear()
    if ms1 is None or len(ms1[0]) == 0 or ms1_path is None:
        return df_out, file_mode, None, accurate_ms1, accurate_ms2, ms1_ppm_floor, ion_trap
    np.savez(ms1_path, rts = ms1[0], mzs = ms1[1], ints = ms1[2], offsets = ms1[3])
    return df_out, file_mode, ms1_path, accurate_ms1, accurate_ms2, ms1_ppm_floor, ion_trap


def filter_top_n_isomers(df_in, top_n = 5):
    df_out = df_in.copy(deep = True)
    # Rank the isomer groups (categories) at each mass within each file: carrying a prediction first, then by the number of files they were seen in, then by
    # abundance in that file; a group stays if it is among the top_n of any file. One cap over all files dropped real isomers once more files brought more
    # groups, and compared intensities across files. Groups without a prediction stay, so finalise_predictions sees the rows it sees in wrap_inference
    keys = ['mass_label', 'category_label']
    cats = df_out.groupby(keys).agg(has_pred = ('top1_pred', lambda x: x.notna().any()), file_presences = ('condition_label', 'nunique')).reset_index()
    cats = df_out.groupby(keys + ['condition_label'])['rel_abundance'].sum().reset_index().merge(cats, on = keys)
    cats = cats.sort_values(['has_pred', 'file_presences', 'rel_abundance'], ascending = False, kind = 'stable').groupby(['mass_label', 'condition_label']).head(top_n)
    permitted = dict(zip(zip(cats.mass_label, cats.category_label), cats.file_presences))
    df_out['file_presences'] = [permitted.get(k, 0) for k in zip(df_out.mass_label, df_out.category_label)]
    return df_out[df_out['file_presences'] > 0]


def assign_modal_category_prediction(assigned_cats):
    # Each isomer group takes the structure most of its rows rank first, ties going to the highest probability summed over all rows' candidates
    # (a 1:1 tie between two files went to the later file); every row keeps its own candidates and its own probability of that structure
    keys = list(zip(assigned_cats.mass_label, assigned_cats.category_label))
    votes, probs = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(list))
    for key, preds in zip(keys, assigned_cats['predictions']):
        if preds:
            votes[key][preds[0][0]] += 1
        for p in preds:
            if len(p) > 1:
                probs[key][p[0]].append(p[1])
    modal = {key: max(v, key = lambda g: (v[g], sum(probs[key][g]))) for key, v in votes.items()}
    top1, predictions = [], []
    for key, preds, comp in zip(keys, assigned_cats['predictions'], assigned_cats['composition']):
        g = modal.get(key)
        # A row without a prediction (e.g., one that failed the class or confidence filters in its file) takes the group's structure if it has its composition
        if g is None or not (preds or (isinstance(comp, dict) and get_comp(g) == plain_composition(comp))):
            top1.append(None)
            predictions.append(preds)
            continue
        own = next((p[1] for p in preds if len(p) > 1 and p[0] == g), None)
        top1.append(g)
        predictions.append([(g, own if own is not None else round(float(np.mean(probs[key][g] or [0])), 4))] + [p for p in preds if p[0] != g])
    assigned_cats['top1_pred'], assigned_cats['predictions'] = top1, predictions
    return assigned_cats


def assign_categories(all_ms2_spectra, intra_cat_thresh = 3):
    all_mass_dfs = []
    condition_labels = all_ms2_spectra.condition_label.unique()
    # Splits the rows by mass and file once instead of masking all rows for every mass and file
    for _, mass_df in all_ms2_spectra.groupby('mass_label', sort = False):
        condition_dfs = dict(list(mass_df.groupby('condition_label', sort = False)))
        mass_group_dfs = [condition_dfs.get(c, mass_df.iloc[:0]).assign(RT_group = lambda x: range(len(x))) for c in condition_labels]
        cats_mass_dfs = mass_dfs_to_categories(mass_group_dfs, intra_cat_thresh)
        all_mass_dfs.append(cats_mass_dfs)
    return pd.concat([p for q in all_mass_dfs for p in q])


def mass_dfs_to_categories(mass_range_dfs, inter_sample_thresh):
    RT_groups = create_RT_groups(mass_range_dfs)
    categories = {0: []}
    for x in RT_groups[0]:
        add_new_category(categories, x)
    categories = expand_RT_categories(RT_groups, categories, inter_sample_thresh)
    sample_cats = RT_cats_to_sample_cats(categories, RT_groups)
    cat_dfs = sample_categories_to_df(sample_cats, mass_range_dfs)
    return cat_dfs


def create_RT_groups(mass_range_dfs):
    # Every row is its own RT group (RT_group numbers the rows of each file)
    return [[[rt] for rt in sample_df['RT']] for sample_df in mass_range_dfs]


def add_new_category(categories, cluster):
    new_id = max(categories)
    categories[new_id + 1] = [cluster]
    return categories


def expand_RT_categories(all_sample_RT_groups, categories, inter_sample_thresh):
    for sample in all_sample_RT_groups[1:]:
        all_candidate_categories = calculate_candidate_clusters(sample, categories, inter_sample_thresh)
        orphan_idxs = []
        for idx, (orphan_cluster, empty_candidates) in enumerate(zip(sample, all_candidate_categories)):
            if len(empty_candidates) == 0:
                add_new_category(categories, orphan_cluster)
                orphan_idxs.append(idx)
        sample = [x for u, x in enumerate(sample) if u not in orphan_idxs]
        all_candidate_categories = [x for u, x in enumerate(all_candidate_categories) if u not in orphan_idxs]
        assigned = settle_category_conflict(sample, all_candidate_categories, categories)
        while single := [u for u, x in enumerate(all_candidate_categories) if len(x) == 1 and u not in assigned]:
            cat = next(iter(all_candidate_categories[single[0]]))
            categories[cat].append(sample[single[0]])
            assigned.add(single[0])
            all_candidate_categories = [x - {cat} for x in all_candidate_categories]
        for u, x in enumerate(all_candidate_categories):
            if not x and u not in assigned:
                add_new_category(categories, sample[u])
                assigned.add(u)
        sample = [x for u, x in enumerate(sample) if u not in assigned]
        all_candidate_categories = [x for u, x in enumerate(all_candidate_categories) if u not in assigned]
        if [x for x in all_candidate_categories if x]:
            optim_cats = find_closest_categories_largest(sample, all_candidate_categories, categories)
            for cluster, optim_cat in zip(sample, optim_cats):
                if optim_cat:
                    categories[optim_cat].append(cluster)
                else:
                    add_new_category(categories, cluster)
    return categories


def calculate_candidate_clusters(sample, categories, inter_sample_thresh):
    all_candidate_categories = []
    for cluster in sample:
        candidate_categories = set()
        for category, cat_pop in categories.items():
            if cat_pop:
                if abs(np.mean(cluster) - np.mean([p for q in cat_pop for p in q])) < inter_sample_thresh:
                    candidate_categories.add(category)
        all_candidate_categories.append(candidate_categories)
    return all_candidate_categories


def settle_category_conflict(sample, cand_categories, categories):
    settled = set()
    for cat in set().union(*cand_categories):
        conflicting = [i for i, x in enumerate(cand_categories) if x == {cat}]
        if len(conflicting) > 1:
            closest_cluster_idx = conflicting[np.argmin(
                [abs(np.mean([p for q in categories[cat] for p in q]) - np.mean(sample[i])) for i in conflicting])]
            categories[cat].append(sample[closest_cluster_idx])
            for other_idx in [i for i in conflicting if i != closest_cluster_idx]:
                categories = add_new_category(categories, sample[other_idx])
            settled.update(conflicting)
            for x in cand_categories:
                x.discard(cat)
    return settled


def find_closest_categories_largest(sample, sample_candidates, categories):
    cat_means = [{cat: np.mean([p for q in categories[cat] for p in q]) for cat in sorted(x)} for x in sample_candidates]
    cluster_means = [x[0] for x in sample]
    cat_mean_diffs = []
    for cluster_mean, cat_mean in zip(cluster_means, cat_means):
        cat_mean_diffs.append({k: abs(v - cluster_mean) for k, v in cat_mean.items()})
    cats_out = [set() for _ in cluster_means]
    selected_RTs = []
    sorted_categories = dict(sorted(categories.items(), key = lambda x: len(x[1]), reverse = True))
    for cat in sorted_categories:
        closest_RTs = sorted(
            [(diffs[cat], sample_RT) for diffs, sample_RT in zip(cat_mean_diffs, cluster_means) if cat in diffs if
             sample_RT not in selected_RTs])
        if not closest_RTs:
            continue
        selected_RT = closest_RTs[0][1]
        selected_RTs.append(selected_RT)
        cats_out[cluster_means.index(selected_RT)] = cat
    return cats_out


def RT_cats_to_sample_cats(RT_categories, RT_groups):
    sample_group_categories = {x: [] for x in RT_categories}
    for k, v in RT_categories.items():
        for cluster in v:
            for i, x in enumerate(RT_groups):
                sample_group_categories[k].extend((i, j) for j, c in enumerate(x) if c is cluster)
    return sample_group_categories


def sample_categories_to_df(sample_group_categories, mass_range_df):
    category_dfs = []
    for cat, group in sample_group_categories.items():
        for x in group:
            cat_df = mass_range_df[x[0]][mass_range_df[x[0]]['RT_group'] == x[1]]
            cat_df = cat_df.assign(category_label = cat)
            category_dfs.append(cat_df)
    return category_dfs


def supplement_prediction(df_in, glycan_class, mode = 'negative', modification = 'reduced',
                          sample_prep = 'underivatized',
                          mass_tag = None):
    """searches for biosynthetic precursors of CandyCrunch predictions that could explain peaks\n
   | Arguments:
   | :-
   | df_in (pandas dataframe): output file produced by wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP or 'custom'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None\n
   | Returns:
   | :-
   | Returns dataframe with supplemented predictions based on biosynthetic network
   """
    df = copy.deepcopy(df_in)
    preds = [k[0][0] for k in df['predictions'] if k]
    permitted_roots = {
        'free': {"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"},
        'lipid': {"Glc", "Gal"},
        'O': {'GalNAc', 'Fuc', 'Man'},
        'N': {'GlcNAc(b1-4)GlcNAc'}
    }.get(glycan_class, {})
    if glycan_class == 'free':
        preds = [f"{k}-ol" for k in preds]
    # Positions on the m/z index raise a KeyError (caught in augment_predictions) whenever a row lacks a prediction, which keeps this step inactive as
    # measured; so it fails before building the network, which took 2-4 s per file for nothing
    unexplained_idx = [idx for idx, pred in enumerate(df['predictions']) if not pred]
    unexplained = df.index[unexplained_idx].tolist()
    charges = df.charge[unexplained_idx].tolist()
    net = construct_network(preds, permitted_roots = permitted_roots)
    if glycan_class == 'free':
        net = evoprune_network(net)
    preds_set = set(preds)
    new_nodes = [k for k in net.nodes() if k not in preds_set]
    explained_idx = [[unexplained_idx[k] for k, check in enumerate([mass_check(j, node,
                                                                               modification = modification, mode = mode,
                                                                               mass_tag = mass_tag,
                                                                               sample_prep = sample_prep,
                                                                               permitted_charges = [abs(c)]) for j, c in
                                                                    zip(unexplained, charges)]) if check] for node in
                     new_nodes]
    new_nodes = [(node, idx) for node, idx in zip(new_nodes, explained_idx) if idx]
    explained = {k: [] for k in set(unwrap(explained_idx))}
    for node, indices in new_nodes:
        for index in indices:
            explained[index].append(node)
    pred_idx = df.columns.get_loc('predictions')
    for index, values in explained.items():
        df.iat[index, pred_idx] = [(t, 0) for t in values[:5]]
    return df

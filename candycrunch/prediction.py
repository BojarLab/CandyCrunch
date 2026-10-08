import ast
import copy
import os
import re
import pickle
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict
import numpy as np
import opentfraw
import pandas as pd
import torch
import torch.nn.functional as F
from glycowork.glycan_data.loader import df_glycan, stringify_dict, unwrap
from glycowork.motif.graph import subgraph_isomorphism, glycan_to_nxGraph, compare_glycans, \
    graph_to_string
from glycowork.motif.processing import enforce_class
from glycowork.motif.annotate import get_molecular_properties
from glycowork.motif.tokenization import (composition_to_mass, get_ion_mzs,
                                          glycan_to_composition, PROTON_MASS,
                                          glycan_to_mass, modification_formula_dict, calculate_adduct_mass,
                                          mz_to_composition, structure_to_basic, mass_dict)
from glycowork.network.biosynthesis import construct_network, evoprune_network
from candycrunch.model import (CandyCrunch_CNN, SimpleDataset, transform_mz, transform_rt)
from candycrunch.analysis import CandyCrumbs, PEPTIDE_ION_TYPES, supporting_ions
from candycrunch.utils import read_mzml, read_mzxml, read_mgf

this_dir, this_filename = os.path.split(__file__)
data_path = os.path.join(this_dir, 'glycans.pkl')
glycans = pickle.load(open(data_path, 'rb'))
data_path = os.path.join(this_dir, 'glytoucan_mapping.pkl')
glytoucan_mapping = pickle.load(open(data_path, 'rb'))
# Choose the correct computing architecture
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
sdict = os.path.join(this_dir, 'candycrunch.pt')
sdict = torch.load(sdict, map_location = device, weights_only = True)
sdict = {k.replace('module.', ''): v for k, v in sdict.items()}
candycrunch = CandyCrunch_CNN(2048, num_classes = len(glycans), input_precursor_dim = 12).to(device)
candycrunch.load_state_dict(sdict)
candycrunch = candycrunch.eval()
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

NEGATIVE_ADDUCTS = ['Acetate', 'Formate', 'HCO3-']
POSITIVE_ADDUCTS = ['Na+', 'K+', 'NH4+']
temperature = torch.Tensor([1.15]).to(device)
comp_vector_order = ['dHex', 'Hex', 'HexA', 'HexN', 'HexNAc', 'Kdn', 'Me', 'Neu5Ac', 'Neu5Gc', 'P', 'Pen', 'S']
MZ_REF = 1600  # reference m/z used internally to turn the user's single ppm tolerance into the flat-Da window for binning/fragments; corresponds to ~0.5 Da at 300ppm
ISOTOPE_SPACING = 1.003355  # 13C - 12C
# Label mass a reducing-end modification adds (H2 for reduction, label minus O for reductive amination)
modification_mass_dict = {k: calculate_adduct_mass(v) for k, v in modification_formula_dict.items()}
# Natural abundances of each element's isotopes at +0, +1, +2, ... Da, for the share of a glycan's molecules within the integrated isotope peaks
ISOTOPE_ABUNDANCES = {'C': [0.9893, 0.0107], 'H': [0.999885, 0.000115], 'N': [0.99636, 0.00364], 'O': [0.99757, 0.00038, 0.00205],
                      'S': [0.9499, 0.0075, 0.0425, 0, 0.0001], 'P': [1.0]}


def get_adduct_list(mode):
    return NEGATIVE_ADDUCTS if mode == 'negative' else POSITIVE_ADDUCTS


def refine_precursor_mz(ms1, mzs, rts, scans, refine, mz_tolerance = 0.3, isotope_tolerance = 0.25, min_explained = 0.7):
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
   | min_explained (float): share of a peak's intensity that the isotope envelope of a lighter peak has to explain for the peak to count as its isotope; default:0.7\n
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
        if not len(scan_mzs) or np.abs(scan_mzs - mz).min() > mz_tolerance:
            continue
        k = np.argmin(np.abs(scan_mzs - mz))
        mono = k
        # DDA often triggers on a heavier isotope once the monoisotopic peak is excluded, so step down to the lightest peak (z = 1, then
        # z = 2 spacing) whose isotope envelope explains this peak, unless that peak itself is explained as an isotope of an even lighter one
        for z in (1, 2):
            chain = [k]
            for j in range(1, 4):
                near = np.where(np.abs(scan_mzs - (scan_mzs[k] - j * ISOTOPE_SPACING / z)) <= isotope_tolerance)[0]
                if not len(near):
                    break
                chain.append(near[np.argmax(scan_ints[near])])
            for j in range(len(chain) - 1, 0, -1):
                below = np.where(np.abs(scan_mzs - (scan_mzs[chain[j]] - ISOTOPE_SPACING / z)) <= isotope_tolerance)[0]
                below = below[np.argmax(scan_ints[below])] if len(below) else None
                if (scan_ints[chain[j]] * envelope(scan_mzs[chain[j]] * z)[j] >= min_explained * scan_ints[k] and
                        (below is None or scan_ints[below] * envelope(scan_mzs[below] * z)[1] < min_explained * scan_ints[chain[j]])):
                    mono = chain[j]
                    break
            if mono != k:
                break
        walked[i] = mono != k
        # Ion-trap centroids jitter between scans, so average the monoisotopic centroid over the neighbouring survey scans
        pos, wts = [], []
        for t in range(max(s - 1, 0), min(s + 2, len(ms1_rts))):
            t_mzs, t_ints = ms1_mzs[ms1_offsets[t]:ms1_offsets[t + 1]], ms1_ints[ms1_offsets[t]:ms1_offsets[t + 1]]
            if len(t_mzs) and np.abs(t_mzs - scan_mzs[mono]).min() <= 0.2:
                pos.append(t_mzs[np.argmin(np.abs(t_mzs - scan_mzs[mono]))])
                wts.append(t_ints[np.argmin(np.abs(t_mzs - scan_mzs[mono]))])
        out[i] = float(np.average(pos, weights = wts))
    # An isotope-triggered MS2 repeats its monoisotopic precursor's fragmentation with partly 13C-shifted fragments; if that precursor was
    # fragmented itself within a minute, the isotope spectrum only blurs the cluster (and can become its apex), so it is dropped
    out_arr, walked, rts = np.array(out), np.array(walked), np.asarray(rts, dtype = float)
    drop = np.array([w and bool(np.any(~walked & (np.abs(out_arr - m) < mz_tolerance) & (np.abs(rts - rt) <= 1.0))) for m, rt, w in
                     zip(out_arr, rts, walked)], dtype = bool)
    return out, drop


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
   | tuples per MS2 spectrum)
   """
    highest_i_dict = {}
    rts, intensities, mzs, charges, scans, activations = [], [], [], [], [], []
    detected_mode, detected_trap = None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine = [], [], [], [], []
    ms3s, row_of = [], {}
    for spectrum in read_mzml(filepath, centroid_levels = (ms_level, ms_level + 1)):
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
            peaks_raw = spectrum['peaks']
            if len(peaks_raw) > 0:
                ms1_rts.append(spectrum['rt'])
                # float32 halves the memory of MS1 data and still resolves ~0.2 mDa at m/z 2000
                ms1_mzs.append(peaks_raw[:, 0].astype(np.float32))
                ms1_ints.append(peaks_raw[:, 1].astype(np.float32))
        if spectrum['ms_level'] == ms_level:
            if detected_mode is None or detected_trap is None:
                ns_uri = '{http://psi.hupo.org/ms/mzml}'
                for cv in spectrum['element'].iter(f'{ns_uri}cvParam'):
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
                    elif acc in ('MS:1000484', 'MS:1000079'):
                        # orbitrap / FT-ICR analyzer terms; vendor-neutral fallback when no Thermo filter string is present
                        detected_trap = 'orbitrap'
            peaks = spectrum['peaks']
            mz_i_dict = dict(peaks[peaks[:, 1].argsort()][-num_peaks:])
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
                # Without a charge state the instrument never determined the monoisotopic peak, so its trigger m/z gets refined from MS1
                ms1_scans.append(len(ms1_rts) - 1)
                refine.append(raw_charge is None)
                # Vendor software can default to charge=1 when undetermined;
                # only trust explicit multiply-charged assignments
                if raw_charge is not None and abs(int(raw_charge)) == 1:
                    raw_charge = None
                charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
                if intensity:
                    inty = spectrum['precursor'].get('i', np.nan)
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
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, refine)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
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
    detected_mode, detected_trap = None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine = [], [], [], [], []
    ms3s, row_of = [], {}
    for spectrum in read_mzxml(filepath):
        if spectrum['msLevel'] == ms_level + 1 and spectrum.get('precursorMz') and len(spectrum['m/z array']) and mzs:
            # As in process_mzML_stack; mzXML names the parent scan as precursorScanNum (nested scans are yielded one by one)
            prec = next((p for p in spectrum['precursorMz'] if int(p.get('precursorScanNum', -1)) in row_of), spectrum['precursorMz'][0])
            row = row_of.get(int(prec.get('precursorScanNum', -1)), len(mzs) - 1 if 'precursorScanNum' not in prec else None)
            if row is not None:
                top_idx = np.argsort(spectrum['intensity array'])[::-1][:num_peaks]
                ms3s[row].append((float(prec['precursorMz']), {float(m): float(i) for m, i in zip(spectrum['m/z array'][top_idx],
                                                                                                spectrum['intensity array'][top_idx])}))
        if spectrum['msLevel'] == 1 and len(spectrum['m/z array']):
            order = np.argsort(spectrum['m/z array'], kind = 'stable')
            ms1_rts.append(float(spectrum['retentionTime']))
            ms1_mzs.append(spectrum['m/z array'][order].astype(np.float32))
            ms1_ints.append(spectrum['intensity array'][order].astype(np.float32))
        if spectrum['msLevel'] == ms_level:
            # mzXML scans carry their polarity and, if converted from Thermo files, the filter line naming the analyzer
            if detected_mode is None and spectrum.get('polarity') in ('+', '-'):
                detected_mode = 'negative' if spectrum['polarity'] == '-' else 'positive'
            if detected_trap is None and str(spectrum.get('filterLine', '')).startswith(('ITMS', 'FTMS')):
                detected_trap = 'linear' if spectrum['filterLine'].startswith('ITMS') else 'orbitrap'
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            num_peaks_to_extract = min(num_peaks, len(mz_array))
            top_idx = np.argsort(intensity_array)[::-1][:num_peaks_to_extract]
            mz_i_dict = {mz: i for mz, i in zip(mz_array[top_idx], intensity_array[top_idx])}
            if mz_i_dict:
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
                ms1_scans.append(len(ms1_rts) - 1)
                refine.append(raw_charge is None)
                if raw_charge is not None and abs(int(raw_charge)) == 1:
                    raw_charge = None
                charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
                if intensity:
                    inty = spectrum['precursorMz'][0].get('precursorIntensity', np.nan)
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
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, refine)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
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
    raw = opentfraw.RawFile(filepath)
    table = raw.scan_table()
    # opentfraw 2.0.0 misreads the scan events of version 66 files from ion traps (LTQ, LTQ XL, LTQ Orbitrap Velos with Xcalibur 4): they are
    # self-describing (136-byte preamble, then counted 56-byte reactions, 16-byte scan windows and 8-byte coefficients, then 3 u32), not of the
    # fixed size it assumes. Its peaks, profile intensities, retention times and trailers are right, so if this layout walks exactly over the event
    # stream, the columns read below are rebuilt from the events and the trailer
    index_addr, data_addr = 0, 0
    with open(filepath, 'rb') as f:
        f.seek(36)
        if int.from_bytes(f.read(4), 'little') >= 66:
            # Skips the sequence row (64 bytes, 31 strings with a u32 after the 16th) and the autosampler info (24 bytes, a string), each string
            # being a u32 count of UTF-16 characters followed by them
            f.seek(1420)
            for size in [None] * 16 + [4] + [None] * 15 + [24, None]:
                f.seek(size or 2 * int.from_bytes(f.read(4), 'little'), 1)
            info = f.read(824)
            addrs = f.read(16 * max(int.from_bytes(info[28:32], 'little'), 2))
            # The MS controller's run header has scan events or the file's data address
            for a in range(0, len(addrs), 16):
                f.seek(int.from_bytes(addrs[a:a + 8], 'little'))
                run_header = f.read(7464)
                n_events = int.from_bytes(run_header[7376:7380], 'little')
                if int.from_bytes(addrs[a:a + 8], 'little') and (n_events or run_header[7416:7424] == info[808:816]):
                    break
            trailer_addr, params_addr = int.from_bytes(run_header[7448:7456], 'little'), int.from_bytes(run_header[7456:7464], 'little')
            f.seek(trailer_addr + 4)
            stream = f.read(max(params_addr - trailer_addr - 4, 0))
            starts, pos = [], 0
            while pos + 148 <= len(stream) and len(starts) < n_events:
                starts.append(pos)
                pos += 140 + 56 * int.from_bytes(stream[pos + 136:pos + 140], 'little')
                pos += 4 + 16 * int.from_bytes(stream[pos:pos + 4], 'little')
                pos += 16 + 8 * int.from_bytes(stream[pos:pos + 4], 'little')
            if pos == len(stream) and len(starts) == len(table['ms_level']) and len(set(np.diff(starts + [pos]))) > 1:
                index_addr, data_addr = int.from_bytes(run_header[7408:7416], 'little'), int.from_bytes(info[808:816], 'little')
                for i, s in enumerate(starts):
                    level = max(stream[s + 6], 1)
                    reactions = [np.frombuffer(stream, '<f8', 3, s + 140 + 56 * k) for k in range(int.from_bytes(stream[s + 136:s + 140], 'little'))]
                    act = {1: 'hcd', 3: 'etd', 4: 'cid', 5: 'ecd'}.get(stream[s + 24], '')
                    params = (raw.scan_parameters(raw.first_scan + i) or {}) if level > 1 else {}
                    mono, charge = params.get('Monoisotopic M/Z:') or 0, int(params.get('Charge State:') or 0)
                    table['ms_level'][i] = level
                    table['polarity'][i] = {0: '-', 1: '+'}.get(stream[s + 4])
                    table['scan_mode'][i] = {0: 'centroid', 1: 'profile'}.get(stream[s + 5])
                    table['analyzer'][i] = {0: 'ITMS', 1: 'TQMS', 2: 'SQMS', 3: 'TOFMS', 4: 'FTMS', 5: 'Sector'}.get(stream[s + 40])
                    table['filter_string'][i] = ' '.join(f'{mz:.4f}@{act}{energy:.2f}' for mz, width, energy in reactions if mz > 0)
                    table['precursor_mz'][i] = mono if mono > 0 else float(reactions[-1][0]) if reactions else None
                    table['charge'][i] = charge if charge > 0 else None
                    table['isolation_width'][i] = params.get('MS2 Isolation Width:') or params.get('MSn Isolation Width:')
    peak_ds, rts, intensities, mzs, charges, scans, activations = [], [], [], [], [], [], []
    detected_mode, detected_trap = None, None
    ms1_rts, ms1_mzs, ms1_ints, ms1_scans, refine, ms1_index = [], [], [], [], [], {}
    ms3s, isolations = [], []
    for i, scan in enumerate(range(raw.first_scan, raw.last_scan + 1)):
        if table['ms_level'][i] not in (1, ms_level, ms_level + 1):
            continue
        peak_mzs, peak_ints = raw.peaks(scan)
        # Ion-trap profile scans come without centroids. Like Thermo's centroider, which made the centroids of mzML files from such runs, profile
        # points below 1 are dropped, the profile is split at the lowest point between maxima and neighboring centroids closer than 0.5 m/z are
        # merged, each carrying the summed signal of its share of the profile at its intensity-weighted m/z
        if not len(peak_mzs) and table['scan_mode'][i] == 'profile' and table['analyzer'][i] != 'FTMS':
            prof_mzs, prof_ints = raw.profile(scan)
            if index_addr:
                # opentfraw 2.0.0 also converts these profiles with coefficients of the misread events, but ion-trap profile bins are m/z already,
                # so each chunk's m/z are rebuilt from the scan's packet: 40-byte header, 8 bytes per further segment, then the profile
                with open(filepath, 'rb') as f:
                    f.seek(index_addr + 88 * i + 72)
                    f.seek(data_addr + int.from_bytes(f.read(8), 'little'))
                    head = f.read(40)
                    f.seek(8 * max(int.from_bytes(head[:4], 'little') - 1, 0), 1)
                    prof = f.read(4 * int.from_bytes(head[4:8], 'little'))
                (first_value, step), fudge, pos, prof_mzs = np.frombuffer(prof, '<f8', 2), head[12] > 0, 24, [np.zeros(0)]
                for _ in range(int.from_bytes(prof[16:20], 'little')):
                    first_bin, n = np.frombuffer(prof, '<u4', 2, pos)
                    fudge_mz = np.frombuffer(prof, '<f4', 1, pos + 8)[0] if fudge else 0
                    prof_mzs.append(first_value + np.arange(first_bin, first_bin + n) * step + fudge_mz)
                    pos += 8 + 4 * fudge + 4 * int(n)
                prof_mzs = np.concatenate(prof_mzs)
            prof_ints = np.where(prof_ints < 1, 0, prof_ints)
            apex = np.where((prof_ints[1:-1] > prof_ints[:-2]) & (prof_ints[1:-1] >= prof_ints[2:]) & (prof_ints[1:-1] > 0))[0] + 1
            edges = np.array([0] + [a + 1 + np.argmin(prof_ints[a + 1:b]) for a, b in zip(apex[:-1], apex[1:])], dtype = np.int64)
            while len(apex):
                peak_ints = np.add.reduceat(prof_ints, edges)
                peak_mzs = np.add.reduceat(prof_ints * prof_mzs, edges) / peak_ints
                gaps = np.diff(peak_mzs)
                if not len(gaps) or gaps.min() >= 0.5:
                    break
                # Merges every pair closer than 0.5 m/z whose gap is the smallest among its neighbouring gaps, until none is left
                left, right = np.concatenate([[np.inf], gaps[:-1]]), np.concatenate([gaps[1:], [np.inf]])
                edges = np.delete(edges, np.where((gaps < 0.5) & (gaps <= left) & (gaps < right))[0] + 1)
            if len(apex):
                # Thermo's centroids of up to 8 profile points sit one bin above their weighted mean on this axis, those of wider peaks on it
                peak_mzs = peak_mzs + (prof_mzs[1] - prof_mzs[0]) * (
                            np.add.reduceat((prof_ints > 0).astype(int), edges) < 9)
        if not len(peak_mzs):
            continue
        if table['ms_level'][i] == 1:
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
        charges.append(raw_charge if raw_charge is not None and raw_charge > 1 else None)
        if intensity:
            # ThermoRawFileParser's precursor intensity: summed centroids within 1.5 m/z of the isolation m/z in the trailer's master scan (tribrids
            # acquire MS2 in parallel with the next survey scan), else in the preceding survey scan; LTQ trailers only carry a 'Master Index:', no scan
            s = ms1_index.get((raw.scan_parameters(scan) or {}).get('Master Scan Number:'), len(ms1_rts) - 1)
            intensities.append(
                float(ms1_ints[s][(ms1_mzs[s] >= iso - 1.5) & (ms1_mzs[s] < iso + 1.5)].sum()) if s >= 0 else np.nan)
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
    df_out['m/z'], drop = refine_precursor_mz(ms1, mzs, rts, ms1_scans, refine)
    df_out = df_out[~drop].reset_index(drop = True)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    if extract_ms1:
        df_out.attrs['ms1'] = ms1
    return df_out


def average_dicts(dicts, mode = 'mean', round_dp = False):
    """averages a list of dictionaries containing spectra\n
   | Arguments:
   | :-
   | dicts (list): list of dictionaries of form (fragment) m/z : intensity
   | mode (string): whether to average by mean or by max\n
   | Returns:
   | :-
   | Returns a single dictionary of form (fragment) m/z : intensity
   """
    result = defaultdict(list)
    for d in dicts:
        for mass, intensity in d.items():
            if round_dp:
                key_mass = np.round(mass, round_dp)
                result[key_mass].append(intensity)
            else:
                result[mass].append(intensity)
    return {mass: np.mean(intensities) if mode == 'mean' else max(intensities) for mass, intensities in result.items()}


def bin_intensities(peak_d, frames):
    """sums up intensities for each bin across a spectrum\n
   | Arguments:
   | :-
   | peak_d (dict): dictionary of form (fragment) m/z : intensity
   | frames (list): m/z boundaries separating each bin\n
   | Returns:
   | :-
   | (1) a list of binned intensities
   | (2) a list of the difference (bin edge - m/z of highest peak in bin) for each bin
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


def ms1_mz_calibration(ms1, target_mzs, rt_centers, charges, mz_tolerance = 0.5):
    """estimates the MS1 m/z offset and spread from the most intense centroid near each singly charged precursor\n
    | Arguments:
    | :-
    | ms1 (tuple): flat MS1 data from process_mzML_stack, (rts sorted ascending, mzs, intensities, scan offsets)
    | target_mzs (array-like): (theoretical) precursor m/z values
    | rt_centers (array-like): retention time of each precursor
    | charges (array-like): charge of each precursor; only singly charged ones are used, as low-resolution MS1 merges the isotopes of the others
    | mz_tolerance (float): m/z tolerance within which to look for the centroid; default:0.5\n
    | Returns:
    | :-
    | Returns a tuple of (median m/z offset, half-width covering the m/z spread), or (0, mz_tolerance) with fewer than 5 usable precursors
    """
    ms1_rts, ms1_mzs, ms1_ints, ms1_offsets = ms1
    target_mzs, charges = np.asarray(target_mzs, dtype = np.float64), np.abs(np.asarray(charges, dtype = np.float64))
    nearest = np.clip(np.searchsorted(ms1_rts, np.asarray(rt_centers, dtype = np.float64)), 0, len(ms1_rts) - 1)
    offsets = []
    for t in np.where(charges <= 1)[0]:
        s0, s1 = ms1_offsets[nearest[t]], ms1_offsets[nearest[t] + 1]
        lo, hi = np.searchsorted(ms1_mzs[s0:s1], target_mzs[t] - mz_tolerance), np.searchsorted(ms1_mzs[s0:s1], target_mzs[t] + mz_tolerance)
        if hi > lo:
            offsets.append(ms1_mzs[s0 + lo + np.argmax(ms1_ints[s0 + lo:s0 + hi])] - target_mzs[t])
    if len(offsets) < 5:
        return 0.0, mz_tolerance
    shift = float(np.median(offsets))
    return shift, min(mz_tolerance, 4 * 1.4826 * float(np.median(np.abs(np.array(offsets) - shift))))


def extract_xic_areas(ms1, target_mzs, rt_centers, charges = None, rt_window = 1.0, mz_tolerance = 0.5, search_window = 0,
                      isotopes = 3, weights = None, mz_calibration = None, glycans = None, sample_prep = 'underivatized'):
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
    | glycans (list): if given, each area is divided by the share of the glycan's molecules within the summed isotope peaks (computed from its elemental formula; None entries are interpolated by mass); default:None
    | sample_prep (string): underivatized/permethylated/peracetylated, for the elemental formula; default:'underivatized'\n
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
    for t in range(len(target_mzs)):
        s0, s1 = scan_starts[t], scan_ends[t]
        if s1 - s0 < 3:
            continue
        rts = ms1_rts[s0:s1]
        centers = target_mzs[t] + shift + spacing[t] * np.arange(isotopes + 1)
        ints_arr = np.empty(s1 - s0)
        for j, si in enumerate(range(s0, s1)):
            mzs = ms1_mzs[ms1_offsets[si]:ms1_offsets[si + 1]]
            cum = np.concatenate([[0], np.cumsum(ms1_ints[ms1_offsets[si]:ms1_offsets[si + 1]], dtype = np.float64)])
            ints_arr[j] = np.sum(cum[np.searchsorted(mzs, centers + widths[t])] - cum[np.searchsorted(mzs, centers - widths[t])])
        # Smoothing (sigma of 0.05 min, at least 0.7 scans) only steers apex and boundary finding; the area comes from the raw trace
        sigma = max(0.7, 0.05 / np.median(np.diff(rts)))
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
    if weights is not None:
        weights = np.nan_to_num(np.asarray(weights, dtype = np.float64))
        shared = [np.where((apex_scans == apex_scans[t]) & (charges == charges[t]) & (np.abs(target_mzs - target_mzs[t]) <= widths[t]))[0] if apex_scans[t] >= 0 else [t] for t in range(len(target_mzs))]
        areas = np.array([areas[t] * (weights[t] / weights[same].sum() if weights[same].sum() > 0 else 1 / len(same)) for t, same in enumerate(shared)])
    if glycans is not None:
        # Divide by the share of each glycan's molecules within the summed isotope peaks, from its elemental formula plus any derivatization groups; glycans without a formula are interpolated by mass
        props = get_molecular_properties([g for g in set(glycans) if isinstance(g, str)]) if any(isinstance(g, str) for g in glycans) else pd.DataFrame()
        formulas = props['molecular_formula'].to_dict() if 'molecular_formula' in props.columns else {}
        unit = {'permethylated': 'CH2', 'peracetylated': 'C2H2O'}.get(sample_prep)
        share = np.full(len(target_mzs), np.nan)
        for t, g in enumerate(glycans):
            if g not in formulas:
                continue
            counts = defaultdict(int, {el: int(k or 1) for el, k in re.findall(r'([A-Z][a-z]?)(\d*)', formulas[g])})
            if unit:
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
    return (areas, apex_rts, apex_ints, is_peak) if search_window else areas


def process_for_inference(df, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear', rt_max_default = 30.0, tta_thresh = None):
    """processes averaged spectra for them being inputs to CandyCrunch\n
   | Arguments:
   | :-
   | df (dataframe): condensed dataframe from condense_dataframe
   | glycan_class (int): 0 = O-linked, 1 = N-linked, 2 = lipid/free
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | tta_thresh (float): inputs of spectra whose best annotation_score exceeds this get 5 augmented copies, the others one plain copy (column 'tta'); default: None (all get 5)\n
   | Returns:
   | :-
   | (1) a dataloader used for model prediction (5-copy inputs first, then 1-copy inputs, each in input_id order)
   | (2) a preliminary df_out dataframe
   """
    df = df.assign(glycan_type = glycan_class,
                   mode = int(mode == 'negative'),
                   lc = np.select([lc == 'PGC', lc == 'C18'], [0, 1], 2),
                   modification = np.select([modification == 'reduced', modification == 'permethylated'], [0, 1], 2),
                   trap = np.select([trap == 'linear', trap == 'orbitrap', trap == 'amazon'], [0, 1, 2], 3))
    df['glycan'] = [0] * len(df)
    # Retention time normalization
    max_rt = max(max(df['RT']), rt_max_default)
    df['RT2'] = df['RT'] / max_rt
    # Candidate structures of one composition share spectrum and composition vector, i.e., the model input, so every distinct input is
    # predicted once (input_id points each row to its input)
    input_ids = {}
    df['input_id'] = [input_ids.setdefault((s, tuple(v)), len(input_ids)) for s, v in zip(df['spec_id'], df['compositional_vector'])]
    df['tta'] = True if tta_thresh is None else (df.groupby('spec_id')['annotation_score'].transform('max') > tta_thresh).values
    df_in = df.drop_duplicates('input_id')
    # Dataloader generation
    X = list(zip(df_in.binned_intensities.values.tolist(), df_in.mz_remainder.values.tolist(),
                 df_in.compositional_vector.values.tolist(), df_in.glycan_type.values.tolist(),
                 df_in.RT2.values.tolist(), df_in['mode'].values.tolist(), df_in.lc.values.tolist(),
                 df_in.modification.values.tolist(), df_in.trap.values.tolist()))
    # Test-time augmentation (5 augmented copies, aggregated by max) only where the prediction can reach the output
    tta = df_in['tta'].values
    X_tta = unwrap([[k] * 5 for k, t in zip(X, tta) if t])
    X_plain = [k for k, t in zip(X, tta) if not t]
    dset = torch.utils.data.ConcatDataset([SimpleDataset(X_tta, pd.Series([0] * len(X_tta)), transform_mz = transform_mz, transform_rt = transform_rt),
                                           SimpleDataset(X_plain, pd.Series([0] * len(X_plain)))])
    dloader = torch.utils.data.DataLoader(dset, batch_size = 256, shuffle = False)
    idx_col = 'm/z' if 'm/z' in df.columns else 'reducing_mass'
    df.set_index(idx_col, inplace = True)
    drop_cols = ['binned_intensities', 'mz_remainder', 'RT2', 'mode', 'modification', 'trap', 'glycan', 'glycan_type',
                 'lc']
    df.drop(drop_cols, axis = 1, inplace = True)
    return dloader, df


def get_topk(dataloader, model, glycans, k = 25, temp = False, temperature = temperature):
    """yields topk CandyCrunch predictions for spectra in dataloader\n
   | Arguments:
   | :-
   | dataloader (PyTorch): dataloader from process_for_inference
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch
   | k (int): how many top predictions to provide for each spectrum; default:25
   | temp (bool): whether to calibrate logits by temperature factor; default:False
   | temperature (float): the temperature factor used to calibrate logits; default:1.2097\n
   | Returns:
   | :-
   | (1) a nested list of topk glycans for each spectrum
   | (2) a nested list of associated prediction confidences, for each spectrum
   """
    n_samples = len(dataloader.dataset)
    preds = np.empty((n_samples, k), dtype = int)
    conf = np.empty((n_samples, k), dtype = float)
    start_idx = 0
    with torch.inference_mode():
        for data in dataloader:
            mz_list, mz_remainder, precursor, glycan_type, rt, mode, lc, modification, trap, y = data
            mz_list = torch.stack([mz_list, mz_remainder], dim = 1)
            batch_size = len(y)
            inputs = [x.to(device, non_blocking = True) for x in
                      [mz_list, precursor, glycan_type, rt, mode, lc, modification, trap]]
            pred = model(*inputs)
            if temp:
                pred = pred / temperature
            pred = F.softmax(pred, dim = 1)
            conf_topk, idx_topk = torch.topk(pred, k, dim = 1)
            end_idx = start_idx + batch_size
            preds[start_idx:end_idx, :] = idx_topk.cpu().numpy()
            conf[start_idx:end_idx, :] = conf_topk.cpu().numpy()
            start_idx = end_idx
    preds = [[glycans[i] for i in j] for j in preds]
    return preds, conf.tolist()


comp_cache = {}


def get_comp(glycan):
    if glycan not in comp_cache:
        try:
            comp_cache[glycan] = glycan_to_composition(glycan)
        except Exception:
            comp_cache[glycan] = {}
    return comp_cache[glycan]


def mass_check(mass, glycan, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized',
               mass_tag = None, double_thresh = 900,
               triple_thresh = 1500, quadruple_thresh = 3500, mass_tolerance = 0.5, permitted_charges = [1, 2, 3, 4]):
    """determine whether glycan could explain m/z\n
   | Arguments:
   | :-
   | mass (float): observed m/z
   | glycan (string): glycan in IUPAC-condensed nomenclature
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', '2AA', '2AB', 'procainamide', or 'custom'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/peracetylated
   | mass_tag (float): label mass to add when calculating possible m/z if modification == 'custom'; default:0
   | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
   | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1500
   | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500
   | mass_tolerance (float): maximum allowed mass difference to return True; default:0.5
   | permitted_charges (list): charges of ions used to check mass against; default:[1,2,3,4]\n
   | Returns:
   | :-
   | Returns True if glycan could explain mass and False if not
   """
    try:
        mz = glycan_to_mass(glycan, sample_prep = sample_prep, modification = modification) if isinstance(glycan,
                                                                                                          str) else glycan + composition_to_mass(
            {}, sample_prep = sample_prep, modification = modification) - composition_to_mass({}, sample_prep = sample_prep)
    except:
        return False
    ions = get_ion_mzs(mz + (mass_tag or 0), max_charge = int(max(permitted_charges)) * (1 if mode == 'positive' else -1),
                       adducts = get_adduct_list(mode), min_mass = {2: double_thresh, 3: triple_thresh, 4: quadruple_thresh})
    # ion names end in their charge, e.g., '[M-H]-' or '[M+Acetate-H]2-'
    return [m for ion, m in ions.items() if int(ion.rsplit(']', 1)[1][:-1] or 1) in permitted_charges and abs(mass - m) < mass_tolerance]


def condense_dataframe(df, mz_diff = 0.5, rt_diff = 1.0, min_mz = 39.714, max_mz = 3000, bin_num = 2048):
    """groups spectra and combines the clusters into averaged and binned spectra\n
    | Arguments:
    | :-
    | df (dataframe): dataframe from load_spectra_filepath
    | mz_diff (float): mass tolerance for assigning spectra to the same peak; default:0.5
    | rt_diff (float): retention time tolerance (in minutes) for assigning spectra to the same peak; default:1.0
    | min_mz (float): minimal m/z used for binning; don't change; default:39.714
    | max_mz (float): maximal m/z used for binning; don't change; default:3000
    | bin_num (int): number of bins for binning; don't change; default: 2048\n
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
        for cluster in active:
            last_max = cluster['apex_int']
            if last_max > 0:
                last_mz, last_rt = cluster['apex_mz'], cluster['apex_rt']
            else:
                last_mz, last_rt = cluster['m/z'][-1], cluster['RT'][-1]
            if abs(last_mz - mz) <= mz_diff and abs(last_rt - rt) <= rt_diff:
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
    # Create a condensed dataframe
    condensed_data = []
    for cluster in clusters:
        highest_intensity_index = np.argmax(cluster['intensity'])
        highest_intensity = cluster['intensity'][highest_intensity_index]
        if highest_intensity > 0:
            rep_mz = cluster['m/z'][highest_intensity_index]
            mean_rt = cluster['RT'][highest_intensity_index]
        else:
            rep_mz = min(cluster['m/z'])
            mean_rt = np.mean(cluster['RT'])
        # Cluster fragment peaks across spectra by mass proximity, then weight-average mass and sum intensity
        mi_pairs = sorted([(m, i) for spec in cluster['peak_d'] for m, i in spec.items()], key = lambda x: x[0])
        pk = {}
        if mi_pairs:
            cur_grp = [mi_pairs[0]]
            grp_start = mi_pairs[0][0]
            for m, i in mi_pairs[1:]:
                if m - grp_start <= mz_diff / 2:
                    cur_grp.append((m, i))
                else:
                    ms, ints = zip(*cur_grp)
                    pk[np.average(ms, weights = ints)] = sum(ints)
                    cur_grp = [(m, i)]
                    grp_start = m
            if cur_grp:
                ms, ints = zip(*cur_grp)
                pk[np.average(ms, weights = ints)] = sum(ints)
        peaks = dict(sorted(pk.items(), key = lambda x: x[1], reverse = True))
        binned_intensities, mz_remainder = zip(*[bin_intensities(c, frames) for c in cluster['peak_d']])
        binned_intensities = np.mean(np.array(binned_intensities), axis = 0)
        mz_remainder = np.mean(np.array(mz_remainder), axis = 0)
        # Bin intensity normalization
        binned_intensities = binned_intensities / binned_intensities.sum()
        num_spectra = len(cluster['RT'])
        rep_charge = cluster['precursor_charge'][highest_intensity_index]
        # Without MS1, the apex precursor intensity is the abundance: it tracks MS1 peak areas far better than summing or integrating
        # the precursor intensities of however many MS2 spectra dynamic exclusion happened to allow
        condensed_data.append(
            [rep_mz, mean_rt, highest_intensity, peaks, binned_intensities, mz_remainder, num_spectra, rep_charge])
    df_out = pd.DataFrame(condensed_data,
                          columns = ['m/z', 'RT', 'intensity', 'peak_d', 'binned_intensities', 'mz_remainder',
                                     'num_spectra', 'precursor_charge'])
    # The MS3 spectra of all MS2 spectra in a cluster
    if 'ms3' in df.columns:
        df_out['ms3'] = [c['ms3'] for c in clusters]
    return df_out


def create_struct_map(df_glycan, glycan_class, filter_out = None, phylo_level = 'Kingdom', phylo_filter = 'Animalia'):
    processed_df_use = df_glycan[
        df_glycan[f"{phylo_level}"].apply(lambda x: phylo_filter in x) & (df_glycan['glycan_type'] == glycan_class)]
    if filter_out:
        processed_df_use = processed_df_use.iloc[
            [i for i, x in enumerate(processed_df_use.Composition) if not filter_out.intersection(x)]]
    processed_df_use = processed_df_use.assign(comp_str = [stringify_dict(x) for x in processed_df_use.Composition])
    processed_df_use = processed_df_use.assign(
        ref_counts = processed_df_use.loc[:, 'ref'].map(len) + processed_df_use.loc[:, 'tissue_ref'].map(len) +
                     processed_df_use.loc[:, 'disease_ref'].map(len))
    processed_df_use = processed_df_use[~(processed_df_use['glycan'].str.contains("}"))]
    processed_df_use = processed_df_use.sort_values('ref_counts', ascending = False)
    processed_df_use = processed_df_use.assign(topology = [structure_to_basic(x) for x in processed_df_use['glycan']])
    processed_df_use = processed_df_use.assign(
        glycan = [x.replace('-ol', '').replace('1Cer', '') for x in processed_df_use.glycan])
    small_comps = processed_df_use[[0 < sum(x.values()) < 6 for x in processed_df_use.Composition]]
    df_use_unq_topos = small_comps.groupby('topology').first().groupby('comp_str').agg(list)
    topology_map = dict(zip(df_use_unq_topos.index, df_use_unq_topos.glycan))
    df_use_unq_comps = processed_df_use.groupby('comp_str').first()
    common_struct_map = dict(zip(df_use_unq_comps.index, df_use_unq_comps.glycan))
    return common_struct_map, processed_df_use, topology_map


def assign_candidate_structures(df_in, df_glycan_in, comp_struct_map, topo_struct_map, mass_tolerance, mode, mass_tag,
                                modification = 'reduced', sample_prep = 'underivatized', max_charge = -3):
    idx_col = 'm/z' if 'm/z' in df_in.columns else 'reducing_mass'
    red_masses = np.array(df_in[idx_col])
    known_charges = [None if pd.isna(c) else int(c) for c in
                     df_in['precursor_charge'].values] if 'precursor_charge' in df_in.columns else [None] * len(
        red_masses)
    tag = mass_tag if mass_tag else 0
    all_comps = [x for x in df_glycan_in.groupby('comp_str').first()['Composition']]
    comps_in = copy.deepcopy(all_comps)
    comp_masses = np.array(
        [composition_to_mass(x, mass_value = 'monoisotopic', sample_prep = sample_prep,
                             modification = modification) + tag for
         x in comps_in])
    comps_out = [(None, 0)] * len(red_masses)
    comps_with_none = comps_in + [None]
    # A top 5 fragment heavier than 1.1x the precursor m/z rules out z = 1 (as in domain_filter)
    heavy_frags = [any(m > 1.1 * mz for m in list(peaks)[:5]) for mz, peaks in zip(red_masses, df_in['peak_d'])]

    def _match_chunked(candidate_masses):
        out = []
        for mz_chunk in np.array_split(red_masses, max(1, len(red_masses) // 1000)):
            row_idx, comp_idx = np.where(
                np.abs(candidate_masses.reshape(1, -1) - mz_chunk.reshape(-1, 1)) < mass_tolerance)
            values, indices, _ = np.unique(row_idx, return_counts = True, return_index = True)
            subarrays = np.split(comp_idx, indices)[1:]
            comps_all = [None] * len(mz_chunk)
            for x, y in zip(values, subarrays):
                comps_all[x] = y
            out.extend([[comps_with_none[mc] for mc in x] if x is not None else x for x in comps_all])
        return out

    def _update(chunked, charge, replace = False):
        # Only overwrite where this scenario found a match, the slot is still empty (or, with replace, holds z = 1 matches that the
        # spectrum rules out and domain_filter would empty), and any known charge agrees
        return [(y, charge) if ((not x[0] or (replace and heavy and x[1] == 1)) and y and (kc is None or kc == charge)) else x
                for x, y, kc, heavy in zip(comps_out, chunked, known_charges, heavy_frags)]

    # Try each charge state separately; higher charges produce lower observed m/z for the same neutral mass, so a ruled-out singly charged
    # match must not hide a multiply charged one at the same m/z (e.g., HexNAc1Neu5Ac2 [M-H]- vs. Hex3HexNAc4dHex2 [M-2H]2- at 804.3)
    for charge in range(1, abs(max_charge) + 1):
        if mode == 'negative':
            charged_comp_masses = (comp_masses - charge * PROTON_MASS) / charge
        else:
            charged_comp_masses = (comp_masses + charge * PROTON_MASS) / charge
        comps_out = _update(_match_chunked(charged_comp_masses), charge, replace = True)
    valid_adducts = [(a, mass_dict[a]) for a in get_adduct_list(mode) if mass_dict.get(a, 999) != 999]
    for adduct, adduct_mass in valid_adducts:
        comps_out = _update(_match_chunked(comp_masses + adduct_mass), 1)
    # Multiply-charged adduct ions: only fill gaps not explained by protonated or singly-charged adduct matches
    threshold_dict = {2: 900, 3: 1500, 4: 3500}
    for adduct, adduct_mass in valid_adducts:
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
    for adduct, adduct_mass in valid_adducts:
        for charge in range(2, abs(max_charge) + 1):
            threshold = threshold_dict.get(charge, 9999)
            # All charge carriers are the adduct (no protons): m/z = (M + z×adduct) / z
            multi_adduct_masses = (comp_masses + charge * adduct_mass) / charge
            multi_adduct_masses = np.where(comp_masses + charge * adduct_mass > threshold, multi_adduct_masses, 9999)
            comps_out = _update(_match_chunked(multi_adduct_masses), charge)
    df_in['composition'] = [x[0] for x in comps_out]
    df_in['charge'] = [x[1] if x[0] else None for x in comps_out]
    candidate_data = []
    for matched_comps_str, matched_comps in [([stringify_dict(y) for y in x], x) if x else (x, x) for x in
                                             df_in.composition]:
        if not matched_comps:
            candidate_data.append(([None], [None]))
        else:
            # Prefer topology-level structures when available; fall back to the most common structure for that composition
            structures = [s for comp_str in matched_comps_str for s in
                          topo_struct_map.get(comp_str, [comp_struct_map[comp_str]])]
            compositions = [comp for comp, comp_str in zip(matched_comps, matched_comps_str) for _ in
                            topo_struct_map.get(comp_str, [comp_struct_map[comp_str]])]
            candidate_data.append((structures, compositions))
    df_in['candidate_structure'], df_in['composition'] = zip(*candidate_data)
    df_in = df_in.explode(['composition', 'candidate_structure']).reset_index(names = 'spec_id')
    # Build a fixed-length composition vector aligned to comp_vector_order for model input
    df_in['compositional_vector'] = [
        np.array([x.get(m, 0) for m in comp_vector_order]) if x else None
        for x in df_in.composition
    ]
    return df_in


def deisotope_ms2(peaks: Dict[float, float], precursor_charge: int,
                  mass_tolerance: float = 0.2, sum_intensities: bool = True,
                  min_intensity: float = 0.0, min_isotope_count: int = 2,
                  validate_pattern: bool = True) -> Dict[float, float]:
    """De-isotope MS2 spectrum identifying direct isotope patterns."""
    sorted_peaks = sorted([(m, i) for m, i in peaks.items() if i >= min_intensity])
    mass_arr = np.array([m for m, _ in sorted_peaks])
    consumed = np.zeros(len(sorted_peaks), dtype = bool)
    deisotoped = {}
    spacings = [1.0034 / z for z in range(1, precursor_charge + 1)]
    for i, (current_mass, current_intensity) in enumerate(sorted_peaks):
        # Skip if already processed
        if consumed[i]: continue
        consumed[np.searchsorted(mass_arr, current_mass - mass_tolerance, 'left'):np.searchsorted(mass_arr,
                                                                                                  current_mass + mass_tolerance,
                                                                                                  'right')] = True
        best_pattern = [(current_mass, current_intensity)]
        best_charge = 0
        # Check all charge states
        for charge, spacing in enumerate(spacings, 1):
            next_mass = current_mass
            charge_pattern = [(current_mass, current_intensity)]
            start_idx = i + 1
            # Fast-forward to relevant mass range
            while start_idx < len(sorted_peaks) and sorted_peaks[start_idx][0] < next_mass + spacing - mass_tolerance:
                start_idx += 1
            for j in range(start_idx, len(sorted_peaks)):
                candidate_mass, candidate_intensity = sorted_peaks[j]
                if candidate_mass > next_mass + spacing + mass_tolerance:
                    break
                if consumed[j]:
                    continue
                if abs(candidate_mass - next_mass - spacing) <= mass_tolerance:
                    if validate_pattern and not (candidate_intensity <= charge_pattern[-1][1] * 1.20 or
                                                 candidate_intensity >= charge_pattern[-1][1] * 0.05):
                        continue
                    charge_pattern.append((candidate_mass, candidate_intensity))
                    next_mass = candidate_mass
            if len(charge_pattern) > len(best_pattern):
                best_pattern = charge_pattern
                best_charge = charge
        if len(best_pattern) >= min_isotope_count:
            mono_mass = best_pattern[0][0]
            for peak_mass, _ in best_pattern[1:]:
                consumed[np.searchsorted(mass_arr, peak_mass - mass_tolerance, 'left'):
                                                           np.searchsorted(mass_arr, peak_mass + mass_tolerance, 'right')] = True
            deisotoped[mono_mass] = sum(i for _, i in best_pattern) if sum_intensities else best_pattern[0][1]
        else:
            deisotoped[current_mass] = current_intensity
    return dict(sorted(deisotoped.items(), key = lambda x: x[1], reverse = True))


def assign_annotation_scores_pooled(df_in, multiplier, mass_tag, mass_tolerance, modification = 'reduced',
                                    sample_prep = 'underivatized'):
    mode = 'negative' if multiplier == -1 else 'positive'
    adduct_list = get_adduct_list(mode)
    idx_col = 'm/z' if 'm/z' in df_in.columns else 'reducing_mass'
    unq_structs = df_in[df_in['candidate_structure'].notnull()].groupby('candidate_structure').first().reset_index()
    groups = df_in.groupby('candidate_structure', sort = False).indices
    comp_map = dict(zip(unq_structs.candidate_structure, unq_structs.composition))
    charge_vals = np.asarray(df_in['charge'].values, dtype = float)
    mz_vals, peak_vals, spec_ids = df_in[idx_col].values, df_in['peak_d'].values, df_in['spec_id'].values
    scores_out = np.zeros(len(df_in))
    # A spectrum is a row of every candidate structure it matched, so its deisotoped fragments are computed once per charge state
    rounded_masses = {}
    for struct, rows in groups.items():
        grp_charges = charge_vals[rows]
        row_charge = np.nanmax(grp_charges) if not np.all(np.isnan(grp_charges)) else 1.0
        comp_mass = composition_to_mass(comp_map[struct], sample_prep = sample_prep, modification = modification) + (
            mass_tag if mass_tag else 0)
        charges_arr = np.abs(charge_vals[rows])
        spec_masses = mz_vals[rows] * charges_arr
        is_adduct = any(
            np.any(np.abs(comp_mass + mass_dict.get(adduct, 999) - spec_masses) < mass_tolerance) or
            np.any(np.abs(comp_mass + mass_dict.get(adduct, 999) * charges_arr - spec_masses) < mass_tolerance)
            for adduct in adduct_list)
        rounded_mass_rows = [rounded_masses[key] if (key := (spec, row_charge)) in rounded_masses else
                             rounded_masses.setdefault(key, [np.round(y, 1) for y in deisotope_ms2(x, int(abs(row_charge)), 0.05)][:15])
                             for spec, x in zip(spec_ids[rows], peak_vals[rows])]
        # MS3 spectra (if the file has them) are pooled per isolated MS2 fragment, as CandyCrumbs only explains their peaks with fragments of
        # what that fragment is in this structure
        ms3_rows = [[(round(p), p, [np.round(y, 1) for y in deisotope_ms2(x, int(abs(row_charge)), 0.05)][:15]) for p, x in ms3] for ms3 in
                    df_in['ms3'].values[rows]] if 'ms3' in df_in.columns else []
        pools = {None: (None, set([x for y in rounded_mass_rows for x in y]))}
        for key, p, masses in [x for y in ms3_rows for x in y]:
            pools.setdefault(key, (p, set()))[1].update(masses)
        tester_mass_scores = {}
        for key, (ms3_precursor, unq_rounded_masses) in pools.items():
            cc_out = CandyCrumbs(struct, unq_rounded_masses, mass_tolerance, simplify = False,
                                 charge = int(multiplier * abs(row_charge)),
                                 disable_global_mods = (not is_adduct or mode == "negative"), disable_X_cross_rings = True,
                                 max_cleavages = 2, mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0),
                                 sample_prep = sample_prep, ms3_precursor = ms3_precursor)
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
        # A row's MS3 evidence is the mean score of its MS3 spectra, so it does not grow with how often the instrument repeated them
        if ms3_rows:
            row_scores = [score + (np.mean([sum(tester_mass_scores[(key, x)] for x in masses) for key, _, masses in ms3]) if ms3 else 0)
                          for score, ms3 in zip(row_scores, ms3_rows)]
        scores_out[rows] = row_scores
    df_in['annotation_score'] = scores_out
    return df_in


def deduplicate_predictions(df, mz_diff = 0.5, rt_diff = 1.0):
    """removes/unifies duplicate predictions\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference
   | mz_diff (float): mass tolerance for assigning spectra to the same peak; default:0.5
   | rt_diff (float): retention time tolerance (in minutes) for assigning spectra to the same peak; default:1.0\n
   | Returns:
   | :-
   | Returns a deduplicated dataframe
   """
    # Sort by index and 'RT'
    df.sort_values(by = 'RT', inplace = True)
    df.sort_index(inplace = True)
    idx_vals = df.index.values
    rt_vals = df['RT'].values
    preds_col = df['predictions'].values
    first_preds = np.array([p[0][0] if p else None for p in preds_col], dtype = object)
    conf_first = np.array([p[0][1] if p else -np.inf for p in preds_col])
    abund = df['rel_abundance'].values if 'rel_abundance' in df.columns else None
    lo = np.searchsorted(idx_vals, idx_vals - mz_diff, side = 'right')
    hi = np.searchsorted(idx_vals, idx_vals + mz_diff, side = 'left')
    # Each row is represented by the most confident row with its top prediction within mz_diff and rt_diff (rows without a prediction by themselves)
    reps = np.arange(len(df))
    for k in range(len(df)):
        if first_preds[k] is not None:
            window = np.arange(lo[k], hi[k])
            sel = window[(np.abs(rt_vals[window] - rt_vals[k]) < rt_diff) & (first_preds[window] == first_preds[k])]
            reps[k] = sel[np.argmax(conf_first[sel])]
    dedup_df = df.iloc[pd.unique(reps)].copy()
    if abund is not None:
        # A representative carries the abundance of the rows it represents, so a row linking two representatives is not counted twice
        dedup_df['rel_abundance'] = [np.nansum(abund[reps == r]) for r in pd.unique(reps)]
        dedup_df = dedup_df.astype({'rel_abundance': df['rel_abundance'].dtype})
    return dedup_df


def domain_filter(df_out, glycan_class, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized',
                  max_charge = -3, mass_tolerance = 0.5, filter_out = set(), df_use = None, mass_tag = None):
    """filters out false-positive predictions\n
   | Arguments:
   | :-
   | df_out (dataframe): df_out generated within wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', or 'other'/'none'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/peracetylated
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:None
   | df_use (dataframe): glycan database used to check whether compositions are valid; default: df_glycan
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:0.0\n
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
        [composition_to_mass(comp, sample_prep = sample_prep, modification = modification) + tag for comp in
         df_out['composition'].values])
    raw_masses = df_out.index.values * np.abs(df_out['charge'].values)
    df_out['adduct'] = None
    charges_abs = np.abs(df_out['charge'].values)
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
    new_preds = []
    top_fragments_col = df_out['top_fragments'].tolist()
    predictions_col = df_out['predictions'].tolist()
    adduct_col = df_out['adduct'].tolist()
    charge_col = df_out['charge'].tolist()
    index_vals = df_out.index.tolist()
    double_mass_tolerance = 2 * mass_tolerance
    has_sulfate = {}
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
        to_append = len(current_preds) > 0
        top_frags = top_fragments_col[k]
        float_frags = [j for j in top_frags if isinstance(j, float)]
        adduct_name = adduct_col[k]
        for i, m in enumerate(current_preds):
            m = m[0]
            truth = [True]
            # Diagnostic ions: a sialic acid in the structure must leave its diagnostic fragment
            for sia in ('Neu5Ac', 'Neu5Gc', 'Kdn'):
                if sia in m:
                    truth.append(any(abs(mass_dict[sia] + PROTON_MASS * multiplier - j) < double_mass_tolerance or
                                     abs(assumed_mass - mass_dict[sia] - j) < double_mass_tolerance or
                                     abs(precursor_mz - ((mass_dict[sia] - addy) / c) - j) < double_mass_tolerance for j
                                     in float_frags))
                if 'Neu5Gc' not in m:
                    truth.append(not any(abs(mass_dict['Neu5Gc'] + PROTON_MASS * multiplier - j) < mass_tolerance
                                         for j in top_frags[:5] if isinstance(j, float)))
                if 'Neu5Ac' not in m and 'Neu5Gc' not in m:
                    truth.append(not any(abs(mass_dict['Neu5Ac'] + PROTON_MASS * multiplier - j) < mass_tolerance
                                         for j in top_frags[:5] if isinstance(j, float)))
                if 'Neu5Ac' not in m and (m.count('Fuc') + m.count('dHex') > 1):
                    truth.append(
                        not any(abs(mass_dict['Neu5Ac'] + PROTON_MASS * multiplier - j) < double_mass_tolerance or
                                     abs(precursor_mz - mass_dict['Neu5Ac'] - j) < double_mass_tolerance
                                     for j in top_frags[:10] if isinstance(j, float)))
            if 'S' in m and len(current_preds) == 1:
                # Rows exploded from one spectrum share top_frags, so each fragment's composition lookup is cached
                for t in top_frags[:20]:
                    if t not in has_sulfate:
                        has_sulfate[t] = 'S' in (mz_to_composition(t, max_charge = max_charge, mass_tolerance = mass_tolerance,
                                                                   glycan_class = glycan_class, df_use = df_use,
                                                                   filter_out = filter_out, modification = modification,
                                                                   sample_prep = sample_prep)[0:1] or ({},))[0].keys()
                    if has_sulfate[t]:
                        break
                truth.append(any(has_sulfate[t] for t in top_frags[:20]))
            # Check fragment size distribution
            if c > 1:
                truth.append(any(j > precursor_mz * 1.2 for j in top_frags[:15]))
            if c == 1:
                truth.append(all(j < precursor_mz * 1.1 for j in top_frags[:5]))
            if len(top_frags) < 2:
                truth.append(False)
            # Check neutral loss of adduct for adducts
            if isinstance(adduct_name, str):
                neutral_loss = mass_dict.get(adduct_name, 999) - (PROTON_MASS * multiplier)
                expected_frag = precursor_mz - neutral_loss / c
                truth.append(any(abs(expected_frag - j) < mass_tolerance for j in top_frags[:10]))
            if all(truth):
                if to_append:
                    keep.append(current_preds[i])
                else:
                    pass
            else:
                if to_append:
                    pass
                else:
                    keep.append('remove')
        new_preds.append(keep)
    df_out['predictions'] = new_preds
    return df_out[df_out['predictions'].apply(lambda x: 'remove' not in x[:1])]


def backfill_missing(df):
    """finds rows with composition-only that match existing predictions wrt mass and RT and propagates\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference\n
   | Returns:
   | :-
   | Returns backfilled dataframe
   """
    predictions = df['predictions'].values
    compositions = df['composition'].apply(stringify_dict).values
    charges = df['charge'].values
    RTs = df['RT'].values
    masses = df.index.values * np.abs(charges) - charges * PROTON_MASS
    for k in range(len(df)):
        if not len(predictions[k]) > 0:
            target_mass = masses[k]
            target_RT = RTs[k]
            target_composition = compositions[k]
            mass_diffs = np.abs(masses - target_mass)
            RT_diffs = np.abs(RTs - target_RT)
            same_compositions = compositions == target_composition
            idx = np.where((mass_diffs < 0.5) & (RT_diffs < 1) & same_compositions)[0]
            if len(idx) > 0:
                df.iat[k, 0] = predictions[idx[0]]
    return df


def impute(df_out, pred_thresh, mode = 'negative', modification = 'reduced', sample_prep = 'underivatized',
           mass_tag = 0.0,
           glycan_class = "O"):
    """searches for specific isomers that could be added to the prediction dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are 'reduced', or 'other'/'none'
    | sample_prep (string): underivatized/permethylated/peracetylated
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:0.0
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"\n
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
    if glycan_class == "O":
        # O-glycans may carry a sulfated GlcNAc branch that is interchangeable with the unsulfated form
        seqs = [p[0][0] for p in predictions_list if p and ("GlcNAc6S(b1-6)" in p[0][0] or "GlcNAc(b1-6)" in p[0][0])]
        variants.update(set(unwrap([_get_all_variants(s, 'GlcNAc6S(b1-6)', 'GlcNAc(b1-6)') for s in seqs])))
    # Every empty row tries the same variants (first match in sorted order, as mass_check would find it), so their masses and ion m/z are
    # computed once per charge state instead of once per row and variant
    variants = sorted(variants)
    variant_masses, ion_mzs = [], {}
    for v in variants:
        try:
            variant_masses.append(glycan_to_mass(v, sample_prep = sample_prep, modification = modification))
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
    top1_preds = set([k[0][0] for k in predictions_list if k and k[0]])
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


def make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = 'Mammalia',
                  sample_prep = 'underivatized'):
    """generates a mass dict that can be used in the possibles() function\n
   | Arguments:
   | :-
   | glycans (list): glycans used for training CandyCrunch
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | taxonomy_class (string): which taxonomic class to use for selecting possible glycans; default:'Mammalia'
   | sample_prep (string): underivatized/permethylated/peracetylated\n
   | Returns:
   | :-
   | Returns a dictionary of form mass : list of glycans
   """
    exp_glycans = set(df_use.glycan.values.tolist())
    class_glycans = [k for k in glycans if enforce_class(k, glycan_class)]
    exp_glycans.update(class_glycans)
    mass_dic = {9999: []}
    for k in exp_glycans:
        try:
            composition = get_comp(k)
            if not filter_out.intersection(composition.keys()):
                mass = glycan_to_mass(k, sample_prep = sample_prep)
                mass_dic.setdefault(mass, []).append(k)
            else:
                mass_dic[9999].append(k)
        except:
            mass_dic[9999].append(k)
    return mass_dic


def canonicalize_biosynthesis(df_out, pred_thresh):
    """regularize predictions by incentivizing biosynthetic feasibility\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01\n
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
    prediction_column = []
    for k, row in df_out[::-1].iterrows():
        new_preds = []
        preds = row['predictions']
        if len(preds) == 0:
            prediction_column.append(new_preds)
            continue
        for p in [x for x in preds if len(x) == 2 if x[1]]:
            p_list = list(p)
            if len(p_list) == 1:
                p_list.append(0)
            p_list[1] += 0.1 * sum(subgraph_isomorphism(p_list[0], t) for t in rest_top1 if t != p_list[0])
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
    return df_out.loc[df_out.index.sort_values(), :]


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
    ext = os.path.splitext(spectra_filepath)[1].lower()
    if ext == ".mzml":
        return process_mzML_stack(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
    if ext == ".mzxml":
        return process_mzXML_stack(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
    if ext == ".raw":
        return process_raw_stack(spectra_filepath, intensity = True, extract_ms1 = extract_ms1)
    if ext == ".mgf":
        rows = []
        for spectrum in read_mgf(spectra_filepath):
            params = spectrum['params']
            if not len(spectrum['m/z array']):
                continue
            if 'rtinseconds' not in params:
                raise ValueError(f"MGF spectrum '{params.get('title', '')}' has no RTINSECONDS entry; CandyCrunch needs retention times")
            # Same conventions as process_mzML_stack: top 1000 peaks sorted by intensity, charge 1 treated as undetermined
            charge = abs(int(params['charge'][0])) if params.get('charge') else None
            peak_d = dict(sorted(zip(spectrum['m/z array'].tolist(), spectrum['intensity array'].tolist()), key = lambda x: x[1], reverse = True)[:1000])
            rows.append([float(params['pepmass'][0]), peak_d, float(params['rtinseconds']) / 60,
                         charge if charge != 1 else None, params['pepmass'][1] if params['pepmass'][1] is not None else np.nan,
                         int(params['scans']) if str(params.get('scans')).isdigit() else params.get('scans'), None])
        return pd.DataFrame(rows, columns = ['m/z', 'peak_d', 'RT', 'precursor_charge', 'intensity', 'scan', 'activation'])
    if ext == ".pkl":
        loaded_file = pd.read_pickle(spectra_filepath)
        return loaded_file
    if ext == ".xlsx":
        loaded_file = pd.read_excel(spectra_filepath)

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
        # Files written by extract_spectra carry the ion mode and analyzer detected from their raw file
        for k in ('mode', 'trap'):
            if k in loaded_file.columns:
                detected = loaded_file.pop(k).dropna()
                loaded_file.attrs[f'detected_{k}'] = detected.iloc[0] if len(detected) else None
        return loaded_file
    if ext == ".csv":
        storage = DictStorage()
        loaded_file = storage.read(spectra_filepath, 'peak_d')
        return loaded_file
    raise FileNotFoundError(
        'Incorrect filepath or extension, please ensure it is in the intended directory and is one of the supported formats')


def extract_spectra(spectra_filepath, output_filepath = None):
    """extracts the MS/MS spectra of a .raw/.mzML/.mzXML/.mgf file into a lightweight .xlsx that wrap_inference reads directly\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".raw", ".mzML", ".mzXML", or ".mgf"
   | output_filepath (string): .xlsx filepath to write; default:None (spectra_filepath with an .xlsx extension)\n
   | Returns:
   | :-
   | Returns the filepath of the written .xlsx; precursor m/z values are already refined from MS1 and isotope-triggered repeat spectra removed,
   | MS3 spectra are kept (column ms3), but MS1 itself is not, so predictions from the .xlsx are quantified by precursor intensity instead of XIC areas
   """
    df = load_spectra_filepath(spectra_filepath)
    # Rounding keeps the file small and every peak dictionary below Excel's 32,767-character cell limit; 4 decimals and 4 significant
    # digits sit far below the binning and fragment-annotation tolerances
    df['peak_d'] = [str({round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for d in df['peak_d']]
    if 'ms3' in df.columns:
        df['ms3'] = [str([(round(p, 4), {round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for p, d in x]) for x in df['ms3']]
    # Ion mode and analyzer detected from the raw file would not survive the export, so they travel as columns that load_spectra_filepath
    # turns back into the attrs wrap_inference checks
    for k in ('mode', 'trap'):
        df[k] = df.attrs.get(f'detected_{k}')
    output_filepath = output_filepath or os.path.splitext(spectra_filepath)[0] + '.xlsx'
    df.to_excel(output_filepath, index = False)
    return output_filepath


def combine_charge_states(df_out):
    """looks for several charges at the same RT with the same top prediction and combines their relative abundances\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe where the singly-charged state now carries the sum of abundances
    """
    df_out['top_pred'] = [k[0][0] if len(k) > 0 else np.nan for k in df_out.predictions]
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
        # Every lowest-charge row absorbs the higher charge states of this glycan that co-elute with it
        for idx, row in pred_rows[pred_rows['charge'].abs() == lowest].iterrows():
            partners = pred_rows[(pred_rows['charge'].abs() > lowest) & ((pred_rows['RT'] - row['RT']).abs() < 1) & pred_rows.index.isin(df_filtered.index)]
            df_filtered.at[idx, 'rel_abundance'] += partners['rel_abundance'].sum()
            df_filtered = df_filtered.drop(partners.index)
    df_out = pd.concat([df_out[~df_out['top_pred'].isin(filtered_top_pred)], df_filtered]).sort_index()
    df_out.drop(['top_pred'], axis = 1, inplace = True)
    return df_out


def combine_adduct_species(df_out, rt_diff = 1.0):
    """combines abundances when the same glycan appears as different adduct forms (e.g., [M-H]⁻ and [M+Acetate]⁻)\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | rt_diff (float): retention time tolerance (in minutes) for considering two rows the same species; default:1.0\n
    | Returns:
    | :-
    | Returns prediction dataframe with adduct abundances consolidated into the primary ion form
    """
    df_out['top_pred'] = [k[0][0] if len(k) > 0 else np.nan for k in df_out.predictions]
    repeated = df_out['top_pred'].value_counts()
    repeated = repeated[repeated > 1].index.tolist()
    # Only act when multiple rows share a prediction but differ in m/z (i.e., different ionization forms)
    rows_to_drop = []
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
                if df_out.at[idx_i, 'rel_abundance'] >= df_out.at[idx_j, 'rel_abundance']:
                    df_out.at[idx_i, 'rel_abundance'] = df_out.at[idx_i, 'rel_abundance'] + df_out.at[idx_j, 'rel_abundance']
                    df_out.at[idx_i, 'num_spectra'] = df_out.at[idx_i, 'num_spectra'] + df_out.at[idx_j, 'num_spectra']
                    rows_to_drop.append(idx_j)
                    used.add(idx_j)
                else:
                    df_out.at[idx_j, 'rel_abundance'] = df_out.at[idx_j, 'rel_abundance'] + df_out.at[idx_i, 'rel_abundance']
                    df_out.at[idx_j, 'num_spectra'] = df_out.at[idx_j, 'num_spectra'] + df_out.at[idx_i, 'num_spectra']
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
        df_out["Original_Prediction"] = df_out["predictions"].apply(
            lambda x: x[0][0] if x else None
        )
    # Iterate through unique Gc predictions
    for gc_pred in df_out[df_out["Original_Prediction"].str.contains("Neu5Gc", na = False)][
        "Original_Prediction"
    ].unique():
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
        # Clean up temporary RT_diff column
        df_out.drop(columns = ["RT_diff"], inplace = True, errors = "ignore")
    # Remove the helper column
    df_out.drop(columns = ["Original_Prediction"], inplace = True, errors = "ignore")
    return df_out


def filter_delayed_rts(df_out, mass_tolerance):
    """function to filter out duplicates if they come toward the end of the run\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe in which, if possible, fake isomers have been deduplicated
    """
    # Create an empty list to store indices of rows to discard
    rows_to_discard = []
    # Group by m/z with a tolerance of +/- 0.5 and identical top1 predictions
    df_work = df_out.assign(
        _cxn = np.array([p[0][1] if p else 0.0 for p in df_out['predictions']]) * df_out['num_spectra'].to_numpy())
    for mz_val in df_out.index.unique():
        mz_group = df_work.loc[
            (df_work.index >= mz_val - mass_tolerance) & (df_work.index <= mz_val + mass_tolerance)
            ]
        # Further group by top1 prediction
        for top1_pred, group in mz_group.groupby('top1_pred'):
            # Sort the group by RT to easily find the first instance
            group_sorted = group.sort_values(by = "RT")
            first_rt = group_sorted.iloc[0]["RT"]
            max_cxn = group['_cxn'].max()
            # Delayed rows that lose to the group maximum
            rows_to_discard.extend(
                group_sorted.index[(group_sorted["RT"] > first_rt + 10) & (group_sorted['_cxn'] < max_cxn)])
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
                        mass_tag, filter_out, taxonomy_class, mass_tolerance, mass_dic, sample_prep = 'underivatized',
                        max_charge = -3):
    """adds and reorders predictions based on possible structures\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of filtered predictions
    | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks
    | experimental (bool): whether to impute missing predictions via database searches etc.
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
    | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
    | modification (string): chemical modification of glycans; options are 'reduced', '2AA', '2AB', 'procainamide', or 'custom'; default:'reduced'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
    | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia'
    | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
    | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
    | sample_prep (string): underivatized/permethylated/peracetylated
    | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3\n
    | Returns:
    | :-
    | Returns a dataframe of predictions
    """
    # Construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
    if supplement:
        try:
            df_out = supplement_prediction(df_out, glycan_class, mode = mode, modification = modification,
                                           mass_tag = mass_tag, sample_prep = sample_prep)
            df_out['evidence'] = [
                'medium' if pd.isna(evidence) and preds else evidence
                for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
            ]
        except:
            pass
    # Check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
    if experimental:
        df_out = impute(df_out, pred_thresh, mode = mode, modification = modification, mass_tag = mass_tag,
                        glycan_class = glycan_class, sample_prep = sample_prep)
        try:
            df_out = filter_delayed_rts(df_out, mass_tolerance)
            df_out = Ac_follows_Gc(df_out)
        except ValueError:
            pass
        ionization = -PROTON_MASS if mode == 'negative' else PROTON_MASS
        # Reducing-end modification as glycowork adds it, including the group a derivatized alditol gains on its extra free OH
        mass_offset = composition_to_mass({}, sample_prep = sample_prep, modification = modification) - composition_to_mass(
            {}, sample_prep = sample_prep) + (mass_tag or 0) + ionization
        mass_dic = mass_dic if mass_dic else make_mass_dic(glycans, glycan_class, filter_out, df_use,
                                                           taxonomy_class = taxonomy_class, sample_prep = sample_prep)
        df_out = possibles(df_out, mass_dic, mass_offset)
        df_out['evidence'] = [
            'weak' if pd.isna(evidence) and preds else evidence
            for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
        ]
    # Filter out wrong predictions via diagnostic ions etc.
    if supplement or experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, modification = modification,
                               mass_tolerance = mass_tolerance, df_use = df_use, sample_prep = sample_prep,
                               mass_tag = mass_tag, max_charge = max_charge)
        df_out['predictions'] = [[(k[0].replace('-ol', '').replace('1Cer', ''), k[1]) if len(k) > 1 else (
            k[0].replace('-ol', '').replace('1Cer', ''),) for k in j] if j else j for j in df_out['predictions']]
    return df_out


def finalise_predictions(df_out, get_missing, pred_thresh, mode, modification, mass_tag, ppm_thresh, rt_diff,
                         sample_prep = 'underivatized', glycan_class = 'O', mass_tolerance = 0.5, ms1 = None):
    """Cleans up incorrect structure predictions, quantifies, and formats dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of augmented predictions
    | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition
    | pred_thresh (float): prediction confidence threshold used for filtering
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are 'reduced', '2AA', '2AB', 'procainamide', or 'custom'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | ppm_thresh (float): ppm error threshold; the file's own robust ppm error spread can only make it stricter
    | rt_diff (float): maximum retention time difference (in minutes) for combining adduct species of one glycan
    | sample_prep (string): underivatized/permethylated/peracetylated
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
    | mass_tolerance (float): flat-Da mass tolerance used for composition matching and XIC extraction; default:0.5
    | ms1 (tuple): flat MS1 data from process_mzML_stack; if given, abundances are XIC areas; default:None\n
    | Returns:
    | :-
    | Returns a tuple of (dataframe of corrected predictions with unnormalized rel_abundance, list of spectra)
    """
    # Reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
    try:
        df_out = canonicalize_biosynthesis(df_out, pred_thresh)
    except ValueError:
        pass
    # Keep or remove spectra that still lack a prediction after all this (canonicalize_biosynthesis drops confidence-less database guesses)
    if not get_missing:
        df_out = df_out[df_out['predictions'].str.len() > 0]
    df_out = flag_coeluting_substructures(df_out, glycan_class)
    # Calculate ppm error
    valid_indices, ppm_errors, theo_mzs = [], [], []
    for preds, obs_mass in zip(df_out['predictions'], df_out.index):
        theo_mass = mass_check(obs_mass, preds[0][0], modification = modification, mass_tag = mass_tag, mode = mode,
                               sample_prep = sample_prep, mass_tolerance = mass_tolerance) if preds else None
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
    known_ppm = df_out['ppm_error'].dropna().values
    if len(known_ppm) >= 5:
        med = np.median(known_ppm)
        ppm_thresh = min(ppm_thresh, med + 3.0 * np.median(np.abs(known_ppm - med)) * 1.4826)
    df_out = df_out[~(df_out['ppm_error'] >= ppm_thresh)]
    # Retention-time outlier removal: drop predictions whose RT lies far outside the file's overall
    # elution distribution, scaled to the observed spread (tight cluster => strict, wide spread => permissive)
    has_pred = df_out['predictions'].apply(len) > 0
    rts_pred = df_out.loc[has_pred, 'RT'].values
    if len(rts_pred) >= 5:
        rt_med = np.median(rts_pred)
        rt_mad = np.median(np.abs(rts_pred - rt_med))
        if rt_mad > 0:
            rt_cut = 3.0 * rt_mad * 1.4826  # ~3-sigma on robust spread
            df_out = df_out[(~has_pred) | (np.abs(df_out['RT'] - rt_med) <= rt_cut)]
    # Clean-up
    df_out['composition'] = [get_comp(k[0][0]) if k else val for k, val in
                             zip(df_out['predictions'], df_out['composition'])]
    df_out['charge'] = round(df_out['composition'].apply(lambda x: composition_to_mass(x, sample_prep = sample_prep,
                                                                                       modification = modification)) / df_out.index) * (-1 if mode == 'negative' else 1)
    df_out = df_out.astype({'num_spectra': 'int', 'charge': 'int'})
    # Quantify via MS1 where available, else keep precursor intensities: each row's isotope envelope at its theoretical m/z, over its own chromatographic peak, before charge states and adducts of a glycan are summed
    if ms1 is not None and len(ms1[0]) > 0 and len(df_out) > 0:
        calibration = ms1_mz_calibration(ms1, df_out['theo_mz'].values, df_out['RT'].values, df_out['charge'].values,
                                         mz_tolerance = mass_tolerance)
        xic_areas = extract_xic_areas(ms1, df_out['theo_mz'].values, df_out['RT'].values, charges = df_out['charge'].values,
                                      mz_tolerance = mass_tolerance, weights = df_out['rel_abundance'].values,
                                      mz_calibration = calibration, sample_prep = sample_prep,
                                      glycans = [k[0][0] if k else None for k in df_out['predictions']])
        if (xic_areas > 0).any():
            df_out['rel_abundance'] = xic_areas
            df_out.attrs['ms1_calibration'] = calibration
    df_out = df_out.drop(columns = ['theo_mz'])
    df_out = combine_charge_states(df_out)
    df_out = combine_adduct_species(df_out, rt_diff = rt_diff)
    # Map GlyTouCan IDs of the structure reported, which can differ from the first candidate
    df_out["GlyTouCan_ID"] = [glytoucan_mapping.get(g, '') if isinstance(g, str) else '' for g in df_out['top1_pred']]
    # MS2 peaks that support the reported structure, per residue: intense single glycosidic cleavages that no other placement of that residue could form
    # (residues without any other placement are not counted), and peaks that only another placement explains
    supported_by = []
    for top1, preds, peaks, charge in zip(df_out['top1_pred'], df_out['predictions'], df_out['peak_d'], df_out['charge']):
        support = supporting_ions(top1, peaks, charge = int(charge), candidates = [p[0] for p in preds if p[0] != top1],
                                  mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0), sample_prep = sample_prep) if isinstance(
            top1, str) and isinstance(peaks, dict) else {'residues': []}
        tested, texts = [e for e in support['residues'] if e['support'] or e['open']], []
        for e in tested:
            names = [re.sub(r'_(\d+)_(.).*', lambda m: m[1] + m[2].lower(), s[2][0]) for s in e['support'][:2]]
            texts += [e['name'] + ' (' + ', '.join(f'{s[0]:.1f} {n}' for s, n in zip(e['support'], names)) + ')'] if names else []
            texts += [f"against {e['name']} ({a[0]:.1f} fits {a[4]})" for a in e['against'][:1]]
        supported_by.append(f"{sum(bool(e['support']) for e in tested)}/{len(tested)} residues supported" + (': ' + '; '.join(texts) if texts else '') if tested else '')
    df_out['supported_by'] = supported_by
    if (df_out['rel_abundance'] == 0).all():
        df_out = df_out.drop(columns = ['rel_abundance'])
    spectra_out = df_out.pop('peak_d').values.tolist()
    df_out = df_out[['top1_pred', 'predictions'] + [c for c in df_out.columns if c not in ('top1_pred', 'predictions')]]
    df_out.index.name = "m/z"
    return df_out, spectra_out


def flag_coeluting_substructures(df_out, glycan_class, rt_tolerance = 0.25):
    """Flag potential in-source fragments and O-glycan peeling products"""
    df_out['notes'] = ''
    top1_data = [(idx, row['predictions'][0][0], row['RT'])
                 for idx, row in df_out.iterrows()
                 if row['predictions']]
    if len(top1_data) < 2:
        return df_out
    pred_graphs = {}
    for _, pred, _ in top1_data:
        if pred not in pred_graphs:
            try:
                pred_graphs[pred] = glycan_to_nxGraph(pred)
            except Exception:
                pass
    # In-source fragments: substructure co-elutes with superstructure
    for i, (idx_a, pred_a, rt_a) in enumerate(top1_data):
        if pred_a not in pred_graphs:
            continue
        notes = []
        for j, (idx_b, pred_b, rt_b) in enumerate(top1_data):
            if i == j or pred_b not in pred_graphs:
                continue
            if pred_b.count('(') - pred_a.count('(') != 1:
                continue
            if abs(rt_a - rt_b) > rt_tolerance:
                continue
            if subgraph_isomorphism(pred_graphs[pred_b], pred_graphs[pred_a]):
                notes.append(f"Possible in-source fragment of {pred_b}")
        if notes:
            df_out.at[idx_a, 'notes'] = '; '.join(notes)
    # Peeling products: O-glycan minus reducing-end mono, independent RT
    if glycan_class == 'O':
        peeled_strings = {}
        for pred, g in pred_graphs.items():
            if len(g.nodes()) < 3:
                continue
            root = max(g.nodes())
            link_children = list(g.successors(root))
            if len(link_children) != 1:
                continue
            peeled = g.copy()
            peeled.remove_nodes_from([root, link_children[0]])
            if len(peeled.nodes()) > 0:
                try:
                    peeled_strings[pred] = graph_to_string(peeled)
                except Exception:
                    pass
        for i, (idx_a, pred_a, rt_a) in enumerate(top1_data):
            if pred_a not in pred_graphs:
                continue
            peeling_notes = []
            for j, (idx_b, pred_b, rt_b) in enumerate(top1_data):
                if i == j or pred_b not in peeled_strings:
                    continue
                try:
                    if compare_glycans(pred_a, peeled_strings[pred_b]):
                        peeling_notes.append(f"Possible peeling product of {pred_b}")
                except Exception:
                    pass
            if peeling_notes:
                existing = df_out.at[idx_a, 'notes']
                combined = '; '.join([existing, '; '.join(peeling_notes)]) if existing else '; '.join(peeling_notes)
                df_out.at[idx_a, 'notes'] = combined
    return df_out


def wrap_inference(spectra_filepath, glycan_class, model = candycrunch, glycans = glycans, bin_num = 2048,
                   max_charge = -3, frag_num = 100, modification = 'reduced', mass_tag = None, lc = 'PGC', trap = 'linear',
                   rt_min = 0, rt_max = 0, rt_diff = 1.0, rt_max_default = 30.0,
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False,
                   extra_thresh = 0.2, crumbs_thresh = 3, ppm_thresh = 300,
                   filter_out = {'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True,
                   experimental = True, mass_dic = None, sample_prep = 'underivatized',
                   taxonomy_level = 'Class', taxonomy_filter = 'Mammalia', df_use = None, plot_glycans = False,
                   _return_intermediate = False):
    """wrapper function to get & curate CandyCrunch predictions\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".raw" (Thermo), ".mzML", ".mzXML", ".mgf", or ".xlsx" pointing to a file containing spectra or preprocessed spectra;
   |                            MS3 spectra in it (.raw/.mzML/.mzXML, or .xlsx from extract_spectra) are used automatically, as fragment evidence and to rank isomers
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | modification (string): chemical modification of glycans; options are 'reduced', '2AA', '2AB', 'procainamide', or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0
   | rt_max (float): whether only spectra up to a maximum retention time (in minutes) should be considered; default:0
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
   | temperature (float): the temperature factor used to calibrate logits; default:1.15
   | spectra (bool): whether to also output the actual spectra used for prediction; default:False
   | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
   | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); below it, they are only allowed if more likely than every in-class structure of that composition; default:0.2
   | crumbs_thresh (float): threshold for annotation score to keep predictions; default:3
   | ppm_thresh (float): ppm error threshold for filtering; default:300
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:{'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}
   | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
   | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
   | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
   | sample_prep (string): underivatized/permethylated/peracetylated
   | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia'
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | plot_glycans (bool): whether to save an .xlsx file with SNFG images of all top1 predictions next to spectra_filepath, named like it plus _output; default:False\n
   | Returns:
   | :-
   | Returns dataframe of predictions for spectra in file; if spectra=True, a tuple of (dataframe, list of its MS2 spectra), and files with MS3 spectra
   | then also have a column ms3 (list of (isolated MS2 fragment m/z, MS3 peak dictionary) per row)
   """
    # ppm_thresh is the only tolerance the user sets; derive the flat-Da window everything downstream needs from it here
    mass_tolerance = ppm_thresh * MZ_REF / 1e6
    # 'custom' labels are described by mass_tag alone; glycowork's mass functions only know the named modifications
    modification = None if modification == 'custom' else modification
    mode = "negative" if max_charge < 0 else "positive"
    if not _return_intermediate:
        print(
            f"Your chosen settings are: {glycan_class} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type == glycan_class])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
    multiplier = -1 if mode == 'negative' else 1
    loaded_file = load_spectra_filepath(spectra_filepath, extract_ms1 = spectra_filepath.lower().endswith(('.mzml', '.mzxml', '.raw')))
    ms1 = loaded_file.attrs.pop('ms1', None)
    detected_mode = getattr(loaded_file, 'attrs', {}).get('detected_mode')
    detected_trap = getattr(loaded_file, 'attrs', {}).get('detected_trap')
    if detected_mode and detected_mode != mode:
        print(
            f"WARNING: File contains {detected_mode}-mode spectra but mode='{mode}' was specified. Overriding to '{detected_mode}'.")
        mode = detected_mode
        multiplier = -1 if mode == 'negative' else 1
        # The sign of max_charge is the ion mode for everything that takes it (e.g., mz_to_composition in domain_filter)
        max_charge = abs(max_charge) * multiplier
    if detected_trap and detected_trap != trap:
        print(
            f"WARNING: File was acquired on {detected_trap} but trap='{trap}' was specified. Overriding to '{detected_trap}'.")
        trap = detected_trap
    loaded_file = filter_rts(loaded_file, rt_min, rt_max)
    if loaded_file.empty:
        # Without MS2 spectra there are no glycan peaks, so this is the empty result of any other file without them
        print(f"WARNING: {os.path.basename(spectra_filepath)} has no MS2 spectra left after RT filtering, so it has no glycan peaks.")
        df_out = pd.DataFrame(columns = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d', 'annotation_score', 'rel_abundance',
                                         'top_fragments'], index = pd.Index([], name = 'm/z'))
        if _return_intermediate:
            df_out.attrs.update(ms1 = ms1, mode = mode)
            return df_out
        df_out.insert(0, 'top1_pred', [])
        df_out['ppm_error'] = []
        return (df_out, []) if spectra else df_out
    intensity = 'intensity' in loaded_file.columns and not (loaded_file['intensity'] == 0).all() and not loaded_file[
        'intensity'].isnull().all()
    if intensity:
        loaded_file.loc[loaded_file['intensity'].isnull(), 'intensity'] = 0
    else:
        loaded_file['intensity'] = [0] * len(loaded_file)
    # Prepare file for processing
    loaded_file.dropna(subset = ['peak_d'], inplace = True)
    idx_col = 'm/z' if 'm/z' in loaded_file.columns else 'reducing_mass'
    loaded_file[idx_col] += np.random.uniform(10 ** (-20), 0.00001, size = len(loaded_file))
    coded_class = {'O': 0, 'N': 1, 'free': 2, 'lipid': 2}[glycan_class]
    # Group spectra by mass/retention isomers and process them for being inputs to CandyCrunch
    df_out = condense_dataframe(loaded_file, mz_diff = mass_tolerance, rt_diff = rt_diff, bin_num = bin_num)
    common_structure_map, df_use, topo_struct_map = create_struct_map(df_use, glycan_class, filter_out = filter_out,
                                                                      phylo_level = taxonomy_level,
                                                                      phylo_filter = taxonomy_filter)
    df_out = assign_candidate_structures(df_out, df_use, common_structure_map, topo_struct_map, mass_tolerance, mode,
                                         mass_tag, modification = modification, max_charge = max_charge,
                                         sample_prep = sample_prep)
    df_out = assign_annotation_scores_pooled(df_out, multiplier, mass_tag, mass_tolerance, modification = modification,
                                             sample_prep = sample_prep)
    df_out = df_out[df_out['compositional_vector'].notnull()].reset_index(drop = True)
    # Spectra without an annotation above crumbs_thresh are dropped unless rescued, so their inputs get one plain pass instead of 5 augmented ones
    loader, df_out = process_for_inference(df_out, coded_class, mode = mode, modification = modification, lc = lc,
                                           trap = trap,
                                           rt_max_default = rt_max_default, tta_thresh = crumbs_thresh)
    # Predict glycans from spectra
    preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
    # Max over the 5 augmented copies of each test-time augmented input (they come first in the loader), a plain input has one copy
    tta = df_out.drop_duplicates('input_id')['tta'].values
    order = np.concatenate([np.flatnonzero(tta), np.flatnonzero(~tta)])
    sizes = np.where(tta[order], 5, 1)
    preds_in, conf_in = [None] * len(tta), [None] * len(tta)
    for i, start, n in zip(order, np.cumsum(sizes) - sizes, sizes):
        combs = [{p: c for p, c in zip(cp, cc)} for cp, cc in zip(preds[start:start + n], pred_conf[start:start + n])]
        combs = dict(sorted(average_dicts(combs, mode = 'max').items(), key = lambda x: x[1], reverse = True))
        preds_in[i], conf_in[i] = list(combs.keys()), list(combs.values())
    preds, pred_conf = preds_in, conf_in
    df_out['rel_abundance'] = df_out['intensity']
    df_out['predictions'] = [[(pred, conf) for pred, conf in zip(preds[i], pred_conf[i])] for i in df_out['input_id']]
    _raw_predictions = df_out['predictions'].tolist()
    # Check correctness of glycan class & mass
    df_out['predictions'] = [[g for g in preds if g[1] > pred_thresh and get_comp(g[0]) == comp] for preds, comp in
                             zip(df_out.predictions, df_out.composition)]
    # A cross-class structure needs extra_thresh, unless it is more likely than every in-class isomer of that composition:
    # the class prior may veto a composition's cross-class structures, but should not make the row pick a less likely isomer
    best_in_class = [max([g[1] for g in preds if enforce_class(g[0], glycan_class)], default = 0) for preds in df_out.predictions]
    df_out['predictions'] = [
        [(g[0], round(g[1], 4)) for g in preds if
         enforce_class(g[0], glycan_class, g[1], extra_thresh = extra_thresh) or 0 < best_in <= g[1]][:5]
        for preds, best_in in zip(df_out.predictions, best_in_class)
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
                if (enforce_class(g_name, glycan_class, g_conf, extra_thresh = extra_thresh)
                        and mass_check(df_out.index[i], g_name, mode = mode, modification = modification,
                                       mass_tag = mass_tag, sample_prep = sample_prep,
                                       permitted_charges = [abs(int(df_out['charge'].iloc[i]))],
                                       mass_tolerance = mass_tolerance)):
                    _model_tops[i] = (g_name, g_conf)
                    break
        _struct_to_indices = defaultdict(list)
        for i, (g_name, _) in _model_tops.items():
            _struct_to_indices[g_name].append(i)
        for _struct, _indices in _struct_to_indices.items():
            _pred_comp = get_comp(_struct)
            if not _pred_comp:
                continue
            _pred_comp_mass = composition_to_mass(_pred_comp, sample_prep = sample_prep,
                                                  modification = modification) + _tag
            _charges_arr = np.array([abs(int(df_out['charge'].iloc[i])) for i in _indices])
            _spec_masses = np.array([df_out.index[i] * _charges_arr[j] for j, i in enumerate(_indices)])
            _is_adduct = any(
                np.any(np.abs(_pred_comp_mass + mass_dict.get(adduct, 999) - _spec_masses) < mass_tolerance) or
                np.any(
                    np.abs(_pred_comp_mass + mass_dict.get(adduct, 999) * _charges_arr - _spec_masses) < mass_tolerance)
                for adduct in _adduct_list)
            _row_charge = int(max(_charges_arr))
            _peak_dicts = [df_out['peak_d'].iloc[i] for i in _indices]
            _rounded_mass_rows = [[np.round(y, 1) for y in deisotope_ms2(pd, _row_charge, 0.05)][:15] for pd in
                                  _peak_dicts]
            _unq_rounded = set(m for row in _rounded_mass_rows for m in row)
            try:
                _cc_out = CandyCrumbs(_struct, _unq_rounded, mass_tolerance, simplify = False,
                                      charge = int(multiplier * _row_charge),
                                      disable_global_mods = (not _is_adduct or mode == "negative"),
                                      disable_X_cross_rings = True, max_cleavages = 2,
                                      mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0),
                                      sample_prep = sample_prep)
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
                    df_out.iat[_idx, df_out.columns.get_loc('predictions')] = [(_gn, round(_gc, 4))]
                    df_out.iat[_idx, df_out.columns.get_loc('composition')] = get_comp(_gn)
                    df_out.iat[_idx, df_out.columns.get_loc('annotation_score')] = float(_score)
    df_out['charge'] = [c * multiplier for c in df_out['charge']]
    df_out['RT'] = df_out['RT'].round(2)
    # Extract & sort the top 100 fragments by intensity
    df_out['top_fragments'] = [
        [round(float(frag[0]), 4) for frag in sorted(peak_d.items(), key = lambda x: x[1], reverse = True)[:frag_num]]
        for peak_d in df_out['peak_d']
    ]
    # Filter out wrong predictions via diagnostic ions etc.
    if experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, sample_prep = sample_prep,
                               mass_tag = mass_tag, max_charge = max_charge,
                               modification = modification, mass_tolerance = mass_tolerance,
                               df_use = df_use).reset_index()
    else:
        df_out = df_out.reset_index()
    df_out = df_out.sort_values(['spec_id', 'annotation_score'], ascending = False).groupby('spec_id').first()
    df_out = df_out[df_out['annotation_score'] > crumbs_thresh].drop(columns = ['candidate_structure']).set_index('m/z')
    if 'ms3' in df_out.columns:
        # MS3 spectra resolve isomers whose MS2 fragments look alike: an isomer explaining a row's MS3 spectra with more fragments of the fragment
        # they isolated (scored as in assign_annotation_scores_pooled) moves up, isomers explaining them equally well keep the model's order
        reranked = []
        for preds, ms3, charge in zip(df_out['predictions'], df_out['ms3'], df_out['charge']):
            pools = {}
            for p, x in ms3 if len(preds) > 1 else []:
                pools.setdefault(round(p), (p, []))[1].append([np.round(y, 1) for y in deisotope_ms2(x, int(abs(charge)), 0.05)][:15])
            scores = [0] * len(preds)
            for i, (g, _) in enumerate(preds):
                for p, ms3_spectra in pools.values():
                    cc_out = CandyCrumbs(g, set(m for masses in ms3_spectra for m in masses), mass_tolerance, simplify = False, charge = int(charge),
                                         disable_global_mods = True, disable_X_cross_rings = True, max_cleavages = 2,
                                         mass_tag = modification_mass_dict.get(modification, 0) + (mass_tag or 0), sample_prep = sample_prep,
                                         ms3_precursor = p)
                    counts = {k: sum(not (len(a) > 1 and {c.split('_')[0][-1] for c in a} & {'A', 'X'}) and not (
                        len(a) > 2 and 'M' in {c.split('_')[0][-1] for c in a}) for a in v['Domon-Costello nomenclatures']) if v else 0
                              for k, v in cc_out.items()}
                    scores[i] += np.mean([sum(counts[m] for m in masses) for masses in ms3_spectra])
            reranked.append([preds[i] for i in sorted(range(len(preds)), key = lambda i: -scores[i])])
        df_out['predictions'] = reranked
    df_out = df_out[
        ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d', 'annotation_score', 'rel_abundance',
         'top_fragments'] + (['ms3'] if 'ms3' in df_out.columns else [])]
    if not df_out.empty:
        # Deduplicate identical predictions for different spectra
        df_out = deduplicate_predictions(df_out, mz_diff = mass_tolerance, rt_diff = rt_diff)
        df_out['evidence'] = ['strong' if preds else np.nan for preds in df_out['predictions']]
    if _return_intermediate:
        df_out.attrs.update(ms1 = ms1, mode = mode)
        return df_out
    if df_out.empty:
        df_out.insert(0, 'top1_pred', [])
        df_out['ppm_error'] = []
        return (df_out, []) if spectra else df_out
    df_out['top1_pred'] = [x[0][0] if x else np.nan for x in df_out.predictions]
    # Same augmentation and clean-up as every file in wrap_inference_batch
    if supplement or experimental:
        df_out = augment_predictions(df_out, pred_thresh, supplement, experimental, glycan_class, df_use, mode,
                                     modification, mass_tag, filter_out, taxonomy_filter, mass_tolerance, mass_dic,
                                     sample_prep = sample_prep, max_charge = max_charge)
    df_out, spectra_out = finalise_predictions(df_out, get_missing, pred_thresh, mode, modification, mass_tag,
                                               ppm_thresh, rt_diff, sample_prep = sample_prep,
                                               glycan_class = glycan_class, mass_tolerance = mass_tolerance, ms1 = ms1)
    if 'rel_abundance' in df_out.columns:
        df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
    if plot_glycans:
        from glycowork.motif.draw import plot_glycans_excel
        plot_glycans_excel(df_out.drop(columns = ['ms3'], errors = 'ignore').reset_index(),
                           os.path.splitext(spectra_filepath)[0] + '_output.xlsx', glycan_col_num = 'top1_pred')
    # MS3 spectra are output like the MS2 spectra, only with spectra=True
    return (df_out, spectra_out) if spectra else df_out.drop(columns = ['ms3'], errors = 'ignore')


def wrap_inference_batch(spectra_filepath_list, glycan_class, intra_cat_thresh, top_n_isomers = 5, model = candycrunch,
                         glycans = glycans, bin_num = 2048, max_charge = -3,
                         frag_num = 100, modification = 'reduced', mass_tag = None, lc = 'PGC',
                         trap = 'linear', rt_min = 0, rt_max = 0, rt_diff = 1.0, rt_max_default = 30.0,
                         pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False,
                         extra_thresh = 0.2, crumbs_thresh = 3, ppm_thresh = 300,
                         filter_out = {'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True,
                         experimental = True, mass_dic = None, sample_prep = 'underivatized',
                         taxonomy_level = 'Class', taxonomy_filter = 'Mammalia', df_use = None, plot_glycans = False,
                         n_jobs = 1):
    """wrapper function to get & curate CandyCrunch predictions, then harmonize them across multiple files\n
   | Arguments:
   | :-
   | spectra_filepath_list (list): list of absolute filepaths ending in ".raw" (Thermo), ".mzML", ".mzXML", ".mgf", or ".xlsx" pointing to files containing spectra
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | intra_cat_thresh (float): minutes the RT of a structure can differ from the mean of a group
   | top_n_isomers (int): number of different isomer groups at each composition to retain; default:5
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | max_charge (int): maximum signed charge to consider for composition matching etc.; default -3
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | modification (string): chemical modification of glycans; options are 'reduced', '2AA', '2AB', 'procainamide', or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0
   | rt_max (float): whether only spectra up to a maximum retention time (in minutes) should be considered; default:0
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
   | rt_max_default (float): minimum maximum retention time to normalize to; default: 30.0
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
   | temperature (float): the temperature factor used to calibrate logits; default:1.15
   | spectra (bool): whether to also output the actual spectra used for prediction; default:False
   | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
   | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); below it, they are only allowed if more likely than every in-class structure of that composition; default:0.2
   | crumbs_thresh (float): threshold for annotation score to keep predictions; default:3
   | ppm_thresh (float): ppm error threshold for filtering; default:300
   | filter_out (set): set of monosaccharide or modification types to filter out; default:{'Ac', 'Kdn', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}
   | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
   | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
   | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
   | sample_prep (string): underivatized/permethylated/peracetylated
   | taxonomy_level (string): taxonomy level to filter by; default:'Class'
   | taxonomy_filter (string): which taxonomy to pull glycans for; default:'Mammalia'
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
   | plot_glycans (bool): whether to save a <file>_output.xlsx file per input file with SNFG images of all top1 predictions; default:False
   | n_jobs (int): number of files to process in parallel (separate processes); default:1\n
   | Returns:
   | :-
   | Returns a tuple of (feature table with one row per isomer group: top1_pred, consensus m/z, RT, charge, composition, GlyTouCan_ID, number of files with MS2 evidence, then per file its rel_abundance (num_spectra if no file has intensities) and evidence_<file>; dict of per-file dataframes, or of (dataframe, spectra) tuples if spectra=True, with the column ms3 for files with MS3 spectra, as in wrap_inference)
   """
    mode = "negative" if max_charge < 0 else "positive"
    mass_tolerance = ppm_thresh * MZ_REF / 1e6
    modification = None if modification == 'custom' else modification
    print(
        f"Your chosen settings are: {glycan_class} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type == glycan_class])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)].reset_index(drop = True)
    # Built once here instead of once per file inside augment_predictions
    if experimental and not mass_dic:
        mass_dic = make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = taxonomy_filter,
                                 sample_prep = sample_prep)
    # Files with the same name in different folders still get distinct labels
    file_labels = [os.path.splitext(os.path.basename(fp))[0] for fp in spectra_filepath_list]
    file_labels = [label if file_labels.count(label) == 1 else f"{label}_{i}" for i, label in enumerate(file_labels)]
    # Core inference per file via wrap_inference intermediate return; MS1 data is parked on disk so only one file's MS1 is in memory at a time
    ms1_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
    inference_kwargs = dict(glycan_class = glycan_class, glycans = glycans, bin_num = bin_num, max_charge = max_charge,
                            frag_num = frag_num, modification = modification, mass_tag = mass_tag, lc = lc, trap = trap,
                            rt_min = rt_min, rt_max = rt_max, rt_diff = rt_diff, rt_max_default = rt_max_default,
                            pred_thresh = pred_thresh, temperature = temperature, get_missing = get_missing,
                            extra_thresh = extra_thresh, crumbs_thresh = crumbs_thresh, ppm_thresh = ppm_thresh,
                            filter_out = filter_out, supplement = False, experimental = experimental,
                            mass_dic = mass_dic, sample_prep = sample_prep, taxonomy_level = taxonomy_level,
                            taxonomy_filter = taxonomy_filter, df_use = df_use, _return_intermediate = True)
    # Worker processes load the default model themselves, so only a custom model has to be sent to them
    if model is not candycrunch:
        inference_kwargs['model'] = model
    tasks = [(fp, os.path.join(ms1_dir.name, f'{i}.npz'), inference_kwargs) for i, fp in
             enumerate(spectra_filepath_list)]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers = n_jobs, initializer = torch.set_num_threads,
                                 initargs = (max(1, (os.cpu_count() or 1) // n_jobs),)) as pool:
            results = list(pool.map(_batch_inference_file, *zip(*tasks)))
    else:
        results = [_batch_inference_file(*task) for task in tasks]
    inference_dfs, file_modes, ms1_paths = {}, {}, {}
    for file_label, (df_out, file_mode, ms1_path) in zip(file_labels, results):
        # wrap_inference overrides the ion mode if the file says otherwise, so downstream steps have to follow it
        file_modes[file_label], ms1_paths[file_label] = file_mode, ms1_path
        inference_dfs[file_label] = df_out.assign(condition_label = file_label) if not df_out.empty else df_out
    # The table of a file without any glycan peak
    empty_table = pd.DataFrame(columns = ['top1_pred', 'predictions', 'composition', 'num_spectra', 'charge', 'RT', 'rel_abundance'],
                               index = pd.Index([], name = 'm/z'))
    non_empty = {k: v for k, v in inference_dfs.items() if not v.empty}
    if not non_empty:
        return pd.DataFrame(), {k: (empty_table.copy(), []) if spectra else empty_table.copy() for k in file_labels}
    all_ms2 = pd.concat(non_empty.values())
    # Cluster precursor m/z across files: split at gaps above the mass tolerance, then split clusters wider than it at their largest gap
    mz_sorted = np.unique(all_ms2.index.values)
    to_split, mz_clusters = np.split(mz_sorted, np.where(np.diff(mz_sorted) > mass_tolerance)[0] + 1), []
    while to_split:
        cluster = to_split.pop()
        if cluster[-1] - cluster[0] <= mass_tolerance:
            mz_clusters.append(cluster)
        else:
            cut = np.argmax(np.diff(cluster)) + 1
            to_split.extend([cluster[:cut], cluster[cut:]])
    mass_labels = {mz: round(float(np.median(cluster)), 4) for cluster in mz_clusters for mz in cluster}
    all_ms2['mass_label'] = all_ms2.index.map(mass_labels)
    # Cross-file harmonization: align RT drift, resolve variant predictions, link DDA gaps
    assigned_cats = assign_categories(all_ms2, intra_cat_thresh = intra_cat_thresh, maximise_cat_size = True)
    smoothed_category_predictions = assign_modal_category_prediction(assigned_cats)
    prevailing_category_predictions = filter_top_n_isomers(smoothed_category_predictions, top_n = top_n_isomers,
                                                           keep_unpredicted = get_missing)
    # Per-file augment, finalize, and quantify
    harmonized_labels = set(prevailing_category_predictions.condition_label.unique())
    for file_label, spectra_filepath in zip(file_labels, spectra_filepath_list):
        file_mode = file_modes[file_label]
        df_out, spectra_out = empty_table.copy(), []
        if file_label in harmonized_labels:
            df_out = prevailing_category_predictions[prevailing_category_predictions['condition_label'] == file_label].sort_index()
            if supplement or experimental:
                df_out = augment_predictions(df_out, pred_thresh, supplement, experimental, glycan_class, df_use,
                                             file_mode, modification, mass_tag, filter_out, taxonomy_filter,
                                             mass_tolerance, mass_dic, sample_prep = sample_prep,
                                             max_charge = abs(max_charge) * (-1 if file_mode == 'negative' else 1))
            if len(df_out) > 0:
                ms1 = None
                if ms1_paths[file_label]:
                    with np.load(ms1_paths[file_label]) as ms1_file:
                        ms1 = (ms1_file['rts'], ms1_file['mzs'], ms1_file['ints'], ms1_file['offsets'])
                df_out, spectra_out = finalise_predictions(df_out, get_missing, pred_thresh, file_mode, modification,
                                                           mass_tag, ppm_thresh, rt_diff, sample_prep = sample_prep,
                                                           glycan_class = glycan_class, mass_tolerance = mass_tolerance,
                                                           ms1 = ms1)
        inference_dfs[file_label] = (df_out, spectra_out)
    # Consensus of each isomer group that survived clean-up in at least one file, for MS1 gap filling in the others
    finalized = [v[0] for v in inference_dfs.values() if not v[0].empty]
    # Rows with a top1_pred first, so every value of a group comes from one row (groupby.first skips NaN column by column)
    master = pd.concat(finalized).reset_index().sort_values('top1_pred', key = lambda s: s.isna(), kind = 'stable').groupby(
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
        # Cross-file MS1 gap filling: propagate isomer groups to files without MS2 for them when MS1 shows their peak
        missing = master[~master['present_in'].apply(lambda s: file_label in s)]
        if ms1_paths[file_label] and not missing.empty:
            with np.load(ms1_paths[file_label]) as ms1_file:
                ms1 = (ms1_file['rts'], ms1_file['mzs'], ms1_file['ints'], ms1_file['offsets'])
            # Look for the apex within intra_cat_thresh of the group's consensus RT, on the monoisotopic trace around the consensus m/z
            _, apex_rts, apex_ints, is_peak = extract_xic_areas(ms1, missing['repr_mz'].values, missing['repr_rt'].values,
                                                                mz_tolerance = mass_tolerance, search_window = intra_cat_thresh,
                                                                isotopes = 0, mz_calibration = (0, mass_tolerance))
            # Then quantify that peak exactly like the MS2 rows of this file, so both share one abundance scale
            theo_mzs = [(mass_check(mz, top1, mode = file_mode, modification = modification, mass_tag = mass_tag,
                                    sample_prep = sample_prep, mass_tolerance = mass_tolerance) or [mz])[0] if isinstance(top1, str) else mz
                        for mz, top1 in zip(missing['repr_mz'], missing['repr_top1'])]
            areas = extract_xic_areas(ms1, theo_mzs, np.where(np.isnan(apex_rts), missing['repr_rt'].values, apex_rts),
                                      charges = missing['repr_charge'].values, mz_tolerance = mass_tolerance,
                                      mz_calibration = df_out.attrs.get('ms1_calibration'), sample_prep = sample_prep,
                                      glycans = missing['repr_top1'].tolist())
            # Only fill with a genuine peak at least as intense as the weakest precursor that got MS2 in this file,
            # and never with the peak of a feature that already has MS2 in this file
            own = all_ms2[all_ms2['condition_label'] == file_label]
            own_apex = extract_xic_areas(ms1, own.index.values, own['RT'].values, mz_tolerance = mass_tolerance,
                                         search_window = rt_diff, isotopes = 0,
                                         mz_calibration = (0, mass_tolerance))[2] if len(own) else np.zeros(0)
            floor = own_apex[own_apex > 0].min() if (own_apex > 0).any() else 0
            own_rts = own.groupby('mass_label')['RT'].apply(np.array).to_dict()
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
                'notes': 'MS1 signal only; propagated from other files'}
                for (_, row), area, apex_rt, apex_int, peak in
                zip(missing.iterrows(), areas, apex_rts, apex_ints, is_peak)
                if peak and apex_int >= floor and not (
                            np.abs(own_rts.get(row['mass_label'], np.zeros(0)) - apex_rt) <= rt_diff).any()]
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
        inference_dfs[file_label] = (df_out, spectra_out) if spectra else df_out.drop(columns = ['ms3'],
                                                                                      errors = 'ignore')
    all_outputs = [d for d in (v[0] if spectra else v for v in inference_dfs.values()) if not d.empty]
    if not all_outputs:
        return pd.DataFrame(), inference_dfs
    all_outputs = pd.concat(all_outputs).reset_index().sort_values('top1_pred', key = lambda s: s.isna(), kind = 'stable')
    # Feature table: one row per isomer group with its consensus annotation, then abundance and evidence per file
    # Without any intensities (e.g., only .mgf files), spectrum counts stand in for abundances
    value_col = 'rel_abundance' if 'rel_abundance' in all_outputs.columns else 'num_spectra'
    group_keys = ['mass_label', 'category_label']
    combined_batch = all_outputs.groupby(group_keys).agg(top1_pred = ('top1_pred', 'first'), mz = ('m/z', 'median'),
                                                         RT = ('RT', 'median'), charge = ('charge', 'first'),
                                                         composition = ('composition', 'first'),
                                                         GlyTouCan_ID = ('GlyTouCan_ID', 'first'))
    combined_batch['n_files_ms2'] = all_outputs[all_outputs['evidence'] != 'ms1_only'].groupby(group_keys)[
        'condition_label'].nunique().reindex(combined_batch.index, fill_value = 0)
    abundances = all_outputs.pivot_table(index = group_keys, columns = 'condition_label', values = value_col,
                                         aggfunc = 'sum')
    evidence = all_outputs.pivot_table(index = group_keys, columns = 'condition_label', values = 'evidence',
                                       aggfunc = 'first')
    # Every input file keeps its columns, even without any feature, so they can serve as groups downstream
    combined_batch = combined_batch.join(abundances.reindex(columns = file_labels)).join(
        evidence.reindex(columns = file_labels).add_prefix('evidence_'))
    combined_batch = combined_batch.rename(columns = {'mz': 'm/z'}).sort_values(['m/z', 'RT']).reset_index(drop = True)
    return combined_batch, inference_dfs


def _batch_inference_file(spectra_filepath, ms1_path, inference_kwargs):
    """runs the per-file part of wrap_inference_batch and parks the file's MS1 data on disk; module-level so that it can run in a process pool\n
   | Returns:
   | :-
   | Returns a tuple of (intermediate prediction dataframe, ion mode used, path to the MS1 .npz or None)
   """
    df_out = wrap_inference(spectra_filepath, **inference_kwargs)
    ms1, file_mode = df_out.attrs.pop('ms1', None), df_out.attrs.pop('mode')
    df_out.attrs.clear()
    if ms1 is None or len(ms1[0]) == 0:
        return df_out, file_mode, None
    np.savez(ms1_path, rts = ms1[0], mzs = ms1[1], ints = ms1[2], offsets = ms1[3])
    return df_out, file_mode, ms1_path


def filter_top_n_isomers(df_in, top_n = 5, keep_unpredicted = False):
    df_out = df_in.copy(deep = True)
    # Rank the isomer groups (categories) at each mass: carrying a prediction first, then by the number of files they were seen in, then by abundance
    cats = df_out.groupby(['mass_label', 'category_label']).agg(has_pred = ('top1_pred', lambda x: x.notna().any()),
                                                                file_presences = ('condition_label', 'nunique'),
                                                                abundance = ('rel_abundance', 'sum')).reset_index()
    if not keep_unpredicted:
        cats = cats[cats['has_pred']]
    cats = cats.sort_values(['has_pred', 'file_presences', 'abundance'], ascending = False, kind = 'stable').groupby('mass_label').head(top_n)
    permitted = dict(zip(zip(cats.mass_label, cats.category_label), cats.file_presences))
    df_out['file_presences'] = [permitted.get(k, 0) for k in zip(df_out.mass_label, df_out.category_label)]
    return df_out[df_out['file_presences'] > 0]


def assign_modal_category_prediction(assigned_cats):
    assigned_cats['top1_pred'] = [x[0][0] if x else None for x in assigned_cats['predictions']]
    most_common_group_preds = dict(assigned_cats[['mass_label', 'category_label', 'top1_pred']].groupby(
        ['mass_label', 'category_label']).value_counts())
    most_common_mapping = {}
    unq_groups = set([(x[0], x[1]) for x in most_common_group_preds])
    for unq in unq_groups:
        prevalence_sort = sorted([(k, v) for k, v in most_common_group_preds.items() if (k[0], k[1]) == unq],
                                 key = lambda x: x[1])
        mode_pred = prevalence_sort[-1]
        most_common_mapping[(mode_pred[0][0], mode_pred[0][1])] = mode_pred[0][2]
    assigned_cats['top1_pred'] = [most_common_mapping[(ml, cl)] if pd.notna(tp) else None for ml, cl, tp in
                                  zip(assigned_cats.mass_label, assigned_cats.category_label, assigned_cats.top1_pred)]
    assigned_cats['predictions'] = [[(top1_p, 0.888)] + [p for p in preds[1:] if p[0] != top1_p] if preds else [] for top1_p, preds in
                                    zip(assigned_cats.top1_pred, assigned_cats.predictions)]
    return assigned_cats


def assign_categories(all_ms2_spectra, intra_cat_thresh = 3, maximise_cat_size = True):
    all_mass_dfs = []
    condition_labels = all_ms2_spectra.condition_label.unique()
    # Splits the rows by mass and file once instead of masking all rows for every mass and file
    for search_mass, mass_df in all_ms2_spectra.groupby('mass_label', sort = False):
        condition_dfs = dict(list(mass_df.groupby('condition_label', sort = False)))
        mass_group_dfs = [condition_dfs.get(c, mass_df.iloc[:0]).assign(RT_group = lambda x: range(len(x))) for c in condition_labels]
        cats_mass_dfs = mass_dfs_to_categories(mass_group_dfs, intra_cat_thresh, maximise_cat_size = maximise_cat_size)
        all_mass_dfs.append(cats_mass_dfs)
    return pd.concat([p for q in all_mass_dfs for p in q])


def mass_dfs_to_categories(mass_range_dfs, inter_sample_thresh, maximise_cat_size = True):
    RT_groups = create_RT_groups(mass_range_dfs)
    categories = initialise_categories(RT_groups)
    categories = expand_RT_categories(RT_groups, categories, inter_sample_thresh, maximise_cat_size = maximise_cat_size)
    sample_cats = RT_cats_to_sample_cats(categories, RT_groups)
    cat_dfs = sample_categories_to_df(sample_cats, mass_range_dfs)
    return cat_dfs


def create_RT_groups(mass_range_dfs):
    # Every row is its own RT group (RT_group numbers the rows of each file)
    return [[[rt] for rt in sample_df['RT']] for sample_df in mass_range_dfs]


def initialise_categories(all_sample_RT_groups):
    categories = {0: []}
    for x in all_sample_RT_groups[0]:
        add_new_category(categories, x)
    return categories


def add_new_category(categories, cluster):
    new_id = max([x for x in categories])
    categories[new_id + 1] = []
    categories[new_id + 1].append(cluster)
    return categories


def expand_RT_categories(all_sample_RT_groups, categories, inter_sample_thresh, maximise_cat_size = False):
    for i, sample in enumerate(all_sample_RT_groups[1:]):
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
            if maximise_cat_size:
                optim_cats = find_closest_categories_largest(sample, all_candidate_categories, categories)
            else:
                optim_cats = find_closest_categories(sample, all_candidate_categories, categories)
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
    cat_means = get_category_means([sorted(x) for x in sample_candidates], categories)
    cluster_means = [x[0] for x in sample]
    cat_mean_diffs = []
    for cluster_mean, cat_mean in zip(cluster_means, cat_means):
        cat_mean_diffs.append({k: abs(v - cluster_mean) for k, v in cat_mean.items()})
    cats_out = [set() for m in cluster_means]
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


def find_closest_categories(sample, sample_candidates, categories):
    cat_means = get_category_means([sorted(x) for x in sample_candidates], categories)
    cluster_means = sample
    cat_mean_diffs = []
    for cluster_mean, cat_mean in zip(cluster_means, cat_means):
        cat_mean_diffs.append({k: abs(v - cluster_mean) for k, v in cat_mean.items()})
    disallowed_list, chosen_list = [], []
    sorted_idx = [sorted(cat_mean_diffs, key = lambda x: min([y for y in x.values()])).index(x) for i, x in
                  enumerate(cat_mean_diffs)]
    for cat_cands in sorted(cat_mean_diffs, key = lambda x: min([y for y in x.values()])):
        sorted_cands = sorted(cat_cands.items(), key = lambda x: x[1])
        filtered_cands = [x for x in sorted_cands if x[0] not in disallowed_list]
        if filtered_cands:
            chosen_cand = filtered_cands[0]
            chosen_list.append(chosen_cand)
            disallowed_list.append(chosen_cand[0])
        else:
            chosen_list.append((set(), None))
    cats_out = [x[0] for x in chosen_list]
    cats_out = [cats_out[x] for x in sorted_idx]
    return cats_out


def get_category_means(candidate_categories, categories):
    category_means = []
    for cand_cats in candidate_categories:
        category_means.append({cat: np.mean([p for q in categories[cat] for p in q]) for cat in cand_cats})
    return category_means


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
            cat_df = cat_df.assign(category_label = [cat for x in cat_df['RT']])
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
   | modification (string): chemical modification of glycans; options are 'reduced', or 'other'/'none'; default:'reduced'
   | sample_prep (string): underivatized/permethylated/peracetylated
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
    net = construct_network(preds, permitted_roots = permitted_roots)
    if glycan_class == 'free':
        net = evoprune_network(net)
    unexplained_idx = [idx for idx, pred in enumerate(df['predictions']) if not pred]
    unexplained = df.index[unexplained_idx].tolist()
    charges = df.charge[unexplained_idx].tolist()
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

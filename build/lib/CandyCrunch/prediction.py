import ast
import copy
import operator
import os
import pickle
from itertools import combinations
from collections import defaultdict

import numpy as np
import numpy_indexed as npi
import pandas as pd
import pymzml
import torch
import torch.nn.functional as F
from glycowork.glycan_data.loader import df_glycan, stringify_dict, unwrap
from glycowork.motif.analysis import get_differential_expression
from glycowork.motif.graph import subgraph_isomorphism
from glycowork.motif.processing import enforce_class, expand_lib, get_lib
from glycowork.motif.tokenization import (composition_to_mass,
                                          glycan_to_composition,
                                          glycan_to_mass, mapping_file,
                                          mz_to_composition)
from glycowork.network.biosynthesis import construct_network, evoprune_network
from pyteomics import mzxml
from sklearn.cluster import AgglomerativeClustering

from CandyCrunch.model import (CandyCrunch_CNN, SimpleDataset, transform_mz,
                               transform_prec, transform_rt)

this_dir, this_filename = os.path.split(__file__)
data_path = os.path.join(this_dir, 'glycans.pkl')
glycans = pickle.load(open(data_path, 'rb'))
lib = get_lib(glycans)

fp_in = "drive/My Drive/CandyCrunch/"

# Choose the correct computing architecture
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

sdict = os.path.join(this_dir, 'candycrunch.pt')
sdict = torch.load(sdict, map_location = device)
sdict = {k.replace('module.', ''): v for k, v in sdict.items()}
candycrunch = CandyCrunch_CNN(2048, num_classes = len(glycans)).to(device)
candycrunch.load_state_dict(sdict)
candycrunch = candycrunch.eval()

mass_dict = dict(zip(mapping_file.composition, mapping_file["underivatized_monoisotopic"]))
modification_mass_dict = {'reduced': 1.0078, '2AA': 137.14, '2AB': 120.2}
abbrev_dict = {'S': 'Sulphate', 'P': 'Phosphate', 'Ac': 'Acetate'}
temperature = torch.Tensor([1.15]).to(device)


def T_scaling(logits, temperature):
    return torch.div(logits, temperature)


def process_mzML_stack(filepath, num_peaks = 1000,
                       ms_level = 2,
                       intensity = False):
    """function extracting all MS/MS spectra from .mzML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzML file
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | ms_level (int): which MS^n level to extract; default:2
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, and intensity if True
   """
    run = pymzml.run.Reader(filepath)
    highest_i_dict = {}
    rts, intensities, reducing_mass = [], [], []
    for spectrum in run:
        if spectrum.ms_level == ms_level:
            try:
                temp = spectrum.highest_peaks(2)
            except:
                continue
        mz_i_dict = {}
        num_actual_peaks = min(num_peaks, len(spectrum.peaks("raw")))
        for mz, i in spectrum.highest_peaks(num_actual_peaks):
            mz_i_dict[mz] = i
        if mz_i_dict:
            key = f"{spectrum.ID}_{spectrum.selected_precursors[0]['mz']}"
            highest_i_dict[key] = mz_i_dict
            reducing_mass.append(float(key.split('_')[-1]))
            rts.append(spectrum.scan_time_in_minutes())
        if intensity:
            inty = spectrum.selected_precursors[0].get('i', np.nan)
            intensities.append(inty)
        df_out = pd.DataFrame({
            'reducing_mass': reducing_mass,
            'peak_d': list(highest_i_dict.values()),
            'RT': rts,
            })
    if intensity:
        df_out['intensity'] = intensities
    return df_out


def process_mzXML_stack(filepath, num_peaks = 1000, ms_level = 2, intensity = False):
    """function extracting all MS/MS spectra from .mzXML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzXML file
   | num_peaks (int): max number of peaks to extract from spectrum; default:1000
   | ms_level (int): which MS^n level to extract; default:2
   | intensity (bool): whether to extract precursor ion intensity from spectra; default:False\n
   | Returns:
   | :-
   | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, and intensity if True
    """
    highest_i_dict = {}
    rts, intensities, reducing_mass = [], [], []
    with mzxml.read(filepath) as reader:
        for spectrum in reader:
            if spectrum['msLevel'] == ms_level:
                mz_array = spectrum['m/z array']
                intensity_array = spectrum['intensity array']
                num_peaks_to_extract = min(num_peaks, len(mz_array))
                mz_i_dict = {mz: i for mz, i in zip(mz_array[:num_peaks_to_extract], intensity_array[:num_peaks_to_extract])}
                if mz_i_dict:
                    precursor_mz = spectrum['precursorMz'][0]['precursorMz']
                    key = f"{spectrum['id']}_{precursor_mz}"
                    highest_i_dict[key] = mz_i_dict
                    reducing_mass.append(float(precursor_mz))
                    rts.append(spectrum['retentionTime'])
                    if intensity:
                        inty = spectrum['precursorMz'][0].get('precursorIntensity', np.nan)
                        intensities.append(inty)
    df_out = pd.DataFrame({
        'reducing_mass': reducing_mass, 
        'peak_d': list(highest_i_dict.values()), 
        'RT': rts,
        })
    if intensity:
        df_out['intensity'] = intensities
    return df_out


def average_dicts(dicts, mode = 'mean'):
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
            result[mass].append(intensity)
    if mode == 'mean':
        result = {mass: np.mean(intensities) for mass, intensities in result.items()}
    else:
        result = {mass: max(intensities) for mass, intensities in result.items()}
    return result


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
    binned_intensities  = np.zeros(num_frames)
    mz_diff = np.zeros(num_frames)
    mzs = np.array(list(peak_d.keys()), dtype = 'float32')
    intensities = np.array(list(peak_d.values()))
    bin_indices = np.digitize(mzs, frames, right = True)
    mz_remainder = mzs - frames[bin_indices - 1]
    unique_bins, summed_intensities = npi.group_by(bin_indices).sum(intensities)
    _, max_mz_remainder = npi.group_by(bin_indices).max(mz_remainder)
    binned_intensities[unique_bins - 1] = summed_intensities
    mz_diff[unique_bins - 1] = max_mz_remainder
    return binned_intensities , mz_diff


def normalize_dict(peak_d):
    keys, values = zip(*peak_d.items())
    values = np.array(values)
    normalized_values = values / values.sum()
    return dict(zip(keys, normalized_values))


def process_for_inference(keys, values, rts, num_spectra, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear', min_mz = 39.714, max_mz = 3000, bin_num = 2048):
    """processes averaged spectra for them being inputs to CandyCrunch\n
   | Arguments:
   | :-
   | keys (list): list of m/z values
   | values (list): list of spectra in form peak m/z:intensity
   | rts (list): list of retention times
   | num_spectra (list): list of number of spectra for each cluster
   | glycan_class (int): 0 = O-linked, 1 = N-linked, 2 = lipid/free
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | min_mz (float): minimal m/z used for binning; don't change; default:39.714
   | max_mz (float): maximal m/z used for binning; don't change; default:3000
   | bin_num (int): number of bins for binning; don't change; default: 2048\n
   | Returns:
   | :-
   | (1) a dataloader used for model prediction
   | (2) a preliminary df_out dataframe
   """
    data = {'reducing_mass': keys, 'RT': rts, 'peak_d': values, 'num_spectra': num_spectra}
    df = pd.DataFrame(data)
    df.assign(glycan_type = glycan_class,
              mode = (mode == 'negative').astype(int),
              lc = np.select([lc == 'PGC', lc == 'C18'], [0, 1], 2),
              modification = np.select([modification == 'reduced', modification == 'permethylated'], [0, 1], 2),
              trap = np.select([trap == 'linear', trap == 'orbitrap', trap == 'amazon'], [0, 1, 2], 3),
              inplace = True)
    # Intensity normalization
    df['peak_d'] = df['peak_d'].apply(normalize_dict)
    # Retention time normalization
    max_rt = max(max(df['RT']), 30)
    df['RT2'] = df['RT'] / max_rt
    # Intensity binning
    step = (max_mz - min_mz) / (bin_num - 1)
    frames = np.array([min_mz + step * i for i in range(bin_num)])
    binned_data = df['peak_d'].apply(lambda x: bin_intensities(x, frames))
    df[['binned_intensities', 'mz_remainder']] = pd.DataFrame(binned_data.tolist())
    # Dataloader generation
    X = list(zip(df.binned_intensities.values.tolist(), df.mz_remainder.values.tolist(), df.reducing_mass.values.tolist(), df.glycan_type.values.tolist(),
                 df.RT2.values.tolist(), df['mode'].values.tolist(), df.lc.values.tolist(), df.modification.values.tolist(), df.trap.values.tolist()))
    X = unwrap([[k]*5 for k in X])
    y = df['glycan'].repeat(5).reset_index(drop = True)
    dset = SimpleDataset(X, y, transform_mz = transform_mz, transform_prec = transform_prec, transform_rt = transform_rt)
    dloader = torch.utils.data.DataLoader(dset, batch_size = 256, shuffle = False)
    df.set_index('reducing_mass', inplace = True)
    drop_cols = ['reducing_mass', 'binned_intensities', 'mz_remainder', 'RT2', 'mode', 'modification', 'trap', 'glycan', 'glycan_type', 'lc']
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
    for data in dataloader:
        mz_list, mz_remainder, precursor, glycan_type, rt, mode, lc, modification, trap, y = data
        mz_list = torch.stack([mz_list, mz_remainder], dim = 1)
        batch_size = len(y)
        inputs = [mz_list, precursor, glycan_type, rt, mode, lc, modification, trap]
        inputs = [x.to(device) for x in inputs]
        pred = model(*inputs)
        if temp:
            pred = T_scaling(pred, temperature)
        pred = F.softmax(pred, dim = 1)
        pred = pred.cpu().detach().numpy()
        idx_topk = np.argsort(pred, axis = 1)[:, ::-1][:, :k]
        conf_topk = -np.sort(-pred)[:, :k]
        end_idx = start_idx + batch_size
        preds[start_idx:end_idx, :] = idx_topk
        conf[start_idx:end_idx, :] = conf_topk
        start_idx = end_idx
    preds = [[glycans[i] for i in j] for j in preds]
    return preds, conf.tolist()


def assign_dict_labels(dicty):
    """bin dictionary items according to key values\n
   | Arguments:
   | :-
   | dicty (dict): dictionary of form m/z : spectrum\n
   | Returns:
   | :-
   | Returns a list of labels used for binning/grouping the dict items
   """
    k = 0
    labels = []
    prev_key = None
    for curr_key in dicty.keys():
        if prev_key is not None and abs(curr_key - prev_key) < 0.5:
            labels.append(k)
        else:
            k += 1
            labels.append(k)
        prev_key = curr_key
    return labels


def determine_threshold(m):
    """determine appropriate fluctuation threshold of m/z\n
   | Arguments:
   | :-
   | m (float): an m/z value\n
   | Returns:
   | :-
   | Returns the allowed error (float) to establish equality for m
   """
    return 0.5 if m < 1500 else 0.75


def mass_check(mass, glycan, libr = None, mode = 'negative', modification = 'reduced', mass_tag = None,
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500):
    """determine whether glycan could explain m/z\n
   | Arguments:
   | :-
   | mass (float): observed m/z
   | glycan (string): glycan in IUPAC-condensed nomenclature
   | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB, or 'custom'; default:'reduced'
   | mass_tag (float): label mass to add when calculating possible m/z if modification == 'custom'; default:None
   | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
   | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1500
   | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500\n
   | Returns:
   | :-
   | Returns True if glycan could explain mass and False if not
   """
    if libr is None:
        libr = lib
    try:
        mz = glycan_to_mass(glycan, sample_prep='permethylated' if modification == 'permethylated' else None)
    except:
        return False
    if modification in modification_mass_dict:
        mz += modification_mass_dict[modification]
    if modification == 'custom':
        mz += mass_tag
    adduct_mass = mass_dict['Acetate'] if mode == 'negative' else mass_dict['Na+']
    og_list = [mz, mz + adduct_mass]
    mz_list = og_list.copy()
    thresh = determine_threshold(mass)
    for z, threshold, charge_adjust in zip(
        [2, 3, 4],
        [double_thresh, triple_thresh, quadruple_thresh],
        [-0.5, -0.66, -0.75] if mode == 'negative' else [0.5, 0.66, 0.75]
        ):
        if mz > threshold:
            mz_list += [(m / z + charge_adjust) for m in og_list]
    return [m for m in mz_list if abs(mass - m) < thresh]


def break_alert(t):
    """split list if there is a discontinuity\n
   | Arguments:
   | :-
   | t (list): list of ordered floats (retention times in this case)\n
   | Returns:
   | :-
   | Returns two lists, that may be split by a discontinuity of > 0.5
   """
    for k in range(len(t)-1):
        try:
            temp = t[k+1][0] - t[k][0]
        except:
            temp = t[k+1] - t[k]
        if temp > 0.5:
            return t[:k+1], t[k+1:]
    return (t,)


def break_alert_loop(t):
    """iteratively split list if there is a discontinuity\n
   | Arguments:
   | :-
   | t (list): list of ordered floats (retention times in this case)\n
   | Returns:
   | :-
   | (1) nested lists of grouped retention times, split by discontinuities
   | (2) list of length of each group of retention times
   """
    out = []
    out_len = []
    temp = break_alert(t)
    out.append(temp[0])
    out_len.append(len(temp[0]))
    while len(temp) > 1:
        temp = break_alert(temp[1])
        out.append(temp[0])
        out_len.append(len(temp[0]))
    return out, out_len


def get_rep_spectra(rt_label_in, intensity = False):
    """select representative spectra from the middle of a peak\n
   | Arguments:
   | :-
   | rt_label_in (list): list of retention times (float)
   | intensity (bool): whether to use intensity for relative abundance estimation; default: False\n
   | Returns:
   | :-
   | (1) list of indices for the respective representative spectra
   | (2) nested lists of indices for each spectrum group, to group their intensities, if intensity=True; else empty list
   | (3) list of length of each group of retention times (i.e., number of spectra)
   """
    if len(rt_label_in) <= 1:
        return [rt_label_in[0]], [], [1] if len(rt_label_in) == 1 else [], [], []
    rt_label = sorted(rt_label_in)
    X = np.array(rt_label).reshape(-1, 1)
    clustering = AgglomerativeClustering(n_clusters = None,distance_threshold = 0.5).fit(X)
    unq_labels = set(clustering.labels_)
    grouped_rt = [
        [rt_label[i] for i, label in enumerate(clustering.labels_) if label == unique_label]
        for unique_label in unique_labels
        ]
    group_lengths = [len(group) for group in grouped_rt]
    median_indices = [
        group.index(np.percentile(group, 50, method = 'nearest'))
        for group in grouped_rt
        ]
    median_indices_in_input = [
        rt_label_in.index(grouped_rt[i][median_idx])
        for i, median_idx in enumerate(median_indices)
        ]
    if intensity:
        intensity_indices = [
            [rt_label_in.index(rt) for rt in group]
            for group in grouped_rt
            ]
        return median_indices_in_input, intensity_indices, group_lengths
    return median_indices_in_input, [], group_lengths


def build_mean_dic(dicty, rt, intensity):
    """organizes a spectra cluster into an averaged spectrum\n
   | Arguments:
   | :-
   | dicty (dict): dictionary of form m/z : spectrum
   | rt (list): list of retention times
   | intensity (list): list of intensities\n
   | Returns:
   | :-
   | (1) list of peak m/z
   | (2) list of spectra, of form peak m/z:intensity
   | (3) list of retention times
   | (4) list of number of spectra per spectra cluster
   | (5) (optional) list of intensities for each spectra cluster
   """
    inty_check = len(intensity) > 0
    # Sort in preparation for discontinuity check; bring everything in same order defined by sort_idx
    sort_idx = np.argsort(list(dicty.keys()))
    dicty = dict(sorted(dicty.items()))
    # Detect mass groups of spectra by scanning for m/z discontinuities > 0.5
    labels = assign_dict_labels(dicty)
    unique_labels = set(labels)
    rt = np.array(rt)[sort_idx]
    grouped_rt = [rt[labels == label] for label in unique_labels]
    # Detect retention groups of mass groups & retrieve indices representative spectra
    rt_idx, inty_idx, num_spectra = list(zip(*[get_rep_spectra(k, intensity = True) for k in grouped_rt]))
    num_spectra = unwrap(num_spectra)
    if inty_check:
        intensity = np.array(intensity)[sort_idx]
        intensity = [intensity[labels == label] for label in unique_labels]
        intensity = [[[intensity[k][z] for z in j] if isinstance(j, list) else [intensity[k][j]] for j in inty_idx[k]] for k in range(len(intensity))]
        intensity = unwrap([[sum(j) for j in k] if isinstance(k[0], list) else [k] for k in intensity])
    # Get m/z, predictions, and prediction confidence of those representative spectra
    keys = [[list(dicty.keys())[k] for k in range(len(labels)) if labels[k] == j] for j in unique_labels]
    keys = [[[keys[k][z] for z in j] if isinstance(j, list) else [keys[k][j]] for j in inty_idx[k]] for k in range(len(keys))]
    keys = unwrap([[np.min(j) for j in k] if isinstance(k[0], list) else [k] for k in keys])
    values = [[list(dicty.values())[k] for k in range(len(labels)) if labels[k] == j] for j in unique_labels]
    rep_values = unwrap([[values[k][j] for j in rt_idx[k]] for k in range(len(values))])
    values = [[[values[k][z] for z in j] if isinstance(j, list) else [values[k][j]] for j in inty_idx[k]] for k in range(len(values))]
    values = unwrap([[average_dicts(j) for j in k] if isinstance(k[0], list) else [k] for k in values])
    rts = [[[rt_labels[k][z] for z in j] if isinstance(j, list) else [rt_labels[k][j]] for j in inty_idx[k]] for k in range(len(rt_labels))]
    rts = unwrap([[np.mean(j) for j in k] if isinstance(k[0], list) else [k] for k in rts])
    return keys, values, rts, num_spectra, rep_values, intensity if inty_check else []


def deduplicate_predictions(df):
    """removes/unifies duplicate predictions\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference\n
   | Returns:
   | :-
   | Returns a deduplicated dataframe
   """
    # Keep track of abundances if relevant
    if 'rel_abundance' in df.columns:
        struc_abundance = df.apply(lambda row: (row['predictions'][0][0] if row['predictions'] else "ID" + str(row.name), row['rel_abundance']), axis = 1).to_dict()
    drop_idx = []
    pred_list = df['predictions'].tolist()
    index_list = df.index.tolist()
    for k in range(len(df)-1):
        # Basically, if we have the same structure with same mass & charge state twice or more --> just go with the most confident one
        if k not in drop_idx and pred_list[k]:
            check = [abs(index_list[k]-index_list[j]) < determine_threshold(index_list[k])+0.5 and pred_list[k][0][0] == pred_list[j][0][0] if len(pred_list[j]) > 0 else False for j in range(len(df))]
            if any(check):
                idx = np.where(np.array(check) == True)[0]
                winner = idx[np.argmax([pred_list[i][0][1] for i in idx])]
                rest = [i for i in idx if i != winner]
                if rest:
                    drop_idx.extend(rest)
                    if 'rel_abundance' in df.columns:
                        if 'rel_abundance' in df.columns and pred_list[k]:
                            key = pred_list[k][0][0]
                            struc_abundance[key] = sum(df.loc[idx, 'rel_abundance'])
    drop_idx = set(drop_idx)
    df.drop(index = [index_list[k] for k in drop_idx], inplace = True)
    if 'rel_abundance' in df.columns:
        df['rel_abundance'] = df.apply(lambda row: struc_abundance.get(row['predictions'][0][0]) if row['predictions'] else struc_abundance.get("ID" + str(row.name)), axis = 1)
    return df


def combinatorics(comp):
    """given a composition, create a crude approximation of possible B/C/Y/Z fragments\n
   | Arguments:
   | :-
   | comp (dict): composition in dictionary form\n
   | Returns:
   | :-
   | Returns a list of rough masses to check against fragments
   """
    clist = unwrap([[k]*v for k, v in comp.items()])
    verbose_clist = [abbrev_dict[x] if x in abbrev_dict else x for x in clist]
    masses = set()
    for i in range(1, len(verbose_clist) + 1):
        for comb in combinations(verbose_clist, i):
            mass = sum(mass_dict.get(k, 0) for k in comb)
            masses.add(mass)
            masses.add(mass + 18.01056)
    return list(masses)


def domain_filter(df_out, glycan_class, libr = None, mode = 'negative', modification = 'reduced',
                  mass_tolerance = 0.5, filter_out = set(), df_use = None):
    """filters out false-positive predictions\n
   | Arguments:
   | :-
   | df_out (dataframe): df_out generated within wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
   | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:None
   | df_use (dataframe): glycan database used to check whether compositions are valid; default: df_glycan\n
   | Returns:
   | :-
   | Returns a filtered prediction dataframe
   """
    if df_use is None:
        df_use = df_glycan
    if libr is None:
        libr = lib
    reduced = 1.0078 if modification == 'reduced' else 0
    multiplier = -1 if mode == 'negative' else 1
    df_out = adduct_detect(df_out, mode, modification)
    for k in range(len(df_out)):
        keep = []
        addy = df_out['charge'].iloc[k]*multiplier-1
        c = abs(df_out['charge'].iloc[k])
        assumed_mass = df_out.index[k]*c + addy
        cmasses = np.array(combinatorics(df_out['composition'].iloc[k]))
        current_preds = df_out['predictions'].iloc[k] if len(df_out['predictions'].iloc[k]) > 0 else [''.join(list(df_out['composition'].iloc[k].keys()))]
        to_append = len(df_out['predictions'].iloc[k]) > 0
        # Check whether it's a glycan spectrum
        top_fragments = np.array(df_out['top_fragments'].iloc[k][:10])
        if not np.any(np.abs(top_fragments[:, None] - cmasses) < 1.5):
            df_out.at[k, 'predictions'] = ['remove']
            continue
        for i, m in enumerate(current_preds):
            m = m[0]
            truth = [True]
            # Check diagnostic ions
            if 'Neu5Ac' in m:
                truth.append(any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Neu5Ac']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Neu5Ac']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j, float)]))
            if 'Neu5Gc' in m:
                truth.append(any([abs(mass_dict['Neu5Gc']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Neu5Gc']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Neu5Gc']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j, float)]))
            if 'Kdn' in m:
                truth.append(any([abs(mass_dict['Kdn']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Kdn']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Kdn']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j, float)]))
            if 'Neu5Gc' not in m:
                truth.append(not any([abs(mass_dict['Neu5Gc']+(1.0078*multiplier)-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j, float)]))
            if 'Neu5Ac' not in m and 'Neu5Gc' not in m:
                truth.append(not any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j, float)]))
            if 'Neu5Ac' not in m and (m.count('Fuc') + m.count('dHex') > 1):
                truth.append(not any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 1 or abs(df_out.index.tolist()[k]-mass_dict['Neu5Ac']-j) < 1 for j in df_out.top_fragments.values.tolist()[k][:10] if isinstance(j, float)]))
            if 'S' in m and len(df_out.predictions.values.tolist()[k]) < 1:
                truth.append(any(['S' in (mz_to_composition(t, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class,
                                  df_use = df_use, filter_out = filter_out, reduced = reduced>0)[0:1] or ({},))[0].keys() for t in df_out.top_fragments.values.tolist()[k][:20]]))
            # Check fragment size distribution
            if c > 1:
                truth.append(any([j > df_out.index.values[k]*1.2 for j in df_out.top_fragments.values.tolist()[k][:15]]))
            if c == 1:
                truth.append(all([j < df_out.index.values[k]*1.1 for j in df_out.top_fragments.values.tolist()[k][:5]]))
            if len(df_out.top_fragments.values.tolist()[k]) < 5:
                truth.append(False)
            # Check M-adduct for adducts
            if isinstance(df_out.adduct.values.tolist()[k], str):
                truth.append(any([abs(df_out.index.tolist()[k]-mass_dict[df_out.adduct.values.tolist()[k]]-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:10]]))
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
            df_out.at[k, 'predictions'] = keep
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
    masses = df.index.values * np.abs(charges) + (np.abs(charges) - 1)
    for k in range(len(df)):
        if not len(predictions[k]) > 0:
            target_mass = masses[k]
            target_RT = RTs[k]
            target_composition = compositions[k]
            mass_diffs = np.abs(masses - target_mass)
            RT_diffs = np.abs(RTs - target_RT)
            same_compositions = compositions == target_composition
            idx = np.where((mass_diffs < determine_threshold(masses)) & (RT_diffs < 1) & same_compositions)[0]
            if len(idx) > 0:
                df.iat[k, 0] = predictions[idx[0]]
    return df


def adduct_detect(df, mode, modification):
    """checks which spectra contains adducts and records them\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', or 'other'/'none'\n
   | Returns:
   | :-
   | Returns adduct-filled dataframe
   """
    adduct = 'Acetate' if mode == 'negative' else 'Na+'
    adduct_mass = mass_dict[adduct]
    if modification == 'reduced':
        adduct_mass += 1.0078
    compositions = df['composition'].values
    charges = df['charge'].values
    indices = df.index.values
    computed_masses = np.array([composition_to_mass(composition) for composition in compositions])
    observed_masses = indices * np.abs(charges) + (np.abs(charges) - 1)
    adduct_check = np.abs(computed_masses + adduct_mass - observed_masses) < 0.5
    df['adduct'] = np.where(adduct_check, adduct, np.nan)
    return df


def average_preds(preds, conf, k = 5):
    """takes in data-augmentation based prediction variants and averages them\n
   | Arguments:
   | :-
   | preds (list): nested list of predictions (glycan strings) for each spectrum cluster
   | conf (list): nested list of prediction confidences (floats) for each spectrum cluster
   | k (int): how many predictions should be averaged for one cluster; default:5, do not change lightly\n
   | Returns:
   | :-
   | Returns averaged predictions and averaged prediction confidences
   """
    pred_chunks = [preds[i:i + k] for i in range(0, len(preds), k)]
    conf_chunks = [conf[i:i + k] for i in range(0, len(conf), k)]
    out_p, out_c = [], []
    for this_pred, this_conf in zip(pred_chunks, conf_chunks):
        combs = [{pred: conf for pred, conf in zip(chunk_pred, chunk_conf)} for chunk_pred, chunk_conf in zip(this_pred, this_conf)]
        combs = average_dicts(combs, mode = 'max')
        combs = dict(sorted(combs.items(), key = lambda x: x[1], reverse = True))
        out_p.append(list(combs.keys()))
        out_c.append(list(combs.values()))
    return out_p, out_c


def impute(df_out):
    """searches for specific isomers that could be added to the prediction dataframe\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference\n
   | Returns:
   | :-
   | Returns prediction dataframe with imputed predictions (if possible)
   """
    predictions_list = df_out.predictions.values.tolist()
    index_list = df_out.index.tolist()
    charge_list = df_out.charge.values.tolist()
    composition_list = df_out.composition.values.tolist()
    for k in range(len(df_out)):
        pred_k = predictions_list[k]
        index_k = index_list[k]
        charge_k = charge_list[k]
        if len(pred_k) > 0:
            for j in range(len(df_out)):
                pred_j = predictions_list[j]
                index_j = index_list[j]
                charge_j = charge_list[j]
                composition_j = composition_list[j]
                if len(pred_j) < 1:
                    charge_max_k = max([abs(charge_k), 1])
                    # Condition for 'Neu5Gc'
                    if abs(index_k + (16.0051 / charge_max_k) - index_j) < 0.5 and 'Neu5Gc' in composition_j.keys():
                        df_out.iat[j, 0] = [(m[0].replace('Neu5Ac', 'Neu5Gc', 1),) for m in pred_k]
                    # Condition for 'Neu5Ac'
                    elif abs(index_k - (16.0051 / charge_max_k) - index_j) < 0.5 and 'Neu5Ac' in composition_j.keys():
                        df_out.iat[j, 0] = [(m[0].replace('Neu5Gc', 'Neu5Ac', 1),) for m in pred_k]
    return df_out


def possibles(df_out, mass_dic, reduced):
    """searches for known glycans that could explain the observed m/z value if we don't have a prediction there\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference
   | mass_dic (dict): dictionary of form mass : list of glycans
   | reduced (int): 1 if modification = 'reduced' and 0 otherwise\n
   | Returns:
   | :-
   | Returns prediction dataframe with imputed predictions (if possible)
   """
    predictions_list = df_out.predictions.values.tolist()
    index_list = df_out.index.tolist()
    mass_keys = np.array(list(mass_dic.keys()))
    for k in range(len(df_out)):
        if len(predictions_list[k]) < 1:
            check_mass = index_list[k] - reduced
            diffs = np.abs(mass_keys - check_mass)
            min_diff_index = np.argmin(diffs)
        if diffs[min_diff_index] < 0.5:
            possible = mass_dic[mass_keys[min_diff_index]]
            df_out.iat[k, 0] = [(m,) for m in possible]
    return df_out


def make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = 'Mammalia'):
    """generates a mass dict that can be used in the possibles() function\n
   | Arguments:
   | :-
   | glycans (list): glycans used for training CandyCrunch
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | taxonomy_class (string): which taxonomic class to use for selecting possible glycans; default:'Mammalia'\n
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
            composition = glycan_to_composition(k)
            if not filter_out.intersection(composition.keys()):
                mass = glycan_to_mass(k)
                mass_dic.setdefault(mass, []).append(k)
            else:
                mass_dic[9999].append(k)
        except:
            mass_dic[9999].append(k)
    return mass_dic


def canonicalize_biosynthesis(df_out, libr, pred_thresh):
    """regularize predictions by incentivizing biosynthetic feasibility\n
   | Arguments:
   | :-
   | df_out (dataframe): prediction dataframe generated within wrap_inference
   | libr (list): library of monosaccharides
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01\n
   | Returns:
   | :-
   | Returns prediction dataframe with re-ordered predictions, based on observed biosynthetic activities
   """
    df_out['true_mass'] = df_out.index * abs(df_out['charge']) - (df_out['charge'] + np.sign(df_out['charge'].sum()))
    df_out.sort_values(by = 'true_mass', inplace = True)
    strong_evidence_preds = [pred[0][0] for pred, evidence in zip(df_out['predictions'], df_out['evidence']) if len(pred) > 0 and evidence == 'strong']
    rest_top1 = set(strong_evidence_preds)
    for k, row in df_out[::-1].iterrows():
        preds = row['predictions']
        if len(preds) == 0:
            continue
        new_preds = []
        for p in preds:
            p_list = list(p)
            if len(p_list) == 1:
                p_list.append(0)
            p_list[1] += 0.1 * sum(subgraph_isomorphism(p_list[0], t, libr = libr, wildcards_ptm = True) for t in rest_top1 if t != p_list[0])
            new_preds.append(tuple(p_list))
        new_preds.sort(key = lambda x: x[1], reverse = True)
        total = sum(p[1] for p in new_preds)
        if total > 1:
            new_preds = [(p[0], p[1] / total) for p in new_preds][:5]
        else:
            new_preds = new_preds[:5]
        df_out.at[k, 'predictions'] = new_preds
    df_out.drop(['true_mass'], axis = 1, inplace = True)
    return df_out.loc[df_out.index.sort_values(), :]


def load_spectra_filepath(spectra_filepath):
    if spectra_filepath.endswith(".mzML"):
        return process_mzML_stack(spectra_filepath, intensity = True)
    if spectra_filepath.endswith(".mzXML"):
        return process_mzXML_stack(spectra_filepath, intensity = True)
    if spectra_filepath.endswith(".xlsx"):
        loaded_file = pd.read_excel(spectra_filepath)
        mask = loaded_file['peak_d'].str.endswith('}', na = False)
        loaded_file = loaded_file[mask]
        loaded_file['peak_d'] = loaded_file['peak_d'].apply(ast.literal_eval)
        return loaded_file
    raise FileNotFoundError('Incorrect filepath or extension, please ensure it is in the intended directory and is one of the supported formats')


def calculate_ppm_error(theoretical_mass,observed_mass):
    ppm_error = ((theoretical_mass-observed_mass)/theoretical_mass)* (10**6)
    return ppm_error


def wrap_inference(spectra_filepath, glycan_class, model = candycrunch, glycans = glycans, libr = None, bin_num = 2048,
                   frag_num = 100, mode = 'negative', modification = 'reduced', mass_tag = None, lc = 'PGC', trap = 'linear',
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False, mass_tolerance = 0.5, extra_thresh = 0.2,
                   filter_out = {'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True, experimental = True, mass_dic = None,
                   taxonomy_class = 'Mammalia', df_use = None):
    """wrapper function to get & curate CandyCrunch predictions\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".mzML",".mzXML", or ".xlsx" pointing to a file containing spectra or preprocessed spectra 
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'; default:'reduced'
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
   | temperature (float): the temperature factor used to calibrate logits; default:1.15
   | spectra (bool): whether to also output the actual spectra used for prediction; default:False
   | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
   | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
   | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); default:0.2
   | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:{'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}
   | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
   | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
   | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
   | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia'
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan\n
   | Returns:
   | :-
   | Returns dataframe of predictions for spectra in file
   """
    print(f"Your chosen settings are: {glycan_class} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if libr is None:
        libr = lib
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[(df_glycan.glycan_type==glycan_class) & (df_glycan.Class.str.contains(taxonomy_class))])
        df_use['Composition'] = df_use['Composition'].apply(ast.literal_eval)
    reduced = 1.0078 if modification == 'reduced' else 0
    multiplier = -1 if mode == 'negative' else 1
    loaded_file = load_spectra_filepath(spectra_filepath)
    intensity = 'intensity' in loaded_file.columns and not (loaded_file['intensity'] == 0).all() and not loaded_file['intensity'].isnull().all()
    # Prepare file for processing
    loaded_file.dropna(subset = ['peak_d'], inplace = True)
    loaded_file['reducing_mass'] += np.random.uniform(0.00001, 10**(-20), size = len(loaded_file))
    inty = loaded_file['intensity'].values if intensity else []
    coded_class = {'O':0,'N':1,'free':2,'lipid':2}[glycan_class]
    spec_dic = {mass: peak for mass, peak in zip(loaded_file['reducing_mass'].values, loaded_file['peak_d'].values)}
    # Group spectra by mass/retention isomers and process them for being inputs to CandyCrunch
    keys, values, RT, num_spectra, rep_values, intensity = build_mean_dic(spec_dic, loaded_file.RT.values.tolist(), inty)
    loader, df_out = process_for_inference(keys, values, RT, num_spectra, coded_class, mode = mode, modification = modification, lc = lc, trap = trap, bin_num = bin_num)
    df_out['peak_d'] = rep_values
    # Predict glycans from spectra
    preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
    preds, pred_conf = average_preds(preds, pred_conf)
    if intensity:
        df_out['rel_abundance'] = intensity
    df_out['predictions'] = [[(pred, conf) for pred, conf in zip(pred_row, conf_row)] for pred_row, conf_row in zip(preds, pred_conf)]
    # Check correctness of glycan class & mass
    df_out['predictions'] = [
        [(g[0], round(g[1], 4)) for g in preds if
         enforce_class(g[0], glycan_class, g[1], extra_thresh = extra_thresh) and
         g[1] > pred_thresh and
         mass_check(mass, g[0], libr = libr, modification = modification, mass_tag = mass_tag, mode = mode)][:5]
        for preds, mass in zip(df_out.predictions, df_out.index)
        ]
    # Get composition of predictions
    df_out['composition'] = [glycan_to_composition(g[0][0]) if g and g[0] else np.nan for g in df_out.predictions]
    df_out.composition = [
        k if isinstance(k, dict) else mz_to_composition(m, mode = mode, mass_tolerance = mass_tolerance,
                                                        glycan_class = glycan_class, df_use = df_use, filter_out = filter_out,
                                                        reduced = reduced > 0)
        for k, m in zip(df_out['composition'], df_out.index)
        ]
    df_out.composition = [np.nan if isinstance(k, list) and not k else k[0] if isinstance(k, list) and len(k) > 0 else k for k in df_out['composition']]
    df_out.dropna(subset = ['composition'], inplace = True)
    # Calculate precursor ion charge
    df_out['charge'] = [
        round(composition_to_mass(comp) / idx) * multiplier
        for comp, idx in zip(df_out['composition'], df_out.index)
        ]
    df_out['RT'] = df_out['RT'].round(2)
    cols = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d']
    if intensity:
        cols.append('rel_abundance')
    df_out = df_out[cols]
    # Fill up possible gaps of singly-/doubly-charged spectra of the same structure
    df_out = backfill_missing(df_out)
    # Extract & sort the top 100 fragments by intensity
    df_out['top_fragments'] = [
        [round(frag[0], 4) for frag in sorted(peak_d.items(), key = lambda x: x[1], reverse = True)[:frag_num]]
        for peak_d in df_out['peak_d']
        ]
    # Filter out wrong predictions via diagnostic ions etc.
    df_out = domain_filter(df_out, glycan_class, libr = libr, mode = mode, filter_out = filter_out,
                           modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
    # Deduplicate identical predictions for different spectra
    df_out = deduplicate_predictions(df_out)
    df_out['evidence'] = ['strong' if preds else np.nan for preds in df_out['predictions']]
    # Construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
    if supplement:
        try:
            df_out = supplement_prediction(df_out, glycan_class, libr = libr, mode = mode, modification = modification, mass_tag = mass_tag)
            df_out['evidence'] = [
                'medium' if pd.isna(evidence) and preds else evidence
                for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
                ]
        except:
            pass
    # Check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
    if experimental:
        df_out = impute(df_out)
        mass_dic = mass_dic if mass_dic else make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = taxonomy_class)
        df_out = possibles(df_out, mass_dic, reduced)
        df_out['evidence'] = [
            'weak' if pd.isna(evidence) and preds else evidence
            for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
            ]
    # Filter out wrong predictions via diagnostic ions etc.
    if supplement or experimental:
        df_out = domain_filter(df_out, glycan_class, libr = libr, mode = mode, filter_out = filter_out, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
        df_out['predictions'] = [[(k[0].replace('-ol', '').replace('1Cer', ''), k[1]) if len(k) > 1 else (k[0].replace('-ol', '').replace('1Cer', ''),) for k in j] if j else j for j in df_out['predictions']]
    # Keep or remove spectra that still lack a prediction after all this
    if not get_missing:
        df_out = df_out[df_out['predictions'].str.len() > 0]
    # Reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
    df_out = canonicalize_biosynthesis(df_out, libr, pred_thresh)
    spectra_out = df_out.pop('peak_d').values.tolist()
    # Calculate  ppm error
    valid_indices, ppm_errors = [], []
    for preds, obs_mass in zip(df_out['predictions'], df_out.index):
        theo_mass = mass_check(obs_mass, preds[0][0], modification=modification, mass_tag=mass_tag, mode=mode)
        if theo_mass:
            valid_indices.append(True)
            ppm_errors.append(abs(calculate_ppm_error(theo_mass[0], obs_mass)))
        else:
            valid_indices.append(False)
    df_out = df_out[valid_indices]
    df_out['ppm_error'] = ppm_errors
    # Clean-up
    df_out['composition'] = [glycan_to_composition(k[0][0]) if k else val for k, val in zip(df_out['predictions'], df_out['composition'])]
    df_out['charge'] = round(df_out['composition'].apply(composition_to_mass) / df_out.index) * multiplier
    # Normalize relative abundances if relevant
    if intensity:
        df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
    return (df_out, spectra_out) if spectra else df_out


def supplement_prediction(df_in, glycan_class, libr = None, mode = 'negative', modification = 'reduced',mass_tag=None):
    """searches for biosynthetic precursors of CandyCrunch predictions that could explain peaks\n
   | Arguments:
   | :-
   | df_in (pandas dataframe): output file produced by wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', or 'other'/'none'; default:'reduced'\n
   | Returns:
   | :-
   | Returns dataframe with supplemented predictions based on biosynthetic network
   """
    if libr is None:
        libr = lib
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
        libr = expand_lib(libr, preds)
    net = construct_network(preds, permitted_roots = permitted_roots, libr = libr)
    if glycan_class == 'free':
        net = evoprune_network(net, libr = libr)
    unexplained_idx = [idx for idx, pred in enumerate(df['predictions']) if not pred]
    unexplained = df.index[unexplained_idx].tolist()
    preds_set = set(preds)
    new_nodes = [k for k in net.nodes() if k not in preds_set]
    explained_idx = [[unexplained_idx[k] for k in np.where([mass_check(j, node,
                                                                       modification = modification, mode = mode, mass_tag = mass_tag) for j in unexplained])[0]] for node in new_nodes]
    new_nodes = [(node, idx) for node, idx in zip(new_nodes, explained_idx) if idx]
    explained = {k: [] for k in set(unwrap(explained_idx))}
    for node, indices in new_nodes:
        for index in indices:
            explained[index].append(node)
    pred_idx = df.columns.get_loc('predictions')
    for index, values in explained.items():
        df.iat[index, pred_idx] = [(t,) for t in values]
    return df

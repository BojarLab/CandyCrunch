import ast
import copy
import operator
import os
import re
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
from glycowork.motif.processing import enforce_class
from glycowork.motif.tokenization import (composition_to_mass,
                                          glycan_to_composition,
                                          glycan_to_mass, mapping_file,
                                          mz_to_composition)
from glycowork.network.biosynthesis import construct_network, evoprune_network
from pyteomics import mzxml

from CandyCrunch.model import (CandyCrunch_CNN, SimpleDataset, transform_mz,
                               transform_prec, transform_rt)

this_dir, this_filename = os.path.split(__file__)
data_path = os.path.join(this_dir, 'glycans.pkl')
glycans = pickle.load(open(data_path, 'rb'))
data_path = os.path.join(this_dir, 'glytoucan_mapping.pkl')
glytoucan_mapping = pickle.load(open(data_path, 'rb'))

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
    # Sort the highest_i_dict by values
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key = lambda x: x[1], reverse = True))
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
    # Sort the highest_i_dict by values
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key = lambda x: x[1], reverse = True))
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


def process_for_inference(df, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear'):
    """processes averaged spectra for them being inputs to CandyCrunch\n
   | Arguments:
   | :-
   | df (dataframe): condensed dataframe from condense_dataframe
   | glycan_class (int): 0 = O-linked, 1 = N-linked, 2 = lipid/free
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'\n
   | Returns:
   | :-
   | (1) a dataloader used for model prediction
   | (2) a preliminary df_out dataframe
   """
    df = df.assign(glycan_type = glycan_class,
              mode = int(mode == 'negative'),
              lc = np.select([lc == 'PGC', lc == 'C18'], [0, 1], 2),
              modification = np.select([modification == 'reduced', modification == 'permethylated'], [0, 1], 2),
              trap = np.select([trap == 'linear', trap == 'orbitrap', trap == 'amazon'], [0, 1, 2], 3))
    df['glycan'] = [0]*len(df)
    # Retention time normalization
    max_rt = max(max(df['RT']), 30)
    df['RT2'] = df['RT'] / max_rt
    # Dataloader generation
    X = list(zip(df.binned_intensities.values.tolist(), df.mz_remainder.values.tolist(), df.reducing_mass.values.tolist(), df.glycan_type.values.tolist(),
                 df.RT2.values.tolist(), df['mode'].values.tolist(), df.lc.values.tolist(), df.modification.values.tolist(), df.trap.values.tolist()))
    y  = df['glycan']
    if device != 'cpu':
        X = unwrap([[k]*5 for k in X])
        y = y.repeat(5).reset_index(drop = True)
    dset = SimpleDataset(X, y, transform_mz = transform_mz, transform_prec = transform_prec, transform_rt = transform_rt)
    dloader = torch.utils.data.DataLoader(dset, batch_size = 256, shuffle = False)
    df.set_index('reducing_mass', inplace = True)
    drop_cols = ['binned_intensities', 'mz_remainder', 'RT2', 'mode', 'modification', 'trap', 'glycan', 'glycan_type', 'lc']
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


def mass_check(mass, glycan, mode = 'negative', modification = 'reduced', mass_tag = 0,
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500, mass_thresh = 0.5):
    """determine whether glycan could explain m/z\n
   | Arguments:
   | :-
   | mass (float): observed m/z
   | glycan (string): glycan in IUPAC-condensed nomenclature
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB, or 'custom'; default:'reduced'
   | mass_tag (float): label mass to add when calculating possible m/z if modification == 'custom'; default:0
   | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
   | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1500
   | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500
   | mass_thresh (float): maximum allowed mass difference to return True; default:0.5\n
   | Returns:
   | :-
   | Returns True if glycan could explain mass and False if not
   """
    try:
        mz = glycan_to_mass(glycan, sample_prep= modification if modification in ["permethylated", "peracetylated"] else 'underivatized') if isinstance(glycan, str) else glycan
    except:
        return False
    mz += modification_mass_dict.get(modification, mass_tag)
    adduct_list = ['Acetonitrile', 'Acetate', 'Formate'] if mode == 'negative' else ['Na+', 'K+', 'NH4+']
    og_list = [mz] + [mz + mass_dict.get(adduct, 999) for adduct in adduct_list]
    charge_adjustments = [-0.5, -0.66, -0.75] if mode == 'negative' else [0.5, 0.66, 0.75]
    thresholds = [double_thresh, triple_thresh, quadruple_thresh]
    mz_list = og_list + [
        (m / z + charge_adjust) for z, threshold, charge_adjust in zip([2, 3, 4], thresholds, charge_adjustments)
        for m in og_list if m > threshold
    ]
    return [m for m in mz_list if abs(mass - m) < mass_thresh]


def normalize_array(input_array):
    array_sum = input_array.sum()
    return input_array / array_sum


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
    # Initialize an empty dictionary to hold the results
    results_dict = {}
    clusters= []
    # Sort the dataframe by 'reducing_mass' and 'RT'
    df['rounded_reducing_mass'] = np.round(df['reducing_mass'] * 2) / 2
    df['rounded_RT'] = np.round(df['RT'], 1)
    df.sort_values(by = ['rounded_reducing_mass', 'rounded_RT'], inplace = True)
    df.drop(['rounded_reducing_mass', 'rounded_RT'], axis = 1, inplace = True)
    # Initialize the first cluster
    first_row = df.iloc[0]
    clusters.append({
        'reducing_mass': [first_row['reducing_mass']],
        'RT': [first_row['RT']],
        'intensity': [first_row['intensity']],
        'peak_d': [first_row['peak_d']],
        'max_intensity': [first_row['intensity']]
    })

    # Loop through the sorted dataframe starting from the second row
    for _, row in df.iloc[1:].iterrows():
        rm = row['reducing_mass']
        rt = row['RT']
        intensity = row['intensity']
        peak_d = row['peak_d']
        found = False
        for cluster in clusters:
            last_max = cluster['max_intensity'][-1]
            if last_max > 0:
                idx = np.argmax(cluster['max_intensity'])
                last_rm, last_rt = cluster['reducing_mass'][idx], cluster['RT'][idx]
            else:
                last_rm, last_rt = cluster['reducing_mass'][-1], cluster['RT'][-1]
            if abs(last_rm - rm) <= mz_diff and abs(last_rt - rt) <= rt_diff:
                cluster['reducing_mass'].append(rm)
                cluster['RT'].append(rt)
                cluster['intensity'].append(intensity)
                cluster['peak_d'].append(peak_d)
                cluster['max_intensity'].append(intensity if intensity > last_max else last_max)
                if len(cluster['intensity']) == 3:
                  if cluster['intensity'][0] > cluster['intensity'][1] > cluster['intensity'][2]:
                    cluster = None
                found = True
                break
        if not found:
            clusters.append({
                'reducing_mass': [rm],
                'RT': [rt],
                'intensity': [intensity],
                'peak_d': [peak_d],
                'max_intensity': [intensity],
            })

    # Create a condensed dataframe
    condensed_data = []
    for cluster in clusters:
        highest_intensity_index = np.argmax(cluster['intensity'])
        highest_intensity = cluster['intensity'][highest_intensity_index]
        if highest_intensity > 0:
            min_rm = cluster['reducing_mass'][highest_intensity_index]
            mean_rt = cluster['RT'][highest_intensity_index]
        else:
            min_rm = min(cluster['reducing_mass'])
            mean_rt = np.mean(cluster['RT'])
        sum_intensity = np.nansum(cluster['intensity'])
        binned_intensities, mz_remainder = zip(*[bin_intensities(c, frames) for c in cluster['peak_d']])
        binned_intensities = np.mean(np.array(binned_intensities), axis = 0)
        mz_remainder = np.mean(np.array(mz_remainder), axis = 0)
        # Bin intensity normalization
        binned_intensities = normalize_array(binned_intensities)
        num_spectra = len(cluster['RT'])
        condensed_data.append([min_rm, mean_rt, sum_intensity, binned_intensities, mz_remainder, num_spectra])
    condensed_df = pd.DataFrame(condensed_data, columns = ['reducing_mass', 'RT', 'intensity', 'binned_intensities', 'mz_remainder', 'num_spectra'])
    return condensed_df


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
    max_conf_rows = []
    # Loop through the DataFrame to find duplicates
    for idx, row in df.iterrows():
        # Set a mask for close enough index values and RT values
        mask = (np.abs(df.index - idx) < mz_diff) & (np.abs(df['RT'] - row['RT']) < rt_diff)
        # Filter DataFrame based on mask
        sub_df = df[mask]
        # Get the first prediction from the tuple
        first_pred = row['predictions'][0][0] if row['predictions'] else None
        if first_pred is None:
          continue
        # Filter sub_df based on the first prediction value
        pred_mask = sub_df['predictions'].apply(lambda x: x[0][0] if x else None) == first_pred
        # Choose the row with the max confidence for this prediction
        max_conf_row = sub_df.loc[pred_mask].iloc[np.argmax([p[0][1] for p in sub_df.loc[pred_mask, 'predictions']])]
        if 'rel_abundance' in df.columns:
          # Sum 'rel_abundance' for this subset
          summed_abundance = np.nansum(sub_df.loc[pred_mask, 'rel_abundance'])
          # Update 'rel_abundance'
          max_conf_row['rel_abundance'] = summed_abundance
        # Store in max_conf_rows
        max_conf_rows.append(max_conf_row)
    dedup_df = pd.DataFrame(max_conf_rows,columns = df.columns)
    dedup_df = dedup_df.astype(dict(df.dtypes))
    # Drop duplicate rows based on index and 'predictions'
    dedup_df.drop_duplicates(subset=['predictions'], keep = 'first', inplace = True)
    return dedup_df


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


def domain_filter(df_out, glycan_class, mode = 'negative', modification = 'reduced',
                  mass_tolerance = 0.5, filter_out = set(), df_use = None):
    """filters out false-positive predictions\n
   | Arguments:
   | :-
   | df_out (dataframe): df_out generated within wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
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
        found_match = False  # Flag to track whether a match is found
        for top_fragment in top_fragments:
            for cmass in cmasses:
                if mass_check(top_fragment, cmass, mass_thresh = 1.5):
                    found_match = True
                    break  # Stop comparing once the first True value is found
            if found_match:
                break
        if not found_match:
            df_out.iat[k, 0] = ['remove']
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
                                  df_use = df_use, filter_out = filter_out, reduced = reduced > 0)[0:1] or ({},))[0].keys() for t in df_out.top_fragments.values.tolist()[k][:20]]))
            # Check fragment size distribution
            if c > 1:
                truth.append(any([j > df_out.index.values[k]*1.2 for j in df_out.top_fragments.values.tolist()[k][:15]]))
            if c == 1:
                truth.append(all([j < df_out.index.values[k]*1.1 for j in df_out.top_fragments.values.tolist()[k][:5]]))
            if len(df_out.top_fragments.values.tolist()[k]) < 5:
                truth.append(False)
            # Check M-adduct for adducts
            if isinstance(df_out.adduct.values.tolist()[k], str):
                truth.append(any([abs(df_out.index.tolist()[k]-mass_dict[df_out.adduct.values.tolist()[k]]-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5]]))
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
        df_out.iat[k, 0] = keep
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
            idx = np.where((mass_diffs < 0.5) & (RT_diffs < 1) & same_compositions)[0]
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
    adduct_list = ['Acetonitrile', 'Acetate', 'Formate'] if mode == 'negative' else ['Na+', 'K+', 'NH4+']
    compositions = df['composition'].values
    charges = df['charge'].values
    indices = df.index.values
    computed_masses = np.array([composition_to_mass(composition) for composition in compositions])
    observed_masses = indices * np.abs(charges) + (np.abs(charges) - 1)
    df['adduct'] = None
    for adduct in adduct_list:
        adduct_mass = mass_dict.get(adduct, 999)
        if modification == 'reduced':
            adduct_mass += 1.0078
        adduct_check = np.abs(computed_masses + adduct_mass - observed_masses) < 0.5
        df.loc[adduct_check, 'adduct'] = adduct
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


def generate_variants_ac_gc(sequence):
    """generates all possible Neu5Ac/Neu5Gc substitutions of sequence\n
   | Arguments:
   | :-
   | sequence (string): glycan in IUPAC-condensed nomenclature\n
   | Returns:
   | :-
   | Returns a list of sequences with possible Neu5Ac/Neu5Gc substitutions
   """
    # Identify all occurrences of Neu5Ac and Neu5Gc
    occurrences = [(m.start(), m.group()) for m in re.finditer(r'Neu5(Ac|Gc)', sequence)]
    # Generate all combinations of substitutions
    variants = [sequence]
    for index, original in occurrences:
        new_variants = []
        for variant in variants:
            if original == 'Neu5Ac':
                # Replace Neu5Ac with Neu5Gc
                new_variant = variant[:index] + 'Neu5Gc' + variant[index + 6:]
            else:
                # Replace Neu5Gc with Neu5Ac
                new_variant = variant[:index] + 'Neu5Ac' + variant[index + 6:]
            new_variants.append(new_variant)
        variants.extend(new_variants)
    return variants


def generate_variants_6S(sequence):
    """generates all possible GlcNAc/GlcNAc6S substitutions of sequence\n
   | Arguments:
   | :-
   | sequence (string): glycan in IUPAC-condensed nomenclature\n
   | Returns:
   | :-
   | Returns a list of sequences with possible GlcNAc/GlcNAc6S substitutions
   """
    # Identify all occurrences of GlcNAc and GlcNAc6S
    occurrences = [(m.start(), m.group()) for m in re.finditer(r'GlcNAc(6S){,1}\(b1\-6\)', sequence)]
    # Generate all combinations of substitutions
    variants = [sequence]
    for index, original in occurrences:
        new_variants = []
        for variant in variants:
            if original == 'GlcNAc(b1-6)':
                # Replace GlcNAc with GlcNAc6S
                new_variant = variant[:index] + 'GlcNAc6S(b1-6)' + variant[index + 12:]
            else:
                # Replace GlcNAc6S with GlcNAc
                new_variant = variant[:index] + 'GlcNAc(b1-6)' + variant[index + 14:]
            new_variants.append(new_variant)
        variants.extend(new_variants)
    return variants


def impute(df_out, mode = 'negative', modification = 'reduced', mass_tag = 0,
           glycan_class = "O"):
    """searches for specific isomers that could be added to the prediction dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', or 'other'/'none'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"\n
    | Returns:
    | :-
    | Returns prediction dataframe with imputed predictions (if possible)
    """
    predictions_list = df_out.predictions.values.tolist()
    index_list = df_out.index.tolist()
    seqs = [p[0][0] for p in predictions_list if p and ("Neu5Ac" in p[0][0] or "Neu5Gc" in p[0][0])]
    variants = set(unwrap([generate_variants_ac_gc(s) for s in seqs]))
    if glycan_class == "O":
      seqs = [p[0][0] for p in predictions_list if p and ("GlcNAc6S(b1-6)" in p[0][0] or "GlcNAc(b1-6)" in p[0][0])]
      variants.update(set(unwrap([generate_variants_6S(s) for s in seqs])))
    for i, k in enumerate(predictions_list):
        if len(k) < 1:
            for v in variants:
                if mass_check(index_list[i], v, mode = mode, modification = modification, mass_tag = mass_tag):
                    df_out.iat[i, 0] = [(v, )]
                    break
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
    top1_preds = set([k[0][0] for k in predictions_list if k and k[0]])
    index_list = df_out.index.tolist()
    mass_keys = np.array(list(mass_dic.keys()))
    for k in range(len(df_out)):
        if len(predictions_list[k]) < 1:
            check_mass = index_list[k] - reduced
            diffs = np.abs(mass_keys - check_mass)
            min_diff_index = np.argmin(diffs)
            if diffs[min_diff_index] < 0.5:
                possible = mass_dic[mass_keys[min_diff_index]]
                df_out.iat[k, 0] = [(m,) for m in possible if m not in top1_preds]
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
    df_out = df_out.assign(true_mass = df_out.index * abs(df_out['charge']) - (df_out['charge'] + np.sign(df_out['charge'].sum())))
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
            p_list[1] += 0.1 * sum(subgraph_isomorphism(p_list[0], t, wildcards_ptm = True) for t in rest_top1 if t != p_list[0])
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


def calculate_ppm_error(theoretical_mass, observed_mass):
    return ((theoretical_mass-observed_mass)/theoretical_mass)* (10**6)


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
        idx = df_filtered.index[len(df_filtered) - 1 - df_filtered.top_pred.values.tolist()[::-1].index(pred)]
        idx_rt = df_filtered.loc[idx, 'RT']
        for k, row in df_filtered[:idx-1][::-1].iterrows():
            if row['top_pred'] == pred and abs(row['RT'] -idx_rt) < 1:
                df_filtered.at[idx, 'rel_abundance'] += row['rel_abundance']
                df_filtered.drop(k, inplace = True)
    df_out = pd.concat([df_out[~df_out['top_pred'].isin(filtered_top_pred)], df_filtered]).sort_index()
    df_out.drop(['top_pred'], axis = 1 , inplace = True)
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
    for gc_pred in df_out[df_out["Original_Prediction"].str.contains("Neu5Gc", na=False)][
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


def filter_delayed_rts(df_out):
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
    for mz_val in df_out.index.unique():
        mz_group = df_out.loc[
            (df_out.index >= mz_val - 0.5) & (df_out.index <= mz_val + 0.5)
        ]
        # Further group by top1 prediction
        for top1_pred, group in mz_group.groupby(
            lambda idx: mz_group.loc[idx, "predictions"][0][0]
            if mz_group.loc[idx, "predictions"]
            else None
        ):
            if group.empty:
                continue
            # Sort the group by RT to easily find the first instance
            group_sorted = group.sort_values(by = "RT")
            first_rt = group_sorted.iloc[0]["RT"]
            # Find rows with RT more than 10 units after the first instance
            delayed_rows = group_sorted[group_sorted["RT"] > first_rt + 10]
            # Calculate the confidence * num_spectra for each row and find the maximum in the group
            group["conf_times_num_spectra"] = group.apply(
                lambda row: row["predictions"][0][1] * row["num_spectra"], axis = 1
            )
            max_conf_times_num_spectra = group["conf_times_num_spectra"].max()
            # Check each delayed row against the maximum value
            for idx, row in delayed_rows.iterrows():
                if row["predictions"][0][1] * row["num_spectra"] < max_conf_times_num_spectra:
                    rows_to_discard.append(idx)
            # Clean up temporary column
            group.drop(columns = ["conf_times_num_spectra"], inplace = True, errors = "ignore")
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
      loaded_file = loaded_file[loaded_file['RT'] < 0.9*loaded_file['RT'].max()].reset_index(drop = True)
    return loaded_file


def get_helper_vars(df_use, modification, mode, taxonomy_class, glycan_class):
    """generates variables used for inference\n
    | Arguments:
    | :-
    | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"\n
    | Returns:
    | :-
    | (1) a library of monosaccharides
    | (2) a sugarbase-like database of glycans with species associations
    | (3) mass to add to glycans based on given modification 
    | (4) sign of ion mode
    """
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_class])
        df_use = df_use[df_use['Class'].apply(lambda x: taxonomy_class in x)]
    reduced = 1.0078 if modification == 'reduced' else 0
    multiplier = -1 if mode == 'negative' else 1
    return df_use, reduced, multiplier


def spectra_filepath_to_condensed_df(spectra_filepath,rt_min,rt_max,rt_diff,mass_tolerance,bin_num=2048):
    """Loads and clusters spectra into distinct mass and RT groups\n
    | Arguments:
    | :-
    | spectra_filepath (string): absolute filepath ending in ".mzML",".mzXML", or ".xlsx" pointing to a file containing spectra or preprocessed spectra 
    | rt_min (float): whether only spectra from a minimum retention time (in minutes) onward should be considered
    | rt_max (float): whether only spectra up to a maximum retention time (in minutes) should be considered
    | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak
    | mass_tolerance (float): the general mass tolerance that is used for composition matching
    | bin_num (int): number of bins for binning; don't change; default: 2048\n
    | Returns:
    | :-
    | Returns a preliminary df_out dataframe of clustered spectra with no predictions 
    """
    loaded_file = load_spectra_filepath(spectra_filepath)
    loaded_file = filter_rts(loaded_file,rt_min,rt_max)
    intensity = 'intensity' in loaded_file.columns and not (loaded_file['intensity'] == 0).all() and not loaded_file['intensity'].isnull().all()
    if intensity:
        loaded_file['intensity'].fillna(0, inplace = True)
    else:
        loaded_file['intensity'] = [0]*len(loaded_file)
    # Prepare file for processing
    loaded_file.dropna(subset = ['peak_d'], inplace = True)
    loaded_file['reducing_mass'] += np.random.uniform(0.00001, 10**(-20), size = len(loaded_file))
    spec_dic = {mass: peak for mass, peak in zip(loaded_file['reducing_mass'].values, loaded_file['peak_d'].values)}
    # Group spectra by mass/retention isomers and process them for being inputs to CandyCrunch
    df_out = condense_dataframe(loaded_file, mz_diff = mass_tolerance, rt_diff = rt_diff, bin_num = bin_num)
    # df_out.set_index('reducing_mass', inplace = True,drop=False)
    df_out['peak_d'] = [spec_dic[m] for m in df_out['reducing_mass']]
    return df_out


def assign_predictions(df_out,glycan_class,model,glycans,mode,modification,lc,trap,mass_tag,pred_thresh,temperature,extra_thresh):
    """generates dataframe initial predictions\n
    | Arguments:
    | :-
    | df_out (dataframe): a preliminary df_out dataframe of clustered spectra with no predictions 
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
    | model (PyTorch): trained CandyCrunch model
    | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'
    | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'
    | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | pred_thresh (float): prediction confidence threshold used for filtering
    | temperature (float): the temperature factor used to calibrate logits
    | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples)\n
    | Returns:
    | :-
    | Returns a preliminary df_out dataframe with initial predictions 
    """
    coded_class = {'O': 0, 'N': 1, 'free': 2, 'lipid': 2}[glycan_class]
    loader, df_out = process_for_inference(df_out, coded_class, mode = mode, modification = modification, lc = lc, trap = trap)
    # Predict glycans from spectra
    preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
    if device != 'cpu':
        preds, pred_conf = average_preds(preds, pred_conf)
    df_out['rel_abundance'] = df_out['intensity']
    df_out['predictions'] = [[(pred, conf) for pred, conf in zip(pred_row, conf_row)] for pred_row, conf_row in zip(preds, pred_conf)]
    # Check correctness of glycan class & mass
    df_out['predictions'] = [
        [(g[0], round(g[1], 4)) for g in preds if
         enforce_class(g[0], glycan_class, g[1], extra_thresh = extra_thresh) and
         g[1] > pred_thresh and
         mass_check(mass, g[0], modification = modification, mass_tag = mass_tag, mode = mode)][:5]
        for preds, mass in zip(df_out.predictions, df_out.index)
        ]
    return df_out
    

def prediction_post_processing(df_out,mode,mass_tolerance,glycan_class,df_use,
                               filter_out,reduced,multiplier,modification,
                               rt_diff,frag_num):
    """filters initial predictions\n
    | Arguments:
    | :-
    | df_out (dataframe) a preliminary df_out dataframe with initial predictions
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | mass_tolerance (float): the general mass tolerance that is used for composition matching
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
    | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
    | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
    | reduced (float): mass to add to glycans based on given modification 
    | multiplier (int): sign of ion mode
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'
    | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak
    | frag_num (int): how many top fragments to show in df_out per spectrum\n
    | Returns:
    | :-
    | Returns a dataframe of filtered predictions 
    """
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
    cols = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d','rel_abundance']
    df_out = df_out[cols]
    # Fill up possible gaps of singly-/doubly-charged spectra of the same structure
    df_out = backfill_missing(df_out)
    # Extract & sort the top 100 fragments by intensity
    df_out['top_fragments'] = [
        [round(frag[0], 4) for frag in sorted(peak_d.items(), key = lambda x: x[1], reverse = True)[:frag_num]]
        for peak_d in df_out['peak_d']
        ]
    # Filter out wrong predictions via diagnostic ions etc.
    df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out,
                           modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
    # Deduplicate identical predictions for different spectra
    df_out = deduplicate_predictions(df_out, mz_diff = mass_tolerance, rt_diff = rt_diff)
    df_out['evidence'] = ['strong' if preds else np.nan for preds in df_out['predictions']]
    return df_out


def augment_predictions(df_out,supplement,experimental,glycan_class,df_use,mode,modification,mass_tag,filter_out,taxonomy_class,reduced,mass_tolerance,mass_dic):
    """adds and reorders predictions based on possible structures\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of filtered predictions
    | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks
    | experimental (bool): whether to impute missing predictions via database searches etc.
    | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
    | df_use (dataframe): sugarbase-like database of glycans with species associations etc.
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'; default:'reduced'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | filter_out (set): set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
    | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia'
    | reduced (float): mass to add to glycans based on given modification 
    | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
    | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally\n
    | Returns:
    | :-
    | Returns a dataframe of predictions 
    """
    # Construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
    if supplement:
        try:
            df_out = supplement_prediction(df_out, glycan_class, mode = mode, modification = modification, mass_tag = mass_tag)
            df_out['evidence'] = [
                'medium' if pd.isna(evidence) and preds else evidence
                for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
                ]
        except:
            pass
    # Check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
    if experimental:
        df_out = impute(df_out, mode = mode, modification = modification, mass_tag = mass_tag, glycan_class = glycan_class)
        try:
            df_out = filter_delayed_rts(df_out)
            df_out = Ac_follows_Gc(df_out)
        except ValueError:
            pass
        mass_dic = mass_dic if mass_dic else make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = taxonomy_class)
        df_out = possibles(df_out, mass_dic, reduced)
        df_out['evidence'] = [
            'weak' if pd.isna(evidence) and preds else evidence
            for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
            ]
    # Filter out wrong predictions via diagnostic ions etc.
    if supplement or experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
        df_out['predictions'] = [[(k[0].replace('-ol', '').replace('1Cer', ''), k[1]) if len(k) > 1 else (k[0].replace('-ol', '').replace('1Cer', ''),) for k in j] if j else j for j in df_out['predictions']]
    return df_out
    
    
def finalise_predictions(df_out,get_missing,pred_thresh,mode,modification,mass_tag,multiplier,plot_glycans,spectra_filepath,spectra):
    """Cleans up incorrect structure predictions and formats dataframe\n
    | Arguments:
    | :-
    | df_out (dataframe): a dataframe of augmented predictions
    | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition
    | pred_thresh (float): prediction confidence threshold used for filtering
    | mode (string): mass spectrometry mode, either 'negative' or 'positive'
    | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'
    | mass_tag (float): mass of custom reducing end tag that should be considered if relevant
    | multiplier (int): sign of ion mode
    | plot_glycans (bool): whether you want to save an output.xlsx file that contains SNFG images of all top1 predictions
    | spectra_filepath (string): absolute filepath ending in ".mzML",".mzXML", or ".xlsx" pointing to a file containing spectra or preprocessed spectra 
    | spectra (bool): whether to also output the actual spectra used for prediction; default:False\n
    | Returns:
    | :-
    | Returns a dataframe of corrected predictions 
    """
    # Keep or remove spectra that still lack a prediction after all this
    if not get_missing:
        df_out = df_out[df_out['predictions'].str.len() > 0]
    # Reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
    try:
        df_out = canonicalize_biosynthesis(df_out, pred_thresh)
    except ValueError:
        pass
    spectra_out = df_out.pop('peak_d').values.tolist()
    # Calculate  ppm error
    valid_indices, ppm_errors = [], []
    for preds, obs_mass in zip(df_out['predictions'], df_out.index):
        theo_mass = mass_check(obs_mass, preds[0][0], modification = modification, mass_tag = mass_tag, mode = mode)
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
    df_out = df_out.astype({'num_spectra': 'int', 'charge': 'int'})
    df_out = combine_charge_states(df_out)
    # Map GlyTouCan IDs
    df_out["GlyTouCan_ID"] = [glytoucan_mapping[g[0][0]] if g and g[0][0] in glytoucan_mapping else '' for g in df_out["predictions"]]
    # Normalize relative abundances if relevant
    if not all(np.array(df_out['rel_abundance'])==0):
        df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
    else:
        df_out.drop(['rel_abundance'], axis = 1, inplace = True)
    df_out.index.name = "m/z"
    if plot_glycans:
        from glycowork.motif.draw import plot_glycans_excel
        plot_glycans_excel(df_out, '/'.join(spectra_filepath.split("\\")[:-1])+'/', glycan_col_num = 0)
    return (df_out, spectra_out) if spectra else df_out


def wrap_inference(spectra_filepath, glycan_class, model = candycrunch, glycans = glycans, bin_num = 2048,
                   frag_num = 100, mode = 'negative', modification = 'reduced', mass_tag = None, lc = 'PGC', trap = 'linear', rt_min = 0, rt_max = 0, rt_diff = 1.0,
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False, mass_tolerance = 0.5, extra_thresh = 0.2,
                   filter_out = {'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True, experimental = True, mass_dic = None,
                   taxonomy_class = 'Mammalia', df_use = None, plot_glycans = False):
    """wrapper function to get & curate CandyCrunch predictions\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".mzML",".mzXML", or ".xlsx" pointing to a file containing spectra or preprocessed spectra 
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0
   | rt_max (float): whether only spectra up to a maximum retention time (in minutes) should be considered; default:0
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
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
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | plot_glycans (bool): whether you want to save an output.xlsx file that contains SNFG images of all top1 predictions; default:False\n
   | Returns:
   | :-
   | Returns dataframe of predictions for spectra in file
   """
    print(f"Your chosen settings are: {glycan_class} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_class])
        df_use = df_use[df_use['Class'].apply(lambda x: taxonomy_class in x)]
    reduced = 1.0078 if modification == 'reduced' else 0
    multiplier = -1 if mode == 'negative' else 1
    loaded_file = load_spectra_filepath(spectra_filepath)
    loaded_file = filter_rts(loaded_file,rt_min,rt_max)
    intensity = 'intensity' in loaded_file.columns and not (loaded_file['intensity'] == 0).all() and not loaded_file['intensity'].isnull().all()
    if intensity:
        loaded_file['intensity'].fillna(0, inplace = True)
    else:
        loaded_file['intensity'] = [0]*len(loaded_file)
    # Prepare file for processing
    loaded_file.dropna(subset = ['peak_d'], inplace = True)
    loaded_file['reducing_mass'] += np.random.uniform(0.00001, 10**(-20), size = len(loaded_file))
    coded_class = {'O': 0, 'N': 1, 'free': 2, 'lipid': 2}[glycan_class]
    spec_dic = {mass: peak for mass, peak in zip(loaded_file['reducing_mass'].values, loaded_file['peak_d'].values)}
    # Group spectra by mass/retention isomers and process them for being inputs to CandyCrunch
    df_out = condense_dataframe(loaded_file, mz_diff = mass_tolerance, rt_diff = rt_diff, bin_num = bin_num)
    loader, df_out = process_for_inference(df_out, coded_class, mode = mode, modification = modification, lc = lc, trap = trap)
    df_out['peak_d'] = [spec_dic[m] for m in df_out.index]
    # Predict glycans from spectra
    preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
    if device != 'cpu':
        preds, pred_conf = average_preds(preds, pred_conf)
    df_out['rel_abundance'] = df_out['intensity']
    df_out['predictions'] = [[(pred, conf) for pred, conf in zip(pred_row, conf_row)] for pred_row, conf_row in zip(preds, pred_conf)]
    # Check correctness of glycan class & mass
    df_out['predictions'] = [
        [(g[0], round(g[1], 4)) for g in preds if
         enforce_class(g[0], glycan_class, g[1], extra_thresh = extra_thresh) and
         g[1] > pred_thresh and
         mass_check(mass, g[0], modification = modification, mass_tag = mass_tag, mode = mode)][:5]
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
    cols = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d', 'rel_abundance']
    df_out = df_out[cols]
    # Fill up possible gaps of singly-/doubly-charged spectra of the same structure
    df_out = backfill_missing(df_out)
    # Extract & sort the top 100 fragments by intensity
    df_out['top_fragments'] = [
        [round(frag[0], 4) for frag in sorted(peak_d.items(), key = lambda x: x[1], reverse = True)[:frag_num]]
        for peak_d in df_out['peak_d']
        ]
    # Filter out wrong predictions via diagnostic ions etc.
    df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out,
                           modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
    # Deduplicate identical predictions for different spectra
    df_out = deduplicate_predictions(df_out, mz_diff = mass_tolerance, rt_diff = rt_diff)
    df_out['evidence'] = ['strong' if preds else np.nan for preds in df_out['predictions']]
    # Construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
    if supplement:
        try:
            df_out = supplement_prediction(df_out, glycan_class, mode = mode, modification = modification, mass_tag = mass_tag)
            df_out['evidence'] = [
                'medium' if pd.isna(evidence) and preds else evidence
                for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
                ]
        except:
            pass
    # Check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
    if experimental:
        df_out = impute(df_out, mode = mode, modification = modification, mass_tag = mass_tag, glycan_class = glycan_class)
        df_out = filter_delayed_rts(df_out)
        df_out = Ac_follows_Gc(df_out)
        mass_dic = mass_dic if mass_dic else make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = taxonomy_class)
        df_out = possibles(df_out, mass_dic, reduced)
        df_out['evidence'] = [
            'weak' if pd.isna(evidence) and preds else evidence
            for evidence, preds in zip(df_out['evidence'], df_out['predictions'])
            ]
    # Filter out wrong predictions via diagnostic ions etc.
    if supplement or experimental:
        df_out = domain_filter(df_out, glycan_class, mode = mode, filter_out = filter_out, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
        df_out['predictions'] = [[(k[0].replace('-ol', '').replace('1Cer', ''), k[1]) if len(k) > 1 else (k[0].replace('-ol', '').replace('1Cer', ''),) for k in j] if j else j for j in df_out['predictions']]
    # Keep or remove spectra that still lack a prediction after all this
    if not get_missing:
        df_out = df_out[df_out['predictions'].str.len() > 0]
    # Reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
    df_out = canonicalize_biosynthesis(df_out, pred_thresh)
    spectra_out = df_out.pop('peak_d').values.tolist()
    # Calculate  ppm error
    valid_indices, ppm_errors = [], []
    for preds, obs_mass in zip(df_out['predictions'], df_out.index):
        theo_mass = mass_check(obs_mass, preds[0][0], modification = modification, mass_tag = mass_tag, mode = mode)
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
    df_out = df_out.astype({'num_spectra': 'int', 'charge': 'int'})
    df_out = combine_charge_states(df_out)
    # Map GlyTouCan IDs
    df_out["GlyTouCan_ID"] = [glytoucan_mapping[g[0][0]] if g and g[0][0] in glytoucan_mapping else '' for g in df_out["predictions"]]
    # Normalize relative abundances if relevant
    if intensity:
        df_out['rel_abundance'] = df_out['rel_abundance'] / df_out['rel_abundance'].sum() * 100
    else:
        df_out.drop(['rel_abundance'], axis = 1, inplace = True)
    df_out.index.name = "m/z"
    if plot_glycans:
        from glycowork.motif.draw import plot_glycans_excel
        plot_glycans_excel(df_out, '/'.join(spectra_filepath.split("\\")[:-1])+'/', glycan_col_num = 0)
    return (df_out, spectra_out) if spectra else df_out


def wrap_inference_batch(spectra_filepath_list, glycan_class, intra_cat_thresh ,top_n_isomers, model = candycrunch, glycans = glycans, bin_num = 2048,
                   frag_num = 100, mode = 'negative', modification = 'reduced', mass_tag = None, lc = 'PGC', trap = 'linear', rt_min = 0, rt_max = 0, rt_diff = 1.0,
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False, mass_tolerance = 0.5, extra_thresh = 0.2,
                   filter_out = {'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, supplement = True, experimental = True, mass_dic = None,
                   taxonomy_class = 'Mammalia', df_use = None, plot_glycans = False):
    """wrapper function to get & curate CandyCrunch predictions, then harmonize them across multiple files\n
   | Arguments:
   | :-
   | spectra_filepath (string): absolute filepath ending in ".mzML",".mzXML", or ".xlsx" pointing to a file containing spectra or preprocessed spectra 
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | intra_cat_thresh (float): minutes the RT of a structure can differ from the mean of a group
   | top_n_isomers (int): number of different isomer groups at each composition to retain
   | model (PyTorch): trained CandyCrunch model
   | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
   | bin_num (int): number of bins for binning; don't change; default: 2048
   | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB' or 'custom'; default:'reduced'
   | mass_tag (float): mass of custom reducing end tag that should be considered if relevant; default:None
   | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
   | trap (string): type of mass detector; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
   | rt_min (float): whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0
   | rt_max (float): whether only spectra up to a maximum retention time (in minutes) should be considered; default:0
   | rt_diff (float): maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0
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
   | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
   | plot_glycans (bool): whether you want to save an output.xlsx file that contains SNFG images of all top1 predictions; default:False\n
   | Returns:
   | :-
   | Returns a dictionary of dataframes, one for each input file, keyed by their filenames
   """
    print(f"Your chosen settings are: {glycan_class} glycans, {mode} ion mode, {modification} glycans, {lc} LC, and {trap} ion trap. If any of that seems off to you, please restart with correct parameters.")
    df_use,reduced,multiplier = get_helper_vars(df_use,modification,mode,taxonomy_class,glycan_class)
    inference_dfs = {}
    for spectra_filepath in spectra_filepath_list:
        file_label = spectra_filepath.split('/')[-1].split('.')[0]
        df_out = spectra_filepath_to_condensed_df(spectra_filepath,rt_min,rt_max,rt_diff,mass_tolerance)
        df_out = assign_predictions(df_out,glycan_class,model,glycans,mode,modification,lc,trap,mass_tag,pred_thresh,temperature,extra_thresh)
        df_out = prediction_post_processing(df_out,mode,mass_tolerance,glycan_class,df_use,filter_out,reduced,multiplier,modification,rt_diff,frag_num)
        df_out['mass_label'] = np.round(df_out.index * 2) / 2
        df_out = df_out.assign(condition_label = spectra_filepath.split('/')[-1].split('.')[0])
        inference_dfs[file_label] = df_out
    assigned_cats = assign_categories(pd.concat([x for x in inference_dfs.values()]),intra_cat_thresh=intra_cat_thresh,maximise_cat_size=True) 
    smoothed_category_predictions = assign_modal_category_prediction(assigned_cats)
    prevailing_category_predictions = filter_top_n_isomers(smoothed_category_predictions,top_n=top_n_isomers)
    for file_label in prevailing_category_predictions.condition_label.unique():
        df_out = prevailing_category_predictions[prevailing_category_predictions['condition_label'] == file_label]
        df_out = df_out.sort_index().drop_duplicates(subset=['RT','rel_abundance'])
        df_out = augment_predictions(df_out,supplement,experimental,glycan_class,df_use,mode,modification,mass_tag,filter_out,taxonomy_class,reduced,mass_tolerance,mass_dic)
        df_out = finalise_predictions(df_out,get_missing,pred_thresh,mode,modification,mass_tag,multiplier,plot_glycans,spectra_filepath,spectra)
        inference_dfs[file_label] = df_out
    return inference_dfs


def filter_top_n_isomers(df_in,top_n=3):
    df_out = df_in.copy(deep=True)
    df_out['paired_categories'] = [(x,y) for x,y in zip(df_out.mass_label,df_out.category_label)]
    grouped_bc = df_out[['mass_label','category_label','condition_label']].groupby(['mass_label','category_label']).nunique()
    df_out['file_presences'] = df_out['paired_categories'].map(grouped_bc['condition_label'].to_dict())
    permitted_mass_cats_df = df_out.sort_values(['mass_label','file_presences'],ascending=False).groupby(['mass_label','top1_pred'],as_index=False).first(top_n)
    permitted_mass_cats = list(zip(permitted_mass_cats_df.mass_label,permitted_mass_cats_df.category_label))
    df_out = df_out[df_out['paired_categories'].isin(permitted_mass_cats)]
    return df_out


def assign_modal_category_prediction(assigned_cats):
    assigned_cats['top1_pred'] = [x[0][0] if x else None for x in assigned_cats['predictions']]
    most_common_group_preds = dict(assigned_cats[['mass_label','category_label','top1_pred']].groupby(['mass_label','category_label']).value_counts())
    most_common_mapping = {} 
    unq_groups = set([(x[0],x[1]) for x in most_common_group_preds])
    for unq in unq_groups:
        prevalence_sort = sorted([(k,v) for k,v in most_common_group_preds.items() if (k[0],k[1]) == unq],key=lambda x:x[1])
        mode_pred = prevalence_sort[-1]
        most_common_mapping[(mode_pred[0][0],mode_pred[0][1])] = mode_pred[0][2]
    assigned_cats['top1_pred'] = [most_common_mapping[(ml,cl)] if tp else None for ml,cl,tp in zip(assigned_cats.mass_label,assigned_cats.category_label,assigned_cats.top1_pred)]
    assigned_cats['predictions'] = [[(top1_p,0.888)]+preds[1:] if preds else [] for top1_p,preds in zip(assigned_cats.top1_pred,assigned_cats.predictions)]
    return assigned_cats


def assign_categories(all_ms2_spectra,intra_cat_thresh = 3,maximise_cat_size=True):
    all_mass_dfs = []
    for search_mass in all_ms2_spectra.mass_label.unique():
        mass_group_dfs = []
        for condition_label in all_ms2_spectra.condition_label.unique():
            mass_group = all_ms2_spectra[(all_ms2_spectra['mass_label'] == search_mass)&(all_ms2_spectra['condition_label'] == condition_label)].copy(deep=True)
            mass_group = mass_group.assign(RT_group = [i for i in range(len(mass_group))])
            mass_group_dfs.append(mass_group)
        cats_mass_dfs = mass_dfs_to_categories(mass_group_dfs,intra_cat_thresh,maximise_cat_size=maximise_cat_size)
        all_mass_dfs.append(cats_mass_dfs)
    return pd.concat([p for q in all_mass_dfs for p in q])


def mass_dfs_to_categories(mass_range_dfs,inter_sample_thresh,maximise_cat_size=True):
    RT_groups = create_RT_groups(mass_range_dfs)
    categories = initialise_categories(RT_groups)
    categories = expand_RT_categories(RT_groups,categories,inter_sample_thresh,maximise_cat_size=maximise_cat_size)
    sample_cats = RT_cats_to_sample_cats(categories,RT_groups)
    cat_dfs = sample_categories_to_df(sample_cats,mass_range_dfs)
    return cat_dfs


def create_RT_groups(mass_range_dfs):
    all_sample_RT_groups = []
    for sample_df in mass_range_dfs:
        RT_groups = []
        for RT_group in sample_df.groupby('RT_group').agg(list).RT:
            RT_groups.append(RT_group)
        all_sample_RT_groups.append(RT_groups)
    return all_sample_RT_groups


def initialise_categories(all_sample_RT_groups):
    categories = {0:[]}
    for x in all_sample_RT_groups[0]:
        add_new_category(categories,x)
    return categories


def add_new_category(categories,cluster):
    new_id = max([x for x in categories])
    categories[new_id+1] = []
    categories[new_id+1].append(cluster)
    return categories


def expand_RT_categories(all_sample_RT_groups,categories,inter_sample_thresh,maximise_cat_size=False):
    for i,sample in enumerate(all_sample_RT_groups[1:]):
        all_candidate_categories = calculate_candidate_clusters(sample,categories,inter_sample_thresh)
        orphan_idxs = []
        for idx,(orphan_cluster,empty_candidates) in enumerate(zip(sample,all_candidate_categories)):
            if len(empty_candidates) == 0:
                add_new_category(categories,orphan_cluster)
                orphan_idxs.append(idx)
        sample = [x for u,x in enumerate(sample) if u not in orphan_idxs]
        all_candidate_categories = [x for u,x in enumerate(all_candidate_categories) if u not in orphan_idxs]
        settle_category_conflict(sample,all_candidate_categories,categories)
        while [x for x in all_candidate_categories if len(x) ==1]:
            for clusters,cand_categories in zip(sample,all_candidate_categories):
                if len(cand_categories) == 1:
                    categories[list(cand_categories)[0]].append(clusters)
                    all_candidate_categories = [x - cand_categories for x in all_candidate_categories]
        valid_idxs = [i for i,x in enumerate(all_candidate_categories) if x]
        sample = [x for u,x in enumerate(sample) if u in valid_idxs]
        all_candidate_categories = [x for u,x in enumerate(all_candidate_categories) if u in valid_idxs]
        if [x for x in all_candidate_categories if x]:
            if maximise_cat_size:
                optim_cats = find_closest_categories_largest(sample,all_candidate_categories,categories)
            else: 
                optim_cats = find_closest_categories(sample,all_candidate_categories,categories)
            for cluster,optim_cat in zip(sample,optim_cats):
                if optim_cat:
                    categories[optim_cat].append(cluster)
                else:
                    add_new_category(categories,cluster)
    return categories


def calculate_candidate_clusters(sample,categories,inter_sample_thresh):
    all_candidate_categories = []
    for cluster in sample:
        candidate_categories = set()
        for category,cat_pop in categories.items():
            if cat_pop:
                if abs(np.mean(cluster)-np.mean([p for q in cat_pop for p in q])) < inter_sample_thresh:
                    candidate_categories.add(category)
        all_candidate_categories.append(candidate_categories)
    return all_candidate_categories


def settle_category_conflict(sample,cand_categories,categories):
    for cat in set().union(*cand_categories):
        if len([x for x in cand_categories if x == {cat}]) >1:
            closest_cluster_idx = np.argmin([abs(np.mean([p for q in categories[cat] for p in q])-np.mean(j)) for j in sample])
            categories[cat].append(sample[closest_cluster_idx])
            for other_cluster in [x for i,x in enumerate(sample) if i != closest_cluster_idx]:
                categories = add_new_category(categories,other_cluster)
    return categories


def find_closest_categories_largest(sample,sample_candidates,categories):
    cat_means = get_category_means([sorted(x) for x in sample_candidates],categories)
    cluster_means = [x[0] for x in sample]
    cat_mean_diffs = []
    for cluster_mean,cat_mean in zip(cluster_means,cat_means):
        cat_mean_diffs.append({k:abs(v-cluster_mean) for k,v in cat_mean.items()})
    cats_out = [set() for m in cluster_means]
    selected_RTs = []
    sorted_categories = dict(sorted(categories.items(),key = lambda x:len(x[1]),reverse=True))
    for cat in sorted_categories:
        closest_RTs = sorted([(diffs[cat],sample_RT) for diffs,sample_RT in zip(cat_mean_diffs,cluster_means) if cat in diffs if sample_RT not in selected_RTs])
        if not closest_RTs:
            continue
        selected_RT = closest_RTs[0][1]
        selected_RTs.append(selected_RT)
        cats_out[cluster_means.index(selected_RT)] = cat
    return cats_out


def find_closest_categories(sample,sample_candidates,categories):
    cat_means = get_category_means([sorted(x) for x in sample_candidates],categories)
    cluster_means = sample
    cat_mean_diffs = []
    for cluster_mean,cat_mean in zip(cluster_means,cat_means):
        cat_mean_diffs.append({k:abs(v-cluster_mean) for k,v in cat_mean.items()})
    disallowed_list = []
    chosen_list = []
    sorted_idx = [sorted(cat_mean_diffs,key = lambda x: min([y for y in x.values()])).index(x) for i,x in enumerate(cat_mean_diffs)]
    for cat_cands in sorted(cat_mean_diffs,key = lambda x: min([y for y in x.values()])):
        sorted_cands = sorted(cat_cands.items(),key=lambda x:x[1])
        filtered_cands = [x for x in sorted_cands if x[0] not in disallowed_list]
        if filtered_cands:
            chosen_cand = filtered_cands[0]
            chosen_list.append(chosen_cand)
            disallowed_list.append(chosen_cand[0])
        else:
            chosen_list.append((set(),None))
    cats_out = [x[0] for x in chosen_list]
    cats_out = [cats_out[x] for x in sorted_idx]
    return cats_out


def get_category_means(candidate_categories,categories):
    category_means = []
    for cand_cats in candidate_categories:
        category_means.append({cat:np.mean([p for q in categories[cat] for p in q]) for cat in cand_cats})
    return category_means 


def RT_cats_to_sample_cats(RT_categories,RT_groups):
    sample_group_categories = {x:[] for x in RT_categories}
    for k,v in RT_categories.items():
        for cluster in v:
            for i,x in enumerate(RT_groups):
                if cluster in x:
                    sample_group_categories[k].append((i,x.index(cluster)))
    return sample_group_categories


def sample_categories_to_df(sample_group_categories,mass_range_df):
    category_dfs = []
    for cat,group in sample_group_categories.items():
        for x in group:
            cat_df = mass_range_df[x[0]][mass_range_df[x[0]]['RT_group'] == x[1]]
            cat_df = cat_df.assign(category_label = [cat for x in cat_df['RT']])
            category_dfs.append(cat_df)
    return category_dfs


def supplement_prediction(df_in, glycan_class, mode = 'negative', modification = 'reduced', mass_tag = None):
    """searches for biosynthetic precursors of CandyCrunch predictions that could explain peaks\n
   | Arguments:
   | :-
   | df_in (pandas dataframe): output file produced by wrap_inference
   | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free"
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', or 'other'/'none'; default:'reduced'
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
    preds_set = set(preds)
    new_nodes = [k for k in net.nodes() if k not in preds_set]
    explained_idx = [[unexplained_idx[k] for k, check in enumerate([mass_check(j, node,
                                                                               modification = modification, mode = mode, mass_tag = mass_tag) for j in unexplained]) if check] for node in new_nodes]
    new_nodes = [(node, idx) for node, idx in zip(new_nodes, explained_idx) if idx]
    explained = {k: [] for k in set(unwrap(explained_idx))}
    for node, indices in new_nodes:
        for index in indices:
            explained[index].append(node)
    pred_idx = df.columns.get_loc('predictions')
    for index, values in explained.items():
        df.iat[index, pred_idx] = [(t,0) for t in values[:5]]
    return df

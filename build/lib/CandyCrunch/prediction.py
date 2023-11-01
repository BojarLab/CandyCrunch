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


def mass_check(mass, glycan, libr = None, mode = 'negative', modification = 'reduced', mass_tag = 0,
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500):
    """determine whether glycan could explain m/z\n
   | Arguments:
   | :-
   | mass (float): observed m/z
   | glycan (string): glycan in IUPAC-condensed nomenclature
   | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
   | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
   | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated', '2AA', '2AB, or 'custom'; default:'reduced'
   | mass_tag (float): label mass to add when calculating possible m/z if modification == 'custom'; default:0
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
        mz = glycan_to_mass(glycan, sample_prep= modification if modification in ["permethylated", "peracetylated"] else 'underivatized')
    except:
        return False
    mz += modification_mass_dict[modification] if modification in modification_mass_dict else mass_tag
    adduct_list = ['Acetonitrile', 'Acetate', 'Formate'] if mode == 'negative' else ['Na+', 'K+', 'NH4+']
    og_list = [mz] + [mz + mass_dict.get(adduct, 999) for adduct in adduct_list]
    mz_list = og_list.copy()
    thresh = 0.5
    for z, threshold, charge_adjust in zip(
        [2, 3, 4],
        [double_thresh, triple_thresh, quadruple_thresh],
        [-0.5, -0.66, -0.75] if mode == 'negative' else [0.5, 0.66, 0.75]
        ):
        if mz > threshold:
            mz_list += [(m / z + charge_adjust) for m in og_list]
    return [m for m in mz_list if abs(mass - m) < thresh]


def normalize_array(input_array):
    array_sum = input_array.sum()
    return input_array / array_sum


def condense_dataframe(df, min_mz = 39.714, max_mz = 3000, bin_num = 2048):
    """groups spectra and combines the clusters into averaged and binned spectra\n
    | Arguments:
    | :-
    | df (dataframe): dataframe from load_spectra_filepath
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
            if abs(last_rm - rm) <= 0.5 and abs(last_rt - rt) <= 1:
                cluster['reducing_mass'].append(rm)
                cluster['RT'].append(rt)
                cluster['intensity'].append(intensity)
                cluster['peak_d'].append(peak_d)
                cluster['max_intensity'].append(intensity if intensity > last_max else last_max)
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


def deduplicate_predictions(df):
    """removes/unifies duplicate predictions\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference\n
   | Returns:
   | :-
   | Returns a deduplicated dataframe
   """
    # Sort by index and 'RT'
    df.sort_values(by = ['RT'], inplace = True)
    df.sort_index(inplace = True) 
    # Initialize an empty DataFrame for deduplicated records
    dedup_df = pd.DataFrame(columns = df.columns)  
    # Loop through the DataFrame to find duplicates
    for idx, row in df.iterrows():
        # Set a mask for close enough index values and RT values
        mask = (np.abs(df.index - idx) < 0.5) & (np.abs(df['RT'] - row['RT']) < 1)
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
        # Store in dedup_df
        dedup_df = pd.concat([dedup_df, pd.DataFrame([max_conf_row])])
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
    if loaded_file['RT'].max() > 20:
      loaded_file = loaded_file[loaded_file['RT'] >= 2].reset_index(drop = True)
    if loaded_file['RT'].max() > 40:
      loaded_file = loaded_file[loaded_file['RT'] < 0.9*loaded_file['RT'].max()].reset_index(drop = True)
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
    df_out = condense_dataframe(loaded_file, bin_num = bin_num)
    loader, df_out = process_for_inference(df_out, coded_class, mode = mode, modification = modification, lc = lc, trap = trap)
    df_out['peak_d'] = [spec_dic[m] for m in df_out.index]
    # Predict glycans from spectra
    preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
    if device != 'cpu':
        preds, pred_conf = average_preds(preds, pred_conf)
    if intensity:
        df_out['rel_abundance'] = df_out['intensity']
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
        theo_mass = mass_check(obs_mass, preds[0][0], libr = libr, modification = modification, mass_tag = mass_tag, mode = mode)
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
    explained_idx = [[unexplained_idx[k] for k, check in enumerate([mass_check(j, node, libr = libr,
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

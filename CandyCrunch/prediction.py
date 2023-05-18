import numpy as np
import pandas as pd
import numpy_indexed as npi
from collections import defaultdict
from itertools import combinations
from glycowork.motif.processing import enforce_class, get_lib, expand_lib
from glycowork.motif.graph import subgraph_isomorphism
from glycowork.motif.tokenization import mapping_file, glycan_to_composition, glycan_to_mass, mz_to_composition, mz_to_composition2, composition_to_mass
from glycowork.network.biosynthesis import construct_network, plot_network, evoprune_network
from glycowork.glycan_data.loader import unwrap, stringify_dict, df_glycan
from CandyCrunch.model import CandyCrunch_CNN, SimpleDataset, transform_mz, transform_prec, transform_rt
import os
import ast
import copy
import torch
import pickle
import pymzml
import operator
import torch.nn.functional as F

this_dir, this_filename = os.path.split(__file__) 
data_path = os.path.join(this_dir, 'glycans.pkl')
glycans = pickle.load(open(data_path, 'rb'))
lib = get_lib(glycans)

fp_in = "drive/My Drive/CandyCrunch/"

#choose the correct computing architecture
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

sdict = os.path.join(this_dir, 'candycrunch.pt')
sdict = torch.load(sdict, map_location = device)
sdict = {k.replace('module.',''):v for k,v in sdict.items()}
candycrunch = CandyCrunch_CNN(2048, num_classes = len(glycans)).to(device)
candycrunch.load_state_dict(sdict)
candycrunch = candycrunch.eval()

mass_dict = dict(zip(mapping_file.composition, mapping_file["underivatized_monoisotopic"]))
abbrev_dict = {'S':'Sulphate','P':'Phosphate','Ac':'Acetate'}
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
  highest_i_dict = defaultdict(dict)
  number_of_peaks_to_extract = num_peaks
  rts = []
  charges = []
  intensities = []
  prev_len = 0
  for spectrum in run:
    if spectrum.ms_level == ms_level:
      if len(spectrum.peaks("raw")) < number_of_peaks_to_extract:
        ex_num = len(spectrum.peaks("raw"))
      else:
        ex_num = number_of_peaks_to_extract
      try:
        temp = spectrum.highest_peaks(2)
        for mz, i in spectrum.highest_peaks(ex_num):
          highest_i_dict[str(spectrum.ID) + '_' + str(spectrum.selected_precursors[0]['mz'])][mz] = i
        if len(highest_i_dict.keys())>prev_len:
          rts.append(spectrum.scan_time_in_minutes())
          if intensity:
            inty = spectrum.selected_precursors[0]['i'] if 'i' in spectrum.selected_precursors[0].keys() else np.nan
            intensities.append(inty)
          prev_len = len(highest_i_dict.keys())
      except:
        pass
  reducing_mass = [float(k.split('_')[-1]) for k in list(highest_i_dict.keys())]
  peak_d = list(highest_i_dict.values())
  df_out = pd.DataFrame([reducing_mass, peak_d]).T
  df_out.columns = ['reducing_mass', 'peak_d']
  df_out['RT'] = rts
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
  result = {}
  for d in dicts:
    for mass, intensity in d.items():
      if mass in result:
        result[mass].append(intensity)
      else:
        result[mass] = [intensity]
  for mass, intensities in result.items():
    if mode == 'mean':
      result[mass] = sum(intensities) / len(intensities)
    else:
      result[mass] = max(intensities)
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
  out_list = np.zeros(len(frames))
  out_list2 = np.zeros(len(frames))
  keys = list(dict.fromkeys(peak_d))
  filled_bins = np.digitize(np.array(keys, dtype = 'float32'), frames, right = True)
  mz_remainder = keys-frames[filled_bins-1]
  unq, ids = np.unique(filled_bins, return_inverse = True)
  vals = np.array(list(map(peak_d.get, keys)))
  a2 = mz_remainder[npi.group_by(ids).argmax(vals)[1]]
  for b,s,m in zip(unq, np.bincount(ids, np.array(list(map(peak_d.get, keys)))), np.bincount(range(len(a2)), a2)):
    out_list[b-1] = s
    out_list2[b-1] = m
  return out_list, out_list2

def process_for_inference(keys, values, rts, num_spectra, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear', min_mz = 39.714, max_mz = 3000, bin_num = 2048):
  """processes averaged spectra for them being inputs to CandyCrunch\n
  | Arguments:
  | :-
  | keys (list): list of m/z values
  | values (list): list of spectra in form peak m/z:intensity
  | rts (list): list of retention times
  | num_spectra (list): list of number of spectra for each cluster
  | glycan_class (int): 0 = O-linked, 1 = N-linked, 2 = lipid/free, 3 = other
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
  data = {'reducing_mass':keys, 'RT':rts,'peak_d':values,'num_spectra':num_spectra}
  dd = list(zip(*data.values()))
  df = pd.DataFrame(dd, columns= data.keys())
  df['glycan_type'] = [glycan_class]*len(df)
  df['glycan'] = [0]*len(df)
  df['mode'] = [0]*len(df) if mode == 'negative' else [1]*len(df)
  df['lc'] = [0]*len(df) if lc == 'PGC' else [1]*len(df) if lc == 'C18' else [2]*len(df)
  df['modification'] = [0]*len(df) if modification == 'reduced' else [1]*len(df) if modification == 'permethylated' else [2]*len(df)
  df['trap'] = [0]*len(df) if trap == 'linear' else [1]*len(df) if trap == 'orbitrap' else [2]*len(df) if trap == 'amazon' else [3]*len(df)
  #intensity normalization
  df.peak_d = [{k: v / sum(d.values()) for k, v in d.items()} for d in df.peak_d.values.tolist()]
  #retention time normalization
  df['RT2'] = [k/max(max(df.RT.values.tolist()),30) for k in df.RT.values.tolist()]
  #intensity binning
  step = (max_mz - min_mz) / (bin_num - 1)
  frames = np.array([min_mz + step * i for i in range(bin_num)])
  df['binned_intensities'], df['mz_remainder'] = list(zip(*[bin_intensities(df.peak_d.values.tolist()[k], frames) for k in range(len(df))]))
  #dataloader generation
  X = list(zip(df.binned_intensities.values.tolist(),df.mz_remainder.values.tolist(),df.reducing_mass.values.tolist(),df.glycan_type.values.tolist(),
               df.RT2.values.tolist(), df['mode'].values.tolist(), df.lc.values.tolist(), df.modification.values.tolist(), df.trap.values.tolist()))
  X = unwrap([[k]*5 for k in X])
  y = df.glycan.values.tolist()
  y = unwrap([[k]*5 for k in y])
  dset = SimpleDataset(X, y, transform_mz=transform_mz, transform_prec=transform_prec, transform_rt=transform_rt)
  dloader = torch.utils.data.DataLoader(dset, batch_size = 256, shuffle = False)
  df.index=df.reducing_mass.values.tolist()
  df.drop(['reducing_mass','binned_intensities', 'mz_remainder', 'RT2', 'mode', 'modification',
           'trap', 'glycan', 'glycan_type', 'lc'], axis = 1, inplace = True)
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
  preds = []
  conf = []
  for data in dataloader:
    mz_list, mz_remainder, precursor, glycan_type, rt, mode, lc, modification, trap, y = data
    mz_list = mz_list.to(device)
    mz_remainder = mz_remainder.to(device)
    mz_list = torch.stack([mz_list, mz_remainder], dim = 1)
    precursor = precursor.to(device)
    glycan_type = glycan_type.to(device)
    rt = rt.to(device)
    mode = mode.to(device)
    lc = lc.to(device)
    modification = modification.to(device)
    trap = trap.to(device)
    pred = model(mz_list, precursor, glycan_type, rt, mode, lc, modification, trap)
    if temp:
      pred = T_scaling(pred, temperature)
    pred = F.softmax(pred, dim = 1)
    pred = pred.cpu().detach().numpy()
    idx = np.argsort(pred, axis = 1)[:, ::-1]
    idx = idx[:, :k].tolist()
    preds.append(idx)
    pred = -np.sort(-pred)
    conf_idx = pred[:, :k].tolist()
    conf.append(conf_idx)
  preds = unwrap(preds)
  conf = unwrap(conf)
  preds = [[glycans[k] for k in j] for j in preds]
  return preds, conf

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
  labels = [0]
  keys = list(dicty.keys())
  for d in range(1,len(keys)):
    if abs(keys[d-1] - keys[d]) < 0.5:
      labels.append(k)
    else:
      k+=1
      labels.append(k)
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
  if m < 1500:
    return 0.5
  else:
    return 0.75

def mass_check(mass, glycan, libr = None, mode = 'negative', modification = 'reduced', 
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500):
  """determine whether glycan could explain m/z\n
  | Arguments:
  | :-
  | mass (float): observed m/z
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
  | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1500
  | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500\n
  | Returns:
  | :-
  | Returns True if glycan could explain mass and False if not
  """
  if libr is None:
    libr = lib
  if modification == 'permethylated':
    mz = glycan_to_mass(glycan, sample_prep = 'permethylated')
  else:
    try:
      mz = glycan_to_mass(glycan)
    except:
      return False
  if modification == 'reduced':
    mz += 1
  if mode == 'negative':
    og_list = [mz, mz + mass_dict['Acetate']]
  else:
    og_list = [mz, mz + mass_dict['Na+']]
  thresh = determine_threshold(mass)
  mz_list = []
  if mz > double_thresh:
    if mode == 'negative':
      mz_list += [m/2 - 0.5 for m in og_list]
    elif mode == 'positive':
      mz_list += [m/2 + 0.5 for m in og_list]
  if mz > triple_thresh:
    if mode == 'negative':
      mz_list += [m/3 - 0.66 for m in og_list]
    elif mode == 'positive':
      mz_list += [m/3 + 0.66 for m in og_list]
  if mz > quadruple_thresh:
    if mode == 'negative':
      mz_list += [m/4 - 0.75 for m in og_list]
    elif mode == 'positive':
      mz_list += [m/4 + 0.75 for m in og_list]
  mz_list += og_list
  return any([abs(mass-m) < thresh for m in mz_list])

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
  rt_label = sorted(rt_label_in)
  tt, tt_len = break_alert_loop(rt_label)
  medIdx = [t.index(np.percentile(t, 50, interpolation = 'nearest')) for t in tt]
  medIdx = [rt_label_in.index(tt[k][i]) if isinstance(tt[k][i], float) else rt_label_in.index(tt[k][i][0]) for k,i in enumerate(medIdx)]
  if intensity:
    idx_for_inty = [[rt_label_in.index(j) for j in t] for t in tt]
    return medIdx, idx_for_inty, tt_len
  else:
    return medIdx, [], tt_len

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
  if len(intensity) > 0:
    inty_check = True
  else:
    inty_check = False
  #sort in preparation for discontinuity check; bring everything in same order defined by sort_idx
  sort_idx = np.argsort(list(dicty.keys()))
  dicty = dict(sorted(dicty.items()))
  #detect mass groups of spectra by scanning for m/z discontinuities > 0.5
  labels = assign_dict_labels(dicty)
  unq_labels = list(set(labels))
  rt = [rt[k] for k in sort_idx]
  rt_labels = [[rt[k] for k in range(len(rt)) if labels[k]==j] for j in unq_labels]
  #detect retention groups of mass groups & retrieve indices representative spectra
  rt_idx, inty_idx, num_spectra = list(zip(*[get_rep_spectra(k, intensity = True) for k in rt_labels]))
  num_spectra = unwrap(num_spectra)
  if inty_check:
    intensity = [intensity[k] for k in sort_idx]
    intensity = [[intensity[k] for k in range(len(intensity)) if labels[k]==j] for j in unq_labels]
    intensity = [[[intensity[k][l] for l in j] if isinstance(j, list) else [intensity[k][j]] for j in inty_idx[k]] for k in range(len(intensity))]
    intensity = unwrap([[sum(j) for j in k] if isinstance(k[0], list) else [k] for k in intensity])
  #get m/z, predictions, and prediction confidence of those representative spectra
  keys = [[list(dicty.keys())[k] for k in range(len(labels)) if labels[k]==j] for j in unq_labels]
  keys = [[[keys[k][l] for l in j] if isinstance(j, list) else [keys[k][j]] for j in inty_idx[k]] for k in range(len(keys))]
  keys = unwrap([[np.mean(j) for j in k] if isinstance(k[0], list) else [k] for k in keys])
  values = [[list(dicty.values())[k] for k in range(len(labels)) if labels[k]==j] for j in unq_labels]
  rep_values = unwrap([[values[k][j] for j in rt_idx[k]] for k in range(len(values))])
  values = [[[values[k][l] for l in j] if isinstance(j, list) else [values[k][j]] for j in inty_idx[k]] for k in range(len(values))]
  values = unwrap([[average_dicts(j) for j in k] if isinstance(k[0], list) else [k] for k in values])
  rts = [[[rt_labels[k][l] for l in j] if isinstance(j, list) else [rt_labels[k][j]] for j in inty_idx[k]] for k in range(len(rt_labels))]
  rts = unwrap([[np.mean(j) for j in k] if isinstance(k[0], list) else [k] for k in rts])
  if inty_check:
    return keys, values, rts, num_spectra, rep_values, intensity
  else:
    return keys, values, rts, num_spectra, rep_values, []

def deduplicate_predictions(df):
  """removes/unifies duplicate predictions\n
  | Arguments:
  | :-
  | df (dataframe): df_out generated within wrap_inference\n
  | Returns:
  | :-
  | Returns a deduplicated dataframe
  """
  drop_idx = []
  #keep track of abundances if relevant
  if 'rel_abundance' in df.columns:
    struc_abundance = {(df.predictions.values.tolist()[k][0][0] if len(df.predictions.values.tolist()[k])>0 else "ID"+str(df.index.tolist()[k])): df.rel_abundance.values.tolist()[k] for k in range(len(df))}
  for k in range(len(df)-1):
    #basically, if we have the same structure with same mass & charge state twice or more --> just go with the most confident one
    if k not in drop_idx and len(df.predictions.values.tolist()[k])>0:
      check = [abs(df.index.tolist()[k]-df.index.tolist()[j]) < determine_threshold(df.index.tolist()[k])+0.5 and df.predictions.values.tolist()[k][0][0] == df.predictions.values.tolist()[j][0][0] if len(df.predictions.values.tolist()[j])>0 else False for j in range(len(df))]
      if any(check):
        idx = np.where(np.array(check) == True)[0]
        winner = idx[np.argmax([df.predictions.values.tolist()[i][0][1] for i in idx])]
        rest = [i for i in idx if i != winner]
        if len(rest)>0:
          drop_idx.append(rest)
          if 'rel_abundance' in df.columns:
            if len(df.predictions.values.tolist()[k]) > 0:
              struc_abundance[df.predictions.values.tolist()[k][0][0]] = sum([df.iat[i, df.columns.tolist().index('rel_abundance')] for i in idx])
  drop_idx = set(unwrap(drop_idx))
  df = df.drop([df.index.tolist()[k] for k in drop_idx], axis = 0)
  if 'rel_abundance' in df.columns:
    df.rel_abundance = [struc_abundance[df.predictions.values.tolist()[k][0][0]] if len(df.predictions.values.tolist()[k])>0 else struc_abundance["ID"+str(df.index.tolist()[k])] for k in range(len(df))]
  return df

def deduplicate_retention(df):
  """removes/unifies duplicate predictions based on retention time\n
  | Arguments:
  | :-
  | df (dataframe): df_out generated within wrap_inference\n
  | Returns:
  | :-
  | Returns a deduplicated dataframe
  """
  drop_idx = []
  for k in range(len(df)-1):
    if k not in drop_idx and len(df.predictions.values.tolist()[k])<1:
      check = [abs(df.RT.values.tolist()[k]-df.RT.values.tolist()[j]) < 0.5 and abs(df.index.tolist()[k]-df.index.tolist()[j]) < determine_threshold(df.index.tolist()[k])+0.5 if len(df.predictions.values.tolist()[j])<1 else False for j in range(len(df))]
      if any(check):
        idx = np.where(np.array(check) == True)[0]
        winner = idx[0]
        rest = [i for i in idx if i!=winner]
        if len(rest)>0:
          drop_idx.append(rest)
  drop_idx = set(unwrap(drop_idx))
  return df.drop([df.index.tolist()[k] for k in drop_idx], axis = 0)

def combinatorics(comp):
  """given a composition, create a crude approximation of possible B/C/Y/Z fragments\n
  | Arguments:
  | :-
  | comp (dict): composition in dictionary form\n
  | Returns:
  | :-
  | Returns a list of rough masses to check against fragments
  """
  clist = unwrap([[k]*v for k,v in comp.items()])
  verbose_clist = [abbrev_dict[x] if x in abbrev_dict else x for x in clist]
  all_combinations = set(unwrap([[comb for comb in combinations(verbose_clist, i)] for i in range(1, len(clist)+1)]))
  masses = [sum([mass_dict[k] for k in j]) for j in all_combinations]
  return masses + [k+18.01056 for k in masses]

def domain_filter(df_out, glycan_class, libr = None, mode = 'negative', modification = 'reduced',
                  mass_tolerance = 0.5, filter_out = None, df_use = None):
  """filters out false-positive predictions\n
  | Arguments:
  | :-
  | df_out (dataframe): df_out generated within wrap_inference
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
  | filter_out (list): list of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:None
  | df_use (dataframe): glycan database used to check whether compositions are valid; default: df_glycan\n
  | Returns:
  | :-
  | Returns a filtered prediction dataframe
  """
  if df_use is None:
    df_use = df_glycan
  if libr is None:
    libr = lib
  if modification == 'reduced':
    reduced = 1.0078
  else:
    reduced = 0
  multiplier = -1 if mode == 'negative' else 1
  df_out = adduct_detect(df_out, mode, modification)
  for k in range(len(df_out)):
    keep = []
    addy = df_out.charge.values.tolist()[k]*multiplier-1
    c = abs(df_out.charge.tolist()[k])
    assumed_mass = df_out.index.tolist()[k]*c + addy
    cmasses = np.array(combinatorics(df_out.composition.values.tolist()[k]))
    if len(df_out.predictions.values.tolist()[k]) > 0:
      current_preds = df_out.predictions.values.tolist()[k]
      to_append = True
    else:
      current_preds = [''.join(list(df_out.composition.values.tolist()[k].keys()))]
      to_append = False
    #check if it's a glycan spectrum
    if not np.any(np.abs(np.array(df_out.top_fragments.values.tolist()[k][:10])[:, None] - cmasses) < 1.5):
      df_out.iat[k,0] = ['remove']
      continue
    for i,m in enumerate(current_preds):
      m = m[0]
      truth = [True]
      #check diagnostic ions
      if 'Neu5Ac' in m:
        truth.append(any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Neu5Ac']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Neu5Ac']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
      if 'Neu5Gc' in m:
        truth.append(any([abs(mass_dict['Neu5Gc']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Neu5Gc']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Neu5Gc']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
      if 'Kdn' in m:
        truth.append(any([abs(mass_dict['Kdn']+(1.0078*multiplier)-j) < 1 or abs(assumed_mass-mass_dict['Kdn']-j) < 1 or abs(df_out.index.tolist()[k]-((mass_dict['Kdn']-addy)/c)-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
      if 'Neu5Gc' not in m:
        truth.append(not any([abs(mass_dict['Neu5Gc']+(1.0078*multiplier)-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j,float)]))
      if 'Neu5Ac' not in m and 'Neu5Gc' not in m:
        truth.append(not any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j,float)]))
      if 'Neu5Ac' not in m and (m.count('Fuc') + m.count('dHex') > 1):
        truth.append(not any([abs(mass_dict['Neu5Ac']+(1.0078*multiplier)-j) < 1 or abs(df_out.index.tolist()[k]-mass_dict['Neu5Ac']-j) < 1 for j in df_out.top_fragments.values.tolist()[k][:10] if isinstance(j,float)]))
      if 'S' in m and len(df_out.predictions.values.tolist()[k]) < 1:
        truth.append(any(['S' in (mz_to_composition2(t-reduced, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class,
                                  df_use = df_use, filter_out = filter_out)[0:1] or ({},))[0].keys() for t in df_out.top_fragments.values.tolist()[k][:20]]))
      #check fragment size distribution
      if c > 1:
        truth.append(any([j > df_out.index.values[k]*1.2 for j in df_out.top_fragments.values.tolist()[k][:15]]))
      if c == 1:
        truth.append(all([j < df_out.index.values[k]*1.1 for j in df_out.top_fragments.values.tolist()[k][:5]]))
      if len(df_out.top_fragments.values.tolist()[k]) < 5:
        truth.append(False)
      #check M-adduct for adducts
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
    df_out.iat[k,0] = keep
  idx = [k for k in range(len(df_out)) if 'remove' not in df_out.predictions.values.tolist()[k][:1]]
  return df_out.iloc[idx,:]

def backfill_missing(df):
  """finds rows with composition-only that match existing predictions wrt mass and RT and propagates\n
  | Arguments:
  | :-
  | df (dataframe): df_out generated within wrap_inference\n
  | Returns:
  | :-
  | Returns backfilled dataframe
  """
  str_dics = [stringify_dict(k) for k in df.composition]
  for k in range(len(df)):
    if not len(df.predictions.values.tolist()[k]) > 0:
      idx = [j for j in range(len(str_dics))
             if str_dics[j] == str_dics[k]
             and abs(df.index.tolist()[j]*abs(df.charge.values.tolist()[j])+(abs(df.charge.values.tolist()[j])-1)-df.index.tolist()[k]*abs(df.charge.values.tolist()[k])+(abs(df.charge.values.tolist()[k])-1)) < determine_threshold(df.index.tolist()[j])
             and abs(df.RT.values.tolist()[j]-df.RT.values.tolist()[k]) < 1]
      if len(idx) > 0:
        df.iat[k,0] = df.predictions.values.tolist()[idx[0]]
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
  if mode == 'negative':
    adduct = 'Acetate'
  else:
    adduct = 'Na+'
  adduct_mass = mass_dict[adduct]
  if modification == 'reduced':
    adduct_mass += 1.0078
  adduct_check = []
  for k in range(len(df)):
    if abs(composition_to_mass(df.composition.values.tolist()[k]) + adduct_mass - (df.index.tolist()[k]*abs(df.charge.values.tolist()[k])+(abs(df.charge.values.tolist()[k])-1))) < 0.5:
      adduct_check.append(adduct)
    else:
      adduct_check.append(np.nan)
  df['adduct'] = adduct_check
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
  out_p = []
  out_c = []
  for c in range(len(pred_chunks)):
    this_pred = pred_chunks[c]
    this_conf = conf_chunks[c]
    combs = [{this_pred[j][m]:this_conf[j][m] for m in range(len(this_pred[j]))} for j in range(len(this_pred))]
    combs = average_dicts(combs, mode = 'max')
    combs = dict(sorted(combs.items(), key = operator.itemgetter(1), reverse = True))
    out_p.append(list(combs.keys()))
    out_c.append(list(combs.values()))
  return out_p, out_c

def map_to_comp(mass, reduced, mode, mass_tolerance, glycan_class, df_use, filter_out):
  """robust workflow to map an m/z value to a composition\n
  | Arguments:
  | :-
  | mass (float): observed m/z value
  | reduced (int): 1 if modification = 'reduced' and 0 otherwise
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
  | df_use (dataframe): database to use for searching valid compositions; should only be used with some version of df_glycan
  | filter_out (list): list of composition elements that are not allowed in the final composition
  | libr (list): library of monosaccharides\n
  | Returns:
  | :-
  | Returns composition (as a dict)
  """
  comp = mz_to_composition2(mass-reduced, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class, df_use = df_use, filter_out = filter_out)
  if len(comp) < 1:
    new_mass = (mass+0.5)*2-reduced if mode == 'negative' else (mass-0.5)*2-reduced
    comp = mz_to_composition2(new_mass, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class, df_use = df_use, filter_out = filter_out)
  return comp

def impute(df_out):
  """searches for specific isomers that could be added to the prediction dataframe\n
  | Arguments:
  | :-
  | df_out (dataframe): prediction dataframe generated within wrap_inference\n
  | Returns:
  | :-
  | Returns prediction dataframe with imputed predictions (if possible)
  """
  for k in range(len(df_out)):
    for j in range(len(df_out)):
      if len(df_out.predictions.values.tolist()[k]) > 0 and len(df_out.predictions.values.tolist()[j]) < 1:
        if abs(df_out.index.tolist()[k]+(16.0051/max([abs(df_out.charge.values.tolist()[k]),1]))-df_out.index.tolist()[j]) < 0.5 and 'Neu5Gc' in df_out.composition.values.tolist()[j].keys():
          df_out.iat[j,0] = [(m[0].replace('Neu5Ac', 'Neu5Gc', 1),) for m in df_out.predictions.values.tolist()[k]]
        elif abs(df_out.index.tolist()[k]-(16.0051/max([abs(df_out.charge.values.tolist()[k]),1]))-df_out.index.tolist()[j]) < 0.5 and 'Neu5Ac' in df_out.composition.values.tolist()[j].keys():
          df_out.iat[j,0] = [(m[0].replace('Neu5Gc', 'Neu5Ac', 1),) for m in df_out.predictions.values.tolist()[k]]
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
  for k in range(len(df_out)):
    if len(df_out.predictions.values.tolist()[k]) < 1:
      check_mass = df_out.index.tolist()[k]-reduced
      diffs = [abs(j-check_mass) for j in mass_dic.keys()]
      if min(diffs) < 0.5:
        possible = mass_dic[list(mass_dic.keys())[np.argmin(diffs)]]
        df_out.iat[k,0] = [(m,) for m in possible]
  return df_out

def make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = 'Mammalia'):
  """generates a mass dict that can be used in the possibles() function\n
  | Arguments:
  | :-
  | glycans (list): glycans used for training CandyCrunch
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
  | filter_out (list): list of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen)
  | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan
  | taxonomy_class (string): which taxonomic class to use for selecting possible glycans; default:'Mammalia'\n
  | Returns:
  | :-
  | Returns a dictionary of form mass : list of glycans
  """
  exp_glycans = df_use.glycan.values.tolist()
  class_glycans = [k for k in glycans if enforce_class(k, glycan_class)]
  exp_glycans = list(set(class_glycans+exp_glycans))
  masses = []
  for k in exp_glycans:
    try:
      if not any([j in glycan_to_composition(k).keys() for j in filter_out]):
        masses.append(glycan_to_mass(k))
      else:
        masses.append(9999)
    except:
      masses.append(9999)
  unq_masses = set(masses)
  mass_dic = {u:[exp_glycans[j] for j in [i for i,m in enumerate(masses) if m == u]] for u in unq_masses}
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
  df_out['true_mass'] = [df_out.index.tolist()[j]*abs(df_out.charge.values.tolist()[j])+(abs(df_out.charge.values.tolist()[j])-1) for j in range(len(df_out))]
  idx = df_out.index.tolist()
  df_out.sort_values(by = 'true_mass', inplace = True)
  for k in reversed(range(len(df_out))):
    preds = df_out.iloc[k, 0]
    if len(preds) > 0:
      rest_top1 = set([df_out.predictions.values.tolist()[j][0][0] for j in range(k) if len(df_out.predictions.values.tolist()[j]) > 0 and df_out.evidence.values.tolist()[j] == 'strong'])
      for i,p in enumerate(preds):
        p = list(p)
        if len(p) == 1:
          p.append(0)
        p[1] += 0.1*sum([subgraph_isomorphism(p[0], t, libr = libr, wildcards_ptm = True)*(not t == p[0]) for t in rest_top1])
        p = tuple(p)
        preds[i] = p
      preds = sorted(preds, key = lambda x: x[1], reverse = True)
      total = sum([p[1] for p in preds])
      if total > 1:
        df_out.iat[k,0] = [(p[0],p[1]/total) for p in preds][:5]
      else:
        df_out.iat[k,0] = [(p[0],p[1]) for p in preds][:5]
  df_out.drop(['true_mass'], axis = 1, inplace = True)
  return df_out.loc[idx,:]

def wrap_inference(filename, glycan_class, model = candycrunch, glycans = glycans, libr = None, filepath = fp_in + "for_prediction/", bin_num = 2048,
                   frag_num = 100, mode = 'negative', modification = 'reduced', lc = 'PGC', trap = 'linear',
                   pred_thresh = 0.01, temperature = temperature, spectra = False, get_missing = False, mass_tolerance = 0.5, extra_thresh = 0.2,
                   filter_out = ['Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'], supplement = True, experimental = True, mass_dic = None,
                   taxonomy_class = 'Mammalia', df_use = None):
  """wrapper function to get & curate CandyCrunch predictions\n
  | Arguments:
  | :-
  | filename (string or dataframe): if string, filepath +filename+ ".xlsx" must point to file; datafile containing extracted spectra
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
  | model (PyTorch): trained CandyCrunch model
  | glycans (list): full list of glycans used for training CandyCrunch; don't change default without changing model
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | filepath (string): absolute filepath to filename, used as filepath +filename+ ".xlsx"
  | bin_num (int): number of bins for binning; don't change; default: 2048
  | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
  | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
  | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
  | temperature (float): the temperature factor used to calibrate logits; default:1.15
  | spectra (bool): whether to also output the actual spectra used for prediction; default:False
  | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
  | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
  | extra_thresh (float): prediction confidence threshold at which to allow cross-class predictions (e.g., N-glycans in O-glycan samples); default:0.2
  | filter_out (list): list of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:['Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me']
  | supplement (bool): whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True
  | experimental (bool): whether to impute missing predictions via database searches etc.; default:True
  | mass_dic (dict): dictionary of form mass : list of glycans; will be generated internally
  | taxonomy_class (string): which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia'
  | df_use (dataframe): sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan\n
  | Returns:
  | :-
  | Returns dataframe of predictions for spectra in file
  """
  print("Your chosen settings are: " + glycan_class + " glycans, " + mode + " ion mode, " + modification + " glycans, " + lc + " LC, and " + trap + " ion trap. If any of that seems off to you, please restart with correct parameters.")
  if libr is None:
    libr = lib
  if df_use is None:
    df_use = df_glycan[(df_glycan.glycan_type==glycan_class) & (df_glycan.Class.str.contains(taxonomy_class))]
  if isinstance(filename, str):
    loaded_file = pd.read_excel(filepath + filename + ".xlsx")
  else:
    loaded_file = filename
  if modification == 'reduced':
    reduced = 1.0078
  else:
    reduced = 0
  intensity = True if 'intensity' in loaded_file.columns else False
  multiplier = -1 if mode == 'negative' else 1
  #prepare file for processing
  loaded_file = loaded_file.iloc[[k for k in range(len(loaded_file)) if loaded_file.peak_d.values.tolist()[k][-1] == '}' and loaded_file.RT[k] > 2],:]
  loaded_file.peak_d = [ast.literal_eval(k) if k[-1] == '}' else np.nan for k in loaded_file.peak_d]
  loaded_file.dropna(subset = ['peak_d'], inplace = True)
  loaded_file.reducing_mass = [k+np.random.uniform(0.00001, 10**(-20)) for k in loaded_file.reducing_mass]
  if intensity:
    inty = loaded_file.intensity.values.tolist()
  else:
    inty = []
  coded_class = 0 if glycan_class == 'O' else 1 if glycan_class == 'N' else 2 if any([glycan_class == 'free', glycan_class == 'lipid']) else 3
  spec_dic = {loaded_file.reducing_mass.values.tolist()[k]:loaded_file.peak_d.values.tolist()[k] for k in range(len(loaded_file))}
  #group spectra by mass/retention isomers and process them for being inputs to CandyCrunch
  keys, values, RT, num_spectra, rep_values, intensity = build_mean_dic(spec_dic, loaded_file.RT.values.tolist(), inty)
  loader, df_out = process_for_inference(keys, values, RT, num_spectra, coded_class, mode = mode, modification = modification, lc = lc, trap = trap, bin_num = bin_num)
  df_out.peak_d = rep_values
  #predict glycans from spectra
  preds, pred_conf = get_topk(loader, model, glycans, temp = True, temperature = temperature)
  preds, pred_conf = average_preds(preds, pred_conf)
  if intensity:
    df_out['rel_abundance'] = intensity
  df_out['predictions'] = [[(preds[k][j], pred_conf[k][j]) for j in range(len(preds[k]))] for k in range(len(preds))]
  #check correctness of glycan class & mass
  df_out.predictions = [[gly for gly in v if enforce_class(gly[0], glycan_class, gly[1], extra_thresh = extra_thresh) and gly[1] > pred_thresh] for v in df_out.predictions]
  df_out.predictions = [[(gly[0], round(gly[1],4)) for gly in df_out.predictions.values.tolist()[v] if mass_check(df_out.index.tolist()[v], gly[0], libr = libr, modification = modification, mode = mode)][:5] for v in range(len(df_out))]
  #get composition of predictions
  df_out['composition'] = [glycan_to_composition(g[0][0]) if len(g) > 0 and len(g[0]) > 0 else np.nan for g in df_out.predictions]
  df_out.composition = [k if isinstance(k, dict) else map_to_comp(df_out.index.tolist()[i], reduced, mode, mass_tolerance, glycan_class, df_use, filter_out) for i,k in enumerate(df_out.composition.values.tolist())]
  df_out.composition = [np.nan if isinstance(k, list) and len(k) < 1 else k[0] if isinstance(k, list) and len(k) > 0 else k for k in df_out.composition]
  df_out.dropna(subset = ['composition'], inplace = True)
  #calculate precursor ion charge
  df_out['charge'] = [round(composition_to_mass(df_out.composition.values.tolist()[k])/df_out.index.values.tolist()[k])*multiplier for k in range(len(df_out))]
  df_out.RT = [round(k,2) for k in df_out.RT.values.tolist()]
  cols = ['predictions', 'composition', 'num_spectra', 'charge', 'RT', 'peak_d']
  if intensity:
    cols += ['rel_abundance']
  df_out = df_out[cols]
  #fill up possible gaps of singly-/doubly-charged spectra of the same structure
  df_out = backfill_missing(df_out)
  #extract & sort the top 100 fragments by intensity
  top_frags = [sorted(k.items(), key = lambda x: x[1], reverse = True)[:frag_num] for k in df_out.peak_d]
  df_out['top_fragments'] = [[round(j[0],4) for j in k] for k in top_frags]
  #filter out wrong predictions via diagnostic ions etc.
  df_out = domain_filter(df_out, glycan_class, libr = libr, mode = mode, filter_out = filter_out, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
  #deduplicate identical predictions for different spectra
  df_out = deduplicate_predictions(df_out)
  df_out['evidence'] = ['strong' if len(k) > 0 else np.nan for k in df_out.predictions]
  #construct biosynthetic network from top1 predictions and check whether intermediates could be a fit for some of the spectra
  if supplement:
    #if len(df_out) > 200:
    #  print("Very large number of glycans detected; biosynthetic network construction could take a while. If you're in a hurry, restart with supplement=False")
    try:
      df_out = supplement_prediction(df_out, glycan_class, libr = libr, mode = mode, modification = modification)
      df_out.evidence = ['medium' if isinstance(df_out.evidence.values.tolist()[k], float) and len(df_out.predictions.values.tolist()[k]) > 0 else df_out.evidence.values.tolist()[k] for k in range(len(df_out))]
    except:
      pass
  #check for Neu5Ac-Neu5Gc swapped structures and search for glycans within SugarBase that could explain some of the spectra
  if experimental:
    df_out = impute(df_out)
    if mass_dic is None:
        mass_dic = make_mass_dic(glycans, glycan_class, filter_out, df_use, taxonomy_class = taxonomy_class)
    df_out = possibles(df_out, mass_dic, reduced)
    df_out.evidence = ['weak' if isinstance(df_out.evidence.values.tolist()[k], float) and len(df_out.predictions.values.tolist()[k]) > 0 else df_out.evidence.values.tolist()[k] for k in range(len(df_out))]
  #filter out wrong predictions via diagnostic ions etc.
  if supplement or experimental:
    df_out = domain_filter(df_out, glycan_class, libr = libr, mode = mode, filter_out = filter_out, modification = modification, mass_tolerance = mass_tolerance, df_use = df_use)
    df_out.predictions = [[(k[0].replace('-ol','').replace('1Cer',''), k[1]) if len(k) > 1 else (k[0].replace('-ol','').replace('1Cer',''),) for k in j] if len(j) > 0 else j for j in df_out.predictions]
  #keep or remove spectra that still lack a prediction after all this
  if get_missing:
    pass
  else:
    idx = [k for k in range(len(df_out)) if len(df_out.predictions.values.tolist()[k]) > 0]
    df_out = df_out.iloc[idx,:]
  #reprioritize predictions based on how well they are explained by biosynthetic precursors in the same file (e.g., core 1 O-glycan making extended core 1 O-glycans more likely)
  df_out = canonicalize_biosynthesis(df_out, libr, pred_thresh)
  spectra_out = df_out.peak_d.values.tolist()
  df_out.drop(['peak_d'], axis = 1, inplace = True)
  #clean-up
  df_out.composition = [glycan_to_composition(k[0][0]) if len(k) > 0 else df_out.composition.values.tolist()[i] for i,k in enumerate(df_out.predictions)]
  df_out.charge = [round(composition_to_mass(df_out.composition.values.tolist()[k])/df_out.index.tolist()[k])*multiplier for k in range(len(df_out))]
  #normalize relative abundances if relevant
  if intensity:
    df_out.rel_abundance = [k/sum(df_out.rel_abundance.values.tolist())*100 for k in df_out.rel_abundance]
  if spectra:
    return df_out, spectra_out
  else:
    return df_out

def supplement_prediction(df_in, glycan_class, libr = None, mode = 'negative', modification = 'reduced'):
  """searches for biosynthetic precursors of CandyCrunch predictions that could explain peaks\n
  | Arguments:
  | :-
  | df_in (pandas dataframe): output file produced by wrap_inference
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
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
  preds  = [k[0][0] for k in df.predictions if len(k) > 0]
  if glycan_class == 'free':
    preds = [k+'-ol' for k in preds]
    libr = expand_lib(libr, preds)
    permitted_roots = {"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}
  elif glycan_class == 'lipid':
    permitted_roots = {"Glc", "Gal"}
  elif glycan_class == 'O':
    permitted_roots = {'GalNAc', 'Fuc', 'Man'}
  elif glycan_class == 'N':
    permitted_roots = {'GlcNAc(b1-4)GlcNAc'}
  net = construct_network(preds, permitted_roots = permitted_roots, libr = libr)
  if glycan_class == 'free':
    net = evoprune_network(net, libr = libr)
  unexplained_idx = [k for k in range(len(df)) if len(df.predictions.values.tolist()[k]) < 1]
  unexplained = [df.index.tolist()[k] for k in unexplained_idx]
  new_nodes = [k for k in net.nodes() if k not in preds]
  explained_idx = [np.where([mass_check(j, k, modification = modification, mode = mode) for j in unexplained])[0] for k in new_nodes]
  explained_idx = [[unexplained_idx[k] for k in j] for j in explained_idx]
  new_nodes = [(new_nodes[k], explained_idx[k]) for k in range(len(new_nodes)) if len(explained_idx[k]) > 0]
  explained = {k:[] for k in list(set(unwrap(explained_idx)))}
  for n in new_nodes:
    for nn in n[1]:
      explained[nn].append(n[0])
  pred_idx = df.columns.tolist().index('predictions')
  for k in explained.keys():
    df.iat[k,pred_idx] = [(t,) for t in explained[k]]#.sort(key=len)
  return df

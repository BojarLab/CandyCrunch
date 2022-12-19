import numpy as np
import pandas as pd
import numpy_indexed as npi
from glycowork.motif.processing import enforce_class, get_lib
from glycowork.motif.tokenization import mapping_file, glycan_to_composition, calculate_theoretical_mass, mz_to_composition, mz_to_composition2, composition_to_mass
from glycowork.network.biosynthesis import construct_network, plot_network, evoprune_network
from glycowork.glycan_data.loader import unwrap, stringify_dict, lib, df_glycan
from CandyCrunch.model import SimpleDataset
import ast
import copy
import torch
import torch.nn.functional as F

fp_in = "drive/My Drive/CandyCrunch/"

#choose the correct computing architecture
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

mass_dict = dict(zip(mapping_file.composition, mapping_file["underivatized_monoisotopic"]))
temperature = torch.Tensor([1.2097]).to(device)
def T_scaling(logits, temperature):
  return torch.div(logits, temperature)

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
  filled_bins = np.digitize(np.array(keys,dtype='float32'),frames,right=True)
  mz_remainder = keys-frames[filled_bins-1]
  unq, ids = np.unique(filled_bins, return_inverse=True)
  vals = np.array(list(map(peak_d.get, keys)))
  a2 = mz_remainder[npi.group_by(ids).argmax(vals)[1]]
  for b,s,m in zip(unq,np.bincount(ids, np.array(list(map(peak_d.get, keys)))), np.bincount(range(len(a2)), a2)):
    out_list[b-1] = s
    out_list2[b-1] = m
  return out_list, out_list2

def process_for_inference(df_in, glycan_class, mode = 'negative', modification = 'reduced', lc = 'PGC',
                          trap = 'linear', min_mz = 39.714, max_mz = 3000, bin_num = 2048,
                          #trap = 'linear', min_mz = 64.95, max_mz = 2500, bin_num = 2048,
                          intensity = False):
  """processes extracted spectra for them being inputs to CandyCrunch\n
  | Arguments:
  | :-
  | df_in (dataframe): extracted spectra with columns: reducing_mass, peak_d, and (optional) RT, (optional) intensity
  | glycan_class (int): 0 = O-linked, 1 = N-linked, 2 = lipid/free, 3 = other
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
  | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
  | min_mz (float): minimal m/z used for binning; don't change; default:39.714
  | max_mz (float): maximal m/z used for binning; don't change; default:3000
  | bin_num (int): number of bins for binning; don't change; default: 2048
  | intensity (bool): whether to use intensity for relative abundance estimation; default: False\n
  | Returns:
  | :-
  | (1) a dataloader used for model prediction
  | (2) a list of retention times
  | (3) a list of intensities (if intensity = True, else an empty list)
  """
  df = copy.deepcopy(df_in)
  df['glycan_type'] = [glycan_class]*len(df)
  df['glycan'] = [0]*len(df)
  df['mode'] = [0]*len(df) if mode == 'negative' else [1]*len(df)
  df['lc'] = [0]*len(df) if lc == 'PGC' else [1]*len(df) if lc == 'C18' else [2]*len(df)
  df['modification'] = [0]*len(df) if modification == 'reduced' else [1]*len(df) if modification == 'permethylated' else [2]*len(df)
  df['trap'] = [0]*len(df) if trap == 'linear' else [1]*len(df) if trap == 'orbitrap' else [2]*len(df) if trap == 'amazon' else [3]*len(df)
  df.peak_d = [ast.literal_eval(k) if k[-1] == '}' else np.nan for k in df.peak_d.values.tolist()]
  df.dropna(subset = ['peak_d'], inplace=True)
  #intensity normalization
  df.peak_d = [{k: v / sum(d.values()) for k, v in d.items()} for d in df.peak_d.values.tolist()]
  if 'RT' not in df.columns.tolist():
    df['RT'] = [0]*len(df)
  #retention time normalization
  df['RT2'] = [k/max(max(df.RT.values.tolist()),30) for k in df.RT.values.tolist()]
  #intensity binning
  step = (max_mz - min_mz) / (bin_num - 1)
  frames = np.array([min_mz + step * i for i in range(bin_num)])
  df['binned_intensities'], df['mz_remainder'] = list(zip(*[bin_intensities(df.peak_d.values.tolist()[k], frames) for k in range(len(df))]))
  #dataloader generation
  X = list(zip(df.binned_intensities.values.tolist(),df.mz_remainder.values.tolist(),df.reducing_mass.values.tolist(),df.glycan_type.values.tolist(),
               df.RT2.values.tolist(), df['mode'].values.tolist(), df.lc.values.tolist(), df.modification.values.tolist(), df.trap.values.tolist()))
  y = df.glycan.values.tolist()
  dset = SimpleDataset(X, y)
  dloader = torch.utils.data.DataLoader(dset, batch_size = 256, shuffle = False)
  if intensity:
    return dloader, df.RT.values.tolist(), df.intensity.values.tolist()
  else:
    return dloader, df.RT.values.tolist(), []

def get_topk(dataloader, model, glycans, k=10, temp = False):
  """yields topk CandyCrunch predictions for spectra in dataloader\n
  | Arguments:
  | :-
  | dataloader (PyTorch): dataloader from process_for_inference
  | model (PyTorch): trained CandyCrunch model
  | glycans (list): full list of glycans used for training CandyCrunch
  | k (int): how many top predictions to provide for each spectrum; default:10
  | temp (bool): whether to calibrate logits by temperature factor; default:False\n
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
    thresh = 0.5
  elif 1500 < m < 2500:
    thresh = 1.0
  elif 2500 < m < 3500:
    thresh = 1.0
  else:
    thresh = 1.0
  return thresh

def mass_check(mass, glycan, libr = None, mode = 'negative', modification = 'reduced', 
               double_thresh = 900, triple_thresh = 1200, quadruple_thresh = 3500):
  """determine whether glycan could explain m/z\n
  | Arguments:
  | :-
  | mass (float): observed m/z
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | double_thresh (float): mass threshold over which to consider doubly-charged ions; default:900
  | triple_thresh (float): mass threshold over which to consider triply-charged ions; default:1200
  | quadruple_thresh (float): mass threshold over which to consider quadruply-charged ions; default:3500\n
  | Returns:
  | :-
  | Returns True if glycan could explain mass and False if not
  """
  if libr is None:
    libr = lib
  if modification == 'permethylated':
    mz = calculate_theoretical_mass(glycan, libr = libr, sample_prep = 'permethylated', go_fast = True)
  else:
    mz = calculate_theoretical_mass(glycan, libr = libr, go_fast = True)
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
  rt_label = list(sorted(rt_label_in))
  tt, tt_len = break_alert_loop(rt_label)
  medIdx = [t.index(np.percentile(t, 50, interpolation = 'nearest')) for t in tt]
  medIdx = [rt_label_in.index(tt[k][i]) if isinstance(tt[k][i], float) else rt_label_in.index(tt[k][i][0]) for k,i in enumerate(medIdx)]
  if intensity:
    idx_for_inty = [[rt_label_in.index(j) for j in t] for t in tt]
    return medIdx, idx_for_inty, tt_len
  else:
    return medIdx, [], tt_len

def build_mean_dic(dicty, rt, pred_conf, intensity, libr = None, glycan_class = 'O', pred_thresh = 0.1, mode = 'negative', modification = 'reduced',
                   get_missing = False, mass_tolerance = 0.5, filter_out = None, df_use = None):
  """organizes spectrum predictions into 1 representative prediction for a spectrum cluster\n
  | Arguments:
  | :-
  | dicty (dict): dictionary of form m/z : list of predictions
  | rt (list): list of retention times
  | intensity (list): list of intensities
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"; default: "O"
  | pred_thresh (float): prediction confidence threshold used for filtering; default:0.1
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
  | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
  | filter_out (list): list of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:None
  | df_use (dataframe): glycan database used to check whether compositions are valid; default: df_glycan\n
  | Returns:
  | :-
  | (1) dictionary of form m/z : list of remaining predictions with their prediction confidence
  | (2) list of number of spectra per item in dictionary
  """
  if libr is None:
    libr = lib
  if df_use is None:
    df_use = df_glycan
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
  pred_conf = [pred_conf[k] for k in sort_idx]
  rt_labels = [[rt[k] for k in range(len(rt)) if labels[k]==j] for j in unq_labels]
  #detect retention groups of mass groups & retrieve indices representative spectra
  rt_idx, inty_idx, num_spectra = list(zip(*[get_rep_spectra(k, intensity = inty_check) for k in rt_labels]))
  num_spectra = unwrap(num_spectra)
  if inty_check:
    intensity = [intensity[k] for k in sort_idx]
    intensity = [[intensity[k] for k in range(len(intensity)) if labels[k]==j] for j in unq_labels]
    intensity = [[[intensity[k][l] for l in j] if isinstance(j, list) else [intensity[k][j]] for j in inty_idx[k]] for k in range(len(intensity))]
    intensity = unwrap([[sum(j) for j in k] if isinstance(k[0], list) else [k] for k in intensity])
  #get m/z, predictions, and prediction confidence of those representative spectra
  keys = [[list(dicty.keys())[k] for k in range(len(labels)) if labels[k]==j] for j in unq_labels]
  keys = unwrap([[keys[k][j] for j in rt_idx[k]] for k in range(len(keys))])
  values = [[list(dicty.values())[k] for k in range(len(labels)) if labels[k]==j] for j in unq_labels]
  values = unwrap([[values[k][j] for j in rt_idx[k]] for k in range(len(values))])
  pred_conf = [[pred_conf[k] for k in range(len(labels)) if labels[k]==j] for j in unq_labels]
  pred_conf = unwrap([[pred_conf[k][j] for j in rt_idx[k]] for k in range(len(pred_conf))])
  num_spectra = {keys[k]:num_spectra[k] for k in range(len(keys))}
  #make dictionary of m/z : (predictions, prediction confidences, [intensities])
  if inty_check:
    reduced_dic = {keys[k]:list(zip(values[k], pred_conf[k], [intensity[k]]*len(values[k]))) for k in range(len(keys))}
  else:
    reduced_dic = {keys[k]:list(zip(values[k], pred_conf[k])) for k in range(len(keys))}
  #filter out predictions of (i) wrong glycan class, (ii) too low prediction confidence, (iii) wrong mass
  ranking = {}
  for k,v in reduced_dic.items():
    if intensity:
      inty = v[0][2]
    g = [gly for gly in v if enforce_class(gly[0], glycan_class) and gly[1] > pred_thresh]
    g = [(gly[0], round(gly[1],4)) for gly in g if mass_check(k, gly[0], libr = libr, modification = modification, mode = mode)]
    #get composition of predictions
    if len(g)>0:
      if intensity:
        ranking[k] = ([j[:2] for j in g], glycan_to_composition(g[0][0], libr = libr, go_fast = True), inty)
      else:
        ranking[k] = (g, glycan_to_composition(g[0][0], libr = libr, go_fast = True))
    #if no valid prediction (and get_missing = True), also keep representative spectra with a valid glycan composition
    else:
      if get_missing:
        if modification == 'reduced':
          k_c = k-1
        else:
          k_c = k
        comp = mz_to_composition2(k_c, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class,
                                 df_use = df_use, filter_out = filter_out, libr = libr)
        if len(comp) < 1:
          k_c2 = (k_c+0.5) * 2 if mode == 'negative' else (k_c-0.5) * 2 
          comp = mz_to_composition2(k_c2, mode = mode, mass_tolerance = mass_tolerance/2, glycan_class = glycan_class,
                                 df_use = df_use, filter_out = filter_out, libr = libr)
        if len(comp)>0:
          if intensity:
            ranking[k] = ([], comp[0], inty)
          else:
            ranking[k] = ([], comp[0])
  return ranking, [num_spectra[k] for k in ranking.keys()]

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
  drop_idx = list(set(unwrap(drop_idx)))
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
  drop_idx = list(set(unwrap(drop_idx)))
  df = df.drop([df.index.tolist()[k] for k in drop_idx], axis = 0)
  return df

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
  keep = []
  if modification == 'reduced':
    reduced = 1
  else:
    reduced = 0
  for k in range(len(df_out)):
    truth = [True]
    #check diagnostic ions
    if 'Neu5Ac' in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid']:
      truth.append(any([abs(290-j) < 1 or abs(df_out.index.tolist()[k]-291-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
    if 'Neu5Gc' in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid']:
      truth.append(any([abs(306-j) < 1 or abs(df_out.index.tolist()[k]-307-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
    if 'Kdn' in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid']:
      truth.append(any([abs(249-j) < 1 or abs(df_out.index.tolist()[k]-250-j) < 1 for j in df_out.top_fragments.values.tolist()[k] if isinstance(j,float)]))
    if 'Neu5Gc' not in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid']:
      truth.append(not any([abs(306-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j,float)]))
    if 'Neu5Ac' not in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid'] and 'Neu5Gc' not in df_out.composition.values.tolist()[k].keys():
      truth.append(not any([abs(290-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:5] if isinstance(j,float)]))
    if 'S' in df_out.composition.values.tolist()[k].keys() and glycan_class in ['O', 'free', 'lipid'] and len(df_out.predictions.values.tolist()[k]) < 1:
      truth.append(any(['S' in (mz_to_composition2(t-reduced, libr = libr, mode = mode, mass_tolerance = mass_tolerance, glycan_class = glycan_class,
                                 df_use = df_use, filter_out = filter_out)[0:1] or ({},))[0].keys() for t in df_out.top_fragments.values.tolist()[k][:20]]))
    #check fragment size distribution
    if df_out.charge.values.tolist()[k] > 1:
      truth.append(any([j > df_out.index.values[k]*1.2 for j in df_out.top_fragments.values.tolist()[k][:15]]))
    if df_out.charge.values.tolist()[k] == 1:
      truth.append(all([j < df_out.index.values[k]*1.1 for j in df_out.top_fragments.values.tolist()[k][:5]]))
    ##if len(df_out.top_fragments.values.tolist()[k])<20:
    ##  truth.append(False)
    #check M-adduct for adducts
    if isinstance(df_out.adduct.values.tolist()[k], str):
      truth.append(any([abs(df_out.index.tolist()[k]-mass_dict[df_out.adduct.values.tolist()[k]]-j) < 0.5 for j in df_out.top_fragments.values.tolist()[k][:10]]))
    if all(truth):
      keep.append(k)
  df_out = df_out.iloc[keep,:]
  return df_out

def backfill_missing(df):
  """finds rows with composition-only that match existing predictions and propagates\n
  | Arguments:
  | :-
  | df (dataframe): df_out generated within wrap_inference\n
  | Returns:
  | :-
  | Returns backfilled dataframe
  """
  str_dics = [stringify_dict(k) for k in df.composition.values.tolist()]
  for k in range(len(df)):
    if not len(df.predictions.values.tolist()[k]) > 0:
      idx = [j for j in range(len(str_dics)) if str_dics[j] == str_dics[k] and abs(df.index.tolist()[j]-df.index.tolist()[k]) > determine_threshold(df.index.tolist()[j])]
      if len(idx) > 0:
        df.iat[k,0] = df.predictions.values.tolist()[idx[0]]
  return df

def adduct_detect(df, mode, modification):
  """checks which spectra contains adducts and records them\n
  | Arguments:
  | :-
  | df (dataframe): df_out generated within wrap_inference
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'\n
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
    adduct_mass += 1
  adduct_check = []
  for k in range(len(df)):
    if abs(composition_to_mass(df.composition.values.tolist()[k]) + adduct_mass - df.index.tolist()[k]) < 0.5:
      adduct_check.append(adduct)
    else:
      adduct_check.append(np.nan)
  df['adduct'] = adduct_check
  return df

def wrap_inference(filename, glycan_class, model, glycans, libr = None, filepath = fp_in + "for_prediction/", bin_num = 2048,
                   frag_num = 100, mode = 'negative', modification = 'reduced', lc = 'PGC', trap = 'linear',
                   pred_thresh = 0.01, spectra = False, get_missing = False, mass_tolerance = 0.5,
                   filter_out = ['Kdn', 'P', 'HexA', 'Pen', 'HexN']):
  """wrapper function to get & curate CandyCrunch predictions\n
  | Arguments:
  | :-
  | filename (string or dataframe): if string, filepath +filename+ ".xlsx" must point to file; datafile containing extracted spectra
  | glycan_class (string): glycan class as string, options are "O", "N", "lipid", "free", "other"
  | model (PyTorch): trained CandyCrunch model
  | glycans (list): full list of glycans used for training CandyCrunch
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | filepath (string): absolute filepath to filename, used as filepath +filename+ ".xlsx"
  | bin_num (int): number of bins for binning; don't change; default: 2048
  | frag_num (int): how many top fragments to show in df_out per spectrum; default:100
  | mode (string): mass spectrometry mode, either 'negative' or 'positive'; default: 'negative'
  | modification (string): chemical modification of glycans; options are 'reduced', 'permethylated' or 'other'/'none'; default:'reduced'
  | lc (string): type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC'
  | trap (string): type of ion trap; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear'
  | pred_thresh (float): prediction confidence threshold used for filtering; default:0.01
  | spectra (bool): whether to also output the actual spectra used for prediction; default:False
  | get_missing (bool): whether to also organize spectra without a matching prediction but a valid composition; default:False
  | mass_tolerance (float): the general mass tolerance that is used for composition matching; default:0.5
  | filter_out (list): list of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:['Kdn', 'P', 'HexA', 'Pen', 'HexN']\n
  | Returns:
  | :-
  | Returns dataframe of predictions for spectra in file
  """
  if libr is None:
    libr = lib
  if isinstance(filename, str):
    loaded_file = pd.read_excel(filepath + filename + ".xlsx")
  else:
    loaded_file = filename
  intensity = True if 'intensity' in loaded_file.columns else False
  loaded_file.reducing_mass = [k+np.random.uniform(0.00001, 10**(-20)) for k in loaded_file.reducing_mass.values.tolist()]
  coded_class = 0 if glycan_class == 'O' else 1 if glycan_class == 'N' else 2 if any([glycan_class == 'free', glycan_class == 'lipid']) else 3
  loader, RT, inty = process_for_inference(loaded_file, coded_class, mode = mode, modification = modification,
                                           lc = lc, trap = trap, bin_num = bin_num, intensity = intensity)
  preds, pred_conf = get_topk(loader, model, glycans, temp = True)
  preds = {loaded_file.reducing_mass.values.tolist()[k]:preds[k] for k in range(len(preds))}
  out, num_spectra = build_mean_dic(preds, RT, pred_conf, inty, libr = libr, glycan_class = glycan_class, pred_thresh = pred_thresh,
                       modification = modification, mode = mode, get_missing = get_missing, mass_tolerance = mass_tolerance,
                       filter_out = filter_out)
  df_out = pd.DataFrame.from_dict(out, orient = 'index')
  if intensity:
    df_out.columns = ['predictions', 'composition', 'rel_abundance']
    df_out.rel_abundance = [k/sum(df_out.rel_abundance.values.tolist())*100 for k in df_out.rel_abundance.values.tolist()]
  else:
    df_out.columns = ['predictions', 'composition']
  df_out['num_spectra'] = num_spectra
  df_out = deduplicate_predictions(df_out)
  df_out['charge'] = [round(composition_to_mass(df_out.composition.values.tolist()[k])/df_out.index.values.tolist()[k]) for k in range(len(df_out))]
  idx = [loaded_file.reducing_mass.values.tolist().index(k) for k in df_out.index.values]
  spectra_out = loaded_file.iloc[idx, :].reset_index(drop = True)
  df_out['RT'] = [round(k,2) for k in spectra_out.RT.values.tolist()]
  if get_missing:
    #df_out = deduplicate_retention(df_out)
    df_out = backfill_missing(df_out)
    idx = [loaded_file.reducing_mass.values.tolist().index(k) for k in df_out.index.values]
    spectra_out = loaded_file.iloc[idx, :].reset_index(drop = True)  
  spectra_out.peak_d = [ast.literal_eval(k) for k in spectra_out.peak_d.values.tolist()]
  top_frags = [sorted(k.items(), key = lambda x: x[1], reverse = True)[:frag_num] for k in spectra_out.peak_d.values.tolist()]
  df_out['top_fragments'] = [[round(j[0],4) for j in k] for k in top_frags]
  df_out = adduct_detect(df_out, mode, modification)
  df_out = domain_filter(df_out, glycan_class, libr = libr, mode = mode, filter_out = filter_out, modification = modification,
                         mass_tolerance = mass_tolerance)
  idx = [loaded_file.reducing_mass.values.tolist().index(k) for k in df_out.index.values]
  spectra_out = loaded_file.iloc[idx, :].reset_index(drop = True)
  spectra_out.peak_d = [ast.literal_eval(k) for k in spectra_out.peak_d.values.tolist()]
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
  preds  = [k[0][0] for k in df_out.predictions.values.tolist() if len(k) > 0]
  if glycan_class == 'free':
    preds = [k+'-ol' for k in preds]
    reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol','GlcNAc6S-ol', 'GlcNAc6P-ol',
                    'GlcNAc1P-ol','Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol']
    libr = libr + reducing_end
    permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  elif glycan_class == 'lipid':
    reducing_end = ['Glc','Gal']
    permitted_roots = ["Glc", "Gal"]
  elif glycan_class == 'O':
    reducing_end = ['GalNAc', 'Fuc', 'Man']
    permitted_roots = ['GalNAc', 'Fuc', 'Man']
  elif glycan_class == 'N':
    reducing_end = ['GlcNAc']
    permitted_roots = ['Man(b1-4)GlcNAc(b1-4)GlcNAc']
  net = construct_network(preds, reducing_end = reducing_end, permitted_roots = permitted_roots, libr = libr)
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
    df.iat[k,pred_idx] = explained[k]#.sort(key=len)
  return df

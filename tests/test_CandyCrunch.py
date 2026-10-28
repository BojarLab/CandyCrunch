import pytest
import os
import base64
import pathlib
import sqlite3
import struct
import zlib
import hashlib
import inspect
import json
from tabulate import tabulate
import numpy as np
from collections import defaultdict

from candycrunch.prediction import *
import candycrunch.utils
from glycowork.motif.graph import compare_glycans, get_possible_topologies, graph_to_string
from glycowork.motif.annotate import get_glycan_similarity
from itertools import product
from scipy.optimize import linear_sum_assignment
from scipy.stats import exponnorm, spearmanr
from glycowork.motif.processing import canonicalize_iupac, canonicalize_composition
import time
import random
import torch

BASE_DIR = pathlib.Path(__file__).parent.parent  # Go up one level from the test file
# CC_TEST_DATA_DIR: an alternate data folder (e.g., hardlinks with regenerated .xlsx), for measuring regenerated inputs
TEST_DATA_DIR = pathlib.Path(os.environ["CC_TEST_DATA_DIR"]) if os.environ.get("CC_TEST_DATA_DIR") else BASE_DIR / "tests" / "data"
TEST_DICTS = [
    {'name':'milk','args': {'glycan_class':'free'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000350','args': {'glycan_class':'O'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'O.' in x]},
    {'name':'GPST000350','args': {'glycan_class':'N'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'N' in x]},
    {'name':'GPST000017','args': {'glycan_class':'O'}, 'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000017/") if 'PGMb' not in x if 'JC' in x]},
    {'name':'GPST000029','args': {'glycan_class':'O'}},
    {'name':'PMC8950484_CHO','args': {'glycan_class':'O'}},
    {'name':'GPST000307','args': {'glycan_class':'O'}},
    {'name':'GPST000487','args': {'glycan_class':['N', 'O']},'test_files':os.listdir(f"{TEST_DATA_DIR}/GPST000487/")},
    # Four O-glycan and four N-glycan replicate runs (Bruker HCT) share one folder; 'folder' names it, 'name' the ground truth (df_mz_<name>.csv)
    {'name':'GPST000169_O','folder':'GPST000169','args': {'glycan_class':'O'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000169/") if 'OGlycan' in x]},
    {'name': 'GPST000169_N', 'folder': 'GPST000169', 'args': {'glycan_class': 'N'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000169/") if 'NGlycan' in x]},
    # Brain glycosphingolipid, N-glycan, and O-glycan runs (Thermo LTQ .raw, and their extract_spectra .xlsx) share one folder, one ground truth per glycan class
    {'name': 'GPST000374_GSL', 'folder': 'GPST000374', 'args': {'glycan_class': 'lipid'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000374/") if '_GSL_' in x]},
    {'name': 'GPST000374_N', 'folder': 'GPST000374', 'args': {'glycan_class': 'N'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000374/") if '_NG_' in x]},
    {'name': 'GPST000374_O', 'folder': 'GPST000374', 'args': {'glycan_class': 'O'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000374/") if '_OG_' in x]},
    # RNase B N-glycans with a free reducing end (a 'custom' label without mass), positive mode; the .raw runs with --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000134', 'args': {'glycan_class': 'N', 'modification': 'custom', 'max_charge': 3},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000134/") if 'glycans_1' in x]},
    # N-glycans with Neu5Gc (Thermo LTQ Velos .raw, ion-trap MS1 and MS2 at ~1000x lower intensities), one run; the .raw runs with --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000596', 'args': {'glycan_class': 'N'}},
    # Human serum N-glycans of the septic shock study (Thermo LTQ Velos .raw, ion-trap MS1 and MS2 at low intensities, header charges), one run; the .raw runs
    # with --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000503', 'args': {'glycan_class': 'N'}},
    # O-glycans from slide-fixed tissue (Bruker amaZon speed .d, read from its Analysis.yep); the .d runs with --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000316', 'args': {'glycan_class': 'O'}},
    # Adipose-derived stem cell N- and O-glycans (Agilent LC/MSD Trap XCT Ultra II .D, a Bruker-built ion trap read from its Analysis.yep), one run each; the .D runs with --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000153_N', 'folder': 'GPST000153', 'args': {'glycan_class': 'N'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000153/") if 'ADSC_Passage3_NG' in x]},
    {'name': 'GPST000153_O', 'folder': 'GPST000153', 'args': {'glycan_class': 'O'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000153/") if 'ADSC_Passage4_OG' in x]},
    # Milk oligosaccharides up to Hex8HexNAc6dHex4, the larger ones as [M-2H]2- (Thermo LTQ .raw, ion-trap MS1 and MS2), one run; the .raw runs with
    # --input-format mzml, the .xlsx otherwise
    {'name': 'GPST000556', 'args': {'glycan_class': 'free'}},
    # Bruker ion-trap N-glycans (mzXML, its extract_spectra .xlsx otherwise); the dot in the file name makes the GT run column end at '-35C-0'
    {'name': 'GPST000302', 'args': {'glycan_class': 'N'}},
    # Permethylated reduced human serum N-glycans (Mechref lab, C18, Q Exactive HF, positive HCD; .raw with --input-format mzml, its .xlsx otherwise);
    # trisialylated glycans are mostly [M+4H]4+, so max_charge 4
    {'name': 'GPST000548', 'args': {'glycan_class': 'N', 'sample_prep': 'permethylated', 'max_charge': 4, 'lc': 'C18'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000548/") if 'Serum_control' in x]},
    # Permethylated reduced rat serum N-glycans with Neu5Gc (Mechref lab, C18, Q Exactive HF, positive HCD), the DDA run; .raw with --input-format mzml, its
    # .xlsx otherwise
    {'name': 'GPST000403', 'args': {'glycan_class': 'N', 'sample_prep': 'permethylated', 'max_charge': 4, 'lc': 'C18'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000403/") if 'Rat_Glyphosate' in x]},
    # The PRM run of the same sample (30 scheduled targets, no survey scans, so no precursor intensities; every target sits on the 13C peak of its ion, those
    # of z = 3 isolate [M+2H+NH4]3+ ions); ground truth from the DDA one (RT 2.2-3.7 min later), each row backed by its PRM MS2
    {'name': 'GPST000403_PRM', 'folder': 'GPST000403', 'args': {'glycan_class': 'N', 'sample_prep': 'permethylated', 'max_charge': 4, 'lc': 'C18'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000403/") if x.startswith('PRM_')]},
    # N-glycans of carcinoembryonic antigen (human, Merck preparation; Bruker amaZon speed ETD, msconvert profile mzML with MS1, its extract_spectra .xlsx otherwise)
    {'name': 'GPST000054', 'args': {'glycan_class': 'N'}, 'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000054/") if x.startswith('CEA_Human_Merk')]},
    # Glycosphingolipid glycans of SW1116 colorectal cancer cells with a GT1b spike (Bruker HCT, profile mzXML, its extract_spectra .xlsx otherwise)
    {'name': 'GPST000239', 'args': {'glycan_class': 'lipid'}, 'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000239/") if 'SW1116' in x]},
    # Permethylated reduced human milk O-glycans and free oligosaccharides (Mechref lab, MGC column, Thermo orbitrap MS1 with ion-trap CID MS2, positive),
    # one run each from one folder with one open-peak ground truth per run (.raw with --input-format mzml, its .xlsx otherwise); rt_min 21 as the gradient
    # starts at ~21 min (earlier MS2 are re-triggered targets), Homo_sapiens as Mammalia let the database fallback fill Neu5Gc compositions; the GSL run of
    # the project is not a test, as its MS2 hold 1-2 peaks and its ground truth would be lactose alone
    {'name': 'GPST000461_O', 'folder': 'GPST000461', 'args': {'glycan_class': 'O', 'sample_prep': 'permethylated', 'rt_min': 21, 'taxonomy_level': 'Species', 'taxonomy_filter': 'Homo_sapiens'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000461/") if x.startswith('MilkOglycan')]},
    {'name': 'GPST000461_free', 'folder': 'GPST000461', 'args': {'glycan_class': 'free', 'sample_prep': 'permethylated', 'rt_min': 21, 'taxonomy_level': 'Species', 'taxonomy_filter': 'Homo_sapiens'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000461/") if x.startswith('MilkOligos')]},
    # Tear-film N-glycans (unreduced, a 'custom' label without mass) and reduced O-glycans of the hawk/caiman/sea turtle project, sample Ana_05 (Lebrilla lab,
    # PGC chip, Agilent Q-TOF CID, positive), one run each as Agilent mgf (MS2 only) with its extract_spectra .xlsx; open-peak ground truths audited from the
    # runs' MassHunter compound lists
    {'name': 'GPST000570_N', 'folder': 'GPST000570', 'args': {'glycan_class': 'N', 'modification': 'custom', 'max_charge': 3},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000570/") if x.startswith('Ana_05_N-gly') and x.endswith('.xlsx')]},
    {'name': 'GPST000570_O', 'folder': 'GPST000570', 'args': {'glycan_class': 'O', 'max_charge': 3},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000570/") if x.startswith('Ana_05_O-gly') and x.endswith('.xlsx')]},
    # 178 N-glycan standards (Thermo ion-trap CID, orbitrap survey scans, negative, free reducing ends, infused): each standard's target-precursor MS2 as a
    # synthetic LC peak, isomers in different RT slots, over two files; ground truth from the GlyTouCan records
    {'name': 'GPST000578', 'args': {'glycan_class': 'N', 'modification': 'custom', 'lc': 'other'}},
    # Trisialylated N-glycans of chicken colon labeled with 2-aminopyridine (PA), Table S1's DEAE fraction 8 on ODS (C18),
    # Thermo LTQ .raw (ion-trap MS1 and CID MS2, positive, mostly [M+3H]3+), one run (.raw with --input-format mzml, its .xlsx otherwise); Aves as the sample is chicken
    {'name': 'GPST000237', 'args': {'glycan_class': 'N', 'modification': 'PA', 'lc': 'C18', 'max_charge': 3, 'taxonomy_level': 'Class',
                                    'taxonomy_filter': 'Aves'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000237/") if x.startswith('chicken_colon_fr5_exp2')]},
    # Human plasma N-glycans labeled with 2-aminobenzamide (NIBRT, HILIC, Thermo orbitrap MS1 and HCD MS2, negative), one pooled run (.raw with --input-format
    # mzml, its .xlsx otherwise); ground truth audited from the published identification table
    {'name': 'GPST000651', 'args': {'glycan_class': 'N', 'modification': '2AB', 'lc': 'other'},
     'test_files': [x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000651/") if x.startswith('Plasma pooled sample')]},
]
# timsTOF Pro serum N-glycans (negative, reduced, PGC, DDA-PASEF .d with --input-format mzml, its extract_spectra .xlsx otherwise) live in tests/untracked,
# which git does not track, so the set is only added where its (preliminary, open-peak) ground truth is present; 'folder' is relative to TEST_DATA_DIR
UNTRACKED_DIR = BASE_DIR / "tests" / "untracked"
if os.path.exists(UNTRACKED_DIR / 'df_mz_timsTOF_serum.csv'):
    TEST_DICTS.append({'name': 'timsTOF_serum', 'folder': '../untracked', 'args': {'glycan_class': 'N'},
                       'test_files': [x for x in os.listdir(UNTRACKED_DIR) if x.startswith('20260831_TIMS_SC_NG') and x.lower().endswith(('.d', '.xlsx'))]})
# Pooled human urine N-glycans labelled with 2-aminopyridine (PA), DEAE fraction A1 sub-run 1, fraction 8 after aminolysis-SALSA (a2-6 isopropylamide, a2-3/8
# methylamide), Thermo LTQ (ion-trap MS1 and CID MS2, positive), one run (.raw with --input-format mzml, its .xlsx otherwise); in tests/untracked like timsTOF_serum
if os.path.exists(UNTRACKED_DIR / 'GPST000750' / 'df_mz_GPST000750.csv'):
    TEST_DICTS.append({'name': 'GPST000750', 'folder': '../untracked/GPST000750',
                       'args': {'glycan_class': 'N', 'modification': 'PA', 'lc': 'C18', 'max_charge': 3, 'sialic_derivatization': 'SALSA'}})
# Misprediction reports from the app (fetched from Sentry by the misprediction cell into tests/untracked/mispredictions): one set per report, the .cc of
# the peak's surroundings and an .xlsx made from it (older reports: the pooled MS2 spectrum as a one-spectrum .xlsx)
if os.path.exists(UNTRACKED_DIR / 'mispredictions' / 'index.json'):
    with open(UNTRACKED_DIR / 'mispredictions' / 'index.json', encoding = 'utf-8') as file:
        # JSON has no sets, wrap_inference takes filter_out as one
        TEST_DICTS.extend({**d, 'folder': '../untracked/mispredictions', 'args': {k: set(v) if k == 'filter_out' else v for k, v in d['args'].items()}} for d in json.load(file))
# Spectra files with MS1 (lower-case, Thermo writes .RAW too) and Bruker .d folders, run by --input-format mzml
MS1_FORMATS = ('.mzml', '.mzxml', '.raw', '.d', '.cc')
AVG_THRESHOLD = 0.05
MASS_TOLERANCE = 0.5
RT_TOLERANCE = 1.0
# Each batch set: glycan class and its runs as (ground truth name, folder, file); runs sharing a ground truth are replicates, scored together by their mean
BATCH_SETS = {'pgm': ('O', [('GPST000029', 'GPST000029', 'CA_PGMLAD_OG_051017.mzML'), ('GPST000017', 'GPST000017', 'JC_141128PGMa.mzML')]),
              'GPST000169_O': ('O', [('GPST000169_O', 'GPST000169', x) for x in sorted(os.listdir(f"{TEST_DATA_DIR}/GPST000169/")) if 'OGlycan' in x and x.endswith('.mzXML')]),
              'GPST000169_N': ('N', [('GPST000169_N', 'GPST000169', x) for x in sorted(os.listdir(f"{TEST_DATA_DIR}/GPST000169/")) if 'NGlycan' in x and x.endswith('.mzXML')])}
# Batch F1 on rows with MS2 evidence was 0.701-0.720 (GPST000029) and 0.705-0.713 (GPST000017) over seeds 0-2; the mean over the four replicates
# 0.646 (GPST000169_O) and 0.741 (GPST000169_N) at seed 0
BATCH_F1_THRESHOLDS = {'GPST000029': 0.65, 'GPST000017': 0.65, 'GPST000169_O': 0.59, 'GPST000169_N': 0.69}
# Quantification against the abundances the authors published for a run: ground truth name: (folder, run file, column of the published abundances (of the
# 'given' rows), column whose rows the authors quantified as one (Mechref's compositions) or None)
QUANT_SETS = {'GPST000374_N': ('GPST000374', '20230623_SD_NG_SSSMuG_2.raw', '20230623_SD_NG_SSSMuG_2', None),
              'GPST000374_O': ('GPST000374', '20230625_SD_OG_SSSMuG_3.raw', '20230625_SD_OG_SSSMuG_3', None),
              'GPST000374_GSL': ('GPST000374', '20230622_SD_GSL_SSSMuG_5.raw', '20230622_SD_GSL_SSSMuG_5', None),
              'GPST000596': ('GPST000596', '20240402_PVelos_KC_NG_PLT008_K2.raw', '20240402_PVelos_KC_NG_PLT008_K2', None),
              'GPST000556': ('GPST000556', 'JC_231121MN16.raw', 'JC_231121MN16', None),
              'GPST000239': ('GPST000239', '52-GSL glycans -SW1116 - technical replicate 2.mzXML', '52-GSL glycans -SW1116 - technical replicate 2', None),
              'GPST000548': ('GPST000548', 'Serum_control_1.raw', 'Serum_control_1', 'source_code'),
              'GPST000403': ('GPST000403', 'Rat_Glyphosate_Serum_Glycomics_SR1_100.raw', 'source_abundance', 'source_code')}
# Loading a .raw or mzXML takes 10-60 s, quantifying its ground truth about a second, so the MS1 of each run is kept here, keyed by file and loader code
QUANT_CACHE = pathlib.Path.home() / '.cache' / 'candycrunch_tests'


def match_spectra(array1, array2, mass_threshold = MASS_TOLERANCE, rt_threshold = RT_TOLERANCE, array1_alt = None, array2_alt = None):
    # One-to-one assignment minimizing the summed normalized m/z and RT distances (as GlycoGauntlet's evaluator): a greedy pass in ground-truth
    # order let a peak take the row of its neighboring isomer, and the result depended on how the ground truth was sorted
    array1_alt = array1 if array1_alt is None else array1_alt
    array2_alt = array2 if array2_alt is None else array2_alt
    # fmin, so a row without a charge (NaN singly charged equivalent) still matches on its m/z
    mass_diffs = np.fmin(np.abs(array1[:, None, 0] - array2[None, :, 0]), np.abs(array1_alt[:, None, 0] - array2_alt[None, :, 0]))
    rt_diffs = np.abs(array1[:, None, 1] - array2[None, :, 1])
    feasible = (mass_diffs <= mass_threshold) & (rt_diffs <= rt_threshold)
    rows, cols = linear_sum_assignment(np.where(feasible, mass_diffs / mass_threshold + rt_diffs / rt_threshold, 1e6))
    return [(i, j) for i, j in zip(rows, cols) if feasible[i, j]]


def add_pred_column(df_in, col_name, matches, pred_df, rt_col):
  df_in = df_in.copy()
  df_in[col_name] = None
  df_in['in_ground_truth'] = True
  for gt_idx,pred_idx in matches:
    df_in.at[gt_idx,col_name] = pred_df.iloc[pred_idx,:]['top1_pred']
  extra_preds = pred_df[~(pred_df.index.isin([x[1] for x in matches]))][['m/z','RT','top1_pred']].rename(columns={'m/z':'Mass','top1_pred':col_name,'RT':rt_col})
  extra_preds['in_ground_truth'] = False
  extra_preds['glycan'] = None
  df_in = pd.concat([df_in,extra_preds]).sort_values(['Mass',rt_col])
  return df_in


def evaluate_predictions(predictions, gt, rt_col, mass_thresh, RT_thresh, verbose = False):
  # Same rules as GlycoGauntlet's evaluator: rows without a structure are no predictions, structures are canonicalized, and both sides are also
  # compared as singly charged equivalents (a ground truth without a charge column counts as singly charged)
  predictions = predictions.reset_index()
  predictions = predictions[predictions['top1_pred'].notna()].reset_index(drop = True)
  if len(predictions)==0:
    print('empty preds')
    return 0, 0, 0, 0, 0, 0, 0, 0, 0
  predictions['top1_pred'] = predictions['top1_pred'].map({g: canonicalize_iupac(g) for g in predictions['top1_pred'].unique()})
  # a blank charge (e.g. an open mapping-file row whose ion is unknown) counts as singly charged, as a ground truth without the column
  gt_charges = np.nan_to_num(gt['charge'].values.astype(float), nan = -1) if 'charge' in gt.columns else np.full(len(gt), -1)
  pairs = predictions[['m/z', 'RT']].round(2).values
  pairs_converted = np.column_stack([predictions['m/z'] * predictions['charge'].abs() - (predictions['charge'] - np.sign(predictions['charge'])) * PROTON_MASS, predictions['RT']]).round(2)
  gt_pairs = gt[['Mass', rt_col]].round(2).values
  gt_pairs_converted = np.column_stack([gt['Mass'].values * np.abs(gt_charges) - (gt_charges - np.sign(gt_charges)) * PROTON_MASS, gt[rt_col].values]).round(2)
  matched_pairs = match_spectra(gt_pairs, pairs, mass_threshold = mass_thresh, rt_threshold = RT_thresh, array1_alt = gt_pairs_converted, array2_alt = pairs_converted)
  merge_df = gt[['Mass', rt_col, 'glycan']].reset_index(drop = True)
  new_md = add_pred_column(merge_df,'batch_pred', matched_pairs, predictions, rt_col)
  similarity_scores = []
  for gt_glycan, pred_glycan in zip(new_md['glycan'],new_md['batch_pred']):
    if not (isinstance(gt_glycan, str) and isinstance(pred_glycan, str)):
      similarity_scores.append(0.0)
      continue
    if '{' in gt_glycan:
      # get_possible_topologies places one floating part per call, so it runs until none is left (a GT with two placed none of them before, and no prediction
      # could match it exactly); a prediction written with the same floating parts is the GT itself
      possible_structures = [gt_glycan]
      while any('{' in p for p in possible_structures):
        possible_structures = list(dict.fromkeys(q for p in possible_structures for q in ([graph_to_string(x) for x in get_possible_topologies(p, exhaustive = True)] if '{' in p else [p])))
      similarity_scores.append(1.0 if compare_glycans(gt_glycan, pred_glycan) else max([1.0 if compare_glycans(p, pred_glycan) else get_glycan_similarity(p, pred_glycan) for p in possible_structures]))
    elif compare_glycans(gt_glycan, pred_glycan):
      similarity_scores.append(1.0)
    else:
      similarity_scores.append(get_glycan_similarity(gt_glycan, pred_glycan))
  new_md['similarity_score'] = similarity_scores
  unevaluable = len(np.where((new_md['in_ground_truth'])&(new_md['glycan'].isnull())&(new_md['batch_pred'].notnull()))[0])
  fp = len(np.where((~new_md['in_ground_truth'])&(new_md['batch_pred'].notnull()))[0])
  tp = new_md[new_md['glycan'].notnull()]['similarity_score'].sum() + 0.5 * unevaluable
  empty_glycan_not_predicted = len(np.where((new_md['in_ground_truth'])&(new_md['glycan'].isnull())&(new_md['batch_pred'].isnull()))[0])
  # A ground-truth peak without a structure weighs half either way: a hit is 0.5 TP, a miss 0.5 FN (a miss used to be a full FN against 0.5 TP for a hit)
  fn = (new_md[new_md['glycan'].notnull()]['similarity_score'].apply(lambda x: 1-x)).sum() + 0.5 * empty_glycan_not_predicted
  peaks_not_picked = len(np.where((new_md['in_ground_truth'])&(new_md['batch_pred'].isnull()))[0])
  incorrect_predictions = len(
      np.where((new_md['glycan'].notnull()) & (new_md['batch_pred'].notnull()) & (new_md['similarity_score'] < 1.0))[0])
  if verbose:
      np_rows = new_md[(new_md['in_ground_truth']) & (new_md['batch_pred'].isnull())]
      if len(np_rows) > 0:
          print('\n--- NotPicked (ground-truth m/z with no prediction) ---')
          print(tabulate(np_rows[['Mass', rt_col, 'glycan']].values, headers = ['Mass', 'RT', 'correct_glycan'],
                         tablefmt = 'grid'))
      wrong_rows = new_md[
          (new_md['glycan'].notnull()) & (new_md['batch_pred'].notnull()) & (new_md['similarity_score'] < 1.0)]
      if len(wrong_rows) > 0:
          print('\n--- Wrong (predicted, similarity < 1.0) ---')
          print(tabulate(wrong_rows[['Mass', rt_col, 'batch_pred', 'glycan', 'similarity_score']].round(
              {'similarity_score': 3}).values, headers = ['Mass', 'RT', 'predicted', 'correct_glycan', 'sim'],
                         tablefmt = 'grid'))
      fp_rows = new_md[(~new_md['in_ground_truth']) & (new_md['batch_pred'].notnull())]
      if len(fp_rows) > 0:
          print('\n--- FP (prediction not in ground truth) ---')
          print(tabulate(fp_rows[['Mass', rt_col, 'batch_pred']].values, headers = ['Mass', 'RT', 'predicted'],
                         tablefmt = 'grid'))
  Precision = tp / (tp + fp + 1e-8)
  Recall = tp / (tp + fn + 1e-8)
  F1_score = 2 * (Precision * Recall) / (Precision + Recall + 1e-8)
  return F1_score, Precision, Recall, peaks_not_picked, incorrect_predictions, tp, fp, fn, unevaluable

def posthoc_process_df(df_in, posthoc_params):
    for arg in posthoc_params:
        df_arg = arg.split('posthoc_')[1]
        if 'lowerthan' in arg:
            df_arg = df_arg.split('lowerthan_')[1]
            df_in = df_in[(df_in[df_arg] < posthoc_params[arg])]
        else:
            df_in = df_in[(df_in[df_arg] > posthoc_params[arg])]
    return df_in

extra_param_dict = {
        'test_dict': TEST_DICTS,
        'supplement': [True],
        'experimental': [True]
    }

test_params = [
    dict(zip(extra_param_dict, combo))
    for combo in product(*extra_param_dict.values())
]


def pytest_generate_tests(metafunc):
    # With --input-format both, the .xlsx and the MS1 files (.mzML/.mzXML/.raw) of a dataset are separate tests, so pytest-xdist can run them in parallel;
    # their scores were already logged and regression-checked per format
    if metafunc.function.__name__ == 'test_candycrunch_accuracy':
        input_format = metafunc.config.getoption('--input-format')
        groups = ['xlsx', 'mzml'] if input_format == 'both' else [input_format]
        metafunc.parametrize('test_params, file_group', [(p, g) for p in test_params for g in groups],
                             ids = [f"{p['test_dict']['name']}-{'+'.join(glycan_classes(p['test_dict']['args']['glycan_class']))}-{g}" for p in test_params for g in groups])


def test_candycrunch_accuracy(test_params, file_group, result_collector, input_format, verbose):
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    test_outputs = []
    test_dict = test_params['test_dict']
    folder = test_dict.get('folder', test_dict['name'])
    test_files = test_dict.get('test_files')
    if test_files is None:
        test_files = os.listdir(f"{TEST_DATA_DIR}/{folder}")
    test_files = [x for x in test_files if 'df_mz' not in x if not x.startswith(".")]
    # mzXML and Thermo .raw files count as mzML (raw spectra with MS1), so they never enter an xlsx run
    if file_group == "xlsx":
        test_files = [x for x in test_files if not x.lower().endswith(MS1_FORMATS)]
    else:
        mzml = [x for x in test_files if x.lower().endswith(MS1_FORMATS)]
        # mzml alone falls back to the xlsx of a dataset without an MS1 file; with both, its xlsx test already runs it
        test_files = mzml if mzml or input_format == "both" else test_files
    if not test_files:
        pytest.skip(f"no test file for {test_dict['name']} {test_dict['args']}")
    scores_by_key = defaultdict(list)
    for filename in test_files:
        inference_params = {k: v for k,v in test_params.items() if 'posthoc' not in k if k not in ('test_dict', 'format')}
        posthoc_params = {k: v for k,v in test_params.items() if 'posthoc' in k}
        print(filename)
        print(inference_params|posthoc_params)
        # Seeded, so a score only moves with the code: unseeded, test-time augmentation alone moved milk by 0.04 between runs
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        start_time = time.time()
        preds_out = wrap_inference(f"{TEST_DATA_DIR}/{folder}/{filename}",**test_dict['args'],
                                **inference_params)
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"\nTest execution time: {execution_time:.2f} seconds")  # Print execution time
        preds_out = posthoc_process_df(preds_out,posthoc_params)
        loaded_gt = pd.read_csv(f"{TEST_DATA_DIR}/{folder}/df_mz_{test_dict['name']}.csv")
        col_name = filename.split(".")[0]
        rt_col_name = 'RT' if 'RT' in loaded_gt.columns else col_name+'_RT'
        eval_scores = evaluate_predictions(preds_out, loaded_gt[loaded_gt[col_name] > 0], rt_col_name, MASS_TOLERANCE, RT_TOLERANCE, verbose = verbose)
        print('True Positives',eval_scores[-4])
        print('False Positives',eval_scores[-3])
        print('Unevaluable',eval_scores[-1])
        print('False Negatives',eval_scores[-2])
        print('incorrect_preds',eval_scores[4])
        print('peaks_not_picked',eval_scores[-6])
        test_outputs.append(eval_scores)
        # .raw gets a format of its own, so its scores keep their own baseline in the results log
        file_format = 'raw' if filename.lower().endswith('.raw') else 'mzML' if filename.lower().endswith(MS1_FORMATS) else 'xlsx'
        print(f'file_score:{eval_scores[0]} ({file_format})')
        test_params['format'] = file_format
        result_collector.add_result(test_params, eval_scores[0], eval_scores)
        param_key = tuple(
            test_params[key] if key != 'test_dict' else test_params['test_dict']['name']
            for key in test_params
        )
        scores_by_key[param_key].append(eval_scores[0])
    # The baseline is a dataset's mean F1 over its files, so the regression check takes that mean (replicates differ by more than its 0.04 tolerance)
    for param_key, scores in scores_by_key.items():
        result_collector.check_performance(test_dict['name'], param_key, np.mean(scores), loaded_gt)
    print("Adding results to collector")  # Debug print
    if test_outputs:
        print(f'avg_score:{np.mean([x[0] for x in test_outputs])}')
        assert np.mean([x[0] for x in test_outputs]) > AVG_THRESHOLD


@pytest.mark.parametrize("batch_set", list(BATCH_SETS))
def test_candycrunch_batch(batch_set, result_collector, verbose):
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    glycan_class, runs = BATCH_SETS[batch_set]
    if not runs or not all(os.path.exists(f"{TEST_DATA_DIR}/{folder}/{file}") for _, folder, file in runs):
        pytest.skip(f"{batch_set}: not every run has its mzML/mzXML")
    labels = [os.path.splitext(file)[0] for _, _, file in runs]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    combined, outputs = wrap_inference_batch([f"{TEST_DATA_DIR}/{folder}/{file}" for _, folder, file in runs], glycan_class, intra_cat_thresh = 1.0,
                                             spectra = True)
    assert list(outputs) == labels
    scores, gts = defaultdict(list), {}
    for (name, folder, _), label in zip(runs, labels):
        df_out, spectra_out = outputs[label]
        gaps = (df_out['evidence'] == 'ms1_only').values
        # Spectra stay aligned with the rows, MS1-only gap rows included
        assert len(spectra_out) == len(df_out)
        assert all(s is None for s, g in zip(spectra_out, gaps) if g)
        # Gap rows share the MS2 rows' abundance scale, so they cannot dominate a file
        assert abs(df_out['rel_abundance'].sum() - 100) < 1e-6
        assert df_out.loc[gaps, 'rel_abundance'].sum() < 50
        gts[name] = gt = pd.read_csv(f"{TEST_DATA_DIR}/{folder}/df_mz_{name}.csv")
        eval_scores = evaluate_predictions(df_out[~gaps].copy(), gt[gt[label] > 0],
                                           'RT' if 'RT' in gt.columns else label + '_RT', MASS_TOLERANCE, RT_TOLERANCE,
                                           verbose = verbose)
        print(f'{label} batch F1 (rows with MS2 evidence): {eval_scores[0]:.3f}, MS1-only gap rows: {gaps.sum()}')
        # Report like the per-file tests, so batch scores reach the summary tables, the results log and the regression check
        result_collector.add_result({'test_dict': {'name': f'{name}_batch'}, 'supplement': True, 'experimental': True,
                                     'format': 'mzML'}, eval_scores[0], eval_scores)
        scores[name].append(eval_scores[0])
    for name, name_scores in scores.items():
        result_collector.check_performance(f'{name}_batch', (f'{name}_batch', True, True, 'mzML'), np.mean(name_scores), gts[name])
        assert np.mean(name_scores) > BATCH_F1_THRESHOLDS.get(name, AVG_THRESHOLD)
    # Feature table: one row per isomer group with per-file abundances and evidence
    assert {'top1_pred', 'm/z', 'RT', 'n_files_ms2'}.issubset(combined.columns)
    assert all(label in combined.columns and f'evidence_{label}' in combined.columns for label in labels)
    assert (combined['n_files_ms2'] > 0).all()


def test_reducing_end_check(capsys):
    # The example run (reduced PGM O-glycans) set to unreduced or 2-AB-labeled glycans: the reducing end check finds the reduced reducing end and
    # takes it over, so the run gives the glycans of the reduced run (without the check, almost none are predicted)
    example = str(BASE_DIR / 'candycrunch' / 'example_PGM_O-glycans.xlsx')
    outs = {m: wrap_inference(example, 'O', modification = m) for m in ['reduced', 'custom', '2AB']}
    assert capsys.readouterr().out.count("Overriding the reducing end to 'reduced'") == 2
    reduced = set(outs['reduced']['top1_pred'].dropna())
    assert len(reduced) >= 15 and all(len(set(outs[m]['top1_pred'].dropna()) & reduced) >= 0.8 * len(reduced) for m in ['custom', '2AB'])


def test_onnx_model_matches_torch():
    # The built app runs the weights of model.py's CandyCrunch_CNN on ONNX Runtime: same logits as PyTorch, also for a norm scale below zero (which cannot
    # be folded into the convolution after it), and the same predictions for the example run
    pytest.importorskip('onnxruntime')
    from candycrunch.inference import CandyCrunchCNN, load_weights
    from candycrunch.model import CandyCrunch_CNN
    weights = load_weights(BASE_DIR / 'candycrunch' / 'candycrunch.pt')
    rng = np.random.default_rng(0)
    mz = np.zeros((8, 2, 2048), np.float32)
    for row in mz:
        peaks = rng.choice(2048, 40, replace = False)
        row[:, peaks] = rng.random((2, 40))
    inputs = (mz, rng.random((8, 12)).astype(np.float32), rng.integers(0, 3, 8), rng.random((8, 1)).astype(np.float32), *rng.integers(0, 3, (4, 8)))
    for state in (weights, {**weights, 'res3.ln2.weight': -weights['res3.ln2.weight']}):
        model = CandyCrunch_CNN(2048, num_classes = len(glycans), input_precursor_dim = 12)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        with torch.inference_mode():
            logits = model.eval()(*[torch.from_numpy(x if x.ndim > 1 else x[:, None]) for x in inputs]).numpy()
        assert np.abs(CandyCrunchCNN(state)(*inputs) - logits).max() < 1e-3
    example = str(BASE_DIR / 'candycrunch' / 'example_PGM_O-glycans.xlsx')
    onnx, default = wrap_inference(example, 'O', model = CandyCrunchCNN(weights)), wrap_inference(example, 'O')
    assert onnx.index.equals(default.index) and onnx['top1_pred'].equals(default['top1_pred'])


def test_xic_quantification():
    # Two isomers 0.6 min apart (4:1) and a doubly charged ion, with isotope envelopes, on a 5 s MS1 cycle: each gets its own peak area
    rts, sigma, envelope = np.arange(0, 20, 0.08), 0.08, np.array([0.6, 0.3, 0.08, 0.02])
    peaks = [(600.2, 1, 8.0, 4e5), (600.2, 1, 8.6, 1e5), (850.3, 2, 12.0, 2e5)]
    scans = [sorted((mz + i * 1.003355 / z, h * p * np.exp(-0.5 * ((rt - apex) / sigma) ** 2)) for mz, z, apex, h in peaks for i, p in enumerate(envelope)) for rt in rts]
    ms1 = (rts, np.array([m for scan in scans for m, _ in scan], dtype = np.float32), np.array([i for scan in scans for _, i in scan], dtype = np.float32),
           np.arange(len(rts) + 1, dtype = np.int64) * len(scans[0]))
    areas = extract_xic_areas(ms1, [p[0] for p in peaks], [8.05, 8.55, 12.1], charges = [p[1] for p in peaks], weights = [4, 1, 2])
    truth = np.array([p[3] for p in peaks]) * sigma * np.sqrt(2 * np.pi)
    assert np.allclose(areas, truth, rtol = 0.05)
    # The record of each integration (what the app shows) gives its area back, on its own peak, and asking for it changes no area
    details = []
    assert np.array_equal(extract_xic_areas(ms1, [p[0] for p in peaks], [8.05, 8.55, 12.1], charges = [p[1] for p in peaks], weights = [4, 1, 2], details = details), areas)
    assert np.allclose([(d['area'] + sum(o['area'] for o in d['others'])) * d['scale'] for d in details], areas)
    assert [round(float(d['times'][d['apex']]), 1) for d in details] == [8.0, 8.6, 12.0] and all(d['left'] < d['apex'] < d['right'] for d in details)


def test_quantification_synthetic(result_collector):
    # Peaks of known area injected into the survey scans of packaging/self_test.mzML (GPST000029, ion-trap MS1, RT 25-29.5 min), over its real background:
    # per seed 20 single peaks, 6 isomer pairs 0.25-1 min apart and 6 doubly charged glycans with 20-60% of their signal as [M-H]-, each with a tailing peak,
    # its isotope envelope (merged pairwise for z = 2, as in an ion trap), m/z jitter and intensity noise, quantified as finalise_predictions does. The score
    # is the share of peaks within 20% of their injected area (isotopes M..M+3, as measured), tracked like F1
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    rts, mzs, ints, offsets = load_spectra_filepath(str(BASE_DIR / 'packaging' / 'self_test.mzML'), extract_ms1 = True).attrs['ms1']
    base = np.median([ints[offsets[i]:offsets[i + 1]].max() for i in range(len(rts))])
    tol, designs, errors = 300 * MZ_REF / 1e6, [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        # (neutral mass, charge, share of the singly charged ion, apex, area, sigma, tau)
        design = [(rng.uniform(500, 1800), 1, 1.0, rng.uniform(25.5, 29), base * 10 ** rng.uniform(-2.5, 0.5), rng.uniform(0.06, 0.15), rng.uniform(0, 0.15)) for _ in range(20)]
        for _ in range(6):
            m, apex, area = rng.uniform(500, 1800), rng.uniform(25.5, 28), base * 10 ** rng.uniform(-2, 0.5)
            design += [(m, 1, 1.0, apex, area, 0.08, 0.08), (m, 1, 1.0, apex + rng.uniform(0.25, 1.0), area * rng.uniform(0.15, 1), 0.08, 0.08)]
        design += [(rng.uniform(1200, 2800), 2, rng.uniform(0.2, 0.6), rng.uniform(25.5, 29), base * 10 ** rng.uniform(-2, 0.5), rng.uniform(0.06, 0.15), rng.uniform(0, 0.15)) for _ in range(6)]
        design = pd.DataFrame(design, columns = ['mass', 'charge', 'z1_share', 'apex', 'area', 'sigma', 'tau'])
        added, truth = [[] for _ in rts], np.zeros(len(design))
        for k, (m, z, share, apex, area, sigma, tau) in enumerate(design.itertuples(index = False)):
            envelope = np.exp(-4.4e-4 * m) * (4.4e-4 * m) ** np.arange(5) / np.array([1, 1, 2, 6, 24])
            for zz, frac in ([(1, 1.0)] if z == 1 else [(2, 1 - share), (1, share)]):
                profile = area * frac * exponnorm.pdf(rts, tau / sigma, loc = apex, scale = sigma)
                truth[k] += np.sum((profile[1:] + profile[:-1]) * np.diff(rts)) / 2 * envelope[:4].sum()
                for i in np.flatnonzero(profile > 1e-3 * profile.max()):
                    peaks = [((m - zz * PROTON_MASS) / zz + j * ISOTOPE_SPACING / zz, profile[i] * p * rng.lognormal(0, 0.15)) for j, p in enumerate(envelope)]
                    peaks = peaks if zz == 1 else [((a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]) for a, b in zip(peaks[0::2], peaks[1::2])] + [peaks[4]]
                    added[i] += [(mz + rng.normal(0, 0.06), x) for mz, x in peaks]
        scans = [sorted(list(zip(mzs[offsets[i]:offsets[i + 1]], ints[offsets[i]:offsets[i + 1]])) + added[i]) for i in range(len(rts))]
        ms1 = (rts, np.array([m for s in scans for m, _ in s], dtype = np.float32), np.array([x for s in scans for _, x in s], dtype = np.float32),
               np.cumsum([0] + [len(s) for s in scans]).astype(np.int64))
        # MS2 near the apex, noisy precursor intensities as weights, every other charge state the neutral mass allows
        target_mzs, ms2_rts = ((design['mass'] - design['charge'] * PROTON_MASS) / design['charge']).values, design['apex'].values + rng.uniform(-0.1, 0.1, len(design))
        measured = extract_xic_areas(ms1, target_mzs, ms2_rts, charges = design['charge'].values, mz_tolerance = tol,
                                     mz_calibration = ms1_mz_calibration(ms1, target_mzs, ms2_rts, design['charge'].values, mz_tolerance = tol),
                                     weights = design['area'].values * rng.lognormal(0, 0.2, len(design)), mode = 'negative',
                                     other_charges = [[x for x in (1, 2) if x != z and m >= {1: 0, 2: 900}[x]] for m, z in zip(design['mass'], design['charge'])])
        designs.append(design)
        errors.append(np.abs(np.log2(np.maximum(measured, 1e-9) / truth)))
    errors = np.concatenate(errors)
    score = float(np.mean(errors <= np.log2(1.2)))
    groups = {g: float(np.mean(np.concatenate([e[sl] for e in np.split(errors, 3)]) <= np.log2(1.2))) for g, sl in [('single', slice(0, 20)), ('isomer pairs', slice(20, 32)), ('z = 2', slice(32, 38))]}
    print(f'synthetic quantification: {score:.3f} within 20% (' + ', '.join(f'{g} {v:.2f}' for g, v in groups.items()) + f'), median |log2 error| {np.median(errors):.3f}')
    result_collector.add_result({'test_dict': {'name': 'synthetic_quant'}, 'supplement': True, 'experimental': True, 'format': 'quant'}, score)
    result_collector.check_performance('synthetic_quant', ('synthetic_quant', True, True, 'quant'), score, pd.concat(designs))
    assert score > 0.6


@pytest.mark.parametrize("quant_set", list(QUANT_SETS))
def test_quantification_published(quant_set, result_collector):
    # The abundances of a run's ground-truth peaks as the pipeline quantifies a row (calibration, envelope over its own peak, co-eluting charge states no other
    # row reports), against the abundances its authors published; no model, so it tracks quantification alone. The score is the correlation of the log
    # shares (a peak quantified as 0 at half the smallest measured share), tracked like F1
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    folder, run, column, by = QUANT_SETS[quant_set]
    path = TEST_DATA_DIR / folder / run
    if not path.exists():
        pytest.skip(f"{quant_set}: no {run}")
    key = hashlib.sha1((run + str(path.stat().st_size) + str(int(path.stat().st_mtime)) + pathlib.Path(candycrunch.utils.__file__).read_text() + ''.join(
        inspect.getsource(f) for f in (load_spectra_filepath, process_mzML_stack, process_mzXML_stack, process_raw_stack, refine_precursor_mz))).encode()).hexdigest()[:16]
    cache = QUANT_CACHE / f'{path.stem}_{key}.npz'
    if not cache.exists():
        df = load_spectra_filepath(str(path), extract_ms1 = True)
        cache.parent.mkdir(parents = True, exist_ok = True)
        with open(cache.with_suffix('.tmp'), 'wb') as file:
            np.savez(file, *df.attrs['ms1'], mode = df.attrs.get('detected_mode') or 'negative', ms1_trap = str(df.attrs.get('detected_ms1_trap')))
        os.replace(cache.with_suffix('.tmp'), cache)
    with np.load(cache) as file:
        ms1, mode, accurate = (file['arr_0'], file['arr_1'], file['arr_2'], file['arr_3']), str(file['mode']), str(file['ms1_trap']) in ('orbitrap', 'tof')
    loaded_gt = pd.read_csv(TEST_DATA_DIR / folder / f'df_mz_{quant_set}.csv')
    gt = loaded_gt.assign(**{column: pd.to_numeric(loaded_gt[column], errors = 'coerce')})
    gt = gt[(gt['source'] == 'given') & (gt[column] > 0)].reset_index(drop = True)
    zs, rts = np.abs(gt['charge'].values).astype(int), gt['RT'].values
    comps = [frozenset((canonicalize_composition(c) if isinstance(c, str) and c else get_comp(g) if isinstance(g, str) and g else {}).items()) or None
             for g, c in zip(gt['glycan'], gt['composition'] if 'composition' in gt.columns else [None] * len(gt))]
    neutral = (gt['Mass'].values - (-1 if mode == 'negative' else 1) * PROTON_MASS) * zs
    other_charges = [[] if comps[i] is None else [z for z in range(1, zs.max() + 1) if z != zs[i] and neutral[i] >= {1: 0, 2: 900, 3: 1500}.get(z, 3500)
                      and not any(comps[j] == comps[i] and zs[j] == z and abs(rts[j] - rts[i]) < 1 for j in range(len(gt)))] for i in range(len(gt))]
    tol = 300 * MZ_REF / 1e6
    sample_prep = next((d['args'].get('sample_prep', 'underivatized') for d in TEST_DICTS if d['name'] == quant_set), 'underivatized')
    gt['measured'] = extract_xic_areas(ms1, gt['Mass'].values, rts, charges = zs, mz_tolerance = tol, sample_prep = sample_prep, mode = mode,
                                       mz_calibration = ms1_mz_calibration(ms1, gt['Mass'].values, rts, zs, mz_tolerance = tol, accurate = accurate),
                                       glycans = [g if isinstance(g, str) and g else None for g in gt['glycan']], other_charges = other_charges)
    if by:
        gt = gt.groupby(by).agg(measured = ('measured', 'sum'), **{column: (column, 'first')}).reset_index()
    measured, published = gt['measured'].values / gt['measured'].sum(), gt[column].values / gt[column].sum()
    floored = np.maximum(measured, measured[measured > 0].min() / 2)
    log_ratio = np.log2(floored / published)
    score = float(np.corrcoef(np.log2(floored), np.log2(published))[0, 1])
    print(f'{quant_set} quantification ({len(gt)} {"compositions" if by else "peaks"}): r of log shares {score:.3f}, Spearman {spearmanr(measured, published)[0]:.3f}, '
          f'{np.mean(np.abs(log_ratio) <= 1):.2f} within 2x, median |log2 ratio| {np.median(np.abs(log_ratio)):.3f}, {(measured == 0).sum()} not found')
    result_collector.add_result({'test_dict': {'name': f'{quant_set}_quant'}, 'supplement': True, 'experimental': True, 'format': 'quant'}, score)
    result_collector.check_performance(f'{quant_set}_quant', (f'{quant_set}_quant', True, True, 'quant'), score, loaded_gt)
    assert score > 0.5


def test_precursor_refinement():
    # Three survey scans: a jittering z = 1 isotope envelope at 600.2 and a precursor at 700.2 with an unrelated weak peak 1 Da below it
    jitter, scale = [-0.02, 0.02, 0.0], [0.9, 1.0, 1.1]
    scans = [[(600.2 + d, 1000 * f), (601.2 + d, 300 * f), (602.2 + d, 60 * f), (699.2, 100), (700.2, 1000)] for d, f in zip(jitter, scale)]
    ms1 = (np.array([10.0, 10.05, 10.1]), np.array([m for scan in scans for m, _ in scan], dtype = np.float32),
           np.array([i for scan in scans for _, i in scan], dtype = np.float32), np.arange(4, dtype = np.int64) * 5)
    mono = np.average([600.18, 600.22, 600.2], weights = [900, 1000, 1100])
    out, drop = refine_precursor_mz(ms1, [600.31, 601.25, 601.25, 700.3], [10.06, 10.07, 10.08, 10.06], [1, 1, 1, 1], [True, True, False, True])
    # The trigger moves onto the averaged monoisotopic centroid, an M+1 trigger is walked down and dropped as its monoisotopic precursor was
    # fragmented itself, a header charge state is trusted, and an unrelated lighter peak is not mistaken for the monoisotopic one
    assert np.allclose(out[:2], mono, atol = 1e-3) and out[2] == 601.25 and abs(out[3] - 700.2) < 1e-3
    assert list(drop) == [False, True, False, False]


def test_label_run_mapping_rules(monkeypatch):
    # label_run on inputs as wrap_inference(return_inputs = True) gives them: mapping RTs taken 1.3 min early in another run are shifted onto the run by
    # its 5 compositions listed once, and a mapping row without RT labels only its composition's most intense inputs above crumbs_thresh, as many as the
    # mapping has rows of that composition
    import training.pipeline_labels as pl
    comps = [{'Hex': 3, 'HexNAc': 2}, {'Hex': 5, 'HexNAc': 2}, {'Hex': 6, 'HexNAc': 2}, {'Hex': 7, 'HexNAc': 2}, {'Hex': 8, 'HexNAc': 2},
             {'Hex': 5, 'HexNAc': 4, 'Neu5Ac': 1}]
    glycans = ['Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', 'Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
               'Man(a1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
               'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
               'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc']
    sia = ['Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc']
    mzs = [911.33, 1235.43, 698.24, 779.27, 860.29, 965.84]
    # Each high-mannose glycan at RT 20 + k; four inputs of the sialylated composition: two isomer peaks, a more intense one failing crumbs_thresh, a weak third
    rows = [(mzs[k], 20.0 + k, 1e5, 5, comps[k], 12.0) for k in range(5)] + [(965.84, 30.0, 4e5, 6, comps[5], 15.0), (965.84, 31.0, 3e5, 4, comps[5], 14.0),
                                                                            (965.85, 35.0, 9e5, 2, comps[5], 2.0), (965.84, 33.0, 1e4, 1, comps[5], 9.0)]
    inputs = pd.DataFrame({'spec_id': range(len(rows)), 'RT': [r[1] for r in rows], 'intensity': [r[2] for r in rows], 'peak_d': [{500.0: 1.0}] * len(rows),
                           'rt_range': [(r[1], r[1]) for r in rows], 'num_spectra': [r[3] for r in rows], 'composition': [r[4] for r in rows],
                           'charge': [1, 1, 2, 2, 2, 2, 2, 2, 2], 'annotation_score': [r[5] for r in rows], 'binned_intensities': [np.eye(1, 2048, 300).ravel()] * len(rows),
                           'mz_remainder': [np.zeros(2048)] * len(rows), 'RT2': [r[1] / 35 for r in rows], 'glycan_type': 1,
                           'compositional_vector': [np.zeros(12)] * len(rows)}, index = pd.Index([r[0] for r in rows], name = 'm/z'))
    inputs.attrs.update(mode = 'negative', trap = 'linear', lc = 'PGC', modification = 'reduced', mass_tag = None, sample_prep = 'underivatized')
    monkeypatch.setattr(pl, 'wrap_inference', lambda *args, **kwargs: inputs.copy())
    monkeypatch.setattr(pl, 'load_spectra_filepath', lambda *args, **kwargs: pd.DataFrame({'activation': ['CID']}))
    mapping = pd.DataFrame({'Mass': mzs[:5] + [965.84, 965.84], 'RT': [18.7 + k for k in range(5)] + [np.nan, np.nan], 'glycan': glycans + sia})
    labelled, unlabelled = pl.label_run('run.cc', mapping, 'N')
    assert np.isclose(labelled.attrs['rt_shift'], 1.3) and np.array_equal(labelled['model_input'].iloc[0], np.array([[300], [1], [0]], dtype = np.float32))
    assert labelled['spec_id'].tolist() == [0, 1, 2, 3, 4, 5, 6] and len(unlabelled) == 2
    # The sialylated composition's two kept inputs, 1 min apart on PGC, split its two RT-less rows by elution order: a2-6 first
    assert labelled['glycan_candidates'].iloc[5:].tolist() == [[sia[1]], [sia[0]]] and labelled['glycan_match_status'].iloc[5] == 'unique'
    # Where each label came from, and every mapping row labelled
    assert labelled.attrs['rt_alignment'] == 'median' and labelled.attrs['rt_anchors'] == 5
    assert labelled['label_source'].tolist() == ['rt_median'] * 5 + ['no_rt'] * 2 and np.allclose(labelled['rt_error'].iloc[:5], 0, atol = 1e-6)
    assert [r['status'] for r in labelled.attrs['map_rows']] == ['labelled'] * 7 and [r['labels'] for r in labelled.attrs['map_rows']][5:] == [2, 0]
    # Mapping RTs of another gradient (the 5 anchors disagree): the rows count as RT-less, so each composition's strongest passing input is labeled
    shuffled = mapping.assign(RT = [22.7, 18.7, 25.7, 19.7, 30.7, np.nan, np.nan])
    labelled, unlabelled = pl.label_run('run.cc', shuffled, 'N')
    assert labelled.attrs['rt_shift'] == 0.0 and labelled['spec_id'].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert labelled.attrs['rt_alignment'] == 'dropped' and labelled['label_source'].tolist() == ['rt_dropped'] * 5 + ['no_rt'] * 2
    # What became of mapping rows that label nothing: no input at their mass, inputs of another composition there, their composition outside the RT window
    extra = pd.DataFrame({'Mass': [1500.0, 911.33, 698.24], 'RT': [20.0, 20.0, 40.0], 'glycan': [glycans[4], glycans[1], glycans[2]]})
    labelled, unlabelled = pl.label_run('run.cc', pd.concat([mapping.assign(RT = mapping['RT'] + 1.3), extra], ignore_index = True), 'N')
    assert [r['status'] for r in labelled.attrs['map_rows']] == ['labelled'] * 7 + ['no_input_at_mass', 'other_composition', 'outside_rt']
    assert labelled.attrs['rt_alignment'] == 'none' and labelled['spec_id'].tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_label_inputs_accurate_ms1():
    # With accurate survey scans an input takes a mapping row only within ppm_tolerance of its Mass or of its first two 13C peaks at the input's charge, a Mass
    # with fewer decimals within half its last one: 720.3207 (another ion, 655 ppm off N3H3 [M+2Na]2+ 719.8493) is only labelled at 0.5 m/z, the 13C pick
    # of N4H5S2 [M+3Na]3+ is, and so is an input 29 ppm off a Mass given as 1031.5
    import training.pipeline_labels as pl
    comps = [{'Hex': 5, 'HexNAc': 2}, {'Hex': 3, 'HexNAc': 3}, {'Hex': 5, 'HexNAc': 4, 'Neu5Ac': 2}, {'Hex': 4, 'HexNAc': 4, 'dHex': 1}]
    glycans = ['Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', 'GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
               'Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
               'Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc']
    mzs, masses, charges = [801.3864, 720.3207, 946.1202 + 1.003355 / 3 + 3e-4, 1031.53], [801.3859, 719.8493, 946.1202, 1031.5], [2, 2, 3, 2]
    inputs = pd.DataFrame({'spec_id': range(4), 'RT': [10.0, 11.0, 12.0, 13.0], 'intensity': 1e5, 'peak_d': [{500.0: 1.0}] * 4, 'rt_range': [(r, r) for r in range(10, 14)],
                           'num_spectra': 3, 'composition': comps, 'charge': charges, 'annotation_score': 12.0, 'model_input': [np.zeros((3, 1), np.float32)] * 4,
                           'RT2': 0.3, 'glycan_type': 1, 'compositional_vector': [np.zeros(12)] * 4}, index = pd.Index(mzs, name = 'm/z'))
    inputs.attrs.update(mode = 'positive', trap = 'other', lc = 'C18', modification = None, mass_tag = None, sample_prep = 'permethylated', ms1_trap = 'tof', accurate_ms1 = True)
    mapping = pd.DataFrame({'Mass': masses, 'RT': [10.0, 11.0, 12.0, 13.0], 'glycan': glycans})
    labelled, unlabelled = pl.label_inputs(inputs, mapping)
    assert sorted(labelled['spec_id']) == [0, 2, 3] and [r['status'] for r in labelled.attrs['map_rows']] == ['labelled', 'no_input_at_mass', 'labelled', 'labelled']
    assert labelled['mz_error'].tolist() == pytest.approx([0.0005, 1.003355 / 3 + 3e-4, 0.03])
    # Without accurate survey scans every input within 0.5 m/z of a Mass takes its row
    inputs.attrs['accurate_ms1'] = False
    labelled, _ = pl.label_inputs(inputs, mapping)
    assert sorted(labelled['spec_id']) == [0, 1, 2, 3]


def test_label_inputs_open_rows():
    # A mapping row without structure but with Mass and RT (an isomer the mapping leaves at composition level) keeps the inputs nearer it than any
    # structured row of their mass: the input 0.7 min from the structured row and 0.1 min from the open row stays unlabelled, and a structured row whose
    # only input sits nearer an open row says so
    import training.pipeline_labels as pl
    comps = [{'Hex': 1, 'HexNAc': 2}, {'Hex': 1, 'HexNAc': 2}, {'Hex': 2, 'HexNAc': 2}]
    glycans = ['Gal(b1-3)[GlcNAc(b1-6)]GalNAc', 'Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc']
    mzs = [587.2305, 587.2305, 749.2833]
    inputs = pd.DataFrame({'spec_id': range(3), 'RT': [10.0, 10.7, 12.6], 'intensity': 1e5, 'peak_d': [{500.0: 1.0}] * 3, 'rt_range': [(r, r) for r in (10.0, 10.7, 12.6)],
                           'num_spectra': 3, 'composition': comps, 'charge': [1, 1, 1], 'annotation_score': 12.0, 'model_input': [np.zeros((3, 1), np.float32)] * 3,
                           'RT2': 0.3, 'glycan_type': 1, 'compositional_vector': [np.zeros(12)] * 3}, index = pd.Index(mzs, name = 'm/z'))
    inputs.attrs.update(mode = 'negative', trap = 'other', lc = 'PGC', modification = 'reduced', mass_tag = None, sample_prep = 'underivatized')
    mapping = pd.DataFrame({'Mass': [587.2305, 587.2305, 749.2833, 749.2833], 'RT': [10.0, 10.8, 12.0, 12.5], 'glycan': [glycans[0], np.nan, glycans[1], np.nan]})
    labelled, unlabelled = pl.label_inputs(inputs, mapping)
    assert labelled['spec_id'].tolist() == [0] and len(unlabelled) == 2
    assert [r['status'] for r in labelled.attrs['map_rows']] == ['labelled', 'no_structure', 'open_row_nearer', 'no_structure']

def test_member_spectra_repool(monkeypatch):
    # The member spectra run_inputs keeps of a pooled input re-pool to the input the model got: their summed binned intensities, normalized, and the mean
    # of their m/z remainders over every member (one without a peak in the binned range counts too)
    import training.pipeline_labels as pl
    df = pd.DataFrame({'m/z': [900.3, 900.31, 900.29], 'RT': [10.0, 10.1, 10.2], 'intensity': [1e5, 2e5, 5e4],
                       'peak_d': [{204.087: 100.0, 366.14: 50.0}, {204.09: 80.0, 528.19: 20.0, 2990.0: 1.0}, {3500.0: 10.0}]})
    pooled = condense_dataframe(df.copy())
    monkeypatch.setattr(pl, 'wrap_inference', lambda *args, **kwargs: pooled.assign(spec_id = range(len(pooled))))
    monkeypatch.setattr(pl, 'load_spectra_filepath', lambda *args, **kwargs: df)
    row = pl.run_inputs('run.cc', 'N').iloc[0]
    b, r = np.zeros(2048), np.zeros(2048)
    for m in row['member_inputs']:
        b[m[0].astype(int)] += m[1]
        r[m[0].astype(int)] += m[2]
    pb, pr = np.zeros(2048), np.zeros(2048)
    pb[row['model_input'][0].astype(int)], pr[row['model_input'][0].astype(int)] = row['model_input'][1], row['model_input'][2]
    assert len(row['member_inputs']) == 3 and np.allclose(b / b.sum(), pb, atol = 1e-6) and np.allclose(r / 3, pr, atol = 1e-5)
    assert np.allclose(row['member_rts'], [10.0, 10.1, 10.2]) and 'binned_intensities' not in row.index and 'member_peaks' not in row.index


def test_label_run_rivals_and_settings(monkeypatch):
    # A labeled spectrum keeps its other candidate compositions (negatives for a composition-level loss, rival_mapped where the mapping lists them there
    # too) and the run's activation and survey scan analyzer
    import training.pipeline_labels as pl
    comps = [{'Hex': 3, 'HexNAc': 2}, {'Hex': 2, 'HexNAc': 2, 'dHex': 1}]
    inputs = pd.DataFrame({'spec_id': [0, 0], 'RT': [20.0, 20.0], 'intensity': [1e5, 1e5], 'peak_d': [{500.0: 1.0}] * 2, 'rt_range': [(20.0, 20.0)] * 2,
                           'num_spectra': [3, 3], 'composition': comps, 'charge': [1, 1], 'annotation_score': [12.0, 5.0],
                           'binned_intensities': [np.eye(1, 2048, 300).ravel()] * 2, 'mz_remainder': [np.zeros(2048)] * 2, 'RT2': [0.5] * 2, 'glycan_type': 1,
                           'compositional_vector': [np.ones(12), np.zeros(12)]}, index = pd.Index([911.33, 911.35], name = 'm/z'))
    inputs.attrs.update(mode = 'negative', trap = 'orbitrap', lc = 'PGC', modification = 'reduced', mass_tag = None, sample_prep = 'underivatized',
                        activation = 'HCD', ms1_trap = 'orbitrap')
    monkeypatch.setattr(pl, 'wrap_inference', lambda *args, **kwargs: inputs.copy())
    monkeypatch.setattr(pl, 'load_spectra_filepath', lambda *args, **kwargs: pd.DataFrame({'activation': ['HCD']}))
    mapping = pd.DataFrame({'Mass': [911.33], 'RT': [20.0], 'glycan': ['Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc']})
    labelled, unlabelled = pl.label_run('run.cc', mapping, 'N')
    assert len(labelled) == 1 and unlabelled.empty and labelled['activation'].iloc[0] == 'HCD' and labelled['ms1_trap'].iloc[0] == 'orbitrap'
    assert labelled['rival_compositions'].iloc[0] == [stringify_dict(comps[1])] and labelled['rival_scores'].iloc[0] == [5.0]
    assert labelled['rival_mapped'].iloc[0] == [False] and np.array_equal(labelled['rival_vectors'].iloc[0][0], np.zeros(12))


def test_run_inputs_activation_split(monkeypatch):
    # A run with CID and HCD scans of its precursors gives the inputs of each activation's spectra, their pooled spectra numbered apart; a run of one
    # activation goes to wrap_inference whole
    import training.pipeline_labels as pl
    loaded = pd.DataFrame({'m/z': [900.3, 900.3, 911.3], 'RT': [10.0, 10.0, 12.0], 'activation': ['CID', 'HCD', 'CID']})
    calls = []

    def wrap(path, glycan_class, _loaded = None, **kwargs):
        calls.append(_loaded['activation'].tolist())
        out = pd.DataFrame({'spec_id': range(len(_loaded)), 'binned_intensities': [np.eye(1, 2048, 300).ravel()] * len(_loaded),
                            'mz_remainder': [np.zeros(2048)] * len(_loaded)}, index = pd.Index(_loaded['m/z'].values, name = 'm/z'))
        out.attrs.update(mode = 'negative', activation = _loaded['activation'].iloc[0])
        return out

    monkeypatch.setattr(pl, 'wrap_inference', wrap)
    monkeypatch.setattr(pl, 'load_spectra_filepath', lambda *args, **kwargs: loaded)
    inputs = pl.run_inputs('run.cc', 'N')
    assert calls == [['CID', 'CID'], ['HCD']] and inputs['activation'].tolist() == ['CID', 'CID', 'HCD'] and inputs['spec_id'].tolist() == [0, 1, 2]
    assert inputs.attrs['mode'] == 'negative' and inputs.attrs['activation'] is None
    calls.clear()
    monkeypatch.setattr(pl, 'load_spectra_filepath', lambda *args, **kwargs: loaded.assign(activation = 'CID'))
    inputs = pl.run_inputs('run.cc', 'N')
    assert calls == [['CID'] * 3] and 'activation' not in inputs.columns and inputs.attrs['activation'] == 'CID'


def test_charge_carrier_candidates():
    # A sodiated permethylated run (free reducing end, adducts Na+) with accurate survey scans (ms1_ppm): every charge is carried by Na+, so Man5 [M+2Na]2+ 801.3864 is no
    # Hex1HexNAc1Neu5Ac3 [M+Na+H]2+ (8 ppm off) and FM3 [M+2Na]2+ 684.3261 no Hex6HexNAc2dHex1P1 [M+Na+2H]3+ (0.33 Da off), the matches tried first before;
    # and a match within ms1_ppm replaces an in-class one further off: Hex5HexNAc2Neu5Ac1 [M+Na+H]2+ 970.9818 is no Hex5HexNAc3P1 [M+2Na]2+ (43 ppm off)
    comp_map, df_use, topo_map = create_struct_maps(df_glycan, ['N'], filter_out = {'HexA'}, phylo_level = 'Class', phylo_filter = 'Mammalia')
    peaks = {300.0: 1.0, 1500.0: 0.5}
    candidates = lambda ms1_ppm: assign_candidate_structures(pd.DataFrame({'m/z': [801.3864, 684.3261, 970.9818], 'peak_d': [peaks] * 3}), df_use, comp_map, topo_map,
                                                             0.5, 'positive', 0, modification = None, sample_prep = 'permethylated', max_charge = 4, ms1_ppm = ms1_ppm,
                                                             adducts = ['Na+'])
    out = candidates(20)
    found = {(s, stringify_dict(c), z) for s, c, z in zip(out['spec_id'], out['composition'], out['charge'])}
    assert (0, 'Hex5HexNAc2', 2) in found and (1, 'Hex3HexNAc2dHex1', 2) in found and (2, 'Hex5HexNAc2Neu5Ac1', 2) in found
    assert not {f for f in found if f[1] in ('Hex1HexNAc1Neu5Ac3', 'Hex6HexNAc2dHex1P1', 'Hex5HexNAc3P1')}
    # Without accurate survey scans the first match within 0.5 Da stays
    out = candidates(None)
    assert 'Hex5HexNAc3P1' in {stringify_dict(c) for s, c in zip(out['spec_id'], out['composition']) if s == 2}


def test_mixed_charge_carrier_candidates():
    # A procainamide run that fragments [M+2H+K]3+ and [M+H+NH4]2+ beside [M+3H]3+ (adducts K+ and NH4+) with accurate survey scans: a precursor keeps every
    # reading within ms1_ppm for the annotation scores, so protonated Hex5HexNAc4dHex1Neu5Ac2 is no longer only Hex4HexNAc4Neu5Ac3 [M+2H+NH4]3+, and
    # Hex5HexNAc4dHex3 [M+H+NH4]2+ no longer only Hex4HexNAc4Neu5Ac1dHex2 [M+2H]2+; the K+ ions keep their reading
    comp_map, df_use, topo_map = create_struct_maps(df_glycan, ['N'], filter_out = {'HexA'}, phylo_level = 'Class', phylo_filter = 'Mammalia')
    mass = lambda c: composition_to_mass(c, modification = 'procainamide')
    sialylated, fucosylated = {'Hex': 5, 'HexNAc': 4, 'dHex': 1, 'Neu5Ac': 2}, {'Hex': 5, 'HexNAc': 4, 'dHex': 3}
    ions = [(sialylated, (mass(sialylated) + 3 * PROTON_MASS) / 3, 3), (sialylated, (mass(sialylated) + 2 * PROTON_MASS + mass_dict['K+']) / 3, 3),
            (fucosylated, (mass(fucosylated) + PROTON_MASS + mass_dict['NH4+']) / 2, 2)]
    candidates = lambda ms1_ppm: assign_candidate_structures(pd.DataFrame({'m/z': [mz * (1 + 10e-6) for _, mz, _ in ions], 'peak_d': [{300.0: 1.0, 1500.0: 0.5}] * 3}),
                                                             df_use, comp_map, topo_map, 0.5, 'positive', 0, modification = 'procainamide', max_charge = 3,
                                                             ms1_ppm = ms1_ppm, adducts = ['K+', 'NH4+'])
    out = candidates(20)
    found = {(s, stringify_dict(c), z) for s, c, z in zip(out['spec_id'], out['composition'], out['charge'])}
    assert all((k, stringify_dict(c), z) in found for k, (c, _, z) in enumerate(ions))
    # Without accurate survey scans the first reading stays
    out = candidates(None)
    assert (0, stringify_dict(sialylated), 3) not in {(s, stringify_dict(c), z) for s, c, z in zip(out['spec_id'], out['composition'], out['charge'])}


def test_ms1_peak_regrouping():
    # Unreduced anomers 1 min apart (1:3) fragmented on their flanks only, and a broad peak (sigma 1 min) whose MS2 every 1.2 min differ: with accurate survey
    # scans each anomer is a cluster at its MS1 apex, and the broad peak one cluster at its apex
    rts = np.arange(0, 20, 0.01)
    peaks = [(900.3, 10.0, 0.1, 1e5), (900.3, 11.0, 0.1, 3e5), (1100.4, 15.0, 1.0, 2e5)]
    trace = lambda mz, t: sum(h * np.exp(-0.5 * ((t - apex) / w) ** 2) for m, apex, w, h in peaks if m == mz) + 1
    ms1 = (rts, np.repeat([[900.3, 1100.4]], len(rts), axis = 0).ravel().astype(np.float32),
           np.array([[trace(900.3, t), trace(1100.4, t)] for t in rts]).ravel().astype(np.float32), np.arange(len(rts) + 1, dtype = np.int64) * 2)
    spectra = [(900.3, t) for t in (9.95, 10.1, 10.9, 11.05)] + [(1100.4, t) for t in (12.6, 13.8, 15.0, 16.2, 17.4)]
    df = pd.DataFrame({'m/z': [m for m, _ in spectra], 'RT': [t for _, t in spectra], 'intensity': [trace(m, t) for m, t in spectra],
                       'peak_d': [{204.087: 100.0, 366.14: 50.0, 1000.0 + 37 * k: 80.0} for k in range(len(spectra))]})
    out = condense_dataframe(df.copy(), mz_ppm = 20, ms1 = ms1).sort_values('RT')
    assert np.allclose(out['RT'], [10.0, 11.0, 15.0], atol = 0.02) and out['num_spectra'].tolist() == [2, 2, 5]
    # Without survey scans the anomers stay one cluster and the broad peak five
    assert len(condense_dataframe(df.copy(), mz_ppm = 20)) == 6


def test_condense_isotope_apex():
    # GPST000315 IL_VB_NG_260717 Man7 [M-2H]2- 779.27 (precursor m/z, RT and intensity as loaded from its .cc): the ion trap re-triggered on the merged 13C
    # centroid 779.80-779.93, once at the apex, and the cluster is reported at its strongest monoisotopic spectrum; a spectrum 0.4 below an apex with a tenth
    # of its intensity is another ion, and a peak of zero intensity in every spectrum of a cluster is left out instead of stopping the run
    spectra = [(779.274, 32.777, 4224), (779.927, 32.798, 4152), (779.275, 32.847, 7899), (779.922, 32.878, 8025), (779.386, 32.948, 12896),
               (779.919, 32.959, 13263), (779.435, 33.050, 20655), (779.470, 33.153, 22386), (779.804, 33.164, 22616), (779.479, 33.256, 18707),
               (779.275, 33.358, 12072), (779.466, 33.503, 4898), (1100.0, 40.0, 1000), (1100.4, 40.05, 10000), (1100.42, 40.1, 9000)]
    df = pd.DataFrame({'m/z': [m for m, _, _ in spectra], 'RT': [t for _, t, _ in spectra], 'intensity': [float(i) for _, _, i in spectra],
                       'peak_d': [{698.3: 100.0, 596.2: 50.0, 1438.6: 30.0} if m < 800 else {500.0: 10.0, 300.0: 0.0} for m, _, _ in spectra]})
    out = condense_dataframe(df.copy(), mz_diff = 0.48).sort_values('RT')
    assert np.allclose(out['m/z'], [779.470, 1100.4]) and out['num_spectra'].tolist() == [12, 3]
    assert all(300.0 not in p for p in out['peak_d'])


def test_ion_trap_header_charge():
    # One LTQ survey scan and four doubly charged precursors whose MS2 headers carry the instrument's charge and "monoisotopic" m/z (GPST000374): the survey
    # merges M and M+1 of 1160.93 into one centroid 0.28 below the header's 13C pick, resolves 864.30 0.5 below the header 864.79, and shows 1111.90 and
    # 1059.39, whose headers are right, with their merged or resolved 13C peaks above them
    scan = [(864.23, 194), (864.80, 354), (1059.38, 3209), (1060.06, 4720), (1111.90, 923), (1112.43, 948), (1161.19, 5161), (1161.95, 1499)]
    ms1 = (np.array([20.0]), np.array([m for m, _ in scan], dtype = np.float32), np.array([i for _, i in scan], dtype = np.float32),
           np.array([0, len(scan)], dtype = np.int64))
    out, drop = refine_precursor_mz(ms1, [1161.47, 864.79, 1111.94, 1059.40], [20.05] * 4, [0] * 4, [True] * 4, charges = [2, 2, 2, 2])
    assert np.allclose(out, [1161.47 - 1.003355 / 2, 864.79 - 1.003355 / 2, 1111.94, 1059.40]) and not drop.any()


def test_bruker_mzxml(tmp_path):
    # Bruker HCT profile scans (msconvert mzXML): a doubly charged envelope at 965.844 whose monoisotopic top dips in the middle, and a singly charged
    # peak at 675.36 with a noise shoulder 0.6 m/z above it; MS2 headers give the isolation window centre (966.20 at charge 2, 675.81 and 675.42 at
    # charge 1)
    grid = np.arange(600, 1000, 0.03125)
    def profile(peaks):
        ints = sum(h * np.exp(-0.5 * ((grid - mz) / 0.05) ** 2) for mz, h in peaks)
        return np.where(ints < 1, 0, ints)
    survey = [(965.79, 6e5), (965.9, 6e5), (965.844 + 1.003355 / 2, 9e5), (965.844 + 1.003355, 5.4e5), (675.36, 2e6), (675.97, 5e3)]
    def scan(num, level, rt, ints, precursor = ''):
        data = base64.b64encode(np.stack([grid, ints], axis = 1).astype('>f8').tobytes()).decode()
        return (f'<scan num="{num}" msLevel="{level}" centroided="0" polarity="-" retentionTime="PT{rt}S" msInstrumentID="1">{precursor}'
                f'<peaks precision="64" byteOrder="network" compressionType="none" contentType="m/z-int">{data}</peaks></scan>')
    fragments = profile([(675.3, 1e4), (800.3, 3e4), (900.4, 2e4)])
    path = tmp_path / 'hct.mzXML'
    path.write_text('<?xml version="1.0"?><mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun><msInstrument msInstrumentID="1">'
                    '<msManufacturer category="msManufacturer" value="Bruker Daltonics"/><msModel category="msModel" value="Bruker Daltonics HCT Series"/>'
                    '<msMassAnalyzer category="msMassAnalyzer" value="radial ejection linear ion trap"/></msInstrument>' +
                    scan(1, 1, 3000, profile(survey)) +
                    scan(2, 2, 3002, fragments, '<precursorMz precursorIntensity="0" precursorCharge="2" activationMethod="CID">966.204528808594</precursorMz>') +
                    scan(3, 2, 3004, fragments, '<precursorMz precursorIntensity="0" precursorCharge="1" activationMethod="CID">675.813842773438</precursorMz>') +
                    scan(4, 2, 3006, fragments, '<precursorMz precursorIntensity="0" precursorCharge="1" activationMethod="CID">675.418823242188</precursorMz>') +
                    '</msRun></mzXML>')
    df = process_mzXML_stack(str(path), intensity = True, extract_ms1 = True)
    _, ms1_mzs, _, _ = df.attrs['ms1']
    # One centroid per isotope (the Gaussian fit made two of the dipping top), and the noise shoulder stays apart
    assert np.sum(np.abs(ms1_mzs - 965.85) < 0.3) == 1 and np.sum(np.abs(ms1_mzs - 966.35) < 0.2) == 1 and np.sum(np.abs(ms1_mzs - 675.97) < 0.1) == 1
    # Every precursor moves onto its monoisotopic centroid (the nearest centroid to 675.81 is the shoulder), keeps its header charge, and none is
    # dropped as isotope-triggered, as a Bruker isolation window always holds the whole envelope
    assert np.allclose(df['m/z'], [965.844, 675.36, 675.36], atol = 0.02) and df['precursor_charge'].tolist()[0] == 2
    assert df.attrs['detected_trap'] == 'linear' and df.attrs['detected_mode'] == 'negative'
    # The scans' details: analyzer and polarity, nothing the file does not state, the header m/z as the shift refinement applied
    assert df['analyzer'].tolist() == ['linear'] * 3 and df['polarity'].tolist() == ['-'] * 3 and df['collision_energy'].isna().all()
    assert np.allclose(df['m/z'] + df['header_shift'], [966.204528808594, 675.813842773438, 675.418823242188])


def test_mzxml_without_ms2_or_rt(tmp_path):
    # Runs without MS2 spectra (full-MS profiling, MSn files of MS3 and higher stages only) load as an empty table, which extract_spectra refuses
    # with its own error; MALDI spectra exported to mzXML have no retention times
    data = base64.b64encode(np.array([[500.0, 10.0], [600.0, 20.0]]).astype('>f8').tobytes()).decode()
    def write(name, scans):
        path = tmp_path / name
        path.write_text('<?xml version="1.0"?><mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun>' + ''.join(
            f'<scan num="{num}" msLevel="{level}" centroided="1" polarity="+"{rt}>{precursor}<peaks precision="64" byteOrder="network" '
            f'compressionType="none" contentType="m/z-int">{data}</peaks></scan>' for num, level, rt, precursor in scans) + '</msRun></mzXML>')
        return str(path)
    ms3 = '<precursorMz precursorIntensity="0" precursorScanNum="1">600.0</precursorMz>'
    path = write('ms1_ms3.mzXML', [(1, 1, ' retentionTime="PT60S"', ''), (2, 3, ' retentionTime="PT61S"', ms3)])
    df = process_mzXML_stack(path, intensity = True, extract_ms1 = True)
    assert df.empty and set(SPECTRUM_DETAILS) <= set(df.columns) and len(df.attrs['ms1'][0]) == 1
    with pytest.raises(ValueError, match = 'has no MS2 spectra'):
        extract_spectra(path, str(tmp_path / 'ms1_ms3.cc'))
    path = write('maldi.mzXML', [(1, 2, '', '<precursorMz precursorIntensity="0">1288.6</precursorMz>')])
    with pytest.raises(ValueError, match = 'has no retentionTime'):
        process_mzXML_stack(path)


def test_exactive_mzxml(tmp_path):
    # msconvert writes a Q Exactive's mzXML without filter lines and names only its quadrupole as analyzer; the model names the orbitrap, so the survey
    # scans count as accurate MS1 (read as an unknown analyzer, candidates were matched at 0.5 Da, where [M+H]+ readings took the [M+NH4]+ ions)
    data = base64.b64encode(np.array([[500.0, 10.0], [797.3034, 2e6]]).astype('>f8').tobytes()).decode()
    path = tmp_path / 'qe.mzXML'
    path.write_text('<?xml version="1.0"?><mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun><msInstrument msInstrumentID="1">'
                    '<msManufacturer category="msManufacturer" value="Thermo Scientific"/><msModel category="msModel" value="Q Exactive Plus"/>'
                    '<msMassAnalyzer category="msMassAnalyzer" value="quadrupole"/></msInstrument>' + ''.join(
                        f'<scan num="{num}" msLevel="{level}" centroided="1" polarity="+" retentionTime="PT{rt}S" msInstrumentID="1">{precursor}<peaks '
                        f'precision="64" byteOrder="network" compressionType="none" contentType="m/z-int">{data}</peaks></scan>' for num, level, rt, precursor in
                        [(1, 1, 600, ''), (2, 2, 601, '<precursorMz precursorIntensity="2e6" activationMethod="HCD">797.3034</precursorMz>')]) + '</msRun></mzXML>')
    df = process_mzXML_stack(str(path), intensity = True, extract_ms1 = True)
    assert df.attrs['detected_trap'] == df.attrs['detected_ms1_trap'] == 'orbitrap' and df['analyzer'].tolist() == ['orbitrap']


def test_bruker_ion_trap_mzml(tmp_path):
    # The scans of test_bruker_mzxml as msconvert writes a Bruker ion trap .d to mzML: profiles, the HCT series model with the amaZon named in a userParam,
    # and the isolation window centre as selected ion m/z, also with a charge state; mzML and mzXML of one run give the same spectra and precursors
    grid = np.arange(600, 1000, 0.03125)
    def profile(peaks):
        ints = sum(h * np.exp(-0.5 * ((grid - mz) / 0.05) ** 2) for mz, h in peaks)
        return np.where(ints < 1, 0, ints)
    def cv(acc, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="" value="{value}"/>'
    def spectrum(i, level, rt, ints, precursor = None, charge = None):
        arrays = ''.join(f'<binaryDataArray encodedLength="0">{cv("MS:1000523")}{cv("MS:1000576")}{cv(acc)}<binary>'
                         f'{base64.b64encode(values.astype("<f8").tobytes()).decode()}</binary></binaryDataArray>' for acc, values in (('MS:1000514', grid), ('MS:1000515', ints)))
        prec = (f'<precursorList count="1"><precursor><isolationWindow>{cv("MS:1000827", precursor)}</isolationWindow><selectedIonList count="1"><selectedIon>'
                f'{cv("MS:1000744", precursor)}{cv("MS:1000041", charge)}</selectedIon></selectedIonList><activation>{cv("MS:1000133")}</activation></precursor>'
                f'</precursorList>') if precursor else ''
        return (f'<spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(grid)}">{cv("MS:1000511", level)}{cv("MS:1000129")}{cv("MS:1000128")}'
                f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitName="second"/></scan></scanList>'
                f'{prec}<binaryDataArrayList count="2">{arrays}</binaryDataArrayList></spectrum>')
    survey = [(965.79, 6e5), (965.9, 6e5), (965.844 + 1.003355 / 2, 9e5), (965.844 + 1.003355, 5.4e5), (675.36, 2e6), (675.97, 5e3)]
    fragments = profile([(675.3, 1e4), (800.3, 3e4), (900.4, 2e4)])
    spectra = [spectrum(0, 1, 3000, profile(survey)), spectrum(1, 2, 3002, fragments, 966.204528808594, 2), spectrum(2, 2, 3004, fragments, 675.813842773438, 1),
               spectrum(3, 2, 3006, fragments, 675.418823242188, 1)]
    path = tmp_path / 'amazon.mzML'
    path.write_text('<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><referenceableParamGroupList count="1">'
                    f'<referenceableParamGroup id="CommonInstrumentParams">{cv("MS:1000697")}<userParam name="instrument model" value="amaZon speed ETD"/>'
                    '</referenceableParamGroup></referenceableParamGroupList><instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">'
                    f'<referenceableParamGroupRef ref="CommonInstrumentParams"/><componentList count="1"><analyzer order="2">{cv("MS:1000083")}</analyzer></componentList>'
                    f'</instrumentConfiguration></instrumentConfigurationList><run id="r"><spectrumList count="4">{"".join(spectra)}</spectrumList></run></mzML>')
    df = process_mzML_stack(str(path), intensity = True, extract_ms1 = True)
    _, ms1_mzs, _, _ = df.attrs['ms1']
    # One centroid per isotope (the Gaussian fit made two of the dipping top), and the noise shoulder stays apart, as in test_bruker_mzxml
    assert np.sum(np.abs(ms1_mzs - 965.85) < 0.3) == 1 and np.sum(np.abs(ms1_mzs - 966.35) < 0.2) == 1 and np.sum(np.abs(ms1_mzs - 675.97) < 0.1) == 1
    # Every precursor moves onto its monoisotopic centroid, also the charged one, and keeps its header charge
    assert np.allclose(df['m/z'], [965.844, 675.36, 675.36], atol = 0.02) and df['precursor_charge'].tolist()[0] == 2
    assert df.attrs['detected_trap'] == df.attrs['detected_ms1_trap'] == 'amazon' and df.attrs['detected_mode'] == 'negative'


@pytest.mark.parametrize('model', ['amaZon speed', 'HCTultra', 'LC/MSD Trap XCT ultra II'])
def test_bruker_yep(tmp_path, model):
    # A Bruker ion trap .d folder (amaZon): Analysis.yep pages of zlib records, a spectrum header (type 316) and a line spectrum (type 315, kind 2000) per
    # spectrum, and the parameter definitions; MS2 headers give the isolation target (966.20 for a [M-2H]2- at 965.844) and the trigger charge. HCT files
    # (pwiz's Sample_1-A,1_01_985.d) put 80 bytes with an A0 BF signature before the block chain and start it with an A1 BF block; Agilent's LC/MSD Trap XCT
    # (GPST000153) starts it with A1 BF blocks of types 10, 11 and 12 and has no type 2 block
    hct, xct = model.startswith('HCT'), model.startswith('LC/MSD')
    def record(kind, number, spectrum, body):
        packed = zlib.compress(body)
        key = (31, 0, 0, spectrum, 2000 if spectrum else 1000, 11)
        return struct.pack('<4s5I6I', b'\xfa\x0b\xdb\xdb', len(packed), 3, kind, number, 3, *key) + packed + bytes(16) + b'   '
    def spectrum(number, ticks, peaks, precursor = None, charge = 0):
        params = struct.pack('<5I', 2, 0x6b, 1, 0x403, charge)
        header = struct.pack('<ifIff3f5I', ticks, sum(i for _, i in peaks), number, 100, 2000, 0, max(i for _, i in peaks), 0, ticks, 0, 3,
                             0 if precursor is None else 4, 20)
        header = header if precursor is None else header[:48] + struct.pack('<f', precursor) + header[48:]
        start = (struct.pack('<HIH48x', 56, 10, 0xbfa1) + struct.pack('<HIH24x', 32, 11, 0xbfa1) + struct.pack('<HIH12x', 20, 12, 0xbfa1) if xct else
                 struct.pack('<HIH4x', 12, 2, 0xbfa1 if hct else 0xbfa2))
        body = ((bytes(20) + struct.pack('<HH', 4, 0xbfa0) + bytes(56) if hct or xct else b'') + start +
                struct.pack('<HIH', 8 + len(params), 8, 0xbfa0) + params + np.array(peaks, dtype = '<f4').tobytes())
        return record(316, 2 * number, number, header) + record(315, 2 * number + 1, number, body)
    definitions = b''.join(struct.pack('<II', pid, 2) + b"\x1d'\xab\xbf" + struct.pack('<I', len(name) + 1) + name + b'\x00' for pid, name in
                           [(0x6b, b'iIonNeg'), (0x403, b'iTrgChrgAto02')]) + b"\x1d'\xab\xbf" + struct.pack('<I', len(model) + 1) + model.encode() + b'\x00'
    survey = [(675.36, 2e6), (965.844, 6e5), (965.844 + 1.003355 / 2, 9e5), (965.844 + 1.003355, 5.4e5), (966.47, 4e3)]
    fragments = [(290.09, 1e4), (675.3, 1e4), (800.3, 3e4)]
    records = (record(315, 1, 0, definitions) + spectrum(1, 2812500, survey) + spectrum(2, 2813000, fragments, 966.2045, 2) +
               spectrum(3, 2813500, fragments, 675.8138, 1))
    folder = tmp_path / 'run.d'
    folder.mkdir()
    (folder / 'Analysis.yep').write_bytes(b'Yellop1.1\x00' + bytes(8) + b'\xfa\x0b\x00\xee' + struct.pack('<I', 24 + len(records)) + bytes(20) + records)
    df = load_spectra_filepath(str(folder), extract_ms1 = True)
    # Both precursors move onto their monoisotopic centroids; Bruker's trigger charge 2 is kept, its charge 1 counts as undetermined
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 0.01)
    assert df['precursor_charge'].tolist()[0] == 2 and np.isnan(df['precursor_charge'].tolist()[1])
    # RT ticks are 64 us, and the MS2 peaks are the line spectrum's
    assert np.allclose(df['RT'], np.array([2813000, 2813500]) * 64e-6 / 60)
    assert len(df['peak_d'][0]) == 3 and max(df['peak_d'][0], key = df['peak_d'][0].get) == pytest.approx(800.3)
    assert df.attrs['detected_trap'] == ('linear' if hct or xct else 'amazon') and df.attrs['detected_mode'] == 'negative' and len(df.attrs['ms1'][0]) == 1
    assert df['analyzer'].tolist() == [df.attrs['detected_trap']] * 2 and df['polarity'].tolist() == ['-'] * 2 and df['isolation_width'].isna().all()


def test_bruker_baf(tmp_path):
    # A Bruker Q-TOF .d folder (otofControl): analysis.baf blocks (length, type with the A0 BF signature, header length) with the parameter definitions
    # (3001: entries with their number and short name at byte 20), the method's values (2001: a set of (number, 8-byte value) records), and per spectrum a
    # header (2002: number, RT in ms; sub-blocks with its values, precursor list and TOF calibration), a profile (1001, not read) and a line spectrum (1002:
    # float64 positions on the digitizer's time axis, float32 intensities). The spectra's C1 is the method's times 1 + e (temperature compensation)
    delay, timebase, c0, c1, c2, c4, e = 17312.0, 0.2, 312.48, 155567.06, 9.18e-05, -0.00143, -1.2e-06
    names = {'Mode_IonPolarity': 0, 'Mode_ScanBegin': 1213, 'Collision_Energy_Act': 1480, 'MSMS_IsolationMass_Act': 1521, 'MSMS_PreCursorChargeState': 1538,
             'Quadrupole_IsolationResolution_Act': 1449, 'Calibration_Tof2CalC1': 1149, 'TOF_DeviceTempGradient1': 1284}

    def block(kind, header, body = b''):
        return struct.pack('<3I', 12 + len(header) + len(body), 0xbfa00000 | kind, 12 + len(header)) + header + body

    def records(values):
        return b''.join(struct.pack('<H', k) + (struct.pack('<q', v) if isinstance(v, int) else struct.pack('<d', v)) for k, v in values.items())

    def line_spectrum(number, peaks, gradient):
        # Positions from the method's C1 and m/z times 1 + e or, where the instrument compensates with its own temperature gradient, the spectrum's C1
        cal, f = (c1 * (1 + e), 1) if gradient else (c1, 1 + e)
        positions = np.array([(c0 + np.sqrt(1e12 / cal * (m * f + c4)) + c2 * m * f - delay) / timebase for m, _ in peaks], dtype = '<f8')
        n = len(peaks)
        return block(0x1002, struct.pack('<4I', number, number, 3, n) + bytes(24), struct.pack('<4I', n, 40, 40 + 8 * n, 40 + 12 * n) + bytes(24) +
                     positions.tobytes() + np.array([i for _, i in peaks], dtype = '<f4').tobytes() + np.ones(n, dtype = '<f4').tobytes())

    definitions = b''.join(struct.pack('<2HBBH6H', 20 + len(name) + 1, number, 5, 1, 0, 20, 0, 0, 0, 0, 0) + name.encode() + b'\x00' for name, number in names.items())
    survey = [(675.36, 2e6), (965.844, 9e5), (965.844 + 1.003355 / 2, 6e5), (965.844 + 1.003355, 3e5), (966.47, 4e3)]
    fragments = [(290.09, 1e4), (675.3, 1e4), (800.3, 3e4), (965.85, 5e4), (966.35, 4e4)]

    def run(gradient):
        # The method's values, with TOF_DeviceTempGradient1 (ppm/C) where the instrument compensates C1 itself
        method = {0: 1, 1213: 50.0, 1149: c1, **({1284: gradient} if gradient else {})}
        data = block(0x3001, b'', definitions) + block(0x2001, bytes(8), block(0x000d, struct.pack('<2I', len(method), 0x200), records(method)))
        for number, rt, precursor, charge, peaks in [(1, 150000, None, 0, survey), (2, 150600, 965.844, 2, fragments), (3, 151200, 675.3605, 1, fragments)]:
            values = {1480: 25.0, 1449: 4.0} if precursor else {1480: 7.0}
            values.update({1521: precursor, 1538: charge} if precursor else {})
            subs = (block(0x0001, struct.pack('<I', len(values)), records(values)) +
                    block(0x0005, struct.pack('<I', 1 if precursor else 0), struct.pack('<f', precursor) if precursor else b'') +
                    block(0x0006, struct.pack('<I', 10), struct.pack('<7d3I2d', delay, timebase, c0, c1 * (1 + e), c2, 0.0, c4, 3, 68, 2, 26.1, 25.6) + bytes(24)))
            data += (block(0x2002, struct.pack('<5I', 1, number, 3, rt, 0) + struct.pack('<If', 0x200, -1.0), subs) + block(0x1001, bytes(32)) +
                     line_spectrum(number, peaks, gradient))
        return data

    folder = tmp_path / 'run.d'
    folder.mkdir()
    data = run(0)
    (folder / 'analysis.baf').write_bytes(data)
    df = load_spectra_filepath(str(folder), extract_ms1 = True)
    # Both precursors move onto their monoisotopic survey centroids; Bruker's charge 2 is kept, its charge 1 counts as undetermined
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 1e-4)
    assert df['precursor_charge'].tolist()[0] == 2 and np.isnan(df['precursor_charge'].tolist()[1])
    # The line positions give back the m/z through the TOF calibration and the temperature factor; the doubly charged precursor's envelope is no fragment
    assert sorted(df['peak_d'][0]) == pytest.approx([290.09, 675.3, 800.3], abs = 1e-6) and len(df['peak_d'][1]) == 5
    assert np.allclose(df['RT'], [2.51, 2.52]) and df['scan'].tolist() == [2, 3] and df['collision_energy'].tolist() == [25.0, 25.0]
    assert df['isolation_width'].tolist() == [4.0, 4.0] and df['scan_low'].tolist() == [50.0, 50.0]
    assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and df.attrs['detected_mode'] == 'negative'
    assert df['analyzer'].tolist() == ['tof'] * 2 and df['polarity'].tolist() == ['-'] * 2 and df['activation'].tolist() == ['HCD'] * 2
    assert len(df.attrs['ms1'][0]) == 1 and df.attrs['ms1'][1][0] == pytest.approx(675.36, abs = 1e-4)
    # An instrument compensating C1 with its own temperature gradient (a compact) used the spectrum's C1 as it is
    (folder / 'analysis.baf').write_bytes(run(45.0))
    df = load_spectra_filepath(str(folder))
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 1e-4) and sorted(df['peak_d'][0]) == pytest.approx([290.09, 675.3, 800.3], abs = 1e-6)
    # A run stored as profile spectra only is refused with a hint
    (folder / 'analysis.baf').write_bytes(data.replace(struct.pack('<I', 0xbfa01002), struct.pack('<I', 0xbfa01003)))
    with pytest.raises(ValueError, match = 'profile spectra only'):
        load_spectra_filepath(str(folder))


def test_bruker_tsf(tmp_path, monkeypatch):
    # A timsTOF .d without ion mobility (timsTOF fleX, ESI): analysis.tsf (SQLite: Frames, FrameMsMsInfo, MzCalibration, GlobalMetadata) and per frame in
    # analysis.tsf_bin its byte size, the size of its zstd block, then the block of float64 line positions on the digitizer's time axis and float32
    # intensities. Survey frames are MsMsType 0, MS/MS frames are listed with their trigger mass, isolation width, charge and collision energy
    try:
        from compression.zstd import compress
    except ImportError:
        from zstandard import compress
    timebase, delay, c0, c1, c2, c4, dc1, t1_ref = 0.125, 31272.375, 278.333, 156145.33, -0.00065, 0.0, 20.0, 25.75

    def position(mz, t1):
        mz = mz * (1 + dc1 * 1e-6 * (t1 - t1_ref))
        return (c0 + np.sqrt(1e12 / c1 * (mz + c4)) + c2 * mz - delay) / timebase

    survey = [(675.36, 2e6), (965.844, 9e5), (965.844 + 1.003355 / 2, 6e5), (965.844 + 1.003355, 3e5), (966.47, 4e3)]
    fragments = [(290.09, 1e4), (675.3, 1e4), (800.3, 3e4), (965.85, 5e4), (966.35, 4e4)]
    # Frame 1 survey, 2 and 3 MS/MS of 965.84 (charge 2) and 675.36 (charge 1) with trigger masses at their monoisotopic survey lines, 4 an MS/MS frame
    # without lines
    rows = [(1, 150.0, 0, survey, 26.1), (2, 150.6, 2, fragments, 26.3), (3, 151.2, 2, fragments, 26.3), (4, 151.8, 2, [], 26.3)]
    msms = [(2, 1, 965.8441, 3.0, 2, -25.0), (3, 1, 675.3605, 1.0, 1, -20.0), (4, 1, 500.0, 1.0, None, -20.0)]

    def write(folder, metadata, maldi = False):
        folder.mkdir()
        data, frames = b'', []
        for frame, time, kind, peaks, t1 in rows:
            block = compress(np.array([position(m, t1) for m, _ in peaks], '<f8').tobytes() + np.array([i for _, i in peaks], '<f4').tobytes())
            frames.append((frame, time, '-', 1, kind, len(data), len(peaks), 1, t1, 24.0))
            data += struct.pack('<2I', 8 + len(block), len(block)) + block
        (folder / 'analysis.tsf_bin').write_bytes(data)
        db = sqlite3.connect(folder / 'analysis.tsf')
        db.execute('CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT)')
        db.executemany('INSERT INTO GlobalMetadata VALUES (?, ?)', metadata.items())
        db.execute('CREATE TABLE Frames (Id INTEGER, Time REAL, Polarity TEXT, ScanMode INTEGER, MsMsType INTEGER, TimsId INTEGER, NumPeaks INTEGER, MzCalibration INTEGER, T1 REAL, T2 REAL)')
        db.executemany('INSERT INTO Frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', frames)
        db.execute('CREATE TABLE MzCalibration (Id INTEGER, ModelType INTEGER, DigitizerTimebase REAL, DigitizerDelay REAL, T1 REAL, T2 REAL, dC1 REAL, dC2 REAL, C0, C1, C2, C3, C4)')
        db.execute('INSERT INTO MzCalibration VALUES (1, 1, ?, ?, ?, 24.0, ?, 0.0, ?, ?, ?, 0.0, ?)', (timebase, delay, t1_ref, dc1, c0, c1, c2, c4))
        db.execute('CREATE TABLE FrameMsMsInfo (Frame INTEGER, Parent INTEGER, TriggerMass REAL, IsolationWidth REAL, PrecursorCharge INTEGER, CollisionEnergy REAL)')
        db.executemany('INSERT INTO FrameMsMsInfo VALUES (?, ?, ?, ?, ?, ?)', msms)
        if maldi:
            db.execute('CREATE TABLE MaldiFrameInfo (Frame INTEGER, Chip INTEGER, SpotName TEXT)')
            db.execute("INSERT INTO MaldiFrameInfo VALUES (1, 0, 'A1')")
        db.commit()
        db.close()
        return str(folder)

    metadata = {'TimsCompressionType': '3', 'HasLineSpectra': '1', 'HasProfileSpectra': '0', 'MzAcqRangeLower': '150.000000'}
    run = write(tmp_path / 'run.d', metadata)
    df = load_spectra_filepath(run, extract_ms1 = True)
    # Both precursors move onto their monoisotopic survey lines; the empty MS/MS frame gives no row
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 1e-4)
    # Bruker writes no charge where it determined none, so its charge 1 is kept
    assert df['precursor_charge'].tolist() == [2, 1]
    # The line positions give back the m/z through the TOF calibration and the frame's temperature correction; the charge 2 precursor's envelope is no fragment
    assert sorted(df['peak_d'][0]) == pytest.approx([290.09, 675.3, 800.3], abs = 1e-6) and len(df['peak_d'][1]) == 5
    assert np.allclose(df['RT'], [2.51, 2.52]) and df['scan'].tolist() == [2, 3] and df['collision_energy'].tolist() == [25.0, 20.0]
    assert df['isolation_width'].tolist() == [3.0, 1.0] and df['scan_low'].tolist() == [150.0, 150.0]
    assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and df.attrs['detected_mode'] == 'negative'
    assert df['polarity'].tolist() == ['-'] * 2 and df['activation'].tolist() == ['HCD'] * 2
    assert len(df.attrs['ms1'][0]) == 1 and df.attrs['ms1'][1][0] == pytest.approx(675.36, abs = 1e-4) and len(df.attrs['ms1'][1]) == 5
    # Survey frames keep their TIMS_MS1_PEAKS most intense lines
    monkeypatch.setattr(candycrunch.utils, 'TIMS_MS1_PEAKS', 4)
    assert load_spectra_filepath(run, extract_ms1 = True).attrs['ms1'][1].tolist() == pytest.approx([675.36, 965.844, 966.3457, 966.8474], abs = 1e-3)
    # Headers only (the app's run detection) need no analysis.tsf_bin
    os.remove(os.path.join(run, 'analysis.tsf_bin'))
    assert [s['precursor'] for s in candycrunch.utils.read_tsf(run, peaks = False)][1] == {'mz': 965.8441, 'charge': 2}
    # Profile-only and MALDI runs are refused with a hint
    with pytest.raises(ValueError, match = 'profile spectra only'):
        load_spectra_filepath(write(tmp_path / 'profile.d', {**metadata, 'HasLineSpectra': '0', 'HasProfileSpectra': '1'}))
    with pytest.raises(ValueError, match = 'MALDI'):
        load_spectra_filepath(write(tmp_path / 'maldi.d', metadata, maldi = True))


def test_waters_raw(tmp_path):
    # A Waters MassLynx .raw folder (Fast DDA): _FUNCTNS.INF (a 416-byte record per function: scan type, ionization mode, centroid and lock mass bits; mass
    # range at 160), per function an .IDX (30-byte or older 22-byte records: points, bit 30 for calibrated m/z, TIC, RT, offset), an .STS (parameter
    # definitions, then a record per scan) and a .DAT of 12-byte centroids or 8-byte continuum points (intensity and m/z as mantissa and exponent); the
    # continuum MS/MS function is stored uncalibrated and takes the header's calibration, a polynomial in sqrt(m/z) (here sqrt(m/z) = c0 + c1 sqrt(stored))
    cal = [4e-3, 0.9998, 0.0, 0.0, 0.0, 0.0]
    folder = tmp_path / 'run.raw'
    folder.mkdir()

    def pack(value, bits, shift):
        e = int(np.floor(np.log2(value))) + 1 if value else 0
        return round(value * 2.0 ** (bits - e)) | e << shift if value else 0

    def function(k, head, scans, width, size, sts):
        data, index = b'', b''
        for rt, peaks in scans:
            info = len(peaks) | (1 << 30 if width == 12 else 0)
            index += struct.pack('<2I2f6sQ', 0, info, 0.0, rt, b'', len(data)) if size == 30 else struct.pack('<2I2f6s', len(data), info, 0.0, rt, b'')
            data += b''.join(struct.pack('<2I', pack(i, 21, 22), pack(m, 27, 27)) + bytes(width - 8) for m, i in peaks)
        defs = b''.join(struct.pack('<3H', number, 3, at) + name.ljust(26, b'\x00') + struct.pack('<H', 4) + bytes(14)
                        for number, at, name in [(62, 0, b'Collision Energy'), (77, 4, b'Set Mass')])
        stats = struct.pack('<4H', 32 + len(defs), 1, 8, 2) + bytes(24) + defs + b''.join(struct.pack('<2f', ce, mass) for ce, mass in sts)
        for suffix, content in (('IDX', index), ('DAT', data), ('STS', stats)):
            (folder / f'_FUNC{k:03d}.{suffix}').write_bytes(content)
        return struct.pack('<H', head) + bytes(158) + struct.pack('<f', 50.0) + bytes(124) + struct.pack('<f', 2000.0) + bytes(124)

    # Survey scan (function 1, ES-: ionization mode 9, centroided): a 3+ glycan at 965.844 and a 1+ one at 675.36
    survey = [(675.36, 2e6), (965.844, 9e5), (965.844 + 1.003355 / 3, 1e6), (965.844 + 2 * 1.003355 / 3, 6e5), (1200.5, 3e4)]
    # MS/MS (function 2, continuum): each fragment a peak of points evenly spaced in sqrt(m/z), as on the TOF grid, flanked by zeros
    fragments = [290.09, 512.2, 800.3]
    profile = [(((np.sqrt(f) + 2.2e-4 * k - cal[0]) / cal[1]) ** 2, h) for f in fragments for k, h in zip(range(-3, 4), [0, 20, 70, 100, 70, 20, 0])]
    # The DDA set masses sit above the survey centroids: on the 13C peak of the 3+ ion, and 0.06 above the 1+ one; function 3 is the lock mass reference
    (folder / '_FUNCTNS.INF').write_bytes(function(1, 18 | 9 << 5 | 0x400, [(10.0, survey)], 12, 30, [(4.0, 0.0)]) +
                                          function(2, 16 | 9 << 5, [(10.01, profile), (10.02, profile)], 8, 22, [(25.0, 965.844 + 1.003355 / 3 + 0.03), (20.0, 675.42)]) +
                                          function(3, 18 | 9 << 5 | 0x400 | 0x8000, [(10.015, [(554.2615, 5e4)])], 12, 30, [(4.0, 0.0)]))
    (folder / '_HEADER.TXT').write_text('\n'.join(f'$$ Cal Function {k}: ' + ','.join(map(str, cal)) + ',T1' for k in (1, 2, 3)))
    assert [s['scan'] for s in candycrunch.utils.read_waters(str(folder), peaks = False)] == ['function=1 process=0 scan=1', 'function=2 process=0 scan=1',
                                                                                             'function=2 process=0 scan=2']
    df = load_spectra_filepath(str(folder), extract_ms1 = True)
    # The 3+ precursor walks down its isotopes from the 13C peak, the 1+ one is found 0.06 below its set mass; MassLynx stores no charge states
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 1e-4) and df['precursor_charge'].isna().all()
    # Continuum points are calibrated and centroided (calibrated centroids are not calibrated again); the lock mass function is no survey scan
    assert sorted(df['peak_d'][0]) == pytest.approx(fragments, abs = 1e-3) and list(df['peak_d'][0].values()) == pytest.approx([280] * 3)
    assert len(df.attrs['ms1'][0]) == 1 and not np.any(np.abs(df.attrs['ms1'][1] - 554.2615) < 0.01) and df.attrs['ms1'][1][0] == pytest.approx(675.36, abs = 1e-4)
    assert df['scan'].tolist() == ['function=2 process=0 scan=1', 'function=2 process=0 scan=2'] and np.allclose(df['RT'], [10.01, 10.02])
    assert df['collision_energy'].tolist() == [25.0, 20.0] and df['scan_low'].tolist() == [50.0, 50.0] and df['polarity'].tolist() == ['-'] * 2
    assert df.attrs['detected_mode'] == 'negative' and df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof'
    # Ion mobility (HDMS) runs keep their spectra in .cdt files, which are not read
    (folder / '_func002.cdt').write_bytes(b'')
    with pytest.raises(ValueError, match = 'ion mobility'):
        load_spectra_filepath(str(folder))


def test_agilent_masshunter(tmp_path):
    # An Agilent MassHunter .d folder: AcqData/MSScan.bin (header with the offset of the first record at byte 88, one 284-byte record per scan),
    # MSTS.xml (number of scans), MSMassCal.bin (per scan: a count, A, t0, polynomial step) and MSPeak.bin (per scan: float64 flight times, then
    # float32 abundances); m/z = (A (t - t0))^2 minus the polynomial step (here c t, ValueUseFlags 2 in DefaultMassCal.xml). MS2 records give the isolation
    # target and MassHunter's charge state
    from candycrunch.utils import MASSHUNTER_SCAN
    a, t0, c = 0.000572794, 1024.818, -2e-8
    survey = [(675.36, 2e6), (965.844, 9e5), (965.844 + 1.003355 / 2, 6e5), (965.844 + 1.003355, 3e5), (966.47, 4e3)]
    fragments = [(290.09, 1e4), (675.3, 1e4), (800.3, 3e4), (965.85, 5e4), (966.35, 4e4)]
    scans = [(1, 2.5, 1, survey, 0.0, 0), (2, 2.51, 2, fragments, 965.846, 2), (3, 2.52, 2, fragments, 675.3605, 1)]
    records, peaks, calibration = np.zeros(len(scans), dtype = MASSHUNTER_SCAN), bytes(68), bytes(72)
    for k, (scan, rt, level, spectrum, mz, charge) in enumerate(scans):
        records[k] = (scan, rt, level, 0, 10.0 * (level - 1), mz, charge, len(calibration), 2, len(peaks), len(spectrum), 1)
        calibration += struct.pack('<i5d', 5, a, t0, 19996.0, 92106.0, c)
        times = np.array([np.sqrt(m) / a + t0 for m, _ in spectrum])
        for _ in range(3):
            times = np.sqrt(np.array([m for m, _ in spectrum]) + c * times) / a + t0
        peaks += times.astype('<f8').tobytes() + np.array([i for _, i in spectrum], dtype = '<f4').tobytes()
    folder = tmp_path / 'run.d'
    (folder / 'AcqData').mkdir(parents = True)
    (folder / 'AcqData' / 'MSScan.bin').write_bytes(bytes(88) + struct.pack('<i', 228) + bytes(136) + records.tobytes())
    (folder / 'AcqData' / 'MSTS.xml').write_text(f'<TimeSegments><TimeSegment><NumOfScans>{len(scans)}</NumOfScans></TimeSegment></TimeSegments>')
    (folder / 'AcqData' / 'MSMassCal.bin').write_bytes(calibration)
    (folder / 'AcqData' / 'DefaultMassCal.xml').write_text('<DefaultCalibration DefaultCalibrationID="1"><Step Number="2"><CalibrationFormula>Polynomial'
                                                           '</CalibrationFormula><ValueUseFlags>2</ValueUseFlags></Step></DefaultCalibration>')
    (folder / 'AcqData' / 'MSPeak.bin').write_bytes(peaks)
    # The third scan as from a time segment stored as centroids only: its centroid block comes first, then an empty profile block
    index = bytearray(records.tobytes())
    index[2 * 284 + 156:2 * 284 + 220], index[2 * 284 + 220:2 * 284 + 284] = index[2 * 284 + 220:2 * 284 + 284], struct.pack('<i', 1) + bytes(60)
    (folder / 'AcqData' / 'MSScan.bin').write_bytes(bytes(88) + struct.pack('<i', 228) + bytes(136) + bytes(index))
    df = load_spectra_filepath(str(folder), extract_ms1 = True)
    # Both precursors move onto their monoisotopic survey centroids; MassHunter's charge 2 is kept, its charge 1 counts as undetermined
    assert np.allclose(df['m/z'], [965.844, 675.36], atol = 1e-4)
    assert df['precursor_charge'].tolist()[0] == 2 and np.isnan(df['precursor_charge'].tolist()[1])
    # The doubly charged precursor's envelope is no fragment (without a charge state the envelope is not known, as in Q-TOF mzML); flight times give back the m/z
    assert sorted(df['peak_d'][0]) == pytest.approx([290.09, 675.3, 800.3], abs = 1e-6) and len(df['peak_d'][1]) == 5
    assert df['RT'].tolist() == [2.51, 2.52] and df['scan'].tolist() == [2, 3] and df['collision_energy'].tolist() == [10.0, 10.0]
    assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and df.attrs['detected_mode'] == 'positive'
    assert df['analyzer'].tolist() == ['tof'] * 2 and df['polarity'].tolist() == ['+'] * 2 and df['activation'].tolist() == ['HCD'] * 2
    assert len(df.attrs['ms1'][0]) == 1 and df.attrs['ms1'][1][0] == pytest.approx(675.36, abs = 1e-4)
    # A scan stored as a profile only (no centroid block) is skipped; a run whose survey scans are all profiles is refused
    index[2 * 284 + 156:2 * 284 + 160] = struct.pack('<i', 1)
    (folder / 'AcqData' / 'MSScan.bin').write_bytes(bytes(88) + struct.pack('<i', 228) + bytes(136) + bytes(index))
    assert load_spectra_filepath(str(folder))['scan'].tolist() == [2]
    index[220:224] = struct.pack('<i', 1)
    (folder / 'AcqData' / 'MSScan.bin').write_bytes(bytes(88) + struct.pack('<i', 228) + bytes(136) + bytes(index))
    with pytest.raises(ValueError, match = 'profile spectra only'):
        load_spectra_filepath(str(folder))
    # A run stored as profile spectra only is refused with a hint
    (folder / 'AcqData' / 'MSPeak.bin').unlink()
    with pytest.raises(ValueError, match = 'profile spectra only'):
        load_spectra_filepath(str(folder))


def test_sciex_wiff(tmp_path):
    # A SCIEX .wiff (OLE compound file, version 3 here: streams below 4096 bytes in the root's mini stream) with the streams read_wiff takes: per experiment
    # ExperimentHeaderEx (bins summed per point at 56, mass range at 66), and of Sample1 the scan index Idx, the IDA slots DDERealTimeData, TDCInfo (TOF bin in
    # ns) and TOFCalibrationData (default and per-scan A, t0); the .wiff.scan beside it holds each spectrum as a run-length byte code of its TOF profile
    from candycrunch.utils import WIFF_SCAN, read_compound_file

    def compound_file(streams):
        # Small streams in 64-byte mini sectors of the mini stream, the others, the mini stream, the mini FAT, the directory and the FAT in 512-byte sectors
        sectors, fat, ministream, minifat = [], [], b'', []

        def place(data, table, size):
            first = len(table)
            n = max(1, -(-len(data) // size))
            table += list(range(first + 1, first + n)) + [0xFFFFFFFE]
            return first, data.ljust(n * size, b'\0')

        entries = [['Root Entry', 5, -1, -1, -1, 0, 0]]
        for path, data in streams.items():
            parent = 0
            for k, name in enumerate(path.split('/')):
                child = next((i for i, e in enumerate(entries) if e[0] == name and e[7:] == [parent]), None)
                if child is None:
                    child = len(entries)
                    entries.append([name, 2 if k == len(path.split('/')) - 1 else 1, -1, -1, -1, 0, 0, parent])
                    # Siblings as a chain of right links under their parent's child link
                    last = entries[parent][4]
                    while last != -1 and entries[last][3] != -1:
                        last = entries[last][3]
                    entries[last if last != -1 else parent][3 if last != -1 else 4] = child
                parent = child
            if len(data) < 4096:
                start, block = place(data, minifat, 64)
                ministream += block
            else:
                start, block = place(data, fat, 512)
                sectors.append(block)
            entries[parent][5:7] = [start, len(data)]
        entries[0][5:7] = list(place(ministream, fat, 512)[:1]) + [len(ministream)]
        sectors.append(ministream.ljust(-(-len(ministream) // 512) * 512, b'\0'))
        first_minifat, block = place(np.array(minifat, '<u4').tobytes(), fat, 512)
        sectors.append(block)
        directory = b''.join(e[0].encode('utf-16-le').ljust(64, b'\0') + struct.pack('<HBB3i16x4x16x', 2 * len(e[0]) + 2, e[1], 1, *e[2:5]) +
                             struct.pack('<IQ', *e[5:7]) for e in entries)
        first_dir, block = place(directory, fat, 512)
        sectors.append(block)
        n_fat = -(-(len(fat) + 1) * 4 // 512)
        fat += [0xFFFFFFFD] * n_fat
        sectors.append(np.array(fat, '<u4').tobytes().ljust(n_fat * 512, b'\xff'))
        header = (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + bytes(16) + struct.pack('<5H6x', 0x3E, 3, 0xFFFE, 9, 6) +
                  struct.pack('<9I', 0, n_fat, first_dir, 0, 4096, first_minifat, -(-len(minifat) * 4 // 512), 0xFFFFFFFE, 0) +
                  struct.pack('<109I', *range(len(fat) - n_fat, len(fat)), *[0xFFFFFFFF] * (109 - n_fat)))
        return header + b''.join(sectors)

    def spectrum(first, summed, peaks):
        # Each peak as five points around the stored point nearest its m/z (point k at TOF bin first + summed (k + 1)), coded as intensities (below 7C, or
        # 7C/7D/7E + 1/2/4 bytes) and gaps of empty points (80-FB, or FC/FD/FE + 1/2/4 bytes), FF at the end and FF padding to 4 bytes; gives the m/z of
        # the peaks' central points
        code, pos, centres = bytearray(), 0, []
        for mz, heights in sorted(peaks):
            centre = round(((np.sqrt(mz) / a + t0) / bin_ns - first) / summed - 1)
            centres.append((a * ((first + summed * (centre + 1)) * bin_ns - t0)) ** 2)
            for step, value in zip(range(centre - 2, centre + 3), heights):
                gap = step - pos
                code += b'' if not gap else bytes([0x80 + gap]) if gap < 0x7C else b'\xfc' + bytes([gap]) if gap < 256 else b'\xfd' + struct.pack('<H', gap) if gap < 65536 else b'\xfe' + struct.pack('<I', gap)
                code += bytes([value]) if value < 0x7C else b'\x7c' + bytes([value]) if value < 256 else b'\x7d' + struct.pack('<H', value) if value < 65536 else b'\x7e' + struct.pack('<I', value)
                pos = step + 1
        block = b'\xff' * 4 + struct.pack('<I', first) + b'\0' + code + b'\xff'
        return block + b'\xff' * (-len(block) % 4), centres

    a, t0, bin_ns = 0.000702096, 0.04, 0.0251173474
    envelope = [(965.844, (300, 9000, 31869, 9000, 300)), (965.844 + 1.003355 / 2, (200, 6000, 21000, 6000, 200)), (965.844 + 1.003355, (50, 1500, 5000, 1500, 50)),
                (675.36, (100, 2000, 70000, 2000, 100))]
    fragments = [(290.09, (30, 200, 300, 200, 30)), (800.3, (40, 255, 700, 255, 40)), (1500.7, (5, 50, 124, 50, 5))]
    # Two cycles of a survey scan and two dependent scans: the first dependent scan of each cycle fragmented a precursor, the second found none (no data)
    rows = [(1000.0, 1389000, 4, envelope), (1150.0, 400000, 8, fragments), (1300.0, None, 8, None), (1600.0, 1389100, 4, envelope),
            (1750.0, 400100, 8, fragments), (1900.0, None, 8, None)]
    scan_file, index, centres = bytearray(44), np.zeros(len(rows), WIFF_SCAN), []
    for k, (rt, first, summed, peaks) in enumerate(rows):
        block, mzs = spectrum(first, summed, peaks) if peaks else (b'', [])
        centres.append(mzs)
        index[k] = (len(scan_file) - 44, len(block), rt, 0, 0, 0)
        scan_file += block
    slots = np.zeros((2, 2, 4))
    slots[0, 0, [0, 2]], slots[1, 0, [0, 2]] = (965.846, 31869), (675.3605, 70000)
    calibration = struct.pack('<2dI', a, t0, 0) + struct.pack('<I', len(rows)) + b''.join(struct.pack('<2dI', a, t0, 0) for _ in rows)
    streams = {f'MethodSubtree/Method1/DeviceMethod0/Period0/Experiment{e}/ExperimentHeaderEx': bytes(56) + struct.pack('<I6x2d4x', 8 if e else 4, 50.0 if e else 600.0, 2000.0)
               for e in range(3)}
    streams.update({'SampleSubtree/Sample1/Idx': bytes(32) + index.tobytes(), 'SampleSubtree/Sample1/DDERealTimeData': bytes(32) + slots.tobytes(),
                    'SampleSubtree/Sample1/TDCInfo': bytes(32) + struct.pack('<d', bin_ns) + bytes(32), 'SampleSubtree/Sample1/TOFCalibrationData': bytes(32) + calibration,
                    # A stream above the mini stream cutoff, read from regular sectors
                    'SampleSubtree/Sample1/Log': bytes(range(256)) * 20})
    path = tmp_path / 'run.wiff'
    path.write_bytes(compound_file(streams))
    assert read_compound_file(str(path)) == streams
    (tmp_path / 'run.wiff.scan').write_bytes(bytes(scan_file))
    for default_only in (False, True):
        if default_only:
            # Without per-scan records every scan takes the default calibration
            streams['SampleSubtree/Sample1/TOFCalibrationData'] = bytes(32) + calibration[:20] + struct.pack('<I', 0)
            path.write_bytes(compound_file(streams))
        df = load_spectra_filepath(str(path), extract_ms1 = True)
        # Both precursors move onto their survey centroids, a few mDa from the IDA m/z; SCIEX stores no charge states
        assert np.allclose(df['m/z'], [centres[0][1], centres[3][0]], atol = 1e-5) and df['precursor_charge'].isna().all()
        assert [sorted(d) for d in df['peak_d']] == [pytest.approx(centres[1], abs = 1e-5), pytest.approx(centres[4], abs = 1e-5)]
        assert df['intensity'].tolist() == [31869, 70000] and df['RT'].tolist() == pytest.approx([1150.0 / 60000, 1750.0 / 60000])
        assert df['scan'].tolist() == ['sample=1 period=1 cycle=1 experiment=2', 'sample=1 period=1 cycle=2 experiment=2'] and df['scan_low'].tolist() == [50.0, 50.0]
        assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and df.attrs['detected_mode'] is None
        assert df['analyzer'].tolist() == ['tof'] * 2 and df['activation'].tolist() == ['HCD'] * 2
        rts, mzs, ints, offsets = df.attrs['ms1']
        assert len(rts) == 2 and np.allclose(mzs[:offsets[1]], centres[0], atol = 1e-4) and ints[:offsets[1]].tolist() == [74200, 50469, 33400, 8100]
    # A run stopped within a cycle: the index ends with a survey scan and one dependent scan, the IDA slots with that scan's slot
    index = np.concatenate([index, np.zeros(2, WIFF_SCAN)]).astype(WIFF_SCAN)
    for k, (rt, first, summed, peaks) in enumerate([(2200.0, 1389200, 4, envelope), (2350.0, 400200, 8, fragments)]):
        block, mzs = spectrum(first, summed, peaks)
        centres.append(mzs)
        index[len(rows) + k] = (len(scan_file) - 44, len(block), rt, 0, 0, 0)
        scan_file += block
    streams['SampleSubtree/Sample1/Idx'] = bytes(32) + index.tobytes()
    streams['SampleSubtree/Sample1/DDERealTimeData'] = bytes(32) + slots.tobytes() + struct.pack('<4d', 965.846, 0, 31869, 0)
    path.write_bytes(compound_file(streams))
    (tmp_path / 'run.wiff.scan').write_bytes(bytes(scan_file))
    df = load_spectra_filepath(str(path))
    assert df['scan'].tolist() == [f'sample=1 period=1 cycle={c} experiment=2' for c in (1, 2, 3)] and df['m/z'].tolist()[2] == pytest.approx(centres[6][1], abs = 1e-4)
    # The spectra live in the .wiff.scan, which has to come with the .wiff
    (tmp_path / 'run.wiff.scan').unlink()
    with pytest.raises(ValueError, match = 'wiff.scan beside it'):
        load_spectra_filepath(str(path))


def test_tims_scans(tmp_path):
    # msconvert writes a timsTOF run, unless asked to combine ion mobility scans, as one spectrum per scan (native ID frame=f scan=s), each at its
    # frame's retention time. Each frame becomes one survey scan and each precursor's scans one MS2 spectrum, centroided on the TOF grid; taken one
    # by one, the scans of a frame were survey scans at one retention time, which crashed the XIC smoothing (OverflowError: cannot convert float
    # infinity to integer), and Bruker's charge 1 counted as undetermined
    def cv(acc, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="" value="{value}"/>'
    def spectrum(i, frame, scan, level, rt, peaks, precursor = None):
        arrays = ''.join(f'<binaryDataArray encodedLength="0">{cv("MS:1000523")}{cv("MS:1000576")}{cv(acc)}<binary>'
                         f'{base64.b64encode(np.array(values, dtype = "<f8").tobytes()).decode()}</binary></binaryDataArray>'
                         for acc, values in (('MS:1000514', [m for m, _ in peaks]), ('MS:1000515', [x for _, x in peaks])))
        prec = (f'<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>{cv("MS:1000744", precursor)}{cv("MS:1000041", 1)}'
                f'</selectedIon></selectedIonList><activation>{cv("MS:1000133")}</activation></precursor></precursorList>') if precursor else ''
        return (f'<spectrum index="{i}" id="frame={frame} scan={scan}" defaultArrayLength="{len(peaks)}">{cv("MS:1000511", level)}{cv("MS:1000130")}'
                f'{cv("MS:1000127")}<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" '
                f'unitName="second"/></scan></scanList>{prec}<binaryDataArrayList count="2">{arrays}</binaryDataArrayList></spectrum>')
    # Peaks on the TOF grid (equal steps in sqrt(m/z)), five bins each, spread over three ion mobility scans
    def tof_peaks(mzs, height):
        bins = [round((np.sqrt(mz) - 10) / 7.8e-5) for mz in mzs]
        return [[((10 + 7.8e-5 * (b + d)) ** 2, height * w / 3) for b in bins for d, w in zip(range(-2, 3), (1, 3, 5, 3, 1))] for _ in range(3)]
    spectra, frame = [], 0
    for rt, height in zip(np.arange(600, 615, 1.1), np.exp(-0.5 * ((np.arange(600, 615, 1.1) - 607) / 3) ** 2) * 1e4):
        frame += 1
        spectra += [(frame, s + 1, 1, rt, peaks) for s, peaks in enumerate(tof_peaks([384.1529, 385.1562], height))]
        frame += 1
        spectra += [(frame, s + 10, 2, rt + 0.1, peaks, 384.1529) for s, peaks in enumerate(tof_peaks([204.0867, 366.1395], 100))]
    path = tmp_path / 'tims.mzML'
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="r"><spectrumList count="{len(spectra)}">' +
                    ''.join(spectrum(i, *x) for i, x in enumerate(spectra)) + '</spectrumList></run></mzML>')
    df = process_mzML_stack(str(path), intensity = True, extract_ms1 = True)
    rts, mzs, ints, offsets = df.attrs['ms1']
    assert len(df) == len(rts) == 14 and len(set(rts)) == 14 and df['precursor_charge'].tolist() == [1] * 14
    # One centroid per peak, carrying the signal of all its bins and scans
    assert np.allclose(mzs[offsets[6]:offsets[7]], [384.1529, 385.1562], atol = 2e-4) and np.isclose(ints[offsets[6]:offsets[7]].sum(), 2 * 13 * 1e4 * np.exp(-0.5 * (606.6 - 607) ** 2 / 9), rtol = 1e-3)
    assert np.allclose(sorted(df['peak_d'][0]), [204.0867, 366.1395], atol = 2e-4) and np.allclose(df['m/z'], 384.1529)
    area = extract_xic_areas(df.attrs['ms1'], [384.1529], [10.1], [1])
    assert np.isfinite(area).all() and area[0] > 0


def test_tof_precursors():
    # One timsTOF survey frame: a z = 2 isotope envelope at 1111.392 and an unrelated z = 1 peak at 1000.0. Bruker's monoisotopic pick on the 13C peak walks
    # down, the other picks keep their m/z, no spectrum is dropped (the PASEF isolation window holds the whole envelope), and the intensity is the isotope
    # envelope M..M+3 within 20 ppm, not the window around it
    peaks = [(1000.0, 500), (1111.392, 100), (1111.892, 98), (1112.393, 50), (1112.895, 18)]
    ms1 = (np.array([10.0]), np.array([m for m, _ in peaks], dtype = np.float32), np.array([i for _, i in peaks], dtype = np.float32), np.array([0, 5]))
    # Picks a few ppm off their survey centroid (Bruker's per-mobility-feature m/z) move onto it
    out, drop, intensities = refine_tof_precursors(ms1, [1111.3935, 1111.892, 1000.003], [10.0, 10.0, 10.0], [0, 0, 0], [2, 2, 1])
    assert abs(out[0] - 1111.392) < 1e-3 and abs(out[1] - 1111.392) < 1e-3 and abs(out[2] - 1000.0) < 1e-4 and not drop.any()
    assert np.allclose(intensities, [266, 266, 500])


def test_tdf_frames(tmp_path):
    # A DDA-PASEF .d with one survey frame and one PASEF frame, written as compression type 2 (zstd) and as type 1 (LZF per scan, as literal runs and one
    # overlapping back reference): survey peaks come back at the m/z of their TOF index, summed over scans, and the precursor's MS2 spectrum sums only its
    # own scans (2-5; the peak in scan 8 belongs to another precursor)
    try:
        from compression.zstd import compress
    except ImportError:
        from zstandard import compress
    def tof(mz):
        return round((300 + np.sqrt(1e12 / 157000 * mz) - 24864) / 0.2)
    def mz(tof):
        return (tof * 0.2 + 24864 - 300) ** 2 * 157000 / 1e12
    survey = {3: [(tof(1000.0), 500)], 5: [(tof(1111.392), 100)], 6: [(tof(1111.392), 50)]}
    pasef = {2: [(tof(500.0), 40)], 3: [(tof(600.0), 100), (tof(600.0) + 1, 100), (tof(600.0) + 2, 100)], 4: [(tof(500.0), 60), (tof(700.0), 30)],
             8: [(tof(800.0), 1000)]}
    outputs = []
    for compression in ('2', '1'):
        folder = tmp_path / f'type{compression}.d'
        folder.mkdir()
        frames, tims_ids = b'', []
        for scans in (survey, pasef):
            tims_ids.append(len(frames))
            if compression == '2':
                values = [10] + [2 * len(scans.get(s, [])) for s in range(9)]
                for s in range(10):
                    last = -1
                    for t, i in scans.get(s, []):
                        values += [t - last, i]
                        last = t
                data = compress(np.array(values, '<u4').view(np.uint8).reshape(-1, 4).T.tobytes())
                frames += struct.pack('<2I', 8 + len(data), 10) + data
                continue
            blocks = []
            for s in range(10):
                raw, last = [], 0
                for t, i in scans.get(s, []):
                    raw += ([last - t] if t > last else []) + [i]
                    last = t + 1
                raw = np.array(raw, '<i4').tobytes()
                if s == 3 and scans is pasef:
                    # The three equal intensities of PASEF scan 3: the skip and the first as literals, then 8 bytes copied from 4 bytes back
                    blocks.append(bytes([7]) + raw[:8] + bytes([0xc0, 3]))
                else:
                    blocks.append(b''.join(bytes([len(raw[k:k + 32]) - 1]) + raw[k:k + 32] for k in range(0, len(raw), 32)))
            offsets = np.cumsum([8 + 44] + [len(b) for b in blocks])
            frames += struct.pack('<2I', offsets[-1], 10) + np.array(offsets, '<u4').tobytes() + b''.join(blocks)
        (folder / 'analysis.tdf_bin').write_bytes(frames)
        db = sqlite3.connect(folder / 'analysis.tdf')
        db.execute('CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT)')
        db.execute("INSERT INTO GlobalMetadata VALUES ('TimsCompressionType', ?)", (compression,))
        db.execute('CREATE TABLE Frames (Id INTEGER, Time REAL, Polarity TEXT, MsMsType INTEGER, TimsId INTEGER, NumPeaks INTEGER, MzCalibration INTEGER, T1 REAL, T2 REAL)')
        db.executemany('INSERT INTO Frames VALUES (?, ?, ?, ?, ?, ?, 1, 25.0, 25.0)', [(1, 600.0, '-', 0, tims_ids[0], 3), (2, 600.1, '-', 8, tims_ids[1], 7)])
        db.execute('CREATE TABLE MzCalibration (Id INTEGER, ModelType INTEGER, DigitizerTimebase REAL, DigitizerDelay REAL, T1 REAL, T2 REAL, dC1 REAL, dC2 REAL, C0 REAL, C1 REAL, C2 REAL, C3 REAL, C4 REAL)')
        db.execute('INSERT INTO MzCalibration VALUES (1, 1, 0.2, 24864, 25.0, 25.0, 0, 0, 300, 157000, 0, 0, 0)')
        db.execute('CREATE TABLE Precursors (Id INTEGER, LargestPeakMz REAL, MonoisotopicMz REAL, Charge INTEGER, Intensity REAL, Parent INTEGER)')
        db.execute('INSERT INTO Precursors VALUES (1, ?, ?, 2, 100, 1)', (mz(tof(1111.392)), mz(tof(1111.392))))
        db.execute('CREATE TABLE PasefFrameMsMsInfo (Frame INTEGER, ScanNumBegin INTEGER, ScanNumEnd INTEGER, Precursor INTEGER, IsolationMz REAL, IsolationWidth REAL)')
        db.execute('INSERT INTO PasefFrameMsMsInfo VALUES (2, 2, 6, 1, 1111.4, 3.0)')
        db.commit()
        db.close()
        outputs.append(process_tdf_stack(str(folder), intensity = True, extract_ms1 = True))
    for df in outputs:
        _, ms1_mzs, ms1_ints, _ = df.attrs['ms1']
        assert np.allclose(ms1_mzs, [mz(tof(1000.0)), mz(tof(1111.392))], rtol = 1e-7) and np.allclose(ms1_ints, [500, 150])
        assert len(df) == 1 and abs(df['m/z'][0] - mz(tof(1111.392))) < 1e-4 and df['precursor_charge'][0] == 2 and abs(df['RT'][0] - 10.0) < 1e-9
        peaks = sorted(df['peak_d'][0].items())
        assert np.allclose([m for m, _ in peaks], [mz(tof(500.0)), mz(tof(600.0) + 1), mz(tof(700.0))], rtol = 1e-6) and np.allclose([i for _, i in peaks], [100, 300, 30])
    assert outputs[0]['peak_d'][0] == outputs[1]['peak_d'][0]
    # Details from the PASEF table and frames; this file has no collision energy column and no acquisition range
    assert outputs[0][['analyzer', 'polarity', 'isolation_width']].values.tolist() == [['tof', '-', 3.0]] and outputs[0][['collision_energy', 'scan_low']].isna().all(axis = None)


def test_thermo_raw(tmp_path):
    # A version 66 Thermo .raw file with an ion-trap profile survey scan (a peak at 1000.0 on Thermo's axis, stored one bin lower) and two centroided MS2
    # scans, one with an undetermined precursor (isolation m/z and no charge) and one with the instrument's monoisotopic m/z, charge and master scan;
    # their scan events differ in size, and the trailer extra header sits behind the error log
    def text(s):
        return struct.pack('<I', len(s)) + s.encode('utf-16-le')
    def event(level, mode, reactions):
        preamble = bytearray(136)
        preamble[4:7] = bytes([0, mode, level])
        return (bytes(preamble) + struct.pack('<I', len(reactions)) + b''.join(struct.pack('<3dI28x', mz, 2.0, energy, code << 1) for mz, energy, code in reactions)
                + struct.pack('<I2dI3I', 1, 150.0, 2000.0, 0, 0, 0, 0))
    profile = struct.pack('<2d2I2I5f', 100.0, 0.1, 1, 0, 8997, 5, 10, 50, 100, 50, 10)
    centroids = [[(400.1, 50.0), (600.2, 100.0)], [(300.0, 20.0)]]
    packets = [struct.pack('<4I24x', 1, len(profile) // 4, 0, 0) + profile] + [struct.pack('<4I24x', 1, 0, 1 + 2 * len(c), 0) + struct.pack('<I', len(c))
                                                                                  + b''.join(struct.pack('<2f', *p) for p in c) for c in centroids]
    events = event(1, 1, []) + event(2, 0, [(1000.0, 35.0, 0)]) + event(2, 0, [(1000.5, 30.0, 5)])
    header = struct.pack('<I', 3) + b''.join(struct.pack('<2I', kind, 0) + text(label) for kind, label in [(11, 'Monoisotopic M/Z:'), (6, 'Charge State:'), (8, 'Master Scan Number:')])
    trailer = struct.pack('<dhi', 0, 0, 0) + struct.pack('<dhi', 0, 0, 1) + struct.pack('<dhi', 1000.02, 2, 1)
    start = 1356 + 64 + 32 * 4 + 24 + 4
    run_header, data = start + 824 + 32, start + 824 + 32 + 7464
    index = data + sum(len(p) for p in packets)
    errors = index + 3 * 88
    events_addr = errors + 4 + len(header)
    params = events_addr + 4 + len(events)
    head = bytearray(1356)
    head[0:18], head[36:40] = b'\x01\xa1' + 'Finnigan'.encode('utf-16-le'), struct.pack('<I', 66)
    info = bytearray(824)
    info[28:32] = struct.pack('<I', 1)
    rh = bytearray(7464)
    rh[8:24] = struct.pack('<4I', 1, 3, 0, 0)
    rh[7376:7380] = struct.pack('<I', 3)
    rh[7408:7464] = struct.pack('<7Q', index, data, 0, errors, 0, events_addr, params)
    entries = b''
    for rt, packet in zip([10.0, 10.01, 10.02], np.cumsum([0] + [len(p) for p in packets[:-1]])):
        entry = bytearray(88)
        entry[24:32], entry[72:80] = struct.pack('<d', rt), struct.pack('<Q', int(packet))
        entries += bytes(entry)
    path = tmp_path / 'run.raw'
    path.write_bytes(bytes(head) + bytes(64) + text('') * 16 + bytes(4) + text('') * 15 + bytes(24) + text('') + bytes(info) + struct.pack('<Q24x', run_header)
                     + bytes(rh) + b''.join(packets) + entries + bytes(4) + header + bytes(4) + events + trailer)
    table = candycrunch.utils.read_raw_scan_table(str(path))
    assert table['ms_level'] == [1, 2, 2] and table['filter_string'] == ['', '1000.0000@cid35.00', '1000.5000@hcd30.00']
    assert table['precursor_mz'] == [None, 1000.0, 1000.02] and table['charge'] == [None, None, 2] and table['master_scan'] == [None, 1, 1]
    assert table['scan_range'] == [(150.0, 2000.0)] * 3
    df = process_raw_stack(str(path), intensity = True, extract_ms1 = True)
    _, ms1_mzs, ms1_ints, _ = df.attrs['ms1']
    assert np.allclose(ms1_mzs, [1000.0]) and np.allclose(ms1_ints, [220])
    assert df.attrs['detected_mode'] == 'negative' and df.attrs['detected_trap'] == df.attrs['detected_ms1_trap'] == 'linear'
    assert np.allclose(df['m/z'], [1000.0, 1000.02]) and pd.isna(df['precursor_charge'][0]) and df['precursor_charge'][1] == 2
    assert list(df['scan']) == [2, 3] and list(df['activation']) == ['CID', 'HCD'] and np.allclose(df['intensity'], [220, 220]) and np.allclose(df['RT'], [10.01, 10.02])
    assert np.allclose(list(df['peak_d'][0]), [600.2, 400.1]) and list(df['peak_d'][0].values()) == [100, 50]
    assert df['analyzer'].tolist() == ['linear'] * 2 and df['polarity'].tolist() == ['-'] * 2 and df['collision_energy'].tolist() == [35.0, 30.0]
    assert df['scan_low'].tolist() == [150.0] * 2 and df['header_shift'].tolist() == [0.0, 0.0]


def test_qtof_mzml(tmp_path):
    # A Waters Q-TOF DDA run as msconvert writes it (pwiz's ATEHLSTLSEK_LM_684.3469.raw): the generic Waters model without analyzers, native IDs
    # function=f process=0 scan=s, survey scans in function 1, the lockmass function 3 written as MS1 scans between them and the MS2 spectra (function 2),
    # and the set mass as precursor m/z without a charge state; a SCIEX-style MS2 spectrum names the TOF analyzer, a charge state and the isolation window
    def cv(acc, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="" value="{value}"/>'
    def spectrum(i, native, level, rt, peaks, precursor = None, charge = None, ic = 'IC1'):
        arrays = ''.join(f'<binaryDataArray encodedLength="0">{cv("MS:1000523")}{cv("MS:1000576")}{cv(acc)}<binary>'
                         f'{base64.b64encode(np.array(values, dtype = "<f8").tobytes()).decode()}</binary></binaryDataArray>'
                         for acc, values in (('MS:1000514', [m for m, _ in peaks]), ('MS:1000515', [x for _, x in peaks])))
        prec = (f'<precursorList count="1"><precursor><isolationWindow>{cv("MS:1000827", precursor)}{cv("MS:1000828", 0.7)}{cv("MS:1000829", 0.7)}</isolationWindow>'
                f'<selectedIonList count="1"><selectedIon>{cv("MS:1000744", precursor)}{cv("MS:1000041", charge) if charge else ""}</selectedIon></selectedIonList>'
                f'<activation>{cv("MS:1000422")}</activation></precursor></precursorList>') if precursor else ''
        return (f'<spectrum index="{i}" id="{native}" defaultArrayLength="{len(peaks)}">{cv("MS:1000511", level)}{cv("MS:1000130")}{cv("MS:1000127")}'
                f'<scanList count="1"><scan instrumentConfigurationRef="{ic}"><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" '
                f'unitName="minute"/></scan></scanList>{prec}<binaryDataArrayList count="2">{arrays}</binaryDataArrayList></spectrum>')
    rng = np.random.default_rng(0)
    survey = [(500.2, 2e5), (608.3163, 8e5), (608.8172, 4.8e5), (609.319, 1.8e5), (609.82, 5e4)]
    envelope = [(608.3163, 9e4), (608.8172, 5e4), (609.319, 2e4)]
    fragments = [(float(m), 1e4 * (k + 1)) for k, m in enumerate(np.linspace(200, 1100, 20))] + [(float(m), float(i)) for m, i in zip(rng.uniform(100, 1200, 300), rng.uniform(5, 30, 300))]
    spectra = [spectrum(0, 'function=3 process=0 scan=1', 1, 22.39, [(556.2771, 5e4)]), spectrum(1, 'function=1 process=0 scan=1', 1, 22.40, survey),
               spectrum(2, 'function=3 process=0 scan=2', 1, 22.41, [(556.2771, 5e4), (608.3156, 633)]),
               spectrum(3, 'function=2 process=0 scan=1', 2, 22.42, fragments + envelope, 608.321),
               spectrum(4, 'function=2 process=0 scan=2', 2, 22.43, fragments + envelope, 608.3245, 2, 'IC2')]
    path = tmp_path / 'qtof.mzML'
    path.write_text('<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><instrumentConfigurationList count="2">'
                    f'<instrumentConfiguration id="IC1">{cv("MS:1000126")}</instrumentConfiguration><instrumentConfiguration id="IC2">{cv("MS:1000121")}'
                    f'<componentList count="2"><analyzer order="2">{cv("MS:1000081")}</analyzer><analyzer order="3">{cv("MS:1000084")}</analyzer></componentList>'
                    f'</instrumentConfiguration></instrumentConfigurationList><run id="r"><spectrumList count="5">{"".join(spectra)}</spectrumList></run></mzML>')
    df = process_mzML_stack(str(path), intensity = True, extract_ms1 = True)
    # The lockmass scans are no survey scans: the precursors (8 ppm and 14 ppm off) move onto the function 1 centroid and take its intensities
    assert len(df.attrs['ms1'][0]) == 1 and np.allclose(df['m/z'], 608.3163, atol = 1e-4) and np.allclose(df['intensity'], 1.51e6)
    # TOF MS2 keep their 100 most intense peaks, and the unfragmented envelope of a precursor with a charge state goes
    assert [len(p) for p in df['peak_d']] == [100, 100] and not any(608.2 < m < 609.5 for m in df['peak_d'][1])
    assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof'


def test_agilent_mgf(tmp_path):
    # An Agilent MassHunter Q-TOF run as msconvert writes its mgf (GPST000570): native IDs scanId=, no survey scans, and precursor intensities from the
    # moment DDA triggered, which are dropped; without intensities, two stretches of spectra 0.65 min apart (anomers of an unreduced N-glycan) are two clusters
    path = tmp_path / 'run.mgf'
    path.write_text(''.join(f'BEGIN IONS\nTITLE=run.{i}.{i}. File:"run.d", NativeID:"scanId={i}"\nRTINSECONDS={rt * 60}\nPEPMASS=642.9155 {2e4 + i}\nCHARGE=3+\n'
                            '204.0867 500\n366.1395 300\n528.1923 100\nEND IONS\n' for i, rt in enumerate([15.28, 15.39, 16.04, 16.16])))
    df = load_spectra_filepath(str(path))
    assert df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and df['intensity'].isna().all()
    # wrap_inference sets missing intensities to 0 before condensing
    assert condense_dataframe(df.assign(intensity = 0))['num_spectra'].tolist() == [2, 2]


def test_mzdata(tmp_path):
    # An Agilent MassHunter Q-TOF run as MzDataExport writes it (GPST000097): mzData 1.05 without a namespace, PSI terms with names, peak-detected spectra
    # as base64 of 64-bit m/z and 32-bit intensities, the isolation target as precursor m/z with a charge state for some precursors (1 for others)

    def cv(acc, name, value):
        return f'<cvParam cvLabel="psi" accession="{acc}" name="{name}" value="{value}" />'

    def spectrum(i, level, rt, peaks, precursor = None, charge = None, endian = 'little'):
        arrays = ''.join(f'<{kind}><data precision="{bits}" endian="{endian}" length="{len(values)}">{base64.b64encode(np.array(values, dtype = (">" if endian == "big" else "<") + dtype).tobytes()).decode()}'
                         f'</data></{kind}>' for kind, bits, dtype, values in (('mzArrayBinary', 64, 'f8', [m for m, _ in peaks]), ('intenArrayBinary', 32, 'f4', [x for _, x in peaks])))
        prec = (f'<precursorList count="1"><precursor msLevel="2" spectrumRef="1"><ionSelection>{cv("PSI:1000040", "MassToChargeRatio", precursor)}'
                f'{cv("PSI:1000041", "ChargeState", charge) if charge else ""}</ionSelection><activation>{cv("PSI:1000044", "FragmentationMode", "CID")}'
                f'{cv("PSI:1000045", "CollisionEnergy", 20.5)}</activation></precursor></precursorList>') if precursor else ''
        return (f'<spectrum id="{i}"><spectrumDesc><spectrumSettings><acqSpecification spectrumType="discrete" methodOfCombination="average" count="1">'
                f'<acquisition acqNumber="{1000 + i}" /></acqSpecification><spectrumInstrument msLevel="{level}">{cv("PSI:1000036", "ScanMode", "Scan")}'
                f'{cv("PSI:1000037", "Polarity", "Positive")}{cv("PSI:1000038", "TimeInMinutes", rt)}</spectrumInstrument></spectrumSettings>{prec}</spectrumDesc>'
                f'{arrays}</spectrum>')

    rng = np.random.default_rng(0)
    survey = [(500.2, 2e5), (608.3163, 8e5), (608.8180, 4.8e5), (609.3197, 1.8e5), (609.82, 5e4), (1001.353, 3e5), (1002.356, 1.5e5)]
    envelope = [(608.3163, 9e4), (608.8180, 5e4), (609.3197, 2e4)]
    fragments = [(float(m), 1e4 * (k + 1)) for k, m in enumerate(np.linspace(200, 590, 20))] + [(float(m), float(x)) for m, x in zip(rng.uniform(100, 1200, 300), rng.uniform(5, 30, 300))]
    small = [(204.0867, 900.0), (366.1395, 500.0), (528.1923, 300.0), (1001.35, 50.0)]
    text = ('<?xml version="1.0" encoding="utf-8"?><mzData version="1.05" accessionNumber="psi-ms:100"><description><instrument><analyzerList count="1"><analyzer>'
            f'{cv("PSI:1000010", "AnalyzerType", "QuadrupoleTimeOfFlight")}</analyzer></analyzerList></instrument></description><spectrumList count="4">'
            f'{spectrum(1, 1, 22.40, survey)}{spectrum(2, 2, 22.42, fragments + envelope, 608.321, 2)}{spectrum(3, 2, 22.43, small, 1001.36, 1)}'
            f'{spectrum(4, 2, 22.44, small, 1001.36, endian = "big")}</spectrumList></mzData>')
    path = tmp_path / 'run.mzdata.xml'
    path.write_text(text)
    df = load_spectra_filepath(str(path), extract_ms1 = True)
    # The isolation targets (8 and 7 ppm off) move onto their survey centroids and, without a header intensity, take the survey centroids around them
    assert np.allclose(df['m/z'], [608.3163, 1001.353, 1001.353], atol = 1e-4) and np.allclose(df['intensity'], [1.51e6, 4.5e5, 4.5e5])
    assert df['precursor_charge'].tolist()[0] == 2 and np.isnan(df['precursor_charge'].tolist()[1:]).all()
    # TOF MS2 keep their 100 most intense peaks, and the unfragmented envelope of a precursor with a charge state goes; big-endian arrays read the same
    assert len(df['peak_d'][0]) == 100 and not any(608.2 < m < 609.5 for m in df['peak_d'][0]) and df['peak_d'][1] == df['peak_d'][2] == dict(small)
    assert df['scan'].tolist() == [2, 3, 4] and df['RT'].tolist() == [22.42, 22.43, 22.44] and df['collision_energy'].tolist() == [20.5] * 3
    assert df['activation'].tolist() == ['HCD'] * 3 and df['analyzer'].tolist() == ['tof'] * 3 and df['polarity'].tolist() == ['+'] * 3
    assert df.attrs['detected_mode'] == 'positive' and df.attrs['detected_trap'] == 'other' and df.attrs['detected_ms1_trap'] == 'tof' and len(df.attrs['ms1'][0]) == 1
    # A .cc keeps it all; an .mzdata file reads the same, a cut-short download says so, and an .xml that is no mzData is no spectra file
    assert load_spectra_filepath(extract_spectra(str(path), str(tmp_path / 'run.cc')))['m/z'].tolist() == df['m/z'].tolist()
    (tmp_path / 'run.mzdata').write_text(text)
    assert load_spectra_filepath(str(tmp_path / 'run.mzdata'))['m/z'].tolist() == df['m/z'].tolist()
    (tmp_path / 'cut.mzdata.xml').write_text(text[:len(text) // 2])
    with pytest.raises(ValueError, match = 'not a complete mzData file'):
        load_spectra_filepath(str(tmp_path / 'cut.mzdata.xml'))
    (tmp_path / 'other.xml').write_text('<?xml version="1.0"?><config/>')
    with pytest.raises(FileNotFoundError):
        load_spectra_filepath(str(tmp_path / 'other.xml'))


def test_spectra_scan_activation(tmp_path):
    # A glycoproteomics run alternates HCD and EThcD (ETD + supplemental beam-type CID) scans of one precursor; both are kept apart by scan
    # number and activation, which a search engine's identification and CandyCrumbs' fragmentation_method need
    def cv(acc, name, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="{name}" value="{value}"/>'
    def spectrum(i, native_id, level, rt, activation = ()):
        arrays = ''.join(f'<binaryDataArray encodedLength="0">{cv("MS:1000523", "64-bit float")}{cv("MS:1000576", "no compression")}{cv(acc, name)}'
                         f'<binary>{base64.b64encode(np.array(values, dtype = "<f8").tobytes()).decode()}</binary></binaryDataArray>'
                         for acc, name, values in (('MS:1000514', 'm/z array', [204.0867, 1189.512, 1392.591]), ('MS:1000515', 'intensity array', [100.0, 50.0, 80.0])))
        precursor = (f'<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>{cv("MS:1000744", "selected ion m/z", 878.69)}'
                     f'{cv("MS:1000041", "charge state", 3)}</selectedIon></selectedIonList><activation>{"".join(cv(a, "") for a in activation)}</activation>'
                     '</precursor></precursorList>') if level == 2 else ''
        return (f'<spectrum index="{i}" id="{native_id}" defaultArrayLength="3">{cv("MS:1000511", "ms level", level)}'
                f'{cv("MS:1000130", "positive scan")}<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" '
                f'unitName="minute"/></scan></scanList>{precursor}'
                f'<binaryDataArrayList count="2">{arrays}</binaryDataArrayList></spectrum>')
    def write(name, spectra):
        path = tmp_path / name
        path.write_text(f'<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="r"><spectrumList count="{len(spectra)}">' +
                        ''.join(spectrum(i, *x) for i, x in enumerate(spectra)) + '</spectrumList></run></mzML>')
        return str(path)
    thermo = write('thermo.mzML', [(f'controllerType=0 controllerNumber=1 scan={i + 1}', *x) for i, x in
                                   enumerate([(1, 10.0), (2, 10.01, ['MS:1000422']), (2, 10.02, ['MS:1000598', 'MS:1002678'])])])
    df = load_spectra_filepath(thermo)
    assert df['scan'].tolist() == [2, 3] and df['activation'].tolist() == ['HCD', 'EThcD']
    assert df.attrs['detected_mode'] == 'positive' and df['precursor_charge'].tolist() == [3, 3]
    # Both survive the .xlsx export that wrap_inference reads directly
    df_xlsx = load_spectra_filepath(extract_spectra(thermo))
    assert df_xlsx['scan'].tolist() == [2, 3] and df_xlsx['activation'].tolist() == ['HCD', 'EThcD']
    # SCIEX native IDs have no scan= number and repeat their experiment number every cycle, so the whole ID identifies a spectrum; EAD gives
    # c/z ions like ECD, and ETciD and low-energy CID have terms of their own
    sciex = write('sciex.mzML', [(f'sample=1 period=1 cycle={c} experiment={e}', *x) for c, e, x in
                                 [(1, 1, (1, 10.0)), (1, 2, (2, 10.01, ['MS:1003294'])), (2, 2, (2, 10.02, ['MS:1000433'])), (3, 2, (2, 10.03, ['MS:1003182']))]])
    df = load_spectra_filepath(sciex)
    assert df['scan'].tolist() == [f'sample=1 period=1 cycle={c} experiment=2' for c in (1, 2, 3)]
    assert df['activation'].tolist() == ['ECD', 'CID', 'ETciD']
    # Text scan IDs travel in the .cc export's JSON header
    assert load_spectra_filepath(extract_spectra(sciex, file_format = 'cc'))['scan'].tolist() == df['scan'].tolist()
    # MGF scan numbers are integers as in mzML
    mgf_path = tmp_path / 'glycopeptides.mgf'
    mgf_path.write_text('BEGIN IONS\nTITLE=s1\nPEPMASS=878.69 1000\nCHARGE=3+\nRTINSECONDS=600\nSCANS=1234\n204.0867 100\n1189.512 50\nEND IONS\n')
    assert load_spectra_filepath(str(mgf_path))['scan'].tolist() == [1234]


def test_ms3_spectra(tmp_path):
    # Thermo DDA isolates the base peak of an MS2 spectrum for MS3; the MS3 precursors are listed latest stage first, each referencing its parent
    # scan, so MS3 spectra end up with their MS2 spectrum (in .xlsx exports, too), and files without MS3 get no column
    def cv(acc, name, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="{name}" value="{value}"/>'

    def spectrum(i, level, rt, peaks, precursors = ()):
        arrays = ''.join(
            f'<binaryDataArray encodedLength="0">{cv("MS:1000523", "64-bit float")}{cv("MS:1000576", "no compression")}{cv(acc, name)}'
            f'<binary>{base64.b64encode(np.array(values, dtype = "<f8").tobytes()).decode()}</binary></binaryDataArray>'
            for acc, name, values in (('MS:1000514', 'm/z array', [m for m, _ in peaks]),
                                      ('MS:1000515', 'intensity array', [x for _, x in peaks])))
        precursor = ''.join(
            f'<precursor spectrumRef="scan={ref}"><selectedIonList count="1"><selectedIon>{cv("MS:1000744", "selected ion m/z", mz)}'
            f'</selectedIon></selectedIonList><activation>{cv("MS:1000133", "")}</activation></precursor>' for ref, mz
            in precursors)
        return (
            f'<spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(peaks)}">{cv("MS:1000511", "ms level", level)}{cv("MS:1000129", "negative scan")}'
            f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitName="minute"/></scan></scanList>'
            f'{"<precursorList>" + precursor + "</precursorList>" if precursors else ""}<binaryDataArrayList count="2">{arrays}</binaryDataArrayList></spectrum>')

    def write(name, spectra):
        path = tmp_path / name
        path.write_text(
            f'<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="r"><spectrumList count="{len(spectra)}">' +
            ''.join(spectrum(i, *x) for i, x in enumerate(spectra)) + '</spectrumList></run></mzML>')
        return str(path)

    ms1, ms2 = (1, 10.0, [(675.25, 1e5), (676.25, 3e4)]), [(290.09, 100.0), (384.15, 40.0)]
    path = write('ms3.mzML', [ms1, (2, 10.01, ms2, [(1, 675.25)]), (2, 10.02, ms2, [(1, 675.25)]),
                              (3, 10.03, [(272.08, 20.0), (170.05, 10.0)], [(2, 290.09), (1, 675.25)])])
    df = load_spectra_filepath(path)
    assert [[(p, list(d)) for p, d in x] for x in df['ms3']] == [[(290.09, [272.08, 170.05])], []]
    assert load_spectra_filepath(extract_spectra(path))['ms3'].tolist() == df['ms3'].tolist()
    # The .cc export keeps every column, the MS3 spectra and the survey scans as loaded (survey scan intensities to 0.02%)
    full, cc = load_spectra_filepath(path, extract_ms1 = True), load_spectra_filepath(extract_spectra(path, file_format = 'cc'), extract_ms1 = True)
    assert cc.drop(columns = ['peak_d', 'ms3']).equals(full.drop(columns = ['peak_d', 'ms3'])) and cc.attrs['detected_mode'] == full.attrs['detected_mode'] == 'negative'
    assert cc['peak_d'].tolist() == full['peak_d'].tolist() and cc['ms3'].tolist() == full['ms3'].tolist()
    assert all(np.array_equal(cc.attrs['ms1'][i], full.attrs['ms1'][i]) for i in (0, 1, 3)) and np.allclose(cc.attrs['ms1'][2], full.attrs['ms1'][2], rtol = 3e-4, atol = 0)
    assert 'ms3' not in load_spectra_filepath(write('ms2.mzML', [ms1, (2, 10.01, ms2, [(1, 675.25)])])).columns


def test_spectra_readers(tmp_path):
    # MS-Numpress arrays (encoded with the reference implementation, pynumpress), mzXML scans nested in their MS1 scan (yielded by scan
    # number, so the MS1 scan comes first), and the mgf CHARGE and PEPMASS formats
    def cv(acc, value = ''):
        return f'<cvParam cvRef="MS" accession="{acc}" name="" value="{value}"/>'
    def spectrum(i, level, arrays):
        return (f'<spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="4">{cv("MS:1000511", level)}<scanList count="1"><scan>'
                f'<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="600" unitName="second"/></scan></scanList>'
                f'<binaryDataArrayList count="2">' + ''.join(f'<binaryDataArray encodedLength="0">{cv(comp)}{cv(acc)}<binary>{data}</binary></binaryDataArray>'
                                                            for acc, comp, data in arrays) + '</binaryDataArrayList></spectrum>')
    path = tmp_path / 'numpress.mzML'
    path.write_text('<?xml version="1.0" encoding="utf-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="r"><spectrumList count="2">' +
                    spectrum(0, 1, [('MS:1000514', 'MS:1002312', 'QVBZ7oAAAADmUyQ0rGuLXYBHSuj3'), ('MS:1000515', 'MS:1002313', 'ZGYjYFcw')]) +
                    spectrum(1, 2, [('MS:1000514', 'MS:1002746', 'eJxzDIh818DAwPAsWMVkTXZ3bIO714vvAFVHCNk='), ('MS:1000515', 'MS:1002314', 'QMu8AAAAAAD//xjawfPlTA==')]) +
                    '</spectrumList></run></mzML>')
    ms1, ms2 = read_mzml(str(path))
    assert ms1['ms_level'] == 1 and ms1['rt'] == 10.0 and ms1['precursor'] is None
    assert np.allclose(ms1['peaks'], [[204.0867, 100], [366.1395, 50], [528.1923, 80], [1189.512, 3]], rtol = 1e-9)
    assert np.array_equal(ms2['peaks'][:, 0], ms1['peaks'][:, 0]) and np.allclose(ms2['peaks'][:, 1], ms1['peaks'][:, 1], rtol = 1e-4)
    path = tmp_path / 'nested.mzXML'
    path.write_text('<?xml version="1.0"?><mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun>'
                    '<scan num="1" msLevel="1" retentionTime="PT600.5S" polarity="-"><peaks precision="32" byteOrder="network" compressionType="zlib">'
                    'eJxz0bjA4H44gMFFU4DB7VUCAwApEgSk</peaks><scan num="2" msLevel="2" retentionTime="PT601S"><precursorMz precursorScanNum="1" '
                    'precursorCharge="2" activationMethod="CID">675.25</precursorMz><peaks precision="32" byteOrder="network" compressionType="none">'
                    'Q5ELhULIAABDwBMzQiAAAA==</peaks></scan></scan></msRun></mzXML>')
    scans = list(read_mzxml(str(path)))
    assert [s['num'] for s in scans] == ['1', '2'] and scans[0]['retentionTime'] == 600.5 / 60
    assert scans[1]['precursorMz'] == [{'precursorScanNum': '1', 'precursorCharge': 2, 'activationMethod': 'CID', 'precursorMz': 675.25}]
    assert scans[0]['m/z array'].tolist() == [675.25, 676.25] and scans[1]['intensity array'].tolist() == [100.0, 40.0]
    path = tmp_path / 'charges.mgf'
    path.write_text('CHARGE=2+\nBEGIN IONS\nPEPMASS=675.25\nRTINSECONDS=600\n290.09 100\nEND IONS\nBEGIN IONS\nPEPMASS=675.25 1e5 3-\nRTINSECONDS=601\n'
                    '290.09 100 1+\nEND IONS\nBEGIN IONS\nPEPMASS=675.25\nCHARGE=2+ and 3+\n# a comment\nRTINSECONDS=602\nEND IONS\n')
    params = [s['params'] for s in read_mgf(str(path))]
    assert [p['charge'] for p in params] == [[2], [-3], [2, 3]] and [p['pepmass'] for p in params] == [(675.25, None), (675.25, 1e5), (675.25, None)]
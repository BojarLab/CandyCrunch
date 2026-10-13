import pytest
import unittest
import os
import base64
import pathlib
import sys
from tabulate import tabulate
import numpy as np
from collections import defaultdict

from candycrunch.prediction import *
from glycowork.motif.graph import compare_glycans, get_possible_topologies, graph_to_string
from glycowork.motif.annotate import get_glycan_similarity
from itertools import product
from scipy.optimize import linear_sum_assignment
from glycowork.motif.processing import canonicalize_iupac
import time

BASE_DIR = pathlib.Path(__file__).parent.parent  # Go up one level from the test file
TEST_DATA_DIR = BASE_DIR / "tests" / "data"
TEST_DICTS = [
    {'name':'milk','args': {'glycan_class':'free'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000350','args': {'glycan_class':'O'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'O.' in x]},
    {'name':'GPST000350','args': {'glycan_class':'N'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'N' in x]},
    {'name':'GPST000017','args': {'glycan_class':'O'}, 'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000017/") if 'PGMb' not in x if 'JC' in x]},
    {'name':'GPST000029','args': {'glycan_class':'O'}},
    {'name':'PMC8950484_CHO','args': {'glycan_class':'O'}},
    {'name':'GPST000307','args': {'glycan_class':'O'}},
    {'name':'GPST000487','args': {'glycan_class':'N'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000487/")]},
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
]
# Spectra files with MS1 (lower-case, Thermo writes .RAW too), run by --input-format mzml
MS1_FORMATS = ('.mzml', '.mzxml', '.raw')
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


def match_spectra(array1, array2, mass_threshold = MASS_TOLERANCE, rt_threshold = RT_TOLERANCE, array1_alt = None, array2_alt = None):
    # One-to-one assignment minimizing the summed normalized m/z and RT distances (as GlycoGauntlet's evaluator): a greedy pass in ground-truth
    # order let a peak take the row of its neighbouring isomer, and the result depended on how the ground truth was sorted
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
  gt_charges = gt['charge'].values if 'charge' in gt.columns else np.full(len(gt), -1)
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
      possible_structures = [graph_to_string(x) for x in get_possible_topologies(gt_glycan, exhaustive = True)]
      similarity_scores.append(max([1.0 if compare_glycans(p, pred_glycan) else get_glycan_similarity(p, pred_glycan) for p in possible_structures]))
    else:
      if compare_glycans(gt_glycan, pred_glycan):
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
    dict(zip([x for x in extra_param_dict], combo))
    for combo in product(*[v for v in extra_param_dict.values()])
]


def pytest_generate_tests(metafunc):
    # With --input-format both, the .xlsx and the MS1 files (.mzML/.mzXML/.raw) of a dataset are separate tests, so pytest-xdist can run them in parallel;
    # their scores were already logged and regression-checked per format
    if metafunc.function.__name__ == 'test_candycrunch_accuracy':
        input_format = metafunc.config.getoption('--input-format')
        groups = ['xlsx', 'mzml'] if input_format == 'both' else [input_format]
        metafunc.parametrize('test_params, file_group', [(p, g) for p in test_params for g in groups],
                             ids = [f"{p['test_dict']['name']}-{p['test_dict']['args']['glycan_class']}-{g}" for p in test_params for g in groups])


def test_candycrunch_accuracy(test_params, file_group, result_collector, input_format, verbose, test_files = None):
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    start_time = time.time()  # Start timing
    test_outputs = []
    test_dict = test_params['test_dict']
    folder = test_dict.get('folder', test_dict['name'])
    test_files = test_params['test_dict'].get('test_files',None)
    if test_files is None:
        test_files = [x for x in os.listdir(f"{TEST_DATA_DIR}/{folder}")]
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
        start_time = time.time()
        preds_out = wrap_inference(f"{TEST_DATA_DIR}/{folder}/{filename}",**test_dict['args'],
                                **inference_params)
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"\nTest execution time: {execution_time:.2f} seconds")  # Print execution time
        preds_out = posthoc_process_df(preds_out,posthoc_params)
        loaded_gt = pd.read_csv(f"{TEST_DATA_DIR}/{folder}/df_mz_{test_dict['name']}.csv")
        col_name  =  filename.split(".")[0]
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
    out, drop, intensities = refine_tof_precursors(ms1, [1111.392, 1111.892, 1000.0], [10.0, 10.0, 10.0], [0, 0, 0], [2, 2, 1])
    assert out[0] == 1111.392 and abs(out[1] - 1111.392) < 1e-3 and out[2] == 1000.0 and not drop.any()
    assert np.allclose(intensities, [266, 266, 500])


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
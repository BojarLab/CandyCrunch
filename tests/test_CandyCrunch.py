import pytest
import unittest
import os
import pathlib
import sys
from tabulate import tabulate
import numpy as np
from collections import defaultdict

from candycrunch.prediction import *
from glycowork.motif.graph import compare_glycans, get_possible_topologies, graph_to_string
from glycowork.motif.annotate import get_glycan_similarity
from itertools import product
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
   # {'name':'GPST000134','args': {'glycan_class':'N', 'mode':'positive'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000134/") if 'glycans_1' in x][:1]}
]
AVG_THRESHOLD = 0.05
MASS_TOLERANCE = 0.5
RT_TOLERANCE = 1.0
# Batch F1 on rows with MS2 evidence was 0.709-0.721 (GPST000029) and 0.634 (GPST000017) over seeds 0-2
BATCH_F1_THRESHOLDS = {'GPST000029': 0.65, 'GPST000017': 0.58}


def match_spectra(array1, array2, mass_threshold = MASS_TOLERANCE, rt_threshold = RT_TOLERANCE, array2_alt = None):
    matches = []
    used_predictions = set()
    for i, (mass1, rt1) in enumerate(array1):
        # Find all potential matches based on mass
        mass_diffs = np.abs(array2[:, 0] - mass1)
        potential_matches = np.where(mass_diffs <= mass_threshold)[0]
        if array2_alt is not None:
            mass_diffs_alt = np.abs(array2_alt[:, 0] - mass1)
            potential_alt = np.where(mass_diffs_alt <= mass_threshold)[0]
            potential_matches = list(set(list(potential_matches) + list(potential_alt)))
        potential_matches = [j for j in potential_matches if j not in used_predictions]
        if len(potential_matches) == 0:
            continue
        # If only one match, check retention time
        if len(potential_matches) == 1:
            j = potential_matches[0]
            if abs(rt1 - array2[j, 1]) <= rt_threshold:
                matches.append((i, j))
                used_predictions.add(j)
        else:
            # Multiple matches, find the closest retention time
            rt_diffs = np.abs(array2[potential_matches, 1] - rt1)
            best_match = potential_matches[np.argmin(rt_diffs)]
            if rt_diffs[np.argmin(rt_diffs)] <= rt_threshold:
                matches.append((i, best_match))
                used_predictions.add(best_match)
    return matches


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
  assert len(predictions)>0
  if len(predictions)==0:
    print('empty preds')
    return 0, 0, 0, 0, 0, 0, 0, 0, 0
  predictions['converted_masses'] = [m_z * abs(charge) + (abs(charge) - 1) for m_z, charge in zip(predictions.reset_index()['m/z'], predictions['charge'])]
  pairs = predictions.reset_index()[['m/z', 'RT']].round(2).values
  pairs_converted = predictions[['converted_masses', 'RT']].round(2).values
  gt_pairs = gt.reset_index()[['Mass', rt_col]].round(2).values
  matched_pairs = match_spectra(gt_pairs, pairs, mass_threshold = mass_thresh, rt_threshold = RT_thresh, array2_alt = pairs_converted)
  merge_df = gt[['Mass', rt_col, 'glycan']].reset_index(drop = True)
  new_md = add_pred_column(merge_df,'batch_pred', matched_pairs, predictions.reset_index(), rt_col)
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
  fn = (new_md[new_md['glycan'].notnull()]['similarity_score'].apply(lambda x: 1-x)).sum() + empty_glycan_not_predicted
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

@pytest.mark.parametrize("test_params", test_params)
def test_candycrunch_accuracy(test_params, result_collector, input_format, verbose, test_files = None):
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    start_time = time.time()  # Start timing
    test_outputs = []
    test_dict = test_params['test_dict']
    test_files = test_params['test_dict'].get('test_files',None)
    if test_files is None:
        test_files = [x for x in os.listdir(f"{TEST_DATA_DIR}/{test_dict['name']}")]
    test_files = [x for x in test_files if 'df_mz' not in x if not x.startswith(".")]
    if input_format == "xlsx":
        test_files = [x for x in test_files if not x.endswith(".mzML")]
    elif input_format == "mzml":
        mzml = [x for x in test_files if x.endswith(".mzML")]
        test_files = mzml if mzml else test_files  # fall back to xlsx if no mzML present
    for filename in test_files:
        inference_params = {k: v for k,v in test_params.items() if 'posthoc' not in k if k not in ('test_dict', 'format')}
        posthoc_params = {k: v for k,v in test_params.items() if 'posthoc' in k}
        print(filename)
        print(inference_params|posthoc_params)
        start_time = time.time()
        preds_out = wrap_inference(f"{TEST_DATA_DIR}/{test_dict['name']}/{filename}",**test_dict['args'],
                                **inference_params)
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"\nTest execution time: {execution_time:.2f} seconds")  # Print execution time
        preds_out = posthoc_process_df(preds_out,posthoc_params)
        loaded_gt = pd.read_csv(f"{TEST_DATA_DIR}/{test_dict['name']}/df_mz_{test_dict['name']}.csv")
        col_name  =  filename.split(".")[0]
        rt_col_name = 'RT' if 'RT' in loaded_gt.columns else col_name+'_RT'
        eval_scores = evaluate_predictions(preds_out, loaded_gt[loaded_gt[col_name] > 0], rt_col_name, MASS_TOLERANCE, RT_TOLERANCE, verbose = verbose)
        print('True Positives',eval_scores[-4])
        print('False Positives',eval_scores[-3])
        print('Unevaluable',eval_scores[-1])
        print('False Negatives',eval_scores[-2])
        print('incorrect_preds',eval_scores[-3])
        print('peaks_not_picked',eval_scores[-6])
        test_outputs.append(eval_scores)
        file_format = 'mzML' if filename.endswith('.mzML') else 'xlsx'
        print(f'file_score:{eval_scores[0]} ({file_format})')
        test_params['format'] = file_format
        result_collector.add_result(test_params, eval_scores[0], eval_scores)
        param_key = tuple(
            test_params[key] if key != 'test_dict' else test_params['test_dict']['name']
            for key in test_params
        )
        result_collector.check_performance(test_dict['name'], param_key, eval_scores[0])
    print("Adding results to collector")  # Debug print
    if test_outputs:
        print(f'avg_score:{np.mean([x[0] for x in test_outputs])}')
        assert np.mean([x[0] for x in test_outputs]) > AVG_THRESHOLD


def test_candycrunch_batch(result_collector, verbose):
    if result_collector.param_names is None:
        result_collector.param_names = {k: k for k in list(extra_param_dict.keys()) + ['format']}
    files = {'GPST000029': 'CA_PGMLAD_OG_051017', 'GPST000017': 'JC_141128PGMa'}
    combined, outputs = wrap_inference_batch([f"{TEST_DATA_DIR}/{name}/{label}.mzML" for name, label in files.items()],
                                             'O', intra_cat_thresh = 1.0, spectra = True)
    assert list(outputs) == list(files.values())
    for name, label in files.items():
        df_out, spectra_out = outputs[label]
        gaps = (df_out['evidence'] == 'ms1_only').values
        # Spectra stay aligned with the rows, MS1-only gap rows included
        assert len(spectra_out) == len(df_out)
        assert all(s is None for s, g in zip(spectra_out, gaps) if g)
        # Gap rows share the MS2 rows' abundance scale, so they cannot dominate a file
        assert abs(df_out['rel_abundance'].sum() - 100) < 1e-6
        assert df_out.loc[gaps, 'rel_abundance'].sum() < 50
        gt = pd.read_csv(f"{TEST_DATA_DIR}/{name}/df_mz_{name}.csv")
        eval_scores = evaluate_predictions(df_out[~gaps].copy(), gt[gt[label] > 0],
                                           'RT' if 'RT' in gt.columns else label + '_RT', MASS_TOLERANCE, RT_TOLERANCE,
                                           verbose = verbose)
        print(f'{name} batch F1 (rows with MS2 evidence): {eval_scores[0]:.3f}, MS1-only gap rows: {gaps.sum()}')
        # Report like the per-file tests, so batch scores reach the summary tables, the results log and the regression check
        result_collector.add_result({'test_dict': {'name': f'{name}_batch'}, 'supplement': True, 'experimental': True,
                                     'format': 'mzML'}, eval_scores[0], eval_scores)
        result_collector.check_performance(f'{name}_batch', (f'{name}_batch', True, True, 'mzML'), eval_scores[0])
        assert eval_scores[0] > BATCH_F1_THRESHOLDS[name]
    # Feature table: one row per isomer group with per-file abundances and evidence
    assert {'top1_pred', 'm/z', 'RT', 'n_files_ms2'}.issubset(combined.columns)
    assert all(label in combined.columns and f'evidence_{label}' in combined.columns for label in files.values())
    assert (combined['n_files_ms2'] > 0).all()


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
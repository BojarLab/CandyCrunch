import pytest
import unittest
import os
import pathlib
import sys
from tabulate import tabulate
import numpy as np
from collections import defaultdict

from CandyCrunch.prediction import *
from glycowork.motif.graph import compare_glycans,get_possible_topologies,graph_to_string
from itertools import product
import time

BASE_DIR = pathlib.Path(__file__).parent.parent  # Go up one level from the test file
TEST_DATA_DIR = BASE_DIR / "tests" / "data"
TEST_DICTS = [
    {'name':'milk','args': {'glycan_class':'free'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000350','args': {'glycan_class':'O','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'O.' in x],'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000350','args': {'glycan_class':'N','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000350/") if 'N' in x],'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000017','args': {'glycan_class':'O','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'}, 'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000017/") if 'PGMb' not in x if 'JC' in x],'mass_threshold':0.5, 'RT_threshold':2},
    {'name':'GPST000029','args': {'glycan_class':'O','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'PMC8950484_CHO','args': {'glycan_class':'O','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000307','args': {'glycan_class':'O','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'}, 'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000487','args': {'glycan_class':'N','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000487/")],'mass_threshold':0.5, 'RT_threshold':1},
    {'name':'GPST000134','args': {'glycan_class':'N','taxonomy_level':'Kingdom','taxonomy_filter':'Animalia','mode':'positive'},'test_files':[x for x in os.listdir(f"{TEST_DATA_DIR}/GPST000134/") if 'glycans_1' in x][:1],'mass_threshold':0.5, 'RT_threshold':1}

]

def match_spectra(array1, array2, mass_threshold=0.3, rt_threshold=0.9):
    matches = []
    
    for i, (mass1, rt1) in enumerate(array1):
        # Find all potential matches based on mass
        mass_diffs = np.abs(array2[:, 0] - mass1)
        potential_matches = np.where(mass_diffs <= mass_threshold)[0]
        
        if len(potential_matches) == 0:
            continue
        
        # If only one match, check retention time
        if len(potential_matches) == 1:
            j = potential_matches[0]
            if abs(rt1 - array2[j, 1]) <= rt_threshold:
                matches.append((i, j))
        else:
            # Multiple matches, find the closest retention time
            rt_diffs = np.abs(array2[potential_matches, 1] - rt1)
            best_match = potential_matches[np.argmin(rt_diffs)]
            if rt_diffs[np.argmin(rt_diffs)] <= rt_threshold:
                matches.append((i, best_match))
    
    return matches
    
def add_pred_column(df_in,col_name,matches,pred_df,rt_col):
    df_in[col_name] = None
    for gt_idx,pred_idx in matches:
        df_in.at[gt_idx,col_name] = pred_df.iloc[pred_idx,:]['top1_pred']
    extra_preds = pred_df[~(pred_df.index.isin([x[1] for x in matches]))][['m/z','RT','top1_pred']].rename(columns={'m/z':'Mass','top1_pred':col_name,'RT':rt_col})
    df_in = pd.concat([df_in,extra_preds]).sort_values(['Mass',rt_col])
    return df_in

def evaluate_predictions(predictions,gt,rt_col,mass_thresh,RT_thresh):
    assert len(predictions)>0
    if len(predictions)==0:
        print('empty preds')
        return 0,0,0,0,0,0
    predictions['converted_masses'] = [m_z for m_z,charge in zip(predictions.reset_index()['m/z'],predictions['charge'])]
    pairs = predictions.reset_index()[['m/z','RT']].round(2).values
    gt_pairs = gt.reset_index()[['Mass',rt_col]].round(2).values
    matched_pairs = match_spectra(gt_pairs, pairs,mass_threshold=mass_thresh,rt_threshold=RT_thresh)
    merge_df = gt[['Mass',rt_col,'glycan']].reset_index(drop=True)
    new_md = add_pred_column(merge_df,'batch_pred',matched_pairs,predictions.reset_index(),rt_col)
    # correct_preds = [compare_glycans(x,y) if (isinstance(x,str) and isinstance(y,str)) else False for x,y in zip(new_md['glycan'],new_md['batch_pred'])]
    correct_preds = []
    for gt_glycan, pred_glycan in zip(new_md['glycan'],new_md['batch_pred']):
        if not (isinstance(gt_glycan,str) and isinstance(pred_glycan,str)):
            correct_preds.append(False)
            continue
        if '{' in gt_glycan:
            possible_structures = [graph_to_string(x) for x in get_possible_topologies(gt_glycan)]
            correct_preds.append(any([compare_glycans(p,pred_glycan) for p in possible_structures]))
        else:
            correct_preds.append(compare_glycans(gt_glycan,pred_glycan))
    new_md['correct_preds'] = correct_preds
    tp = len(np.where(new_md['correct_preds'] == True)[0])
    fp = len(np.where((new_md['glycan'].isnull())&(new_md['batch_pred'].notnull()))[0])
    fn = len(np.where((new_md['glycan'].notnull())&(new_md['correct_preds'] == False))[0])
    peaks_not_picked = len(np.where((new_md['glycan'].notnull())&(new_md['batch_pred'].isnull()))[0])
    incorrect_predictions = len(np.where((new_md['glycan'].notnull())&(new_md['batch_pred'].notnull())&(new_md['correct_preds'] == False))[0])
    Precision = (tp / (tp + fp)+1e-4)+1e-4
    Recall = (tp / (tp + fn)+1e-4)+1e-4
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    return F1_score,Precision,Recall,peaks_not_picked,incorrect_predictions,tp,fp,fn

def posthoc_process_df(df_in,posthoc_params):
    for arg in posthoc_params:
        df_arg = arg.split('posthoc_')[1]
        if 'lowerthan' in arg:
            df_arg = df_arg.split('lowerthan_')[1]
            df_in = df_in[(df_in[df_arg]<posthoc_params[arg])] 
        else:
            df_in = df_in[(df_in[df_arg]>posthoc_params[arg])]
    return df_in 

AVG_THRESHOLD = 0.05

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
def test_candycrunch_accuracy(test_params,result_collector,test_files=None):
    if result_collector.param_names is None:
        result_collector.param_names = {k:k for k in extra_param_dict.keys()}
    start_time = time.time()  # Start timing
    test_outputs = []
    test_dict = test_params['test_dict']
    test_files = test_params['test_dict'].get('test_files',None)
    if not test_files:
        test_files = [x for x in os.listdir(f"{TEST_DATA_DIR}/{test_dict['name']}")]
    test_files = [x for x in test_files if 'df_mz' not in x if not x.startswith(".")]
    for filename in test_files:
        inference_params = {k:v for k,v in test_params.items() if 'posthoc' not in k if k!= 'test_dict'}
        posthoc_params = {k:v for k,v in test_params.items() if 'posthoc' in k}
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
        eval_scores = evaluate_predictions(preds_out,loaded_gt[loaded_gt[col_name]>0].dropna(subset='glycan'),rt_col_name,test_dict['mass_threshold'],test_dict['RT_threshold'])
        print('True Positives',eval_scores[-3])
        print('False Positives',eval_scores[-2])
        print('False Negatives',eval_scores[-1])
        print('incorrect_preds',eval_scores[-4])
        print('peaks_not_picked',eval_scores[-5])
        test_outputs.append(eval_scores)
        print(f'file_score:{eval_scores[0]}')
        result_collector.add_result(test_params, eval_scores[0])
        
        param_key = tuple(
            test_params[key] if key != 'test_dict' else test_params['test_dict']['name']
            for key in test_params
        )
        result_collector.check_performance(test_dict['name'], param_key, eval_scores[0])
    print("Adding results to collector")  # Debug print
    print(f'avg_score:{np.mean([x[0] for x in test_outputs])}')
    assert np.mean([x[0] for x in test_outputs])>AVG_THRESHOLD
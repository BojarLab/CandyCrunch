from collections import Counter
import itertools
import ast
import numpy as np
import pandas as pd
from glycowork.motif.processing import enforce_class
from glycowork.motif.tokenization import composition_to_mass,glycan_to_composition
from prediction import average_dicts,mass_check,process_for_inference,get_topk,average_preds,domain_filter,bin_intensities
from prediction import candycrunch,glycans,temperature
import matplotlib.pyplot as plt

def add_new_category(categories,cluster):
    new_id = max([x for x in categories])
    categories[new_id+1] = []
    categories[new_id+1].append(cluster)
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

def create_RT_groups(mass_range_dfs):
    all_sample_RT_groups = []
    for sample_df in mass_range_dfs:
        RT_groups = []
        for RT_group in sample_df.groupby('RT_group').agg(list).RT:
            RT_groups.append(RT_group)
        all_sample_RT_groups.append(RT_groups)
    return all_sample_RT_groups

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

def mass_dfs_to_categories(mass_range_dfs,inter_sample_thresh):
    RT_groups = create_RT_groups(mass_range_dfs)
    categories = initialise_categories(RT_groups)
    categories = expand_RT_categories(RT_groups,categories,inter_sample_thresh)
    sample_cats = RT_cats_to_sample_cats(categories,RT_groups)
    cat_dfs = sample_categories_to_df(sample_cats,mass_range_dfs)
    return cat_dfs

def initialise_categories(all_sample_RT_groups):
    categories = {0:[]}
    for x in all_sample_RT_groups[0]:
        add_new_category(categories,x)
    return categories

def expand_RT_categories(all_sample_RT_groups,categories,inter_sample_thresh):
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
            optim_cats = find_closest_categories(sample,all_candidate_categories,categories)
            for cluster,optim_cat in zip(sample,optim_cats):
                if optim_cat:
                    categories[optim_cat].append(cluster)
                else:
                    add_new_category(categories,cluster)
    return categories


def replace_impossible_combinations(sample,sample_candidates,test_perms):
    perm_distance_sums = []
    few_repeats_perms = []
    lowest_dupes = 99
    for perm in test_perms:
        if [p for p in Counter([cand_cat[0] for cand_cat in perm]).values() if p>1][0] < lowest_dupes:
            lowest_dupes = [p for p in Counter([cand_cat[0] for cand_cat in perm]).values() if p>1][0]
    for perm in test_perms:
        if [p for p in Counter([cand_cat[0] for cand_cat in perm]).values() if p>1][0] == lowest_dupes:
            perm_distance_sums.append(sum([x[1] for x in perm]))
            few_repeats_perms.append(perm)
    duplicate_classes = [x for x in few_repeats_perms[np.argmin(perm_distance_sums)] if x[0] in [k for k,v in Counter([cand_cat[0] for cand_cat in few_repeats_perms[np.argmin(perm_distance_sums)]]).items() if v >1]]
    highest_dupe_idx = np.argmax([x[1] for x in duplicate_classes])
    return [(i,x) if x !=duplicate_classes[highest_dupe_idx] else (i,(set(),0)) for i,x in enumerate(few_repeats_perms[np.argmin(perm_distance_sums)])]

def find_closest_categories(sample,sample_candidates,categories):
    cat_means = get_category_means([sorted(x) for x in sample_candidates],categories)
    cluster_means = [np.mean(x) for x in sample]
    cat_mean_diffs = []
    for cluster_mean,cat_mean in zip(cluster_means,cat_means):
        cat_mean_diffs.append({k:abs(v-cluster_mean) for k,v in cat_mean.items()})
    cat_diff_sums = []
    cat_diff_perms = []
    try:
        for perm in itertools.product(*[list(y.items()) for y in cat_mean_diffs]):
            if not [p for p in Counter([cand_cat[0] for cand_cat in perm]).values() if p>1]:
                cat_diff_sums.append(sum([x[1] for x in perm]))
                cat_diff_perms.append(perm)
        optim_cats = [optim_cat[0] for optim_cat in cat_diff_perms[np.argmin(cat_diff_sums)]]
    except ValueError:
        tester_impossible_optims = replace_impossible_combinations(sample,sample_candidates,list(itertools.product(*[list(y.items()) for y in cat_mean_diffs])))
        return [x[1][0] for x in tester_impossible_optims]
    return optim_cats

def settle_category_conflict(sample,cand_categories,categories):
    for cat in set().union(*[{'A'},{'B','C'},{'A'}]):
        if len([x for x in cand_categories if x == {cat}]) >1:
            closest_cluster_idx = np.argmin([abs(np.mean([p for q in categories[cat] for p in q])-np.mean(j)) for j in sample])
            categories[cat].append(sample[closest_cluster_idx])
            for other_cluster in [x for i,x in enumerate(sample) if i != closest_cluster_idx]:
                categories = add_new_category(categories,other_cluster)
    return categories

def get_category_means(candidate_categories,categories):
    category_means = []
    for cand_cats in candidate_categories:
        category_means.append({cat:np.mean([p for q in categories[cat] for p in q]) for cat in cand_cats})
    return category_means

def load_ms2_spectra(directory,filename_label_map):
    loaded_spectra_list = []
    for filename,(condition_label,replicate_label) in filename_label_map.items():
        loaded_spectra = pd.read_excel(directory+filename)
        split_filename = filename.split('.')[0]
        loaded_spectra = loaded_spectra.assign(condition_label = [condition_label for x in loaded_spectra.reducing_mass])
        loaded_spectra = loaded_spectra.assign(replicate_label = [replicate_label for x in loaded_spectra.reducing_mass])
        loaded_spectra_list.append(loaded_spectra)
    return pd.concat(loaded_spectra_list,ignore_index=True)  

def assign_RT_group(single_mass_df,RT_gap):
    single_mass_df = single_mass_df.assign(RT_group = abs(single_mass_df.RT - single_mass_df.RT.shift(1)) > RT_gap)
    single_mass_df = single_mass_df.assign(RT_group = single_mass_df['RT_group'].cumsum())
    return single_mass_df

# def assign_mass_groups(all_ms2_spectra):
#     all_ms2_spectra = all_ms2_spectra.assign(mass_group = all_ms2_spectra.reducing_mass.round(0))
#     return all_ms2_spectra

def assign_mass_groups(loaded_ms2_spectra,mass_window,min_spectra):
    mass_groups = create_mass_groups(loaded_ms2_spectra,mass_window,min_spectra)
    for x in mass_groups:
        loaded_ms2_spectra.loc[(loaded_ms2_spectra.reducing_mass>x-mass_window)&(loaded_ms2_spectra.reducing_mass<x+mass_window),'mass_label'] = x
    return loaded_ms2_spectra

def assign_dict_labels_list(listy,break_dist=0.5):
  k = 0
  labels = [0]
  for d in range(1, len(listy)):
    if abs(listy[d-1] - listy[d]) < break_dist:
      labels.append(k)
    else:
      k += 1
      labels.append(k)
  return labels

def repeated_discontinuity_split(list_to_split,max_group_size,starting_break_dist,all_groups = []):
    if np.ptp(list_to_split)<max_group_size:
        all_groups.append(list_to_split)
        return None
    dict_labels = assign_dict_labels_list(list_to_split,break_dist=starting_break_dist)
    all_labelled_masses = pd.DataFrame.from_dict(list(zip(dict_labels,list_to_split)))
    all_labelled_masses.columns=['label','mass']
    mass_groups = all_labelled_masses.groupby('label').agg(list)['mass'].tolist()
    for cat in mass_groups:
            repeated_discontinuity_split(cat,max_group_size,starting_break_dist=starting_break_dist*0.95,all_groups = all_groups)
    return all_groups

def create_mass_groups_repeated_split(sorted_masses,max_group_size,starting_break_dist):
    mass_groups = repeated_discontinuity_split(sorted_masses,max_group_size,starting_break_dist,all_groups=[])
    mass_groups_means = [[y]*len(x) for x,y in zip(mass_groups,[np.mean(x) for x in mass_groups])]
    return [x for y in mass_groups_means for x in y]

def generate_basic_human_masses(max_monosaccharides):
    #Could probably replace this with a sugarbase search
    masses_list = []
    for i in range(1,max_monosaccharides+1):
        masses_list.extend([(composition_to_mass(Counter(x))+1,x) for x in itertools.combinations_with_replacement(['Hex','HexNAc','Neu5Ac','dHex','Neu5Gc'],i)])
    return masses_list

def assign_categories(all_ms2_spectra,filename_label_map,search_masses):
    all_mass_dfs = []
    for search_mass in search_masses:
        mass_group_dfs = []
        for condition_label in sorted({x[0] for x in filename_label_map.values()}):
            for replicate_label in [rep for cond,rep in filename_label_map.values() if cond == condition_label]:
                mass_group = all_ms2_spectra[(all_ms2_spectra['mass_label'] == search_mass)&(all_ms2_spectra['condition_label'] == condition_label)&(all_ms2_spectra['replicate_label'] == replicate_label)].copy(deep=True)
                mass_group = mass_group.sort_values('RT')
                mass_group = assign_RT_group(mass_group,0.8)
                mass_group_dfs.append(mass_group)
        cats_mass_dfs = mass_dfs_to_categories(mass_group_dfs,2.5)
        all_mass_dfs.append(cats_mass_dfs)
    return pd.concat([p for q in all_mass_dfs for p in q])

def normalise_spectrum(peak_dict):
    dict_total = sum(peak_dict.values())
    normalised_peak_dict = {k:v for k,v in (pd.Series(peak_dict) / dict_total).items()}
    return normalised_peak_dict  

def group_df_by_category(single_mass_df):
    mean_RT = single_mass_df[['reducing_mass','category_label','RT']].groupby('category_label').mean()
    count_RT = single_mass_df[['reducing_mass','category_label','RT']].groupby('category_label').count()
    sum_RT = single_mass_df[['intensity','category_label']].groupby('category_label').sum()
    grouped_categories = pd.concat([mean_RT,sum_RT,count_RT],axis=1).iloc[:,:-1]
    grouped_categories.columns = ['reducing_mass','RT','intensity','num_spectra']
    if 'mass_label' in single_mass_df.columns:
        grouped_categories['mass_label'] = single_mass_df['mass_label'].tolist()[0]
    return grouped_categories

def average_category_dicts(single_mass_df):
    category_averaged_dicts = {}
    for x in single_mass_df['category_label'].unique():
        current_cat = single_mass_df[single_mass_df['category_label'] == x]
        cat_dicts = [y for y in current_cat.peak_d]
        averaged_dict = average_dicts(cat_dicts)
        category_averaged_dicts[x] = averaged_dict
    return category_averaged_dicts

def group_category_spectra(single_mass_df):
    cat_avg_dicts = average_category_dicts(single_mass_df)
    single_mass_df = group_df_by_category(single_mass_df)
    single_mass_df = single_mass_df.assign(peak_d = [cat_avg_dicts[x] for x in single_mass_df.index])
    return single_mass_df

def generate_valid_predictions(single_mass_df,preds_out,glycan_class,modification='reduced',mode='negative'):
    class_map = {0:'O',1:'N',2:'free'}
    candidates = [[y for y in preds_out[idx][:5] if mass_check(mass, y, modification = modification, mode = mode)] for mass,idx in zip(single_mass_df.reducing_mass,single_mass_df.index)]
    candidates = [[y for y in x if enforce_class(y,class_map[glycan_class])] for x in candidates]
    return candidates

def select_unique_predictions_by_abundance(candidate_predictions):
    final_predictions = []
    for category_num in range(len(candidate_predictions)):
        if candidate_predictions[category_num]:
            final_prediction = candidate_predictions[category_num][0]
        else:
            final_prediction = None
        candidate_predictions = [[y for y in x if y!=final_prediction] for x in candidate_predictions]
        final_predictions.append(final_prediction)
    return final_predictions

def grouped_df_to_preds(grouped_df,glycan_class):
    inf_loader, inf_df_out = process_for_inference(grouped_df.reducing_mass,grouped_df.peak_d,grouped_df.RT,grouped_df.num_spectra,glycan_class=glycan_class)
    preds, pred_conf = get_topk(inf_loader, candycrunch, glycans, temp = True, temperature = temperature)
    preds, pred_conf = average_preds(preds, pred_conf)
    return preds,pred_conf

def filter_grouped_df_predictions(grouped_df,preds,glycan_class):
    filtered_preds_df = grouped_df.copy(deep=True)
    filtered_preds_df = filtered_preds_df.reset_index()
    filtered_preds_df = filtered_preds_df[(filtered_preds_df.RT>5)&(filtered_preds_df.RT<40)]
    filtered_preds_df = filtered_preds_df.sort_values('num_spectra',ascending=False)
    valid_predictions = generate_valid_predictions(filtered_preds_df,preds,glycan_class)
    unique_predictions = select_unique_predictions_by_abundance(valid_predictions)
    filtered_preds_df['final_predictions'] = unique_predictions
    filtered_preds_df = filtered_preds_df[~filtered_preds_df.final_predictions.isnull()]
    return filtered_preds_df

def categories_df_to_unique_predictions(df_out_categories,glycan_class,search_masses):
    all_filtered_dfs = []
    for search_mass in search_masses:
        single_mass_df = df_out_categories[df_out_categories.mass_label == search_mass].copy(deep=True)
        grouped_df = group_category_spectra(single_mass_df)
        p1,c1 = grouped_df_to_preds(grouped_df,glycan_class)
        filtered_grouped_df = filter_grouped_df_predictions(grouped_df,p1,glycan_class)
        all_filtered_dfs.append(filtered_grouped_df)
    return pd.concat(all_filtered_dfs,ignore_index=True)

def concat_to_annotation_table(concat_df,label_column,intensity_column):
    annotation_dfs = []
    for category in concat_df[label_column].unique().tolist():
        single_category_df = concat_df[concat_df[label_column] == category][['reducing_mass','RT',intensity_column,'replicate_label','condition_label']].groupby(['replicate_label','condition_label']).sum()[[intensity_column]]
        single_category_df = single_category_df.rename(columns = {intensity_column:category})
        annotation_dfs.append(single_category_df)
    ant_table = pd.concat(annotation_dfs,axis=1).reset_index()
    ant_table['cols'] = [f'condition{x}_rep{y}' for x,y in zip(ant_table.condition_label,ant_table.replicate_label)]
    ant_table = ant_table[[x for x in ant_table.columns if 'label' not in x]]
    ant_table = ant_table.set_index('cols').T
    ant_table.index.name = 'Glycan'
    return ant_table.reset_index()

def map_predictions_to_spectra(df_out,df_out_unique_preds):
    label_to_pred_map = {(ml,cl):fp for ml,cl,fp in zip(df_out_unique_preds.mass_label,df_out_unique_preds.category_label,df_out_unique_preds.final_predictions)}
    df_out['final_pred'] = [label_to_pred_map[(ml,cl)] if (ml,cl) in label_to_pred_map else None for ml,cl in zip(df_out.mass_label,df_out.category_label)]
    return df_out

def wrap_inference_multi_file(spectra_directory,filename_label_map,glycan_class,max_group_size,starting_break_dist,top_n_masses=None):
    loaded_ms2_spectra = load_ms2_spectra(spectra_directory,filename_label_map)
    loaded_ms2_spectra = loaded_ms2_spectra.sort_values('reducing_mass')
    loaded_ms2_spectra['mass_label'] = create_mass_groups_repeated_split(loaded_ms2_spectra.reducing_mass,max_group_size,starting_break_dist)
    sorted_mass_labels = loaded_ms2_spectra.groupby('mass_label').count().sort_values('peak_d',ascending=False).index.tolist()
    if top_n_masses:    
        sorted_mass_labels = sorted_mass_labels[:top_n_masses]
    df_out = assign_categories(loaded_ms2_spectra,filename_label_map,sorted_mass_labels)
    df_out = df_out.assign(peak_d = [ast.literal_eval(x) for x in df_out.peak_d])
    df_out_unique = categories_df_to_unique_predictions(df_out,glycan_class,sorted_mass_labels)
    df_out = map_predictions_to_spectra(df_out,df_out_unique)
    return df_out,df_out_unique

step = (3000 - 39.714) / (2048 - 1)
frames = np.array([39.714 + step * i for i in range(2048)])
def get_top_frags(peak_d):
    actual_top_peaks = frames[np.argsort(bin_intensities(peak_d,frames)[0])[-10:][::-1]]+step/2
    return actual_top_peaks

def harmonise_df_out_unique(df_out_unique):
    df_out_harm = df_out_unique.copy(deep=True)
    df_out_harm['composition'] = [glycan_to_composition(g) for g in df_out_harm.final_predictions]
    df_out_harm['charge'] = [round(composition_to_mass(df_out_harm.composition.values.tolist()[k])/df_out_harm.reducing_mass.values.tolist()[k])*-1 for k in range(len(df_out_harm))]
    df_out_harm['predictions'] = [[(g,0.99)] for g in df_out_harm.final_predictions]
    df_out_harm['top_fragments'] = [get_top_frags(k) for k in df_out_harm.peak_d]
    df_out_harm = df_out_harm[['predictions','final_predictions','composition','num_spectra',	'charge','RT','peak_d','top_fragments','reducing_mass','category_label','mass_label']]
    return df_out_harm

def run_domain_filter(df_in_unique):
    df_out_unique = df_in_unique.copy(deep=True)
    df_out_unique = df_out_unique.set_index('reducing_mass')
    df_out_unique = domain_filter(df_out_unique,'O')
    df_out_unique['string_predictions'] = [str(x) for x in df_out_unique.predictions]
    df_out_unique = df_out_unique[df_out_unique.string_predictions.str.len()>3]
    return df_out_unique                           

def get_glycan_plot_df(annotation_table,species_name):
    plot_df = annotation_table[annotation_table.Glycan == species_name].T.reset_index()
    plot_df = plot_df.iloc[3:]
    plot_df['condition_label'] = [int(x.split('_')[0].split('condition')[1]) for x in plot_df['cols']]
    plot_df['replicate_label'] = [int(x.split('_')[1].split('rep')[1]) for x in plot_df['cols']]
    plot_df = plot_df.rename(columns={plot_df.columns[1]: "intensity" })
    plot_df = plot_df[['intensity','condition_label','replicate_label']]
    plot_df = plot_df.sort_values('condition_label')
    plot_df = plot_df.dropna()
    return plot_df

def plot_glycan_abundance(plot_df):
    norm_df = plot_df.groupby('condition_label').mean()[['intensity']]
    norm_df = norm_df.reset_index()
    norm_df['condition_label'] = norm_df['condition_label'].astype(int)
    norm_df = norm_df.sort_values('condition_label')
    fig, ax = plt.subplots()
    ax.axvspan(0, 12, alpha=0.3, color='navy')
    ax.axvspan(24, 36, alpha=0.3, color='navy')
    sem_df = plot_df.groupby('condition_label').std()[['intensity']]/np.sqrt(plot_df.groupby('condition_label').count()[['intensity']])
    sem_df = sem_df.reset_index()
    sem_df['condition_label'] = sem_df['condition_label'].astype(int)
    sem_df = sem_df.sort_values('condition_label')
    ax.errorbar(norm_df.condition_label,norm_df.intensity, yerr=sem_df.intensity/2,solid_capstyle='projecting',capsize=3,color = 'purple')
    ax.grid(alpha=0.5, linestyle=':')
    plt.show()
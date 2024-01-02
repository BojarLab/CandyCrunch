import numpy as np
from collections import Counter
import itertools

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

def assign_mass_groups(all_ms2_spectra):
    all_ms2_spectra = all_ms2_spectra.assign(mass_group = all_ms2_spectra.reducing_mass.round(0))
    return all_ms2_spectra

def assign_categories(all_ms2_spectra,filename_label_map,masses = None):
    all_mass_dfs = []
    if not masses:
        search_masses = all_ms2_spectra.reducing_mass.round(0).unique()
    else:
        search_masses = masses
    for search_mass in search_masses:
        mass_group_dfs = []
        for condition_label in sorted({x[0] for x in filename_label_map.values()}):
            for replicate_label in [rep for cond,rep in filename_label_map.values() if cond == condition_label]:
                mass_group = all_ms2_spectra[(all_ms2_spectra['reducing_mass'].astype(float).round(0) == search_mass)&(all_ms2_spectra['condition_label'] == condition_label)&(all_ms2_spectra['replicate_label'] == replicate_label)].copy(deep=True)
                mass_group = mass_group.sort_values('RT')
                mass_group = assign_RT_group(mass_group,0.8)
                mass_group_dfs.append(mass_group)
        cats_mass_dfs = mass_dfs_to_categories(mass_group_dfs,2.5)
        all_mass_dfs.append(cats_mass_dfs)
    return pd.concat([p for q in all_mass_dfs for p in q])  
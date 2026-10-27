"""Differential glycomics on CandyCrunch results: glycowork's differential expression (two groups of runs) and glycan ANOVA (three or more groups) on
the feature table of wrap_inference_batch"""
import contextlib
import io
import pandas as pd


def _row_label(structure, composition):
    from glycowork.glycan_data.loader import stringify_dict
    from glycowork.motif.processing import canonicalize_composition
    if isinstance(structure, str) and structure:
        return structure
    if not isinstance(composition, dict) or not composition:
        return None
    # sialic acids of known linkage (from a linkage-specific derivatization) as the E and L of shorthand compositions, as the app writes them
    return canonicalize_composition(composition, as_string = True) if any('_a2-' in k for k in composition) else stringify_dict(composition)


def abundance_table(features, runs, motifs = False):
    """the feature table in glycowork's layout: glycans in the first column, then one column of abundances per run\n
   | Arguments:
   | :-
   | features (dataframe): feature table of wrap_inference_batch (top1_pred, composition, one abundance column per run); rows whose 'excluded' column is True (curation in the app) are left out
   | runs (list): runs (columns of features) to take, in this order
   | motifs (bool): only features with a structure, as motifs are found in structures; default:False\n
   | Returns:
   | :-
   | Returns the table (per row a structure, else the composition; features of the same one summed, as they are the same molecule as far as CandyCrunch can tell, where glycowork would average them; structures first, rows without signal dropped) and the number of features in each row
   """
    labels = pd.Series([_row_label(s, None if motifs else c) for s, c in zip(features['top1_pred'], features['composition'])], index = features.index)
    keep = labels.notna() & ~features.get('excluded', pd.Series(False, index = features.index)).astype(bool)
    values = features.loc[keep, runs].astype(float).fillna(0).groupby(labels[keep].values, sort = False)
    table, counts = values.sum(), values.size()
    table = table[(table > 0).any(axis = 1)]
    # Structures first: glycowork picks its default transform by the first row's glycan class
    order = sorted(table.index, key = lambda g: ('(' not in g, -table.loc[g].mean()))
    return table.loc[order].rename_axis('glycan').reset_index(), counts.loc[order]


def compare_groups(features, groups, pairs = None, motifs = False, transform = None, impute = True, min_samples = 0.1):
    """differential abundance between groups of runs of one batch: glycowork's get_differential_expression for two groups (moderated Welch's t-test,
    or paired, on log-ratio transformed abundances, with Benjamini-Hochberg correction at a sample-size adjusted alpha) and get_glycanova for three or
    more (moderated one-way ANOVA with Tukey's post-hoc test); glycowork tests a glycan or motif not found in any run of a group on its detection
    (Fisher's exact test) instead\n
   | Arguments:
   | :-
   | features (dataframe): feature table of wrap_inference_batch (see abundance_table)
   | groups (dict): group of each run, {run: group}; runs without a group (None or '') are left out; the first group named is the reference, so Log2FC is the second group over the first
   | pairs (dict): two groups only: the subject of each run, {run: subject}, for a paired test of runs of the same subject; default:None
   | motifs (bool): compare motifs (glycowork's known and exhaustive feature sets, counted on the features with a structure) instead of glycans; default:False
   | transform (string): 'CLR' or 'ALR'; default:None (glycowork chooses)
   | impute (bool): replace zeros by glycowork's biosynthetic imputation; default:True
   | min_samples (float): share of runs a glycan has to be found in to be tested; default:0.1\n
   | Returns:
   | :-
   | Returns a dictionary: groups (in order), members ({group: runs}), paired, motifs, table (glycowork's results, plus 'Mean in <group>' columns of measured abundance, 'Found in' (runs per group) and, for glycans, the number of features in each row), posthoc ({glycan: Tukey's table}, three or more groups), values (measured abundance in % per row and run, before imputation), input (the abundance table of the runs compared), test, alpha, transform, and log (what glycowork printed)
   """
    from glycowork.motif.analysis import get_differential_expression, get_glycanova
    from glycowork.motif.annotate import quantify_motifs
    from glycowork.glycan_data import stats
    # glycowork before it had the detection test let a glycan missing from a whole group shift every other one, which made most glycans of split
    # replicates significant; such results are not given
    if not hasattr(stats, 'detection_test'):
        raise ImportError('Comparing groups needs the current glycowork (from its dev branch), whose tests handle glycans missing from a whole group')
    names = list(dict.fromkeys(g for g in groups.values() if g))
    members = {g: [r for r, x in groups.items() if x == g] for g in names}
    if len(names) < 2:
        raise ValueError('Give the runs at least two groups to compare')
    small = [g for g in names if len(members[g]) < 2]
    if small:
        raise ValueError(f'Every group needs at least two runs, to estimate its variance: {", ".join(small)} has one')
    pairs = {r: p for r, p in (pairs or {}).items() if p and groups.get(r)}
    if pairs and len(names) > 2:
        raise ValueError('Pairs only work for two groups (a paired t-test); clear them to compare three or more groups')
    if pairs:
        a, b = ({pairs.get(r): r for r in members[g]} for g in names)
        if None in a or None in b or len(a) != len(members[names[0]]) or len(b) != len(members[names[1]]) or set(a) != set(b):
            raise ValueError('For a paired comparison, every run needs a pair, and every pair one run in each group')
        members[names[1]] = [b[p] for p in a]
    runs = [r for g in names for r in members[g]]
    table, counts = abundance_table(features, runs, motifs = motifs)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        if len(names) == 2:
            results = get_differential_expression(table, group1 = members[names[0]], group2 = members[names[1]], motifs = motifs, paired = bool(pairs),
                                                  impute = impute, min_samples = min_samples, transform = transform, random_state = 0)
            posthoc = {}
        else:
            results, posthoc = get_glycanova(table, groups = [groups[r] for r in runs], motifs = motifs, impute = impute, min_samples = min_samples,
                                             transform = transform, random_state = 0)
        values = table.set_index('glycan')
        values = values / values.sum() * 100
        if motifs:
            values = quantify_motifs(values, feature_set = ['exhaustive', 'known'])
            values = values / values.sum() * 100
    attrs = dict(results.attrs)
    table_out = pd.DataFrame(results).reset_index(drop = True)
    table_out.attrs = {}
    position = table_out.columns.get_loc('Mean abundance') + 1 if 'Mean abundance' in table_out.columns else 1
    for k, g in enumerate(names):
        table_out.insert(position + k, f'Mean in {g}', values.reindex(table_out['Glycan'])[members[g]].mean(axis = 1).values)
    present = (values.reindex(table_out['Glycan']).fillna(0) > 0)
    table_out['Found in'] = [', '.join(f'{g} {present.iloc[i][members[g]].sum()}/{len(members[g])}' for g in names) for i in range(len(table_out))]
    if not motifs:
        table_out['Features'] = counts.reindex(table_out['Glycan']).values
    return {'groups': names, 'members': members, 'paired': bool(pairs), 'motifs': motifs, 'table': table_out,
            'posthoc': {k: pd.DataFrame(v).reset_index(drop = True) for k, v in posthoc.items()}, 'values': values.reindex(table_out['Glycan']),
            'input': table, 'test': attrs.get('test', ''), 'alpha': float(attrs.get('alpha', 0.05)), 'transform': attrs.get('transform', transform or ''),
            'log': log.getvalue()}


def save_volcano(result, path):
    """saves glycowork's volcano plot of a two-group comparison, with SNFG drawings of the significant glycans or motifs\n
   | Arguments:
   | :-
   | result (dict): output of compare_groups for two groups
   | path (string): .svg, .pdf or .png file to write
   """
    import matplotlib
    matplotlib.use('Agg')
    from glycowork.motif.analysis import get_volcano
    get_volcano(result['table'], y_thresh = result['alpha'], filepath = path, title = f'{result["groups"][1]} vs {result["groups"][0]}')

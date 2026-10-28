"""Calibrated confidence of the structures CandyCrunch reports: the probability that a row's structure is the glycan of its peak, as written, from what
the pipeline knows about the row (the model's probability among isomers, fragment evidence, diagnostic peaks, isomer peaks, abundance, ...)"""
import re
import numpy as np
import pandas as pd

# Logistic model over the terms below, fitted (C = 0.3) on the rows the pipeline reported for the development test sets, labeled against their curated
# ground truths: 1 for the exact structure, 0 for another structure or a peak without a ground-truth glycan (rows of ground-truth peaks without a structure
# left out). Leave-one-set-out, it is calibrated within 0.03 (expected calibration error; the model's own probability of the structure: 0.18).
# Per term: coefficient and mean over the fitted rows, so that a row's contribution, coefficient x (value - mean), is how far it moves the log-odds from a
# typical reported peak
CONFIDENCE_INTERCEPT = 0.471
CONFIDENCE_TERMS = {'logit_conf': (0.065, 1.305), 'no_conf': (0.222, 0.02), 'margin': (0.947, 0.573), 'log_score': (0.122, 2.554), 'z2': (0.22, 0.651),
                    'database': (-0.278, 0.211), 'sup_frac': (0.672, -0.166), 'against': (-0.215, 0.405), 'note_isomer': (-0.93, 0.308),
                    'note_insource': (-0.811, 0.015), 'note_same': (-0.471, 0.129), 'no_glytoucan': (-0.803, 0.152), 'log_size': (-0.24, 2.058),
                    'ab_pct': (1.102, 0.027)}
# How the terms are shown: one line per group, which sums the contributions of its terms
CONFIDENCE_GROUPS = [('CandyCrunch model', ['logit_conf', 'no_conf', 'margin']), ('Structure from', ['database']), ('Fragments', ['log_score']),
                     ('Diagnostic peaks', ['sup_frac', 'against']), ('Isomer peaks', ['note_isomer', 'note_same']), ('In-source fragment', ['note_insource']),
                     ('Known glycan', ['no_glytoucan']), ('Size and charge', ['log_size', 'z2']), ('Abundance', ['ab_pct'])]
SOURCES = {'strong': 'the model', 'medium': "a biosynthetic intermediate of the run's other glycans", 'weak': 'a database glycan of this mass',
           'composition': 'a database structure of a composition that fragments confirm'}


def confidence_features(df):
    """the terms of the confidence model for each row\n
   | Arguments:
   | :-
   | df (dataframe): rows as finalise_predictions builds them (top1_pred, predictions, annotation_score, charge, evidence, supported_by, notes, GlyTouCan_ID,
   |                 composition, and rel_abundance if there are abundances), all rows of one run\n
   | Returns:
   | :-
   | Returns a dataframe with one column per term of CONFIDENCE_TERMS, plus conf and second (the model's probability of the structure and of the likeliest other one)
   """
    probs = [[x for x in p if len(x) > 1] if isinstance(p, (list, tuple)) else [] for p in df['predictions']]
    top1 = df['top1_pred'].values
    conf = np.array([next((x[1] for x in p if x[0] == t), np.nan) for p, t in zip(probs, top1)], dtype = float)
    second = np.array([max([x[1] for x in p if x[0] != t] or [0]) for p, t in zip(probs, top1)], dtype = float)
    clipped = np.clip(np.nan_to_num(conf, nan = 0.5), 0.01, 0.99)
    support = [re.match(r'(\d+)/(\d+)', s) if isinstance(s, str) else None for s in df['supported_by']] if 'supported_by' in df.columns else [None] * len(df)
    against = [s.count('against ') if isinstance(s, str) else 0 for s in df['supported_by']] if 'supported_by' in df.columns else [0] * len(df)
    notes = df['notes'].fillna('').astype(str).values if 'notes' in df.columns else np.full(len(df), '')
    ids = df['GlyTouCan_ID'].fillna('').astype(str).values if 'GlyTouCan_ID' in df.columns else np.full(len(df), '')
    # the row's abundance as its percentile among the run's rows with a structure (scale-free: XIC areas and precursor intensities alike)
    named = np.array([isinstance(t, str) for t in top1], dtype = bool)
    abundance = pd.Series(df['rel_abundance'].values.astype(float) if 'rel_abundance' in df.columns else np.full(len(df), np.nan)).where(named)
    return pd.DataFrame({'logit_conf': np.where(np.isnan(conf), 0, np.log(clipped / (1 - clipped))), 'no_conf': np.isnan(conf).astype(float),
                         'margin': np.clip(np.nan_to_num(conf) - second, -1, 1),
                         'log_score': np.log1p(df['annotation_score'].astype(float).fillna(0).values) if 'annotation_score' in df.columns else 0.0,
                         'z2': (np.abs(df['charge'].values.astype(float)) >= 2).astype(float),
                         'database': (df['evidence'].values != 'strong').astype(float) if 'evidence' in df.columns else 1.0,
                         'sup_frac': [int(m[1]) / int(m[2]) - 0.5 if m and int(m[2]) else 0 for m in support], 'against': np.minimum(against, 2),
                         'note_isomer': [float(bool(re.search('likely other|likely another isomer', n))) for n in notes],
                         'note_insource': [float('in-source fragment' in n) for n in notes],
                         'note_same': [float('Same structure as the peaks' in n) for n in notes], 'no_glytoucan': (ids == '').astype(float),
                         'log_size': [np.log(max(sum(c.values()), 1)) if isinstance(c, dict) else 0 for c in df['composition']],
                         'ab_pct': abundance.rank(pct = True).fillna(0.5).values - 0.5, 'conf': conf, 'second': second}, index = df.index)


def calibrated_confidence(df):
    """the calibrated confidence of each row's structure, and how it is built\n
   | Arguments:
   | :-
   | df (dataframe): rows as finalise_predictions builds them, all rows of one run (see confidence_features)\n
   | Returns:
   | :-
   | Returns a tuple of (array of confidences, NaN for rows without a structure; list per row of (term, its value in words, log-odds contribution), the
   | first one the baseline of a typical reported peak, empty for rows without a structure)
   """
    X = confidence_features(df)
    baseline = CONFIDENCE_INTERCEPT + sum(c * m for c, m in CONFIDENCE_TERMS.values())
    contributions = pd.DataFrame({k: c * (X[k].values - m) for k, (c, m) in CONFIDENCE_TERMS.items()}, index = df.index)
    logit = baseline + contributions.sum(axis = 1).values
    named = np.array([isinstance(t, str) for t in df['top1_pred']], dtype = bool)
    confidence = np.where(named, np.round(1 / (1 + np.exp(-logit)), 3), np.nan)
    evidence = df['evidence'].values if 'evidence' in df.columns else np.full(len(df), None)
    score = df['annotation_score'].astype(float).fillna(0).values if 'annotation_score' in df.columns else np.zeros(len(df))
    support = df['supported_by'].values if 'supported_by' in df.columns else np.full(len(df), None)
    notes = df['notes'].fillna('').astype(str).values if 'notes' in df.columns else np.full(len(df), '')
    groups = {label: contributions[keys].sum(axis = 1).values for label, keys in CONFIDENCE_GROUPS}
    abundance = pd.Series(df['rel_abundance'].values.astype(float) if 'rel_abundance' in df.columns else np.full(len(df), np.nan)).where(named)
    ranks = abundance.rank(ascending = False, method = 'min').values
    terms = []
    for i in range(len(df)):
        if not named[i]:
            terms.append([])
            continue
        # (supported_by: the peaks that place each residue, see supporting_ions)
        x, s = X.iloc[i], support[i] if isinstance(support[i], str) else ''
        isomer = next((n.strip() for n in notes[i].split(';') if 'likely other' in n or 'likely another isomer' in n), '')
        values = {'CandyCrunch model': (f"{x['conf']:.2f} for this structure" if x['no_conf'] == 0 else 'no probability for this structure') + (
                      f", {x['second']:.2f} for the likeliest other isomer" if x['second'] > 0 else ''),
                  'Structure from': SOURCES.get(evidence[i], 'not the model'), 'Fragments': f'annotation score {score[i]:.0f}',
                  'Diagnostic peaks': s or 'no residue with another possible placement',
                  'Isomer peaks': isomer or 'no other peak with this structure',
                  'In-source fragment': 'possibly, of a co-eluting glycan' if x['note_insource'] else 'no',
                  'Known glycan': 'in GlyTouCan' if x['no_glytoucan'] == 0 else 'not in GlyTouCan',
                  'Size and charge': f"{np.exp(x['log_size']):.0f} residues, z = {abs(int(df['charge'].iat[i]))}",
                  'Abundance': f"{ranks[i]:.0f}. most abundant of {abundance.notna().sum()} peaks" if not np.isnan(ranks[i]) else 'no abundances'}
        terms.append([('Typical reported peak', f'confidence {1 / (1 + np.exp(-baseline)):.2f}', round(baseline, 3))] + [(label, values[label], round(float(groups[label][i]), 3)) for label, _ in CONFIDENCE_GROUPS])
    return confidence, terms

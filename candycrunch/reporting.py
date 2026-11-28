"""Reports of a batch for the people whose samples it measured: wrap_inference_batch's feature table as one workbook with the glycan profile of every sample,
its glycowork motifs, sample checks against an internal standard, and every peak with what it is"""
import re
import numpy as np
import pandas as pd

# Notes of the peaks outside the glycan profile, by the flags of wrap_inference_batch (FLAG_NOTES); 'other class' names the glycan's class
PEAK_NOTES = {'in-source fragment': 'In-source fragment of a co-eluting glycan', 'peeling product': 'Peeling product of a larger O-glycan',
              'glucose oligomer': 'Glucose oligomer (hexose ladder from glycogen, starch, or dextrins)', 'other class': "not of the runs' glycan class"}
# A glycan by its class (glycowork's get_class), for the note of a glycan of another class than the runs'
CLASS_NAMES = {'N': 'N-glycan', 'O': 'O-glycan', 'free': 'Free oligosaccharide', 'lipid': 'Glycolipid glycan', 'lipid/free': 'Free or glycolipid glycan',
               'repeat': 'Repeating-unit glycan'}
# Mass analyzers by the spectra loaders' names of them (detected_ms1_trap, detected_trap)
ANALYZER_NAMES = {'linear': 'linear ion trap', 'amazon': '3D ion trap', 'orbitrap': 'Orbitrap', 'tof': 'TOF'}


def read_standard(text):
    """an internal standard as the CLI and the app take it: a composition (H5N3A2, Hex5HexNAc3Neu5Ac2, or as glycowork reads it), optionally followed by its
    charge and its RT in minutes, separated by commas (H5N3A2,-2,17.1)\n
   | Arguments:
   | :-
   | text (string): the standard as typed\n
   | Returns:
   | :-
   | Returns a dict of 'composition', 'charge' (None: from a feature of it), and 'rt' (None: likewise), as glycan_report's standards take it; raises
   | ValueError where the text reads as none
   """
    from glycowork.motif.processing import canonicalize_composition
    from glycowork.motif.tokenization import composition_to_mass
    parts = [p.strip() for p in text.split(',')]
    try:
        composition = canonicalize_composition(parts[0])
        composition_to_mass(composition)
        return {'composition': composition, 'charge': int(parts[1]) if len(parts) > 1 and parts[1] else None, 'rt': float(parts[2]) if len(parts) > 2 and parts[2] else None}
    except (ValueError, KeyError):
        raise ValueError(f'{text}: expected a composition, optionally with its charge and RT in minutes (e.g., H5N3A2 or H5N3A2,-2,17.1)') from None


def glycan_report(features, runs, filepath = None, spectra_filepaths = None, standards = None, sample_names = None, modification = 'reduced',
                  sample_prep = 'underivatized', title = 'Glycan profiles', settings = None, drawings = True, linked_slope = 0.3, spectra = True):
    """the report of a batch: wrap_inference_batch's features as the glycan profile of every sample (structures, or compositions, summed over their peaks
    at different retention times, in % of the sample's glycan signal), glycowork's motifs per sample, sample checks, and every peak with a note on what is not in the
    profile; with an internal standard, each sample's glycan signal relative to it, and the standard (and any ion that comes with it) out of the profile; as
    each glycan's evidence, the diagnostic ions of the spectrum behind its best confidence and, with the batch's spectra, that spectrum annotated\n
   | Arguments:
   | :-
   | features (dataframe): the feature table of wrap_inference_batch; rows with a flag (in-source fragment, peeling product, glucose oligomer, other
   |                       class) or excluded (curation) stay out of the profile and are noted in the peaks
   | runs (dict): the per-file tables of wrap_inference_batch (its second output), by run; with spectra = True there, each glycan's spectrum is annotated
   | filepath (string): .xlsx file to write the report to; default:None (only the tables are returned)
   | spectra_filepaths (list): the runs' spectra files (as given to wrap_inference_batch), needed for standards; default:None
   | standards (list): internal standards spiked into every sample, each a composition (dict, or a string as H5N3A2 or Hex5HexNAc3Neu5Ac2) or a dict
   |                   {'composition': ..., 'charge': ..., 'rt': ..., 'name': ...} (charge and RT are taken from a feature of the composition where
   |                   missing, else the charge is the runs' first and the RT the standard's highest MS1 point); integrated in every run, the first is the
   |                   reference of the glycan amounts, and a feature that holds a constant ratio to it while the samples' glycan amounts differ (log slope
   |                   below linked_slope over at least 6 runs spanning a 5-fold range) is taken as part of the standard; default:None
   | sample_names (dict): names of the samples by run; default:None (the run names, in natural order)
   | modification (string or list): reducing end of the runs, as given to wrap_inference_batch, for formulas and theoretical m/z; default:'reduced'
   | sample_prep (string): derivatization of the runs, as given to wrap_inference_batch; default:'underivatized'
   | title (string): title of the report; default:'Glycan profiles'
   | settings (dict): settings of the batch to list in the report (e.g., glycan class, LC, mass analyzer); default:None
   | drawings (bool): whether the glycans sheet shows an SNFG drawing of each glycan; default:True
   | linked_slope (float): slope of a feature's log ratio to the standard against the log glycan amount below which it is part of the standard; default:0.3
   | spectra (bool): whether the workbook has a sheet of each glycan's annotated MS2 spectrum, where runs holds the spectra; default:True\n
   | Returns:
   | :-
   | Returns a dict of dataframes: 'glycans' (one row per glycan by mass, isomers by retention time, then one column per sample), 'motifs', 'samples',
   | 'peaks' (in the same order), and 'spectra' (per glycan with a spectrum, in the order of glycans: the sample and peak of its evidence, and what annotating
   | its spectrum takes)
   """
    from glycowork.motif.processing import canonicalize_composition, get_class
    from glycowork.motif.tokenization import composition_to_mass, get_formula, get_ion_mzs
    from glycowork.motif.annotate import quantify_motifs
    from candycrunch.differential import abundance_table, _row_label
    from candycrunch import __version__
    from candycrunch.utils import format_composition, run_labels
    # (with spectra = True the batch gives each run as its table and its spectra, the peaks behind each row)
    peak_lists = {r: t[1] for r, t in runs.items() if isinstance(t, tuple)}
    runs = {r: t[0] if isinstance(t, tuple) else t for r, t in runs.items()}
    # Runs in natural order (K2 before K10)
    run_cols = sorted([r for r in runs if r in features.columns], key = lambda r: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(r))])
    names = {r: (sample_names or {}).get(r, r) for r in run_cols}
    samples = [names[r] for r in run_cols]
    end = modification[0] if isinstance(modification, (list, tuple)) else modification
    end = None if end in ('custom', 'unreduced') else end
    # (the (de)protonated ion of charge z: glycowork's ions of a mass come by charge, from 1 to z)
    ion_mz = lambda c, z: list(get_ion_mzs(composition_to_mass(c, sample_prep = sample_prep, modification = end), max_charge = z).values())[abs(z) - 1]
    formula = lambda c: get_formula(c, sample_prep = sample_prep, modification = end)
    mass = lambda c: composition_to_mass(c, sample_prep = sample_prep, modification = end) if isinstance(c, dict) and c else np.nan
    flags = features['flag'].fillna('').astype(str) if 'flag' in features.columns else pd.Series('', index = features.index)
    excluded = features['excluded'].astype(bool) if 'excluded' in features.columns else pd.Series(False, index = features.index)
    note = pd.Series(['; '.join(f"{CLASS_NAMES.get(get_class(s) if isinstance(s, str) else '', 'Glycan')}, {PEAK_NOTES[f]}" if f == 'other class' else PEAK_NOTES.get(f, f)
                          for f in fl.split('; ') if f) for fl, s in zip(flags, features['top1_pred'])], index = features.index)
    note[excluded] = 'Excluded in curation'
    # Standards: their compositions, ions, and RTs, and the features that are them
    specs = []
    for s in standards or []:
        s = s if isinstance(s, dict) and 'composition' in s else {'composition': s}
        comp = canonicalize_composition(s['composition']) if isinstance(s['composition'], str) else dict(s['composition'])
        hits = features[[c == comp for c in features['composition']]]
        best = hits.loc[hits['n_files_ms2'].idxmax()] if len(hits) else None
        charge = int(s.get('charge') or (best['charge'] if best is not None else features['charge'].iloc[0]))
        specs.append({'composition': comp, 'charge': charge, 'rt': s.get('rt') or (float(best['RT']) if best is not None else None), 'mz': ion_mz(comp, charge),
                      'name': s.get('name') or format_composition(comp, 'letters')})
        note[[c == comp for c in features['composition']]] = f"Internal standard {specs[-1]['name']}"
    # Per run: the standards' integrated MS1 signal and apex, the run's integrated glycan signal (its (de)protonated rows integrated again at their
    # theoretical m/z and own RT), and the MS1 apex of every candidate glycan of the profile near its RT
    checks, analyzers = {r: {} for r in run_cols}, {}
    in_profile = (note == '') & ~excluded
    candidates = features.index[in_profile & np.array([isinstance(c, dict) and bool(c) for c in features['composition']])]
    cand_mz = [ion_mz(features.loc[i, 'composition'], int(features.loc[i, 'charge'])) for i in candidates]
    measured, genuine = pd.DataFrame(np.nan, index = candidates, columns = run_cols), pd.DataFrame(False, index = candidates, columns = run_cols)
    if specs and spectra_filepaths:
        from candycrunch.prediction import load_spectra_filepath, extract_xic_areas
        by_run = dict(zip(run_labels(spectra_filepaths), spectra_filepaths))
        for r in run_cols:
            df = load_spectra_filepath(by_run[r], extract_ms1 = True)
            ms1, trap = df.attrs['ms1'][:4], df.attrs.get('detected_ms1_trap') in ('linear', 'amazon')
            analyzers[r] = (df.attrs.get('detected_ms1_trap'), df.attrs.get('detected_trap'))
            tol = 0.5 if trap else 0.02
            t = runs[r].reset_index()
            t = t[np.array([isinstance(c, dict) and bool(c) for c in t['composition']]) & t.get('adduct', pd.Series(None, index = t.index)).isna().values & (t['rel_abundance'] > 0).values]
            areas = extract_xic_areas(ms1, [ion_mz(c, z) for c, z in zip(t['composition'], t['charge'])], t['RT'].values, charges = t['charge'].values,
                                      mz_tolerance = tol, weights = t['rel_abundance'].values, glycans = [formula(c) for c in t['composition']]) if len(t) else np.zeros(0)
            checks[r].update({'signal': float(areas.sum()), 'ms2': len(df)})
            if len(candidates):
                # (apex heights, not areas, as integration bounds scatter more; whether the apex is a peak of its own is kept apart, as where a run's RT
                # shift puts the apex near the window's edge it fails that test but still measures the ion)
                found = extract_xic_areas(ms1, cand_mz, features.loc[candidates, 'RT'].values.astype(float), charges = features.loc[candidates, 'charge'].values,
                                          mz_tolerance = tol, search_window = 1.0)
                measured[r], genuine[r] = found[2], found[3]
            span = (float(df['RT'].min()), float(df['RT'].max()))
            for k, spec in enumerate(specs):
                center, window = (spec['rt'], 1.0) if spec['rt'] is not None else (sum(span) / 2, (span[1] - span[0]) / 2)
                area, apex, height = extract_xic_areas(ms1, [spec['mz']], [center], charges = [spec['charge']], mz_tolerance = tol, search_window = window,
                                                       glycans = [formula(spec['composition'])])[:3]
                # (ion-trap precursors read up to a few tenths of an m/z off)
                near = ((df['m/z'] - spec['mz']).abs() <= (0.6 if trap else tol)) & ((df['RT'] - apex[0]).abs() <= 1.5)
                checks[r].update({f'std{k}': float(area[0]), f'std{k}_rt': float(apex[0]) if area[0] > 0 else np.nan, f'std{k}_height': float(height[0]), f'std{k}_ms2': int(near.sum())})
        for k, spec in enumerate(specs):
            spec['found'] = any(checks[r][f'std{k}'] > 0 for r in run_cols)

    def profile_of(keep):
        return abundance_table(values[keep], run_cols)[0].set_index('glycan').reindex(columns = run_cols, fill_value = 0)

    def amounts(table):
        # Glycan signal per run relative to the first standard: the profile's share of the run's rows times their integrated signal, over the standard's
        return pd.Series({r: table[r].sum() / 100 * checks[r]['signal'] / checks[r]['std0'] if checks[r].get('std0', 0) > 0 else np.nan for r in run_cols})

    values = features.copy()
    table = profile_of(in_profile)
    amount, linked = pd.Series(dtype = float), {}
    if specs and spectra_filepaths and specs[0]['found']:
        amount = amounts(table)
        # A background that comes with the standard (an impurity of it, a by-product of its preparation, the reagents spiked with it, or carryover) holds a
        # constant ratio to it in nearly every run, also where a sample holds no glycan, while the samples' glycans follow their glycan amounts. Its share
        # is subtracted, not the glycan: the background is the ion's median ratio to the standard, and a run keeps what rises clearly above it (3 robust
        # standard deviations of the ratio, and 1.5-fold at least), so a glycan that is also the samples' own stays where they hold it
        stds = pd.Series({r: checks[r].get('std0_height', 0) for r in run_cols})
        usable = [r for r in run_cols if stds[r] > 0 and amount[r] > 0]
        for i in candidates:
            # whether the ion comes with the standard is judged on the runs where it has a peak of its own (an absent glycan's window holds noise, which keeps
            # a constant ratio to the standard too); what a run keeps of it, on every run where it was measured
            ratio = (measured.loc[i, usable] / stds[usable]).astype(float)
            own = ratio[genuine.loc[i, usable].values & (ratio > 0)]
            x = np.log10(amount[own.index].astype(float))
            if len(own) >= max(6, 2 / 3 * len(usable)) and x.max() - x.min() >= np.log10(5) and np.polyfit(x, np.log10(own), 1)[0] < linked_slope:
                floor = np.log10(own).median()
                cut = floor + max(3 * 1.4826 * (np.log10(own) - floor).abs().median(), np.log10(1.5))
                ratio = ratio[ratio > 0]
                # (a run where the ion was not measured keeps none of it: nothing shows it above the background there)
                kept = pd.Series(0.0, index = run_cols)
                kept[ratio.index] = np.where(np.log10(ratio) > cut, 1 - 10 ** floor / ratio, 0)
                values.loc[i, run_cols] = features.loc[i, run_cols].astype(float) * kept
                n = int((values.loc[i, run_cols].astype(float).fillna(0) > 0).sum())
                linked[i] = (10 ** floor, n)
                if n:
                    note[i] = f"Background of internal standard {specs[0]['name']} subtracted ({10 ** floor:.0%} of its signal in every sample); above it in {n} sample{'s' if n > 1 else ''}"
                else:
                    note[i] = f"Comes with internal standard {specs[0]['name']} ({10 ** floor:.0%} of its signal in every sample), none above it"
                    in_profile[i] = False
                    values.loc[i, run_cols] = features.loc[i, run_cols]
        if linked:
            table = profile_of(in_profile)
            amount = amounts(table)
    total = table.sum()
    table = table / total.replace(0, np.nan) * 100
    table.columns = samples
    # One row per glycan with its main ion (its most abundant feature), the RTs of its peaks, and glycowork's named motifs it carries
    feats = values[in_profile].assign(label = [_row_label(s, c) for s, c in zip(features.loc[in_profile, 'top1_pred'], features.loc[in_profile, 'composition'])])
    # Per glycan the row of its evidence: of its rows with MS2, the one of its best confidence (a composition: of most spectra), whose diagnostic ions
    # (supported_by) and spectrum the report shows, so evidence and confidence come from one spectrum
    confidence, evidence = {}, {}
    for r in run_cols:
        t = runs[r]
        scores = t['confidence'] if 'confidence' in t.columns else pd.Series(np.nan, index = t.index)
        # (a row excluded in curation, in the app, is evidence of nothing)
        excluded = t['excluded'].astype(bool) if 'excluded' in t.columns else [False] * len(t)
        for k, (s, c, x, n, out) in enumerate(zip(t['top1_pred'], t['composition'], scores, t['num_spectra'] if 'num_spectra' in t.columns else [1] * len(t), excluded)):
            if out:
                continue
            if isinstance(s, str) and pd.notna(x):
                confidence[s] = max(confidence.get(s, 0), float(x))
            g, rank = _row_label(s, c), (float(x) if pd.notna(x) else -1.0, float(n))
            if g is not None and n > 0 and (g not in evidence or rank > evidence[g][0]):
                evidence[g] = (rank, r, k)
    structures = [g for g in table.index if '(' in g]
    # glycowork's motifs per sample (quantify_motifs with its default feature set: each glycan's abundance times the times it carries the motif, identical
    # motif rows merged; a glycan known by its composition alone carries its monosaccharides), and per glycan the motifs of that table it carries
    empty = [s for s, r in zip(samples, run_cols) if not total[r] > 0]
    measured = [s for s in samples if s not in empty]
    if len(table) and measured:
        motif_table, counts = quantify_motifs(table[measured].fillna(0), return_counts = True)
        motif_table = motif_table.reindex(columns = samples)
        motifs = counts[[m for m in motif_table.index if m in counts.columns]] > 0
    else:
        motif_table, motifs = pd.DataFrame(columns = samples), pd.DataFrame()
    rows = []
    for g in table.index:
        f = feats[feats['label'] == g]
        main = f.loc[f[run_cols].fillna(0).mean(axis = 1).idxmax()]
        comp, z, structure = main['composition'], int(main['charge']), g if g in structures else ''
        e = evidence.get(g)
        support = runs[e[1]]['supported_by'].iat[e[2]] if structure and e and 'supported_by' in runs[e[1]].columns else ''
        rows.append({'Structure': structure, 'Composition': format_composition(comp, 'names'),
                     'Class': get_class(structure) if structure else '', 'Formula': formula(comp), 'Mass': mass(comp),
                     'm/z (theoretical)': ion_mz(comp, z), 'Charge': z, 'RT (min)': ', '.join(f'{x:.1f}' for x in sorted(f['RT'])),
                     'GlyTouCan ID': next((x for x in f.get('GlyTouCan_ID', []) if isinstance(x, str) and x), ''), 'Best confidence': confidence.get(structure, np.nan),
                     # (the residues placed and the readings, without the count of residues before them)
                     'Diagnostic ions': re.sub(r'^\d+/\d+ residues supported(: |; |$)', '', support) if isinstance(support, str) else '',
                     'Motifs': ', '.join(m for m in motifs.columns if motifs.loc[g, m]) if g in motifs.index else '', 'Samples detected': int((table.loc[g] > 0).sum()), 'Mean (%)': table.loc[g].mean(), 'Median (%)': table.loc[g].median(), '_composition': comp,
                     '_rt': f['RT'].min()})
    meta = pd.DataFrame(rows, index = table.index, columns = ['Structure', 'Composition', 'Class', 'Formula', 'Mass', 'm/z (theoretical)', 'Charge', 'RT (min)',
                                                              'GlyTouCan ID', 'Best confidence', 'Diagnostic ions', 'Motifs', 'Samples detected', 'Mean (%)', 'Median (%)', '_composition',
                                                              '_rt'])
    # By mass, isomers by retention time
    order = meta.sort_values(['Mass', '_rt'], kind = 'stable').index
    meta, table = meta.loc[order].drop(columns = '_rt'), table.loc[order]
    # Each glycan's evidence spectrum, annotated as the app annotates a run's spectrum: at the run's fragment tolerance (wrap_inference_batch's attrs, else
    # its default mass tolerance), with the reducing end of its row, metal-carrying fragments unless the row is (de)protonated, radical fragmentation
    # where the run used it, and permethylated glycans without cross-rings under collisional activation
    from candycrunch.analysis import precursor_metals, RADICAL_METHODS
    from candycrunch.prediction import crumbs_mass_dict, MZ_REF
    spectrum_rows = []
    for j, g in enumerate(order):
        if g not in evidence or evidence[g][1] not in peak_lists or not spectra:
            continue
        rank, r, k = evidence[g]
        t, peaks = runs[r], peak_lists[r][k]
        if not isinstance(peaks, dict) or not peaks:
            continue
        own = t['reducing_end'].iat[k] if 'reducing_end' in t.columns else None
        row_end = 'custom' if own == 'free' else own if isinstance(own, str) else end
        da, ppm = t.attrs.get('fragment_tolerance', (None, None))
        activation, adduct = t.attrs.get('activation'), t['adduct'].iat[k] if 'adduct' in t.columns else None
        kwargs = {'metal_cations': 'adduct' not in t.columns or (precursor_metals([adduct]) if isinstance(adduct, str) else False)}
        kwargs |= {'fragmentation_method': activation} if activation in RADICAL_METHODS else {'disable_X_cross_rings': True, 'disable_A_cross_rings': True} if sample_prep in ('permethylated', 'deuteropermethylated') else {}
        preds = t['predictions'].iat[k] if 'predictions' in t.columns else None
        spectrum_rows.append({'Glycan': g if g in structures else format_composition(meta.loc[g, '_composition'], 'names'), 'Sample': names[r], 'RT (min)': float(t['RT'].iat[k]),
                              'Precursor m/z': float(t['m/z'].iat[k] if 'm/z' in t.columns else t.index[k]), 'Charge': int(t['charge'].iat[k]), 'Confidence': rank[0] if rank[0] >= 0 else np.nan,
                              '_glycan': j, '_annotation': {'structure': g, 'peaks': peaks, 'charge': int(t['charge'].iat[k]), 'tolerance': da or 300 * MZ_REF / 1e6,
                                                            'ppm': ppm, 'mass_tag': crumbs_mass_dict.get((row_end, sample_prep), 0), 'sample_prep': sample_prep,
                                                            'candidates': [p[0] for p in preds if p[0] != g] if g in structures and isinstance(preds, list) else [],
                                                            'ion_trap': ppm is None and (da or 1) > 0.1, 'kwargs': kwargs}})
    # (most abundant first)
    motif_table = motif_table[(motif_table.fillna(0) > 0).any(axis = 1)]
    motif_table = motif_table.loc[motif_table.mean(axis = 1).sort_values(ascending = False, kind = 'stable').index]
    # Samples
    sample_rows = []
    for r in run_cols:
        row = {'Sample': names[r], 'Run': r, 'MS2 spectra': checks[r].get('ms2'), 'Glycan peaks': int((values.loc[in_profile, r].fillna(0) > 0).sum()),
               'Glycans': int((table[names[r]].fillna(0) > 0).sum())}
        for k, spec in enumerate(specs if checks[r] else []):
            row.update({f"{spec['name']}: MS2 spectra": checks[r][f'std{k}_ms2'], f"{spec['name']}: RT (min)": checks[r][f'std{k}_rt'],
                        f"{spec['name']}: MS1 signal": checks[r][f'std{k}']})
        if len(amount):
            row['Glycan signal / standard'] = amount[r]
        sample_rows.append(row)
    sample_table = pd.DataFrame(sample_rows).dropna(axis = 1, how = 'all')
    if 'Glycan signal / standard' in sample_table.columns:
        sample_table['Relative glycan amount (median sample = 1)'] = sample_table['Glycan signal / standard'] / sample_table['Glycan signal / standard'].median()
    # Every peak, on the scale of its sample's glycan profile (the profile's peaks sum to 100)
    # (a glycan that also comes with the standard shows what rises above its background)
    vals = values[run_cols].astype(float).div(total.replace(0, np.nan), axis = 1) * 100
    vals.columns = samples
    theoretical = pd.Series([ion_mz(c, z) if isinstance(c, dict) and c else np.nan for c, z in zip(features['composition'], features['charge'])], index = features.index)
    # (a peak read as another ion than its (de)protonated one, a metal or anion adduct, keeps only its measured m/z)
    theoretical[(theoretical - features['m/z']).abs() > 1] = np.nan
    peak_table = pd.concat([pd.DataFrame({'Note': note, 'Structure': features['top1_pred'].fillna(''), 'Composition': [format_composition(c, 'names') for c in features['composition']],
                                          'Mass': [mass(c) for c in features['composition']], 'm/z (theoretical)': theoretical,
                                          'm/z (measured)': features['m/z'], 'Charge': features['charge'].astype(int), 'RT (min)': features['RT'],
                                          'GlyTouCan ID': features.get('GlyTouCan_ID', ''), 'Samples with MS2': features['n_files_ms2'],
                                          'Samples detected': (vals > 0).sum(axis = 1)}), vals.where(vals > 0)], axis = 1)
    peak_table = peak_table.sort_values(['Mass', 'RT (min)'], kind = 'stable').reset_index(drop = True)
    report = {'glycans': pd.concat([meta, table.where(table > 0)], axis = 1).reset_index(drop = True), 'motifs': motif_table.rename_axis('Motif').reset_index(),
              'samples': sample_table, 'peaks': peak_table,
              'spectra': pd.DataFrame(spectrum_rows, columns = ['Glycan', 'Sample', 'RT (min)', 'Precursor m/z', 'Charge', 'Confidence', '_glycan', '_annotation'])}
    # The settings, and what the runs tell of their acquisition where the caller does not give it: ion mode by the charges, mass analyzers where the spectra
    # were read, activation as the runs name it
    shown = {'CandyCrunch': __version__} | (settings or {})
    modes = {'negative' if z < 0 else 'positive' for z in features['charge']}
    named = lambda xs: ' and '.join(sorted({ANALYZER_NAMES[x] for x in xs if x in ANALYZER_NAMES}))
    ms1, ms2 = named(a[0] for a in analyzers.values()), named(a[1] for a in analyzers.values())
    activation = ' and '.join(sorted({str(runs[r].attrs['activation']) for r in run_cols if runs[r].attrs.get('activation')}))
    told = {'ion mode': ' and '.join(sorted(modes)), 'mass analyzer': (f'{ms1} (MS1 and MS2)' if ms1 == ms2 else ', '.join(f'{a} ({k})' for a, k in ((ms1, 'MS1'), (ms2, 'MS2')) if a)) if ms1 or ms2 else '',
            'activation': activation}
    shown |= {k: v for k, v in told.items() if v and k not in shown}
    report['glycans'].attrs = {'title': title, 'settings': shown, 'standards': specs, 'samples': samples, 'empty': empty,
                               'linked': [f"{format_composition(features.loc[i, 'composition'])} at m/z {theoretical[i] if theoretical[i] == theoretical[i] else features.loc[i, 'm/z']:.2f}, {features.loc[i, 'RT']:.1f} min, {f:.0%} of the standard's signal"
                                          + (f" (above it in {n} sample{'s' if n > 1 else ''})" if n else '') for i, (f, n) in linked.items()]}
    if filepath:
        write_report(report, filepath, drawings = drawings, spectra = spectra)
    return report


def write_report(report, filepath, drawings = True, spectra = True):
    """writes the report of glycan_report as an .xlsx workbook: About, Glycans (with SNFG drawings), Motifs, Samples, Peaks, and Spectra (each glycan's
    annotated MS2 spectrum, which its diagnostic ions link to)\n
   | Arguments:
   | :-
   | report (dict): what glycan_report returns
   | filepath (string): .xlsx file to write
   | drawings (bool): whether the glycans sheet shows an SNFG drawing of each glycan; default:True
   | spectra (bool): whether the workbook has the sheet of annotated spectra, where the report holds them; default:True
   """
    from openpyxl import Workbook
    from glycowork.motif.draw import add_svg_twins
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from candycrunch.utils import format_composition
    glycans, info = report['glycans'], report['glycans'].attrs
    samples, standards = info['samples'], info['standards']
    head, line = PatternFill('solid', fgColor = 'DCE6F0'), Border(bottom = Side(style = 'thin', color = '8EA9C8'))
    formats = {'Mass': '0.0000', 'm/z (theoretical)': '0.0000', 'm/z (measured)': '0.00', 'Precursor m/z': '0.00', 'RT (min)': '0.00', 'Best confidence': '0.00',
               'Confidence': '0.00', 'Mean (%)': '0.00', 'Median (%)': '0.00',
               'Glycan signal / standard': '0.000', 'Relative glycan amount (median sample = 1)': '0.000', 'MS2 spectra': '#,##0'}
    # (the standards' columns of the samples sheet)
    formats |= {f"{s['name']}: {k}": f for s in standards for k, f in (('MS2 spectra', '#,##0'), ('RT (min)', '0.00'), ('MS1 signal', '#,##0'))}
    # Widths in characters: these columns as wide as their longest entry up to the width given, the rest fixed
    fitted = {'Structure': 60, 'Note': 40, 'Composition': 30, 'Formula': 20, 'Diagnostic ions': 50, 'Motifs': 60, 'Run': 30, 'Motif': 30, 'Glycan': 40,
              'Sample': 20}
    widths = {'Mass': 10, 'Glycan signal / standard': 14, 'Relative glycan amount (median sample = 1)': 16}
    wrapped = ('Structure', 'Note', 'Diagnostic ions', 'Motifs', 'Glycan')
    shown_spectra = report.get('spectra', pd.DataFrame()) if spectra else pd.DataFrame()

    def lines(text, width, bold = False):
        # Lines a wrapped cell takes: its words filled into the column's width, in widths of '0' (plain text fits about a tenth more characters, bold
        # about as many)
        n, used = 1, 0
        for word in str(text).split(' '):
            if used and used + 1 + len(word) > width * (1 if bold else 1.1) - 1:
                n, used = n + 1, len(word)
            else:
                used += len(word) + (1 if used else 0)
        return n

    def sheet(name, df, number_format, first):
        ws = wb.create_sheet(name)
        width = {c: min(fitted[c], max([len(c), *(len(str(v)) for v in df[c] if isinstance(v, str))]) + 2) if c in fitted else widths.get(c, 7 if c in samples else 11)
                 for c in df.columns}
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(row = 1, column = j, value = c)
            cell.font, cell.fill, cell.border, cell.alignment = Font(bold = True), head, line, Alignment(wrap_text = True, vertical = 'bottom')
            ws.column_dimensions[get_column_letter(j)].width = width[c]
        # (rows as high as their wrapped text, as Excel sizes a row to its text only when it edits it, and the drawings set their rows' heights)
        ws.row_dimensions[1].height = 15 * max([lines(c, width[c], bold = True) for c in df.columns] or [1])
        for i, row in enumerate(df.itertuples(index = False), 2):
            for j, (c, v) in enumerate(zip(df.columns, row), 1):
                cell = ws.cell(row = i, column = j, value = None if isinstance(v, float) and np.isnan(v) else v.item() if isinstance(v, np.generic) else v)
                cell.number_format = formats.get(c, number_format if c in samples else 'General')
                cell.alignment = Alignment(wrap_text = c in wrapped, vertical = 'center')
            text = max([lines(v, width[c]) for c, v in zip(df.columns, row) if c in wrapped and isinstance(v, str)] or [1])
            if text > 1:
                ws.row_dimensions[i].height = 15 * text
        ws.freeze_panes = ws.cell(row = 2, column = first + 1)
        if len(df):
            ws.auto_filter.ref = f'A1:{get_column_letter(len(df.columns))}{len(df) + 1}'
        return ws

    wb = Workbook()
    ws = wb.active
    ws.title = 'About'
    listed = lambda xs: xs[0] if len(xs) == 1 else ' and '.join(xs) if len(xs) == 2 else ', '.join(xs[:-1]) + ', and ' + xs[-1]
    text = [(info['title'], 14), ('', 0),
            (f'{len(glycans)} glycans' + (', each with a full or partial structure,' if (glycans['Structure'] != '').all() else f' ({int((glycans["Structure"] != "").sum())} with a full or partial structure)') +
             f' across {len(samples)} samples; '
             f'{int((glycans["Samples detected"] >= len(samples) / 2).sum())} of them in at least half of the samples.', 0)]
    if info['empty']:
        text.append((f"{listed(info['empty'])} hold{'s' if len(info['empty']) == 1 else ''} no glycan of the profile; {'its column is' if len(info['empty']) == 1 else 'their columns are'} empty.", 0))
    if info['settings']:
        text.append(('Settings: ' + '; '.join(f'{k} {v}' for k, v in info['settings'].items()) + '.', 0))
    text += [('', 0), ('Sheets', 12),
             ("Glycans: one row per glycan (a structure's peaks at different retention times, or those of a composition without a structure, summed; their retention times are listed), "
              "by mass and, among isomers, by retention time, per sample in % of that sample's glycan signal, so every sample sums to 100. Mass is the neutral "
              "monoisotopic mass of the glycan as measured, with its reducing end; m/z is the theoretical m/z of its main ion; Best confidence is the calibrated "
              "probability that the structure is right, the best over the samples; Diagnostic ions are the fragments of that best spectrum that place each residue "
              "that could sit elsewhere (each residue with its fragments' m/z and Domon-Costello names), a fragment that only another "
              "placement explains ('against'), and, on ion traps, the linkage of a lone Neu5Ac as its [M-H-Neu5Ac]- ion reads" + (
              "; the cell links to that spectrum, annotated, in the Spectra sheet" if len(shown_spectra) else '') + "; Motifs are the motifs of the Motifs sheet it carries.", 0),
             ("Motifs: glycowork's motifs per sample (quantify_motifs, its named motifs and every monosaccharide and disaccharide): the abundance of each glycan, "
              "times the number of times it carries the motif, summed; a glycan known by its composition alone carries its monosaccharides, so where there is one, "
              "monosaccharides are counted as Hex, HexNAc, and so on for every glycan; motifs that every glycan carries alike are merged.", 0),
             ('Samples: per sample its glycan peaks and glycans' + (', its MS2 spectra, the internal standard, and the glycan signal relative to the standard.' if standards else '.'), 0),
             ('Peaks: every glycan peak by mass and retention time, with its theoretical and measured m/z and charge, per sample on the scale of the glycan profile. Note names '
              'what is not part of the profile (internal standards, in-source fragments, peeling products, glucose oligomers, and glycans of another class) and the '
              'background subtracted from a glycan that also comes with the standard.', 0)]
    if len(shown_spectra):
        text.append(("Spectra: for every glycan the MS2 spectrum of its best confidence (for a glycan known by its composition, that of its peak with the most MS2 scans), with "
                     "its sample, retention time, precursor m/z, and charge, annotated by CandyCrunch's CandyCrumbs: fragments in Domon-Costello nomenclature with "
                     "their SNFG cartoons, the diagnostic ions in lilac.", 0))
    if standards:
        s = standards[0]
        text += [('', 0), ('Internal standard', 12),
                 (f"{s['name']} ({format_composition(s['composition'], 'names')}, m/z {s['mz']:.2f} at charge {s['charge']}) " + (
                  "is integrated in every sample from its MS1 isotope envelope and left out of the profile. Glycan signal / standard: the sample's integrated glycan "
                  "signal over the standard's; the relative glycan amount scales it so that the median sample is 1." if s.get('found') else
                  "has no MS1 peak in any sample, so the samples' glycan amounts are not given."), 0)]
        if info['linked']:
            # (one line per ion: listed in one sentence, they ran on for lines)
            text.append(("Ions that come with the standard hold a constant ratio to it in every sample, also where a sample holds no glycan: that background is "
                         "subtracted, and a glycan keeps what rises clearly above it in a sample (3 robust standard deviations of its ratio to the standard, and "
                         "1.5-fold at least). These ions came with it:", 0))
            text += [(f'\u2022 {x}', 0) for x in info['linked']]
    text += [('', 0), ('Notation', 12),
             ('Structures in IUPAC-condensed notation, without the reducing-end label or alditol. Linkages the spectra do not resolve are written with their '
              'alternatives (b1-3/4) or ?; a residue in braces {} sits at an open position. Short compositions: H Hex, N HexNAc, F dHex (Fuc), A Neu5Ac, G Neu5Gc, '
              'S sulfate, P phosphate. Formula: the neutral glycan as measured, with its reducing end.', 0)]
    for i, (t, size) in enumerate(text, 1):
        cell = ws.cell(row = i, column = 1, value = t)
        cell.alignment = Alignment(wrap_text = True, vertical = 'top')
        if size:
            cell.font = Font(bold = True, size = size)
        ws.row_dimensions[i].height = 1.5 * size if size else 15 * lines(t, 150)
    ws.column_dimensions['A'].width = 150
    shown, svgs = glycans.drop(columns = ['_composition']), {}
    ws = sheet('Glycans', pd.concat([pd.DataFrame({'Drawing': [None] * len(shown)}), shown], axis = 1) if drawings else shown, '0.00', 2 if drawings else 1)
    if len(shown_spectra):
        # Each glycan's diagnostic ions link to its annotated spectrum (a glycan without them: the link alone)
        from openpyxl.worksheet.hyperlink import Hyperlink
        column = list(shown.columns).index('Diagnostic ions') + 1 + bool(drawings)
        for k, j in enumerate(shown_spectra['_glycan']):
            cell = ws.cell(row = j + 2, column = column)
            cell.value = cell.value or 'Annotated spectrum'
            cell.hyperlink = Hyperlink(ref = cell.coordinate, location = f"'Spectra'!A{k + 2}")
            cell.font = Font(color = '0563C1', underline = 'single')
    if drawings:
        # glycowork's SNFG drawings, one scale for every glycan, each with its vector twin for Excel 2016+ once the workbook is saved
        from glycowork.motif.draw import draw_glycans_in_sheet
        from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
        from openpyxl.drawing.xdr import XDRPositiveSize2D
        heights = {i: ws.row_dimensions[i].height or 15 for i in range(2, len(shown) + 2)}
        svgs = draw_glycans_in_sheet(ws, [structure or composition for structure, composition in zip(glycans['Structure'], glycans['_composition'])], 1)
        # (a row as high as its drawing or its text, whichever is higher, and each drawing in the middle of its row, as the text beside it)
        for i, h in heights.items():
            ws.row_dimensions[i].height = max(h, ws.row_dimensions[i].height or 15)
        for image in ws._images:
            cell = ws[image.anchor]
            offset = max(0, ws.row_dimensions[cell.row].height - image.height * 0.75) / 2
            image.anchor = OneCellAnchor(_from = AnchorMarker(col = cell.column - 1, row = cell.row - 1, rowOff = round(offset * 12700)),
                                         ext = XDRPositiveSize2D(round(image.width * 9525), round(image.height * 9525)))
    sheet('Motifs', report['motifs'], '0.0', 1)
    sheet('Samples', report['samples'], 'General', 1)
    sheet('Peaks', report['peaks'], '0.00', 2)
    if len(shown_spectra):
        # Each glycan's spectrum annotated as the app annotates it, drawn by Excel 2016+ from its vector twin, by other readers from its PNG at the size shown
        import struct
        from io import BytesIO
        from openpyxl.drawing.image import Image
        from openpyxl.utils import get_column_letter
        from candycrunch.analysis import plot_annotated_spectrum, spectrum_hits, supporting_ions

        class Picture(Image):
            # (openpyxl calls Pillow only to learn a PNG's size and to hand back its bytes, both of which are known here)
            def __init__(self, data, width, height):
                self.ref, self.width, self.height, self.format = BytesIO(data), width, height, 'png'

            def _data(self):
                return self.ref.getvalue()

        ws = sheet('Spectra', shown_spectra.drop(columns = ['_glycan', '_annotation']).assign(Spectrum = None), '0.00', 1)
        column = get_column_letter(len(shown_spectra.columns) - 1)
        for i, a in enumerate(shown_spectra['_annotation'], 2):
            hits = spectrum_hits(a['structure'], a['peaks'], a['charge'], a['tolerance'], mass_threshold_ppm = a['ppm'], mass_tag = a['mass_tag'],
                                 sample_prep = a['sample_prep'], **a['kwargs'])
            # the peaks that place a residue or tell the structure from the other candidates, and an a2-3 reading of a lone Neu5Ac on ion traps, as in the app
            # (where it cannot place a structure, its spectrum is drawn without diagnostic ions, as there)
            try:
                support = supporting_ions(a['structure'], a['peaks'], charge = a['charge'], candidates = a['candidates'], mass_tag = a['mass_tag'],
                                          sample_prep = a['sample_prep']) if '(' in a['structure'] else None
            except Exception:
                support = None
            diagnostic = sorted({s[0] for e in support['residues'] for s in e['support']} | {s[0] for c in support['candidates'] for s in c['support']} |
                                {e['linkage'][0] for e in support['residues'] if a['ion_trap'] and e.get('linkage') and e['linkage'][2] == 'a2-3'}) if support else None
            _, ax = plot_annotated_spectrum(a['structure'], list(a['peaks']), list(a['peaks'].values()), mass_threshold = a['tolerance'], charge = a['charge'],
                                            mass_tag = a['mass_tag'], sample_prep = a['sample_prep'], hit_dict = hits, diagnostic = diagnostic, figsize = (9, 3.6),
                                            label_fontsize = 7)
            png = ax.figure.to_png(dpi = 96)
            svgs[png] = ax.figure.to_svg().encode('utf-8')
            width, height = struct.unpack('>II', png[16:24])
            ws.add_image(Picture(png, width, height), f'{column}{i}')
            ws.row_dimensions[i].height = max(ws.row_dimensions[i].height or 15, height * 0.75)
            ws.column_dimensions[column].width = width / 7
    wb.save(filepath)
    add_svg_twins(filepath, svgs)

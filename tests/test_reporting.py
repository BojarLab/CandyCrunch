import os
import re
import struct
import zipfile
import numpy as np
import pandas as pd
import pytest
from glycowork.motif.tokenization import composition_to_mass, PROTON_MASS
import candycrunch
import candycrunch.prediction
from candycrunch import cli
from candycrunch.reporting import glycan_report

# name: (structure, composition, charge, RT, height at amount 1, flag); STD is spiked into every sample at one amount, LINKED comes with it at a constant
# ratio (as the core 1 disaccharide with Jin's standard), MIXED has such a background too but is also the samples' own (60 x amount^1.5: notable only in the
# samples richest in glycans, as Gal(b1-3)GalNAc in the colon set), and
# the others follow the sample's glycan amount
GLYCANS = {'G1': ('Neu5Ac(a2-3)Gal(b1-3)GalNAc', {'Hex': 1, 'HexNAc': 1, 'Neu5Ac': 1}, -1, 13.6, 4e4, ''),
           'G2': ('GlcNAc(b1-3)[Neu5Ac(a2-6)]GalNAc', {'HexNAc': 2, 'Neu5Ac': 1}, -1, 12.0, 8e4, ''),
           'G3': ('Gal(b1-3)[GlcNAc(b1-6)]GalNAc', {'Hex': 1, 'HexNAc': 2}, -1, 11.4, 2e4, ''),
           'G4': ('Fuc(a1-2)Gal(b1-3)GalNAc', {'Hex': 1, 'HexNAc': 1, 'dHex': 1}, -1, 21.1, 1e4, ''),
           'GLC': ('Glc(a1-4)Glc(a1-4)Glc', {'Hex': 3}, -1, 12.8, 1e4, 'glucose oligomer'),
           'STD': ('Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc', {'Hex': 5, 'HexNAc': 3, 'Neu5Ac': 2}, -2, 17.1, 5e4, ''),
           'LINKED': ('Gal(b1-3)GalNAc', {'Hex': 1, 'HexNAc': 1}, -1, 8.5, 5e3, ''),
           'MIXED': ('Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', {'Hex': 1, 'HexNAc': 1, 'Neu5Ac': 1}, -1, 11.4, 4e3, ''),
           'COMP': (None, {'Hex': 2, 'HexNAc': 3, 'dHex': 2, 'Neu5Ac': 1, 'S': 1}, -2, 15.5, 1.5e4, '')}
AMOUNTS = dict(zip([f'run{k}' for k in range(1, 9)], np.logspace(-1, 1.5, 8)))


def ion(c, z):
    return (composition_to_mass(c, modification = 'reduced') + z * PROTON_MASS) / abs(z)


def height(name, run):
    rng = np.random.default_rng(int(run[3:]) * 10 + list(GLYCANS).index(name))
    base = GLYCANS[name][4] * (1 if name in ('STD', 'LINKED', 'MIXED') else AMOUNTS[run]) + (60 * AMOUNTS[run] ** 1.5 if name == 'MIXED' else 0)
    return base * rng.lognormal(0, 0.1)


def survey_scans(run):
    """ion-trap-like survey scans (one every 0.02 min): each glycan a Gaussian peak in time (sigma 0.06 min) with its first isotopes, on a low noise floor;
    run2's survey scans miss LINKED, which the batch still reports there (a run where the report cannot measure the ion keeps none of it)"""
    rng = np.random.default_rng(int(run[3:]))
    rts = np.arange(0, 30, 0.02)
    mzs, ints, offsets = [], [], [0]
    for t in rts:
        scan = {round(m, 2): v for m, v in zip(rng.uniform(380, 1200, 60), rng.uniform(20, 60, 60))}
        for name, (_, comp, z, rt, _, _) in GLYCANS.items():
            h = height(name, run) * np.exp(-(t - rt) ** 2 / (2 * 0.06 ** 2)) * (name != 'LINKED' or run != 'run2')
            if h > 1:
                for k, share in enumerate((1, 0.6, 0.25)):
                    scan[round(ion(comp, z) + k * 1.00336 / abs(z), 4)] = scan.get(round(ion(comp, z) + k * 1.00336 / abs(z), 4), 0) + h * share
        keys = sorted(scan)
        mzs.extend(keys)
        ints.extend(scan[k] for k in keys)
        offsets.append(len(mzs))
    return rts, np.array(mzs), np.array(ints), np.array(offsets)


def batch():
    """wrap_inference_batch-like output: per run its rows (% of the run's signal), and the feature table"""
    runs = {}
    for run in AMOUNTS:
        rows = [(ion(comp, z), s, 0.9, comp, z, rt + 0.05 * int(run[3:]) % 3, height(name, run), None, '')
                for name, (s, comp, z, rt, _, _) in GLYCANS.items() if name not in ('G4',) or AMOUNTS[run] > 1]
        t = pd.DataFrame(rows, columns = ['m/z', 'top1_pred', 'confidence', 'composition', 'charge', 'RT', 'rel_abundance', 'adduct', 'notes']).set_index('m/z')
        t['rel_abundance'] = t['rel_abundance'] / t['rel_abundance'].sum() * 100
        runs[run] = t
    features = pd.DataFrame([{'top1_pred': s, 'm/z': ion(comp, z) - 0.3 / abs(z), 'RT': rt, 'charge': z, 'composition': comp, 'GlyTouCan_ID': None,
                              'n_files_ms2': len(AMOUNTS), 'flag': flag} for name, (s, comp, z, rt, _, flag) in GLYCANS.items()])
    for run, t in runs.items():
        features[run] = [t.loc[[s == x if isinstance(s, str) else pd.isna(x) for x in t['top1_pred']], 'rel_abundance'].sum() or np.nan for s in features['top1_pred']]
        features[f'evidence_{run}'] = 'ms2'
    return features, runs


def test_glycan_report(tmp_path, monkeypatch):
    features, runs = batch()

    def load(path, extract_ms1 = False):
        # three MS2 spectra of the standard, and the run's survey scans
        df = pd.DataFrame({'m/z': [ion(GLYCANS['STD'][1], -2) - 0.3] * 3, 'RT': [17.0, 17.1, 17.2]})
        df.attrs = {'ms1': survey_scans(os.path.basename(path)[:-3]), 'detected_ms1_trap': 'linear'}
        return df

    monkeypatch.setattr(candycrunch.prediction, 'load_spectra_filepath', load)
    path = tmp_path / 'report.xlsx'
    report = glycan_report(features, runs, str(path), spectra_filepaths = [str(tmp_path / f'{r}.cc') for r in AMOUNTS], standards = ['H5N3A2'],
                           sample_names = {r: r.replace('run', 'S') for r in AMOUNTS}, title = 'Test', settings = {'glycan class': 'O'})
    glycans, samples = report['glycans'], [f'S{k}' for k in range(1, 9)]
    # The profile: the glycans, also the one known by its composition alone, not the standard, the ion that only comes with it, or the glucose oligomer;
    # every sample sums to 100
    assert set(glycans['Structure']) == {GLYCANS[g][0] for g in ('G1', 'G2', 'G3', 'G4', 'MIXED')} | {''}
    assert glycans.loc[glycans['Structure'] == '', 'Composition'].tolist() == ['Hex2HexNAc3Neu5Ac1S1dHex2']
    assert np.allclose(glycans[samples].fillna(0).sum(), 100)
    assert glycans.loc[glycans['Structure'] == GLYCANS['G2'][0], 'Composition'].item() == 'HexNAc2Neu5Ac1' and 'Composition (short)' not in glycans.columns
    notes = dict(zip(report['peaks']['Structure'], report['peaks']['Note']))
    assert notes[GLYCANS['STD'][0]].startswith('Internal standard') and notes[GLYCANS['LINKED'][0]].startswith('Comes with internal standard')
    assert notes[GLYCANS['GLC'][0]].startswith('Glucose oligomer') and notes[GLYCANS['G1'][0]] == ''
    # A glycan with the standard's background keeps only what rises clearly above it (the ratio scatters ~15% here, so 1.6-fold in S7 does not): nothing
    # in the samples with little glycan, its own share in the richest
    assert notes[GLYCANS['MIXED'][0]].startswith('Background of internal standard')
    mixed = glycans.set_index('Structure').loc[GLYCANS['MIXED'][0], samples]
    assert mixed[samples[:5]].isna().all() and mixed['S8'] > 0
    # (the share of MIXED's own signal that S8 keeps: its profile value over the batch's value of it, both as shares of the run's profile signal)
    profile = [GLYCANS[g][0] for g in ('G1', 'G2', 'G3', 'G4', 'MIXED')]
    batch_share = features.loc[features['top1_pred'] == GLYCANS['MIXED'][0], 'run8'].item() / features.loc[features['top1_pred'].isin(profile) | features['top1_pred'].isna(), 'run8'].sum()
    kept = mixed['S8'] / 100 * (1 - batch_share) / (batch_share * (1 - mixed['S8'] / 100))
    assert abs(kept - 60 * AMOUNTS['run8'] ** 1.5 / (4e3 + 60 * AMOUNTS['run8'] ** 1.5)) < 0.15
    # The glycan amount relative to the standard follows the samples' amounts
    amount = report['samples'].set_index('Sample')['Glycan signal / standard']
    assert np.corrcoef(np.log(amount[samples]), np.log(list(AMOUNTS.values())))[0, 1] > 0.98
    assert (report['samples']['H5N3A2: MS2 spectra'] == 3).all()
    # glycowork's motifs per sample: core 3 is the abundance of the glycans carrying it, and the glycans name their motifs
    motifs = report['motifs'].set_index('Motif')
    core3 = glycans[glycans['Motifs'].str.contains('Oglycan_core3')][samples].fillna(0).sum()
    assert np.allclose(motifs.loc['Oglycan_core3', samples], core3) and {'Oglycan_core1', 'Oglycan_core2'} <= set(motifs.index)
    # the glycan known by its composition alone carries its monosaccharides, which the Motifs sheet counts
    assert {'Neu5Ac', 'S', 'dHex'} <= set(glycans.loc[glycans['Structure'] == '', 'Motifs'].item().split(', ')) and 'S' in motifs.index
    # at glycowork's default feature set: its named motifs and the monosaccharides and disaccharides ('exhaustive')
    from glycowork.glycan_data.loader import motif_list
    assert set(motifs.index) - set(motif_list['motif_name'])
    # The workbook, with a drawing per glycan
    from openpyxl import load_workbook
    wb = load_workbook(path)
    assert wb.sheetnames == ['About', 'Glycans', 'Motifs', 'Samples', 'Peaks'] and len(wb['Glycans']._images) == len(glycans)
    assert 'Ions that come with the standard' in ' '.join(str(c.value) for c in wb['About']['A'])
    # The version of the CandyCrunch that wrote it, whatever the caller lists
    assert f'CandyCrunch {candycrunch.__version__};' in ' '.join(str(c.value) for c in wb['About']['A'])
    # Rows by mass, isomers by retention time, in both tables
    order = list(zip(glycans['Mass'], [float(x.split(', ')[0]) for x in glycans['RT (min)']]))
    assert order == sorted(order) and glycans['Mass'].notna().all()
    peaks = report['peaks']
    assert list(zip(peaks['Mass'], peaks['RT (min)'])) == sorted(zip(peaks['Mass'], peaks['RT (min)']))
    # One scale for every drawing: a monosaccharide is the same size in every row (the size shown is the drawing's extent in the sheet; openpyxl reads back
    # each picture at its own size)
    with zipfile.ZipFile(path) as z:
        targets = {re.search(r'Id="([^"]+)"', x).group(1): re.search(r'Target="[./]*(?:xl/)?([^"]+)"', x).group(1)
                   for x in re.findall(r'<Relationship [^>]+>', z.read('xl/drawings/_rels/drawing1.xml.rels').decode())}
        shown = re.findall(r'<(?:xdr:)?ext cx="(\d+)" cy="\d+"/>.*?r:embed="(rId\d+)"', z.read('xl/drawings/drawing1.xml').decode(), re.S)
        # (glycowork shows each picture at a quarter of its PNG, to the nearest pixel)
        scales = {round(int(cx) / 9525 / struct.unpack('>II', z.read('xl/' + targets[r])[16:24])[0], 2) for cx, r in shown}
        # (each with glycowork's vector twin for Excel 2016+)
        svgs = sum(n.endswith('.svg') for n in z.namelist())
    assert len(shown) == len(glycans) and len(scales) == 1 and svgs == len(glycans)


def test_glycan_report_without_standard():
    features, runs = batch()
    report = glycan_report(features, runs, standards = None, drawings = False)
    # Without a standard everything but the flagged glycan is in the profile, and the samples sheet has no amounts
    assert len(report['glycans']) == 8 and 'Glycan signal / standard' not in report['samples'].columns


def test_cli_report_options(tmp_path, capsys):
    a, b = str(tmp_path / 'a.mzML'), str(tmp_path / 'b.mzML')
    for path in (a, b):
        open(path, 'w').close()
    # the options are checked before any run: a report needs a batch and an .xlsx, a standard needs a report and a composition
    for files, argv, message in (([a], ['--report', str(tmp_path / 'r.xlsx')], 'needs several files'), ([a, b], ['--report', str(tmp_path / 'r.csv')], 'has to end with .xlsx'),
                                 ([a, b], ['--internal_standard', 'H5N3A2'], 'goes with --report'),
                                 ([a, b], ['--report', str(tmp_path / 'r.xlsx'), '--internal_standard', 'H5N3A2,two'], 'expected a composition')):
        with pytest.raises(SystemExit):
            cli.main(['--spectra_filepath', *files, '--glycan_class', 'O', '--output', str(tmp_path / 'out.csv')] + argv)
        assert message in capsys.readouterr().err

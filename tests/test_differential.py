import io
import os
import pickle
import time
import numpy as np
import pandas as pd
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from candycrunch.differential import abundance_table, compare_groups

GLYCANS = ['Gal(b1-3)GalNAc', 'Neu5Ac(a2-3)Gal(b1-3)GalNAc', 'Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', 'Fuc(a1-2)Gal(b1-3)GalNAc',
           'GlcNAc(b1-6)[Gal(b1-3)]GalNAc', 'Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc', 'Neu5Ac(a2-6)GalNAc', 'Neu5Ac(a2-3)Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc',
           'Fuc(a1-2)Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc', 'GalNAc(a1-3)[Fuc(a1-2)]Gal(b1-3)GalNAc', 'Gal(b1-3)[Neu5Ac(a2-6)]GalNAc',
           'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc']


def feature_table(runs, effects, seed = 0):
    """A wrap_inference_batch-like feature table: one feature per structure (the first twice, as two isomer groups of the same top1), one without a
    structure, abundances in % per run with lognormal noise, effects = {run: factor of the first structure}"""
    rng = np.random.default_rng(seed)
    structures = [GLYCANS[0]] + GLYCANS + [None]
    compositions = [{'Hex': 1, 'HexNAc': 1}] * 2 + [None] * (len(GLYCANS) - 1) + [{'Hex': 2, 'HexNAc': 2, 'dHex': 1}]
    base = rng.lognormal(1.5, 0.6, len(structures))
    values = {run: base * rng.lognormal(0, 0.12, len(structures)) for run in runs}
    for run, factor in effects.items():
        values[run][:2] *= factor
    df = pd.DataFrame({'top1_pred': structures, 'composition': [c if c else {'Hex': 1} for c in compositions], 'm/z': np.arange(len(structures)) * 50 + 384.15,
                       'RT': np.linspace(10, 40, len(structures)), 'charge': -1, 'n_files_ms2': len(runs)})
    for run in runs:
        df[run] = values[run] / values[run].sum() * 100
        df[f'evidence_{run}'] = 'strong'
    # a feature not found in one run
    df.loc[5, runs[0]] = np.nan
    return df


def test_abundance_table():
    runs = ['a1', 'a2', 'b1', 'b2']
    df = feature_table(runs, {})
    table, counts = abundance_table(df, runs)
    # the two features of one structure summed, the composition-only feature kept by its composition, structures first, missing as 0
    assert table['glycan'].tolist().count(GLYCANS[0]) == 1 and counts[GLYCANS[0]] == 2
    assert table['glycan'].iloc[-1] == 'Hex2HexNAc2dHex1' and table.shape == (len(GLYCANS) + 1, 5)
    assert np.isclose(table.set_index('glycan').loc[GLYCANS[0], 'a1'], df.loc[:1, 'a1'].sum()) and (table.iloc[:, 1:] >= 0).all().all()
    # motifs only count structures; excluded features are left out
    assert 'Hex2HexNAc2dHex1' not in abundance_table(df, runs, motifs = True)[0]['glycan'].tolist()
    assert GLYCANS[3] not in abundance_table(df.assign(excluded = df['top1_pred'] == GLYCANS[3]), runs)[0]['glycan'].tolist()


def test_compare_two_groups():
    runs = ['ctrl_1', 'ctrl_2', 'ctrl_3', 'ctrl_4', 'tumor_1', 'tumor_2', 'tumor_3', 'tumor_4']
    df = feature_table(runs, {run: 4 for run in runs[4:]})
    result = compare_groups(df, {run: run.split('_')[0] for run in runs})
    table = result['table'].set_index('Glycan')
    assert result['groups'] == ['ctrl', 'tumor'] and not result['paired'] and 't-test' in result['test']
    # the planted 4-fold increase is found, in the direction of the second group over the first
    assert table.loc[GLYCANS[0], 'significant'] and table.loc[GLYCANS[0], 'Log2FC'] > 1
    assert table.loc[GLYCANS[0], 'Mean in tumor'] > 2 * table.loc[GLYCANS[0], 'Mean in ctrl'] and table.loc[GLYCANS[0], 'Features'] == 2
    assert np.allclose(result['values'].sum(), 100)
    # paired by subject, the second group ordered by the first's subjects
    pairs = {run: f'patient {run[-1]}' for run in runs}
    paired = compare_groups(df, {run: run.split('_')[0] for run in runs}, pairs = {**pairs, 'tumor_1': 'patient 2', 'tumor_2': 'patient 1'})
    assert paired['paired'] and paired['members']['tumor'][:2] == ['tumor_2', 'tumor_1'] and 'paired' in paired['test']
    motifs = compare_groups(df, {run: run.split('_')[0] for run in runs}, motifs = True)
    assert motifs['motifs'] and 'Features' not in motifs['table'].columns and len(motifs['table']) > 5


def test_compare_groups_absent_in_a_group():
    runs = ['ctrl_1', 'ctrl_2', 'ctrl_3', 'tumor_1', 'tumor_2', 'tumor_3']
    df = feature_table(runs, {})
    df.loc[8, runs[:3]] = np.nan
    df.loc[9, runs[3:]] = np.nan
    result = compare_groups(df, {run: run.split('_')[0] for run in runs})
    table = result['table'].set_index('Glycan')
    # glycans missing from a whole group are tested on their detection (3/3 against 0/3); their floor used to enter the log-ratio reference and made
    # the glycans found in every run significant
    assert np.allclose(table.loc[[GLYCANS[7], GLYCANS[8]], 'p-val'], 0.1) and not table.drop([GLYCANS[7], GLYCANS[8]])['significant'].any()
    assert table.loc[GLYCANS[7], 'Found in'] == 'ctrl 0/3, tumor 3/3' and table.loc[GLYCANS[7], 'Mean in tumor'] > 0


def test_compare_three_groups_and_errors():
    runs = [f'{g}{k}' for g in 'abc' for k in range(3)]
    df = feature_table(runs, {'c0': 5, 'c1': 5, 'c2': 5})
    result = compare_groups(df, {run: run[0] for run in runs})
    table = result['table'].set_index('Glycan')
    assert result['groups'] == ['a', 'b', 'c'] and 'ANOVA' in result['test'] and table.loc[GLYCANS[0], 'significant']
    # Tukey's tests for the significant glycans
    posthoc = result['posthoc'][GLYCANS[0]]
    assert len(posthoc) == 3 and posthoc.set_index(['group1', 'group2']).loc[('a', 'c'), 'reject']
    # runs without a group are left out
    assert sum(map(len, compare_groups(df, {**{run: run[0] for run in runs}, 'a0': ''})['members'].values())) == 8
    with pytest.raises(ValueError, match = 'two groups'):
        compare_groups(df, {run: 'all' for run in runs})
    with pytest.raises(ValueError, match = 'at least two runs'):
        compare_groups(df, {'a0': 'a', 'a1': 'a', 'b0': 'b'})
    with pytest.raises(ValueError, match = 'Pairs only work'):
        compare_groups(df, {run: run[0] for run in runs}, pairs = {run: run[1] for run in runs})
    with pytest.raises(ValueError, match = 'every pair'):
        compare_groups(df, {'a0': 'a', 'a1': 'a', 'b0': 'b', 'b1': 'b'}, pairs = {'a0': '1', 'a1': '2', 'b0': '1'})


def test_gui_compare_groups():
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from candycrunch import gui
    runs = ['1_ctrl_R1', '2_ctrl_R2', '3_ctrl_R3', '4_tumor_R1', '5_tumor_R2', '6_tumor_R3']
    features = feature_table(runs, {run: 4 for run in runs[3:]}).assign(curation = '', excluded = False)
    features['_top1'] = features['top1_pred']
    session = {'experiments': [{'name': 'Experiment 1', 'files': [f'{run}.mzML' for run in runs], 'settings': dict(gui.DEFAULTS),
                                'tables': {run: (pd.DataFrame(), []) for run in runs}, 'features': features},
                               # too few runs to compare
                               {'name': 'Pilot', 'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (pd.DataFrame(), [])}, 'features': None}]}
    tab = gui.CompareTab(gui.GlycanImages())
    changes = []
    tab.changed.connect(lambda: changes.append(1))
    tab.set_session(session)
    assert tab.experiments.count() == 1 and tab.runs.rowCount() == 6 and not tab.go.isEnabled()
    tab.guess_groups()
    assert [tab.runs.item(r, 1).text() for r in range(6)] == ['ctrl'] * 3 + ['tumor'] * 3 and tab.go.isEnabled() and changes
    assert session['experiments'][0]['comparison']['groups'] == {run: run.split('_')[1] for run in runs}
    tab.run()
    start = time.time()
    while tab.busy is not None and time.time() - start < 120:
        app.processEvents()
        time.sleep(0.05)
    result = session['experiments'][0]['comparison']['result']
    assert result is not None and tab.model.rowCount() == len(result['table']) and 'differ significantly' in tab.summary.text()
    start = time.time()
    while not tab.volcano.figure.axes[0].artists and time.time() - start < 60:
        app.processEvents()
        time.sleep(0.05)
    assert tab.volcano.figure.axes[0].artists and not tab.cartoon_wanted
    tab.tabs.setCurrentWidget(tab.by_group)
    tab.select_row(0)
    assert not tab.tabs.isTabVisible(2) and tab.row == 0
    # saved results with the comparison open with the restricted unpickler
    payload = pickle.loads(pickle.dumps(session))
    reopened = gui._SessionUnpickler(io.BytesIO(pickle.dumps(session))).load()
    assert reopened['experiments'][0]['comparison']['result']['table'].equals(payload['experiments'][0]['comparison']['result']['table'])
    # curation after the comparison is pointed out
    assert not tab.stale(result)
    features.loc[3, 'excluded'] = True
    assert tab.stale(result)
    # new results of the same runs keep their groups
    tab.set_session({'experiments': [{k: v for k, v in session['experiments'][0].items() if k != 'comparison'}]})
    assert tab.groups()[0] == {run: run.split('_')[1] for run in runs} and tab.result is None
    tab.shutdown()

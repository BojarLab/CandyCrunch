import inspect
import os
import pandas as pd
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QtWidgets = pytest.importorskip('PySide6.QtWidgets')


@pytest.fixture(scope = 'module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_settings_match_wrap_inference(app):
    from candycrunch import gui
    from candycrunch.prediction import wrap_inference, wrap_inference_batch
    panel = gui.SettingsPanel()
    values = panel.values()
    params = inspect.signature(wrap_inference).parameters
    # Every setting the app sends must be an argument of wrap_inference(_batch) (the worker turns model_file into model and glycans), and the app's
    # defaults must be the library's
    assert set(values) - set(gui.BATCH_KEYS) - {'model_file'} <= set(params)
    assert set(gui.BATCH_KEYS) <= set(inspect.signature(wrap_inference_batch).parameters)
    assert {k: v for k, v in values.items() if k in params and params[k].default is not inspect.Parameter.empty} == {
        k: params[k].default for k in values if k in params and params[k].default is not inspect.Parameter.empty}
    panel.set_values({**values, 'max_charge': 2, 'modification': 'custom', 'mass_tag': 100.5})
    assert panel.values()['max_charge'] == 2 and panel.values()['mass_tag'] == 100.5
    # the sialic acid derivatization is no combo text: None stays None, as wrap_inference's default
    panel.set_values({**values, 'sialic_derivatization': 'EEA'})
    assert panel.values()['sialic_derivatization'] == 'EEA'
    panel.set_values(values)
    assert panel.values()['sialic_derivatization'] is None


def test_gui_results_curation(app):
    from candycrunch import gui
    df = pd.DataFrame({'top1_pred': ['Gal(b1-3)GalNAc', 'Fuc(a1-2)Gal(b1-3)GalNAc'],
                       'predictions': [[('Gal(b1-3)GalNAc', 0.9), ('Gal(b1-4)GlcNAc', 0.05)], [('Fuc(a1-2)Gal(b1-3)GalNAc', 0.8)]],
                       'composition': [{'Hex': 1, 'HexNAc': 1}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1}], 'num_spectra': [3, 5], 'charge': [-1, -1],
                       'RT': [14.3, 26.8], 'rel_abundance': [40.0, 60.0], 'evidence': ['strong', 'strong'], 'notes': ['', ''],
                       'ppm_error': [10.0, 20.0], 'GlyTouCan_ID': ['', '']}, index = pd.Index([384.15, 530.2], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.set_results({'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (df, [{204.09: 10.0, 384.15: 100.0}, {}])}, 'features': None})
    assert view.model.rowCount() == 2
    view.filter('dHex')
    assert view.proxy.rowCount() == 1
    view.filter('')
    view.select_row(0)
    # A hovered supporting peak must leave the glycan map's selection marker alone
    view.spectrum.support_hovered.emit(('Gal(b1-3)GalNAc', (0,), None))
    view.show_row(1)
    view.show_row(0)
    view.curate('Gal(b1-4)GlcNAc', 'reassigned')
    view.select_row(1)
    view.toggle_excluded()
    out = view.export_frame(view.current_frame())
    assert out['top1_pred'].tolist() == ['Gal(b1-4)GlcNAc'] and out['curation'].tolist() == ['reassigned']


def test_gui_ms3(app):
    from candycrunch import gui
    df = pd.DataFrame({'top1_pred': ['Fuc(a1-2)Gal(b1-3)GalNAc'], 'predictions': [[('Fuc(a1-2)Gal(b1-3)GalNAc', 0.8)]],
                       'composition': [{'Hex': 1, 'HexNAc': 1, 'dHex': 1}],
                       'num_spectra': [5], 'charge': [-1], 'RT': [26.8], 'rel_abundance': [100.0],
                       'evidence': ['strong'], 'notes': [''], 'ppm_error': [20.0],
                       'GlyTouCan_ID': [''], 'ms3': [
            [(384.15, {204.09: 10.0, 222.1: 50.0}), (384.2, {222.0: 30.0}), (325.1, {163.06: 5.0})]]},
                      index = pd.Index([530.2], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.set_results(
        {'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (df, [{384.15: 100.0, 325.1: 40.0}])},
         'features': None})
    view.select_row(0)
    # MS3 spectra are pooled per isolated fragment, merging peaks within half the mass tolerance
    assert [(round(p, 3), len(peaks), n) for p, peaks, n in view.ms3_groups] == [(325.1, 1, 1), (384.175, 2, 2)]
    assert view.tabs.isTabVisible(view.tabs.indexOf(view.ms3_tab))
    view.open_ms3(1)
    assert view.tabs.currentWidget() is view.ms3_tab and view.ms3_panel.request['kwargs']['ms3_precursor'] == \
           view.ms3_groups[1][0]
    assert 'ms3' not in view.export_frame(view.current_frame()).columns


def test_gui_quantification(app):
    from candycrunch import gui
    import numpy as np
    times = np.arange(10, 14, 0.05, dtype = np.float32)
    def record(mz, z, apex, height):
        trace = (height * np.exp(-0.5 * ((times - apex) / 0.1) ** 2)).astype(np.float32)
        left, right = int(np.searchsorted(times, apex - 0.4)), int(np.searchsorted(times, apex + 0.4))
        return {'mz': mz, 'charge': z, 'rt': apex, 'times': times, 'trace': trace, 'left': left, 'apex': int(np.searchsorted(times, apex)), 'right': right,
                'baseline': 0.0, 'area': float(np.sum((trace[left + 1:right + 1] + trace[left:right]) * np.diff(times[left:right + 1])) / 2), 'others': [], 'scale': 1.0,
                'windows': [(mz - 0.25 + i / z, mz + 0.25 + i / z) for i in range(4)], 'scan': (np.array([mz, mz + 1 / z]), np.array([height, height / 2]))}
    # Run a: the first peak sums two integrations (its ion and a charge state folded into it); run b has no record (an older session)
    xic = [[record(530.2, 1, 12.0, 3e5), record(264.6, 2, 12.0, 1e5)], [record(384.15, 1, 11.0, 1e5)]]
    raw = [sum(d['area'] for d in x) for x in xic]
    tables = {}
    for run, values in (('a', [r / sum(raw) * 100 for r in raw]), ('b', [70.0, 30.0])):
        tables[run] = (pd.DataFrame({'top1_pred': ['Fuc(a1-2)Gal(b1-3)GalNAc', 'Gal(b1-3)GalNAc'], 'predictions': [[('Fuc(a1-2)Gal(b1-3)GalNAc', 0.8)], [('Gal(b1-3)GalNAc', 0.9)]],
                                     'composition': [{'Hex': 1, 'HexNAc': 1, 'dHex': 1}, {'Hex': 1, 'HexNAc': 1}], 'num_spectra': [5, 3], 'charge': [-1, -1], 'RT': [12.0, 11.0],
                                     'rel_abundance': values, 'evidence': ['strong', 'strong'], 'notes': ['', ''], 'ppm_error': [20.0, 10.0], 'GlyTouCan_ID': ['', '']},
                                    index = pd.Index([530.2, 384.15], name = 'm/z')).assign(**({'xic': xic} if run == 'a' else {})), [{384.15: 100.0}, {204.09: 50.0}])
    features = pd.DataFrame({'top1_pred': ['Gal(b1-3)GalNAc', 'Fuc(a1-2)Gal(b1-3)GalNAc'], 'm/z': [384.15, 530.2], 'RT': [11.0, 12.0], 'charge': [-1, -1],
                             'composition': [{'Hex': 1, 'HexNAc': 1}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1}], 'GlyTouCan_ID': ['', ''], 'n_files_ms2': [2, 2],
                             'a': [tables['a'][0]['rel_abundance'].iat[1], tables['a'][0]['rel_abundance'].iat[0]], 'b': [30.0, 70.0],
                             'evidence_a': ['strong', 'strong'], 'evidence_b': ['strong', 'ms1_only']})
    view = gui.ResultsView(gui.GlycanImages())
    view.resize(1200, 800)
    view.show()
    view.set_results({'files': ['a.mzML', 'b.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': tables, 'features': features})
    view.tabs.setCurrentWidget(view.quant)
    view.select_row(view.proxy.mapToSource(view.proxy.index(1, 0)).row())
    # A feature's integration in every run with it, MS1 signal only too; the run whose spectrum is shown comes first
    assert view.frame['top1_pred'].iat[view.row] == 'Fuc(a1-2)Gal(b1-3)GalNAc'
    assert [view.quant.run.itemText(i) for i in range(view.quant.run.count())] == ['a', 'b (MS1 signal only)'] and view.quant_source == ('a', 0)
    assert view.quant.ion.count() == 2 and view.quant.ion.isVisibleTo(view.quant) and '75%' in view.quant.summary.text()
    a, before = tables['a'][0], tables['a'][0]['rel_abundance'].copy()
    # Half the peak's range: the run's abundances stay percentages, and the feature table's column of the run follows
    view.quant.select(0, 11.6, 12.0)
    assert np.isclose(a['rel_abundance'].sum(), 100) and a['rel_abundance'].iat[0] < before.iat[0] and np.isclose(features['a'].sum(), 100)
    assert np.isclose(features.loc[features['top1_pred'] == 'Fuc(a1-2)Gal(b1-3)GalNAc', 'a'].iat[0], a['rel_abundance'].iat[0])
    assert a['curation'].iat[0] == 'area edited' and features['curation'].iat[1] == 'area edited' and view.quant.automatic.isVisibleTo(view.quant) and view.dirty
    assert 'xic' not in view.export_frame(a).columns
    # A structure assigned keeps the note; CandyCrunch's range again restores every value
    view.curate('Fuc(a1-2)Gal(b1-4)GlcNAc', 'reassigned')
    assert a['curation'].iat[0] == 'reassigned; area edited'
    view.quant.automatic.click()
    assert np.allclose(a['rel_abundance'], before) and a['curation'].iat[0] == 'reassigned' and 'auto' not in a['xic'].iat[0][0]
    # Results without records (older sessions) and runs without survey scans say why there is nothing to show
    view.quant.run.setCurrentIndex(1)
    assert 'earlier CandyCrunch' in view.quant.message.text()
    view.quant.show([], 50.0)
    assert 'survey scans' in view.quant.message.text()


def test_gui_experiments(app):
    from candycrunch import gui
    panel = gui.SettingsPanel()
    panel.add_paths([os.path.abspath(f'{name}.mzML') for name in ('a', 'b', 'c')])
    gui._set_combo(panel.inputs['glycan_class'], 'O')
    panel.move_runs([panel.files.topLevelItem(0).child(2)], panel.add_experiment(panel.values()))
    gui._set_combo(panel.inputs['glycan_class'], 'N')
    # Every experiment keeps its own settings, and only its own runs are harmonized
    assert [(name, len(files), settings['glycan_class']) for name, files, settings in panel.experiments()] == [('Experiment 1', 2, 'O'), ('Experiment 2', 1, 'N')]
    panel.files.setCurrentItem(panel.files.topLevelItem(0))
    assert panel.inputs['glycan_class'].currentData() == 'O' and panel.batch.isVisibleTo(panel)
    # Moving the last run out of an experiment removes it
    panel.move_runs([panel.files.topLevelItem(1).child(0)], panel.files.topLevelItem(0))
    assert [name for name, _, _ in panel.experiments()] == ['Experiment 1'] and len(panel.paths()) == 3


def test_gui_glycan_class_list(app):
    from candycrunch import gui
    panel = gui.SettingsPanel()
    # N- and O-glycans measured together are one choice, sent to wrap_inference as a list; lists from sessions come back in any order
    panel.set_values({**panel.values(), 'glycan_class': ['O', 'N']})
    assert panel.inputs['glycan_class'].currentData() == 'N+O' and panel.values()['glycan_class'] == ['N', 'O']
    # A list the combo does not offer becomes a choice of its own, and a list of one class is that class
    panel.set_values({**panel.values(), 'glycan_class': ['free', 'O']})
    assert panel.values()['glycan_class'] == ['O', 'free'] and panel.inputs['glycan_class'].currentText() == 'O-glycans and Free oligosaccharides (e.g., milk)'
    panel.set_values({**panel.values(), 'glycan_class': ['lipid']})
    assert panel.values()['glycan_class'] == 'lipid'
    panel.add_paths([os.path.abspath('a.mzML')], detect = False)
    panel.set_values({**panel.values(), 'glycan_class': ['N', 'O']})
    assert panel.experiments()[0][2]['glycan_class'] == ['N', 'O']


def test_gui_supporting_ions(app):
    import time
    from candycrunch import gui
    panel = gui.SpectrumPanel()
    panel.annotate('Fuc(a1-2)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc', [510.19, 715.27], [30.0, 100.0], -1, 2.0156, 'underivatized',
                   candidates = ['Fuc(a1-2)Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc'])
    end = time.time() + 120
    while 'Supported by' not in panel.fragments.toPlainText() and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    # The supporting peaks sit above the fragment table, which is kept, and their count goes into the summary line
    text = panel.fragments.toPlainText()
    assert 'Fuc(a1-2) on Gal(b1-4)GlcNAc' in text and '510.19' in text and 'vs Fuc(a1-2)Gal(b1-3)' in text and 'Observed m/z' in text, panel.message.text() or panel.summary.text()
    assert 'residues placed by diagnostic fragments' in panel.summary.text()


def test_gui_support_hover(app):
    import time
    from matplotlib.backend_bases import MouseEvent
    from candycrunch import gui
    panel, hovered = gui.SpectrumPanel(), []
    panel.support_hovered.connect(hovered.append)
    g = 'Fuc(a1-2)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc'
    panel.annotate(g, [510.19, 715.27], [30.0, 100.0], -1, 2.0156, 'underivatized')
    end = time.time() + 120
    while not panel.hover_targets and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    # The mouse on the lilac marker of the B3 ion highlights its residues (Fuc, Gal, GlcNAc of the 6-arm) and the cleaved GlcNAc(b1-6) bond, and moving
    # off it clears the highlight
    assert panel.hover_targets, panel.message.text() or panel.summary.text()
    artist = panel.hover_targets[0][0]
    x, y = artist.axes.transData.transform(artist.get_offsets()[list(artist.get_offsets()[:, 0]).index(510.19)])
    panel.hover(MouseEvent('motion_notify_event', panel.canvas, x, y))
    panel.hover(MouseEvent('motion_notify_event', panel.canvas, 1, 1))
    assert hovered == [(g, (0, 1, 2), 2), None]
    # The drawing with that highlight renders
    images = gui.GlycanImages()
    while images.get(g, compact = False, highlight = ((0, 1, 2), 2)) is None and time.time() < end:
        time.sleep(0.05)
    assert not images.get(g, compact = False, highlight = ((0, 1, 2), 2)).isNull()


def test_gui_label_tooltip_and_reset(app, monkeypatch):
    import time
    from matplotlib.backend_bases import MouseEvent
    from candycrunch import gui
    # Restore defaults is only enabled while a setting differs from its default, and keeps the glycan class
    settings = gui.SettingsPanel()
    assert not settings.reset.isEnabled()
    settings.inputs['glycan_class'].setCurrentIndex(1)
    settings.inputs['ppm_thresh'].setValue(100)
    assert settings.reset.isEnabled()
    settings.reset.click()
    assert not settings.reset.isEnabled() and settings.values()['glycan_class'] == 'N' and settings.values()['ppm_thresh'] == gui.DEFAULTS['ppm_thresh']
    # Hovering a fragment label shows the observed and theoretical m/z of its peak
    panel, shown, end = gui.SpectrumPanel(), [], time.time() + 120
    panel.annotate('Fuc(a1-2)Gal(b1-3)GalNAc', [384.15, 325.1, 222.1, 530.2], [100.0, 40.0, 30.0, 80.0], -1, None, 'underivatized')
    while not panel.label_targets and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    monkeypatch.setattr(gui.QToolTip, 'showText', lambda pos, text, *args: shown.append(text))
    assert panel.label_targets, panel.message.text() or panel.summary.text()
    text = next(t for t, _ in panel.label_targets if t.xy[0] == 384.15)
    box = text.get_window_extent()
    panel.hover(MouseEvent('motion_notify_event', panel.canvas, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
    assert 'Observed m/z <b>384.1500</b>' in shown[-1] and 'theoretical m/z <b>384.1511</b>' in shown[-1]


def test_gui_ms3_hover_and_menu(app, monkeypatch):
    import time
    from matplotlib.backend_bases import MouseEvent
    from candycrunch import gui
    df = pd.DataFrame({'top1_pred': ['Fuc(a1-2)Gal(b1-3)GalNAc', 'Gal(b1-3)GalNAc'], 'predictions': [[('Fuc(a1-2)Gal(b1-3)GalNAc', 0.8)], [('Gal(b1-3)GalNAc', 0.9)]],
                       'composition': [{'Hex': 1, 'HexNAc': 1, 'dHex': 1}, {'Hex': 1, 'HexNAc': 1}], 'num_spectra': [5, 3], 'charge': [-1, -1], 'RT': [26.8, 14.0],
                       'rel_abundance': [60.0, 40.0], 'evidence': ['strong', 'strong'], 'notes': ['', ''], 'ppm_error': [20.0, 5.0], 'GlyTouCan_ID': ['', ''],
                       'ms3': [[(384.15, {204.09: 10.0, 222.1: 50.0}), (384.2, {222.0: 30.0})], []]}, index = pd.Index([530.2, 384.15], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.resize(1400, 900)
    view.show()
    view.set_results({'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (df, [{384.15: 100.0, 325.1: 40.0}, {204.09: 100.0}])}, 'features': None})
    view.select_row(0)
    panel, shown, end = view.spectrum, [], time.time() + 120
    while not panel.hover_targets and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    # Hovering the MS3 marker names the isolated fragment and what clicking does, with a hand cursor that goes away off the marker
    monkeypatch.setattr(gui.QToolTip, 'showText', lambda pos, text, *args: shown.append(text))
    assert panel.hover_targets, panel.message.text() or panel.summary.text()
    artist = panel.canvas.figure.ms3_artists[0]
    x, y = artist.axes.transData.transform(artist.get_offsets()[0])
    panel.hover(MouseEvent('motion_notify_event', panel.canvas, x, y))
    assert 'isolated for MS3: 2 spectra' in shown[-1] and panel.canvas.cursor().shape() == gui.Qt.PointingHandCursor
    panel.hover(MouseEvent('motion_notify_event', panel.canvas, x + 300, y + 300))
    assert panel.canvas.cursor().shape() == gui.Qt.ArrowCursor
    # Open in CandyCrumbs from the MS3 tab sends the MS3 spectrum shown with its precursor, from the MS2 tab the MS2 spectrum
    sent, tab = [], gui.CrumbsTab(gui.GlycanImages())
    view.open_in_crumbs.connect(sent.append)
    view.open_ms3(0)
    view.send_to_crumbs()
    assert sent[-1]['ms3_precursor'] == view.ms3_groups[0][0] and sent[-1]['peaks'] == view.ms3_groups[0][1]
    tab.load(sent[-1])
    assert abs(tab.spectrum.request['kwargs']['ms3_precursor'] - view.ms3_groups[0][0]) < 1e-4
    view.tabs.setCurrentIndex(0)
    view.send_to_crumbs()
    tab.load(sent[-1])
    assert sent[-1]['ms3_precursor'] is None and 'ms3_precursor' not in tab.spectrum.request['kwargs']
    # Right-clicking a row that is not selected selects it, so the menu acts on it
    monkeypatch.setattr(gui, 'QMenu', type('Menu', (), {'__init__': lambda self, *args: None, 'addAction': lambda self, *args: None, 'exec': lambda self, *args: None}))
    proxy_row = next(i for i in range(view.proxy.rowCount()) if view.proxy.mapToSource(view.proxy.index(i, 0)).row() == 1)
    view.context_menu(view.table.visualRect(view.proxy.index(proxy_row, 2)).center())
    assert view.row == 1


def test_gui_example_run(app):
    from candycrunch import gui
    panel = gui.SettingsPanel()
    gui._set_combo(panel.inputs['glycan_class'], 'N')
    panel.add_example()
    panel.add_example()
    # The bundled example is added once, as an experiment of its own with O-glycan defaults
    assert [(name, files, settings['glycan_class']) for name, files, settings in panel.experiments()] == [
        ('Example', [gui.EXAMPLE_RUN], 'O')]
    assert panel.experiments()[0][2] == {**panel.values(), 'glycan_class': 'O'}
    assert {'m/z', 'peak_d', 'RT', 'precursor_charge', 'intensity', 'mode', 'trap'} <= set(
        pd.read_excel(gui.EXAMPLE_RUN).columns)


def test_gui_filter_refresh_and_pasted_peaks(app):
    from candycrunch import gui
    df = pd.DataFrame({'top1_pred': ['Gal(b1-3)GalNAc', 'Fuc(a1-2)Gal(b1-3)GalNAc'], 'predictions': [[('Gal(b1-3)GalNAc', 0.9)], [('Fuc(a1-2)Gal(b1-3)GalNAc', 0.8)]],
                       'composition': [{'Hex': 1, 'HexNAc': 1}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1}], 'num_spectra': [3, 5], 'charge': [-1, -1], 'RT': [14.3, 26.8],
                       'rel_abundance': [40.0, 60.0], 'evidence': ['strong', 'strong'], 'notes': ['', ''], 'ppm_error': [10.0, 20.0], 'GlyTouCan_ID': ['', '']},
                      index = pd.Index([384.15, 530.2], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.set_results({'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (df, [{204.09: 10.0}, {384.15: 100.0}])}, 'features': None})
    view.select_row(0)
    # A filter that hides every peak leaves none in the detail pane for Exclude to act on; the first match is shown once peaks match again
    view.search.setText('no such glycan')
    view.toggle_excluded()
    assert view.row is None and not view.frame['excluded'].any() and view.search.text() == 'no such glycan'
    view.search.setText('dHex')
    assert view.row == 1
    # A refresh after curation keeps column widths set by hand
    view.table.setColumnWidth(1, 300)
    view.toggle_excluded()
    assert view.table.columnWidth(1) == 300 and view.frame['excluded'].tolist() == [False, True]
    # Lines that start with text (headers, the fields of a pasted mgf spectrum) are no peaks
    tab = gui.CrumbsTab(gui.GlycanImages())
    tab.structure.setText('Fuc(a1-2)Gal(b1-3)GalNAc')
    tab.peaks.setPlainText('BEGIN IONS\nTITLE=run.1234.1234.2\nPEPMASS=530.19 120000\nCHARGE=1-\nm/z\tintensity\n384.15\t100\n325,12;20\nEND IONS')
    tab.annotate()
    assert tab.spectrum.request['mzs'] == [384.15, 325.12] and tab.spectrum.request['intensities'] == [100.0, 20.0]


def test_gui_save_image(app, tmp_path, monkeypatch):
    import time
    from candycrunch import gui
    from candycrunch.analysis import plot_annotated_spectrum
    g, mzs, ints = 'Fuc(a1-2)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc', [510.19, 715.27, 990.0], [30.0, 100.0, 20.0]
    panel = gui.SpectrumPanel()
    panel.annotate(g, mzs, ints, -1, 2.0156, 'underivatized')
    end = time.time() + 120
    while not panel.save.isEnabled() and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    # Drawn again at print size (12 inches at 300 dpi), with the file type of the chosen filter when the name has none
    assert panel.save.isEnabled(), panel.message.text() or panel.summary.text()
    monkeypatch.setattr(gui.QFileDialog, 'getSaveFileName', lambda *args: (str(tmp_path / 'spectrum'), 'PNG image, 300 dpi (*.png)'))
    panel.save_image()
    assert gui.QImage(str(tmp_path / 'spectrum.png')).width() > 3000
    # Its labels give the observed m/z of annotated peaks, and an unannotated peak gets its m/z in grey
    _, ax = plot_annotated_spectrum(g, mzs, ints, charge = -1, mass_tag = 2.0156, mz_labels = True)
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith('510.19 ') for t in texts) and '990.00' in texts


def test_gui_unreduced_and_detection(app):
    import multiprocessing as mp
    import time
    from candycrunch import gui
    panel = gui.SettingsPanel()
    # The pipeline takes a free reducing end as a custom label without mass, and settings saved that way (also as mass 0) show as unreduced
    gui._set_combo(panel.inputs['modification'], 'unreduced')
    assert (panel.values()['modification'], panel.values()['mass_tag']) == ('custom', None) and not panel.sample_form.isRowVisible(panel.inputs['mass_tag'])
    panel.set_values({**panel.values(), 'mass_tag': 0.0})
    assert panel.inputs['modification'].currentData() == 'unreduced'
    results = mp.get_context('spawn').Queue()
    gui._detect_run(gui.EXAMPLE_RUN, results)
    path, found, charges, n_ms2 = results.get(timeout = 10)
    assert found == {'mode': 'negative', 'trap': 'linear', 'glycan_class': 'O'} and max(charges) == 2 and n_ms2 == 275
    # Found settings are set where the user has not changed them since adding the run (here the chromatography)
    experiment = panel.add_experiment({**panel.values(), 'glycan_class': 'N', 'max_charge': 4, 'lc': 'C18'})
    experiment.detecting = (path, {**experiment.settings, 'lc': 'PGC'}, time.time())
    panel.detections.put((path, {**found, 'lc': 'PGC'}, charges, n_ms2))
    end = time.time() + 10
    while panel.waiting() and time.time() < end:
        panel.receive_detections()
        time.sleep(0.05)
    assert (experiment.settings['glycan_class'], experiment.settings['max_charge'], experiment.settings['lc']) == ('O', -3, 'C18') and 'O-glycans' in experiment.note
    experiment = panel.add_experiment({**panel.values(), 'max_charge': -3})
    experiment.detecting = (path, dict(experiment.settings), time.time())
    panel.detections.put((path, {'mode': 'negative'}, {2: 300, 4: 60, 6: 70}, 1000))
    end = time.time() + 10
    while panel.waiting() and time.time() < end:
        panel.receive_detections()
        time.sleep(0.05)
    assert experiment.settings['max_charge'] == -3


def test_gui_reports(app, monkeypatch):
    import json
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from candycrunch import gui
    posts = []
    class Sentry(BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append((self.path, self.headers['X-Sentry-Auth'], self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Sentry)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    monkeypatch.setattr(gui, 'REPORT_DSN', f'http://key@127.0.0.1:{server.server_port}/7')
    home = os.path.expanduser('~')
    try:
        try:
            {}['run']
        except KeyError as error:
            raise ValueError(os.path.join(home, 'run.raw')) from error
    except ValueError as error:
        values = gui._exception_values(error)
    # The cause comes first, and frames name their file, never the folder it sits in
    assert [v['type'] for v in values] == ['KeyError', 'ValueError'] and values[-1]['stacktrace']['frames'][-1]['filename'] == 'test_gui.py'
    event = {'event_id': 'a' * 32, 'level': 'error', 'exception': {'values': values}}
    assert gui._send_report(event, {'log.txt': f'Reading {home}{os.sep}run.raw\nDone', 'empty.txt': ''})
    gui._send_session({'sid': 's', 'started': '2026-10-16T05:00:00Z', 'status': 'exited', 'errors': 1, 'attrs': {'release': 'candycrunch@1.0.0'}}).join(10)
    # A data file goes as bytes, untouched
    assert gui._send_report({'event_id': 'b' * 32}, {'peak_data.cc': b'\x00\x01cc'})
    server.shutdown()
    # A session update is an envelope of one 'session' item, stamped with the time it is sent
    header, item, payload = posts[1][2].split(b'\n')
    assert 'event_id' not in json.loads(header) and json.loads(item)['type'] == 'session' and json.loads(payload)['status'] == 'exited' and 'timestamp' in json.loads(payload)
    assert b'"content_type": "application/octet-stream"' in posts[2][2] and posts[2][2].endswith(b'\x00\x01cc')
    path, auth, body = posts[0]
    assert path == '/api/7/envelope/' and 'sentry_key=key' in auth
    header, rest = body.split(b'\n', 1)
    items = []
    while rest:
        item, rest = rest.split(b'\n', 1)
        item = json.loads(item)
        items.append((item, rest[:item['length']]))
        rest = rest[item['length'] + 1:]
    assert json.loads(header)['event_id'] == 'a' * 32 and [(i['type'], i.get('filename')) for i, _ in items] == [('event', None), ('attachment', 'log.txt')]
    # Paths in the home folder are shortened to ~, also inside the JSON
    assert home.encode() not in body and json.loads(items[0][1])['exception']['values'][-1]['value'] == os.path.join('~', 'run.raw')
    assert items[1][1].decode() == f'Reading ~{os.sep}run.raw\nDone'
    # A bug report needs a description before it can be sent
    dialog = gui.ReportDialog(None, event, {'log.txt': 'Done'})
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    send = next(b for b in box.buttons() if box.buttonRole(b) == QtWidgets.QDialogButtonBox.AcceptRole)
    assert not send.isEnabled()
    dialog.text.setPlainText('The spectrum is empty')
    assert send.isEnabled() and dialog.automatic.isHidden()
    # A Windows home with a non-ASCII user name, in an error message that quotes the path (repr), still loses the user name inside the JSON
    windows_home = 'C:\\Users\\J\u00fcrgen'
    monkeypatch.setattr(os.path, 'expanduser', lambda path: windows_home if path == '~' else path)
    try:
        raise PermissionError(13, 'Permission denied', windows_home + '\\run_output.xlsx')
    except PermissionError as error:
        values = gui._exception_values(error)
    assert 'rgen' not in gui._scrub(json.dumps({'exception': {'values': values}})) and 'rgen' not in gui._scrub(values[-1]['value'])


def test_gui_models_and_misprediction_report(app, tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    import time
    import torch
    from candycrunch import gui
    from candycrunch.model import CandyCrunch_CNN
    import numpy as np
    import candycrunch.prediction as cp
    from candycrunch.prediction import glycans
    # Weights saved with their own vocabulary are a model; weights that do not fit the built-in vocabulary, or a file of anything else, are refused with why
    torch.save({'state_dict': CandyCrunch_CNN(2048, num_classes = len(glycans) + 1, input_precursor_dim = 12).state_dict(), 'glycans': glycans + ['Gal(b1-3)GalNAc6S']},
               tmp_path / 'tuned.pt')
    torch.save(CandyCrunch_CNN(2048, num_classes = 3, input_precursor_dim = 12).state_dict(), tmp_path / 'small.pt')
    (tmp_path / 'notes.pt').write_text('not a model')
    assert gui._check_model(str(tmp_path / 'tuned.pt')) == len(glycans) + 1
    for name, reason in (('small.pt', 'no CandyCrunch model'), ('notes.pt', 'no PyTorch weights file')):
        with pytest.raises(ValueError, match = reason):
            gui._load_model(str(tmp_path / name))
    # The same without PyTorch, as in the built app, where the model runs on ONNX Runtime
    if importlib.util.find_spec('onnxruntime'):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, 'torch', None)
            assert gui._check_model(str(tmp_path / 'tuned.pt')) == len(glycans) + 1
            assert type(gui._load_model(str(tmp_path / 'tuned.pt'))[0]).__name__ == 'CandyCrunchCNN'
            for name, reason in (('small.pt', 'no CandyCrunch model'), ('notes.pt', 'no PyTorch weights file')):
                with pytest.raises(ValueError, match = reason):
                    gui._load_model(str(tmp_path / name))
    # Importing checks the file in a process of its own, then copies it into the model folder and chooses it; a session's model that no longer exists
    # falls back to the built-in one
    panel = gui.SettingsPanel()
    panel.model_dir = str(tmp_path / 'models')
    monkeypatch.setattr(gui.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(tmp_path / 'tuned.pt'), ''))
    for _ in range(2):
        panel.import_model()
        end = time.time() + 120
        while not panel.import_model_button.isEnabled() and time.time() < end:
            app.processEvents()
            time.sleep(0.05)
    assert panel.values()['model_file'] == os.path.join(str(tmp_path / 'models'), 'tuned_2.pt') and os.path.exists(panel.values()['model_file'])
    # The model choice is an advanced setting, opened by an import so the chosen model is in view
    assert panel.reset.isEnabled() and panel.advanced.isAncestorOf(panel.model) and panel.advanced_toggle.isChecked()
    panel.set_values({**panel.values(), 'model_file': str(tmp_path / 'gone.pt')})
    assert panel.values()['model_file'] is None
    # A misprediction report holds the peak shown (with the structure the user assigned as the likely right one), but not the run's name
    df = pd.DataFrame({'top1_pred': ['Gal(b1-3)GalNAc'], 'predictions': [[('Gal(b1-3)GalNAc', 0.9), ('Gal(b1-4)GlcNAc', 0.05)]],
                       'composition': [{'Hex': 1, 'HexNAc': 1}], 'num_spectra': [3], 'charge': [-1], 'RT': [14.3], 'rel_abundance': [100.0],
                       'evidence': ['strong'], 'notes': [''], 'ppm_error': [10.0], 'GlyTouCan_ID': ['']}, index = pd.Index([384.15], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.set_results({'files': ['/data/patient_17.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'patient_17': (df, [{204.09: 10.0, 384.15: 100.0}])},
                      'features': None})
    view.select_row(0)
    assert view.report_button.isEnabled()
    view.curate('Gal(b1-4)GlcNAc', 'reassigned')
    reports = []
    view.report_peak.connect(reports.append)
    view.report_button.click()
    peak = reports[0]
    assert peak['predicted'] == 'Gal(b1-3)GalNAc' and peak['peak']['top1_pred'] == 'Gal(b1-4)GlcNAc' and peak['ms2'] == {204.09: 10.0, 384.15: 100.0}
    assert peak['run_format'] == '.mzml' and 'patient_17' not in json.dumps(peak, default = str)
    # The run's file and its table of peaks stay out of the report itself; with the user's consent, the peak's surroundings are cut out of the file
    assert view.report_data['path'] == '/data/patient_17.mzML' and view.report_data['charge'] == -1 and 'Gal(b1-4)GlcNAc' in view.report_data['table']
    (tmp_path / 'peak.cc').write_bytes(gui._extract_peak_data(gui.EXAMPLE_RUN, 530.29, -1, 26.76))
    with np.load(tmp_path / 'peak.cc') as file:
        meta = json.loads(file['meta'].tobytes())
    spectra = cp.load_spectra_filepath(str(tmp_path / 'peak.cc'))
    assert meta['source'] == 'run.xlsx' and meta['region']['run_ms2'] == 275 and 0 < len(spectra) < 275 and (abs(spectra['RT'] - 26.76) <= 3).all()
    dialog = gui.ReportDialog(None, {'event_id': 'a' * 32}, {'misprediction.json': json.dumps(peak, default = str)}, peak = peak)
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    send = next(b for b in box.buttons() if box.buttonRole(b) == QtWidgets.QDialogButtonBox.AcceptRole)
    assert not send.isEnabled() and dialog.correct.text() == 'Gal(b1-4)GlcNAc' and dialog.include_data.isChecked()
    gui._set_combo(dialog.problem, 'structure')
    assert send.isEnabled()


def test_gui_progress_wrappers(monkeypatch):
    # The app's worker runs the pipeline through the progress wrappers _install_progress puts around prediction.py: a run through them must still predict
    # (get_topk's own batching once failed on a wrapped loader, which only the built app's self-test caught) and count get_topk's batches
    import queue
    from candycrunch import gui
    import candycrunch.prediction as cp
    # The wrappers stay on the module for the rest of the process; restoring its attributes keeps them out of later tests
    for name, value in vars(cp).copy().items():
        if not name.startswith('__'):
            monkeypatch.setattr(cp, name, value)
    monkeypatch.setattr(cp, '_progress_installed', False, raising = False)
    messages = queue.Queue()
    gui._install_progress(messages, gui.STAGE_WEIGHTS, 1)
    df = cp.wrap_inference(gui.EXAMPLE_RUN, 'O')
    stage, progress = None, []
    while not messages.empty():
        kind, payload = messages.get()
        stage = payload[0] if kind == 'stage' else stage
        if kind == 'progress' and stage == 'get_topk':
            progress.append(payload[1])
    # Each get_topk call reports its start; only counted batches move it on
    assert df['top1_pred'].notna().any() and progress and max(progress) > progress[0]
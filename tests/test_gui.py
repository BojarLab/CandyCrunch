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
    # Every setting the app sends must be an argument of wrap_inference(_batch), and the app's defaults must be the library's
    assert set(values) - set(gui.BATCH_KEYS) <= set(params)
    assert set(gui.BATCH_KEYS) <= set(inspect.signature(wrap_inference_batch).parameters)
    assert {k: v for k, v in values.items() if k in params and params[k].default is not inspect.Parameter.empty} == {
        k: params[k].default for k in values if k in params and params[k].default is not inspect.Parameter.empty}
    panel.set_values({**values, 'max_charge': 2, 'modification': 'custom', 'mass_tag': 100.5})
    assert panel.values()['max_charge'] == 2 and panel.values()['mass_tag'] == 100.5


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
    assert 'Fuc(a1-2) on Gal(b1-4)GlcNAc' in text and '510.19' in text and 'vs Fuc(a1-2)Gal(b1-3)' in text and 'Observed m/z' in text
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
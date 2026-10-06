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
    view.curate('Gal(b1-4)GlcNAc', 'reassigned')
    view.select_row(1)
    view.toggle_excluded()
    out = view.export_frame(view.current_frame())
    assert out['top1_pred'].tolist() == ['Gal(b1-4)GlcNAc'] and out['curation'].tolist() == ['reassigned']

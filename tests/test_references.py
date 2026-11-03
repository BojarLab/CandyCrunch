import os
import pandas as pd
import pytest
from candycrunch.references import DEAD_SOURCES, _linkages_agree, add_references, glycan_references
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def test_linkages_agree():
    assert _linkages_agree('Gal(b1-3/4)GlcNAc', 'Gal(b1-4)GlcNAc')
    assert _linkages_agree('Gal(?1-?)GlcNAc', 'Gal(b1-3)GlcNAc-ol')
    assert not _linkages_agree('Gal(b1-3)GlcNAc', 'Gal(b1-4)GlcNAc')
    assert not _linkages_agree('Gal(a1-3)GlcNAc', 'Gal(b1-3)GlcNAc')


def test_glycan_references():
    # The pipeline writes reduced glycans without -ol, the database with it
    exact, related = glycan_references('Gal(b1-3)GalNAc')
    assert exact and exact == glycan_references('Gal(b1-3)GalNAc-ol')[0]
    assert all(r['url'].startswith('http') and r['label'] and not any(s in r['url'] for s in DEAD_SOURCES) for r in exact + related)
    assert not {r['url'] for r in exact} & {r['url'] for r in related}
    human = [r['human'] for r in exact]
    assert human == sorted(human, reverse = True)
    assert all(len(r['label']) <= 30 for r in exact + related)
    # A structure with an unspecified linkage finds those of the structures it can be
    assert glycan_references('Gal(?1-3)GalNAc')[1]
    assert glycan_references('Gal(b1-3)Gal(b1-3)Gal(b1-3)Gal(b1-3)Gal(b1-3)GalNAc') == ([], [])


def test_add_references():
    df = pd.DataFrame({'top1_pred': ['Gal(b1-3)GalNAc', None], 'GlyTouCan_ID': ['', ''], 'RT': [1.0, 2.0]})
    out = add_references(df)
    assert out.columns.tolist() == ['top1_pred', 'GlyTouCan_ID', 'references', 'RT'] and 'references' not in df.columns
    assert out['references'].iat[0].split('; ') == [r['url'] for r in glycan_references('Gal(b1-3)GalNAc')[0]] and out['references'].iat[1] == ''
    assert add_references(df.drop(columns = ['top1_pred'])).columns.tolist() == ['GlyTouCan_ID', 'RT']


def test_gui_references():
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from candycrunch import gui, references
    # The details show references once the index is read, which the app does on a thread; another test may or may not have read it in this process
    references.reference_index()
    structure = 'Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc'
    df = pd.DataFrame({'top1_pred': [structure, 'Gal(b1-3)Gal(b1-3)Gal(b1-3)Gal(b1-3)Gal(b1-3)GalNAc'], 'predictions': [[], []],
                       'composition': [{'Hex': 1, 'HexNAc': 1, 'Neu5Ac': 2}, {'Hex': 5, 'HexNAc': 1}], 'num_spectra': [3, 5], 'charge': [-1, -1], 'RT': [14.3, 26.8],
                       'rel_abundance': [40.0, 60.0], 'evidence': ['strong', 'strong'], 'notes': ['', ''], 'ppm_error': [10.0, 20.0], 'GlyTouCan_ID': ['', '']},
                      index = pd.Index([675.2, 1000.3], name = 'm/z'))
    view = gui.ResultsView(gui.GlycanImages())
    view.set_results({'files': ['a.mzML'], 'settings': dict(gui.DEFAULTS), 'tables': {'a': (df, [{204.09: 10.0}, {204.09: 10.0}])}, 'features': None})
    view.show_row(0)
    n = len(glycan_references(structure)[0])
    # The list has a tab of its own, reached from the count in the facts, where it does not push the candidates down
    assert f'in glycowork ({n})' in view.references.toHtml() and view.references.toHtml().count('href="http') >= n
    assert f'<a href="references">{n}' in view.facts.text()
    view.open_fact_link('references')
    assert view.tabs.currentWidget() is view.references
    view.show_row(1)
    assert 'No references' in view.references.toPlainText() and 'href="references"' not in view.facts.text()
    view.select_row(None)
    assert view.references.toPlainText() == ''
    # A peak's details do not wait for the database: without it, no references yet, and the signal fills them in
    index, references.INDEX = references.INDEX, None
    view.show_row(0)
    assert 'href="references"' not in view.facts.text() and 'in glycowork' not in view.references.toPlainText()
    references.INDEX = index
    view.update_detail()
    assert 'href="references"' in view.facts.text()
    assert view.export_frame(view.current_frame())['references'].iat[0].count('http') == n

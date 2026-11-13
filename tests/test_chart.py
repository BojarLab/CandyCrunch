import os
import subprocess
import sys
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QtWidgets = pytest.importorskip('PySide6.QtWidgets')
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent


@pytest.fixture(scope = 'module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def chart(xs = (1.0, 2.0, 3.0), ys = (10.0, 20.0, 30.0)):
    from candycrunch.gui import ChartView, PALETTE
    view = ChartView(palette = PALETTE)
    view.resize(600, 300)
    view.canvas.resize(600, 300)
    ax = view.new_figure().add_axes([0.1, 0.2, 0.8, 0.7])
    ax.plot(xs, ys)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 40)
    view.canvas.draw_idle()
    return view, ax


def mouse(canvas, kind, ax, x, y, buttons = Qt.LeftButton):
    """A mouse event at data position (x, y)"""
    px, py = canvas.to_widget(*ax.page(x, y))
    button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
    event = QMouseEvent(kind, QPointF(px, py), QPointF(px, py), button, buttons, Qt.NoModifier)
    {QEvent.MouseButtonPress: canvas.mousePressEvent, QEvent.MouseMove: canvas.mouseMoveEvent, QEvent.MouseButtonRelease: canvas.mouseReleaseEvent}[kind](event)


def drag(canvas, ax, start, end):
    mouse(canvas, QEvent.MouseButtonPress, ax, *start)
    mouse(canvas, QEvent.MouseMove, ax, *end)
    mouse(canvas, QEvent.MouseButtonRelease, ax, *end, buttons = Qt.NoButton)


def test_canvas_shows_the_figure_at_its_size_and_ratio(app):
    view, ax = chart()
    image = view.canvas.grab().toImage()
    assert (image.width(), image.height()) == (600, 300)
    # The line in the palette's color where the data say, and white elsewhere
    px, py = view.canvas.to_widget(*ax.page(2.0, 20.0))
    assert image.pixelColor(round(px), round(py)).lightness() < 255
    assert image.pixelColor(590, 10).lightness() == 255
    # A new size moves the axes with it, and page and data stay each other's reverse
    view.canvas.resize(800, 500)
    view.canvas.grab()
    assert view.figure.width == 576 and view.figure.height == 360
    assert np.allclose(ax.data(*ax.page(2.5, 12.0)), (2.5, 12.0))


def test_painter_draws_what_the_rasterizer_draws(app):
    from viscose.qt import paint_page
    from candycrunch.utils import PALETTE, vc
    # Lines, dashes, sticks, bands, transparent points with edges, markers, text with subscripts, a gradient color bar, a legend, and an image with alpha
    figure = vc.Figure(6, 4, PALETTE)
    ax = figure.add_axes([0.12, 0.15, 0.68, 0.75])
    rng = np.random.default_rng(0)
    ax.plot([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 4.0], color = 'a1', lw = 1.5, label = 'line')
    ax.plot([0.0, 3.0], [0.5, 0.5], color = 'b1', dash = (3.0, 1.3), label = 'dashed')
    ax.vlines([0.5, 1.5, 2.5], 0.0, [1.0, 2.0, 3.0], color = ['a0', 'a2', 'b1'], lw = 1.2)
    ax.axvspan(1.0, 1.4, color = 'a2')
    ax.scatter(rng.uniform(0, 3, 40), rng.uniform(0, 4, 40), s = rng.uniform(10, 80, 40), c = rng.uniform(0, 1, 40), edge = ('white', 0.5), alpha = 0.9)
    ax.scatter([0.3, 0.6], [3.5, 3.5], s = 40, color = 'b1', marker = 'v', edge = ('white', 0.6))
    ax.annotate('^{1,3}A_{2\u03b1}', (2.0, 1.0), size = 9, rotation = 90.0, halo = 0.0)
    ax.text(0.5, 1.02, 'A title', size = 11, ha = 'center', va = 'baseline', axes_fraction = True)
    square = np.zeros((20, 20, 4), np.uint8)
    square[..., 2], square[5:15, 5:15, 3] = 200, 255
    ax.image(square, 2.5, 1.1, 20 * 72 / 200, 20 * 72 / 200, y_fraction = True)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    ax.set_xlabel('m/z')
    ax.legend('upper right')
    ax.colorbar('Scale')
    painted = paint_page(figure, 200).convertToFormat(QImage.Format_RGBA8888)
    pixels = np.frombuffer(painted.constBits(), np.uint8).reshape(painted.height(), painted.bytesPerLine())[:, :4 * painted.width()].reshape(painted.height(), painted.width(), 4)[..., :3].astype(float)
    raster = figure.to_rgba(200)[..., :3].astype(float)
    assert pixels.shape == raster.shape
    # Nearly the same everywhere, once a pixel of antialiasing is forgiven: a shifted, mirrored, or missing mark would stand out
    assert np.abs(pixels - raster).mean() < 1.5
    blurred = lambda image: sum(np.roll(np.roll(image, dx, 0), dy, 1) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)) / 25
    assert np.abs(blurred(pixels) - blurred(raster)).max() < 40


def test_painter_redraws_after_a_change_and_stays_on_the_page_size(app):
    from viscose.qt import paint_page
    view, ax = chart()
    before = paint_page(view.figure, 100)
    assert (before.width(), before.height()) == (600, 300)
    ax.set_xlim(1.5, 2.5)
    assert paint_page(view.figure, 100) != before
    # Twice the resolution for a screen with twice the pixels
    assert paint_page(view.figure, 200).width() == 1200


def test_targets_answer_the_mouse(app):
    from candycrunch.gui import ChartEvent, LabelTarget, Target
    view, ax = chart()
    points = Target(ax, [1.0, 1.02, 3.0], [10.0, 10.0, 30.0], radius = 5.0)
    x, y = ax.page(1.02, 10.0)
    hit, info = points.contains(ChartEvent(x, y))
    # The nearest point first
    assert hit and info['ind'][0] == 1 and set(info['ind']) == {0, 1}
    assert not points.contains(ChartEvent(x + 40, y))[0]
    # A stick answers along its whole height, and only up to its top
    sticks = Target(ax, [2.0], [30.0], stick = True)
    x, _ = ax.page(2.0, 0.0)
    assert sticks.contains(ChartEvent(x, ax.page(2.0, 15.0)[1]))[0] and not sticks.contains(ChartEvent(x, ax.page(2.0, 38.0)[1]))[0]
    # A point that zooming left behind does not answer, nor does a label of it
    label = LabelTarget(ax, {'xy': (3.0, 30.0), 'text': 'x', 'box': (-4.0, 3.0, 4.0, 20.0)})
    px, py = ax.page(3.0, 30.0)
    assert label.contains(ChartEvent(px, py + 10))[0] and not label.contains(ChartEvent(px, py - 10))[0]
    ax.set_xlim(0, 2)
    view.canvas.draw_idle()
    px, py = ax.page(3.0, 30.0)
    assert not points.contains(ChartEvent(px, py))[1]['ind'] == [2] and not label.contains(ChartEvent(px, py + 10))[0]


def test_canvas_reports_presses_and_the_wheel(app):
    view, ax = chart()
    pressed, moved, scrolled = [], [], []
    view.canvas.pressed.connect(pressed.append)
    view.canvas.moved.connect(moved.append)
    view.canvas.scrolled.connect(scrolled.append)
    mouse(view.canvas, QEvent.MouseButtonPress, ax, 2.0, 20.0)
    mouse(view.canvas, QEvent.MouseMove, ax, 3.0, 30.0, Qt.NoButton)
    assert np.allclose(ax.data(pressed[0].x, pressed[0].y), (2.0, 20.0), atol = 0.1) and pressed[0].button == 1
    assert np.allclose(ax.data(moved[0].x, moved[0].y), (3.0, 30.0), atol = 0.1)
    px, py = view.canvas.to_widget(*ax.page(2.0, 20.0))
    view.canvas.wheelEvent(QWheelEvent(QPointF(px, py), QPointF(px, py), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert scrolled[0].step == 1


def test_canvas_counts_a_double_click_once(app):
    from PySide6.QtTest import QTest
    from candycrunch.gui import Canvas
    from candycrunch.utils import PALETTE, vc
    figure = vc.Figure(4.0, 3.0, PALETTE)
    figure.add_axes([0.2, 0.2, 0.7, 0.7]).plot([0, 1], [0, 1])
    canvas, presses = Canvas(figure), []
    canvas.pressed.connect(presses.append)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    QTest.mouseDClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    assert len(presses) == 2


def test_label_target_covers_labels_and_cartoons_while_shown(app):
    from candycrunch.gui import ChartEvent, LabelTarget
    from candycrunch.utils import PALETTE, vc
    figure = vc.Figure(4.0, 3.0, PALETTE)
    ax = figure.add_axes([0.2, 0.2, 0.7, 0.5])
    ax.plot([0, 100], [0, 100])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    figure.layout()
    label = LabelTarget(ax, {'xy': (50.0, 50.0), 'box': (-4.0, 3.0, 4.0, 30.0)})
    cartoon = LabelTarget(ax, {'xy': (50.0, 1.2), 'box': (-20.0, -10.0, 20.0, 10.0), 'y_fraction': True})
    px, py = ax.page(50.0, 50.0)
    top = ax.rect[1] + 1.2 * ax.rect[3]
    # The label's margin counts, and the cartoon sits at its fraction of the axes, whatever the y limits
    assert label.contains(ChartEvent(px + 5, py + 10))[0] and not label.contains(ChartEvent(px + 6, py + 10))[0]
    assert cartoon.contains(ChartEvent(px + 15, top))[0] and not cartoon.contains(ChartEvent(px + 15, top - 30))[0]
    ax.set_ylim(60, 90)
    figure.layout()
    px = ax.page(50.0, 70.0)[0]
    assert not label.contains(ChartEvent(px, ax.page(50.0, 50.0)[1] + 10))[0]
    assert cartoon.contains(ChartEvent(px + 15, top))[0]
    # Neither answers once its m/z has left the view, as neither is drawn then
    ax.set_xlim(60, 90)
    figure.layout()
    assert not cartoon.contains(ChartEvent(*ax.page(50.0, 70.0)))[0]


def test_toolbar_pans_zooms_and_returns(app):
    from candycrunch.gui import Toolbar
    view, ax = chart()
    toolbar, pressed = Toolbar(view.canvas), []
    view.canvas.pressed.connect(pressed.append)
    assert [a.text() for a in toolbar.actions()] == ['Home', 'Back', 'Forward', 'Pan', 'Zoom']
    # A zoom into a rectangle, as far as the mouse dragged, sends no press
    toolbar.zoom.setChecked(True)
    assert toolbar.mode == 'zoom' and view.canvas.cursor().shape() == Qt.CrossCursor
    drag(view.canvas, ax, (1.0, 10.0), (3.0, 30.0))
    assert not pressed and np.allclose(ax.get_xlim(), (1.0, 3.0), atol = 0.05) and np.allclose(ax.get_ylim(), (10.0, 30.0), atol = 0.5)
    # Pan and zoom exclude each other, and a pan moves the view
    toolbar.pan.setChecked(True)
    assert toolbar.mode == 'pan' and not toolbar.zoom.isChecked()
    x_lo, x_hi = ax.get_xlim()
    drag(view.canvas, ax, (2.0, 20.0), (2.5, 20.0))
    assert np.allclose(ax.get_xlim(), (x_lo - 0.5, x_hi - 0.5), atol = 0.05)
    toolbar.move_by(-1)
    assert np.allclose(ax.get_xlim(), (x_lo, x_hi))
    toolbar.home()
    assert np.allclose(ax.get_xlim(), (0, 4)) and np.allclose(ax.get_ylim(), (0, 40))
    toolbar.pan.setChecked(False)
    assert toolbar.mode == '' and view.canvas.cursor().shape() == Qt.ArrowCursor


def test_dragged_range(app):
    view, ax = chart()
    ranges = []
    view.canvas.span = lambda lo, hi: ranges.append((lo, hi))
    drag(view.canvas, ax, (1.0, 20.0), (3.0, 25.0))
    assert len(ranges) == 1 and np.allclose(ranges[0], (1.0, 3.0), atol = 0.05)
    # Outside the axes, or without a drag, nothing is selected
    mouse(view.canvas, QEvent.MouseButtonPress, ax, 2.0, 20.0)
    mouse(view.canvas, QEvent.MouseButtonRelease, ax, 2.0, 20.0, Qt.NoButton)
    assert len(ranges) == 1


def test_ring_needs_no_new_drawing(app):
    view, ax = chart()
    view.canvas.grab()
    image = view.canvas.image
    view.canvas.rings = [(ax, 2.0, 20.0)]
    with_ring = view.canvas.grab().toImage()
    assert view.canvas.image is image
    px, py = view.canvas.to_widget(*ax.page(2.0, 20.0))
    assert with_ring.pixelColor(round(px + 6 * 100 / 72), round(py)).lightness() < 200


def test_package_imports_without_matplotlib():
    # The whole app and a figure drawn without it, in a process of its own
    code = ("import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path = None, target = None):\n"
            "        if name.split('.')[0] in ('matplotlib', 'seaborn'):\n"
            "            raise ImportError(name)\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from candycrunch import gui\n"
            "from candycrunch.analysis import plot_annotated_spectrum\n"
            "_, ax = plot_annotated_spectrum('Gal(b1-3)GalNAc', [204.09, 366.14], [50.0, 100.0], charge = -1, mass_threshold = 0.5, draw_glycans = False)\n"
            "assert ax.peak_labels and ax.figure.to_png(72).startswith(b'\\x89PNG')\n")
    subprocess.run([sys.executable, '-c', code], check = True, env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}, stdin = subprocess.DEVNULL)


def test_mirrored_spectrum():
    # A second spectrum below the axis, annotated with the same glycopeptide at its own settings: its labels hang below its peaks in the lower half, the upper ones stay above
    # the axis, and the peptide ladder counts the bonds either spectrum covers
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    seq = 'VQPFNVTQGK'
    b = {i: sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS for i in (2, 3, 4)}
    hcd = {204.0867: 100.0, 366.1395: 60.0, **{m: 40.0 for m in b.values()}}
    ethcd = {sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS + 17.026549: 80.0 - 10 * i for i in (2, 3, 6)} | {204.0867: 30.0}
    _, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', draw_glycans = False,
                                    mirror = {'mzs': list(ethcd), 'intensities': list(ethcd.values()), 'kwargs': {'fragmentation_method': 'EThcD'}, 'title': 'EThcD scan 2'})
    assert ax.get_ylim() == (-104, 104) and ax.peak_labels and ax.mirror_labels
    zero = float(ax.page(0.0, 0.0)[1])
    assert all(label['xy'][1] > 0 and float(ax.page(*label['xy'])[1]) + label['box'][1] >= zero - 0.5 for label in ax.peak_labels)
    assert all(label['xy'][1] < 0 and float(ax.page(*label['xy'])[1]) + label['box'][3] <= zero + 0.5 for label in ax.mirror_labels)
    assert len(ax.mirror_labels) >= 3
    ladder = next(a for a in ax.figure.axes if a is not ax)
    assert any(t.get('kind') == 'text' and t.get('s') == '4/9 backbone bonds' for t in ladder.artists)
    # without a mirror, the figure is the one drawn before: upper half only
    _, alone = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', draw_glycans = False)
    assert alone.get_ylim() == (0, 104) and alone.mirror_labels == []


def test_colored_peaks_are_named():
    # A peak carries its fragment's color only with its label or a cartoon's leader: an annotated peak below the label threshold that the peptide ladder does
    # not count (here a Y ion) stays gray, while a backbone ion the ladder counts is labeled and colored whatever its intensity
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    seq = 'VQPFNVTQGK'
    b = [sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS for i in (2, 3, 4)]
    y1 = sum(AA_masses[x] for x in seq) + 18.010565 + 203.079373 + PROTON_MASS
    hcd = {204.0867: 100.0, 366.1395: 60.0, b[0]: 40.0, b[1]: 2.0, b[2]: 1.0, y1: 1.0}
    hits, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', draw_glycans = False)
    named = {label['xy'][0] for label in ax.peak_labels}
    colored = {x for art in ax.artists if art['kind'] == 'vsegs' and art.get('lw') == 1.3 for x in art['x']}
    assert hits[b[1]] and hits[y1] and colored and colored <= named and {b[0], b[1], b[2]} <= colored and y1 not in colored


def test_ladder_labels_keep_apart():
    # Labels of neighboring bonds that would run into each other take a row each, and the ladder grows by that row
    import viscose as vc
    from candycrunch.analysis import draw_peptide_ladder
    hits = {1.0: {'Domon-Costello nomenclatures': [[['b_4', 'c_4', 'c-1_4']]]}, 2.0: {'Domon-Costello nomenclatures': [[['b_5', 'c_5', 'c-1_5']]]}}
    ax = vc.Figure(5, 1.5).add_axes([0.05, 0.1, 0.9, 0.8])
    draw_peptide_ladder('VQPFNVTQGK', hits, [4], ax = ax)
    y = {art['s']: art['y'] for art in ax.artists if art['kind'] == 'text' and '/' in art['s']}
    assert y['b4/c4/c4-1'] != y['b5/c5/c5-1'] and ax.get_ylim()[1] > 1.25


def test_labels_keep_off_other_peaks():
    # A label that would run across a taller neighbor's stick moves up or aside with a leader, in either half of a mirror, and keeps its text
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    seq = 'VQPFNVTQGK'
    b2 = sum(AA_masses[x] for x in seq[:2]) + PROTON_MASS
    peaks = {204.0867: 100.0, 1000.0: 50.0, b2: 30.0, b2 + 1.5: 100.0}
    _, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(peaks), list(peaks.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', draw_glycans = False,
                                    mirror = {'mzs': list(peaks), 'intensities': list(peaks.values()), 'kwargs': {'fragmentation_method': 'HCD'}})
    tall = float(ax.page(b2 + 1.5, 0.0)[0])
    for labels in (ax.peak_labels, ax.mirror_labels):
        label = next(label for label in labels if abs(label['xy'][0] - b2) < 1e-6)
        x = float(ax.page(*label['xy'])[0])
        assert label['text'] == 'b_{2}' and not x + label['box'][0] < tall < x + label['box'][2]
    leaders = [art for art in ax.artists if art['kind'] == 'line' and art.get('lw') == 0.4]
    assert len(leaders) >= 2


def test_labels_keep_their_peaks_on_real_spectra():
    # On real glycopeptide HCD spectra, labels cross far fewer sticks of other peaks than where labels avoid only each other, every peak named there keeps its name, and
    # no label runs past a side of the axes
    import gzip
    import json
    import candycrunch.analysis as analysis
    with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'glycopeptides', 'glycopeptide_benchmark.json.gz'), 'rt', encoding = 'utf-8') as f:
        spectra = [s for s in json.load(f)['spectra'] if s['activation'] == 'HCD' and s['stratum'] == 'N']
    legacy = analysis.place_peak_label

    def labels_only(*args, sticks = None, xbounds = None, lines = None, home = None, reserved = (), **kwargs):
        return legacy(*args, **kwargs)

    counts = {}
    for name, placer in (('alone', labels_only), ('sticks', legacy)):
        analysis.place_peak_label, across, named, past = placer, 0, [], 0
        try:
            for s in spectra:
                peaks = dict(sorted(map(tuple, s['peaks']), key = lambda x: -x[1])[:100])
                _, ax = analysis.plot_annotated_spectrum({'peptide': s['peptide'], 'glycans': s['glycans'], 'glycosites': s['glycosites']}, list(peaks), list(peaks.values()),
                                                         charge = s['charge'], mass_threshold_ppm = 10, fragmentation_method = 'HCD', draw_glycans = False)
                x, y = ax.page(np.array(list(peaks)), np.array(list(peaks.values())) / max(peaks.values()) * 100)
                for label in ax.peak_labels:
                    px, py = (float(v) for v in ax.page(*label['xy']))
                    box, other = label['box'], np.abs(x - px) > 0.5
                    across += bool(np.any(other & (x > px + box[0]) & (x < px + box[2]) & (y > py + box[1])))
                    past += px + box[0] < ax.rect[0] - 1.5 or px + box[2] > ax.rect[0] + ax.rect[2] + 1.5
                named.append({label['xy'][0] for label in ax.peak_labels})
        finally:
            analysis.place_peak_label = legacy
        counts[name] = (across, named, past)
    assert counts['sticks'][0] <= counts['alone'][0] / 3 and counts['sticks'][2] == 0
    assert all(alone <= now for alone, now in zip(counts['alone'][1], counts['sticks'][1]))


def test_composition_cartoons_name_their_peaks():
    # A glycan composition's cartoon is its peaks' name: they get no text label while it is drawn, a loss is written beside its symbols, and a cartoon switched off
    # gives its peak its text label back
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    seq = 'VQPFNVTQGK'
    b = [sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS for i in (2, 3, 4)]
    hcd = {204.0867: 100.0, 186.0761: 40.0, 366.1395: 60.0, b[0]: 40.0, b[1]: 30.0, b[2]: 20.0}
    _, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD')
    fragments = {box['fragment']: box['peaks'] for box in ax.cartoon_boxes}
    assert fragments['N1'] == [204.0867] and fragments['N1 -H2O'] == [186.0761] and fragments['H1N1'] == [366.1395]
    assert not {204.0867, 186.0761, 366.1395} & {label['xy'][0] for label in ax.peak_labels} and len(ax.peak_labels) == 3
    assert any(art['kind'] == 'text' and art.get('s') == '-H_{2}O' for art in ax.artists)
    _, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', cartoons = {'N1 -H2O': False})
    assert 'N1 -H2O' not in {box['fragment'] for box in ax.cartoon_boxes} and 'HexNAc-H2O' in {label['text'] for label in ax.peak_labels}


def test_z_ions_are_radicals():
    # CandyCrunch's z is the radical (z-dot) and its z+1 is z-dot + H: labels, the fragment table, and the peptide ladder write them with the dot, every other ion type as before
    import viscose as vc
    from candycrunch.analysis import domon_costello_to_label, domon_costello_to_html, draw_peptide_ladder
    assert domon_costello_to_label(['z_5']) == 'z\u2022_{5}' and domon_costello_to_label(['z+1_5']) == 'z\u2022+1_{5}'
    assert domon_costello_to_label(['a_2', 'b_2', 'c_3', 'c-1_3', 'w_4', 'x_4', 'y_5', 'Z_2_Alpha']) == 'a_{2}/b_{2}/c_{3}/c-1_{3}/w_{4}/x_{4}/y_{5}/Z_{2α}'
    assert domon_costello_to_html(['z_5', 'z+1_5', 'y_4', 'c-1_3']) == 'z&bull;<sub>5</sub>, z&bull;+1<sub>5</sub>, y<sub>4</sub>, c-1<sub>3</sub>'
    ax = vc.Figure(5, 1.5).add_axes([0.05, 0.1, 0.9, 0.8])
    draw_peptide_ladder('VQPFNVTQGK', {1.0: {'Domon-Costello nomenclatures': [[['y_3', 'z_3', 'z+1_3']]]}, 2.0: {'Domon-Costello nomenclatures': [[['c_2', 'c-1_2']]]}}, [4], ax = ax)
    texts = {art['s'] for art in ax.artists if art['kind'] == 'text'}
    assert 'y3/z\u20223/z\u20223+1' in texts and 'c2/c2-1' in texts


def test_spectrum_names_what_the_ladder_shows_and_draws_compositions():
    # Every backbone ion the peptide ladder counts is labeled whatever its intensity, a glycopeptide's Y ion is drawn as Pep + its glycan's symbols, and in a
    # mirror the glycan-only ions are drawn too, hanging below their peaks; the mirror's title stays clear of labels
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    from viscose.text import width
    seq = 'VQPFNVTQGK'
    b = [sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS for i in (2, 3)]
    y1 = sum(AA_masses[x] for x in seq) + 18.010565 + 203.079373 + PROTON_MASS
    hcd = {204.0867: 100.0, b[0]: 40.0, b[1]: 1.0, y1: 30.0}
    ethcd = {204.0867: 50.0, 1000.0: 100.0}
    _, ax = plot_annotated_spectrum('VQPFN*H3N4F1*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD',
                                    mirror = {'mzs': list(ethcd), 'intensities': list(ethcd.values()), 'kwargs': {'fragmentation_method': 'EThcD'}, 'title': 'EThcD scan 2'})
    labels = {label['xy'][0]: label['text'] for label in ax.peak_labels}
    assert labels[b[1]] == 'b_{3}' and labels[y1] == 'Pep + N1' and [label['text'] for label in ax.mirror_labels] == ['N1']
    assert any(art['kind'] == 'text' and art.get('s') == 'Pep +' for art in ax.artists)
    corner = (ax.rect[0] + 0.99 * ax.rect[2] - width('EThcD scan 2', 8), ax.rect[1] + 0.02 * ax.rect[3] + 1.2 * 8)
    assert all(float(ax.page(*label['xy'])[0]) + label['box'][2] < corner[0] or float(ax.page(*label['xy'])[1]) + label['box'][1] > corner[1] for label in ax.mirror_labels)


def test_long_labels_lie_where_they_cannot_stand():
    # In the half of a mirror, a long name over a small peak does not fit standing, so it lies across its peak rather than being left out
    from candycrunch.analysis import place_peak_label
    placed = []
    spot = place_peak_label(100.0, 10.0, 'Peptide/Y Hex(1)/HexNAc(2) [+2]', placed, (0.0, 60.0), 7, 6)
    assert spot is not None and spot[2] == 'center' and spot[4][2] - spot[4][0] > spot[4][3] - spot[4][1] and placed


def test_composition_cartoons_stand_near_their_peaks():
    # A dense run of oxonium ions left of empty axes steps up to the right of its peaks rather than running along the band in one row (the oxonium ions of
    # KDNTTVTR H6N5F1A2 at m/z 126-366 had reached m/z 1,300): every cartoon names its peak near it, and no leader crosses another or runs through a cartoon
    import viscose as vc
    from candycrunch.analysis import plot_annotated_spectrum, AA_masses
    from glycowork.motif.tokenization import PROTON_MASS
    seq = 'VQPFNVTQGK'
    b = [sum(AA_masses[x] for x in seq[:i]) + PROTON_MASS for i in (2, 3, 4)]
    hcd = {126.0550: 15.0, 138.0550: 60.0, 144.0655: 20.0, 168.0655: 40.0, 186.0761: 30.0, 204.0867: 100.0, 274.0921: 50.0, 292.1027: 30.0, 366.1395: 70.0,
           657.2349: 20.0, b[0]: 25.0, b[1]: 15.0, b[2]: 10.0}
    figure = vc.Figure(10, 5.15)
    ax, ladder = figure.add_axes([0.065, 0.107, 0.915, 0.42]), figure.add_axes([0.065, 0.62, 0.915, 0.17])
    _, ax = plot_annotated_spectrum('VQPFN*H5N4F1A2*VTQGK', list(hcd), list(hcd.values()), charge = 2, mass_threshold_ppm = 10, fragmentation_method = 'HCD', ax = ax,
                                    ladder_ax = ladder)
    x0, y0, w, h = ax.rect
    page = lambda x, y: (float(ax.page(x, 0.0)[0]), y0 + y * h)
    boxes = [(cx + c['box'][0], cy + c['box'][1], cx + c['box'][2], cy + c['box'][3]) for c in ax.cartoon_boxes for cx, cy in [page(*c['xy'])]]
    assert len(boxes) == 10 and all(shown for _, shown in ax.cartoon_fragments.values())
    distance = [max(box[0] - px, px - box[2], 0.0) for box, c in zip(boxes, ax.cartoon_boxes) for px in [page(c['peaks'][0], 0.0)[0]]]
    assert max(distance) < 200 and np.mean(distance) < 60
    segments = [(p, q) for a in ax.artists if a['kind'] == 'line' and a.get('dash') == (2.0, 1.0) for path in [[page(x, y) for x, y in zip(a['x'], a['y'])]]
                for p, q in zip(path, path[1:])]
    side = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    crosses = lambda s, t: side(*s, t[0]) * side(*s, t[1]) < 0 and side(*t, s[0]) * side(*t, s[1]) < 0
    assert len(segments) >= 10 and not any(crosses(s, t) for i, s in enumerate(segments) for t in segments[i + 1:])
    assert not any(crosses(s, d) for s in segments for box in boxes for d in (((box[0], box[1]), (box[2], box[3])), ((box[0], box[3]), (box[2], box[1]))))

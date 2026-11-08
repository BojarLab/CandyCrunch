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

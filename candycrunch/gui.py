"""CandyCrunch desktop app: run CandyCrunch on LC-MS/MS files, inspect and curate its predictions, and annotate spectra with CandyCrumbs"""
import ast
import importlib.util
import io
import itertools
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
try:
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QSettings, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QRectF, QSize, QUrl, QStringListModel
    from PySide6.QtGui import QTextOption, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPalette, QPixmap, QBrush, QShortcut, QFontDatabase
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton,
                                   QToolButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget,
                                   QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTableView, QAbstractItemView, QSplitter, QStackedWidget, QTabWidget, QScrollArea,
                                   QFrame, QFileDialog, QMessageBox, QInputDialog, QDockWidget, QProgressBar, QStyledItemDelegate, QStyleOptionViewItem,
                                   QStyle, QMenu, QCompleter, QSizePolicy)
except ImportError as error:
    raise ImportError('The CandyCrunch app needs PySide6, which comes with: pip install "candycrunch[gui]"') from error
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

# Defaults and constants are read from prediction.py's source, so the app always offers the library's own defaults without importing torch
_tree = ast.parse(open(importlib.util.find_spec('candycrunch.prediction').origin, encoding = 'utf-8').read())
_batch_def = next(node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name == 'wrap_inference_batch')
DEFAULTS = {'intra_cat_thresh': 1.0}
for _arg, _default in zip(_batch_def.args.args[-len(_batch_def.args.defaults):], _batch_def.args.defaults):
    try:
        DEFAULTS[_arg.arg] = ast.literal_eval(_default)
    except ValueError:
        pass
_constants = {node.targets[0].id: node.value for node in _tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
MZ_REF = ast.literal_eval(_constants['MZ_REF'])
FILTER_CHOICES = sorted(set(ast.literal_eval(_constants['comp_vector_order'])) | DEFAULTS['filter_out'], key = str.lower)
BATCH_KEYS = ('intra_cat_thresh', 'top_n_isomers', 'n_jobs')
FEATURES = '\x00features'
GLYCAN_CLASSES = [('O-glycans', 'O'), ('N-glycans', 'N'), ('Free oligosaccharides (e.g., milk)', 'free'), ('Glycolipid glycans', 'lipid')]
REDUCING_ENDS = [('Reduced (alditol)', 'reduced'), ('2-AB', '2AB'), ('2-AA', '2AA'), ('Procainamide', 'procainamide'), ('Custom tag mass', 'custom')]
SAMPLE_PREPS = [('Underivatized', 'underivatized'), ('Permethylated', 'permethylated'), ('Peracetylated', 'peracetylated')]
LCS = [('PGC', 'PGC'), ('C18', 'C18'), ('Other', 'other')]
TRAPS = [('Linear ion trap', 'linear'), ('Orbitrap', 'orbitrap'), ('Bruker amaZon', 'amazon'), ('Other', 'other')]
TAXONOMY_LEVELS = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain']
FRAGMENTATIONS = [('Any', None), ('CID', 'CID'), ('HCD', 'HCD'), ('ETD', 'ETD'), ('ECD', 'ECD'), ('EThcD', 'EThcD'), ('ETciD', 'ETciD')]
SPECTRA_SUFFIXES = ('.raw', '.mzml', '.mzxml', '.mgf')
# What the worker reports while wrap_inference(_batch) runs, keyed by the prediction.py function that starts each step
STAGES = {'load_spectra_filepath': 'Reading {}', 'condense_dataframe': 'Grouping spectra into chromatographic peaks',
          'assign_candidate_structures': 'Matching precursor compositions', 'assign_annotation_scores_pooled': 'Scoring fragment evidence',
          'get_topk': 'Predicting structures', 'assign_categories': 'Harmonizing isomer groups across files', 'augment_predictions': 'Adding biosynthetic and database candidates',
          'finalise_predictions': 'Quantifying and finalizing'}
# Share of a file's run time per step, measured on the test datasets with 2 CPU cores (model inference ~45%, fragment scoring ~40%, more for
# N-glycans); after each single-file run the app blends in the shares measured on this computer, so its time estimates adapt to the machine
STAGE_WEIGHTS = {'load_spectra_filepath': 0.06, 'condense_dataframe': 0.04, 'assign_candidate_structures': 0.01,
                 'assign_annotation_scores_pooled': 0.37, 'get_topk': 0.45, 'augment_predictions': 0.05,
                 'finalise_predictions': 0.02}
# Share of a batch's time spent on harmonizing, augmenting and finalizing after every file has been predicted
TAIL_WEIGHT = 0.06
# Per process: where progress goes and where each step starts within a file, read by the wrappers _install_progress puts around prediction.py
_PROGRESS = {}
INK, MUTED, LILAC, LILAC_LIGHT, WASH, CANDY = '#2a2133', '#857a90', '#8c5bb5', '#e6d9f1', '#f6f2f9', '#c8243f'
# matplotlib's mathtext parser is not thread-safe, so the annotation threads and the window never lay out text at the same time
MPL_LOCK = threading.Lock()
# glycowork's modules import each other in a cycle, so two threads importing them at once (the drawing, annotation, and taxonomy threads all
# start with an import) can hit Python's import deadlock detection, which killed the drawing thread or failed the first annotation
IMPORT_LOCK = threading.Lock()
CONFIDENCE_MAP = LinearSegmentedColormap.from_list('candy', ['#d9c8ea', LILAC, CANDY])
STYLE = f"""
QWidget#settings {{ background: {WASH}; }}
QLabel[role="heading"] {{ font-weight: 600; color: {INK}; padding-top: 8px; }}
QLabel[role="hint"] {{ color: {MUTED}; }}
QLabel#title {{ font-size: 28pt; font-weight: 300; color: {LILAC}; }}
QLabel#stage {{ font-size: 12pt; color: {INK}; }}
QPushButton#run {{ background: {CANDY}; color: white; border: none; border-radius: 4px; padding: 8px 18px; font-weight: 600; }}
QPushButton#run:hover {{ background: #a91c34; }}
QPushButton#run:disabled {{ background: #e5b8c1; }}
QHeaderView::section {{ background: {WASH}; border: none; border-bottom: 1px solid #ddd0e8; border-right: 1px solid #eee6f3; padding: 4px 6px; font-weight: 600; }}
QTableView {{ gridline-color: #f0e9f5; selection-background-color: {LILAC_LIGHT}; selection-color: {INK}; }}
QListWidget {{ selection-background-color: {LILAC_LIGHT}; selection-color: {INK}; }}
QSplitter::handle {{ background: #e9e0f0; }}
QTabWidget::pane {{ border-top: 1px solid #ddd0e8; }}
"""


def _worker_main(jobs, messages):
    """Runs CandyCrunch in its own process: the window stays responsive, a run is cancelled by ending the process, and the model loads once for all runs"""
    class Pipe(io.TextIOBase):
        def write(self, text):
            messages.put(('log', text))
            return len(text)
    sys.stdout = sys.stderr = Pipe()
    try:
        import candycrunch.prediction as cp
    except Exception:
        messages.put(('fatal', traceback.format_exc()))
        return
    messages.put(('ready', None))
    parent = mp.parent_process()
    while True:
        try:
            job = jobs.get(timeout = 1)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return
            continue
        if job is None:
            return
        experiments, weights = job
        start, results = time.time(), []
        try:
            # Experiments one after the other, each with its own settings; only the runs of one experiment are harmonized into a feature table
            for name, files, settings in experiments:
                _install_progress(messages, weights, len(files))
                kwargs = {k: v for k, v in settings.items() if k not in BATCH_KEYS}
                if len(files) == 1:
                    df, spectra = cp.wrap_inference(files[0], spectra = True, **kwargs)
                    tables, features = {os.path.splitext(os.path.basename(files[0]))[0]: (df, spectra)}, None
                else:
                    features, tables = cp.wrap_inference_batch(files, spectra = True, **kwargs, **{k: settings[k] for k in BATCH_KEYS})
                results.append({'name': name, 'files': files, 'settings': settings, 'tables': tables, 'features': features})
            messages.put(('result', {'experiments': results, 'elapsed': time.time() - start, 'finished': time.strftime('%Y-%m-%d %H:%M')}))
        except Exception:
            messages.put(('error', traceback.format_exc()))


def _install_progress(messages, weights, n_files):
    """Wraps the prediction.py steps named in STAGES (wrap_inference looks them up as module globals at call time) and the loops inside the
    slow ones, in the worker and in wrap_inference_batch's pool processes, so that every process reports its step and how far through its file it is"""
    import candycrunch.prediction as cp
    # In a batch, augmenting and finalizing come after harmonization (the 'tail'), so a file's own share ends with the prediction
    order = [name for name in STAGE_WEIGHTS if n_files == 1 or name not in ('augment_predictions', 'finalise_predictions')]
    total = sum(weights[name] for name in order)
    _PROGRESS.update(messages = messages, weights = weights, n_files = n_files, key = None, stage = None, value = -1.0, time = 0.0, tail = 0,
                     done = 0, todo = 0, shares = {name: weights[name] / total for name in order},
                     starts = dict(zip(order, itertools.accumulate([0.0] + [weights[name] / total for name in order]))))
    if getattr(cp, '_progress_installed', False):
        return
    cp._progress_installed = True
    def report(value, force = False):
        p = _PROGRESS
        if force or value - p['value'] > 0.004 or time.time() - p['time'] > 0.5:
            p['value'], p['time'] = value, time.time()
            p['messages'].put(('progress', (p['key'], min(value, 1.0))))
    def advance(done):
        p = _PROGRESS
        p['done'] += done
        if p['todo'] and p['stage'] in p['shares'] and p['key'] != 'tail':
            report(p['starts'][p['stage']] + p['shares'][p['stage']] * min(p['done'] / p['todo'], 1.0))
    class Batches:
        """get_topk's dataloader, counting batches"""
        def __init__(self, loader):
            self.loader, self.dataset = loader, loader.dataset
        def __iter__(self):
            for batch in self.loader:
                yield batch
                advance(1)
    def staged(func, name):
        def run(*args, **kwargs):
            p = _PROGRESS
            p['messages'].put(('stage', (name, STAGES[name].format(os.path.basename(str(args[0]))) if '{}' in STAGES[name] else STAGES[name])))
            if name == 'load_spectra_filepath' and p['key'] != 'tail':
                p['key'], p['value'] = os.path.normcase(os.path.abspath(str(args[0]))), -1.0
            elif name == 'assign_categories':
                p['key'], p['value'] = 'tail', -1.0
            p['stage'], p['done'], p['todo'] = name, 0, 0
            if p['key'] == 'tail':
                # Harmonization, then augmenting and finalizing every file
                report(p['tail'] / (1 + 2 * p['n_files']), True)
                p['tail'] += 1
            elif name in p['starts']:
                report(p['starts'][name], True)
            if name == 'assign_annotation_scores_pooled':
                p['todo'] = args[0]['candidate_structure'].nunique()
            elif name == 'get_topk':
                p['todo'] = len(args[0])
                args = (Batches(args[0]),) + args[1:]
            return func(*args, **kwargs)
        return run
    for name in STAGES:
        setattr(cp, name, staged(getattr(cp, name), name))
    crumbs = cp.CandyCrumbs
    def counted(*args, **kwargs):
        result = crumbs(*args, **kwargs)
        # The step's todo counts candidate structures, i.e., MS2 calls; the extra calls for MS3 spectra are not counted
        if _PROGRESS['stage'] == 'assign_annotation_scores_pooled' and kwargs.get('ms3_precursor') is None:
            advance(1)
        return result
    cp.CandyCrumbs = counted
    real_read_mzml = cp.read_mzml
    def read_mzml(filepath, *args, **kwargs):
        """read_mzml, counting spectra against the count in the mzML header"""
        with open(filepath, 'rb') as f:
            count = re.search(rb'<spectrumList[^>]*\scount="(\d+)"', f.read(1 << 20))
        _PROGRESS['todo'] = int(count.group(1)) if count else 0
        for n, spectrum in enumerate(real_read_mzml(filepath, *args, **kwargs), 1):
            yield spectrum
            if n % 100 == 0:
                advance(100)
    cp.read_mzml = read_mzml
    def pool(max_workers = None, initializer = None, initargs = ()):
        # Spawned on every platform; each pool process installs the same wrappers and reports through the same queue
        p = _PROGRESS
        return ProcessPoolExecutor(max_workers = max_workers, mp_context = mp.get_context('spawn'), initializer = _pool_initializer,
                                   initargs = (p['messages'], p['weights'], p['n_files'], initializer, initargs))
    cp.ProcessPoolExecutor = pool


def _pool_initializer(messages, weights, n_files, initializer, initargs):
    if initializer is not None:
        initializer(*initargs)
    _install_progress(messages, weights, n_files)


class InferenceRunner(QObject):
    """Owns the worker process and relays its messages as signals"""
    ready, stage, progress, log, finished, failed = Signal(), Signal(object), Signal(object), Signal(str), Signal(object), Signal(str)

    def __init__(self):
        super().__init__()
        self.context, self.busy, self.dead = mp.get_context('spawn'), False, False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.start()

    def start(self):
        # Not a daemon: wrap_inference_batch(n_jobs > 1) starts processes of its own, which daemons may not
        self.jobs, self.messages = self.context.Queue(), self.context.Queue()
        self.process = self.context.Process(target = _worker_main, args = (self.jobs, self.messages), name = 'CandyCrunch worker')
        self.process.start()
        self.timer.start()

    def submit(self, experiments, weights):
        """experiments: (name, files, settings) per experiment"""
        self.busy = True
        self.jobs.put((experiments, weights))

    def cancel(self):
        self.process.terminate()
        self.process.join(5)
        self.busy = False
        self.start()

    def shutdown(self):
        self.timer.stop()
        if self.process.is_alive():
            self.jobs.put(None)
            self.process.join(1.5)
        if self.process.is_alive():
            self.process.terminate()

    def poll(self):
        while True:
            try:
                kind, content = self.messages.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            if kind == 'ready':
                self.ready.emit()
            elif kind == 'stage':
                self.stage.emit(content)
            elif kind == 'progress':
                self.progress.emit(content)
            elif kind == 'log':
                self.log.emit(content)
            elif kind == 'result':
                self.busy = False
                self.finished.emit(content)
            elif kind in ('error', 'fatal'):
                self.busy, self.dead = False, kind == 'fatal'
                self.failed.emit(content)
        if not self.process.is_alive() and not self.dead:
            self.timer.stop()
            if self.busy:
                self.busy = False
                self.failed.emit('The CandyCrunch worker stopped unexpectedly, most likely because it ran out of memory.')
            self.start()


class GlycanImages(QObject):
    """SNFG drawings from GlycoDraw, rendered in a background thread and cached: compact ones for tables and lists, ones with linkages for the detail view"""
    ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.images, self.pending, self.requests, self.order = {}, set(), queue.PriorityQueue(), itertools.count()
        threading.Thread(target = self.render_loop, daemon = True).start()

    def get(self, glycan, compact = True, urgent = False):
        """Returns the QImage (null if GlycoDraw cannot draw it), or None while it is still being rendered"""
        if not isinstance(glycan, str) or not glycan:
            return QImage()
        key = (glycan, compact)
        if key in self.images:
            return self.images[key]
        if key not in self.pending or urgent:
            self.pending.add(key)
            self.requests.put((0 if urgent else 1, next(self.order), key))
        return None

    def render_loop(self):
        try:
            with IMPORT_LOCK:
                from glycowork.motif.draw import GlycoDraw
                from glycorender.render import convert_svg_to_png
        except ImportError:
            # Without the drawing stack every structure is shown as text
            GlycoDraw = convert_svg_to_png = None
        while True:
            key = self.requests.get()[2]
            if key in self.images:
                continue
            try:
                image = QImage.fromData(convert_svg_to_png(GlycoDraw(key[0], compact = key[1], suppress = True).as_svg(), None, return_bytes = True, scale = 2))
            except Exception:
                image = QImage()
            self.images[key] = image
            self.ready.emit(key)


class Annotator(QObject):
    """Runs plot_annotated_spectrum in a background thread; only the newest request is worked on, so scrolling through a table never queues up stale spectra"""
    done = Signal(int, object, object, str)

    def __init__(self):
        super().__init__()
        self.request, self.wake, self.counter = None, threading.Event(), 0
        threading.Thread(target = self.loop, daemon = True).start()

    def submit(self, request):
        self.counter += 1
        self.request = (self.counter, request)
        self.wake.set()
        return self.counter

    def loop(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            request_id, request = self.request
            width, height = request['figsize']
            try:
                with IMPORT_LOCK:
                    from candycrunch.analysis import plot_annotated_spectrum
                figure = Figure(figsize = request['figsize'], dpi = 100)
                FigureCanvasAgg(figure)
                # Margins in inches; with cartoons, the title sits at 1.4 axes heights above the bottom, so the axes top leaves room for it
                bottom = 0.55 / height
                figure.subplots_adjust(left = 0.65 / width, right = 1 - 0.2 / width, bottom = bottom, top = bottom + (1 - 0.4 / height - bottom) / 1.4 if request['cartoons'] else 1 - 0.4 / height)
                with MPL_LOCK:
                    hit_dict, ax = plot_annotated_spectrum(request['structure'], request['mzs'], request['intensities'], mass_threshold = request['tolerance'],
                                                           charge = request['charge'], mass_tag = request['mass_tag'], sample_prep = request['sample_prep'],
                                                           ax = figure.add_subplot(), draw_glycans = request['cartoons'], **request['kwargs'])
                    if request['ms3']:
                        # A triangle under every peak that was isolated for MS3 and an invisible stick over that peak, both clickable
                        mzs, ints = np.array(request['mzs']), np.array(request['intensities'])
                        near = [int(np.argmin(np.abs(mzs - p))) for p in request['ms3']]
                        hit = [abs(mzs[k] - p) <= request['tolerance'] for k, p in zip(near, request['ms3'])]
                        xs = [mzs[k] if h else p for k, h, p in zip(near, hit, request['ms3'])]
                        figure.ms3_artists = (ax.scatter(xs, [0] * len(xs), marker = '^', s = 70, c = CANDY, edgecolors = 'white', linewidths = 0.8, zorder = 6,
                                                         clip_on = False, picker = True),
                                              ax.vlines(xs, 0, [ints[k] / (ints.max() or 1) * 100 if h else 0 for k, h in zip(near, hit)], colors = 'none', picker = 5))
                self.done.emit(request_id, figure, hit_dict, '')
            except Exception as error:
                self.done.emit(request_id, None, None, f'{type(error).__name__}: {error}')


def _combo(items):
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    return box


def _spin(low, high, step, decimals = 0, suffix = ''):
    box = QSpinBox() if decimals == 0 else QDoubleSpinBox()
    if decimals:
        box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _label(text, role = 'heading'):
    label = QLabel(text)
    label.setProperty('role', role)
    label.setWordWrap(role == 'hint')
    return label


def _set_combo(box, value):
    index = box.findData(value)
    if index >= 0:
        box.setCurrentIndex(index)


def _fmt(value, decimals):
    return '' if value is None or (isinstance(value, float) and np.isnan(value)) else f'{value:.{decimals}f}'


def _number_column(title, values, decimals, tip = None):
    raw = [None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v) for v in values]
    return {'title': title, 'kind': 'number', 'raw': raw, 'text': [_fmt(v, decimals) for v in raw], 'tip': tip}


def _mass_tag(settings):
    """The reducing-end mass CandyCrumbs takes, built as in assign_annotation_scores_pooled"""
    with IMPORT_LOCK:
        from glycowork.motif.tokenization import modification_formula_dict, calculate_adduct_mass
    modification = settings.get('modification')
    label = calculate_adduct_mass(modification_formula_dict[modification]) if modification in modification_formula_dict else 0
    return label + (settings.get('mass_tag') or 0)


def _composition(composition):
    with IMPORT_LOCK:
        from glycowork.glycan_data.loader import stringify_dict
    return stringify_dict(composition) if isinstance(composition, dict) and composition else ''


class TableModel(QAbstractTableModel):
    """Read-only view of one result table, column by column as ResultsView lays them out; curation lives in the dataframe's own columns"""
    BAR_ROLE = Qt.UserRole + 1

    def __init__(self, images):
        super().__init__()
        self.images, self.df, self.columns, self.search = images, pd.DataFrame(), [], []
        images.ready.connect(self.image_ready)

    def set_table(self, df, columns):
        self.beginResetModel()
        self.df, self.columns = df, columns
        self.search = [' '.join(str(c['text'][r]).lower() for c in columns) for r in range(len(df))]
        self.endResetModel()

    def rowCount(self, parent = QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent = QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]['title']
        if orientation == Qt.Horizontal and role == Qt.ToolTipRole:
            return self.columns[section].get('tip')
        return None

    def data(self, index, role = Qt.DisplayRole):
        column, row = self.columns[index.column()], index.row()
        if role == Qt.DisplayRole:
            return column['text'][row]
        if role == Qt.UserRole:
            return column['raw'][row]
        if role == self.BAR_ROLE:
            return column.get('bar', [None] * (row + 1))[row]
        if role == Qt.ToolTipRole:
            return column.get('tips', column['text'])[row] or None
        if role == Qt.TextAlignmentRole:
            return int((Qt.AlignRight if column['kind'] in ('number', 'bar') else Qt.AlignLeft) | Qt.AlignVCenter)
        excluded, curated = bool(self.df['excluded'].iat[row]), bool(self.df['curation'].iat[row])
        if role == Qt.ForegroundRole and excluded:
            return QColor(MUTED)
        if role == Qt.FontRole and (excluded or curated):
            font = QFont()
            font.setStrikeOut(excluded)
            font.setItalic(curated)
            return font
        return None

    def image_ready(self, key):
        for c, column in enumerate(self.columns):
            if column['kind'] == 'structure' and len(self.df):
                self.dataChanged.emit(self.index(0, c), self.index(len(self.df) - 1, c), [Qt.DisplayRole])


class TableProxy(QSortFilterProxyModel):
    """Sorts on raw values (empty cells last in ascending order) and filters on the text of every column"""

    def __init__(self):
        super().__init__()
        self.needle = ''
        self.setSortRole(Qt.UserRole)

    def set_needle(self, text):
        # Qt 6.9 replaced invalidateFilter with beginFilterChange/endFilterChange
        modern = hasattr(self, 'beginFilterChange')
        if modern:
            self.beginFilterChange()
        self.needle = text.strip().lower()
        if modern:
            self.endFilterChange()
        else:
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        return not self.needle or all(word in self.sourceModel().search[row] for word in self.needle.split())

    def lessThan(self, left, right):
        a, b = left.data(Qt.UserRole), right.data(Qt.UserRole)
        if a is None or b is None:
            return a is None and b is not None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class StructureDelegate(QStyledItemDelegate):
    """Paints the SNFG drawing of a cell's structure, or its text until the drawing is ready (or if GlycoDraw cannot draw it)"""

    def __init__(self, images, parent):
        super().__init__(parent)
        self.images = images

    def paint(self, painter, option, index):
        image = self.images.get(index.data(Qt.UserRole))
        if image is None or image.isNull():
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ''
        opt.widget.style().drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = opt.rect.adjusted(6, 4, -6, -4)
        scale = min(rect.width() / image.width(), rect.height() / image.height(), 0.2)
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if opt.font.strikeOut():
            painter.setOpacity(0.3)
        painter.drawImage(QRectF(rect.x(), rect.y() + (rect.height() - image.height() * scale) / 2, image.width() * scale, image.height() * scale), image)
        painter.restore()


class BarDelegate(QStyledItemDelegate):
    """Abundance as a bar behind the number; hatched where the abundance only comes from MS1 gap filling"""

    def paint(self, painter, option, index):
        bar = index.data(TableModel.BAR_ROLE)
        if bar is not None and bar[0] > 0:
            rect = option.rect.adjusted(3, 6, -3, -6)
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor('#cbb2e0'), Qt.BDiagPattern) if bar[1] else QColor(LILAC_LIGHT))
            painter.drawRoundedRect(QRectF(rect.x(), rect.y(), max(2.0, rect.width() * min(bar[0], 1.0)), rect.height()), 2, 2)
            painter.restore()
        super().paint(painter, option, index)


class Canvas(FigureCanvasQTAgg):
    """Skips a redraw while an annotation thread holds MPL_LOCK and catches up right after, so the window never blocks on it"""

    def draw(self):
        if MPL_LOCK.acquire(blocking = False):
            try:
                super().draw()
            finally:
                MPL_LOCK.release()
        else:
            QTimer.singleShot(60, self.draw_idle)


class ChartView(QWidget):
    """A matplotlib figure that follows the widget's size"""

    def __init__(self, height = 3.0):
        super().__init__()
        self.figure = Figure(figsize = (6, height), dpi = 100)
        self.canvas = Canvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)


class SpectrumPanel(QWidget):
    """Annotated MS2 (or MS3) spectrum (plot_annotated_spectrum) plus a fragment table; shared by the results view and the CandyCrumbs tab"""
    ms3_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.annotator, self.request, self.latest, self.canvas, self.toolbar = Annotator(), None, 0, None, None
        self.annotator.done.connect(self.show_figure)
        self.controls = QHBoxLayout()
        self.controls.setContentsMargins(6, 4, 6, 0)
        self.summary = _label('', 'hint')
        self.summary.setWordWrap(False)
        # Ignored: the summary's length changes with every annotation, and as a minimum width it resized the panel, which re-annotated it, endlessly
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.tolerance = _spin(0.001, 2.0, 0.05, 3, ' Da')
        self.tolerance.setValue(DEFAULTS['ppm_thresh'] * MZ_REF / 1e6)
        self.tolerance.setToolTip('Maximum difference between an observed peak and a theoretical fragment mass')
        self.cartoons = QCheckBox('Fragment cartoons')
        self.cartoons.setChecked(True)
        self.controls.addStretch(1)
        self.controls.addWidget(QLabel('Tolerance'))
        self.controls.addWidget(self.tolerance)
        self.controls.addWidget(self.cartoons)
        self.message = QLabel()
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setProperty('role', 'hint')
        self.holder = QWidget()
        self.holder_layout = QVBoxLayout(self.holder)
        self.holder_layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.message)
        self.stack.addWidget(self.holder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self.controls)
        self.summary.setContentsMargins(10, 0, 6, 0)
        layout.addWidget(self.summary)
        layout.addWidget(self.stack, 1)
        self.fragments = QTextBrowser()
        self.fragments.setOpenLinks(False)
        self.tolerance.valueChanged.connect(self.resubmit)
        self.cartoons.toggled.connect(self.resubmit)
        # Labels and cartoons are placed for one figure size, so a resized panel gets a freshly laid out spectrum once resizing stops
        self.relayout = QTimer(self)
        self.relayout.setSingleShot(True)
        self.relayout.setInterval(400)
        self.relayout.timeout.connect(self.resubmit)

    def annotate(self, structure, mzs, intensities, charge, mass_tag, sample_prep, ms3 = (), **kwargs):
        """ms3: m/z of the peaks isolated for MS3, which get a clickable marker that emits ms3_clicked with their index"""
        self.request = {'structure': structure, 'mzs': [float(x) for x in mzs], 'intensities': [float(x) for x in intensities], 'charge': int(charge),
                        'mass_tag': mass_tag, 'sample_prep': sample_prep, 'ms3': [float(x) for x in ms3], 'kwargs': kwargs}
        self.resubmit()

    def resubmit(self):
        if self.request is None:
            return
        size = self.stack.size()
        self.request.update(tolerance = self.tolerance.value(), cartoons = self.cartoons.isChecked(),
                            figsize = (max(size.width(), 500) / 100, max(size.height(), 300) / 100))
        self.latest = self.annotator.submit(dict(self.request))
        self.summary.setText(f'Annotating with CandyCrumbs: {self.request["structure"]}')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.request is not None and abs(event.size().height() - event.oldSize().height()) + abs(event.size().width() - event.oldSize().width()) > 40:
            self.relayout.start()

    def show_message(self, text):
        self.request = None
        self.latest = self.annotator.counter + 1
        self.message.setText(text)
        self.stack.setCurrentWidget(self.message)
        self.summary.setText('')
        self.fragments.setHtml(f'<p style="color:{MUTED}">{text}</p>')

    def show_figure(self, request_id, figure, hit_dict, error):
        if request_id != self.latest:
            return
        if error:
            self.message.setText(f'CandyCrumbs could not annotate this spectrum with {self.request["structure"]}.\n\n{error}')
            self.stack.setCurrentWidget(self.message)
            self.summary.setText('')
            return
        for widget in (self.canvas, self.toolbar):
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.canvas = Canvas(figure)
        # Pan and zoom hold the canvas' widgetlock, so clicks while using them never count as picks
        self.canvas.mpl_connect('pick_event', lambda event: event.artist in getattr(figure, 'ms3_artists', ()) and len(event.ind) and self.ms3_clicked.emit(int(event.ind[0])))
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates = False)
        self.toolbar.setIconSize(QSize(16, 16))
        self.controls.insertWidget(0, self.toolbar)
        self.holder_layout.addWidget(self.canvas)
        self.stack.setCurrentWidget(self.holder)
        peaks = dict(zip(self.request['mzs'], self.request['intensities']))
        total, top = sum(peaks.values()) or 1, max(peaks.values()) or 1
        hits = {mz: hit for mz, hit in hit_dict.items() if hit}
        self.summary.setText(f'{len(hits)} of {len(peaks)} peaks annotated, {sum(peaks.get(mz, 0) for mz in hits) / total:.0%} of the ion current' + (
            f'; \u25b2 marks the {len(self.request["ms3"])} peak(s) with MS3 spectra, click one to annotate them' if self.request['ms3'] else ''))
        if self.request['kwargs'].get('ms3_precursor') is not None and not hits:
            self.summary.setText(f'm/z {self.request["kwargs"]["ms3_precursor"]:.2f} is no fragment of {self.request["structure"]}, so none of its MS3 peaks can be annotated')
        from candycrunch.analysis import domon_costello_to_html
        rows = []
        for mz in sorted(hits):
            hit = hits[mz]
            for n, (names, theo, z) in enumerate(zip(hit['Domon-Costello nomenclatures'], hit['Theoretical fragment masses'], hit['Fragment charges'])):
                flat = [x for sub in names for x in sub] if names and isinstance(names[0], list) else list(names)
                first = f'<td>{mz:.4f}</td><td align="right">{peaks.get(mz, 0) / top * 100:.1f}</td>' if n == 0 else '<td></td><td></td>'
                rows.append(f'<tr>{first}<td>{domon_costello_to_html(flat)}</td><td>{theo:.4f}</td><td align="right">{theo - mz:+.3f}</td><td align="right">{z}</td></tr>')
        self.fragments.setHtml(f'<table cellspacing="0" cellpadding="3" width="100%"><tr style="background:{WASH}"><th align="left">Observed m/z</th>'
                               f'<th align="right">Rel. int. (%)</th><th align="left">Fragment</th><th align="left">Theoretical m/z</th><th align="right">Error (Da)</th>'
                               f'<th align="right">z</th></tr>{"".join(rows)}</table>')


class SettingsPanel(QScrollArea):
    """Input files and every wrap_inference / wrap_inference_batch setting"""
    files_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(330)
        body = QWidget()
        body.setObjectName('settings')
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 8, 14, 14)
        layout.addWidget(_label('LC-MS/MS runs'))
        # Experiments (top level, each carrying its own settings as .settings) with their runs; only runs of one experiment are harmonized
        self.files, self.shown = QTreeWidget(), None
        self.files.setHeaderHidden(True)
        self.files.setRootIsDecorated(False)
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files.setMinimumHeight(110)
        self.files.setMaximumHeight(240)
        self.files.setToolTip('Runs of one experiment are predicted together and harmonized into one feature table; experiments are predicted and reported '
                              'separately, each with its own settings. Double-click an experiment to rename it, right-click runs to move them.')
        layout.addWidget(self.files)
        buttons = QGridLayout()
        self.add_files, self.add_folder, self.experiment_button, self.remove = QPushButton('Add files…'), QPushButton('Add folder…'), QPushButton('New experiment'), QPushButton('Remove')
        self.experiment_button.setToolTip('Moves the selected runs into a new experiment with its own settings, for samples that must not be compared with the others')
        for n, button in enumerate((self.add_files, self.add_folder, self.experiment_button, self.remove)):
            buttons.addWidget(button, n // 2, n % 2)
        layout.addLayout(buttons)
        self.file_hint = _label('Thermo .raw, mzML, mzXML, mgf, or an .xlsx made by extract_spectra. Drop files or folders anywhere in this window.', 'hint')
        layout.addWidget(self.file_hint)
        self.inputs = {}
        self.sample_heading = _label('Sample')
        layout.addWidget(self.sample_heading)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.inputs['glycan_class'] = _combo(GLYCAN_CLASSES)
        self.mode = _combo([('Negative', 'negative'), ('Positive', 'positive')])
        self.mode.setToolTip('Overridden by the polarity stored in .raw/mzML/mzXML files')
        self.charge = _spin(1, 8, 1)
        self.charge.setToolTip('Highest absolute precursor charge considered for composition matching')
        self.inputs['modification'] = _combo(REDUCING_ENDS)
        self.inputs['mass_tag'] = _spin(-1000, 3000, 1, 4, ' Da')
        self.inputs['mass_tag'].setToolTip('Mass a custom reducing-end label adds; 0 for a free reducing end')
        self.inputs['sample_prep'] = _combo(SAMPLE_PREPS)
        self.inputs['lc'] = _combo(LCS)
        self.inputs['trap'] = _combo(TRAPS)
        self.inputs['trap'].setToolTip('Overridden by the instrument stored in .raw/mzML/mzXML files')
        form.addRow('Glycans', self.inputs['glycan_class'])
        form.addRow('Ion mode', self.mode)
        form.addRow('Max. charge', self.charge)
        form.addRow('Reducing end', self.inputs['modification'])
        form.addRow('Tag mass', self.inputs['mass_tag'])
        form.addRow('Derivatization', self.inputs['sample_prep'])
        form.addRow('Chromatography', self.inputs['lc'])
        form.addRow('Mass analyzer', self.inputs['trap'])
        layout.addLayout(form)
        self.batch = QWidget()
        batch_layout = QVBoxLayout(self.batch)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(_label('Across runs'))
        form = QFormLayout()
        self.inputs['intra_cat_thresh'] = _spin(0.05, 10, 0.25, 2, ' min')
        self.inputs['intra_cat_thresh'].setToolTip('How far (in minutes) the retention time of a structure may drift between runs and still count as the same peak')
        self.inputs['top_n_isomers'] = _spin(1, 50, 1)
        self.inputs['top_n_isomers'].setToolTip('Isomer groups kept per composition across runs')
        self.inputs['n_jobs'] = _spin(1, max(1, os.cpu_count() or 1), 1)
        self.inputs['n_jobs'].setToolTip('Runs processed in parallel; each needs its own memory, and step-by-step progress is not shown for them')
        form.addRow('RT tolerance', self.inputs['intra_cat_thresh'])
        form.addRow('Isomers per composition', self.inputs['top_n_isomers'])
        form.addRow('Parallel runs', self.inputs['n_jobs'])
        batch_layout.addLayout(form)
        layout.addWidget(self.batch)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText('Advanced settings')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        form = QFormLayout(self.advanced)
        form.setContentsMargins(0, 0, 0, 0)
        self.inputs['rt_min'] = _spin(0, 1000, 1, 2, ' min')
        self.inputs['rt_max'] = _spin(0, 1000, 1, 2, ' min')
        self.inputs['rt_max'].setSpecialValueText('end of run')
        self.inputs['rt_diff'] = _spin(0.05, 10, 0.25, 2, ' min')
        self.inputs['rt_diff'].setToolTip('Maximum retention time difference to a peak apex that is still grouped with that peak')
        self.inputs['ppm_thresh'] = _spin(1, 5000, 10, 0, ' ppm')
        self.inputs['ppm_thresh'].setToolTip('Mass tolerance for grouping spectra, matching compositions, and the ppm error filter')
        self.inputs['pred_thresh'] = _spin(0, 1, 0.005, 3)
        self.inputs['pred_thresh'].setToolTip('Minimum prediction confidence')
        self.inputs['crumbs_thresh'] = _spin(0, 100, 1, 1)
        self.inputs['crumbs_thresh'].setToolTip('Minimum CandyCrumbs fragment annotation score to keep a prediction')
        self.inputs['extra_thresh'] = _spin(0, 1, 0.05, 2)
        self.inputs['extra_thresh'].setToolTip('Confidence from which structures of another glycan class are allowed')
        self.inputs['frag_num'] = _spin(1, 1000, 10)
        self.inputs['frag_num'].setToolTip('Number of top fragments reported per spectrum')
        self.inputs['supplement'] = QCheckBox('Impute biosynthetic intermediates')
        self.inputs['experimental'] = QCheckBox('Search databases for unexplained peaks')
        self.inputs['get_missing'] = QCheckBox('Keep peaks without a structure')
        self.inputs['taxonomy_level'] = QComboBox()
        self.inputs['taxonomy_level'].addItems(TAXONOMY_LEVELS)
        self.inputs['taxonomy_filter'] = QLineEdit()
        self.inputs['taxonomy_filter'].setToolTip('Database searches only use glycans of this taxon')
        self.taxa = QCompleter([])
        self.taxa.setCaseSensitivity(Qt.CaseInsensitive)
        self.taxa.setFilterMode(Qt.MatchContains)
        self.inputs['taxonomy_filter'].setCompleter(self.taxa)
        form.addRow('RT from', self.inputs['rt_min'])
        form.addRow('RT until', self.inputs['rt_max'])
        form.addRow('Peak width', self.inputs['rt_diff'])
        form.addRow('Mass tolerance', self.inputs['ppm_thresh'])
        form.addRow('Min. confidence', self.inputs['pred_thresh'])
        form.addRow('Min. fragment score', self.inputs['crumbs_thresh'])
        form.addRow('Cross-class confidence', self.inputs['extra_thresh'])
        form.addRow('Top fragments', self.inputs['frag_num'])
        form.addRow(self.inputs['supplement'])
        form.addRow(self.inputs['experimental'])
        form.addRow(self.inputs['get_missing'])
        form.addRow('Taxonomic level', self.inputs['taxonomy_level'])
        form.addRow('Taxon', self.inputs['taxonomy_filter'])
        grid = QGridLayout()
        self.filter_boxes = {}
        for n, name in enumerate(FILTER_CHOICES):
            self.filter_boxes[name] = QCheckBox(name)
            grid.addWidget(self.filter_boxes[name], n // 3, n % 3)
        form.addRow(_label('Ignore compositions containing', 'hint'))
        form.addRow(grid)
        self.reset = QPushButton('Restore defaults')
        form.addRow(self.reset)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)
        layout.addStretch(1)
        self.setWidget(body)
        self.advanced_toggle.toggled.connect(lambda on: (self.advanced.setVisible(on), self.advanced_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)))
        self.inputs['modification'].currentIndexChanged.connect(lambda: self.inputs['mass_tag'].setEnabled(self.inputs['modification'].currentData() == 'custom'))
        self.inputs['experimental'].toggled.connect(lambda on: (self.inputs['taxonomy_level'].setEnabled(on), self.inputs['taxonomy_filter'].setEnabled(on)))
        self.reset.clicked.connect(lambda: self.set_values(DEFAULTS))
        self.remove.clicked.connect(self.remove_selected)
        self.experiment_button.clicked.connect(
            lambda: self.move_runs([item for item in self.files.selectedItems() if item.parent() is not None],
                                   self.add_experiment(self.values()), rename = True))
        QShortcut(QKeySequence.Delete, self.files, self.remove_selected)
        self.files.currentItemChanged.connect(lambda: self.show_experiment(self.current_experiment()))
        self.files.itemChanged.connect(self.rename_experiment)
        self.files.customContextMenuRequested.connect(self.files_menu)
        self.files_changed.connect(lambda: self.show_experiment(self.current_experiment()))
        self.set_values(DEFAULTS)
        self.batch.setVisible(False)

    def current_experiment(self):
        """The experiment of the current item, else the last one"""
        item = self.files.currentItem() or self.files.topLevelItem(self.files.topLevelItemCount() - 1)
        return item if item is None or item.parent() is None else item.parent()

    def add_experiment(self, settings, name = None):
        names = {self.files.topLevelItem(i).text(0) for i in range(self.files.topLevelItemCount())}
        item = QTreeWidgetItem(
            [name or next(f'Experiment {n}' for n in itertools.count(1) if f'Experiment {n}' not in names)])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.settings = settings
        self.files.addTopLevelItem(item)
        item.setExpanded(True)
        return item

    def show_experiment(self, item):
        """Keeps the panel's values with the experiment they were shown for, then shows the settings of item"""
        if self.shown is not None and self.files.indexOfTopLevelItem(self.shown) >= 0:
            self.shown.settings = self.values()
        self.shown = item
        if item is not None:
            self.set_values(item.settings)
        self.sample_heading.setText('Sample' if item is None else f'Sample: {item.text(0)}')
        self.batch.setVisible(item is not None and item.childCount() > 1)

    def rename_experiment(self, item):
        if item.parent() is not None:
            return
        others = {self.files.topLevelItem(i).text(0) for i in range(self.files.topLevelItemCount()) if
                  self.files.topLevelItem(i) is not item}
        name = item.text(0).strip() or 'Experiment'
        name = next(x for x in itertools.chain([name], (f'{name} {n}' for n in itertools.count(2))) if x not in others)
        if name != item.text(0):
            item.setText(0, name)
        elif item is self.shown:
            self.sample_heading.setText(f'Sample: {name}')

    def move_runs(self, runs, experiment, rename = False):
        """Moves runs into experiment and shows it; experiments they leave empty are removed"""
        sources = [run.parent() for run in runs]
        for run in runs:
            run.parent().takeChild(run.parent().indexOfChild(run))
            experiment.addChild(run)
        self.files.setCurrentItem(experiment)
        for source in sources:
            if source is not experiment and not source.childCount() and self.files.indexOfTopLevelItem(source) >= 0:
                self.files.takeTopLevelItem(self.files.indexOfTopLevelItem(source))
        self.files_changed.emit()
        if rename:
            self.files.editItem(experiment, 0)

    def files_menu(self, position):
        runs = [item for item in self.files.selectedItems() if item.parent() is not None]
        if not runs:
            return
        menu = QMenu(self)
        for i in range(self.files.topLevelItemCount()):
            menu.addAction(f'Move to {self.files.topLevelItem(i).text(0)}',
                           lambda experiment = self.files.topLevelItem(i): self.move_runs(runs, experiment))
        menu.addAction('Move to a new experiment',
                       lambda: self.move_runs(runs, self.add_experiment(self.values()), rename = True))
        menu.exec(self.files.viewport().mapToGlobal(position))

    def select_experiment(self, name):
        found = self.files.findItems(name, Qt.MatchExactly)
        if found and found[0] is not self.current_experiment():
            self.files.setCurrentItem(found[0])

    def set_experiments(self, experiments):
        """Shows the experiments of saved results: their runs and the settings they were predicted with"""
        self.shown = None
        self.files.blockSignals(True)
        self.files.clear()
        self.files.blockSignals(False)
        for experiment in experiments:
            self.files.setCurrentItem(self.add_experiment(dict(experiment['settings']), experiment['name']))
            self.add_paths(experiment['files'])
        self.files.setCurrentItem(self.files.topLevelItem(0))

    def add_paths(self, paths):
        """Adds runs to the current experiment, or to a new one with the settings shown"""
        existing, target = set(self.paths()), self.current_experiment()
        for path in paths:
            found = [os.path.join(path, x) for x in sorted(os.listdir(path)) if
                     x.lower().endswith(SPECTRA_SUFFIXES)] if os.path.isdir(path) else [path]
            for file in found:
                if file.lower().endswith(SPECTRA_SUFFIXES + ('.xlsx',)) and os.path.abspath(file) not in existing:
                    existing.add(os.path.abspath(file))
                    if target is None:
                        target = self.add_experiment(self.values())
                    item = QTreeWidgetItem([os.path.basename(file)])
                    item.setData(0, Qt.UserRole, os.path.abspath(file))
                    item.setToolTip(0, os.path.abspath(file))
                    target.addChild(item)
        if target is not None and target is not self.current_experiment():
            self.files.setCurrentItem(target)
        self.files_changed.emit()

    def remove_selected(self):
        for item in self.files.selectedItems():
            if item.parent() is None:
                self.files.takeTopLevelItem(self.files.indexOfTopLevelItem(item))
            else:
                item.parent().removeChild(item)
        self.files_changed.emit()

    def experiments(self):
        """(name, paths, settings) of every experiment with runs, with the panel's values kept for the experiment they are shown for"""
        if self.shown is not None and self.files.indexOfTopLevelItem(self.shown) >= 0:
            self.shown.settings = self.values()
        items = [self.files.topLevelItem(i) for i in range(self.files.topLevelItemCount())]
        return [(e.text(0), [e.child(j).data(0, Qt.UserRole) for j in range(e.childCount())], e.settings) for e in items
                if e.childCount()]

    def paths(self):
        return [path for _, paths, _ in self.experiments() for path in paths]

    def values(self):
        values = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                values[key] = widget.currentData() if widget.currentData() is not None else widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
            else:
                values[key] = widget.value()
        values['max_charge'] = self.charge.value() * (-1 if self.mode.currentData() == 'negative' else 1)
        values['filter_out'] = {name for name, box in self.filter_boxes.items() if box.isChecked()}
        values['mass_tag'] = values['mass_tag'] if values['modification'] == 'custom' else None
        return values

    def set_values(self, values):
        for key, widget in self.inputs.items():
            value = values.get(key, DEFAULTS.get(key))
            if value is None:
                continue
            if isinstance(widget, QComboBox):
                _set_combo(widget, value) if widget.findData(value) >= 0 else widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            else:
                widget.setValue(value)
        charge = values.get('max_charge', DEFAULTS['max_charge'])
        _set_combo(self.mode, 'negative' if charge < 0 else 'positive')
        self.charge.setValue(abs(charge))
        for name, box in self.filter_boxes.items():
            box.setChecked(name in set(values.get('filter_out', DEFAULTS['filter_out'])))
        self.inputs['mass_tag'].setEnabled(self.inputs['modification'].currentData() == 'custom')


class ResultsView(QWidget):
    """Result tables with SNFG drawings, a detail pane for the selected glycan peak, its annotated MS2 spectrum, a glycan map, and curation"""
    open_in_crumbs = Signal(object)
    status = Signal(str)
    experiment_shown = Signal(str)

    def __init__(self, images):
        super().__init__()
        self.images, self.payload, self.frame, self.row, self.source, self.glytoucan = images, None, None, None, None, None
        self.alternative, self.ms3_groups, self.ms3_args = None, [], None
        top = QHBoxLayout()
        top.setContentsMargins(8, 6, 8, 4)
        self.experiment = QComboBox()
        self.experiment.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.experiment.setToolTip('Experiments are predicted and harmonized separately')
        self.dataset = QComboBox()
        self.dataset.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter by structure, composition, m/z, GlyTouCan ID, or notes')
        self.search.setClearButtonEnabled(True)
        self.counter = _label('', 'hint')
        self.export = QToolButton()
        self.export.setText('Export')
        self.export.setPopupMode(QToolButton.InstantPopup)
        self.export_menu = QMenu(self)
        self.export.setMenu(self.export_menu)
        top.addWidget(self.experiment)
        top.addWidget(self.dataset)
        top.addWidget(self.search, 1)
        top.addWidget(self.counter)
        top.addWidget(self.export)
        self.model, self.proxy = TableModel(images), TableProxy()
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(58)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.structure_delegate, self.bar_delegate = StructureDelegate(images, self.table), BarDelegate(self.table)
        QShortcut(QKeySequence.Copy, self.table, self.copy_rows)
        # Detail pane: drawing, facts, alternatives, and curation of the selected peak
        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 8)
        self.drawing = QLabel()
        self.drawing.setAlignment(Qt.AlignCenter)
        self.drawing.setMinimumHeight(90)
        # A read-only text box rather than a label: IUPAC strings have no spaces to wrap at, and copying must give the plain string
        self.structure = QPlainTextEdit()
        self.structure.setReadOnly(True)
        self.structure.setFrameShape(QFrame.NoFrame)
        self.structure.setWordWrapMode(QTextOption.WrapAnywhere)
        self.structure.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.structure.viewport().setAutoFillBackground(False)
        self.composition = QLabel()
        self.composition.setStyleSheet('font-weight: 600')
        self.facts = QLabel()
        self.facts.setOpenExternalLinks(True)
        self.facts.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.facts.setWordWrap(True)
        self.alternatives = QListWidget()
        self.alternatives.setIconSize(QSize(170, 40))
        self.alternatives.setMinimumHeight(100)
        self.alternatives.setToolTip('Click a candidate to annotate the spectrum with it; double-click to assign it to this peak')
        actions = QGridLayout()
        self.assign, self.custom, self.exclude, self.crumbs = QPushButton('Assign candidate'), QPushButton('Enter structure…'), QPushButton('Exclude'), QPushButton('Open in CandyCrumbs')
        self.assign.setToolTip('Make the selected candidate this peak\'s structure (top1_pred) in the table and its exports')
        self.custom.setToolTip('Assign a structure that is not among the candidates')
        self.exclude.setToolTip('Leave this peak out of exports')
        actions.addWidget(self.assign, 0, 0)
        actions.addWidget(self.custom, 0, 1)
        actions.addWidget(self.exclude, 1, 0)
        actions.addWidget(self.crumbs, 1, 1)
        info_layout.addWidget(self.drawing)
        info_layout.addWidget(self.structure)
        info_layout.addWidget(self.composition)
        info_layout.addWidget(self.facts)
        info_layout.addWidget(_label('Candidates'))
        info_layout.addWidget(self.alternatives, 1)
        info_layout.addLayout(actions)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setWidget(info)
        info_scroll.setMinimumWidth(300)
        self.spectrum = SpectrumPanel()
        self.spectrum_file = QComboBox()
        self.spectrum_file.setToolTip('Run whose MS2 spectrum is shown for this feature')
        self.spectrum.controls.insertWidget(1, self.spectrum_file)
        # MS3 spectra of the selected peak, pooled per isolated MS2 fragment, annotated as fragments of that fragment
        self.ms3_panel, self.ms3_fragment, self.ms3_tab = SpectrumPanel(), QComboBox(), QSplitter(Qt.Vertical)
        self.ms3_fragment.setToolTip('MS2 fragment whose MS3 spectra are shown (pooled)')
        self.ms3_panel.controls.insertWidget(1, self.ms3_fragment)
        self.ms3_tab.addWidget(self.ms3_panel)
        self.ms3_tab.addWidget(self.ms3_panel.fragments)
        self.ms3_tab.setSizes([320, 120])
        self.map, self.across = ChartView(), ChartView()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.spectrum, 'MS2 spectrum')
        self.tabs.addTab(self.spectrum.fragments, 'Fragments')
        self.tabs.addTab(self.ms3_tab, 'MS3 spectrum')
        self.tabs.setTabVisible(2, False)
        self.tabs.addTab(self.map, 'Glycan map')
        self.tabs.addTab(self.across, 'Across runs')
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(info_scroll)
        bottom.addWidget(self.tabs)
        bottom.setStretchFactor(1, 1)
        bottom.setSizes([380, 900])
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(bottom)
        self.splitter.setSizes([380, 420])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.splitter, 1)
        self.experiment.currentIndexChanged.connect(self.show_experiment)
        self.dataset.currentIndexChanged.connect(lambda: self.show_dataset())
        self.search.textChanged.connect(self.filter)
        self.table.selectionModel().selectionChanged.connect(self.selection_changed)
        self.table.customContextMenuRequested.connect(self.context_menu)
        self.alternatives.itemClicked.connect(self.pick_alternative)
        self.alternatives.itemDoubleClicked.connect(lambda item: self.curate(item.data(Qt.UserRole), 'reassigned'))
        self.assign.clicked.connect(lambda: self.alternatives.currentItem() and self.curate(self.alternatives.currentItem().data(Qt.UserRole), 'reassigned'))
        self.custom.clicked.connect(self.enter_structure)
        self.exclude.clicked.connect(self.toggle_excluded)
        self.crumbs.clicked.connect(self.send_to_crumbs)
        self.spectrum_file.currentIndexChanged.connect(self.show_source)
        self.spectrum.ms3_clicked.connect(self.open_ms3)
        self.ms3_fragment.currentIndexChanged.connect(self.annotate_ms3)
        self.tabs.currentChanged.connect(lambda: self.tabs.currentWidget() is self.ms3_tab and self.annotate_ms3())
        self.map.canvas.mpl_connect('pick_event', lambda event: len(event.ind) and self.select_row(int(event.ind[0])))
        images.ready.connect(self.image_ready)

    def set_results(self, session):
        """Takes a worker result or a saved session (older ones hold a single experiment); tables are kept with m/z as a column and get the curation columns once"""
        if 'experiments' not in session:
            session = {'experiments': [{'name': 'Experiment 1', **{k: session[k] for k in ('files', 'settings', 'tables', 'features')}}],
                       **{k: v for k, v in session.items() if k not in ('files', 'settings', 'tables', 'features')}}
        for experiment in session['experiments']:
            for label, (df, spectra) in experiment['tables'].items():
                df = df.reset_index() if df.index.name == 'm/z' else df
                for column, empty in (('curation', ''), ('excluded', False)):
                    if column not in df.columns:
                        df[column] = empty
                experiment['tables'][label] = (df, list(spectra) + [None] * (len(df) - len(spectra)))
            if experiment['features'] is not None:
                for column, empty in (('curation', ''), ('excluded', False)):
                    if column not in experiment['features'].columns:
                        experiment['features'][column] = empty
        self.session = session
        self.experiment.blockSignals(True)
        self.experiment.clear()
        self.experiment.addItems([experiment['name'] for experiment in session['experiments']])
        self.experiment.blockSignals(False)
        self.experiment.setVisible(self.experiment.count() > 1)
        self.show_experiment()

    def show_experiment(self):
        """self.payload is the experiment shown: its files, settings, tables, and feature table"""
        payload = self.payload = self.session['experiments'][max(self.experiment.currentIndex(), 0)]
        self.dataset.blockSignals(True)
        self.dataset.clear()
        if payload['features'] is not None:
            self.dataset.addItem(f'All {len(payload["tables"])} runs (feature table)', FEATURES)
        for label in payload['tables']:
            self.dataset.addItem(label, label)
        self.dataset.blockSignals(False)
        self.dataset.setVisible(self.dataset.count() > 1)
        self.export_menu.clear()
        for text, kind in (('This table as CSV…', 'csv'), ('This table as Excel…', 'xlsx'), ('This table as Excel with SNFG drawings…', 'snfg')):
            self.export_menu.addAction(text, lambda kind = kind: self.export_table(kind))
        if payload['features'] is not None:
            self.export_menu.addAction('All tables as CSV…', lambda: self.export_all('.csv'))
            self.export_menu.addAction('All tables as Excel…', lambda: self.export_all('.xlsx'))
        self.show_dataset()
        self.experiment_shown.emit(payload['name'])

    def current_frame(self):
        label = self.dataset.currentData()
        return self.payload['features'] if label == FEATURES else self.payload['tables'][label][0]

    def show_dataset(self, keep_row = None):
        label = self.dataset.currentData()
        if label is None:
            return
        df = self.current_frame()
        self.frame, self.row = df, None
        self.tabs.setTabVisible(self.tabs.indexOf(self.across), label == FEATURES)
        if label != FEATURES and self.tabs.currentWidget() is self.across:
            self.tabs.setCurrentIndex(0)
        structures = list(df['top1_pred'])
        compositions = [_composition(c) for c in df['composition']]
        notes_tip = 'Notes from the pipeline, adducts, and your curation'
        columns = [{'title': 'Structure', 'kind': 'structure', 'raw': [s if isinstance(s, str) else None for s in structures],
                    'text': [s if isinstance(s, str) else '' for s in structures]},
                   {'title': 'Composition', 'kind': 'text', 'raw': compositions, 'text': compositions},
                   _number_column('m/z', df['m/z'], 4), _number_column('z', df['charge'], 0, 'Precursor charge'), _number_column('RT', df['RT'], 2, 'Retention time (min)')]
        if label == FEATURES:
            columns.append(_number_column('MS2 runs', df['n_files_ms2'], 0, 'Runs in which this feature has MS2 evidence'))
            for run in self.payload['tables']:
                if run in df.columns:
                    column = _number_column(run, df[run], 2, f'Abundance in {run} (%); hatched where MS1 signal only')
                    evidence = list(df.get(f'evidence_{run}', [None] * len(df)))
                    top = max([v for v in column['raw'] if v is not None] or [1]) or 1
                    column.update(kind = 'bar', bar = [None if v is None else (v / top, e == 'ms1_only') for v, e in zip(column['raw'], evidence)],
                                  tips = [f'{t} ({e})' if isinstance(e, str) else t for t, e in zip(column['text'], evidence)])
                    columns.append(column)
            notes = [c for c in df['curation']]
        else:
            abundance = 'rel_abundance' if 'rel_abundance' in df.columns else 'num_spectra'
            column = _number_column('Abundance' if abundance == 'rel_abundance' else 'Spectra', df[abundance], 2 if abundance == 'rel_abundance' else 0,
                                    'Relative abundance (%), from MS1 peak areas where the file has MS1 scans, else from precursor intensities')
            top = max([v for v in column['raw'] if v is not None] or [1]) or 1
            evidence = list(df.get('evidence', [None] * len(df)))
            column.update(kind = 'bar', bar = [None if v is None else (v / top, e == 'ms1_only') for v, e in zip(column['raw'], evidence)])
            columns.append(column)
            columns.append(_number_column('Confidence', [p[0][1] if isinstance(p, (list, tuple)) and p else None for p in df['predictions']], 2,
                                          'CandyCrunch confidence of the structure shown'))
            columns.append(_number_column('Fragment score', df.get('annotation_score', [None] * len(df)), 0, 'CandyCrumbs fragment annotation score'))
            columns.append(_number_column('Spectra', df['num_spectra'], 0, 'MS2 spectra pooled into this peak'))
            if 'ms3' in df.columns:
                columns.append(_number_column('MS3', [len({round(p) for p, _ in x}) if isinstance(x, list) and x else None for x in df['ms3']], 0,
                                              'MS2 fragments of this peak with MS3 spectra'))
            columns.append(_number_column('ppm', df.get('ppm_error', [None] * len(df)), 0, 'Precursor mass error'))
            texts = [e if isinstance(e, str) else '' for e in evidence]
            columns.append({'title': 'Evidence', 'kind': 'text', 'raw': texts, 'text': texts})
            notes = ['; '.join(str(x) for x in (n, f'adduct {a}' if isinstance(a, str) and a else '', c) if isinstance(x, str) and x)
                     for n, a, c in zip(df.get('notes', [''] * len(df)), df.get('adduct', [''] * len(df)), df['curation'])]
        ids = [g if isinstance(g, str) else '' for g in df.get('GlyTouCan_ID', [''] * len(df))]
        columns.append({'title': 'GlyTouCan', 'kind': 'text', 'raw': ids, 'text': ids})
        columns.append({'title': 'Notes', 'kind': 'text', 'raw': notes, 'text': notes, 'tip': notes_tip})
        sort_column, sort_order = self.table.horizontalHeader().sortIndicatorSection(), self.table.horizontalHeader().sortIndicatorOrder()
        self.model.set_table(df, columns)
        for c, column in enumerate(columns):
            self.table.setItemDelegateForColumn(c, self.structure_delegate if column['kind'] == 'structure' else self.bar_delegate if column['kind'] == 'bar' else None)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, 250)
        for c, column in enumerate(columns):
            if column['kind'] == 'bar':
                self.table.setColumnWidth(c, max(90, self.table.columnWidth(c)))
            elif column['title'] == 'Composition':
                self.table.setColumnWidth(c, min(self.table.columnWidth(c), 210))
        if keep_row is None:
            self.table.sortByColumn(2, Qt.AscendingOrder)
        else:
            self.table.sortByColumn(sort_column, sort_order)
        self.filter(self.search.text())
        self.draw_map()
        self.select_row(keep_row if keep_row is not None else (self.proxy.mapToSource(self.proxy.index(0, 0)).row() if self.proxy.rowCount() else None))

    def filter(self, text):
        self.proxy.set_needle(text)
        shown, total = self.proxy.rowCount(), self.model.rowCount()
        self.counter.setText(f'{total} glycan peaks' if shown == total else f'{shown} of {total} glycan peaks')

    def select_row(self, row):
        if row is None:
            self.row = None
            self.spectrum.show_message('No glycan peaks to show')
            return
        index = self.proxy.mapFromSource(self.model.index(row, 0))
        if not index.isValid():
            self.search.clear()
            index = self.proxy.mapFromSource(self.model.index(row, 0))
        self.table.selectRow(index.row())
        self.table.scrollTo(index)

    def selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_row(self.proxy.mapToSource(rows[0]).row())

    def show_row(self, row):
        self.row, self.alternative = row, None
        df = self.frame
        if self.dataset.currentData() == FEATURES:
            # The feature table keeps no link to the per-run rows, so the row of each run is the one with the same structure closest in m/z and RT
            self.spectrum_file.blockSignals(True)
            self.spectrum_file.clear()
            best = None
            for run, (table, spectra) in self.payload['tables'].items():
                evidence = df[f'evidence_{run}'].iat[row] if f'evidence_{run}' in df.columns else None
                if not isinstance(evidence, str) or evidence == 'ms1_only' or table.empty:
                    continue
                same = (table['top1_pred'] == df['top1_pred'].iat[row]) if isinstance(df['top1_pred'].iat[row], str) else table['top1_pred'].isna()
                distance = (table['m/z'] - df['m/z'].iat[row]).abs() * 10 + (table['RT'] - df['RT'].iat[row]).abs()
                distance = distance[same & ((table['m/z'] - df['m/z'].iat[row]).abs() < 1)]
                if len(distance):
                    self.spectrum_file.addItem(run, (run, int(distance.idxmin())))
                    value = df[run].iat[row] if run in df.columns else 0
                    if best is None or (value or 0) > best[0]:
                        best = (value or 0, self.spectrum_file.count() - 1)
            if best is not None:
                self.spectrum_file.setCurrentIndex(best[1])
            self.spectrum_file.blockSignals(False)
            self.spectrum_file.setVisible(True)
        else:
            self.spectrum_file.setVisible(False)
        self.update_detail()
        self.show_source()
        self.draw_selection()

    def update_detail(self):
        df, row = self.frame, self.row
        structure = df['top1_pred'].iat[row]
        self.structure.setPlainText(structure if isinstance(structure, str) else 'No structure assigned')
        self.structure.setFixedHeight(int(self.structure.document().size().height() * self.structure.fontMetrics().lineSpacing()) + 10)
        self.composition.setText(_composition(df['composition'].iat[row]))
        self.update_drawing()
        facts = [('m/z', _fmt(df['m/z'].iat[row], 4)), ('Charge', _fmt(df['charge'].iat[row], 0)), ('RT', f'{_fmt(df["RT"].iat[row], 2)} min')]
        if self.dataset.currentData() == FEATURES:
            facts.append(('MS2 runs', f'{df["n_files_ms2"].iat[row]} of {len(self.payload["tables"])}'))
        else:
            preds = df['predictions'].iat[row]
            facts += [('Abundance', f'{_fmt(df["rel_abundance"].iat[row], 2)} %' if 'rel_abundance' in df.columns else ''),
                      ('Confidence', _fmt(preds[0][1], 3) if isinstance(preds, (list, tuple)) and preds else ''),
                      ('Fragment score', _fmt(df['annotation_score'].iat[row], 0) if 'annotation_score' in df.columns else ''),
                      ('Spectra', str(df['num_spectra'].iat[row])), ('ppm error', _fmt(df['ppm_error'].iat[row], 1) if 'ppm_error' in df.columns else ''),
                      ('Evidence', str(df['evidence'].iat[row]) if 'evidence' in df.columns and isinstance(df['evidence'].iat[row], str) else ''),
                      ('Notes', df['notes'].iat[row] if 'notes' in df.columns and isinstance(df['notes'].iat[row], str) else '')]
        glytoucan = df['GlyTouCan_ID'].iat[row] if 'GlyTouCan_ID' in df.columns else ''
        if isinstance(glytoucan, str) and glytoucan:
            facts.append(('GlyTouCan', f'<a href="https://glytoucan.org/Structures/Glycans/{glytoucan}">{glytoucan}</a>'))
        if df['curation'].iat[row]:
            facts.append(('Curation', df['curation'].iat[row]))
        facts = [(k, v) for k, v in facts if v]
        cells = [f'<td style="color:{MUTED}; padding-right:8px; white-space:nowrap">{k}</td><td style="padding-right:12px; white-space:nowrap">{v}</td>' for k, v in facts]
        # Short facts pair up in two columns; notes and the like get a row of their own
        rows, pending = [], []
        for (k, v), cell in zip(facts, cells):
            if len(str(v)) > 20 and not str(v).startswith('<a'):
                rows.append(f'<tr><td style="color:{MUTED}; padding-right:8px">{k}</td><td colspan="3">{v}</td></tr>')
            else:
                pending.append(cell)
                if len(pending) == 2:
                    rows.append(f'<tr>{"".join(pending)}</tr>')
                    pending = []
        rows += [f'<tr>{pending[0]}</tr>'] if pending else []
        self.facts.setText('<table cellspacing="0" cellpadding="1">' + ''.join(rows) + '</table>')
        self.exclude.setText('Include again' if df['excluded'].iat[row] else 'Exclude')

    def update_drawing(self):
        structure = self.alternative or self.frame['top1_pred'].iat[self.row]
        image = self.images.get(structure, compact = False, urgent = True)
        if image is None:
            self.drawing.setText('Drawing…')
        elif image.isNull():
            self.drawing.clear()
        else:
            ratio = self.devicePixelRatioF()
            width = min(image.width() * 0.3, max(120, self.drawing.width() - 10))
            pixmap = QPixmap.fromImage(image.scaledToWidth(int(width * ratio), Qt.SmoothTransformation))
            pixmap.setDevicePixelRatio(ratio)
            self.drawing.setPixmap(pixmap)

    def show_source(self):
        """Candidates and spectrum of the selected per-run row (for a feature: of the run picked in the spectrum combobox)"""
        if self.row is None:
            return
        if self.dataset.currentData() == FEATURES:
            self.source = self.spectrum_file.currentData()
        else:
            self.source = (self.dataset.currentData(), self.row)
        self.alternatives.clear()
        if self.source is None:
            self.spectrum.show_message('This feature has MS2 evidence in no run, only MS1 signal')
            return
        table, spectra = self.payload['tables'][self.source[0]]
        preds = table['predictions'].iat[self.source[1]]
        for structure, score in (preds if isinstance(preds, (list, tuple)) else []):
            item = QListWidgetItem(f'{score:.3f}')
            item.setData(Qt.UserRole, structure)
            item.setToolTip(structure)
            self.alternatives.addItem(item)
        self.update_alternative_icons()
        self.annotate()

    def update_alternative_icons(self):
        for i in range(self.alternatives.count()):
            item = self.alternatives.item(i)
            image = self.images.get(item.data(Qt.UserRole), urgent = True)
            if image is not None and not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            elif image is not None:
                item.setText(f'{item.text().split()[0]}  {item.data(Qt.UserRole)}')

    def annotate(self):
        table, spectra = self.payload['tables'][self.source[0]]
        r = self.source[1]
        peaks, settings = spectra[r], self.payload['settings']
        structure = self.alternative or table['top1_pred'].iat[r]
        if not isinstance(structure, str):
            structure = _composition(table['composition'].iat[r]) or None
        # MS3 spectra pooled per isolated MS2 fragment (rounded to 1 m/z, as the pipeline scores them), merging peaks within half the mass tolerance
        ms3, groups, self.ms3_groups = table['ms3'].iat[r] if 'ms3' in table.columns else None, {}, []
        for p, spectrum in ms3 if isinstance(ms3, list) and isinstance(peaks, dict) and peaks and structure else []:
            groups.setdefault(round(p), []).append((p, spectrum))
        for _, members in sorted(groups.items()):
            pairs, pooled, group = sorted((m, i) for _, spectrum in members for m, i in spectrum.items()), {}, []
            for m, i in pairs + [(np.inf, 0)]:
                if group and m - group[0][0] > settings['ppm_thresh'] * MZ_REF / 2e6:
                    ms, ints = zip(*group)
                    pooled[float(np.average(ms, weights = ints)) if sum(ints) else float(np.mean(ms))] = float(
                        sum(ints))
                    group = []
                group.append((m, i))
            if pooled:
                self.ms3_groups.append((float(np.median([p for p, _ in members])), pooled, len(members)))
        self.ms3_args = (structure, int(table['charge'].iat[r]))
        self.ms3_fragment.blockSignals(True)
        self.ms3_fragment.clear()
        for p, _, n in self.ms3_groups:
            self.ms3_fragment.addItem(f'Fragment m/z {p:.2f}, {n} spectr{"um" if n == 1 else "a"}')
        self.ms3_fragment.blockSignals(False)
        if not self.ms3_groups and self.tabs.currentWidget() is self.ms3_tab:
            self.tabs.setCurrentIndex(0)
        self.tabs.setTabVisible(self.tabs.indexOf(self.ms3_tab), bool(self.ms3_groups))
        if not isinstance(peaks, dict) or not peaks:
            self.spectrum.show_message('No MS2 spectrum: this peak was only found by its MS1 signal')
        elif structure is None:
            self.spectrum.show_message('Neither a structure nor a composition to annotate this spectrum with')
        else:
            self.spectrum.annotate(structure, list(peaks.keys()), list(peaks.values()), int(table['charge'].iat[r]),
                                   _mass_tag(settings), settings['sample_prep'],
                                   ms3 = [p for p, _, _ in self.ms3_groups])
            if self.tabs.currentWidget() is self.ms3_tab:
                self.annotate_ms3()

    def open_ms3(self, index):
        self.ms3_fragment.blockSignals(True)
        self.ms3_fragment.setCurrentIndex(index)
        self.ms3_fragment.blockSignals(False)
        if self.tabs.currentWidget() is self.ms3_tab:
            self.annotate_ms3()
        else:
            self.tabs.setCurrentWidget(self.ms3_tab)

    def annotate_ms3(self):
        if 0 <= self.ms3_fragment.currentIndex() < len(self.ms3_groups):
            p, peaks, _ = self.ms3_groups[self.ms3_fragment.currentIndex()]
            settings = self.payload['settings']
            self.ms3_panel.annotate(self.ms3_args[0], list(peaks.keys()), list(peaks.values()), self.ms3_args[1],
                                    _mass_tag(settings), settings['sample_prep'],
                                    ms3_precursor = p)

    def pick_alternative(self, item):
        self.alternative = item.data(Qt.UserRole)
        self.update_drawing()
        self.annotate()

    def image_ready(self, key):
        if self.row is None or self.frame is None:
            return
        if key[1] is False and key[0] == (self.alternative or self.frame['top1_pred'].iat[self.row]):
            self.update_drawing()
        if key[1] is True and any(self.alternatives.item(i).data(Qt.UserRole) == key[0] for i in range(self.alternatives.count())):
            self.update_alternative_icons()

    def draw_map(self):
        df, figure = self.frame, self.map.figure
        figure.clear()
        ax = figure.add_axes([0.08, 0.14, 0.78, 0.8])
        if self.dataset.currentData() == FEATURES:
            color, color_label = df['n_files_ms2'].astype(float).values, 'Runs with MS2'
            size = df[[run for run in self.payload['tables'] if run in df.columns]].fillna(0).mean(axis = 1).values
        else:
            color, color_label = np.array([p[0][1] if isinstance(p, (list, tuple)) and p else np.nan for p in df['predictions']], dtype = float), 'Confidence'
            size = (df['rel_abundance'] if 'rel_abundance' in df.columns else df['num_spectra']).astype(float).values
        size = 12 + 380 * np.sqrt(np.nan_to_num(size) / (np.nanmax(size) or 1))
        known = ~np.isnan(color)
        ax.scatter(df['RT'].values[~known], df['m/z'].values[~known], s = size[~known], c = '#cccccc', edgecolors = 'white', linewidths = 0.5)
        points = ax.scatter(df['RT'].values, df['m/z'].values, s = size, c = np.where(known, color, np.nan), cmap = CONFIDENCE_MAP,
                            edgecolors = 'white', linewidths = 0.5, alpha = 0.9, picker = 4)
        if known.any():
            colorbar = figure.colorbar(points, cax = figure.add_axes([0.89, 0.14, 0.02, 0.8]))
            colorbar.set_label(color_label, fontsize = 8)
            colorbar.ax.tick_params(labelsize = 7)
        self.highlight = ax.scatter([], [], s = 0, facecolors = 'none', edgecolors = INK, linewidths = 1.4)
        ax.set_xlabel('Retention time (min)', fontsize = 8)
        ax.set_ylabel('m/z', fontsize = 8)
        ax.tick_params(labelsize = 7)
        ax.spines[['top', 'right']].set_visible(False)
        self.map.canvas.draw_idle()

    def draw_selection(self):
        df, row = self.frame, self.row
        self.highlight.set_offsets([[df['RT'].iat[row], df['m/z'].iat[row]]])
        self.highlight.set_sizes([140])
        self.map.canvas.draw_idle()
        if self.dataset.currentData() != FEATURES:
            return
        figure = self.across.figure
        figure.clear()
        ax = figure.add_axes([0.25, 0.15, 0.7, 0.8])
        runs = [run for run in self.payload['tables'] if run in df.columns]
        values = [0 if pd.isna(df[run].iat[row]) else df[run].iat[row] for run in runs]
        evidence = [df[f'evidence_{run}'].iat[row] if f'evidence_{run}' in df.columns else None for run in runs]
        bars = ax.barh(range(len(runs)), values, height = 0.6, color = [LILAC if e != 'ms1_only' else 'white' for e in evidence], edgecolor = LILAC, linewidth = 0.8)
        for bar, e, value in zip(bars, evidence, values):
            bar.set_hatch('////' if e == 'ms1_only' else None)
            ax.annotate(f'{value:.2f}' if isinstance(e, str) else 'not found', (value, bar.get_y() + bar.get_height() / 2), xytext = (4, 0),
                        textcoords = 'offset points', va = 'center', fontsize = 7, color = INK if isinstance(e, str) else MUTED)
        ax.set_yticks(range(len(runs)), runs, fontsize = 7)
        ax.invert_yaxis()
        ax.set_xlabel('Relative abundance (%); hatched: MS1 signal only', fontsize = 8)
        ax.tick_params(axis = 'x', labelsize = 7)
        ax.spines[['top', 'right']].set_visible(False)
        self.across.canvas.draw_idle()

    def refresh(self):
        self.show_dataset(keep_row = self.row)

    def curate(self, structure, how):
        if self.row is None or not isinstance(structure, str):
            return
        if self.glytoucan is None:
            self.glytoucan = pickle.load(open(os.path.join(os.path.dirname(__file__), 'glytoucan_mapping.pkl'), 'rb'))
        df = self.frame
        df.loc[df.index[self.row], ['top1_pred', 'curation']] = [structure, how]
        if 'GlyTouCan_ID' in df.columns:
            df.loc[df.index[self.row], 'GlyTouCan_ID'] = self.glytoucan.get(structure, '')
        self.status.emit(f'Assigned {structure}')
        self.refresh()

    def enter_structure(self):
        if self.row is None:
            return
        current = self.frame['top1_pred'].iat[self.row]
        text, ok = QInputDialog.getText(self, 'Enter structure', 'Structure in IUPAC-condensed nomenclature:', text = current if isinstance(current, str) else '')
        if not ok or not text.strip():
            return
        from glycowork.motif.processing import canonicalize_iupac
        from glycowork.motif.tokenization import glycan_to_composition
        try:
            structure = canonicalize_iupac(text.strip())
            composition = glycan_to_composition(structure)
        except Exception as error:
            QMessageBox.warning(self, 'Enter structure', f'glycowork cannot read this structure:\n{error}')
            return
        expected = self.frame['composition'].iat[self.row]
        if isinstance(expected, dict) and expected and composition != expected:
            answer = QMessageBox.question(self, 'Enter structure', f'{structure} is {_composition(composition)}, but this peak was matched to '
                                          f'{_composition(expected)}. Assign it anyway?')
            if answer != QMessageBox.Yes:
                return
        self.curate(structure, 'entered by hand')

    def toggle_excluded(self):
        if self.row is not None:
            df = self.frame
            df.loc[df.index[self.row], 'excluded'] = not df['excluded'].iat[self.row]
            self.refresh()

    def context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        menu = QMenu(self)
        menu.addAction('Copy row', self.copy_rows)
        menu.addAction('Include again' if self.frame['excluded'].iat[self.row] else 'Exclude from export', self.toggle_excluded)
        menu.addAction('Enter structure…', self.enter_structure)
        menu.addAction('Open in CandyCrumbs', self.send_to_crumbs)
        menu.exec(self.table.viewport().mapToGlobal(position))

    def copy_rows(self):
        if self.row is None:
            return
        header = '\t'.join(c['title'] for c in self.model.columns)
        QApplication.clipboard().setText(header + '\n' + '\t'.join(str(c['text'][self.row]) for c in self.model.columns))

    def send_to_crumbs(self):
        if self.row is None or self.source is None:
            return
        table, spectra = self.payload['tables'][self.source[0]]
        r = self.source[1]
        structure = self.alternative or table['top1_pred'].iat[r]
        self.open_in_crumbs.emit({'structure': structure if isinstance(structure, str) else _composition(table['composition'].iat[r]),
                                  'peaks': spectra[r] or {}, 'charge': int(table['charge'].iat[r]), 'settings': self.payload['settings']})

    def export_frame(self, df):
        """What gets written: curated rows as edited, excluded rows dropped, the curation column only if anything was curated"""
        out = df[~df['excluded'].astype(bool)].drop(columns = ['excluded', 'ms3'], errors = 'ignore')
        return out.drop(columns = ['curation']) if not out['curation'].astype(bool).any() else out

    def write(self, df, path, kind):
        if kind == 'snfg':
            from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df, path, glycan_col_num = 'top1_pred')
        elif path.lower().endswith('.csv'):
            df.to_csv(path, index = False)
        else:
            df.map(lambda v: str(v) if isinstance(v, (list, tuple, dict, set)) else v).to_excel(path, index = False)

    def export_table(self, kind):
        label = self.dataset.currentData()
        stem = 'features' if label == FEATURES else label
        suffix = '.csv' if kind == 'csv' else '.xlsx'
        path, _ = QFileDialog.getSaveFileName(self, 'Export table', os.path.join(QSettings().value('folder', os.path.expanduser('~')), stem + suffix),
                                              'CSV (*.csv)' if kind == 'csv' else 'Excel (*.xlsx)')
        if path:
            self.status.emit('Exporting…')
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.write(self.export_frame(self.current_frame()), path, kind)
                self.status.emit(f'Exported {os.path.basename(path)}')
            except Exception as error:
                QMessageBox.critical(self, 'Export', f'Could not write {path}:\n{error}')
            finally:
                QApplication.restoreOverrideCursor()

    def export_all(self, suffix):
        """Same file layout as candycrunch_predict: the feature table, then <name>_<run> per run"""
        path, _ = QFileDialog.getSaveFileName(self, 'Export all tables', os.path.join(QSettings().value('folder', os.path.expanduser('~')), 'candycrunch' + suffix),
                                              'CSV (*.csv)' if suffix == '.csv' else 'Excel (*.xlsx)')
        if not path:
            return
        stem = os.path.splitext(path)[0]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.write(self.export_frame(self.payload['features']), stem + suffix, suffix)
            for run, (table, _) in self.payload['tables'].items():
                if not table.empty:
                    self.write(self.export_frame(table), f'{stem}_{run}{suffix}', suffix)
            self.status.emit(f'Exported {len(self.payload["tables"]) + 1} tables next to {os.path.basename(path)}')
        except Exception as error:
            QMessageBox.critical(self, 'Export', f'Could not write {path}:\n{error}')
        finally:
            QApplication.restoreOverrideCursor()


class CrumbsTab(QWidget):
    """Stand-alone CandyCrumbs: annotate any peak list with any glycan, composition, or glycopeptide"""

    def __init__(self, images):
        super().__init__()
        self.images = images
        form_widget = QWidget()
        form_widget.setObjectName('settings')
        form_widget.setMinimumWidth(330)
        form_widget.setMaximumWidth(420)
        layout = QVBoxLayout(form_widget)
        layout.setContentsMargins(14, 8, 14, 14)
        layout.addWidget(_label('Structure'))
        self.structure = QLineEdit()
        self.structure.setPlaceholderText('Fuc(a1-2)Gal(b1-3)GalNAc, a composition, or a glycopeptide')
        layout.addWidget(self.structure)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(70)
        layout.addWidget(self.preview)
        layout.addWidget(_label('Peaks'))
        self.peaks = QPlainTextEdit()
        self.peaks.setPlaceholderText('One peak per line: m/z, then intensity if you have it.\nColumns can be separated by tabs, commas, or spaces, so '
                                      'pasting straight from Excel or a spectrum viewer works.')
        layout.addWidget(self.peaks, 1)
        form = QFormLayout()
        self.charge = _combo([(f'{z:+d}', z) for z in (-1, -2, -3, -4, 1, 2, 3, 4)])
        self.modification = _combo(REDUCING_ENDS)
        self.mass_tag = _spin(-1000, 3000, 1, 4, ' Da')
        self.sample_prep = _combo(SAMPLE_PREPS)
        self.cleavages = _spin(1, 4, 1)
        self.cleavages.setValue(3)
        self.cleavages.setToolTip('Maximum number of concurrent cleavages per fragment')
        self.fragmentation = _combo(FRAGMENTATIONS)
        self.fragmentation.setToolTip('Restricts peptide backbone ion types for glycopeptides')
        form.addRow('Precursor charge', self.charge)
        form.addRow('Reducing end', self.modification)
        form.addRow('Tag mass', self.mass_tag)
        form.addRow('Derivatization', self.sample_prep)
        form.addRow('Max. cleavages', self.cleavages)
        form.addRow('Fragmentation', self.fragmentation)
        layout.addLayout(form)
        self.go = QPushButton('Annotate spectrum')
        self.go.setObjectName('run')
        layout.addWidget(self.go)
        self.spectrum = SpectrumPanel()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.spectrum, 'Annotated spectrum')
        tabs.addTab(self.spectrum.fragments, 'Fragments')
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(form_widget)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(1, 1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)
        self.spectrum.show_message('Enter a structure and a peak list, then annotate the spectrum')
        self.mass_tag.setEnabled(False)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self.update_preview)
        self.structure.textChanged.connect(self.debounce.start)
        self.structure.returnPressed.connect(self.annotate)
        self.modification.currentIndexChanged.connect(lambda: self.mass_tag.setEnabled(self.modification.currentData() == 'custom'))
        self.go.clicked.connect(self.annotate)
        images.ready.connect(lambda key: key == (self.structure.text().strip(), False) and self.update_preview())

    def update_preview(self):
        text = self.structure.text().strip()
        image = self.images.get(text, compact = False, urgent = True) if text and '*' not in text else QImage()
        if image is None:
            self.preview.setText('Drawing…')
        elif not text or '*' in text:
            self.preview.clear()
        elif image.isNull():
            self.preview.setText('GlycoDraw cannot draw this; CandyCrumbs may still read it')
        else:
            ratio = self.devicePixelRatioF()
            pixmap = QPixmap.fromImage(image.scaledToWidth(int(min(image.width() * 0.3, self.preview.width() - 10) * ratio), Qt.SmoothTransformation))
            pixmap.setDevicePixelRatio(ratio)
            self.preview.setPixmap(pixmap)

    def load(self, request):
        settings = request['settings']
        self.structure.setText(request['structure'] or '')
        self.peaks.setPlainText('\n'.join(f'{mz:.4f}\t{intensity:.1f}' for mz, intensity in sorted(request['peaks'].items())))
        _set_combo(self.charge, request['charge'])
        _set_combo(self.modification, settings.get('modification'))
        self.mass_tag.setValue(settings.get('mass_tag') or 0)
        _set_combo(self.sample_prep, settings.get('sample_prep'))
        self.annotate()

    def annotate(self):
        peaks = [[float(x) for x in re.findall(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line)[:2]] for line in self.peaks.toPlainText().splitlines()]
        peaks = [p for p in peaks if p and p[0] > 0]
        if not self.structure.text().strip() or not peaks:
            self.spectrum.show_message('Enter a structure and at least one peak')
            return
        settings = {'modification': self.modification.currentData(), 'mass_tag': self.mass_tag.value() if self.modification.currentData() == 'custom' else None}
        kwargs = {'max_cleavages': self.cleavages.value()}
        if self.fragmentation.currentData():
            kwargs['fragmentation_method'] = self.fragmentation.currentData()
        self.spectrum.annotate(self.structure.text().strip(), [p[0] for p in peaks], [p[1] if len(p) > 1 else 100.0 for p in peaks],
                               self.charge.currentData(), _mass_tag(settings), self.sample_prep.currentData(), **kwargs)


class MainWindow(QMainWindow):
    taxa_ready = Signal(object)
    update_found = Signal(str)
    update_progress = Signal(int)
    update_downloaded = Signal(bool)

    def __init__(self, paths):
        super().__init__()
        self.store = QSettings()
        self.setWindowTitle('CandyCrunch')
        self.setAcceptDrops(True)
        self.images, self.runner, self.started, self.stage_text = GlycanImages(), InferenceRunner(), None, ''
        self.first_report, self.fraction = None, 0.0
        self.settings = SettingsPanel()
        self.run_button, self.cancel_button = QPushButton('Run CandyCrunch'), QPushButton('Cancel')
        self.run_button.setObjectName('run')
        self.cancel_button.setEnabled(False)
        buttons = QWidget()
        buttons.setObjectName('settings')
        row = QHBoxLayout(buttons)
        row.setContentsMargins(14, 8, 14, 12)
        row.addWidget(self.run_button, 1)
        row.addWidget(self.cancel_button)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.settings, 1)
        left_layout.addWidget(buttons)
        welcome = QWidget()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.addStretch(2)
        self.emblem = QLabel()
        self.emblem.setAlignment(Qt.AlignCenter)
        title = QLabel('CandyCrunch')
        title.setObjectName('title')
        title.setAlignment(Qt.AlignCenter)
        self.welcome_text = _label('Glycan structures from LC-MS/MS runs. Add runs on the left (or drop them anywhere here), check the sample settings, '
                                   'and press Run CandyCrunch.', 'hint')
        self.welcome_text.setAlignment(Qt.AlignCenter)
        self.welcome_text.setFixedWidth(460)
        self.welcome_text.setMinimumHeight(self.welcome_text.heightForWidth(460))
        self.stage_label = QLabel()
        self.stage_label.setObjectName('stage')
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.big_progress = QProgressBar()
        self.big_progress.setFixedWidth(460)
        self.big_progress.setTextVisible(False)
        self.big_progress.hide()
        for widget in (self.emblem, title, self.welcome_text, self.stage_label, self.big_progress):
            welcome_layout.addWidget(widget, 0, Qt.AlignHCenter)
        welcome_layout.addStretch(3)
        self.results = ResultsView(self.images)
        self.pages = QStackedWidget()
        self.pages.addWidget(welcome)
        self.pages.addWidget(self.results)
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(left)
        self.split.addWidget(self.pages)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([360, 1100])
        self.crumbs = CrumbsTab(self.images)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.split, 'Predict')
        self.tabs.addTab(self.crumbs, 'CandyCrumbs')
        self.setCentralWidget(self.tabs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setMaximumBlockCount(20000)
        self.log_dock = QDockWidget('Log', self)
        self.log_dock.setObjectName('log')
        self.log_dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.status_text, self.elapsed = QLabel('Loading the CandyCrunch model…'), QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.elapsed)
        self.statusBar().addPermanentWidget(self.progress)
        self.update_button, self.update_url, self.update_path = QPushButton(), None, None
        self.update_button.setFlat(True)
        self.update_button.setStyleSheet(f'color: {CANDY}; font-weight: 600;')
        self.update_button.hide()
        self.statusBar().addPermanentWidget(self.update_button)
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self.tick)
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction('Add runs…', self.choose_files, QKeySequence.Open)
        file_menu.addAction('Add folder…', self.choose_folder)
        file_menu.addSeparator()
        file_menu.addAction('Open results…', self.open_session)
        self.save_action = file_menu.addAction('Save results…', self.save_session, QKeySequence.Save)
        self.save_action.setEnabled(False)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close, QKeySequence.Quit)
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.log_dock.toggleViewAction())
        help_menu = self.menuBar().addMenu('&Help')
        help_menu.addAction('CandyCrunch on GitHub', lambda: QDesktopServices.openUrl(QUrl('https://github.com/BojarLab/CandyCrunch')))
        help_menu.addAction('About CandyCrunch', self.about)
        self.settings.add_files.clicked.connect(self.choose_files)
        self.settings.add_folder.clicked.connect(self.choose_folder)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.runner.ready.connect(lambda: self.runner.busy or self.status_text.setText('Ready'))
        self.runner.stage.connect(self.show_stage)
        self.runner.progress.connect(self.show_progress)
        self.runner.log.connect(self.append_log)
        self.runner.finished.connect(self.finished)
        self.runner.failed.connect(self.failed)
        self.results.status.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.results.experiment_shown.connect(self.settings.select_experiment)
        self.results.open_in_crumbs.connect(lambda request: (self.tabs.setCurrentWidget(self.crumbs), self.crumbs.load(request)))
        self.settings.inputs['taxonomy_level'].currentTextChanged.connect(self.load_taxa)
        self.taxa_ready.connect(lambda taxa: self.settings.taxa.setModel(QStringListModel(taxa, self.settings.taxa)))
        self.images.ready.connect(self.image_ready)
        self.update_button.clicked.connect(self.install_update)
        self.update_found.connect(lambda tag: (self.update_button.setText(f'Update to CandyCrunch {tag}'), self.update_button.show()))
        self.update_progress.connect(lambda percent: self.update_button.setText(f'Downloading update… {percent}%'))
        self.update_downloaded.connect(lambda ok: (self.update_button.setEnabled(True), self.update_button.setText('Install update and restart' if ok else 'Download failed, retry')))
        self.images.get('Fuc(a1-2)Gal(b1-3)GalNAc', compact = False, urgent = True)
        if self.store.value('settings'):
            stored = json.loads(self.store.value('settings'))
            self.settings.set_values({**DEFAULTS, **stored, 'filter_out': set(stored.get('filter_out', DEFAULTS['filter_out']))})
        if self.store.value('geometry'):
            self.restoreGeometry(self.store.value('geometry'))
            self.restoreState(self.store.value('state'))
        else:
            self.resize(1440, 900)
        sessions = [p for p in paths if p.lower().endswith('.candycrunch')]
        self.settings.add_paths([p for p in paths if p not in sessions and os.path.exists(p)])
        if sessions:
            self.open_session(sessions[0])
        self.load_taxa()
        if getattr(sys, 'frozen', False) and sys.platform == 'win32':
            # The installed app offers the newest GitHub release that carries a Windows installer; without network access it simply offers nothing
            def check():
                try:
                    from importlib.metadata import version
                    from packaging.version import Version
                    with urllib.request.urlopen('https://api.github.com/repos/BojarLab/CandyCrunch/releases/latest', timeout = 10) as response:
                        release = json.load(response)
                    asset = next(a for a in release['assets'] if a['name'].endswith('-Windows-Setup.exe'))
                    if Version(release['tag_name'].lstrip('v')) > Version(version('candycrunch')):
                        self.update_url = asset['browser_download_url']
                        self.update_found.emit(release['tag_name'].lstrip('v'))
                except Exception:
                    pass
            threading.Thread(target = check, daemon = True).start()

    def load_taxa(self):
        level = self.settings.inputs['taxonomy_level'].currentText()
        def work():
            with IMPORT_LOCK:
                from glycowork.glycan_data.loader import df_glycan
                # Imported here only so the first spectrum annotation does not wait for it
                import candycrunch.analysis  # noqa: F401
            self.taxa_ready.emit(sorted({t for taxa in df_glycan[level] for t in (taxa if isinstance(taxa, list) else [taxa]) if isinstance(t, str)}))
        threading.Thread(target = work, daemon = True).start()

    def image_ready(self, key):
        image = self.images.images[key]
        if image.isNull():
            return
        if key == ('Fuc(a1-2)Gal(b1-3)GalNAc', False):
            pixmap = QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(2.8)
            self.emblem.setPixmap(pixmap)

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add LC-MS/MS runs', self.store.value('folder', os.path.expanduser('~')),
                                                'Spectra (*.raw *.RAW *.mzML *.mzXML *.mgf *.xlsx);;All files (*)')
        if paths:
            self.store.setValue('folder', os.path.dirname(paths[0]))
            self.settings.add_paths(paths)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Add every .raw, mzML, mzXML, and mgf file of a folder', self.store.value('folder', os.path.expanduser('~')))
        if path:
            self.store.setValue('folder', path)
            self.settings.add_paths([path])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        sessions = [p for p in paths if p.lower().endswith('.candycrunch')]
        if sessions:
            self.open_session(sessions[0])
        self.settings.add_paths([p for p in paths if p not in sessions])
        self.tabs.setCurrentWidget(self.split)

    def run(self):
        experiments = self.settings.experiments()
        if not experiments:
            self.statusBar().showMessage('Add at least one LC-MS/MS run first', 6000)
            return
        files = [file for _, paths, _ in experiments for file in paths]
        settings = self.settings.values()
        self.store.setValue('settings', json.dumps({**settings, 'filter_out': sorted(settings['filter_out'])}))
        self.weights = {**STAGE_WEIGHTS, **json.loads(self.store.value('stage_weights', '{}'))}
        self.runner.submit(experiments, self.weights)
        self.started, self.stage_text = time.time(), 'Waiting for the CandyCrunch model to load' if self.status_text.text().startswith('Loading') else 'Starting'
        # Progress per file (keyed like the worker keys them, weighted by file size as runs of different length go through in parallel) and of a
        # batch's harmonization; the time estimate starts with the first progress report
        self.file_progress, self.fraction, self.first_report, self.stage_times = dict.fromkeys(map(os.path.normcase, files), 0.0), 0.0, None, []
        # Experiments run one after the other, so a batch's harmonization belongs to the experiment of the latest file that reported
        self.groups, self.tail_progress, self.active = [[os.path.normcase(f) for f in paths] for _, paths, _ in experiments], [0.0] * len(experiments), 0
        self.file_sizes = {os.path.normcase(f): max(os.path.getsize(f), 1) if os.path.exists(f) else 1 for f in files}
        self.history = {}
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        for bar in (self.progress, self.big_progress):
            bar.setRange(0, 0)
            bar.show()
        self.clock.start()
        self.log.clear()
        self.welcome_text.hide()
        # The progress lives on the start page, so earlier results make way for it until this run is done
        self.pages.setCurrentIndex(0)
        self.append_log(f'Running CandyCrunch on {len(files)} file{"s" if len(files) > 1 else ""} in {len(experiments)} experiment{"s" if len(experiments) > 1 else ""}\n')
        self.tick()

    def tick(self):
        seconds = int(time.time() - self.started)
        self.elapsed.setText(f'{seconds // 60}:{seconds % 60:02d}')
        text = self.stage_text
        if self.first_report is not None:
            text += f' ({self.fraction:.0%})'
            # Time left at the pace of the last 30 s, overall and per run: runs going through in parallel finish with the slowest of them, and a
            # run's pace changes with how many cores it shares, so recent pace beats the average since the start
            now, lefts = time.time(), []
            for series in self.history.values():
                old = next(((t, v) for t, v in series if now - t <= 30), series[0])
                if series[-1][1] < 1 and series[-1][1] - old[1] > 0.01 and now - old[0] > 3:
                    lefts.append((1 - series[-1][1]) * (now - old[0]) / (series[-1][1] - old[1]))
            # The first steps go faster than the prediction that dominates, so the estimate waits until the prediction is under way
            if now - self.first_report[0] > 10 and self.fraction > 0.1 and lefts:
                left = max(lefts)
                text += f', about {max(10, round(left / 10) * 10)} s left' if left < 175 else f', about {round(left / 60)} min left'
        self.status_text.setText(text)
        self.stage_label.setText(text)

    def show_stage(self, update):
        name, self.stage_text = update
        self.stage_times.append((name, time.time()))
        self.append_log(f'[{self.elapsed.text() or "0:00"}] {self.stage_text}\n')
        self.tick()

    def show_progress(self, update):
        key, value = update
        if key == 'tail':
            self.file_progress.update(dict.fromkeys(self.groups[self.active], 1.0))
            self.tail_progress[self.active] = value
        elif key in self.file_progress:
            self.active = next(n for n, group in enumerate(self.groups) if key in group)
            self.file_progress[key] = value
            self.history.setdefault(key, []).append((time.time(), value))
        # Every experiment counts by the size of its files, of which a batch spends TAIL_WEIGHT on harmonization
        done = sum(sum(self.file_progress[k] * self.file_sizes[k] for k in group) if len(group) == 1 else
                   sum(((1 - TAIL_WEIGHT) * self.file_progress[k] + TAIL_WEIGHT * tail) * self.file_sizes[k] for k in group) for group, tail in zip(self.groups, self.tail_progress))
        self.fraction = max(self.fraction, done / sum(self.file_sizes.values()))
        self.history.setdefault('all', []).append((time.time(), self.fraction))
        if self.first_report is None:
            self.first_report = (time.time(), self.fraction)
        for bar in (self.progress, self.big_progress):
            bar.setRange(0, 1000)
            bar.setValue(int(self.fraction * 1000))
        self.tick()

    def append_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)

    def stop(self, message):
        self.clock.stop()
        self.progress.hide()
        self.big_progress.hide()
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.stage_label.setText('')
        self.welcome_text.show()
        self.elapsed.setText('')
        self.status_text.setText(message)
        if self.results.payload is not None:
            self.pages.setCurrentWidget(self.results)

    def finished(self, payload):
        seconds = int(time.time() - self.started)
        # A single file shows how this computer splits its time over the steps; blending that in makes the next estimate fit the machine
        steps = [(name, t) for name, t in self.stage_times if name in STAGE_WEIGHTS]
        if len(self.file_progress) == 1 and steps:
            measured = {}
            for (name, start), end in zip(steps, [t for _, t in steps[1:]] + [time.time()]):
                measured[name] = measured.get(name, 0) + end - start
            scale = sum(self.weights[name] for name in measured) / (sum(measured.values()) or 1)
            self.store.setValue('stage_weights', json.dumps({name: 0.5 * self.weights[name] + 0.5 * measured[name] * scale if name in measured else self.weights[name]
                                                             for name in STAGE_WEIGHTS}))
        peaks, runs = sum(len(df) for e in payload['experiments'] for df, _ in e['tables'].values()), sum(len(e['tables']) for e in payload['experiments'])
        self.stop(f'Finished in {seconds // 60}:{seconds % 60:02d}: {peaks} glycan peaks in {runs} run{"s" if runs > 1 else ""}')
        self.append_log(self.status_text.text() + '\n')
        if not peaks:
            QMessageBox.information(self, 'CandyCrunch', 'CandyCrunch found no glycan peaks. Check the glycan class, ion mode, and reducing end, '
                                    'or open the log (View > Log) for details.')
            return
        self.results.set_results(payload)
        self.pages.setCurrentWidget(self.results)
        self.save_action.setEnabled(True)
        QApplication.alert(self)

    def failed(self, text):
        self.stop('CandyCrunch stopped with an error; details are in the log')
        self.append_log(text)
        self.log_dock.show()
        QMessageBox.critical(self, 'CandyCrunch', text.strip().splitlines()[-1])

    def cancel(self):
        self.runner.cancel()
        self.stop('Run cancelled')
        self.append_log('Run cancelled\n')

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save results', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'results.candycrunch'),
                                              'CandyCrunch results (*.candycrunch)')
        if path:
            with open(path, 'wb') as file:
                pickle.dump(self.results.session, file)
            self.statusBar().showMessage(f'Saved {os.path.basename(path)}', 6000)

    def open_session(self, path = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Open results', self.store.value('folder', os.path.expanduser('~')), 'CandyCrunch results (*.candycrunch)')
        if not path:
            return
        try:
            with open(path, 'rb') as file:
                payload = pickle.load(file)
        except Exception as error:
            QMessageBox.critical(self, 'Open results', f'Could not read {path}:\n{error}')
            return
        self.results.set_results(payload)
        self.settings.set_experiments(self.results.session['experiments'])
        self.pages.setCurrentWidget(self.results)
        self.save_action.setEnabled(True)
        self.tabs.setCurrentWidget(self.split)
        self.status_text.setText(f'{os.path.basename(path)}: results from {payload.get("finished", "an earlier run")}')

    def install_update(self):
        if self.update_path:
            # Closing first releases the app's files (and asks if a run is still going); the installer updates the same folder in place and starts the new version
            if self.close():
                subprocess.Popen([self.update_path, '/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/RELAUNCH=1'])
            return
        self.update_button.setEnabled(False)
        self.update_button.setText('Downloading update…')
        def work():
            path = os.path.join(tempfile.gettempdir(), self.update_url.rsplit('/', 1)[1])
            try:
                with urllib.request.urlopen(self.update_url, timeout = 30) as response, open(path + '.part', 'wb') as file:
                    total, done = int(response.headers.get('Content-Length') or 0), 0
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
                        done += len(chunk)
                        if total:
                            self.update_progress.emit(done * 100 // total)
                os.replace(path + '.part', path)
                self.update_path = path
            except Exception:
                pass
            self.update_downloaded.emit(self.update_path is not None)
        threading.Thread(target = work, daemon = True).start()

    def about(self):
        from importlib.metadata import version
        QMessageBox.about(self, 'About CandyCrunch', f'<h3>CandyCrunch {version("candycrunch")}</h3><p>Predicting glycan structure from LC-MS/MS data.</p>'
                          '<p>Urban J, Jin C, Thomsson KA, Karlsson NG, Ives CM, Fadda E, Bojar D. Predicting glycan structure from tandem mass spectrometry via deep '
                          'learning. <i>Nature Methods</i> 21, 1206-1215 (2024).</p><p><a href="https://github.com/BojarLab/CandyCrunch">github.com/BojarLab/CandyCrunch</a></p>')

    def closeEvent(self, event):
        if self.runner.busy and QMessageBox.question(self, 'CandyCrunch', 'A run is still going. Quit anyway?') != QMessageBox.Yes:
            event.ignore()
            return
        self.store.setValue('geometry', self.saveGeometry())
        self.store.setValue('state', self.saveState())
        self.runner.shutdown()
        event.accept()


def main():
    mp.freeze_support()
    if sys.platform == 'win32':
        import ctypes
        # Without its own app ID, Windows groups the window under python.exe and shows Python's taskbar icon
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('BojarLab.CandyCrunch')
    app = QApplication(sys.argv)
    app.setOrganizationName('BojarLab')
    app.setApplicationName('CandyCrunch')
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in ((QPalette.Window, WASH), (QPalette.Base, '#ffffff'), (QPalette.AlternateBase, '#fbf9fc'), (QPalette.Text, INK), (QPalette.WindowText, INK),
                        (QPalette.ButtonText, INK), (QPalette.Button, '#efe8f4'), (QPalette.Highlight, LILAC), (QPalette.HighlightedText, '#ffffff'),
                        (QPalette.ToolTipBase, '#ffffff'), (QPalette.ToolTipText, INK), (QPalette.Link, '#7a3fa8'), (QPalette.PlaceholderText, MUTED)):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'candycrunch_icon.png')))
    if sys.argv[1:2] == ['--self-test']:
        # For the app build: --self-test CLASS FILE [FILE ...] predicts the files (several in parallel) without a window and exits with 0 only if
        # that found glycan peaks
        files = [os.path.abspath(arg) for arg in sys.argv[3:]]
        panel, runner, outcome, log = SettingsPanel(), InferenceRunner(), [], []
        panel.set_values({**DEFAULTS, 'glycan_class': sys.argv[2], 'n_jobs': len(files)})
        runner.log.connect(log.append)
        runner.finished.connect(lambda payload: (outcome.append(sum(len(df) for e in payload['experiments'] for df, _ in e['tables'].values())), app.quit()))
        runner.failed.connect(lambda text: (log.append(text), app.quit()))
        runner.submit([('Self-test', files, panel.values())], STAGE_WEIGHTS)
        app.exec()
        runner.shutdown()
        with open('candycrunch_self_test.log', 'w', encoding = 'utf-8') as file:
            file.write(''.join(log) + f'\nGlycan peaks found: {outcome[0] if outcome else 0}\n')
        sys.exit(0 if outcome and outcome[0] > 0 else 1)
    window = MainWindow([os.path.abspath(p) for p in sys.argv[1:]])
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

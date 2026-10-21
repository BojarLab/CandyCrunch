"""CandyCrunch desktop app: run CandyCrunch on LC-MS/MS files, inspect and curate its predictions, and annotate spectra with CandyCrumbs"""
import ast
import contextlib
import faulthandler
import functools
import glob
import importlib.util
import io
import itertools
import json
import linecache
import multiprocessing as mp
import os
import pickle
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
try:
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QSettings, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QRectF, QSize, QUrl, QStringListModel, \
        QStandardPaths, QEvent
    from PySide6.QtGui import QTextOption, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPalette, QPixmap, QBrush, QShortcut, QFontDatabase, \
        QCursor
    from PySide6.QtWidgets import (QApplication, QMainWindow, QDialog, QDialogButtonBox, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton,
                                   QToolButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget,
                                   QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTableView, QAbstractItemView, QSplitter, QStackedWidget, QTabWidget, QScrollArea,
                                   QFrame, QFileDialog, QMessageBox, QInputDialog, QDockWidget, QProgressBar, QStyledItemDelegate, QStyleOptionViewItem,
                                   QStyle, QMenu, QCompleter, QSizePolicy, QToolTip)
except ImportError as error:
    raise ImportError('The CandyCrunch app needs PySide6, which comes with: pip install "candycrunch[gui]"') from error
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backend_tools import Cursors
from matplotlib.mathtext import MathTextParser

# Defaults and constants are read from prediction.py's source, so the app always offers the library's own defaults without importing torch
_tree = ast.parse(open(importlib.util.find_spec('candycrunch.prediction').origin, encoding = 'utf-8').read())
_batch_def = next(node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name == 'wrap_inference_batch')
# model_file is the app's own setting (an imported model, None for the built-in one), which the worker turns into wrap_inference's model and glycans
DEFAULTS = {'intra_cat_thresh': 1.0, 'model_file': None}
for _arg, _default in zip(_batch_def.args.args[-len(_batch_def.args.defaults):], _batch_def.args.defaults):
    try:
        DEFAULTS[_arg.arg] = ast.literal_eval(_default)
    except ValueError:
        pass
_constants = {node.targets[0].id: node.value for node in _tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
MZ_REF = ast.literal_eval(_constants['MZ_REF'])
FILTER_CHOICES = sorted(set(ast.literal_eval(_constants['comp_vector_order'])) | DEFAULTS['filter_out'], key = str.lower)
ANALYZER_TRAPS, MZXML_ANALYZER_TRAPS = ast.literal_eval(_constants['ANALYZER_TRAPS']), ast.literal_eval(_constants['MZXML_ANALYZER_TRAPS'])
# What a run's name may say about its settings; a setting is only taken from the name if it names exactly one of its choices
NAME_HINTS = {'glycan_class': {'N': r'n-?glycans?|ng|n(?![a-z0-9])', 'O': r'o-?glycans?|og|o(?![a-z0-9])|mucins?', 'lipid': r'gsls?|glycosphingolipids?|glycolipids?|gangliosides?',
                               'free': r'milk|hmos?'},
              'modification': {'reduced': r'(?<!non-)reduced|alditols?', 'unreduced': r'un-?reduced|non-?reduced', '2AB': r'2-?ab', '2AA': r'2-?aa', 'procainamide': r'procainamide',
                               'PA': r'pa-?label\w*|pyridylamin\w*|2-?aminopyridin\w*'},
              'sample_prep': {'permethylated': r'permethyl\w*', 'peracetylated': r'peracetyl\w*'}, 'lc': {'PGC': r'pgc', 'C18': r'c18', 'other': r'hilic'}}
BATCH_KEYS = ('intra_cat_thresh', 'top_n_isomers', 'n_jobs')
FEATURES = '\x00features'
GLYCAN_CLASSES = [('O-glycans', 'O'), ('N-glycans', 'N'), ('Free oligosaccharides (e.g., milk)', 'free'), ('Glycolipid glycans', 'lipid'),
                  ('N- and O-glycans (same run)', 'N+O')]
# Classes measured together are one combo choice, their names joined by '+' in this order; wrap_inference takes them as a list
CLASS_ORDER = ['N', 'O', 'free', 'lipid']
# 'unreduced' exists only in the app: wrap_inference takes a free reducing end as a custom label without mass (its reducing end check calls that 'unreduced')
REDUCING_ENDS = [('Reduced (alditol)', 'reduced'), ('Unreduced (free reducing end)', 'unreduced'), ('2-AB', '2AB'), ('2-AA', '2AA'), ('Procainamide', 'procainamide'),
                 ('PA (2-aminopyridine)', 'PA'), ('Other label (enter its mass)', 'custom')]
SAMPLE_PREPS = [('Underivatized', 'underivatized'), ('Permethylated', 'permethylated'), ('Peracetylated', 'peracetylated')]
# Linkage-specific sialic acid derivatizations of underivatized samples, as wrap_inference's sialic_derivatization
SIALIC_DERIVATIZATIONS = [('None', None), ('Ethyl esterification (EE)', 'EE'), ('Ethyl esterification + amidation (EEA)', 'EEA'), ('Methyl esterification (ME)', 'ME'),
                          ('Dimethylamidation (DMA)', 'DMA'), ('SALSA', 'SALSA'), ('Glycoqueuing, aniline (GQ)', 'GQ'), ('Glycoqueuing, d5/d0-aniline (GQ-d5)', 'GQ-d5')]
SIALIC_TIP = 'Linkage-specific sialic acid derivatization: precursor masses then tell a2-6 from a2-3/8 sialic acids, and fragments carry the derivatized sialic acids'
LCS = [('PGC', 'PGC'), ('C18', 'C18'), ('Other', 'other')]
TRAPS = [('Linear ion trap', 'linear'), ('Orbitrap', 'orbitrap'), ('Bruker amaZon', 'amazon'), ('Other', 'other')]
FRAGMENT_TYPES = [('Automatic', None), ('Glycosidic only', 'glycosidic'), ('All cross-rings', 'cross_rings')]
TAXONOMY_LEVELS = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain']
FRAGMENTATIONS = [('Any', None), ('CID', 'CID'), ('HCD', 'HCD'), ('ETD', 'ETD'), ('ECD', 'ECD'), ('EThcD', 'EThcD'), ('ETciD', 'ETciD'), ('EED', 'EED'),
                  ('EDD', 'EDD'), ('NETD', 'NETD'), ('UVPD', 'UVPD')]
SPECTRA_SUFFIXES = ('.raw', '.d', '.mzml', '.mzxml', '.mgf', '.cc')
# What can be wrong with a peak's prediction, for misprediction reports from the spectrum view; the value tags the report
MISPREDICTIONS = [('Choose what is wrong…', None), ('Wrong structure (the composition is right)', 'structure'), ('Wrong composition', 'composition'),
                  ('No structure, though it is a glycan', 'missing'), ('Not a glycan (noise, contaminant, adduct, or isotope)', 'not_glycan'),
                  ('Wrong charge state', 'charge'), ('Two peaks merged, or one peak split', 'peak_boundaries'), ('Wrong fragment annotation', 'annotation'),
                  ('Something else', 'other')]
# extract_spectra xlsx of 4.5 min (25-29.5) of a pig gastric mucin O-glycan run (GPST000029, as packaging/self_test.mzML): 275 MS2 spectra, ~20 glycan peaks
EXAMPLE_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_PGM_O-glycans.xlsx')
# Crash and bug reports go to the CandyCrunch project on Sentry (its DSN, from Settings > Client Keys: public by design, it can only submit reports);
# without one, or if sending fails, a report opens as a prefilled GitHub issue instead
REPORT_DSN = 'https://5ecf60f667f420d84bd6b3c8c4b437bd@o4512244953776128.ingest.de.sentry.io/4512244961837136'
ISSUES_URL = 'https://github.com/BojarLab/CandyCrunch/issues/new'
# faulthandler writes native crashes and fatal Python errors of the window and the worker here; a process that ends cleanly removes its file
CRASH_FILE = os.path.join(tempfile.gettempdir(), 'CandyCrunch-crash-{}-{}.txt')
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
# matplotlib caches parsed mathtext per parser, 50 entries in all, and every renderer has a parser of its own, so each spectrum parsed its fragment labels
# again (~5 ms each), and so did the window's canvas when it drew it (~30 ms of every spectrum shown). One cache for all parsers of an output type returns
# the same parse for the same arguments, as only the output type of the parser goes into it
_MATHTEXT_PARSERS, _MATHTEXT_PARSE = {}, getattr(MathTextParser._parse_cached, '__wrapped__', None)


@functools.lru_cache(2048)
def _shared_mathtext(output_type, *args):
    return _MATHTEXT_PARSE(_MATHTEXT_PARSERS[output_type], *args)


def _parse_mathtext(self, *args):
    _MATHTEXT_PARSERS.setdefault(self._output_type, self)
    return _shared_mathtext(self._output_type, *args)


if _MATHTEXT_PARSE is not None:
    MathTextParser._parse_cached = _parse_mathtext
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


def _scrub(text):
    """Shortens paths in the user's home folder (which hold the user name) to ~, also as Qt writes them on Windows, with the backslashes doubled as in
    error messages (repr), and each of those as escaped in JSON"""
    home = os.path.expanduser('~')
    variants = {home, home.replace('\\', '/'), home.replace('\\', '\\\\')}
    for variant in sorted(variants | {json.dumps(v)[1:-1] for v in variants}, key = len, reverse = True):
        text = text.replace(variant, '~')
    return text


def _exception_values(error):
    """An exception and those it was raised from or during, oldest first, in Sentry's format; frames name the module and file, not the folder"""
    values, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        frames = [{'module': str(frame.f_globals.get('__name__', '?')), 'filename': os.path.basename(frame.f_code.co_filename), 'function': frame.f_code.co_name,
                   'lineno': lineno, 'context_line': linecache.getline(frame.f_code.co_filename, lineno).strip() or None,
                   'in_app': str(frame.f_globals.get('__name__', '')).startswith('candycrunch')} for frame, lineno in traceback.walk_tb(error.__traceback__)]
        values.insert(0, {'type': type(error).__qualname__, 'module': type(error).__module__, 'value': str(error), 'stacktrace': {'frames': frames}})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return values


def _post_envelope(header, items):
    """Posts a Sentry envelope, its header and items as (item header, payload bytes); True if Sentry took it"""
    if not REPORT_DSN:
        return False
    dsn = urllib.parse.urlsplit(REPORT_DSN)
    prefix, project = dsn.path.rstrip('/').rsplit('/', 1)
    body = json.dumps(header).encode() + b''.join(b'\n' + json.dumps({**item, 'length': len(payload)}).encode() + b'\n' + payload for item, payload in items)
    request = urllib.request.Request(f'{dsn.scheme}://{dsn.netloc.rsplit("@", 1)[-1]}{prefix}/api/{project}/envelope/', data = body, headers = {
        'Content-Type': 'application/x-sentry-envelope', 'X-Sentry-Auth': f'Sentry sentry_version=7, sentry_client=candycrunch-app, sentry_key={dsn.username}'})
    try:
        # A report with the data around a peak can be a few MB, which takes a while on a slow connection
        with urllib.request.urlopen(request, timeout = max(20, len(body) / 50000)) as response:
            return response.status == 200
    except Exception:
        return False


def _send_report(event, files = None):
    """Posts a Sentry event (an error, or a bug report as type 'feedback') and attachments ({file name: text, or bytes for binary files}) as one envelope; True if Sentry took it"""
    items = [({'type': event.get('type', 'event')}, _scrub(json.dumps(event, default = str)).encode())]
    # Text is scrubbed of the home folder; bytes (the .cc of a misprediction's surroundings) go as they are
    items += [({'type': 'attachment', 'filename': name, 'content_type': 'application/octet-stream' if isinstance(data, bytes) else 'text/plain'},
               data if isinstance(data, bytes) else _scrub(data).encode()) for name, data in (files or {}).items() if data]
    return _post_envelope({'event_id': event['event_id'], 'sent_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, items)


def _send_session(update):
    """Release health on Sentry: one session update (release, start, status, error count; nothing about the user or their runs), stamped with the time
    and posted in a background thread, which is returned"""
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    thread = threading.Thread(target = _post_envelope, args = ({'sent_at': now}, [({'type': 'session'}, json.dumps({**update, 'timestamp': now}).encode())]), daemon = True)
    thread.start()
    return thread


def _load_model(path):
    """An imported CandyCrunch model: weights saved with their own vocabulary ({'state_dict': ..., 'glycans': [...]}, as the fine-tuning cell saves them)
    or bare weights for the built-in vocabulary; read with weights_only, so a model file from elsewhere cannot run code. Returns (model, glycans); without
    PyTorch (the built app) the model runs on ONNX Runtime"""
    from candycrunch.inference import CandyCrunchCNN, load_weights
    try:
        import torch
        from candycrunch.model import CandyCrunch_CNN
    except ImportError:
        torch = None
    try:
        state = torch.load(path, map_location = 'cpu', weights_only = True) if torch is not None else load_weights(path)
    except Exception as error:
        raise ValueError(f'{os.path.basename(path)} is no PyTorch weights file, or holds code, which CandyCrunch does not load ({type(error).__name__})') from error
    if not isinstance(state, dict):
        raise ValueError(f'{os.path.basename(path)} holds no model weights')
    glycans = state.get('glycans')
    if glycans is None:
        with open(os.path.join(os.path.dirname(importlib.util.find_spec('candycrunch').origin), 'glycans.pkl'), 'rb') as file:
            glycans = pickle.load(file)
    try:
        if torch is not None:
            model = CandyCrunch_CNN(2048, num_classes = len(glycans), input_precursor_dim = 12)
            model.load_state_dict({k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()})
            return model.to('cuda:0' if torch.cuda.is_available() else 'cpu').eval(), list(glycans)
        model = CandyCrunchCNN(state.get('state_dict', state))
        # One empty input checks the weights against the architecture and the vocabulary
        n = model(np.zeros((1, 2, 2048), np.float32), np.zeros((1, 12), np.float32), np.zeros(1, np.int64), np.zeros((1, 1), np.float32),
                  *[np.zeros(1, np.int64)] * 4).shape[1]
        if n != len(glycans):
            raise ValueError(f'it predicts {n}')
    except Exception as error:
        raise ValueError(f'{os.path.basename(path)} is no CandyCrunch model for {len(glycans)} structures: {" ".join(str(error).split())[:300]}') from error
    return model, list(glycans)


def _check_model(path):
    """Loads a model file in a process of its own (the window never imports torch): the number of structures it predicts, or why it cannot be used"""
    return len(_load_model(path)[1])


def _extract_peak_data(path, mz, charge, rt):
    """Runs in a process of its own (it imports the pipeline and reads the whole run again): the bytes of a .cc of a peak's surroundings for a
    misprediction report (extract_spectra with region: survey scans within REGION_RT min, the MS2/MS3 spectra of its ion at every charge state there)"""
    from candycrunch.prediction import extract_spectra
    with tempfile.TemporaryDirectory() as folder:
        with open(extract_spectra(path, os.path.join(folder, 'peak.cc'), 'cc', region = (mz, charge, rt)), 'rb') as file:
            return file.read()


def _worker_main(jobs, messages, warm_up = False):
    """Runs CandyCrunch in its own process: the window stays responsive, a run is canceled by ending the process, and the model loads once for all runs"""
    class Pipe(io.TextIOBase):
        def write(self, text):
            messages.put(('log', text))
            return len(text)
    sys.stdout = sys.stderr = Pipe()
    # If this process dies, the window reads the traceback of the native crash or fatal Python error from this file
    with contextlib.suppress(OSError):
        faulthandler.enable(open(CRASH_FILE.format('worker', os.getpid()), 'w'), all_threads = True)
    try:
        import candycrunch.prediction as cp
    except Exception as error:
        messages.put(('fatal', (traceback.format_exc(), _exception_values(error))))
        return
    if warm_up and jobs.empty() and os.path.exists(EXAMPLE_RUN):
        # A silent run on the example does the lazy imports, cache filling and first model passes that would otherwise slow down the first real run
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cp.wrap_inference(EXAMPLE_RUN, 'O')
        except Exception:
            pass
    messages.put(('ready', None))
    # Imported models, loaded once per worker
    parent, models = mp.parent_process(), {}
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
                kwargs = {k: v for k, v in settings.items() if k not in BATCH_KEYS and k != 'model_file'}
                if settings.get('model_file'):
                    if settings['model_file'] not in models:
                        models[settings['model_file']] = _load_model(settings['model_file'])
                    kwargs['model'], kwargs['glycans'] = models[settings['model_file']]
                if len(files) == 1:
                    # n_jobs processes score fragments and find supporting peaks, the pure-Python half of a run (the model already uses every core)
                    df, spectra = cp.wrap_inference(files[0], spectra = True, n_jobs = settings['n_jobs'], **kwargs)
                    tables, features = {os.path.splitext(os.path.basename(files[0]))[0]: (df, spectra)}, None
                else:
                    features, tables = cp.wrap_inference_batch(files, spectra = True, **kwargs, **{k: settings[k] for k in BATCH_KEYS})
                results.append({'name': name, 'files': files, 'settings': settings, 'tables': tables, 'features': features})
            messages.put(('result', {'experiments': results, 'elapsed': time.time() - start, 'finished': time.strftime('%Y-%m-%d %H:%M')}))
        except Exception as error:
            messages.put(('error', (traceback.format_exc(), _exception_values(error))))


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
    class Counted:
        """get_topk's model, counting batches (get_topk makes its own batches from a plain loader, so wrapping the loader would not see them)"""
        def __init__(self, model):
            self.model = model
        def __call__(self, *inputs):
            out = self.model(*inputs)
            advance(1)
            return out
    def staged(func, name):
        def run(*args, **kwargs):
            p = _PROGRESS
            if p['stage'] == 'get_topk' and name in ('assign_candidate_structures', 'assign_annotation_scores_pooled'):
                # After the prediction, wrap_inference matches and scores the run's structures at other reducing ends: part of that step, not a new one
                return func(*args, **kwargs)
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
                args = (args[0], Counted(args[1])) + args[2:]
            return func(*args, **kwargs)
        return run
    for name in STAGES:
        setattr(cp, name, staged(getattr(cp, name), name))
    real_jobs = cp.run_jobs

    def run_jobs(jobs, n_jobs = 1):
        """run_jobs, counting the CandyCrumbs calls of fragment scoring; the step's todo counts candidate structures, i.e., MS2 calls (the extra calls
        for MS3 spectra are not counted). Wrapping CandyCrumbs itself would send an unpicklable function to the n_jobs processes, so with them the
        calls go out in chunks"""
        if _PROGRESS['stage'] != 'assign_annotation_scores_pooled':
            return real_jobs(jobs, n_jobs)
        results, size = [], 1 if n_jobs <= 1 else 16 * n_jobs
        for start in range(0, len(jobs), size):
            results += real_jobs(jobs[start:start + size], n_jobs)
            advance(sum(job[2].get('ms3_precursor') is None for job in jobs[start:start + size]))
        return results

    cp.run_jobs = run_jobs
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
    # Cancel and quit end the worker; without this its pool processes would finish their file unseen and then wait forever
    parent = mp.parent_process()
    threading.Thread(target = lambda: (parent.join(), os._exit(1)), daemon = True).start()


def _detect_run(path, results):
    """Runs in a process of its own, as reading a large file takes seconds: what a run's file says about its settings, as the pipeline reads it (ion mode, MS2
    analyzer, how many MS2 spectra carry each precursor charge, and how many MS2 spectra there are), and what its name says (NAME_HINTS); puts (path, found,
    charges, number of MS2 spectra) on results"""
    found, charges, n_ms2, ext = {}, {}, 0, os.path.splitext(os.path.normpath(path))[1].lower()
    try:
        if ext in ('.mzml', '.mzxml', '.mgf'):
            # Only tags and their attributes are read, never the binary data, in chunks cut after a tag (mgf: a line); the first chunk holds the instruments
            mode_re, filter_re, charge_re, ms2_re = [re.compile(x) if x else None for x in {
                '.mzml': (rb'"MS:10001(29|30)"', rb'"MS:1000512"[^>]*?value="(ITMS|FTMS)[^"]* ms2 ', rb'"MS:1000041"[^>]*?value="(\d+)"', rb'"MS:1000511"[^>]*?value="2"'),
                '.mzxml': (rb'polarity="([+-])"', rb'filterLine="(ITMS|FTMS)[^"]* ms2 ', rb'precursorCharge="(\d+)"', rb'msLevel="2"'),
                '.mgf': (rb'(?m)^CHARGE=\d+([+-])', None, rb'(?m)^CHARGE=(\d+)', rb'(?m)^BEGIN IONS')}[ext]]
            cut, tail, header = b'\n' if ext == '.mgf' else b'>', b'', None
            with open(path, 'rb') as file:
                while True:
                    chunk = file.read(1 << 24)
                    text = tail + chunk
                    end = text.rfind(cut) + 1 if chunk else len(text)
                    text, tail, header = text[:end], text[end:], text if header is None else header
                    if 'mode' not in found and (match := mode_re.search(text)):
                        found['mode'] = {b'29': 'negative', b'30': 'positive', b'-': 'negative', b'+': 'positive'}[match.group(1)]
                    if 'trap' not in found and filter_re and (match := filter_re.search(text)):
                        found['trap'] = {b'ITMS': 'linear', b'FTMS': 'orbitrap'}[match.group(1)]
                    for z in charge_re.findall(text):
                        charges[int(z)] = charges.get(int(z), 0) + 1
                    n_ms2 += len(ms2_re.findall(text))
                    if not chunk:
                        break
            # Without a Thermo filter string, the analyzer (or Bruker amaZon model) of the instrument configuration
            if 'trap' not in found and ext == '.mzml':
                found['trap'] = next((t for a, t in ANALYZER_TRAPS.items() if a.encode() in header), None)
            elif 'trap' not in found and ext == '.mzxml':
                instrument = b' '.join(re.findall(rb'<(?:msModel|msMassAnalyzer)[^>]*value="([^"]*)"', header)).decode(errors = 'replace').lower()
                found['trap'] = next((t for a, t in MZXML_ANALYZER_TRAPS.items() if a in instrument), None)
            elif 'trap' not in found and ext == '.mgf' and b'NativeID:"scanId=' in header:
                # Agilent MassHunter native IDs (in msconvert's titles): a Q-TOF, read as TOF by the loader
                found['trap'] = 'other'
        elif ext == '.xlsx':
            # Files written by extract_spectra carry the ion mode and analyzer of their raw file
            df = pd.read_excel(path, usecols = lambda c: c in ('mode', 'trap', 'precursor_charge'))
            found.update({k: df[k].dropna().iloc[0] for k in ('mode', 'trap') if k in df.columns and df[k].notna().any()})
            charges = {int(z): int(n) for z, n in df['precursor_charge'].dropna().abs().value_counts().items()} if 'precursor_charge' in df.columns else {}
            n_ms2 = len(df)
        elif ext == '.cc':
            # extract_spectra's binary files carry the ion mode and analyzer in their JSON header, and precursor charges as a column of byte planes
            with np.load(path) as f:
                meta, n_ms2 = json.loads(f['meta'].tobytes()), f['peak_n'].shape[-1]
                found.update(mode = meta['attrs'].get('detected_mode'), trap = meta['attrs'].get('detected_trap'))
                key = f"col{meta['columns'].index('precursor_charge')}" if 'precursor_charge' in meta['columns'] else None
                if key in f.files:
                    z = np.abs(np.ascontiguousarray(f[key].T).view(meta['dtypes'][key]).ravel().astype(float))
                    charges = {int(c): int(k) for c, k in zip(*np.unique(z[~np.isnan(z)], return_counts = True))}
        elif ext == '.raw':
            from candycrunch.utils import read_raw_scan_table
            table = read_raw_scan_table(path)
            ms2 = [i for i, level in enumerate(table['ms_level']) if level == 2]
            n_ms2 = len(ms2)
            if ms2:
                found.update(mode = {'-': 'negative', '+': 'positive'}.get(table['polarity'][ms2[0]]), trap = {'ITMS': 'linear', 'FTMS': 'orbitrap'}.get(table['analyzer'][ms2[0]]))
            for i in ms2:
                if table['charge'][i]:
                    charges[int(table['charge'][i])] = charges.get(int(table['charge'][i]), 0) + 1
        elif ext == '.d' and os.path.exists(os.path.join(path, 'analysis.tdf')):
            import pathlib
            import sqlite3
            db = sqlite3.connect(pathlib.Path(path, 'analysis.tdf').resolve().as_uri() + '?mode=ro', uri = True)
            try:
                polarity = db.execute('SELECT Polarity FROM Frames WHERE Polarity IS NOT NULL LIMIT 1').fetchone()
                charges = dict(db.execute('SELECT Charge, COUNT(*) FROM Precursors WHERE Charge > 0 GROUP BY Charge').fetchall())
                n_ms2 = db.execute('SELECT COUNT(*) FROM Precursors').fetchone()[0]
            finally:
                db.close()
            found.update(mode = {'+': 'positive', '-': 'negative'}.get(polarity[0] if polarity else None), trap = 'other')
    except Exception:
        # An unreadable file is reported when it is run; here it only tells nothing
        pass
    name = os.path.basename(os.path.normpath(path)).lower()
    for key, choices in NAME_HINTS.items():
        named = [value for value, pattern in choices.items() if re.search(rf'(?<![a-z0-9])(?:{pattern})(?![a-z])', name)]
        if len(named) == 1:
            found[key] = named[0]
    results.put((path, {k: v for k, v in found.items() if v is not None}, charges, n_ms2))


class InferenceRunner(QObject):
    """Owns the worker process and relays its messages as signals"""
    ready, stage, progress, log, finished, failed = Signal(), Signal(object), Signal(object), Signal(str), Signal(object), Signal(str)

    def __init__(self):
        super().__init__()
        self.context, self.busy, self.dead, self.warm_up, self.report = mp.get_context('spawn'), False, False, True, None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.start()

    def start(self):
        # Not a daemon: wrap_inference_batch(n_jobs > 1) starts processes of its own, which daemons may not
        self.jobs, self.messages = self.context.Queue(), self.context.Queue()
        self.process = self.context.Process(target = _worker_main, args = (self.jobs, self.messages, self.warm_up), name = 'CandyCrunch worker')
        self.process.start()
        # Only the first worker warms up: after Cancel or a crash, the user is waiting to run again
        self.warm_up = False
        self.timer.start()

    def submit(self, experiments, weights):
        """experiments: (name, files, settings) per experiment"""
        self.busy = True
        self.jobs.put((experiments, weights))

    def cancel(self):
        self.process.terminate()
        self.process.join(5)
        self.forget()
        self.busy = False
        self.start()

    def shutdown(self):
        self.timer.stop()
        if self.process.is_alive():
            self.jobs.put(None)
            self.process.join(1.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        self.forget()

    def forget(self):
        """Removes the crash file of a worker that has ended"""
        with contextlib.suppress(OSError):
            os.remove(CRASH_FILE.format('worker', self.process.pid))

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
                (text, values), self.busy, self.dead = content, False, kind == 'fatal'
                self.report = {'values': values}
                self.failed.emit(text)
        if not self.process.is_alive() and not self.dead:
            self.timer.stop()
            if self.busy:
                self.busy = False
                try:
                    with open(CRASH_FILE.format('worker', self.process.pid), encoding = 'utf-8', errors = 'replace') as file:
                        dump = file.read()[-100000:]
                except OSError:
                    dump = ''
                self.report = {'files': {'worker_crash.txt': f'Exit code {self.process.exitcode}\n{dump}'}}
                self.failed.emit('The CandyCrunch worker stopped unexpectedly, most likely because it ran out of memory.')
            self.forget()
            self.start()


class GlycanImages(QObject):
    """SNFG drawings from GlycoDraw, rendered in a background thread and cached: compact ones for tables and lists, ones with linkages for the detail view"""
    ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.images, self.pending, self.requests, self.order, self.large = {}, set(), queue.PriorityQueue(), itertools.count(), []
        threading.Thread(target = self.render_loop, daemon = True).start()

    def get(self, glycan, compact = True, urgent = False, highlight = None):
        """Returns the QImage (null if GlycoDraw cannot draw it), or None while it is still being rendered; highlight: (residue indices to keep in full
        color while the rest is faded, index of the residue whose linkage to mark), e.g., the fragment of a supporting peak and its cleaved bond"""
        if not isinstance(glycan, str) or not glycan:
            return QImage()
        key = (glycan, compact) + ((highlight,) if highlight else ())
        # One lookup, as the drawing thread may evict the key between a membership test and the read
        image = self.images.get(key)
        if image is not None:
            return image
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
                # A hover highlight fades everything but the fragment and marks the cleaved bond in red, on the plain drawing's canvas, so nothing moves
                highlight = {'highlight_residues': list(key[2][0]), 'highlight_linkages': [key[2][1]] if key[2][1] is not None else None} if len(key) > 2 else {}
                # Compact drawings are shown at up to 0.4 px per SVG unit (tables, candidate lists), so 0.8 stays sharp on 2x screens at a sixth of the memory
                image = QImage.fromData(convert_svg_to_png(GlycoDraw(key[0], compact = key[1], suppress = True, **highlight).as_svg(), None, return_bytes = True,
                                                           scale = 0.8 if key[1] else 2))
            except Exception:
                image = QImage()
            self.images[key] = image
            if len(key) > 2 or not key[1]:
                # Drawings with linkages (detail view, previews, hover highlights) take 1.5 MB on average and up to 4 MB, so only the newest 60 are kept:
                # all of them held 360 MB once every peak of five runs had been looked at
                self.large.append(key)
                if len(self.large) > 60:
                    old = self.large.pop(0)
                    self.images.pop(old, None)
                    self.pending.discard(old)
            try:
                self.ready.emit(key)
            except RuntimeError:
                # Deleted while this drew (a closed results view, a test ending): nobody waits for drawings any more
                return


class Annotator(QObject):
    """Runs CandyCrumbs, supporting_ions and plot_annotated_spectrum in a background thread; only the newest request is worked on, so scrolling through a
    table never queues up stale spectra, and while idle it draws the spectra likely shown next (prefetch), which the panel then shows at once"""
    done = Signal(int, object, object, object, str)
    prefetched = Signal(str, object, object, object)

    def __init__(self):
        super().__init__()
        self.request, self.ahead, self.wake, self.lock, self.counter, self.handled, self.results = None, [], threading.Event(), threading.Lock(), 0, 0, {}
        threading.Thread(target = self.loop, daemon = True).start()

    def submit(self, request, ahead = ()):
        """request: None only cancels the current work (the panel shows a spectrum it already has); ahead: requests to draw while idle"""
        with self.lock:
            self.counter += 1
            self.request, self.ahead = (self.counter, request), list(ahead)
        self.wake.set()
        return self.counter

    def loop(self):
        rested = True
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    request_id, request = self.request
                    prefetch = request_id == self.handled
                    if prefetch and not self.ahead:
                        break
                    if prefetch and rested:
                        request = self.ahead.pop(0)
                    self.handled = request_id
                if prefetch and not rested:
                    # Spectra ahead wait until the one just shown is on screen: drawn in parallel, they slowed down its drawing in the window
                    rested = True
                    self.wake.wait(0.3)
                    continue
                rested = prefetch
                if request is None:
                    continue
                stale = lambda: self.request[0] != request_id
                width, height = request['figsize']
                try:
                    with IMPORT_LOCK:
                        from candycrunch.analysis import CandyCrumbs, plot_annotated_spectrum, supporting_ions, resolve_spectrum_input, classify_fragment, \
                            fragment_cartoon, fragIUPAC_to_image, is_composition, derivatize_carboxyls, deisotope_ms2
                        # fragIUPAC_to_image imports these itself, now outside MPL_LOCK too
                        with contextlib.suppress(ImportError):
                            import glycowork.motif.draw, glycorender.render, PIL.Image
                    # CandyCrumbs needs no matplotlib and can take long, so it runs before MPL_LOCK is taken: inside it, a slow MS3 annotation stalled the
                    # other panel's spectrum and every redraw in the window. Its results (and those of supporting_ions) are kept for the last spectra, which
                    # are drawn again at another panel size, or after a prefetch was cut short
                    key = repr([request[k] for k in ('structure', 'mzs', 'intensities', 'charge', 'mass_tag', 'sample_prep', 'tolerance', 'ppm')] + sorted(request['kwargs'].items()))
                    if len(self.results) > 200:
                        self.results.clear()
                    if key not in self.results:
                        # Isotope peaks stay unlabeled, as the pipeline scores deisotoped spectra: offered to CandyCrumbs, the M+1 and M+2 of carbon-rich
                        # (permethylated) fragments took cross-ring or three-cleavage labels. A pattern whose apex is more than twice its first peak keeps both: the
                        # first is a noise peak below a fragment as often as the monoisotopic peak of a heavy fragment, whose M+1 is often the most intense one
                        # (and more so when the isolation window centred the precursor's M+1)
                        peaks = dict(zip(request['mzs'], request['intensities']))
                        mono = sorted({m for relabel in (False, True) for m in deisotope_ms2(peaks, max(abs(request['charge']), 1), min(request['tolerance'], 0.05),
                                                                                              relabel = relabel)})
                        # A ppm tolerance alone bounds the matches (CandyCrumbs' Da window, 0.5 by default, is then wider at any fragment m/z)
                        self.results[key] = CandyCrumbs(request['structure'], mono, request['tolerance'] if request['ppm'] is None else None, mass_threshold_ppm = request['ppm'],
                                                        max_cleavages = request['kwargs'].get('max_cleavages', 3),
                                                        simplify = True, charge = request['charge'], mass_tag = request['mass_tag'], sample_prep = request['sample_prep'],
                                                        intensities = [peaks[m] for m in mono], **{k: v for k, v in request['kwargs'].items() if k != 'max_cleavages'})
                    hit_dict = self.results[key]
                    # Drawing a spectrum nobody waits for any more only delays the newer one
                    if stale():
                        continue
                    # The peaks supporting each residue of a structure (not of a composition, glycopeptide, or MS3 spectrum) are found before drawing (up to a
                    # second for large glycans), so the spectrum labels them whatever their intensity and draws their fragments first
                    support = None
                    if isinstance(request['structure'], str) and '(' in request['structure'] and '*' not in request['structure'] and request['kwargs'].get('ms3_precursor') is None:
                        support_key = key + repr(request['candidates'])
                        if support_key not in self.results:
                            try:
                                self.results[support_key] = supporting_ions(request['structure'], dict(zip(request['mzs'], request['intensities'])),
                                                                            charge = request['charge'], candidates = request['candidates'],
                                                                            mass_tag = request['mass_tag'], sample_prep = request['sample_prep'],
                                                                            sialic_derivatization = request['kwargs'].get('sialic_derivatization'))
                            except Exception:
                                self.results[support_key] = {'residues': [], 'candidates': []}
                        support = self.results[support_key]
                        if stale():
                            continue
                    diagnostic = sorted({s[0] for e in support['residues'] for s in e['support']} | {s[0] for c in support['candidates'] for s in c['support']} |
                                        {e['linkage'][0] for e in support['residues'] if request['ion_trap'] and e.get('linkage') and e['linkage'][2] == 'a2-3'}) if support else None
                    # The fragment cartoons plot_annotated_spectrum will place (glycan and oxonium ions of labeled or diagnostic peaks, ~30-50 ms each) are
                    # rendered into its cache here, outside MPL_LOCK (GlycoDraw and glycorender need no matplotlib) and one by one, so a newer request cuts it short
                    glycans = resolve_spectrum_input(request['structure'])[1] if request['cartoons'] else []
                    if glycans:
                        sialic = request['kwargs'].get('sialic_derivatization')
                        glycans = [derivatize_carboxyls(g, sialic) if sialic is not None and isinstance(g, str) and not is_composition(g) else g for g in glycans]
                        top = max(request['intensities'], default = 0)
                        for mz, intensity in zip(request['mzs'], request['intensities']):
                            hit = hit_dict.get(mz)
                            if hit and classify_fragment(hit['Domon-Costello nomenclatures'][0]) in ('oxonium', 'glycan') and (
                                    (top > 0 and intensity / top >= 0.05) or (diagnostic and np.isclose(diagnostic, mz, rtol = 0, atol = 0.0051).any())):
                                frag_iupac = fragment_cartoon(glycans, hit['Domon-Costello nomenclatures'][0])
                                if frag_iupac is not None and request['toggles'].get(frag_iupac) is not False:
                                    # At the resolution plot_annotated_spectrum renders them for: its glycan_zoom at the figure's 100 dpi on this screen
                                    fragIUPAC_to_image(frag_iupac, scale = 0.25 * (100 * request['ratio']) / 72)
                                if stale():
                                    break
                        if stale():
                            continue
                    figure = Figure(figsize = request['figsize'], dpi = 100)
                    FigureCanvasAgg(figure)
                    # Margins in inches; with cartoons, the title sits at 1.4 axes heights above the bottom, so the axes top leaves room for it
                    bottom = 0.55 / height
                    figure.subplots_adjust(left = 0.65 / width, right = 1 - 0.2 / width, bottom = bottom, top = bottom + (1 - 0.4 / height - bottom) / 1.4 if request['cartoons'] else 1 - 0.4 / height)
                    with MPL_LOCK:
                        hit_dict, ax = plot_annotated_spectrum(request['structure'], request['mzs'], request['intensities'], mass_threshold = request['tolerance'],
                                                               charge = request['charge'], mass_tag = request['mass_tag'], sample_prep = request['sample_prep'],
                                                               ax = figure.add_subplot(), draw_glycans = request['cartoons'], hit_dict = hit_dict,
                                                               diagnostic = diagnostic, cartoons = request['toggles'], cartoon_dpi = 100 * request['ratio'],
                                                               **request['kwargs'])
                        # Save image… draws the spectrum again with the same diagnostic peaks
                        figure.diagnostic = diagnostic
                        if request['ms3']:
                            # A triangle under every peak that was isolated for MS3 and an invisible stick over that peak, both clickable
                            mzs, ints = np.array(request['mzs']), np.array(request['intensities'])
                            near = [int(np.argmin(np.abs(mzs - p))) for p in request['ms3']]
                            hit = [abs(mzs[k] - p) <= request['tolerance'] for k, p in zip(near, request['ms3'])]
                            xs = [mzs[k] if h else p for k, h, p in zip(near, hit, request['ms3'])]
                            figure.ms3_artists = (ax.scatter(xs, [0] * len(xs), marker = '^', s = 70, c = CANDY, edgecolors = 'white', linewidths = 0.8, zorder = 6,
                                                             clip_on = False, picker = True),
                                                  ax.vlines(xs, 0, [ints[k] / (ints.max() or 1) * 100 if h else 0 for k, h in zip(near, hit)], colors = 'none', picker = 5))
                    if prefetch:
                        self.prefetched.emit(request['key'], figure, hit_dict, support)
                    else:
                        self.done.emit(request_id, figure, hit_dict, support, '')
                except Exception as error:
                    try:
                        if not prefetch:
                            self.done.emit(request_id, None, None, None, f'{type(error).__name__}: {error}')
                    except RuntimeError:
                        # The panel was deleted while this ran (an emit raised 'Signal source has been deleted'), so nobody waits for the result
                        return


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
        from glycowork.motif.processing import canonicalize_composition
    # sialic acids of known linkage (Neu5Ac_a2-6, from a linkage-specific derivatization) are written as the E and L of shorthand compositions (H5N4E1L1)
    return (canonicalize_composition(composition, as_string = True) if any('_a2-' in k for k in composition) else stringify_dict(composition)) if isinstance(
        composition, dict) and composition else ''


class TableModel(QAbstractTableModel):
    """Read-only view of one result table, column by column as ResultsView lays them out; curation lives in the dataframe's own columns"""
    BAR_ROLE, IMAGE_ROLE = Qt.UserRole + 1, Qt.UserRole + 2

    def __init__(self, images):
        super().__init__()
        self.images, self.df, self.columns, self.search, self.marks = images, pd.DataFrame(), [], [], []
        images.ready.connect(self.image_ready)

    def set_table(self, df, columns):
        self.beginResetModel()
        self.df, self.columns = df, columns
        self.search = [' '.join(str(c['text'][r]).lower() for c in columns) for r in range(len(df))]
        # data() is asked for every role of every cell it sizes and paints, so the excluded and curated flags are not looked up in the dataframe each time
        self.marks = [(bool(e), bool(c)) for e, c in zip(df['excluded'], df['curation'])]
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
        if role == Qt.ItemDataRole.DisplayRole:
            return column['text'][row]
        if role == Qt.ItemDataRole.UserRole:
            return column['raw'][row]
        if role == self.BAR_ROLE:
            return column.get('bar', [None] * (row + 1))[row]
        if role == self.IMAGE_ROLE:
            return column.get('image', column['raw'])[row]
        if role == Qt.ItemDataRole.ToolTipRole:
            return column.get('tips', column['text'])[row] or None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int((Qt.AlignmentFlag.AlignRight if column['kind'] in ('number', 'bar') else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
        excluded, curated = self.marks[row]
        if role == Qt.ItemDataRole.ForegroundRole and excluded:
            return QColor(MUTED)
        if role == Qt.ItemDataRole.FontRole and (excluded or curated):
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
        # The raw values straight from the source model: going through data() cost two Python calls per comparison
        column = self.sourceModel().columns[left.column()]
        raw = column.get('sort', column['raw'])
        a, b = raw[left.row()], raw[right.row()]
        if a is None or b is None:
            # Empty cells go last in either order (Qt sorts in descending order with the arguments swapped)
            return (a is None) != (b is None) and (a is None) == (self.sortOrder() == Qt.SortOrder.DescendingOrder)
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class StructureDelegate(QStyledItemDelegate):
    """Paints the SNFG drawing of a cell's structure, or its text until the drawing is ready (or if GlycoDraw cannot draw it)"""

    def __init__(self, images, parent):
        super().__init__(parent)
        self.images, self.shown = images, {}

    def paint(self, painter, option, index):
        image = self.images.get(index.data(TableModel.IMAGE_ROLE))
        if image is None or image.isNull():
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ''
        opt.widget.style().drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = opt.rect.adjusted(6, 4, -6, -4)
        scale = min(rect.width() / image.width(), rect.height() / image.height(), 0.5)
        # Shrunk to its size on this screen by Qt's area-averaging scaler and kept per size: drawn shrunk, the painter's bilinear filter skipped pixels
        # beyond half size (the 0.8 px per SVG unit of the drawings to 0.4 on a screen without scaling), which made thin strokes grainy
        ratio = painter.device().devicePixelRatioF()
        size = QSize(round(image.width() * scale * ratio), round(image.height() * scale * ratio))
        key = (index.data(TableModel.IMAGE_ROLE), size.width(), size.height())
        if key not in self.shown:
            if len(self.shown) > 2000:
                self.shown.clear()
            self.shown[key] = image if size == image.size() else image.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if opt.font.strikeOut():
            painter.setOpacity(0.3)
        painter.drawImage(QRectF(rect.x(), rect.y() + (rect.height() - image.height() * scale) / 2, image.width() * scale, image.height() * scale), self.shown[key])
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
    # (structure, fragment residues, residue of the cleaved linkage) while the mouse is on a supporting peak's marker, None when it leaves it
    support_hovered = Signal(object)

    def __init__(self):
        super().__init__()
        self.annotator, self.request, self.latest, self.canvas, self.toolbar, self.drawn = Annotator(), None, 0, None, None, None
        self.spec, self.ahead, self.cache = None, [], {}
        self.annotator.done.connect(self.show_figure)
        self.annotator.prefetched.connect(self.store)
        self.fragment_html, self.hover_targets, self.label_targets, self.hovered, self.toggles = '', [], [], None, {}
        self.controls = QHBoxLayout()
        self.controls.setContentsMargins(6, 4, 6, 0)
        self.summary = _label('', 'hint')
        self.summary.setWordWrap(False)
        # Ignored: the summary's length changes with every annotation, and as a minimum width it resized the panel, which re-annotated it, endlessly
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.tolerance = _spin(0.001, 2.0, 0.05, 3)
        self.tolerance.setValue(DEFAULTS['ppm_thresh'] * MZ_REF / 1e6)
        self.tolerance.setToolTip('Maximum difference between an observed peak and a theoretical fragment mass')
        # A flat window for ion-trap MS2, ppm of each theoretical fragment m/z for accurate MS2 (orbitrap), whose errors scale with m/z
        self.unit = _combo([('Da', 'Da'), ('ppm', 'ppm')])
        self.unit.setToolTip('Unit of the tolerance: a flat window in Da, or ppm of each theoretical fragment m/z (accurate MS2, e.g., orbitrap). Switching converts the value at m/z 1000')
        self.cartoons = QCheckBox('Fragment cartoons')
        self.cartoons.setChecked(True)
        self.fragment_types = _combo(FRAGMENT_TYPES)
        self.fragment_types.setToolTip('Fragment types to annotate with. Automatic: glycosidic fragments (B, C, Y, Z, also internal ones) for permethylated glycans, whose '
                                       'cross-ring readings matched no more peaks than in m/z-shifted spectra (CID and HCD of protonated ions); otherwise also X '
                                       'cross-rings, and A cross-rings in negative mode and of the reducing end of metal-adducted unreduced glycans in positive mode. '
                                       'All cross-rings: every A and X cross-ring, e.g., for sodiated permethylated glycans at high collision energy or in MSn')
        # The figure on screen is laid out for the panel, so saving it gave a small picture with cramped labels; this draws it again at print size
        self.save = QPushButton('Save image…')
        self.save.setToolTip('Saves this annotated spectrum at print size (12 inches wide) as PNG (300 dpi), SVG, or PDF, with the m/z of every labeled peak')
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_image)
        self.controls.addStretch(1)
        self.controls.addWidget(QLabel('Tolerance'))
        self.controls.addWidget(self.tolerance)
        self.controls.addWidget(self.unit)
        self.controls.addWidget(QLabel('Fragments'))
        self.controls.addWidget(self.fragment_types)
        self.controls.addWidget(self.cartoons)
        self.controls.addWidget(self.save)
        self.message = _label('', 'hint')
        self.message.setAlignment(Qt.AlignCenter)
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
        self.unit.currentIndexChanged.connect(lambda: self.set_tolerance(self.tolerance.value() * (1000 if self.unit.currentData() == 'ppm' else 1e-3), self.unit.currentData()))
        self.unit.currentIndexChanged.connect(self.resubmit)
        self.cartoons.toggled.connect(self.resubmit)
        self.fragment_types.currentIndexChanged.connect(self.resubmit)
        # Labels and cartoons are placed for one figure size, so a resized panel gets a freshly laid out spectrum once resizing stops
        self.relayout = QTimer(self)
        self.relayout.setSingleShot(True)
        self.relayout.setInterval(400)
        self.relayout.timeout.connect(self.resubmit)

    def set_tolerance(self, value, unit):
        """Shows value in unit ('Da' or 'ppm') in the tolerance box, without annotating again"""
        self.tolerance.blockSignals(True)
        self.unit.blockSignals(True)
        self.unit.setCurrentIndex(self.unit.findData(unit))
        self.tolerance.setDecimals(3 if unit == 'Da' else 1)
        self.tolerance.setRange(*((0.001, 2.0) if unit == 'Da' else (1.0, 2000.0)))
        self.tolerance.setSingleStep(0.05 if unit == 'Da' else 1.0)
        self.tolerance.setValue(value)
        self.tolerance.blockSignals(False)
        self.unit.blockSignals(False)

    def annotate(self, *args, prefetch = (), **kwargs):
        """args and kwargs as make_request takes them; prefetch: (args, kwargs) of the spectra likely shown next (e.g., the neighboring table rows),
        which are drawn ahead while the panel is idle and then appear at once"""
        self.spec, self.ahead, self.toggles = (args, kwargs), list(prefetch), {}
        self.resubmit()

    def make_request(self, structure, mzs, intensities, charge, mass_tag, sample_prep, ms3 = (), candidates = (), toggles = None, ion_trap = False, **kwargs):
        """ms3: (m/z, number of MS3 spectra) of the peaks isolated for MS3, which get a clickable marker that emits ms3_clicked with their index;
        candidates: competing structures whose distinguishing peaks are listed; ion_trap: whether the MS2 spectra are ion-trap CID, the only data the linkage
        reading of a lone Neu5Ac (supporting_ions) was measured on; the request is drawn for the panel's current size and settings"""
        size = self.stack.size()
        choice = self.fragment_types.currentData()
        if choice == 'glycosidic' or (choice is None and sample_prep == 'permethylated'):
            kwargs = {'disable_X_cross_rings': True, 'disable_A_cross_rings': True} | kwargs
        elif choice == 'cross_rings':
            kwargs = {'disable_X_cross_rings': False, 'disable_A_cross_rings': False} | kwargs
        # With a ppm tolerance, 'tolerance' is its Da value at m/z 1000, the flat window of deisotoping and of the MS3 markers
        ppm = self.tolerance.value() if self.unit.currentData() == 'ppm' else None
        request = {'structure': structure, 'mzs': [float(x) for x in mzs], 'intensities': [float(x) for x in intensities], 'charge': int(charge),
                   'mass_tag': mass_tag, 'sample_prep': sample_prep, 'ms3': [float(p) for p, _ in ms3], 'ms3_spectra': [int(n) for _, n in ms3],
                   'candidates': list(candidates), 'kwargs': kwargs, 'tolerance': self.tolerance.value() if ppm is None else ppm * 1e-3, 'ppm': ppm,
                   'cartoons': self.cartoons.isChecked(), 'toggles': dict(toggles or {}), 'ion_trap': bool(ion_trap),
                   'figsize': (max(size.width(), 500) / 100, max(size.height(), 300) / 100), 'ratio': self.devicePixelRatioF()}
        request['key'] = repr(sorted((k, sorted(v.items()) if isinstance(v, dict) else v) for k, v in request.items()))
        return request

    def resubmit(self):
        if self.spec is None:
            return
        self.request = self.make_request(*self.spec[0], toggles = self.toggles, **self.spec[1])
        ahead = [r for r in (self.make_request(*a, **k) for a, k in self.ahead) if r['key'] not in self.cache and r['key'] != self.request['key']]
        if self.request['key'] in self.cache:
            # Drawn before or ahead: shown at once, and the annotation thread drops whatever it was doing for the new neighbors
            entry = self.cache[self.request['key']]
            self.latest = self.annotator.submit(None, ahead)
            self.show_figure(self.latest, entry['figure'], entry['hit_dict'], entry['support'], '')
        else:
            self.latest = self.annotator.submit(dict(self.request), ahead)
            self.summary.setText(f'Annotating with CandyCrumbs: {self.request["structure"]}')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.request is not None and abs(event.size().height() - event.oldSize().height()) + abs(event.size().width() - event.oldSize().width()) > 40:
            self.relayout.start()

    def event(self, event):
        # Moved to a screen of another pixel ratio, the spectrum is drawn again, as its cartoons were rendered for the old one
        if event.type() == QEvent.DevicePixelRatioChange and self.request is not None:
            self.relayout.start()
        return super().event(event)

    def show_message(self, text):
        self.request, self.spec, self.hover_targets, self.label_targets, self.drawn = None, None, [], [], None
        self.save.setEnabled(False)
        if self.hovered is not None:
            self.hovered = None
            self.support_hovered.emit(None)
        self.latest = self.annotator.counter + 1
        self.message.setText(text)
        self.stack.setCurrentWidget(self.message)
        self.summary.setText('')
        self.fragments.setHtml(f'<p style="color:{MUTED}">{text}</p>')

    def store(self, key, figure, hit_dict, support):
        """Keeps the last 12 spectra drawn (current, earlier, or ahead), with their canvas and toolbar once shown, so returning to one or moving on to a
        prefetched one shows it at once"""
        if key not in self.cache:
            self.cache[key] = {'figure': figure, 'hit_dict': hit_dict, 'support': support, 'canvas': None, 'toolbar': None, 'markers': None}
        self.cache[key] = self.cache.pop(key)
        while len(self.cache) > 12:
            old = self.cache.pop(next(iter(self.cache)))
            if old['canvas'] is not None and old['canvas'] is not self.canvas:
                old['canvas'].deleteLater()
                old['toolbar'].deleteLater()

    def show_figure(self, request_id, figure, hit_dict, support, error):
        if request_id != self.latest:
            return
        if error:
            # Nothing of the spectrum shown before stays: its fragment table, or Save image…, which would save that spectrum
            self.drawn = None
            self.save.setEnabled(False)
            self.fragments.setHtml(f'<p style="color:{MUTED}">CandyCrumbs could not annotate this spectrum with {self.request["structure"]}</p>')
            self.message.setText(f'CandyCrumbs could not annotate this spectrum with {self.request["structure"]}.\n\n{error}')
            self.stack.setCurrentWidget(self.message)
            self.summary.setText('')
            return
        self.store(self.request['key'], figure, hit_dict, support)
        entry = self.cache[self.request['key']]
        # The spectrum shown before stays hidden with its canvas while it is cached
        for widget, layout in ((self.canvas, self.holder_layout), (self.toolbar, self.controls)):
            if widget is not None:
                widget.hide()
                layout.removeWidget(widget)
        if self.canvas is not None and all(e['canvas'] is not self.canvas for e in self.cache.values()):
            self.canvas.deleteLater()
            self.toolbar.deleteLater()
        if entry['canvas'] is None:
            entry['canvas'] = Canvas(figure)
            # Pan and zoom hold the canvas' widgetlock, so clicks while using them never count as picks
            entry['canvas'].mpl_connect('pick_event', lambda event: event.artist in getattr(figure, 'ms3_artists', ()) and len(event.ind) and self.ms3_clicked.emit(int(event.ind[0])))
            entry['canvas'].mpl_connect('motion_notify_event', self.hover)
            entry['canvas'].mpl_connect('button_press_event', self.toggle_cartoon)
            entry['toolbar'] = NavigationToolbar2QT(entry['canvas'], self, coordinates = False)
            entry['toolbar'].setIconSize(QSize(16, 16))
            # Its save button would save the small on-screen figure; Save image… draws it at print size instead
            for action in entry['toolbar'].actions():
                if action.text() == 'Save':
                    entry['toolbar'].removeAction(action)
        self.canvas, self.toolbar, self.hover_targets = entry['canvas'], entry['toolbar'], []
        if self.hovered is not None:
            self.hovered = None
            self.support_hovered.emit(None)
        self.drawn = (dict(self.request), hit_dict, getattr(figure, 'diagnostic', None))
        self.save.setEnabled(True)
        self.controls.insertWidget(0, self.toolbar)
        self.holder_layout.addWidget(self.canvas)
        self.toolbar.show()
        self.canvas.show()
        self.stack.setCurrentWidget(self.holder)
        peaks = dict(zip(self.request['mzs'], self.request['intensities']))
        total, top = sum(peaks.values()) or 1, max(peaks.values()) or 1
        hits = {mz: hit for mz, hit in hit_dict.items() if hit}
        self.summary.setText(f'{len(hits)} of {len(peaks)} peaks annotated, {sum(peaks.get(mz, 0) for mz in hits) / total:.0%} of the ion current' + (
            f'; \u25b2 marks the {len(self.request["ms3"])} peak(s) with MS3 spectra, click one to annotate them' if self.request['ms3'] else ''))
        if self.request['kwargs'].get('ms3_precursor') is not None and not hits:
            self.summary.setText(f'm/z {self.request["kwargs"]["ms3_precursor"]:.2f} is no fragment of {self.request["structure"]}, so none of its MS3 peaks can be annotated')
        from candycrunch.analysis import domon_costello_to_html, PROTON_MASS
        if getattr(figure, 'ms3_artists', None):
            # Hovering a peak isolated for MS3 (its marker or the peak itself) names its fragment and says what clicking does
            tips = []
            for x, p, n in zip(np.asarray(figure.ms3_artists[0].get_offsets())[:, 0], self.request['ms3'], self.request['ms3_spectra']):
                names = hits[x]['Domon-Costello nomenclatures'][0] if x in hits else []
                flat = [y for sub in names for y in sub] if names and isinstance(names[0], list) else list(names)
                tips.append((None, f'<b>m/z {p:.2f}</b>' + (f' ({domon_costello_to_html(flat)})' if flat else '') + f' was isolated for MS3: {n} spectr{"um" if n == 1 else "a"}'
                             f'<br>Click to show {"it" if n == 1 else "them"}, pooled and annotated as fragments of this fragment, in the MS3 tab'))
            self.hover_targets += [(artist, tips) for artist in figure.ms3_artists]
        rows, tips = [], {}
        for mz in sorted(hits):
            hit = hits[mz]
            for n, (names, theo, z) in enumerate(zip(hit['Domon-Costello nomenclatures'], hit['Theoretical fragment masses'], hit['Fragment charges'])):
                # CandyCrumbs gives a multiply charged fragment as its singly charged mass, so it is turned into the m/z it is observed at
                theo = (theo + np.sign(z) * (abs(z) - 1) * PROTON_MASS) / abs(z)
                flat = [x for sub in names for x in sub] if names and isinstance(names[0], list) else list(names)
                first = f'<td>{mz:.4f}</td><td align="right">{peaks.get(mz, 0) / top * 100:.1f}</td>' if n == 0 else '<td></td><td></td>'
                rows.append(f'<tr>{first}<td>{domon_costello_to_html(flat)}</td><td>{theo:.4f}</td><td align="right">{theo - mz:+.3f}</td><td align="right">{(theo - mz) / theo * 1e6:+.1f}</td><td align="right">{z}</td></tr>')
                tips[mz] = tips.get(mz, f'Observed m/z <b>{mz:.4f}</b>') + f'<br>{domon_costello_to_html(flat)}: theoretical m/z <b>{theo:.4f}</b> (error {theo - mz:+.4f} Da' + (f', z = {z}' if abs(z) > 1 else '') + ')'
        self.fragment_html = (f'<table cellspacing="0" cellpadding="3" width="100%"><tr style="background:{WASH}"><th align="left">Observed m/z</th>'
                              f'<th align="right">Rel. int. (%)</th><th align="left">Fragment</th><th align="left">Theoretical m/z</th><th align="right">Error (Da)</th>'
                              f'<th align="right">Error (ppm)</th><th align="right">z</th></tr>{"".join(rows)}</table>')
        self.fragments.setHtml(self.fragment_html)
        # Hovering a fragment label shows the observed and theoretical m/z of its peak
        fragments = getattr(figure.axes[0], 'cartoon_fragments', {})
        self.label_targets = [(text, [(None, tips[text.xy[0]] + (f'<br>Click to {"hide" if fragments[text.xy[0]][1] else "draw"} its cartoon' if text.xy[0] in fragments else ''))])
                              for text in figure.axes[0].texts if text.get_text() and text.xy[0] in tips]
        self.show_support(support)

    def save_image(self):
        """Draws the spectrum shown again at print size, with the m/z of every labeled peak, as PNG (300 dpi), SVG, or PDF"""
        if self.drawn is None:
            return
        path, chosen = QFileDialog.getSaveFileName(self, 'Save annotated spectrum', os.path.join(QSettings().value('folder', os.path.expanduser('~')), 'annotated_spectrum.png'),
                                                   'PNG image, 300 dpi (*.png);;SVG vector graphic (*.svg);;PDF (*.pdf)')
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += re.search(r'\*(\.\w+)', chosen).group(1)
        request, hit_dict, diagnostic = self.drawn
        width, height = 12.0, 6.0 if request['cartoons'] else 4.6
        figure = Figure(figsize = (width, height), dpi = 100)
        FigureCanvasAgg(figure)
        bottom = 0.55 / height
        figure.subplots_adjust(left = 0.65 / width, right = 1 - 0.2 / width, bottom = bottom, top = bottom + (1 - 0.4 / height - bottom) / 1.4 if request['cartoons'] else 1 - 0.4 / height)
        with IMPORT_LOCK:
            from candycrunch.analysis import plot_annotated_spectrum
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with MPL_LOCK:
                plot_annotated_spectrum(request['structure'], request['mzs'], request['intensities'], mass_threshold = request['tolerance'], charge = request['charge'],
                                        mass_tag = request['mass_tag'], sample_prep = request['sample_prep'], ax = figure.add_subplot(), draw_glycans = request['cartoons'],
                                        hit_dict = hit_dict, mz_labels = True, label_fontsize = 7, diagnostic = diagnostic, cartoons = request['toggles'],
                                        cartoon_dpi = 300, **request['kwargs'])
                figure.savefig(path, dpi = 300, bbox_inches = 'tight')
        except Exception as error:
            QMessageBox.critical(self, 'Save image', f'Could not write {path}:\n{error}')
        finally:
            QApplication.restoreOverrideCursor()

    def show_support(self, support):
        """Marks the peaks that support a residue's placement (lilac) or fit another placement instead (red) and lists them above the fragment table"""
        if not support or not (support['residues'] or support['candidates']):
            return
        from candycrunch.analysis import domon_costello_to_html, glycan_to_nxGraph
        tested = [e for e in support['residues'] if e['support'] or e['open']]
        partly = sum(bool(e['support'] and e['open']) for e in tested)
        # The linkage of a lone Neu5Ac, read from how intense [M-H-Neu5Ac]- is (ion-trap CID only): no colors of its own, a peak that reads a2-3 gets the
        # lilac marker if the structure says a2-3 and the red one if it says a2-6; a reading of a2-6 is a missing peak, so it is only listed
        read = {id(e): e['linkage'] for e in support['residues'] if e.get('linkage') and self.request['ion_trap']}
        linkage = {k: f'{v[0]:.2f} ' * (v[2] == 'a2-3') + f'[M-H-Neu5Ac]<sup>-</sup> at {v[1]:.0%} of the base peak: {v[2]}' for k, v in read.items()}
        self.summary.setText(self.summary.text() + f'; {sum(bool(e["support"]) and not e["open"] for e in tested)} of {len(tested)} residues placed by diagnostic fragments' + (f', {partly} narrowed down' if partly else '') +
                             ''.join(f'; Neu5Ac linkage reads {v[2]}' + ('' if v[3] else ', not as shown') for v in read.values()))
        entry = self.cache[self.request['key']]
        if entry['markers'] is None:
            # Each marker carries what hovering it shows: the supported residues' fragment and cleaved bond in the drawing (lilac) and a tooltip naming
            # the fragment; added once per cached figure
            ax, points, entry['markers'] = self.canvas.figure.axes[0], {'support': {}, 'against': {}}, []
            for e in support['residues']:
                for s in e['support']:
                    points['support'].setdefault(s[0], [s[1], (self.request['structure'], s[5], s[6]), f'{s[0]:.2f}: {domon_costello_to_html(list(s[2]))} = {s[3]}<br>rules out'])[2] += f'<br>{", ".join(s[4])}'
                for a in e['against']:
                    points['against'].setdefault(a[0], [a[1], None, f'{a[0]:.2f}: {domon_costello_to_html(list(a[2]))} = {a[3]}<br>fits'])[2] += f'<br>{a[4]}'
            for e in support['residues']:
                if id(e) in read and read[id(e)][2] == 'a2-3':
                    residues = range(len(glycan_to_nxGraph(self.request['structure'])) // 2 + 1)
                    points['support' if read[id(e)][3] else 'against'].setdefault(read[id(e)][0], [read[id(e)][1], (self.request['structure'], tuple(i for i in residues if i != e['residue']), e['residue']), f'{linkage[id(e)]}<br>15% or more reads a2-3 (ion-trap CID)'])
            for color, side in ((LILAC, 'support'), (CANDY, 'against')):
                if points[side]:
                    artist = ax.scatter(list(points[side]), [v[0] * 100 + 3 for v in points[side].values()], marker = 'v', s = 45, c = color, edgecolors = 'white',
                                        linewidths = 0.6, zorder = 7, clip_on = False)
                    entry['markers'].append((artist, [v[1:] for v in points[side].values()]))
            self.canvas.draw_idle()
        self.hover_targets += entry['markers']
        rows = [f'<tr><td>{e["name"]}</td><td>{"<br>".join([f"{s[0]:.2f} {domon_costello_to_html(list(s[2]))} = {s[3]}" for s in e["support"][:3]] + [linkage.get(id(e))] * (id(e) in read and read[id(e)][3])) or "none"}</td>'
                f'<td>{", ".join(e["open"][:3]) + (", ..." if len(e["open"]) > 3 else "")}</td>'
                f'<td style="color:{CANDY}">{"<br>".join([f"{a[0]:.2f} fits {a[4]}" for a in e["against"][:2]] + [linkage.get(id(e))] * (id(e) in read and not read[id(e)][3]))}</td></tr>'
                for e in tested + [e for e in support['residues'] if id(e) in read and e not in tested]]
        rows += [f'<tr><td>vs {c["structure"]}</td><td>{"<br>".join(f"{s[0]:.2f} {domon_costello_to_html(list(s[2]))}" for s in c["support"][:3]) or "none"}</td><td></td>'
                 f'<td style="color:{CANDY}">{"<br>".join(f"{a[0]:.2f} {domon_costello_to_html(list(a[2]))}" for a in c["against"][:3])}</td></tr>' for c in support['candidates']]
        self.fragments.setHtml(f'<p><b>Supported by</b> <span style="color:{MUTED}">(intense peaks of a single glycosidic cleavage that no other placement of the residue, '
                               f'or the other candidate, can form with up to two cleavages; marked lilac in the spectrum, peaks that fit only the alternative in red'
                               f'{"; the linkage of a lone Neu5Ac from how intense [M-H-Neu5Ac]<sup>-</sup> is, 15% of the base peak or more for a2-3 in ion-trap CID" if read else ""})'
                               f'</span></p><table cellspacing="0" cellpadding="3" width="100%"><tr style="background:{WASH}"><th align="left">Residue</th>'
                               f'<th align="left">Supporting peaks</th><th align="left">Not told apart from</th><th align="left">Peaks against</th></tr>{"".join(rows)}'
                               f'</table><br>' + self.fragment_html)

    def toggle_cartoon(self, event):
        """A click on a fragment label hides that fragment's cartoon if it is drawn, else draws it (every peak of one fragment shares its cartoon)"""
        # Pan and zoom clicks are not label clicks; hit-testing a label lays out its mathtext, which waits for no annotation thread
        if self.toolbar.mode or event.button != 1 or self.request is None or not MPL_LOCK.acquire(blocking = False):
            return
        try:
            fragments = getattr(self.canvas.figure.axes[0], 'cartoon_fragments', {})
            text = next((t for t in self.canvas.figure.axes[0].texts if t.get_text() and t.xy[0] in fragments and t.contains(event)[0]), None)
        finally:
            MPL_LOCK.release()
        if text is not None:
            frag_iupac, drawn = fragments[text.xy[0]]
            self.toggles[frag_iupac] = not drawn
            self.resubmit()

    def hover(self, event):
        """Shows the fragment of the supporting (or contradicting) peak under the mouse as a tooltip, and its supported part in the SNFG drawing; over a
        peak isolated for MS3, what clicking it does, with a hand cursor"""
        highlight, tip, clickable = None, '', False
        # Hit-testing a label lays out its text, which must wait while an annotation thread draws (matplotlib's mathtext parser is not thread-safe); markers
        # need no lock, so they answer while spectra ahead are drawn
        locked = MPL_LOCK.acquire(blocking = False)
        try:
            # Newest first, so support markers win over the invisible MS3 sticks under them, and fragment labels (text, no 'ind') last
            for artist, meta in list(reversed(self.hover_targets)) + (self.label_targets if locked else []):
                hit, info = artist.contains(event)
                if hit:
                    highlight, tip = meta[info.get('ind', [0])[0]]
                    clickable = artist in getattr(self.canvas.figure, 'ms3_artists', ()) or getattr(artist, 'xy', (None,))[0] in getattr(self.canvas.figure.axes[0], 'cartoon_fragments', {})
                    break
        finally:
            if locked:
                MPL_LOCK.release()
        # While panning or zooming, the toolbar sets its own cursor
        if not self.toolbar.mode:
            self.canvas.set_cursor(Cursors.HAND if clickable else Cursors.POINTER)
        if tip:
            QToolTip.showText(QCursor.pos(), tip, self.canvas)
        elif self.hovered is not None or QToolTip.isVisible():
            QToolTip.hideText()
        if highlight != self.hovered:
            self.hovered = highlight
            self.support_hovered.emit(highlight)


class SettingsPanel(QScrollArea):
    """Input files and every wrap_inference / wrap_inference_batch setting"""
    files_changed = Signal()
    # Every settings check of a run just added (_detect_run) has answered or was given up
    detections_done = Signal()
    # (model file, number of structures it predicts, why it cannot be used), from the process that checked an imported model
    model_checked = Signal(str, object, str)

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
        # As high as its runs, from 110 to 240 px: at its default height the empty list pushed Advanced settings out of view in a 720 px high window
        self.files.setFixedHeight(110)
        self.files.setToolTip('Runs of one experiment are predicted together and harmonized into one feature table; experiments are predicted and reported '
                              'separately, each with its own settings. Double-click an experiment to rename it, right-click runs to move them.')
        layout.addWidget(self.files)
        buttons = QGridLayout()
        self.add_files, self.add_folder, self.experiment_button, self.remove = QPushButton('Add files…'), QPushButton('Add folder…'), QPushButton('New experiment'), QPushButton('Remove')
        self.experiment_button.setToolTip('Moves the selected runs into a new experiment with its own settings, for samples that must not be compared with the others')
        for n, button in enumerate((self.add_files, self.add_folder, self.experiment_button, self.remove)):
            buttons.addWidget(button, n // 2, n % 2)
        layout.addLayout(buttons)
        self.file_hint = _label('Thermo .raw, Bruker timsTOF .d (DDA-PASEF), mzML, mzXML, mgf, or a .cc or .xlsx made by extract_spectra. Drop files or folders anywhere in this window.', 'hint')
        layout.addWidget(self.file_hint)
        self.inputs = {}
        self.sample_heading = _label('Sample')
        layout.addWidget(self.sample_heading)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.inputs['glycan_class'] = _combo(GLYCAN_CLASSES)
        self.inputs['glycan_class'].setToolTip('Glycan class of the samples. N- and O-glycans measured in one run: candidate compositions and structures come from '
                                               'both classes, and only structures of other classes need the cross-class confidence')
        self.mode = _combo([('Negative', 'negative'), ('Positive', 'positive')])
        self.mode.setToolTip('Overridden by the polarity stored in .raw/mzML/mzXML files and in .xlsx files made by extract_spectra')
        self.charge = _spin(1, 8, 1)
        self.charge.setToolTip('Highest absolute precursor charge considered for composition matching')
        self.inputs['modification'] = _combo(REDUCING_ENDS)
        self.inputs['mass_tag'] = _spin(-1000, 3000, 1, 4, ' Da')
        self.inputs['mass_tag'].setToolTip('Mass the reducing-end label adds to the free glycan')
        self.inputs['sample_prep'] = _combo(SAMPLE_PREPS)
        self.sialic = _combo(SIALIC_DERIVATIZATIONS)
        self.sialic.setToolTip(SIALIC_TIP)
        # As wide as their longest choice, these two made the panel wider than its view, which cut off its right edge (the hint, combo arrows, Restore defaults)
        for box in (self.inputs['glycan_class'], self.sialic):
            box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.inputs['lc'] = _combo(LCS)
        self.inputs['trap'] = _combo(TRAPS)
        self.inputs['trap'].setToolTip('Overridden by the instrument stored in .raw/mzML/mzXML files and in .xlsx files made by extract_spectra')
        # Imported models are copies in the app's data folder, so they stay available wherever their file was; each experiment keeps its model
        self.model_dir = os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), 'models')
        self.model = _combo([('CandyCrunch (built-in)', None)] + [(os.path.splitext(os.path.basename(p))[0], p) for p in
                                                                   sorted(glob.glob(os.path.join(self.model_dir, '*.pt')), key = str.lower)])
        self.model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model.setToolTip('Model that predicts the structures. Import… adds a CandyCrunch model file (.pt): weights for the built-in vocabulary, or '
                              'weights saved with their own vocabulary ({"state_dict": ..., "glycans": [...]})')
        self.import_model_button = QPushButton('Import…')
        self.import_model_button.setToolTip('Adds a CandyCrunch model file (.pt) to the models to choose from')
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.import_model_button)
        form.addRow('Glycans', self.inputs['glycan_class'])
        form.addRow('Ion mode', self.mode)
        form.addRow('Max. charge', self.charge)
        form.addRow('Reducing end', self.inputs['modification'])
        form.addRow('Tag mass', self.inputs['mass_tag'])
        form.addRow('Derivatization', self.inputs['sample_prep'])
        form.addRow('Sialic acid linkage', self.sialic)
        form.addRow('Chromatography', self.inputs['lc'])
        form.addRow('Mass analyzer', self.inputs['trap'])
        layout.addLayout(form)
        self.sample_form = form
        # What the first run of the experiment shown set (_detect_run)
        self.detected = _label('', 'hint')
        self.detected.hide()
        layout.addWidget(self.detected)
        self.detections, self.detect_timer = mp.get_context('spawn').Queue(), QTimer(self)
        self.detect_timer.setInterval(200)
        self.detect_timer.timeout.connect(self.receive_detections)
        self.batch = QWidget()
        batch_layout = QVBoxLayout(self.batch)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(_label('Across runs'))
        form = QFormLayout()
        self.inputs['intra_cat_thresh'] = _spin(0.05, 10, 0.25, 2, ' min')
        self.inputs['intra_cat_thresh'].setToolTip('How far (in minutes) the retention time of a structure may drift between runs and still count as the same peak')
        self.inputs['top_n_isomers'] = _spin(1, 50, 1)
        self.inputs['top_n_isomers'].setToolTip('Isomer groups kept per composition in each run')
        self.inputs['n_jobs'] = _spin(1, max(1, os.cpu_count() or 1), 1)
        self.inputs['n_jobs'].setToolTip('Runs processed in parallel, and processes for fragment scoring and supporting peaks within a run; each needs its own memory')
        form.addRow('RT tolerance', self.inputs['intra_cat_thresh'])
        form.addRow('Isomers per composition', self.inputs['top_n_isomers'])
        form.addRow('Parallel processes', self.inputs['n_jobs'])
        batch_layout.addLayout(form)
        layout.addWidget(self.batch)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText('Advanced settings')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        # Outside the advanced settings, so it is in view whatever is expanded; enabled whenever a setting differs from its default
        self.reset = QPushButton('Restore defaults')
        self.reset.setToolTip('Sets every setting of this experiment back to the CandyCrunch default, except the glycan class')
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.advanced_toggle)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.reset)
        layout.addLayout(toggle_row)
        self.advanced = QWidget()
        form = QFormLayout(self.advanced)
        form.setContentsMargins(0, 0, 0, 0)
        self.inputs['rt_min'] = _spin(0, 1000, 1, 2, ' min')
        self.inputs['rt_max'] = _spin(0, 1000, 1, 2, ' min')
        self.inputs['rt_max'].setSpecialValueText('end of run')
        self.inputs['rt_diff'] = _spin(0.05, 10, 0.25, 2, ' min')
        self.inputs['rt_diff'].setToolTip('Maximum retention time difference to a peak apex that is still grouped with that peak')
        self.inputs['ppm_thresh'] = _spin(1, 5000, 10, 0, ' ppm')
        # One tolerance for precursors and fragments: as a flat window of its value at m/z 1600, as the pipeline turns it into Da (MZ_REF)
        self.inputs['ppm_thresh'].valueChanged.connect(lambda v: self.inputs['ppm_thresh'].setToolTip(
            f'One mass tolerance for precursors and fragments. Precursors: the ppm error filter, and a flat {v * MZ_REF / 1e6:.2f} Da window (its value at m/z '
            f'{MZ_REF}) for grouping spectra and matching compositions. Fragments: the same {v * MZ_REF / 1e6:.2f} Da window for CandyCrumbs annotation'))
        self.inputs['pred_thresh'] = _spin(0, 1, 0.005, 3)
        self.inputs['pred_thresh'].setToolTip('Minimum prediction confidence')
        self.inputs['crumbs_thresh'] = _spin(0, 100, 1, 1)
        self.inputs['crumbs_thresh'].setToolTip('A peak is only kept if its fragment score exceeds this (2 lower for files with orbitrap or timsTOF survey scans). The fragment score sums, over the '
                                                '15 most intense deisotoped fragment peaks, how many fragments of a structure of the precursor\'s composition each one matches '
                                                '(up to two cleavages): fragment evidence that the peak is that glycan, independent of the model')
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
        form.addRow('Fragment score above', self.inputs['crumbs_thresh'])
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
        form.addRow('Model', model_row)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)
        layout.addStretch(1)
        self.setWidget(body)
        # Opened, the advanced settings scroll into view (they opened below the window's edge, so nothing seemed to happen)
        self.advanced_toggle.toggled.connect(lambda on: (self.advanced.setVisible(on), self.advanced_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow),
                                                         on and QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(self.advanced_toggle.y() - 8))))
        self.inputs['modification'].currentIndexChanged.connect(lambda: self.sample_form.setRowVisible(self.inputs['mass_tag'], self.inputs['modification'].currentData() == 'custom'))
        self.inputs['experimental'].toggled.connect(lambda on: (self.inputs['taxonomy_level'].setEnabled(on), self.inputs['taxonomy_filter'].setEnabled(on)))
        self.reset.clicked.connect(lambda: self.set_values(DEFAULTS))
        self.import_model_button.clicked.connect(self.import_model)
        self.model_checked.connect(self.model_imported)
        self.remove.clicked.connect(self.remove_selected)
        self.experiment_button.clicked.connect(lambda: self.move_runs([item for item in self.files.selectedItems() if item.parent() is not None],
                                                                      self.add_experiment(self.values()), rename = True))
        # Only while the run list has focus, else Delete in any other widget would remove the selected runs
        QShortcut(QKeySequence.Delete, self.files, self.remove_selected).setContext(Qt.WidgetShortcut)
        self.files.currentItemChanged.connect(lambda: self.show_experiment(self.current_experiment()))
        self.files.itemChanged.connect(self.rename_experiment)
        self.files.customContextMenuRequested.connect(self.files_menu)
        self.files_changed.connect(lambda: self.show_experiment(self.current_experiment()))
        self.files_changed.connect(lambda: self.files.setFixedHeight(min(240, max(110, 6 + max(self.files.sizeHintForRow(0), self.files.fontMetrics().height() + 4) * sum(
            self.files.topLevelItem(i).childCount() + 1 for i in range(self.files.topLevelItemCount()))))))
        for widget in list(self.inputs.values()) + list(self.filter_boxes.values()) + [self.mode, self.charge, self.sialic, self.model]:
            getattr(widget, {QComboBox: 'currentIndexChanged', QCheckBox: 'toggled', QLineEdit: 'textChanged'}.get(type(widget), 'valueChanged')).connect(
                lambda *_: self.reset.setEnabled(any(v != DEFAULTS.get(k, v) for k, v in self.values().items())))
        self.set_values(DEFAULTS)
        self.batch.setVisible(False)

    def current_experiment(self):
        """The experiment of the current item, else the last one"""
        item = self.files.currentItem() or self.files.topLevelItem(self.files.topLevelItemCount() - 1)
        return item if item is None or item.parent() is None else item.parent()

    def add_experiment(self, settings, name = None):
        names = {self.files.topLevelItem(i).text(0) for i in range(self.files.topLevelItemCount())}
        item = QTreeWidgetItem([name or next(f'Experiment {n}' for n in itertools.count(1) if f'Experiment {n}' not in names)])
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
        self.detected.setText(getattr(item, 'note', ''))
        self.detected.setVisible(bool(self.detected.text()))

    def rename_experiment(self, item):
        if item.parent() is not None:
            return
        others = {self.files.topLevelItem(i).text(0) for i in range(self.files.topLevelItemCount()) if self.files.topLevelItem(i) is not item}
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
            self.add_paths(experiment['files'], detect = False)
        self.files.setCurrentItem(self.files.topLevelItem(0))

    def add_paths(self, paths, detect = True):
        """Adds runs to the current experiment, or to a new one with the settings shown; detect: whether the first run of an experiment without runs sets
        what its file and name tell about the experiment's settings"""
        existing, target = set(self.paths()), self.current_experiment()
        empty = target is None or not target.childCount()
        for path in map(os.path.normpath, paths):
            # A timsTOF .d folder is one run; any other folder adds its runs
            found = [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.lower().endswith(SPECTRA_SUFFIXES)] if os.path.isdir(path) and not path.lower().endswith('.d') else [path]
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
        path = target.child(0).data(0, Qt.UserRole) if detect and empty and target is not None and target.childCount() else None
        if path is not None and os.path.exists(path):
            # Read in a process of its own (seconds for a large file); a setting changed in the meantime is kept
            target.detecting = (path, self.values() if target is self.shown else dict(target.settings), time.time())
            mp.get_context('spawn').Process(target = _detect_run, args = (path, self.detections), daemon = True, name = 'CandyCrunch file check').start()
            # Until it answers, the note says that the run is being read (it gave no sign before, and a run started meanwhile used the old settings)
            target.note = f'Reading {os.path.basename(path)} for its ion mode, mass analyzer, and charge states…'
            if target is self.shown:
                self.detected.setText(target.note)
                self.detected.show()
            self.detect_timer.start()

    def receive_detections(self):
        """Sets what the first run of an experiment tells (_detect_run) on the experiment's settings the user has not changed since adding it: the ion mode,
        the analyzer, the highest precursor charge of at least 8% of all MS2 spectra (never below the default), and what the run's name names. Instrument
        charge calls on non-glycan ions spread thin over high charges: Bruker ion traps put 3-6 on 1-6% of the MS2 spectra of GSL glycan runs (z 1-2),
        while runs of glycans at z 4 have them on 11-15%"""
        while True:
            try:
                path, found, charges, n_ms2 = self.detections.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            experiment = next((e for e in self.waiting() if e.detecting[0] == path), None)
            if experiment is None:
                continue
            before, current = experiment.detecting[1], self.values() if experiment is self.shown else dict(experiment.settings)
            del experiment.detecting
            mode = found.pop('mode', None)
            top = max([z for z, n in charges.items() if n >= max(3, 0.08 * n_ms2)] or [0])
            if mode or top:
                found['max_charge'] = (max(abs(DEFAULTS['max_charge']), min(top, 8)) if top else abs(current['max_charge'])) * (
                    -1 if (mode or ('negative' if current['max_charge'] < 0 else 'positive')) == 'negative' else 1)
            if found.get('modification') == 'unreduced':
                found.update(modification = 'custom', mass_tag = None)
            changed = {k: v for k, v in found.items() if current.get(k) == before.get(k) and current.get(k) != v}
            if experiment is self.shown:
                self.set_values({**current, **changed})
            else:
                experiment.settings = {**current, **changed}
            told = [f'{"negative" if v < 0 else "positive"} ions up to charge {abs(v)}' if k == 'max_charge' else
                    self.inputs[k].itemText(self.inputs[k].findData('unreduced' if k == 'modification' and v == 'custom' else v)) for k, v in changed.items() if k != 'mass_tag']
            experiment.note = f'Set from {os.path.basename(path)}: {", ".join(told)}' if told else ''
            if experiment is self.shown:
                self.detected.setText(experiment.note)
                self.detected.setVisible(bool(experiment.note))
        # A check whose process died never answers; after 2 min it is given up, so that a run waiting for it (MainWindow.run) starts
        for experiment in self.waiting():
            if time.time() - experiment.detecting[2] > 120:
                del experiment.detecting
                experiment.note = ''
                if experiment is self.shown:
                    self.detected.hide()
        if not self.waiting():
            self.detect_timer.stop()
            self.detections_done.emit()

    def waiting(self):
        """Experiments whose first run is still being read for its settings (add_paths)"""
        return [e for e in map(self.files.topLevelItem, range(self.files.topLevelItemCount())) if hasattr(e, 'detecting')]

    def add_example(self):
        """Adds the bundled example run once, as an experiment of its own with the default settings for its O-glycans"""
        if EXAMPLE_RUN not in self.paths():
            self.files.setCurrentItem(self.add_experiment({'glycan_class': 'O'}, 'Example'))
            self.add_paths([EXAMPLE_RUN])

    def import_model(self):
        """Checks a model file in a process of its own (the window never imports torch, which takes a few seconds), then adds it (model_imported)"""
        path, _ = QFileDialog.getOpenFileName(self, 'Import a CandyCrunch model', '', 'CandyCrunch models (*.pt *.pth);;All files (*)')
        if not path:
            return
        self.import_model_button.setEnabled(False)
        self.import_model_button.setText('Checking…')
        self.checker = ProcessPoolExecutor(1, mp_context = mp.get_context('spawn'))
        self.checker.submit(_check_model, path).add_done_callback(
            lambda future: self.model_checked.emit(path, None if future.exception() else future.result(), str(future.exception() or '')))

    def model_imported(self, path, structures, error):
        """Copies a checked model file into the app's model folder (under a new name if one is taken) and makes it the shown experiment's model"""
        self.checker.shutdown(wait = False)
        self.import_model_button.setEnabled(True)
        self.import_model_button.setText('Import…')
        if error:
            QMessageBox.warning(self, 'CandyCrunch', f'{os.path.basename(path)} cannot be used: {error}')
            return
        os.makedirs(self.model_dir, exist_ok = True)
        stem, n = os.path.splitext(os.path.basename(path))[0], 1
        target = os.path.join(self.model_dir, f'{stem}.pt')
        while os.path.exists(target):
            n += 1
            target = os.path.join(self.model_dir, f'{stem}_{n}.pt')
        shutil.copyfile(path, target)
        self.model.addItem(os.path.splitext(os.path.basename(target))[0], target)
        self.model.setItemData(self.model.count() - 1, f'{structures} structures; {target}', Qt.ToolTipRole)
        _set_combo(self.model, target)
        # The model choice is the last advanced setting: they open, and the panel scrolls down to it, so the imported model is seen to be chosen
        # (after the scroll to the top of the advanced settings that opening them queues)
        self.advanced_toggle.setChecked(True)
        QTimer.singleShot(0, lambda: self.ensureWidgetVisible(self.model))

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
        return [(e.text(0), [e.child(j).data(0, Qt.UserRole) for j in range(e.childCount())], e.settings) for e in items if e.childCount()]

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
        values['modification'], values['mass_tag'] = ('custom', None) if values['modification'] == 'unreduced' else (values['modification'], values['mass_tag'] if values['modification'] == 'custom' else None)
        values['sialic_derivatization'] = self.sialic.currentData()
        values['model_file'] = self.model.currentData()
        # Classes measured together go to wrap_inference as a list
        if '+' in values['glycan_class']:
            values['glycan_class'] = values['glycan_class'].split('+')
        return values

    def set_values(self, values):
        # A list of classes the combo does not offer (from a session, or settings made outside the app) becomes a choice of its own; a list of one class is that class
        choice = values.get('glycan_class')
        if choice is not None:
            choice = choice if isinstance(choice, str) else '+'.join(sorted(set(choice), key = CLASS_ORDER.index))
            values = {**values, 'glycan_class': choice}
            if self.inputs['glycan_class'].findData(choice) < 0:
                names = {value: label for label, value in GLYCAN_CLASSES}
                self.inputs['glycan_class'].addItem(' and '.join(names[c] for c in choice.split('+')), choice)
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
        # A custom label without mass (as earlier versions saved a free reducing end) is shown as unreduced
        if values.get('modification') == 'custom' and not values.get('mass_tag'):
            _set_combo(self.inputs['modification'], 'unreduced')
        self.sample_form.setRowVisible(self.inputs['mass_tag'], self.inputs['modification'].currentData() == 'custom')
        self.sialic.setCurrentIndex(max(self.sialic.findData(values.get('sialic_derivatization')), 0) if values.get('sialic_derivatization') else 0)
        # A model from a session or the last run that is not in the list is added if its file still exists, else the built-in model is shown
        model = values.get('model_file')
        if model and self.model.findData(model) < 0 and os.path.exists(model):
            self.model.addItem(os.path.splitext(os.path.basename(model))[0], model)
        self.model.setCurrentIndex(max(self.model.findData(model), 0) if model else 0)


class ResultsView(QWidget):
    """Result tables with SNFG drawings, a detail pane for the selected glycan peak, its annotated MS2 spectrum, a glycan map, and curation"""
    open_in_crumbs = Signal(object)
    status = Signal(str)
    experiment_shown = Signal(str)
    # What a misprediction report sends about the peak shown (peak_report)
    report_peak = Signal(object)

    def __init__(self, images):
        super().__init__()
        self.images, self.payload, self.frame, self.row, self.source, self.glytoucan = images, None, None, None, None, None
        self.alternative, self.ms3_groups, self.ms3_args, self.support_highlight, self.dirty, self.drawing_of = None, [], None, None, False, None
        top = QHBoxLayout()
        top.setContentsMargins(8, 6, 8, 4)
        # About 16 and 24 characters wide, not as wide as their longest name: run names of 50 characters squeezed the filter box next to them to a few pixels
        # in a 1280 px window; their lists still show every name in full, and the run shown is the tooltip
        self.experiment, self.dataset = QComboBox(), QComboBox()
        for box, length in ((self.experiment, 16), (self.dataset, 24)):
            box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(length)
        self.experiment.setToolTip('Experiments are predicted and harmonized separately')
        self.dataset.currentTextChanged.connect(self.dataset.setToolTip)
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
        # Only while the table has focus, so Ctrl+C in the structure box still copies its text
        QShortcut(QKeySequence.Copy, self.table, self.copy_rows).setContext(Qt.WidgetShortcut)
        # Ctrl+F (Cmd+F) goes to the filter box, its text selected to type over
        QShortcut(QKeySequence.Find, self, lambda: (self.search.setFocus(), self.search.selectAll()))
        # Detail pane: drawing, facts, alternatives, and curation of the selected peak
        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 8)
        self.drawing = QLabel()
        self.drawing.setAlignment(Qt.AlignCenter)
        self.drawing.setMinimumHeight(90)
        # Its width comes from the pane, never from the drawing shown (a wide drawing widened the pane's content beyond its view, which cut it off)
        self.drawing.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
        self.crumbs.setToolTip('Annotate this spectrum with any structure or setting in the CandyCrumbs tab (from the MS3 tab: the MS3 spectrum shown)')
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
        self.info_scroll = QScrollArea()
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setFrameShape(QFrame.NoFrame)
        self.info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.info_scroll.setWidget(info)
        # The curation buttons stay in view below the details, which scroll if they must (inside them, the buttons were out of view at 1440 x 900), and
        # the candidate list takes the room left instead of its default 192 px height
        self.alternatives.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        actions.setContentsMargins(10, 0, 10, 8)
        details = QWidget()
        details.setMinimumWidth(300)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.addWidget(self.info_scroll, 1)
        details_layout.addLayout(actions)
        self.spectrum = SpectrumPanel()
        self.spectrum_file = QComboBox()
        self.spectrum_file.setToolTip('Run whose MS2 spectrum is shown for this feature')
        # It may shrink below its longest run name (the list shows them in full): 50-character names made the window at least 1430 px wide
        self.spectrum_file.setMinimumWidth(80)
        self.spectrum.controls.insertWidget(1, self.spectrum_file)
        self.report_button = QPushButton('Report misprediction…')
        self.report_button.setToolTip('Sends this peak (its MS2 and MS3 spectra, CandyCrunch\'s predictions and scores, and the settings) to the CandyCrunch '
                                      'developers, with what is wrong about it, so that CandyCrunch can learn from it')
        self.report_button.setEnabled(False)
        self.spectrum.controls.addWidget(self.report_button)
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
        bottom.addWidget(details)
        bottom.addWidget(self.tabs)
        bottom.setStretchFactor(1, 1)
        bottom.setSizes([380, 900])
        # The detail drawing is fitted to the pane's width, so it is fitted again when the pane is resized
        bottom.splitterMoved.connect(lambda: self.row is not None and self.update_drawing())
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
        # A filter that hides every peak empties the detail pane, which kept showing the hidden peak (Exclude and Assign acted on it, and cleared the
        # filter); once peaks match again, the first one is shown
        self.search.textChanged.connect(lambda: self.table.selectionModel().hasSelection() or self.select_row(
            self.proxy.mapToSource(self.proxy.index(0, 0)).row() if self.proxy.rowCount() else None))
        self.table.selectionModel().selectionChanged.connect(self.selection_changed)
        self.table.customContextMenuRequested.connect(self.context_menu)
        # A candidate chosen by click or arrow key is annotated, and Assign candidate assigns it; without one chosen, Assign candidate is disabled (it did
        # nothing, and after arrow keys it assigned a candidate other than the one the spectrum showed)
        self.assign.setEnabled(False)
        self.alternatives.currentItemChanged.connect(lambda item: (self.assign.setEnabled(item is not None), item is not None and self.pick_alternative(item)))
        self.alternatives.itemDoubleClicked.connect(lambda item: self.curate(item.data(Qt.UserRole), 'reassigned'))
        self.assign.clicked.connect(lambda: self.alternatives.currentItem() and self.curate(self.alternatives.currentItem().data(Qt.UserRole), 'reassigned'))
        self.custom.clicked.connect(self.enter_structure)
        self.exclude.clicked.connect(self.toggle_excluded)
        self.crumbs.clicked.connect(self.send_to_crumbs)
        self.report_button.clicked.connect(lambda: self.report_peak.emit(self.peak_report()))
        self.spectrum_file.currentIndexChanged.connect(self.show_source)
        self.spectrum.ms3_clicked.connect(self.open_ms3)
        self.spectrum.support_hovered.connect(lambda highlight: (setattr(self, 'support_highlight', highlight), self.row is not None and self.update_drawing()))
        self.ms3_fragment.currentIndexChanged.connect(self.annotate_ms3)
        self.tabs.currentChanged.connect(lambda: self.tabs.currentWidget() is self.ms3_tab and self.annotate_ms3())
        self.tabs.currentChanged.connect(lambda: self.row is not None and self.tabs.currentWidget() in (self.map, self.across) and self.draw_selection())
        self.map_background, self.highlight = None, None
        self.map.canvas.mpl_connect('draw_event', self.map_drawn)
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
                # _top1 keeps the pipeline's structure, which ties a feature to its per-run rows however either of them is curated
                for column, empty in (('curation', ''), ('excluded', False), ('_top1', df.get('top1_pred'))):
                    if column not in df.columns:
                        df[column] = empty
                experiment['tables'][label] = (df, list(spectra) + [None] * (len(df) - len(spectra)))
            if experiment['features'] is not None:
                for column, empty in (('curation', ''), ('excluded', False), ('_top1', experiment['features'].get('top1_pred'))):
                    if column not in experiment['features'].columns:
                        experiment['features'][column] = empty
        self.session, self.dirty = session, False
        self.experiment.blockSignals(True)
        self.experiment.clear()
        self.experiment.addItems([experiment['name'] for experiment in session['experiments']])
        self.experiment.blockSignals(False)
        self.experiment.setVisible(self.experiment.count() > 1)
        self.show_experiment()

    def show_experiment(self):
        """self.payload is the experiment shown: its files, settings, tables, and feature table"""
        payload = self.payload = self.session['experiments'][max(self.experiment.currentIndex(), 0)]
        # Orbitrap MS2 are annotated at 15 ppm, as their fragment errors scale with m/z. In GPST000548 (Q Exactive HF, fragments at -2 ppm median) it labeled
        # as many real fragments (labels minus those of m/z-shifted decoy spectra) as the pipeline's 0.02 Da, which CandyCrumbs applies to the singly charged
        # mass (0.01 m/z at z = 2), and 10% more with the peaks moved to -9 ppm (calibration offsets seen in orbitrap runs), where 0.01 Da or 10 ppm lost 30-50%
        # and 20 ppm added mostly decoy labels. TOF MS2 (fragments 10-40 ppm off) keep 0.05 Da until audited, ion traps the precursor window until audited
        tolerance = {'orbitrap': (15.0, 'ppm'), 'other': (0.05, 'Da')}.get(payload['settings'].get('trap'), (payload['settings'].get('ppm_thresh', DEFAULTS['ppm_thresh']) * MZ_REF / 1e6, 'Da'))
        for panel in (self.spectrum, self.ms3_panel):
            panel.set_tolerance(*tolerance)
        self.dataset.blockSignals(True)
        self.dataset.clear()
        if payload['features'] is not None and not payload['features'].empty:
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
        with IMPORT_LOCK:
            from candycrunch.utils import mz_order
        structures = list(df['top1_pred'])
        compositions = [_composition(c) for c in df['composition']]
        notes_tip = 'Notes from the pipeline, adducts, and your curation'
        columns = [{'title': 'Structure', 'kind': 'structure', 'raw': [s if isinstance(s, str) else None for s in structures],
                    'text': [s if isinstance(s, str) else '' for s in structures], 'image': [s if isinstance(s, str) else c or None for s, c in zip(structures, compositions)]},
                   {'title': 'Composition', 'kind': 'text', 'raw': [re.sub(r'\d+', lambda m: m.group().zfill(4), c) for c in compositions], 'text': compositions},
                   # sorted by m/z, the rows of one ion (a composition and charge, whose m/z differ by noise) go by RT
                   dict(_number_column('m/z', df['m/z'], 4), sort = mz_order(df['m/z'], df['RT'], df['charge'].values, df['composition'].tolist()).tolist()),
                   _number_column('z', df['charge'], 0, 'Precursor charge'), _number_column('RT', df['RT'], 2, 'Retention time (min)')]
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
            notes = list(df['curation'])
        else:
            abundance = 'rel_abundance' if 'rel_abundance' in df.columns else 'num_spectra'
            column = _number_column('Abundance' if abundance == 'rel_abundance' else 'Spectra', df[abundance], 2 if abundance == 'rel_abundance' else 0,
                                    'Relative abundance (%), from MS1 peak areas where the file has MS1 scans, else from precursor intensities')
            top = max([v for v in column['raw'] if v is not None] or [1]) or 1
            evidence = list(df.get('evidence', [None] * len(df)))
            column.update(kind = 'bar', bar = [None if v is None else (v / top, e == 'ms1_only') for v, e in zip(column['raw'], evidence)])
            columns.append(column)
            columns.append(_number_column('Confidence', [next((x[1] for x in p if x[0] == t and len(x) > 1), None) if isinstance(p, (list, tuple)) else None
                                                         for p, t in zip(df['predictions'], df['top1_pred'])], 2, 'CandyCrunch confidence of the structure shown'))
            columns.append(_number_column('Fragment score', df.get('annotation_score', [None] * len(df)), 0, 'CandyCrumbs fragment annotation score'))
            supported = [s if isinstance(s, str) and s else None for s in df.get('supported_by', [None] * len(df))]
            counts = [re.match(r'(\d+)/(\d+)', s) if s else None for s in supported]
            readings = [re.search(r'Neu5Ac linkage reads (a2-\d)(, not as shown)?', s) if s else None for s in supported]
            columns.append({'title': 'Support', 'kind': 'number', 'raw': [int(m[1]) / int(m[2]) if m and int(m[2]) else None for m in counts],
                            'text': [', '.join(([f'{m[1]}/{m[2]}'] if m else []) + ([('reads ' if r[2] else '') + r[1]] if r else [])) for m, r in zip(counts, readings)],
                            'tips': supported, 'tip': 'Residues placed by diagnostic MS2 peaks, of those with another possible placement, and the linkage of a lone Neu5Ac '
                                                      'that [M-H-Neu5Ac]- reads in ion-trap runs ("reads" where it is not the one shown); hover a cell for the peaks'})
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
        sort_title = self.model.columns[sort_column]['title'] if 0 <= sort_column < len(self.model.columns) else None
        self.model.set_table(df, columns)
        for c, column in enumerate(columns):
            self.table.setItemDelegateForColumn(c, self.structure_delegate if column['kind'] == 'structure' else self.bar_delegate if column['kind'] == 'bar' else None)
        # A refresh after curation keeps the columns as they are, also as resized by hand; measuring them again took ~90% of its time
        if keep_row is None:
            self.table.resizeColumnsToContents()
            self.table.setColumnWidth(0, 250)
            for c, column in enumerate(columns):
                if column['kind'] == 'bar':
                    self.table.setColumnWidth(c, max(90, self.table.columnWidth(c)))
                elif column['title'] == 'Composition':
                    self.table.setColumnWidth(c, min(self.table.columnWidth(c), 210))
            # Another run or the feature table keeps the sort chosen before if it has that column (every switch sorted by m/z again), else by m/z
            titles = [c['title'] for c in columns]
            self.table.sortByColumn(titles.index(sort_title), sort_order) if sort_title in titles else self.table.sortByColumn(2, Qt.AscendingOrder)
        else:
            self.table.sortByColumn(sort_column, sort_order)
        self.filter(self.search.text())
        self.draw_map()
        self.select_row(keep_row if keep_row is not None else (self.proxy.mapToSource(self.proxy.index(0, 0)).row() if self.proxy.rowCount() else None))

    def filter(self, text):
        self.proxy.set_needle(text)
        shown, total = self.proxy.rowCount(), self.model.rowCount()
        excluded = sum(e for e, _ in self.model.marks)
        # Excluded peaks are left out of every export, so how many there are stays in view
        self.counter.setText((f'{total} glycan peaks' if shown == total else f'{shown} of {total} glycan peaks') + (f', {excluded} excluded' if excluded else ''))

    def select_row(self, row):
        if row is None:
            # An empty table (a run without glycan peaks) leaves nothing of the previous table's peak in the detail pane
            self.row, self.source, self.alternative, self.ms3_groups = None, None, None, []
            for button in (self.report_button, self.custom, self.exclude, self.crumbs):
                button.setEnabled(False)
            for widget in (self.structure, self.composition, self.facts, self.drawing, self.alternatives, self.ms3_fragment):
                widget.clear()
            self.spectrum_file.setVisible(False)
            if self.tabs.currentWidget() is self.ms3_tab:
                self.tabs.setCurrentIndex(0)
            self.tabs.setTabVisible(self.tabs.indexOf(self.ms3_tab), False)
            # A filter that matches nothing is told apart from a run without glycan peaks
            self.spectrum.show_message('No glycan peaks match the filter' if self.model.rowCount() else 'No glycan peaks to show')
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
        self.row, self.alternative, self.support_highlight = row, None, None
        df = self.frame
        if self.dataset.currentData() == FEATURES:
            # The feature table keeps no link to the per-run rows, so the row of each run is the one with the same structure closest in m/z and RT
            self.spectrum_file.blockSignals(True)
            self.spectrum_file.clear()
            best = None
            for run, (table, _) in self.payload['tables'].items():
                evidence = df[f'evidence_{run}'].iat[row] if f'evidence_{run}' in df.columns else None
                if not isinstance(evidence, str) or evidence == 'ms1_only' or table.empty:
                    continue
                same = (table['_top1'] == df['_top1'].iat[row]) if isinstance(df['_top1'].iat[row], str) else table['_top1'].isna()
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
                      ('Confidence', _fmt(next((p[1] for p in preds if p[0] == structure and len(p) > 1), None), 3) if isinstance(preds, (list, tuple)) else ''),
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
        structure = self.drawing_of = structure if isinstance(structure, str) else _composition(self.frame['composition'].iat[self.row]) or None
        # While a supporting peak is hovered, its fragment and cleaved bond are highlighted; the plain drawing stays until that version is rendered (or
        # if the installed glycowork cannot draw it)
        image = self.images.get(structure, compact = False, urgent = True, highlight = self.support_highlight[1:]) if self.support_highlight and self.support_highlight[0] == structure else None
        image = image if image is not None and not image.isNull() else self.images.get(structure, compact = False, urgent = True)
        if image is None:
            self.drawing.setText('Drawing…')
        elif image.isNull():
            self.drawing.clear()
        else:
            # At most its natural size, the pane's width and 150 px high, and the label at least as high as the drawing: the label kept 90 px, so it cut off
            # the top and bottom of every drawing taller than that (most N-glycans)
            ratio = self.devicePixelRatioF()
            box = QSize(min(int(image.width() * 0.3), max(120, self.info_scroll.viewport().width() - 30)), 150)
            pixmap = QPixmap.fromImage(image.scaled(box * ratio, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            pixmap.setDevicePixelRatio(ratio)
            self.drawing.setMinimumHeight(max(90, round(pixmap.height() / ratio)))
            self.drawing.setPixmap(pixmap)

    def show_source(self):
        """Candidates and spectrum of the selected per-run row (for a feature: of the run picked in the spectrum combobox)"""
        if self.row is None:
            return
        if self.dataset.currentData() == FEATURES:
            self.source = self.spectrum_file.currentData()
        else:
            self.source = (self.dataset.currentData(), self.row)
        # Without a peak (an empty table) the curation buttons are disabled, and without an MS2 spectrum (a feature with MS1 signal only) Open in CandyCrumbs
        # too, as they did nothing when clicked
        self.custom.setEnabled(True)
        self.exclude.setEnabled(True)
        for button in (self.report_button, self.crumbs):
            button.setEnabled(self.source is not None)
        self.alternatives.clear()
        if self.source is None:
            self.spectrum.show_message('This feature has MS2 evidence in no run, only MS1 signal')
            return
        table = self.payload['tables'][self.source[0]][0]
        preds = table['predictions'].iat[self.source[1]]
        for structure, score in (preds if isinstance(preds, (list, tuple)) else []):
            item = QListWidgetItem(f'{score:.3f}')
            item.setData(Qt.UserRole, structure)
            item.setToolTip(structure)
            self.alternatives.addItem(item)
        self.update_alternative_icons()
        self.annotate()

    def peak_report(self):
        """What a misprediction report sends about the peak shown: its row (for a feature also the feature's, without the other runs' values), its MS2 and
        MS3 spectra, the structure the pipeline gave it and the one shown, and the experiment's settings; not the run's name"""
        table, spectra = self.payload['tables'][self.source[0]]
        r, run = self.source[1], next((f for f in self.payload['files'] if os.path.splitext(os.path.basename(os.path.normpath(f)))[0] == self.source[0]), None)
        report = {'peak': {k: v for k, v in table.iloc[r].items() if k not in ('ms3', 'excluded', '_top1')}, 'predicted': table['_top1'].iat[r],
                  'shown': self.alternative or table['top1_pred'].iat[r], 'ms2': spectra[r], 'ms3': table['ms3'].iat[r] if 'ms3' in table.columns else None,
                  'run_format': os.path.splitext(run)[1].lower() if run is not None else None,
                  'settings': {k: sorted(v) if isinstance(v, set) else v for k, v in self.payload['settings'].items()}, 'tolerance': self.spectrum.tolerance.value(),
                  'tolerance_unit': self.spectrum.unit.currentData()}
        if self.dataset.currentData() == FEATURES:
            report['feature'] = {k: v for k, v in self.frame.iloc[self.row].items() if k not in self.payload['tables'] and not k.startswith('evidence_') and k not in ('excluded', '_top1')}
        # What the report can add if the user agrees, kept out of the report itself as the path names the run: the run's file (to cut out the peak's
        # surroundings) and its table of peaks without spectra
        charge = table['charge'].iat[r]
        self.report_data = {'path': run, 'mz': float(table['m/z'].iat[r]), 'rt': float(table['RT'].iat[r]),
                            'charge': int(charge) if pd.notna(charge) else -1 if self.payload['settings'].get('max_charge', -3) < 0 else 1,
                            'table': table.drop(columns = [c for c in ('ms3', 'excluded', '_top1') if c in table.columns]).to_csv(index = False)}
        return report

    def update_alternative_icons(self):
        for i in range(self.alternatives.count()):
            item = self.alternatives.item(i)
            image = self.images.get(item.data(Qt.UserRole), urgent = True)
            if image is not None and not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            elif image is not None:
                item.setText(f'{item.text().split()[0]}  {item.data(Qt.UserRole)}')

    def spectrum_args(self, r, structure = None):
        """Structure (structure, else the row's top1_pred, else its composition), MS3 groups, and SpectrumPanel.annotate arguments (None without a spectrum
        or anything to annotate it with) of row r of the per-run table shown in the spectrum"""
        table, spectra = self.payload['tables'][self.source[0]]
        peaks, settings = spectra[r], self.payload['settings']
        structure = structure or table['top1_pred'].iat[r]
        if not isinstance(structure, str):
            structure = _composition(table['composition'].iat[r]) or None
        # MS3 spectra pooled per isolated MS2 fragment (rounded to 1 m/z, as the pipeline scores them), merging peaks within half the mass tolerance
        ms3, groups, ms3_groups = table['ms3'].iat[r] if 'ms3' in table.columns else None, {}, []
        for p, spectrum in ms3 if isinstance(ms3, list) and isinstance(peaks, dict) and peaks and structure else []:
            groups.setdefault(round(p), []).append((p, spectrum))
        for _, members in sorted(groups.items()):
            pairs, pooled, group = sorted((m, i) for _, spectrum in members for m, i in spectrum.items()), {}, []
            for m, i in pairs + [(np.inf, 0)]:
                if group and m - group[0][0] > settings['ppm_thresh'] * MZ_REF / 2e6:
                    ms, ints = zip(*group)
                    pooled[float(np.average(ms, weights = ints)) if sum(ints) else float(np.mean(ms))] = float(sum(ints))
                    group = []
                group.append((m, i))
            if pooled:
                ms3_groups.append((float(np.median([p for p, _ in members])), pooled, len(members)))
        if not isinstance(peaks, dict) or not peaks or structure is None:
            return structure, ms3_groups, None
        preds = table['predictions'].iat[r]
        return structure, ms3_groups, ((structure, list(peaks.keys()), list(peaks.values()), int(table['charge'].iat[r]), _mass_tag(settings), settings['sample_prep']),
                                       dict(ms3 = [(p, n) for p, _, n in ms3_groups], candidates = [p[0] for p in preds if p[0] != structure] if isinstance(preds, (list, tuple)) else [],
                                            sialic_derivatization = settings.get('sialic_derivatization'), ion_trap = settings.get('trap') in ('linear', 'amazon'),
                                            metal_cations = 'adduct' not in table.columns or isinstance(table['adduct'].iat[r], str)))

    def annotate(self):
        table, spectra = self.payload['tables'][self.source[0]]
        r = self.source[1]
        structure, self.ms3_groups, args = self.spectrum_args(r, self.alternative)
        self.ms3_args = (structure, int(table['charge'].iat[r]))
        self.ms3_fragment.blockSignals(True)
        self.ms3_fragment.clear()
        for p, _, n in self.ms3_groups:
            self.ms3_fragment.addItem(f'Fragment m/z {p:.2f}, {n} spectr{"um" if n == 1 else "a"}')
        self.ms3_fragment.blockSignals(False)
        if not self.ms3_groups and self.tabs.currentWidget() is self.ms3_tab:
            self.tabs.setCurrentIndex(0)
        self.tabs.setTabVisible(self.tabs.indexOf(self.ms3_tab), bool(self.ms3_groups))
        if not isinstance(spectra[r], dict) or not spectra[r]:
            self.spectrum.show_message('No MS2 spectrum: this peak was only found by its MS1 signal')
        elif structure is None:
            self.spectrum.show_message('Neither a structure nor a composition to annotate this spectrum with')
        else:
            # The rows below and above in the table (as sorted and filtered) are drawn ahead, so stepping through it shows each spectrum at once
            ahead, index = [], self.proxy.mapFromSource(self.model.index(r, 0))
            for step in ((1, -1, 2) if self.dataset.currentData() != FEATURES and index.isValid() else ()):
                if 0 <= index.row() + step < self.proxy.rowCount():
                    ahead.append(self.spectrum_args(self.proxy.mapToSource(self.proxy.index(index.row() + step, 0)).row())[2])
            self.spectrum.annotate(*args[0], prefetch = [a for a in ahead if a is not None], **args[1])
            if self.tabs.currentWidget() is self.ms3_tab:
                self.annotate_ms3()

    def open_ms3(self, index):
        # One click picks both the marker and the invisible stick over the peak; the second pick would annotate the same MS3 spectra again
        if self.tabs.currentWidget() is self.ms3_tab and self.ms3_fragment.currentIndex() == index:
            return
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
            # As the pipeline scores MS3: without X cross-rings, which explain random MS3 peaks as often as real ones (MS3 cleavage counting)
            self.ms3_panel.annotate(self.ms3_args[0], list(peaks.keys()), list(peaks.values()), self.ms3_args[1], _mass_tag(settings),
                                    settings['sample_prep'], ms3_precursor = p, disable_X_cross_rings = True,
                                    sialic_derivatization = settings.get('sialic_derivatization'))

    def pick_alternative(self, item):
        self.alternative, self.support_highlight = item.data(Qt.UserRole), None
        self.update_drawing()
        self.annotate()

    def image_ready(self, key):
        if self.row is None or self.frame is None:
            return
        if key[1] is False and key[0] == self.drawing_of:
            self.update_drawing()
        if key[1] is True and any(self.alternatives.item(i).data(Qt.UserRole) == key[0] for i in range(self.alternatives.count())):
            self.update_alternative_icons()

    def draw_map(self):
        df, figure = self.frame, self.map.figure
        figure.clear()
        self.map_background = None
        ax = figure.add_axes([0.08, 0.14, 0.78, 0.8])
        if self.dataset.currentData() == FEATURES:
            color, color_label = df['n_files_ms2'].astype(float).values, 'Runs with MS2'
            size = df[[run for run in self.payload['tables'] if run in df.columns]].fillna(0).mean(axis = 1).values
        else:
            color, color_label = np.array([next((x[1] for x in p if x[0] == t and len(x) > 1), np.nan) if isinstance(p, (list, tuple)) else np.nan
                                           for p, t in zip(df['predictions'], df['top1_pred'])], dtype = float), 'Confidence'
            size = (df['rel_abundance'] if 'rel_abundance' in df.columns else df['num_spectra']).astype(float).values
        size = 12 + 380 * np.sqrt(np.nan_to_num(size) / (np.nanmax(size, initial = 0) or 1))
        known = ~np.isnan(color)
        ax.scatter(df['RT'].values[~known], df['m/z'].values[~known], s = size[~known], c = '#cccccc', edgecolors = 'white', linewidths = 0.5)
        points = ax.scatter(df['RT'].values, df['m/z'].values, s = size, c = np.where(known, color, np.nan), cmap = CONFIDENCE_MAP,
                            edgecolors = 'white', linewidths = 0.5, alpha = 0.9, picker = 4)
        if known.any():
            colorbar = figure.colorbar(points, cax = figure.add_axes([0.89, 0.14, 0.02, 0.8]))
            colorbar.set_label(color_label, fontsize = 8)
            colorbar.ax.tick_params(labelsize = 7)
            if self.dataset.currentData() == FEATURES:
                from matplotlib.ticker import MaxNLocator
                # A number of runs: whole numbers only (it was labeled 1.2, 1.4, ..., or 0.925 to 1.075 when every feature had MS2 in one run)
                colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer = True, min_n_ticks = 1))
        # The selection marker is animated: a full draw leaves it out (and keeps that background), so selecting another peak only blits the marker
        self.highlight = ax.scatter([], [], s = 0, facecolors = 'none', edgecolors = INK, linewidths = 1.4, animated = True)
        ax.set_xlabel('Retention time (min)', fontsize = 8)
        ax.set_ylabel('m/z', fontsize = 8)
        ax.tick_params(labelsize = 7)
        ax.spines[['top', 'right']].set_visible(False)
        self.map.canvas.draw_idle()

    def map_drawn(self, event = None):
        """After every full draw of the glycan map (or with event None, when only the selection moved): the selection marker blitted onto the map"""
        if self.highlight is None:
            return
        if event is not None:
            self.map_background = self.map.canvas.copy_from_bbox(self.map.figure.bbox)
        elif self.map_background is None:
            return
        else:
            self.map.canvas.restore_region(self.map_background)
        self.map.figure.draw_artist(self.highlight)
        if event is None:
            self.map.canvas.blit(self.map.figure.bbox)

    def draw_selection(self):
        df, row = self.frame, self.row
        self.highlight.set_offsets([[df['RT'].iat[row], df['m/z'].iat[row]]])
        self.highlight.set_sizes([140])
        # Hidden charts are not drawn (a full draw of the map took 0.2-0.5 s on every row change); showing their tab draws them
        if self.map.isVisible():
            self.map_drawn()
        if self.dataset.currentData() != FEATURES or not self.across.isVisible():
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
            with open(os.path.join(os.path.dirname(__file__), 'glytoucan_mapping.pkl'), 'rb') as file:
                self.glytoucan = pickle.load(file)
        if structure not in self.glytoucan:
            with IMPORT_LOCK:
                from glycowork.motif.processing import glytoucan_to_glycan
            # A structure outside CandyCrunch's vocabulary (entered by hand) gets the ID glycowork's database has for it, or for its alditol
            ids = glytoucan_to_glycan([structure, f'{structure}-ol'], revert = True, verbose = False)
            self.glytoucan[structure] = next((i for g, i in zip((structure, f'{structure}-ol'), ids) if i != g), '')
        # A feature's structure also goes to the per-run rows it was matched to, so their tables, exports, and the spectrum shown agree with it
        targets = [(self.frame, self.row)] + ([(self.payload['tables'][run][0], r) for run, r in (self.spectrum_file.itemData(i) for i in range(self.spectrum_file.count()))]
                                             if self.dataset.currentData() == FEATURES else [])
        for df, row in targets:
            df.loc[df.index[row], ['top1_pred', 'curation']] = [structure, how]
            # The pipeline's supporting peaks were found for the old structure; the spectrum panel shows those of the new one
            if 'supported_by' in df.columns:
                df.loc[df.index[row], 'supported_by'] = np.nan
            if 'GlyTouCan_ID' in df.columns:
                df.loc[df.index[row], 'GlyTouCan_ID'] = self.glytoucan.get(structure, '')
        self.dirty = True
        self.status.emit(f'Assigned {structure}')
        self.refresh()

    def enter_structure(self):
        if self.row is None:
            return
        current = self.frame['top1_pred'].iat[self.row]
        text, ok = QInputDialog.getText(self, 'Enter structure', 'Structure in IUPAC-condensed nomenclature:', text = current if isinstance(current, str) else '')
        if not ok or not text.strip():
            return
        with IMPORT_LOCK:
            from glycowork.motif.processing import canonicalize_iupac
            from glycowork.motif.tokenization import glycan_to_composition
        try:
            structure = canonicalize_iupac(text.strip())
            # Like the pipeline's structures, reduced glycans go without -ol, since the experiment's reducing end already says so
            structure = structure[:-3] if structure.endswith('-ol') and self.payload['settings'].get('modification') == 'reduced' else structure
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
            self.dirty = True
            self.refresh()

    def context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        # The actions work on self.row, so the right-clicked row is selected first (they used to act on the previously selected row)
        self.select_row(self.proxy.mapToSource(index).row())
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
        # From the MS3 tab, the MS3 spectra shown go, with the MS2 fragment they were isolated from
        ms3 = self.ms3_groups[self.ms3_fragment.currentIndex()] if self.tabs.currentWidget() is self.ms3_tab and 0 <= self.ms3_fragment.currentIndex() < len(self.ms3_groups) else None
        self.open_in_crumbs.emit({'structure': structure if isinstance(structure, str) else _composition(table['composition'].iat[r]),
                                  'peaks': ms3[1] if ms3 else spectra[r] or {}, 'charge': int(table['charge'].iat[r]), 'settings': self.payload['settings'],
                                  'ms3_precursor': ms3[0] if ms3 else None})

    def export_frame(self, df):
        """What gets written: curated rows as edited, excluded rows dropped, the curation column only if anything was curated"""
        out = df[~df['excluded'].astype(bool)].drop(columns = ['excluded', 'ms3', '_top1'], errors = 'ignore')
        return out.drop(columns = ['curation']) if not out['curation'].astype(bool).any() else out

    def write(self, df, path, kind):
        if kind == 'snfg':
            with IMPORT_LOCK:
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
            runs = [(run, table) for run, (table, _) in self.payload['tables'].items() if not table.empty]
            for run, table in runs:
                self.write(self.export_frame(table), f'{stem}_{run}{suffix}', suffix)
            self.status.emit(f'Exported {len(runs) + 1} tables next to {os.path.basename(path)}')
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
        # Up to 8 as in the Predict tab; glycopeptides often carry 5+ and more
        self.charge = _combo([(f'{z:+d}', z) for z in [*range(-1, -9, -1), *range(1, 9)]])
        self.modification = _combo(REDUCING_ENDS)
        self.mass_tag = _spin(-1000, 3000, 1, 4, ' Da')
        self.sample_prep = _combo(SAMPLE_PREPS)
        self.sialic = _combo(SIALIC_DERIVATIZATIONS)
        self.sialic.setToolTip(SIALIC_TIP)
        self.cleavages = _spin(1, 4, 1)
        self.cleavages.setValue(3)
        self.cleavages.setToolTip('Maximum number of concurrent cleavages per fragment')
        self.fragmentation = _combo(FRAGMENTATIONS)
        self.fragmentation.setToolTip('Restricts peptide backbone ion types for glycopeptides (not EED/EDD/NETD/UVPD); every method but CID/HCD also '
                                      'annotates free-glycan fragments one or two hydrogen atoms off (odd-electron, hydrogen-shifted ions)')
        self.ms3_precursor = _spin(0, 5000, 1, 4, ' m/z')
        self.ms3_precursor.setSpecialValueText('None (MS2 spectrum)')
        self.ms3_precursor.setToolTip('For an MS3 spectrum: m/z of the MS2 fragment that was isolated, so its peaks are only annotated as fragments of that '
                                      'fragment (glycan structures only)')
        form.addRow('Precursor charge', self.charge)
        form.addRow('Reducing end', self.modification)
        form.addRow('Tag mass', self.mass_tag)
        form.addRow('Derivatization', self.sample_prep)
        form.addRow('Sialic acid linkage', self.sialic)
        form.addRow('Max. cleavages', self.cleavages)
        form.addRow('Fragmentation', self.fragmentation)
        form.addRow('MS3 precursor', self.ms3_precursor)
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
        form.setRowVisible(self.mass_tag, False)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self.update_preview)
        self.structure.textChanged.connect(self.debounce.start)
        self.structure.returnPressed.connect(self.annotate)
        self.modification.currentIndexChanged.connect(lambda: form.setRowVisible(self.mass_tag, self.modification.currentData() == 'custom'))
        self.go.clicked.connect(self.annotate)
        # Ctrl+Enter annotates from anywhere in the tab, also from the peak list, where Enter starts a new line
        QShortcut(QKeySequence('Ctrl+Return'), self, self.annotate).setContext(Qt.WidgetWithChildrenShortcut)
        self.go.setToolTip(f'Annotates the peaks with the structure ({QKeySequence("Ctrl+Return").toString(QKeySequence.NativeText)})')
        images.ready.connect(lambda key: key[:2] == (self.structure.text().strip(), False) and self.update_preview())
        self.support_highlight = None
        self.spectrum.support_hovered.connect(lambda highlight: (setattr(self, 'support_highlight', highlight), self.update_preview()))

    def update_preview(self):
        text = self.structure.text().strip()
        # While a supporting peak is hovered, its fragment and cleaved bond are highlighted; the plain drawing stays until that version is rendered (or
        # if the installed glycowork cannot draw it)
        image = self.images.get(text, compact = False, urgent = True, highlight = self.support_highlight[1:]) if self.support_highlight and self.support_highlight[0] == text else None
        image = image if image is not None and not image.isNull() else self.images.get(text, compact = False, urgent = True) if text and '*' not in text else QImage()
        if image is None:
            self.preview.setText('Drawing…')
        elif not text or '*' in text:
            self.preview.clear()
        elif image.isNull():
            self.preview.setText('GlycoDraw cannot draw this; CandyCrumbs may still read it')
        else:
            # As in the results' detail drawing: at most its natural size, the panel's width and 150 px high, and the label as high as the drawing
            ratio = self.devicePixelRatioF()
            pixmap = QPixmap.fromImage(image.scaled(QSize(min(int(image.width() * 0.3), self.preview.width() - 10), 150) * ratio, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            pixmap.setDevicePixelRatio(ratio)
            self.preview.setMinimumHeight(max(70, round(pixmap.height() / ratio)))
            self.preview.setPixmap(pixmap)

    def load(self, request):
        settings = request['settings']
        self.structure.setText(request['structure'] or '')
        self.peaks.setPlainText('\n'.join(f'{mz:.4f}\t{intensity:.1f}' for mz, intensity in sorted(request['peaks'].items())))
        _set_combo(self.charge, request['charge'])
        _set_combo(self.modification, 'unreduced' if settings.get('modification') == 'custom' and not settings.get('mass_tag') else settings.get('modification'))
        self.mass_tag.setValue(settings.get('mass_tag') or 0)
        _set_combo(self.sample_prep, settings.get('sample_prep'))
        self.sialic.setCurrentIndex(max(self.sialic.findData(settings.get('sialic_derivatization')), 0) if settings.get('sialic_derivatization') else 0)
        self.ms3_precursor.setValue(request.get('ms3_precursor') or 0)
        self.annotate()

    def annotate(self):
        # A line split by tabs or semicolons without any '.' has decimal commas (Excel or CSV in comma-decimal locales): 1234,5678<Tab>100; lines that start
        # with text are no peaks (headers, and the TITLE=, PEPMASS=, CHARGE= lines of a pasted mgf spectrum, whose numbers became peaks)
        peaks = [[float(x) for x in re.findall(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line.replace(',', '.') if '.' not in line and re.search(r'[\t;]', line) else line)[:2]]
                 for line in self.peaks.toPlainText().splitlines() if re.match(r'\s*\.?\d', line)]
        peaks = [p for p in peaks if p and p[0] > 0]
        if not self.structure.text().strip() or not peaks:
            self.spectrum.show_message('Enter a structure and at least one peak')
            return
        settings = {'modification': self.modification.currentData(), 'mass_tag': self.mass_tag.value() if self.modification.currentData() == 'custom' else None}
        kwargs = {'max_cleavages': self.cleavages.value()}
        if self.fragmentation.currentData():
            kwargs['fragmentation_method'] = self.fragmentation.currentData()
        if self.sialic.currentData():
            kwargs['sialic_derivatization'] = self.sialic.currentData()
        if self.ms3_precursor.value():
            # X cross-rings within the isolated fragment explain random MS3 peaks as often as real ones
            kwargs.update(ms3_precursor = self.ms3_precursor.value(), disable_X_cross_rings = True)
        self.spectrum.annotate(self.structure.text().strip(), [p[0] for p in peaks], [p[1] if len(p) > 1 else 100.0 for p in peaks],
                               self.charge.currentData(), _mass_tag(settings), self.sample_prep.currentData(), **kwargs)


class ReportDialog(QDialog):
    """The report form, for bug reports, given an error, crash reports, and given a peak (ResultsView.peak_report), misprediction reports: what happened in the
    user's words, an address for a reply, and everything that would be sent, readable before sending"""

    def __init__(self, parent, event, files, summary = None, peak = None):
        super().__init__(parent)
        self.setWindowTitle('Send a crash report' if summary else 'Report a misprediction' if peak else 'Report a bug')
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.addWidget(_label('CandyCrunch ran into an error' if summary else 'What is wrong with the prediction for this peak?' if peak else
                                'Report a bug or suggest an improvement'))
        if summary:
            error = _label(summary, 'hint')
            error.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(error)
        if peak:
            self.problem, self.correct = _combo(MISPREDICTIONS), QLineEdit()
            # A structure the user already assigned to this peak is the likely answer
            self.correct.setText(peak['peak'].get('top1_pred') if peak['peak'].get('curation') and isinstance(peak['peak'].get('top1_pred'), str) else '')
            self.correct.setPlaceholderText('The right structure or composition, if you know it (e.g., Neu5Ac(a2-3)Gal(b1-3)GalNAc or Hex1HexNAc1NeuAc1)')
            layout.addWidget(self.problem)
            layout.addWidget(self.correct)
            self.include_data = QCheckBox('Include the full peak')
            self.include_data.setChecked(True)
            self.include_data.setToolTip('Sending the full information for this peak. This lets the developers rerun '
                                         'CandyCrunch on exactly this peak; without it, they only have the pooled MS2 spectrum')
            layout.addWidget(self.include_data)
        self.text, self.email = QPlainTextEdit(), QLineEdit()
        self.text.setPlaceholderText('What were you doing when it happened? (optional)' if summary else
                                     'Anything else that helps, e.g., which fragments point to the right structure (optional)' if peak else
                                     'What went wrong, and what did you expect instead? Or what would you like CandyCrunch to do?')
        self.text.setMaximumHeight(140)
        self.email.setPlaceholderText('Your email address, if you would like a reply (optional)')
        layout.addWidget(self.text)
        layout.addWidget(self.email)
        layout.addWidget(_label('The report holds this peak: its MS2 and MS3 spectra, CandyCrunch\'s predictions and scores for it, and the settings of its '
                                'experiment, plus the log of this session and the versions of CandyCrunch, its packages, and your operating system; not the '
                                'name of the run or any other peak. Paths in your home folder are shortened to ~.' if peak else
                                f'The report holds {"the error and where in CandyCrunch it happened, " if summary else ""}the log of this session, the settings and '
                                'file types of your runs, and the versions of CandyCrunch, its packages, and your operating system, but none of your data. Paths '
                                'in your home folder are shortened to ~.', 'hint'))
        details = QPlainTextEdit(_scrub(json.dumps(event, indent = 1, default = str) + ''.join(f'\n\n{name}:\n{text}' for name, text in files.items() if text)))
        details.setReadOnly(True)
        details.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        details.setMinimumHeight(240)
        details.hide()
        toggle = QToolButton()
        toggle.setText('Show what is sent')
        toggle.setCheckable(True)
        toggle.toggled.connect(lambda shown: (details.setVisible(shown), toggle.setText('Hide what is sent' if shown else 'Show what is sent'), self.fit(320 if shown else 0)))
        layout.addWidget(toggle, 0, Qt.AlignLeft)
        layout.addWidget(details, 1)
        # Only a report Sentry can take can go out without asking
        self.automatic = QCheckBox('Do this automatically from now on')
        self.automatic.setVisible(bool(summary and REPORT_DSN))
        layout.addWidget(self.automatic)
        buttons = QDialogButtonBox()
        send = buttons.addButton('Send report' if summary else 'Send', QDialogButtonBox.AcceptRole)
        buttons.addButton("Don't send" if summary else 'Cancel', QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # A misprediction report needs what is wrong, a bug report a description
        if peak:
            send.setEnabled(False)
            self.problem.currentIndexChanged.connect(lambda: send.setEnabled(self.problem.currentData() is not None))
        elif not summary:
            send.setEnabled(False)
            self.text.textChanged.connect(lambda: send.setEnabled(bool(self.text.toPlainText().strip())))
        layout.addWidget(buttons)
        # Styled first, so that the wrapped hint text is measured in the font it is shown in
        self.ensurePolished()
        self.fit(0, 720)

    def fit(self, extra, width = None):
        """Sizes the dialog to the height its wrapped text needs at its width, plus extra for the details, within the screen. Qt's own first size and
        adjustSize took the layout's minimum height (566 px at the 560 px minimum width), less than the word-wrapped hints need, so Windows made the
        window taller than asked and Qt warned 'QWindowsWindow::setGeometry: Unable to set geometry'"""
        width = width or self.width()
        screen = (self.parentWidget() or self).screen().availableGeometry()
        self.resize(min(width, screen.width()), min(max(self.heightForWidth(width), self.minimumSizeHint().height()) + extra, int(0.9 * screen.height())))


class MainWindow(QMainWindow):
    taxa_ready = Signal(object)
    update_found = Signal(str)
    update_progress = Signal(int)
    update_downloaded = Signal(bool)
    error_caught = Signal(str, object)
    report_done = Signal(bool, object)

    def __init__(self, paths, crashes = (), ended = {}):
        super().__init__()
        self.store = QSettings()
        self.setWindowTitle('CandyCrunch')
        self.setAcceptDrops(True)
        self.images, self.runner, self.started, self.stage_text = GlycanImages(), InferenceRunner(), None, ''
        self.first_report, self.fraction, self.run_pending = None, 0.0, False
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
                                   'and press Run CandyCrunch. No data at hand? <a href="example">Add the example run</a>.', 'hint')
        self.welcome_text.linkActivated.connect(lambda _: self.settings.add_example())
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
        file_menu.addAction('Add example run', lambda: (self.settings.add_example(), self.tabs.setCurrentWidget(self.split)))
        file_menu.addAction('Import model…', self.settings.import_model)
        file_menu.addSeparator()
        # Through the button, which is disabled while a run goes on (or waits for a run's settings check), from any tab to the Predict tab's progress
        file_menu.addAction('Run CandyCrunch', lambda: (self.tabs.setCurrentWidget(self.split), self.run_button.click()), QKeySequence('Ctrl+R'))
        self.run_button.setToolTip(f'Predicts the glycan structures of every run in the list ({QKeySequence("Ctrl+R").toString(QKeySequence.NativeText)})')
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
        help_menu.addAction('Report a bug…', lambda: self.open_report())
        # triggered, not toggled: Don't send with 'Do this automatically' unticks it and must not reset the choice to asking
        self.auto_action = help_menu.addAction('Send crash reports automatically')
        self.auto_action.setCheckable(True)
        self.auto_action.setChecked(self.store.value('crash_reports', '') == 'always')
        self.auto_action.setVisible(bool(REPORT_DSN))
        self.auto_action.triggered.connect(lambda on: self.store.setValue('crash_reports', 'always' if on else ''))
        help_menu.addSeparator()
        help_menu.addAction('About CandyCrunch', self.about)
        self.settings.add_files.clicked.connect(self.choose_files)
        self.settings.add_folder.clicked.connect(self.choose_folder)
        self.run_button.clicked.connect(self.run)
        self.settings.detections_done.connect(self.run_waiting)
        self.cancel_button.clicked.connect(self.cancel)
        self.runner.ready.connect(lambda: self.runner.busy or self.status_text.setText('Ready'))
        self.runner.stage.connect(self.show_stage)
        self.runner.progress.connect(self.show_progress)
        self.runner.log.connect(self.append_log)
        self.runner.finished.connect(self.finished)
        self.runner.failed.connect(self.failed)
        self.results.status.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.results.experiment_shown.connect(self.settings.select_experiment)
        self.results.report_peak.connect(lambda peak: self.open_report(peak = peak))
        self.results.open_in_crumbs.connect(lambda request: (self.tabs.setCurrentWidget(self.crumbs), self.crumbs.load(request)))
        self.settings.inputs['taxonomy_level'].currentTextChanged.connect(self.load_taxa)
        self.taxa_ready.connect(lambda taxa: self.settings.taxa.setModel(QStringListModel(taxa, self.settings.taxa)))
        self.images.ready.connect(self.image_ready)
        self.update_button.clicked.connect(self.install_update)
        self.update_found.connect(lambda tag: (self.update_button.setText(f'Update to CandyCrunch {tag}'), self.update_button.show()))
        self.update_progress.connect(lambda percent: self.update_button.setText(f'Downloading update… {percent}%'))
        self.update_downloaded.connect(lambda ok: (self.update_button.setEnabled(True), self.update_button.setText('Install update and restart' if ok else 'Download failed, retry')))
        self.error_caught.connect(self.caught_error, Qt.QueuedConnection)
        self.report_done.connect(self.report_sent)
        # Errors in the window and its threads went nowhere, as the app has no console; now they go to the log and become crash reports
        previous, previous_thread, self.reported, self.reporting = sys.excepthook, threading.excepthook, set(), False

        def hook(kind, error, tb):
            if issubclass(kind, Exception):
                with contextlib.suppress(RuntimeError):
                    self.error_caught.emit(''.join(traceback.format_exception(kind, error, tb)), _exception_values(error))
            if sys.stderr is not None:
                previous(kind, error, tb)

        def thread_hook(args):
            if args.exc_value is not None and issubclass(args.exc_type, Exception):
                with contextlib.suppress(RuntimeError):
                    self.error_caught.emit(''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)), _exception_values(args.exc_value))
            if sys.stderr is not None:
                previous_thread(args)

        sys.excepthook, threading.excepthook = hook, thread_hook
        # Release health on Sentry (crash-free sessions and users, adoption per release): a session per window, kept in the settings under the window's
        # process ID until it closes; one still there from a window that never closed (ended, from main) ends as crashed (its crash file has a trace) or
        # abnormal (killed, power cut). None when the user declined crash reports
        self.session = None
        if REPORT_DSN and self.store.value('crash_reports', '') != 'never':
            from importlib.metadata import version, PackageNotFoundError
            self.store.beginGroup('sessions')
            for pid in self.store.childKeys():
                stored, started = json.loads(self.store.value(pid))
                if int(pid) in ended:
                    _send_session({**stored, 'init': False, 'status': 'crashed' if ended[int(pid)] else 'abnormal', 'errors': stored['errors'] + ended[int(pid)]})
                if int(pid) in ended or time.time() - started > 5 * 86400:
                    self.store.remove(pid)
            self.store.endGroup()
            if not self.store.value('install_id'):
                self.store.setValue('install_id', uuid.uuid4().hex)
            with contextlib.suppress(PackageNotFoundError):
                self.session, self.session_start = {'sid': str(uuid.uuid4()), 'did': self.store.value('install_id'), 'init': True, 'status': 'ok', 'errors': 0,
                                                    'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                    'attrs': {'release': f'candycrunch@{version("candycrunch")}',
                                                              'environment': 'app' if getattr(sys, 'frozen', False) else 'package'}}, time.time()
                self.store.setValue(f'sessions/{os.getpid()}', json.dumps([self.session, self.session_start]))
                _send_session(self.session)
        if crashes:
            QTimer.singleShot(0, lambda: self.report_error('CandyCrunch closed unexpectedly the last time it ran', files = {'window_crash.txt': '\n\n'.join(crashes)}))
        self.images.get('Fuc(a1-2)Gal(b1-3)GalNAc', compact = False, urgent = True)
        if self.store.value('settings'):
            stored = json.loads(self.store.value('settings'))
            self.settings.set_values({**DEFAULTS, **stored, 'filter_out': set(stored.get('filter_out', DEFAULTS['filter_out']))})
        # The panes as the user left them: settings and results, table and details, details and spectrum, MS3 spectrum and fragments, CandyCrumbs
        # form and spectrum (restoreState only holds the log dock, so every start reset them)
        self.splitters = (self.split, self.results.splitter, self.results.splitter.widget(1), self.results.ms3_tab, self.crumbs.findChild(QSplitter))
        if self.store.value('geometry'):
            self.restoreGeometry(self.store.value('geometry'))
            self.restoreState(self.store.value('state'))
            for splitter, state in zip(self.splitters, self.store.value('splitters') or []):
                splitter.restoreState(state)
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

    def caught_error(self, text, values):
        self.append_log(text)
        self.report_error(text.strip().splitlines()[-1], values)

    def report_event(self, level, values = None, summary = None):
        """A Sentry event, an error (values from _exception_values) or a message, with where it happened: versions, computer, and the run list"""
        from importlib.metadata import version, PackageNotFoundError
        versions = {}
        for package in ('candycrunch', 'glycowork', 'glycorender', 'torch', 'onnxruntime', 'numpy', 'pandas', 'PySide6'):
            with contextlib.suppress(PackageNotFoundError):
                versions[package] = version(package)
        memory = None
        with contextlib.suppress(Exception):
            if sys.platform == 'win32':
                import ctypes
                class Status(ctypes.Structure):
                    # MEMORYSTATUSEX
                    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong)] + [
                        (f'other{i}', ctypes.c_ulonglong) for i in range(6)]

                status = Status(length = ctypes.sizeof(Status))
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                memory = status.total
            else:
                memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        runs = [{'experiment': name,
                 'files': [os.path.splitext(p)[1].lower() + (f', {os.path.getsize(p) / 1e6:.0f} MB' if os.path.isfile(p) else '') for p in paths],
                 'settings': {k: sorted(v) if isinstance(v, set) else v for k, v in settings.items()}} for name, paths, settings in
                self.settings.experiments()]
        event = {'event_id': uuid.uuid4().hex, 'timestamp': time.time(), 'platform': 'python', 'level': level,
                 'release': f'candycrunch@{versions.get("candycrunch")}',
                 'environment': 'app' if getattr(sys, 'frozen', False) else 'package', 'tags': {'tab': self.tabs.tabText(self.tabs.currentIndex())},
                 'contexts': {'os': {'name': platform.system(), 'version': platform.release(), 'build': platform.version()},
                              'runtime': {'name': 'CPython', 'version': platform.python_version()},
                              'device': {'arch': platform.machine(), 'processor_count': os.cpu_count(), 'memory_size': memory}},
                 'extra': {'versions': versions, 'runs': runs, 'running': self.runner.busy}}
        if values:
            event['exception'] = {'values': values}
        elif summary:
            event['message'] = summary
        return event

    def report_error(self, summary, values = None, files = None, box = False):
        """An error the user did not cause on purpose: sent right away if they chose automatic crash reports, else offered, each kind of error once per
        session; box: tell the user about the error in a message box (a failed run), else in the status bar"""
        if self.session:
            self.session['errors'] += 1
            self.store.setValue(f'sessions/{os.getpid()}', json.dumps([self.session, self.session_start]))
        signature = (values[-1]['type'], *[(f['module'], f['lineno']) for f in values[-1]['stacktrace']['frames'][-3:]]) if values else summary
        choice, new = self.store.value('crash_reports', ''), signature not in self.reported
        self.reported.add(signature)
        note = ''
        if new and choice == 'always' and REPORT_DSN:
            self.send_report(self.report_event('error', values, summary), {**(files or {}), 'log.txt': self.log.toPlainText()[-300000:]})
            note = 'A crash report was sent to the CandyCrunch developers.'
        elif new and choice != 'never' and not self.reporting:
            self.open_report(summary, values, files)
            return
        if box:
            QMessageBox.critical(self, 'CandyCrunch', f'{summary}\n\n{note}'.strip())
        else:
            self.statusBar().showMessage(f'Something went wrong ({summary}). {note} Details are in the log (View > Log).'.replace('  ', ' '), 10000)

    def open_report(self, summary = None, values = None, files = None, peak = None):
        """The report form: a bug report (Help > Report a bug), for an error (summary) the offer to send a crash report, or for a peak of the results (peak,
        from ResultsView.peak_report) a misprediction report"""
        event = self.report_event('error', values, summary) if summary else None
        files = {**({'misprediction.json': json.dumps(peak, indent = 1, default = lambda x: x.item() if isinstance(x, np.generic) else str(x))} if peak else {}), **(files or {}), 'log.txt': self.log.toPlainText()[-300000:]}
        # A bug report is user feedback; with a crash report, a description goes with the error as feedback on it
        feedback = self.report_event('info')
        feedback['type'] = 'feedback'
        dialog, self.reporting = ReportDialog(self, event or feedback, files, summary, peak), True
        accepted, self.reporting = dialog.exec() == QDialog.Accepted, False
        message, email = dialog.text.toPlainText().strip(), dialog.email.text().strip()
        include = peak is not None and dialog.include_data.isChecked()
        if summary and dialog.automatic.isChecked():
            self.store.setValue('crash_reports', 'always' if accepted else 'never')
            self.auto_action.setChecked(accepted)
        dialog.deleteLater()
        if not accepted:
            return
        if peak:
            # Tagged, so that mispredictions can be filtered and grouped by what is wrong; the first line is also the GitHub issue's title
            feedback['tags'].update(report = 'misprediction', problem = dialog.problem.currentData())
            row, predicted = peak['peak'], peak['predicted']
            message = '\n'.join(filter(None, [f'Misprediction: {dialog.problem.currentText()} (m/z {_fmt(row.get("m/z"), 4)}, z = {_fmt(row.get("charge"), 0)}, RT '
                                               f'{_fmt(row.get("RT"), 2)} min)', f'CandyCrunch: {predicted if isinstance(predicted, str) else "no structure"}',
                                               dialog.correct.text().strip() and f'Correct: {dialog.correct.text().strip()}', message]))
        feedback['contexts']['feedback'] = {k: v for k, v in {'message': message, 'contact_email': email, 'source': 'app',
                                                              'associated_event_id': event and event['event_id']}.items() if v}
        if include:
            data = self.results.report_data
            files['run_peaks.csv'] = data['table']
            if data['path'] and os.path.exists(data['path']):
                # The peak's surroundings are cut out of the run file in a process of its own (it reads the whole run again, seconds to a minute); the
                # report goes once that is done
                def work():
                    try:
                        with ProcessPoolExecutor(1, mp_context = mp.get_context('spawn')) as pool:
                            files['peak_data.cc'] = pool.submit(_extract_peak_data, data['path'], data['mz'], data['charge'], data['rt']).result()
                    except Exception as error:
                        files['peak_data_error.txt'] = f'{type(error).__name__}: {error}'
                    self.send_report(event, files, feedback, quiet = False)

                threading.Thread(target = work, daemon = True).start()
                self.statusBar().showMessage('Cutting the data around the peak out of the run; the report is sent when that is done', 15000)
                return
            files['peak_data_error.txt'] = 'The run file is no longer where it was analyzed, so the data around the peak could not be included'
        self.send_report(event, files, feedback if message else None, quiet = False)

    def send_report(self, event, files, feedback = None, quiet = True):
        """Sends in the background; one the user sent by hand that cannot be sent opens as a GitHub issue instead (report_sent)"""

        def work():
            # The files go with the error, or with a bug report, which has none
            ok = (event is None or _send_report(event, files)) and (feedback is None or _send_report(feedback, None if event else files))
            self.report_done.emit(ok, None if quiet else (event, files, feedback))

        threading.Thread(target = work, daemon = True).start()

    def report_sent(self, ok, report):
        if report is None:
            return
        if ok:
            self.statusBar().showMessage('Report sent. Thank you!', 6000)
            return
        event, files, feedback = report
        # The issue form gets what fits into a link, the clipboard everything; the email address stays out, as issues are public
        error = '\n'.join(''.join(
            f'  File "{f["filename"]}", line {f["lineno"]}, in {f["function"]}\n' + (f'    {f["context_line"]}\n' if f['context_line'] else '')
            for f in value['stacktrace']['frames']) + f'{value["type"]}: {value["value"]}' for value in
                          (event or {}).get('exception', {}).get('values', []))
        source = event or feedback
        body = _scrub('\n\n'.join(filter(None, [feedback and feedback['contexts']['feedback']['message'], event and event.get('message'),
                                                error and f'```\n{error}\n```',
                                                'Versions: ' + ', '.join(f'{k} {v}' for k, v in source['extra']['versions'].items()) +
                                                f'; {source["contexts"]["os"]["name"]} {source["contexts"]["os"]["version"]}'])))
        # A data file (the .cc of a misprediction's surroundings) cannot go into a link or the clipboard: it is saved for the user to attach to the issue
        saved = []
        for name, data in files.items():
            if isinstance(data, bytes):
                folder = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                os.makedirs(folder, exist_ok = True)
                saved.append(os.path.join(folder, f'CandyCrunch_report_{name}'))
                with open(saved[-1], 'wb') as file:
                    file.write(data)
        body += _scrub(''.join(f'\n\nPlease attach {path} to this issue.' for path in saved))
        full = body + _scrub(''.join(f'\n\n{name}:\n```\n{text}\n```' for name, text in files.items() if isinstance(text, str) and text))
        title = (event.get('message') or f'{event["exception"]["values"][-1]["type"]}: {event["exception"]["values"][-1]["value"]}') if event else body.splitlines()[0]
        QApplication.clipboard().setText(full)
        QDesktopServices.openUrl(QUrl(f'{ISSUES_URL}?{urllib.parse.urlencode({"title": title[:120], "body": body[:6000]})}'))
        QMessageBox.information(self, 'CandyCrunch',
                                'The report could not be sent from here, so it opens as a GitHub issue in your browser instead (public once '
                                'you submit it; it needs a GitHub account). The full report, with the log, is on the clipboard to paste into the issue.')

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
        image = self.images.images.get(key)
        if image is None or image.isNull():
            return
        if key == ('Fuc(a1-2)Gal(b1-3)GalNAc', False):
            pixmap = QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(2.8)
            self.emblem.setPixmap(pixmap)

    def add_runs(self, paths):
        """Adds files and folders as runs and says so if none of them was a new run: files CandyCrunch cannot read (e.g., .wiff, .pdf), runs already in the
        list, and folders without runs (a folder's .xlsx are not taken, as CandyCrunch writes its results next to the runs as .xlsx) were passed over
        without a word by Add files and Add folder"""
        count = len(self.settings.paths())
        self.settings.add_paths(paths)
        if paths and len(self.settings.paths()) == count:
            self.statusBar().showMessage('No new runs added. CandyCrunch reads Thermo .raw, Bruker .d, mzML, mzXML, mgf, and .cc or .xlsx made by extract_spectra; '
                                         'a folder adds its runs but not its .xlsx files, which are added one by one', 10000)

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add LC-MS/MS runs', self.store.value('folder', os.path.expanduser('~')),
                                                'Spectra (*.raw *.RAW *.mzML *.mzXML *.mgf *.cc *.xlsx);;All files (*)')
        if paths:
            self.store.setValue('folder', os.path.dirname(paths[0]))
            self.add_runs(paths)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Add a timsTOF .d folder, or every .raw, .d, mzML, mzXML, mgf, and .cc run of a folder', self.store.value('folder', os.path.expanduser('~')))
        if path:
            self.store.setValue('folder', os.path.dirname(path) if path.lower().endswith('.d') else path)
            self.add_runs([path])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        sessions = [p for p in paths if p.lower().endswith('.candycrunch')]
        if sessions:
            self.open_session(sessions[0])
        self.add_runs([p for p in paths if p not in sessions])
        self.tabs.setCurrentWidget(self.split)

    def run_waiting(self):
        """Starts the run asked for while a run's settings check was still reading it"""
        if self.run_pending:
            self.run_pending = False
            self.run_button.setText('Run CandyCrunch')
            self.run_button.setEnabled(True)
            self.run()

    def run(self):
        # The settings check of a run just added (SettingsPanel.add_paths) can still change its experiment's settings, so the run waits for it
        if self.settings.waiting():
            self.run_pending = True
            self.run_button.setEnabled(False)
            self.run_button.setText('Reading the run settings…')
            return
        # The results shown are replaced once this run is done
        if not self.keep_curation():
            return
        experiments = self.settings.experiments()
        if not experiments:
            self.statusBar().showMessage('Add at least one LC-MS/MS run first', 6000)
            return
        if self.runner.dead:
            QMessageBox.critical(self, 'CandyCrunch', 'The CandyCrunch model could not be loaded, so nothing can be predicted; the error is in the log (View > Log).')
            return
        files = [file for _, paths, _ in experiments for file in paths]
        missing = [path for path in files if not os.path.exists(path)]
        if missing:
            QMessageBox.warning(self, 'CandyCrunch', 'These runs are no longer where they were added (moved, renamed, or on another computer); remove them and '
                                'add them again:\n\n' + '\n'.join(missing))
            return
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
        # A timsTOF .d folder counts with the size of its files
        self.file_sizes = {os.path.normcase(f): max(sum(os.path.getsize(os.path.join(f, x)) for x in os.listdir(f)) if os.path.isdir(f) else os.path.getsize(f), 1)
                           if os.path.exists(f) else 1 for f in files}
        self.history = {}
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        for bar in (self.progress, self.big_progress):
            bar.setRange(0, 0)
            bar.show()
        self.clock.start()
        # A long run would otherwise end with the computer asleep: Windows stays awake while this thread asks it to, macOS while caffeinate runs
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        elif sys.platform == 'darwin' and shutil.which('caffeinate'):
            self.awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
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
                # The pace of the 30 s up to the latest report, counted down since. Measured up to now instead, a step reporting nothing within it
                # shrank the window's progress while its length stayed 30 s, so the estimate rose to many minutes right before the end
                last = series[-1]
                old = next((t, v) for t, v in series if last[0] - t <= 30)
                if last[1] < 1 and last[1] - old[1] > 0.01 and last[0] - old[0] > 3:
                    lefts.append((1 - last[1]) * (last[0] - old[0]) / (last[1] - old[1]) - (now - last[0]))
            # The first steps go faster than the prediction that dominates, so the estimate waits until the prediction is under way. The last steps
            # report no progress within them and take seconds (xlsx) to minutes (MS1 quantification), so they get none
            last_steps = self.stage_times and self.stage_times[-1][0] in ('assign_categories', 'augment_predictions', 'finalise_predictions')
            if now - self.first_report[0] > 10 and self.fraction > 0.1 and lefts and max(lefts) > 0 and not last_steps:
                left = max(lefts)
                text += f', about {max(10, round(left / 10) * 10)} s left' if left < 175 else f', about {round(left / 60)} min left'
        self.status_text.setText(text)
        self.stage_label.setText(text)
        # Seen in the taskbar while the window is minimized or behind others
        self.setWindowTitle(f'CandyCrunch ({self.fraction:.0%})' if self.first_report is not None and self.clock.isActive() else 'CandyCrunch')

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
            # The reducing-end and glycan-class checks run a file's steps again, which starts its progress over: the pace before does not count
            if value < self.history.get(key, [(0, 0.0)])[-1][1]:
                self.history[key] = []
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
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        elif getattr(self, 'awake', None) is not None:
            self.awake.terminate()
            self.awake = None
        self.progress.hide()
        self.big_progress.hide()
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.stage_label.setText('')
        self.welcome_text.show()
        self.elapsed.setText('')
        self.setWindowTitle('CandyCrunch')
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
        self.session_path = None
        self.pages.setCurrentWidget(self.results)
        self.save_action.setEnabled(True)
        # wrap_inference(_batch) processes files as another glycan class or with another reducing end if their predictions say so, which the user has to know
        notes = [line.removeprefix('WARNING: ') for line in self.log.toPlainText().splitlines() if 'the glycan class' in line or 'the reducing end' in line]
        if notes:
            QMessageBox.warning(self, 'CandyCrunch', '\n\n'.join(notes))
        QApplication.alert(self)

    def failed(self, text):
        self.stop('CandyCrunch stopped with an error; details are in the log')
        self.append_log(text)
        self.log_dock.show()
        report = self.runner.report or {}
        self.report_error(text.strip().splitlines()[-1], report.get('values'), report.get('files'), box = True)

    def cancel(self):
        self.runner.cancel()
        self.stop('Run cancelled')
        self.append_log('Run cancelled\n')

    def save_session(self):
        # Offered where these results were opened from or last saved, else named after their first run: every save offered results.candycrunch in the
        # runs' folder, so saving other results meant renaming them or overwriting the earlier ones
        first = os.path.splitext(os.path.basename(os.path.normpath(self.results.session['experiments'][0]['files'][0])))[0]
        path, _ = QFileDialog.getSaveFileName(self, 'Save results', getattr(self, 'session_path', None) or os.path.join(self.store.value('folder', os.path.expanduser('~')),
                                                                                                                     f'{first}.candycrunch'), 'CandyCrunch results (*.candycrunch)')
        if not path:
            return False
        try:
            with open(path, 'wb') as file:
                pickle.dump(self.results.session, file)
        except OSError as error:
            # A folder without write access or a full disk is no crash; returning False keeps the window (and the curation) open
            QMessageBox.critical(self, 'Save results', f'Could not write {path}:\n{error}')
            return False
        self.results.dirty, self.session_path = False, path
        self.statusBar().showMessage(f'Saved {os.path.basename(path)}', 6000)
        return True

    def keep_curation(self):
        """Offers to save curated results before a run or other results replace them or the app closes; False if the user cancels"""
        if not self.results.dirty:
            return True
        answer = QMessageBox.question(self, 'CandyCrunch', 'Your curation of the current results is not saved yet. Save it first?',
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        return self.save_session() if answer == QMessageBox.Save else answer == QMessageBox.Discard

    def open_session(self, path = None):
        if self.runner.busy:
            # Opening would replace the run list while results of the runs it held are still to come
            self.statusBar().showMessage('Wait for the run to finish or cancel it before opening results', 6000)
            return
        if not self.keep_curation():
            return
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
        self.session_path = path
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
        if not self.keep_curation():
            event.ignore()
            return
        self.store.setValue('geometry', self.saveGeometry())
        self.store.setValue('state', self.saveState())
        self.store.setValue('splitters', [splitter.saveState() for splitter in self.splitters])
        self.runner.shutdown()
        if self.session:
            # Waits a little for the session's end to reach Sentry, as the process ends right after; nothing counts for the session after this
            session, self.session = self.session, None
            self.store.remove(f'sessions/{os.getpid()}')
            _send_session({**session, 'init': False, 'status': 'exited', 'duration': time.time() - self.session_start}).join(3)
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
            file.write(''.join(log) + f'\nGlycan peaks found: {outcome[0] if outcome else 0}\nExample run bundled: {os.path.exists(EXAMPLE_RUN)}\n')
        sys.exit(0 if outcome and outcome[0] > 0 and os.path.exists(EXAMPLE_RUN) else 1)
    # A window crash file that can be removed belongs to an earlier session that did not end cleanly (one in use is a running CandyCrunch's), and is
    # offered as a crash report; worker files are removed by their window, or here once a day old
    crashes, ended = [], {}
    for path in glob.glob(CRASH_FILE.format('*', '*')):
        own = '-window-' in os.path.basename(path)
        with contextlib.suppress(OSError):
            if own or time.time() - os.path.getmtime(path) > 86400:
                with open(path, encoding = 'utf-8', errors = 'replace') as file:
                    text = file.read()
                os.remove(path)
                if own:
                    # The window of that process never closed; faulthandler wrote a trace if it crashed
                    ended[int(os.path.splitext(os.path.basename(path))[0].rsplit('-', 1)[1])] = bool(text.strip())
                if own and text.strip():
                    crashes.append(text[-100000:])
    crash_file = open(CRASH_FILE.format('window', os.getpid()), 'w')
    faulthandler.enable(crash_file, all_threads = True)
    window = MainWindow([os.path.abspath(p) for p in sys.argv[1:]], crashes, ended)
    window.show()
    code = app.exec()
    faulthandler.disable()
    crash_file.close()
    with contextlib.suppress(OSError):
        os.remove(crash_file.name)
    sys.exit(code)


if __name__ == '__main__':
    main()

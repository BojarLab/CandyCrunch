# conftest.py
from tabulate import tabulate
import numpy as np
from collections import defaultdict
import pytest
import json
import os
import pandas as pd
from datetime import datetime

# Each pytest-xdist worker runs its own pipeline, so the workers share the cores instead of each spawning a torch thread per core
if os.environ.get('PYTEST_XDIST_WORKER_COUNT'):
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(os.environ['PYTEST_XDIST_WORKER_COUNT'])))

# Workers inherit this from the main process, which loads this file before it starts them: with a fixed hash seed, set iteration order (which leaks into
# impute's variant choice and subgraph order) and with it every seeded run repeats exactly
os.environ.setdefault('PYTHONHASHSEED', '0')

# Version of evaluate_predictions in test_CandyCrunch.py and of how runs are made; changing either moves scores like a ground-truth edit, so it is part of
# each dataset's key ('-seeded': every run is seeded since 2026-10-13, so the unseeded baselines before it do not compare)
METRIC_VERSION = '-hungarian-halfopen-seeded-braces'


def pytest_addoption(parser):
    parser.addoption(
        "--input-format",
        action="store",
        default="xlsx",
        choices=["xlsx", "mzml", "both"],
        help="Which spectra format to test: xlsx (default), mzml (files with MS1: .mzML, .mzXML, Thermo .raw), or both",
    )


@pytest.fixture(scope="session")
def input_format(request):
    return request.config.getoption("--input-format")


@pytest.fixture(scope="session")
def verbose(request):
    return request.config.getoption("verbose") > 0


class ResultCollector:
    def __init__(self):
        self.results = defaultdict(list)
        self.param_names = None
        self.current_dict = None
        self.dict_results = defaultdict(lambda: defaultdict(list))
        self.dict_full_results = defaultdict(lambda: defaultdict(list))
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results_log.json")
        self.previous_results = self.load_previous_results()
        self.gt_hashes = {}
        self.regressed = set()

    def load_previous_results(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def save_current_results(self):
        # Start from the previous log, so a partial run (one input format, a -k subset) keeps the baselines it did not touch
        final_results = dict(self.previous_results)
        for dict_name, param_results in self.dict_results.items():
            prev = self.previous_results.get(dict_name, {})
            # Baselines scored against a different ground truth are dropped, not carried over
            scores = dict(prev.get('scores', {})) if prev.get('gt_hash') in (None, self.gt_hashes.get(dict_name)) else {}
            for params, run_scores in param_results.items():
                # A regressed score must not become the new baseline, or the regression passes silently on the next run
                if (dict_name, str(params)) not in self.regressed:
                    scores[str(params)] = np.mean(run_scores)
            final_results[dict_name] = {
                'scores': scores,
                'gt_hash': self.gt_hashes.get(dict_name),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        # Save to log file
        with open(self.log_file, 'w') as f:
            json.dump(final_results, f, indent=4)

    def export(self):
        """Results of a pytest-xdist worker, as plain types for the controller, which prints the tables and saves the log"""
        return {'param_names': self.param_names, 'gt_hashes': self.gt_hashes, 'regressed': [list(x) for x in self.regressed],
                'results': [[name, list(params), [float(s) for s in scores]] for name, d in self.dict_results.items() for params, scores in d.items()],
                'full': [[name, list(params), [[float(x) for x in s] for s in scores]] for name, d in self.dict_full_results.items() for params, scores in d.items()]}

    def merge(self, payload):
        self.param_names = self.param_names or payload['param_names']
        self.gt_hashes.update(payload['gt_hashes'])
        self.regressed.update(tuple(x) for x in payload['regressed'])
        for name, params, scores in payload['results']:
            self.dict_results[name][tuple(params)].extend(scores)
        for name, params, scores in payload['full']:
            self.dict_full_results[name][tuple(params)].extend(scores)

    def check_performance(self, test_dict_name, param_key, current_score, gt):
        """Check that the current score has not dropped below the previous run's on the same ground truth"""
        # Editing the ground truth (e.g., adding newly found glycans) moves scores without any code change, so baselines are tied to its content
        self.gt_hashes[test_dict_name] = str(pd.util.hash_pandas_object(gt, index = False).sum()) + METRIC_VERSION
        if test_dict_name in self.previous_results and self.previous_results[test_dict_name].get('gt_hash') in (None, self.gt_hashes[test_dict_name]):
            prev_scores = self.previous_results[test_dict_name]['scores']
            # The log stores scores without the dataset name, so the key must drop it to ever match
            param_key_str = str(param_key[1:])
            if param_key_str in prev_scores:
                prev_score = prev_scores[param_key_str]
                # Runs are seeded, so a drop comes from the code; the margin dates from unseeded runs, which differed by up to 0.02 per dataset (one of milk's 16 peaks is 0.04)
                if current_score < prev_score - 0.04:
                    self.regressed.add((test_dict_name, param_key_str))
                    raise AssertionError(
                        f"\nPerformance regression detected for {test_dict_name}!"
                        f"\nPrevious score: {prev_score:.3f}"
                        f"\nCurrent score: {current_score:.3f}"
                        f"\nParameters: {param_key}")

    def add_result(self, params, score, full_scores = None):
        test_dict_name = params['test_dict']['name']
        param_key = tuple(
            params[key] if key != 'test_dict' else params['test_dict']['name']
            for key in self.param_names
        )
        # Append the individual score to the list of scores for this parameter combination
        self.dict_results[test_dict_name][param_key[1:]].append(score)
        if full_scores is not None:
            self.dict_full_results[test_dict_name][param_key[1:]].append(full_scores)
        # If we've switched to a new test_dict, print the results for the previous one
        if self.current_dict != test_dict_name and self.current_dict is not None:
            self.print_dict_results(self.current_dict)
        self.current_dict = test_dict_name

    def print_dict_results(self, dict_name):
        print(f"\n=== Results for {dict_name} ===")
        # Get results for this test_dict
        dict_specific_results = self.dict_results[dict_name]
        dict_specific_full = self.dict_full_results[dict_name]
        # Generate headers (exclude test_dict since it's the same for all rows)
        headers = list(self.param_names.values())[1:] + ['F1', 'Prec', 'Rec', 'NotPicked', 'Wrong', 'FP']
        table_data = []
        # Calculate averages and sort by F1 score
        averaged_results = {
            params: np.mean(scores)
            for params, scores in dict_specific_results.items()
        }
        # Sort by average F1 score and create table rows
        for params, avg_f1 in sorted(averaged_results.items(), key=lambda x: x[1], reverse=True):
            full = dict_specific_full.get(params, [])
            avg_prec = np.mean([s[1] for s in full]) if full else float('nan')
            avg_rec = np.mean([s[2] for s in full]) if full else float('nan')
            avg_np = np.mean([s[3] for s in full]) if full else float('nan')
            avg_wrong = np.mean([s[4] for s in full]) if full else float('nan')
            avg_fp = np.mean([s[6] for s in full]) if full else float('nan')
            row = list(params) + [f"{avg_f1:.3f}", f"{avg_prec:.3f}", f"{avg_rec:.3f}", f"{avg_np:.1f}", f"{avg_wrong:.1f}", f"{avg_fp:.1f}"]
            table_data.append(row)
        print(tabulate(table_data, headers = headers, tablefmt = 'grid'))
        print()

    def get_table(self):
        if not self.dict_results:
            return "No results collected!"
        self.save_current_results()
        # Print overall results
        print("\n=== Overall Performance Results ===")
        table_data = []
        # Combine results from all test_dicts
        overall_results = defaultdict(list)
        overall_full = defaultdict(list)
        for dict_name, param_results in self.dict_results.items():
            for params, scores in param_results.items():
                overall_key = (dict_name,) + params
                overall_results[overall_key].extend(scores)
        for dict_name, param_results in self.dict_full_results.items():
            for params, scores in param_results.items():
                overall_key = (dict_name,) + params
                overall_full[overall_key].extend(scores)
        # Calculate averages and sort
        averaged_overall = {
            params: np.mean(scores)
            for params, scores in overall_results.items()
        }
        for params, avg_f1 in sorted(averaged_overall.items(), key = lambda x: x[1], reverse = True):
            full = overall_full.get(params, [])
            avg_prec = np.mean([s[1] for s in full]) if full else float('nan')
            avg_rec = np.mean([s[2] for s in full]) if full else float('nan')
            avg_np = np.mean([s[3] for s in full]) if full else float('nan')
            avg_wrong = np.mean([s[4] for s in full]) if full else float('nan')
            avg_fp = np.mean([s[6] for s in full]) if full else float('nan')
            row = list(params) + [f"{avg_f1:.3f}", f"{avg_prec:.3f}", f"{avg_rec:.3f}", f"{avg_np:.1f}",
                                  f"{avg_wrong:.1f}", f"{avg_fp:.1f}"]
            table_data.append(row)
        headers = list(self.param_names.values()) + ['F1', 'Prec', 'Rec', 'NotPicked', 'Wrong', 'FP']
        param_scores = defaultdict(lambda: {'means': [], 'weights': []})
        for param_results in self.dict_results.values():
            for params, scores in param_results.items():
                param_scores[params]['means'].append(np.mean(scores))
                param_scores[params]['weights'].append(len(scores))
        print("\n=== Weighted Summary ===")
        for params, data in sorted(param_scores.items()):
            means = np.array(data['means'])
            weights = np.array(data['weights'])
            weighted_mean = np.average(means, weights = weights)
            n = len(means)
            se = np.std(means, ddof = 1) / np.sqrt(n) if n > 1 else float('nan')
            param_label = ', '.join(
                f"{list(self.param_names.values())[i + 1]}={p}"
                for i, p in enumerate(params)
            )
            line = f"[{param_label}]  score={weighted_mean:.4f}  ±{1.96 * se:.4f} (95% CI, n={n} datasets)"
            print(line)
        return tabulate(table_data, headers = headers, tablefmt = 'grid')

collector = ResultCollector()

@pytest.fixture(scope = "session")
def result_collector():
    return collector


def pytest_configure(config):
    config.collector = collector


# Without timings from an earlier run (pytest's cache), the pipeline tests are assumed to be the slow ones
DEFAULT_DURATIONS = {'test_candycrunch_batch': 300, 'test_candycrunch_accuracy': 60}
test_durations = {}


def pytest_collection_modifyitems(config, items):
    # Longest tests first, each followed by a short one: with pytest-xdist (--dist load --maxschedchunk 1, as in pytest.ini) every worker holds one
    # running and one queued test, so this hands the slow tests to different workers as they become free instead of queueing two behind each other
    durations = config.cache.get('candycrunch/durations', {}) if getattr(config, 'cache', None) else {}
    est = {item.nodeid: durations.get(item.nodeid, DEFAULT_DURATIONS.get(getattr(item, 'originalname', ''), 0)) for item in items}
    slow = sorted([item for item in items if est[item.nodeid] >= 10], key = lambda item: -est[item.nodeid])
    fast = [item for item in items if est[item.nodeid] < 10]
    items[:] = [x for pair in zip(slow, fast) for x in pair] + slow[len(fast):] + fast[len(slow):]


def pytest_runtest_logreport(report):
    if report.when == 'call':
        test_durations[report.nodeid] = report.duration


@pytest.hookimpl(optionalhook = True)
def pytest_testnodedown(node, error):
    # Controller of a pytest-xdist run: collect each worker's results
    payload = getattr(node, 'workeroutput', {}).get('candycrunch')
    if payload:
        collector.merge(json.loads(payload))


def pytest_sessionfinish(session):
    if hasattr(session.config, 'workerinput'):
        session.config.workeroutput['candycrunch'] = json.dumps(collector.export())
        return
    if test_durations and getattr(session.config, 'cache', None):
        session.config.cache.set('candycrunch/durations', {**session.config.cache.get('candycrunch/durations', {}), **test_durations})
    print("\n=== Final Summary ===")
    # Print tables for each test_dict
    for dict_name in collector.dict_results:
        collector.print_dict_results(dict_name)
    # Print overall results (get_table prints its own header)
    print(collector.get_table())
    print()
    if collector.previous_results:
        print("\n=== Comparison with Previous Run ===")
        for dict_name in collector.dict_results:
            if dict_name in collector.previous_results:
                prev_timestamp = collector.previous_results[dict_name]['timestamp']
                prev_score = collector.previous_results[dict_name]['scores']
                print(f"\n{dict_name} (Previous run: {prev_timestamp}, Score: {prev_score})")
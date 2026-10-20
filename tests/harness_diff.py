"""Row-level difference between two harness runs: which ground-truth peaks flipped between exact / partly right / missed, which false positives appeared or
vanished, per set, and the trace stage of each flipped peak in both runs.

Usage (from the repo root):
    python tests/harness_diff.py base.json new.json [--sets GPST000029,milk] [--all]
    python tests/harness_diff.py preds_base preds_new [--sets ...]          (two --save_preds folders: scored here against the ground truth)
--out JSONs of a harness with per-peak outcomes ('peaks', 'fps') are compared directly; for older ones, or for --save_preds folders, the outcomes are
computed from the predictions (needs both folders, or --preds_a/--preds_b next to the JSONs). --all also lists unchanged sets.
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tests'))


def outcomes_from_preds(folder, wanted):
    # Per-peak outcomes of every pickled output table in a --save_preds folder, scored against the ground truth the harness would use
    import harness
    parser = argparse.Namespace(sets = None,
                                private_spectra = str(harness.GG_DIR / 'improved_xlsx_2026-10-10' / 'private_test'),
                                private_gt = str(harness.GG_DIR / 'refined_ground_truths' / 'private_test'), private_ms1 = str(harness.GG_DIR / 'private_files_raw' / 'mzml'), no_private = False)
    tmp = pathlib.Path(folder) / '_gt'
    tmp.mkdir(exist_ok = True)
    out = {}
    for fmt in ['xlsx', 'mzml']:
        parser.format = fmt
        for t in harness.collect_tasks(parser):
            path = pathlib.Path(folder) / f"{t['label'].replace('/', '__')}_seed0.pkl"
            if t['label'] in out or not path.exists() or (wanted and t['set'] not in wanted):
                continue
            t = harness.solution_to_gt(t, tmp)
            gt = pd.read_csv(t['gt'])
            rt_col = 'RT' if 'RT' in gt.columns else t['run'] + '_RT'
            if t['run'] in gt.columns:
                gt = gt[gt[t['run']] > 0]
            peaks, fps = harness.peak_outcomes(pd.read_pickle(path), gt, rt_col)
            out[t['label']] = dict(label = t['label'], set = t['set'], held_out = t['held_out'], peaks = peaks, fps = fps, trace = [])
    return out


def load(path, wanted):
    p = pathlib.Path(path)
    if p.is_dir():
        return outcomes_from_preds(p, wanted), None
    run = json.load(open(p))
    results = {r['label']: r for r in run['results'] if r.get('seed', 0) == 0 and (not wanted or r['set'] in wanted)}
    return results, run


def score(peak):
    # the peak's TP contribution: its similarity, 0.5 for a hit open peak
    if peak['pred'] is None:
        return 0.0
    return 0.5 if peak['gt'] is None else peak['sim']


def kind(peak):
    if peak['pred'] is None:
        return 'missed'
    return 'hit' if peak['gt'] is None else 'exact' if peak['sim'] == 1 else 'partial'


def peaks_f1(result):
    # F1 from the per-peak outcomes, as evaluate_predictions computes it (an open peak counts 0.5 either way)
    tp = sum(score(p) for p in result['peaks'])
    fn = sum((1 - (p['sim'] or 0)) if p['gt'] is not None else (0.5 if p['pred'] is None else 0) for p in result['peaks'])
    fp = len(result['fps'])
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def stage_of(result, peak):
    # the trace row of a peak (missed or wrong), by its mass and RT
    for t in result.get('trace', []):
        if t['kind'] in ('missed', 'wrong') and t['mass'] is not None and abs(t['mass'] - peak['mass']) < 1e-3 and abs(t['rt'] - peak['rt']) < 1e-3:
            return t['stage']
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Row-level difference between two harness runs')
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--sets', default = None)
    parser.add_argument('--preds_a', default = None, help = '--save_preds folder of run a, for a JSON without per-peak outcomes')
    parser.add_argument('--preds_b', default = None)
    parser.add_argument('--all', action = 'store_true')
    args = parser.parse_args()
    wanted = set(args.sets.split(',')) if args.sets else None
    (a, run_a), (b, run_b) = load(args.a, wanted), load(args.b, wanted)
    for side, results, preds in [('a', a, args.preds_a), ('b', b, args.preds_b)]:
        if any('peaks' not in r for r in results.values()):
            if not preds:
                sys.exit(f'{side} has no per-peak outcomes (a harness run before they were recorded): give its --save_preds folder with --preds_{side}')
            computed = outcomes_from_preds(preds, wanted)
            for label, r in results.items():
                r.update({k: computed[label][k] for k in ['peaks', 'fps']} if label in computed else {'peaks': [], 'fps': []})
    by_set = defaultdict(list)
    for label in sorted(set(a) & set(b)):
        by_set[a[label]['set']].append(label)
    totals = defaultdict(float)
    for name in sorted(by_set, key = lambda n: (a[by_set[n][0]].get('held_out', False), n)):
        lines, d_tp, d_fp = [], 0.0, 0
        for label in by_set[name]:
            ra, rb = a[label], b[label]
            pa = {(round(p['mass'], 4), round(p['rt'], 3), p['gt']): p for p in ra['peaks']}
            pb = {(round(p['mass'], 4), round(p['rt'], 3), p['gt']): p for p in rb['peaks']}
            for key in sorted(set(pa) | set(pb)):
                x, y = pa.get(key), pb.get(key)
                if x is None or y is None:
                    lines.append(f"  GT row {key[0]:.2f} @ {key[1]:.2f} only in {'b' if x is None else 'a'} (ground truth differs)")
                    continue
                if (x['pred'], x['sim']) != (y['pred'], y['sim']):
                    d = score(y) - score(x)
                    d_tp += d
                    lines.append(f"  {key[0]:9.2f} @ {key[1]:6.2f}  {kind(x):>7} {stage_of(ra, x):<24} -> {kind(y):<7} {stage_of(rb, y):<24} dTP {d:+.2f}  "
                                 f"GT {key[2] or '(open)'}  |  {x['pred'] or '-'}  ->  {y['pred'] or '-'}")
            fa = {(f['mass'], f['rt'], f['pred']) for f in ra['fps']}
            fb = {(f['mass'], f['rt'], f['pred']) for f in rb['fps']}
            for f in sorted(fa - fb):
                lines.append(f"  FP gone {f[0]:9.2f} @ {f[1]:6.2f}  {f[2]}")
            for f in sorted(fb - fa):
                lines.append(f"  FP new  {f[0]:9.2f} @ {f[1]:6.2f}  {f[2]}")
            d_fp += len(fb) - len(fa)
        f1 = lambda results: sum(results[l]['f1'] if 'f1' in results[l] else peaks_f1(results[l]) for l in by_set[name]) / len(by_set[name])
        totals['tp'] += d_tp
        totals['fp'] += d_fp
        if lines or args.all:
            print(f"{name}: F1 {f1(a):.4f} -> {f1(b):.4f} ({f1(b) - f1(a):+.4f}), TP {d_tp:+.2f}, FP {d_fp:+d}")
            print('\n'.join(lines))
    print(f"\nTotal over {len(by_set)} sets: TP {totals['tp']:+.2f}, FP {totals['fp']:+.0f}")

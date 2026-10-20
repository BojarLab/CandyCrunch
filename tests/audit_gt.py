"""Automatic ground-truth audit of a CandyCrunch test set against the LC-MS/MS run it describes

Usage (from the repo root, in the CandyCrunch environment):
    python tests/audit_gt.py <spectra file> <source GT csv> --glycan_class N [--run NAME] [--modification reduced] [--mass_tag 0]
                             [--sample_prep underivatized] [--output df_mz_X.csv] [--predictions <CandyCrunch output .xlsx/.csv>]
                             [--rt_window 1.5] [--lc PGC] [--library_cache PATH] [--data_dir tests/data] [--mode positive] [--max_charge 4]

<spectra file> is a .raw/.mzML/.mzXML/.d (MS1 used) or an extract_spectra .xlsx (no MS1: only the MS2-based checks apply). <source GT csv> has Mass,
RT (or <run>_RT), glycan, optionally charge, and the run's abundance column (<run>, the spectra file's stem unless --run says otherwise). Writes the
audited GT (default: the source path without '_source', or with '_audited' if it has no such suffix), <output stem>_removed.csv (source rows not kept
and predictions not added, each with its reason) and <output stem>_review.csv (cases that need a human), and prints a summary.

Criteria (.claude/notes/datasets.md, GT audits 2026-10-05 to 2026-10-14): a row needs a distinct MS1 peak of its composition on the envelope XIC of one of its ions
(Thermo ion traps: M-0.3 to M+1.3 m/z, which holds the isotopes they merge at z >= 2; Bruker ion traps, which resolve z = 2 isotopes, and
orbitrap/TOF: one window per isotope M..M+3 after the file's calibration), smoothed in time with a Gaussian of sigma 0.05 min (none at MS1 cycles
above 4 s), whose valley towards the nearest higher point on either side lies below 0.7 of its apex (0.5 if that higher peak's pooled MS2 has
cosine >= 0.9 to its own), at least twice the local median signal, and MS2 of the composition (any charge state) inside its half-height window.
Their pooled top fragments must be explained by CandyCrumbs as fragments of the composition (2 of the 10 most intense, or one with half their
intensity); weak (below the run's median signal) or unfragmented MS2 are inconclusive (kept, listed for review), strong non-glycan MS2 fail. A
later maximum within 3 min, at most half as high, whose signal never fell below 0.3 of its height since the earlier apex and whose MS2 has
cosine >= 0.8 to it is that peak's PGC tail. Rows are assigned to peaks per composition (Hungarian on the distance to the source RT plus the run's
RT shift, fitted on rows with one unambiguous peak, with penalties for peaks failing a criterion); rows of one composition at another charge and the
same source RT are copies. Mass becomes the theoretical m/z of the (de)protonated ion at the dominant charge, RT the apex. Open and added rows take
a structure from the spectral library of the curated test sets (pooled MS2, square-root intensities in 1 Da bins, precursor envelope removed) only
at cosine >= 0.8 with a clear best isomer; 0.7-0.8 goes to the review list. Isotope picks, 1.02 Da isobars (Neu5Ac/2 dHex, Neu5Gc/Hex + dHex),
co-apex in-source fragments of a superset composition, repeated fully specified structures, ambiguous peak choices and spray spikes are detected.
"""
import argparse
import ast
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from glycowork.motif.graph import compare_glycans
from glycowork.motif.processing import canonicalize_composition
from glycowork.motif.tokenization import composition_to_mass, get_ion_mzs, glycan_to_composition, mz_to_composition
from candycrunch.analysis import CandyCrumbs
from candycrunch.prediction import (load_spectra_filepath, ms1_mz_calibration, deisotope_ms2, modification_mass_dict, PROTON_MASS, ISOTOPE_SPACING,
                                    NEGATIVE_ADDUCTS, POSITIVE_ADDUCTS)

DATA_DIR = Path(__file__).resolve().parent / 'data'
VALLEY, VALLEY_SAME_MS2, SAME_MS2 = 0.7, 0.5, 0.9
TAIL_MS2, TAIL_RT, TAIL_RATIO, TAIL_FLOOR = 0.8, 3.0, 2.0, 0.3
LIBRARY_MS2, LIBRARY_REVIEW, LIBRARY_MARGIN, LIBRARY_VERSION = 0.8, 0.7, 0.05, 2
SLOW_CYCLE, SIGMA, BASELINE_WINDOW, MIN_OVER_BASELINE = 4 / 60, 0.05, 2.0, 2.0
GLYCAN_FRAGMENTS, TOP_FRAGMENTS, MIN_FRAGMENTATION = 2, 10, 0.1
COPY_RT, MATCH_MZ, MATCH_RT = 0.3, 0.5, 1.0
# Neutral mass above which a charge state is considered, as in prediction.mass_check
CHARGE_MIN_MASS = {2: 900, 3: 1500, 4: 3500}
STANDARD_COLUMNS = ['Mass', 'charge', 'RT', 'glycan']
AUDIT_COLUMNS = ['composition', 'source', 'RT_source', 'Mass_source', 'glycan_source', 'ms1_audit']


def comp_string(comp):
    return ''.join(f'{k}{v}' for k, v in canonicalize_composition(comp).items())


def comp_key(comp):
    return frozenset(canonicalize_composition(comp).items())


def same_glycan(a, b):
    try:
        return a == b or compare_glycans(a, b)
    except Exception:
        return False


def composition_of(glycan):
    try:
        return canonicalize_composition(glycan_to_composition(glycan))
    except Exception:
        return None


def ion_list(comp, mode, modification, mass_tag, sample_prep, adducts = False, extra_z = ()):
    """(name, |z|, m/z, whether it is the (de)protonated ion) of every ion of a composition up to |z| = 4 (above the CHARGE_MIN_MASS limits, except for
    the charges in extra_z, which the source GT gives for it: GPST000570's MassHunter lists [M+2H]2+ of 790-900 Da, 396.67, and 449.18)"""
    mass = composition_to_mass(comp, sample_prep = sample_prep, modification = modification) + mass_tag
    sign = -1 if mode == 'negative' else 1
    # Positive-mode glycans are mostly seen as metal adducts ([M+K+H]2+ dominated GPST000134), so all their ion forms make up the envelope
    ions = get_ion_mzs(mass, max_charge = 4 * sign, adducts = (NEGATIVE_ADDUCTS if adducts else None) if sign < 0 else POSITIVE_ADDUCTS,
                       min_mass = {z: m for z, m in CHARGE_MIN_MASS.items() if z not in extra_z})
    out = []
    for name, mz in ions.items():
        if not np.isfinite(mz):
            continue
        z = int(name.rsplit(']', 1)[1][:-1] or 1)
        out.append((name, z, float(mz), name in (f'[M-{z if z > 1 else ""}H]{z if z > 1 else ""}-', f'[M+{z if z > 1 else ""}H]{z if z > 1 else ""}+')))
    return out


def compositions_for_mz(mz, mode, modification, mass_tag, sample_prep, glycan_class, charges = (1, 2, 3)):
    """[(composition, |z|, m/z error)] of the glycowork compositions of a class whose (de)protonated ion at any of charges lies within 0.5 m/z, best first;
    mz_to_composition stops at the lowest charge with a hit, so each charge is searched as its singly charged equivalent"""
    sign, out = (-1 if mode == 'negative' else 1), []
    for z in charges:
        for comp in mz_to_composition(z * mz - sign * (z - 1) * PROTON_MASS, max_charge = sign, modification = modification, sample_prep = sample_prep,
                                      mass_tolerance = 0.5 * z, glycan_class = glycan_class, mass_tag = mass_tag or None):
            fits = [abs(m - mz) for _, zz, m, prot in ion_list(comp, mode, modification, mass_tag, sample_prep) if prot and zz == z]
            if fits and fits[0] <= 0.5:
                out.append((canonicalize_composition(comp), z, fits[0]))
    return sorted(out, key = lambda x: x[2])


def pooled_vector(peak_ds, precursor, z):
    """square-root intensities in 1 Da bins summed over spectra, unit length (as condense_dataframe's MS2 cosine), without the bins of the precursor's
    isotope envelope: ion-trap MS2 keep much of the unfragmented precursor, which made any two spectra of one m/z alike (cosine 0.99)"""
    v = np.zeros(3001)
    for d in peak_ds:
        if isinstance(d, dict) and d:
            m = np.clip(np.rint(np.fromiter(d.keys(), float)), 0, 3000).astype(int)
            np.add.at(v, m, np.fromiter(d.values(), float))
    v[max(0, int(round(precursor - 1))):int(round(precursor + 3.5 / z)) + 1] = 0
    v = np.sqrt(v)
    return v / (np.linalg.norm(v) + 1e-12)


def build_library(data_dir, cache_path):
    """pooled MS2 of the curated (structure-carrying) rows of every test set's GT, from its extract_spectra .xlsx; cached by file sizes and times"""
    gts = sorted(p for p in Path(data_dir).glob('*/df_mz_*.csv') if not p.stem.endswith(('_source', '_removed', '_review', '_audited')))
    # extract_spectra files, a .cc in place of the .xlsx of the same run
    xlsxs = sorted({p.with_suffix(''): p for p in sorted(Path(data_dir).glob('*/*.xlsx')) + sorted(Path(data_dir).glob('*/*.cc'))}.values())
    stamp = [LIBRARY_VERSION] + [(str(p), p.stat().st_size, p.stat().st_mtime) for p in gts + xlsxs]
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('stamp') == stamp:
            return cached['entries']
    entries, loaded = [], {}
    for gt_path in gts:
        gt = pd.read_csv(gt_path, encoding = 'utf-8-sig')
        runs = [x for x in xlsxs if x.parent == gt_path.parent and x.stem in gt.columns]
        if not runs:
            runs = [x for x in xlsxs if x.parent == gt_path.parent]
            runs = runs if len(runs) == 1 else []
        for x in runs:
            if x not in loaded:
                df = load_spectra_filepath(str(x))
                loaded[x] = (df['m/z' if 'm/z' in df.columns else 'reducing_mass'].to_numpy(float), df['RT'].to_numpy(float), list(df['peak_d']),
                             df.attrs.get('detected_mode') or 'negative')
            precs, rts, peak_ds, mode = loaded[x]
            rt_col = f'{x.stem}_RT' if f'{x.stem}_RT' in gt.columns else 'RT'
            rows = []
            for _, r in gt.iterrows():
                presence = pd.to_numeric(r.get(x.stem, 1), errors = 'coerce')
                if not isinstance(r['glycan'], str) or not r['glycan'].strip() or pd.isna(r.get(rt_col)) or presence == 0:
                    continue
                comp = composition_of(r['glycan'])
                if comp is None:
                    continue
                given_z = abs(int(r['charge'])) if 'charge' in gt.columns and pd.notna(r['charge']) else None
                # The reducing end is not recorded per test set (GPST000134 is unreduced), so it is the one whose ion fits the Mass
                fits = [(abs(mz - r['Mass']), z, mz, mod) for mod in ('reduced', None) for _, z, mz, prot in ion_list(comp, mode, mod, 0, 'underivatized')
                        if prot and (given_z is None or z == given_z)]
                if not fits or min(f[0] for f in fits) > 0.6:
                    continue
                best = min(fits, key = lambda x: x[0])
                rows.append((r['glycan'].strip(), comp, best[1], best[2], best[3], float(r[rt_col])))
            # Precursors of this run sit at a constant offset from the theoretical m/z (MS1 calibration), estimated from the rows themselves
            offsets = [precs[k] - mz for _, _, _, mz, _, rt in rows for k in np.flatnonzero((np.abs(precs - mz) <= 0.5) & (np.abs(rts - rt) <= 0.3))]
            offset = float(np.median(offsets)) if len(offsets) >= 5 else 0.0
            for glycan, comp, z, mz, mod, rt in rows:
                hits = np.flatnonzero((precs - mz - offset >= -0.3) & (precs - mz - offset <= (0.3 if z == 1 else 0.4)) & (np.abs(rts - rt) <= 0.3))
                if len(hits):
                    vec = pooled_vector([peak_ds[k] for k in hits], mz + offset, z)
                    nz = np.flatnonzero(vec)
                    entries.append({'dataset': gt_path.parent.name, 'gt': gt_path.name, 'run': x.stem, 'glycan': glycan, 'comp': comp_key(comp), 'z': z,
                                    'modification': mod, 'mz': mz, 'RT': rt, 'n_ms2': len(hits), 'bins': nz.astype(np.int16), 'values': vec[nz].astype(np.float32)})
    if cache_path:
        Path(cache_path).parent.mkdir(parents = True, exist_ok = True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'stamp': stamp, 'entries': entries}, f)
    return entries


def library_match(library, comp, z, modification, vec, exclude = ()):
    """best library cosine per structure of the same composition, charge and reducing end, best first: [(cosine, glycan, datasets)]"""
    best, key = {}, comp_key(comp)
    for e in library:
        if e['comp'] != key or e['z'] != z or e['modification'] != modification or e['dataset'] in exclude or e['run'] in exclude:
            continue
        cos = float(vec[e['bins'].astype(int)] @ e['values'])
        same = next((g for g in best if same_glycan(g, e['glycan'])), e['glycan'])
        c, ds = best.get(same, (-1, set()))
        best[same] = (max(c, cos), ds | {e['dataset']})
    return sorted(((c, g, ds) for g, (c, ds) in best.items()), key = lambda x: -x[0])


class Run:
    """MS2 spectra and flat MS1 store of one run, with envelope XICs and their chromatographic maxima"""

    def __init__(self, path, mode = None):
        df = load_spectra_filepath(str(path), extract_ms1 = True)
        if df.empty:
            raise ValueError(f'{Path(path).name} has no MS2 spectra')
        self.mode = mode or df.attrs.get('detected_mode') or 'negative'
        self.accurate = df.attrs.get('detected_ms1_trap') in ('orbitrap', 'tof')
        self.resolved, self.shift, self.half_width = False, 0.0, 0.3
        self.ms2_accurate = df.attrs.get('detected_trap') in ('orbitrap', 'other')
        self.prec, self.rt = df['m/z' if 'm/z' in df.columns else 'reducing_mass'].to_numpy(float), df['RT'].to_numpy(float)
        self.intensity = pd.to_numeric(df['intensity'], errors = 'coerce').to_numpy(float) if 'intensity' in df.columns else np.full(len(df), np.nan)
        self.peak_d = list(df['peak_d'])
        # Each MS2 spectrum's total signal as a percentile of the run's, to tell weak spectra (no evidence either way) from strong non-glycan ones
        tic = np.array([sum(d.values()) if isinstance(d, dict) else 0 for d in self.peak_d])
        self.tic_pct = np.argsort(np.argsort(tic, kind = 'stable'), kind = 'stable') / max(len(tic) - 1, 1)
        self.ms1 = df.attrs.get('ms1')
        self.spikes = np.zeros(0, dtype = bool)
        if self.ms1 is None or not len(self.ms1[0]):
            self.ms1, self.cycle, self.mz_range = None, np.nan, (self.prec.min() - 1, self.prec.max() + 1)
            return
        rts, mzs, ints, offsets = self.ms1
        n = len(rts)
        self.rts, self.mz_range = rts, (float(mzs.min()), float(mzs.max()))
        keys = np.repeat(np.arange(n) * 1e4, np.diff(offsets)) + mzs.astype(np.float64)
        order = np.argsort(keys, kind = 'stable') if np.any(np.diff(keys) < 0) else slice(None)
        self.keys, self.base = keys[order], np.arange(n) * 1e4
        self.cum = np.concatenate([[0], np.cumsum(ints[order], dtype = np.float64)])
        steps = np.diff(rts)
        self.cycle = float(np.median(steps[steps > 0]))
        # Thermo ion traps merge centroids closer than 0.5 m/z (z >= 2 isotopes become one centroid between them); Bruker ion traps resolve z = 2
        # isotopes (665.20/665.68/666.14 in GPST000316), so their centroids are often closer than 0.45
        scan_of = np.repeat(np.arange(n), np.diff(offsets))
        gaps = np.diff(mzs.astype(np.float64))[scan_of[1:] == scan_of[:-1]]
        self.resolved = not self.accurate and len(gaps) > 0 and float((gaps < 0.45).mean()) > 0.02
        # Spray spikes raise every m/z at once (milk 26.22/30.61 min, 4-6x the baseline at m/z without glycans), so they show in the median centroid
        # intensity and the TIC of the scan; their trace values are interpolated from the neighbouring scans, so they never make an apex
        med = np.array([np.median(ints[offsets[i]:offsets[i + 1]]) if offsets[i + 1] > offsets[i] else 0 for i in range(n)])
        tic = np.diff(self.cum[offsets])
        lo, hi = np.searchsorted(rts, rts - 1), np.searchsorted(rts, rts + 1, side = 'right')
        roll_med = np.array([np.median(med[a:b]) for a, b in zip(lo, hi)])
        roll_tic = np.array([np.median(tic[a:b]) for a, b in zip(lo, hi)])
        # Not in orbitrap/TOF survey scans: their centroid lists only hold peaks above the noise, so a big glycan raises their median and TIC alike (every
        # large apex of GPST000651 was flagged and interpolated away)
        self.spikes = (med > 3 * roll_med) & (tic > 2 * roll_tic) & (roll_med > 0) & (not self.accurate)
        # Gaussian smoothing in time (sigma 0.05 min, at least 0.7 cycles), as MS1 cycles vary 0.3-6 s within a run and scan-index smoothing invents humps
        self.weights = None
        if self.cycle <= SLOW_CYCLE:
            sigma = max(SIGMA, 0.7 * self.cycle)
            k = int(min(200, np.max(np.searchsorted(rts, rts + 3 * sigma, side = 'right') - np.arange(n))))
            idx = np.arange(n)[:, None] + np.arange(-k, k + 1)[None, :]
            self.idx = np.clip(idx, 0, n - 1)
            dt = rts[self.idx] - rts[:, None]
            valid = (idx >= 0) & (idx < n)
            w = np.where(valid & (np.abs(dt) <= 3 * sigma), np.exp(-0.5 * (dt / sigma) ** 2), 0)
            self.weights = w / w.sum(1, keepdims = True)

    def trace(self, lo, hi):
        t = self.cum[np.searchsorted(self.keys, self.base + hi, side = 'right')] - self.cum[np.searchsorted(self.keys, self.base + lo)]
        if self.spikes.any():
            t[self.spikes] = np.interp(self.rts[self.spikes], self.rts[~self.spikes], t[~self.spikes])
        return t

    def windows(self, mz, z, isotopes = 3):
        """m/z windows of an ion's isotopes M..M+isotopes, calibrated: one per isotope where MS1 resolves them (orbitrap/TOF within the calibration
        spread, Bruker ion traps up to z = 2), else one window from M-0.3 to M+1.3 that holds the isotopes Thermo ion traps merge"""
        mz, s = mz + self.shift, ISOTOPE_SPACING / z
        if self.accurate or (self.resolved and z <= 2):
            hw = max(self.half_width, 8e-6 * mz) if self.accurate else 0.4 * s
            return [(mz + j * s - hw, mz + j * s + hw) for j in range(isotopes + 1)]
        return [(mz - 0.3, mz + 1.3)] if isotopes else [(mz - 0.3, mz + 0.3)]

    def envelope(self, mz, z, isotopes = 3):
        return sum(self.trace(lo, hi) for lo, hi in self.windows(mz, z, isotopes))

    def smooth(self, t):
        return t if self.weights is None else (t[self.idx] * self.weights).sum(1)

    def maxima(self, raw):
        """maxima of a trace above MIN_OVER_BASELINE x the local median: apex index, RT, smoothed height, the valley towards the nearest higher point on
        either side relative to the apex (the shallower side; 0 without a higher point), the maximum that higher point belongs to, and the half-height window"""
        s, rts, out = self.smooth(raw), self.rts, []
        grid = np.arange(rts[0], rts[-1] + 0.5, 0.5)
        lo, hi = np.searchsorted(rts, grid - BASELINE_WINDOW), np.searchsorted(rts, grid + BASELINE_WINDOW, side = 'right')
        baseline = np.interp(rts, grid, [np.median(raw[a:b]) if b > a else 0 for a, b in zip(lo, hi)])
        # Scans in the uninterrupted stretch of signal around each scan; a peak needs 3 of them unless the MS1 cycle is slow
        signal = raw > 0
        stretch = np.cumsum(signal & ~np.r_[False, signal[:-1]]) * signal
        stretch = np.where(signal, np.bincount(stretch)[stretch], 0)
        # Maxima above twice the local baseline, or, where isomers elute back to back and the baseline median sits on their peaks, maxima above it that
        # are well resolved: GPST000239 Hex4HexNAc2 [M-H]- at 54.98 (valley 0.31 towards the given isomer at 56.14, 9 MS2) lies below twice its baseline
        candidates = (s > np.r_[-1, s[:-1]]) & (s >= np.r_[s[1:], -1]) & (s > 0) & (s >= baseline)
        for i in np.flatnonzero(candidates & ((stretch >= 3) if self.cycle <= SLOW_CYCLE else True)):
            higher_l, higher_r = np.flatnonzero(s[:i] > s[i]), i + 1 + np.flatnonzero(s[i + 1:] > s[i])
            valley, neighbour = 0.0, None
            for j in ([higher_l[-1]] if len(higher_l) else []) + ([higher_r[0]] if len(higher_r) else []):
                v = float(s[min(i, j):max(i, j) + 1].min() / s[i])
                if v >= valley:
                    valley, neighbour = v, j
            if s[i] < MIN_OVER_BASELINE * baseline[i] and valley >= VALLEY_SAME_MS2:
                continue
            if neighbour is not None:
                while 0 < neighbour < len(s) - 1 and max(s[neighbour - 1], s[neighbour + 1]) > s[neighbour]:
                    neighbour += 1 if s[neighbour + 1] > s[neighbour - 1] else -1
            below_l, below_r = np.flatnonzero(s[:i] <= s[i] / 2), i + 1 + np.flatnonzero(s[i + 1:] <= s[i] / 2)
            # Out to the survey scans where the signal has fallen to half: MS2 between the last scan above half height and that one were triggered on the
            # peak (GPST000054 1177.43 at 42.37: its MS2 at 42.60, 0.003 min after the midpoint the window used to end at)
            t0 = rts[below_l[-1]] if len(below_l) else rts[0]
            t1 = rts[below_r[0]] if len(below_r) else rts[-1]
            # The half-height window ends at the valley towards a higher point: a small maximum before a long stretch of higher signal (GPST000054
            # Hex5HexNAc2 [M-H]- at 15.65, 46k, before the 0.2-42M of 17-57 min) otherwise reached to 57.5 min and took the MS2 of Man5 at 27.3
            if len(higher_l):
                t0 = max(t0, rts[higher_l[-1] + int(np.argmin(s[higher_l[-1]:i + 1]))])
            if len(higher_r):
                t1 = min(t1, rts[i + int(np.argmin(s[i:higher_r[0] + 1]))])
            # MS2 triggered by the apex survey scan come up to one cycle after it
            nxt = rts[min(i + 1, len(rts) - 1)]
            out.append({'i': int(i), 'RT': float(rts[i]), 'height': float(s[i]), 'baseline': float(baseline[i]), 'valley': valley, 'neighbour': neighbour,
                        'window': (float(t0) - 0.05, float(max(t1, nxt)) + 0.05)})
        return out


def glycan_fragments(run, hits, comp, z, precursor, cc_kwargs):
    """how many of the TOP_FRAGMENTS most intense deisotoped fragments of the pooled MS2 (outside the precursor envelope) CandyCrumbs explains as
    fragments of the composition, how many there were, the share of their intensity that is explained, and the share of the pooled signal outside
    the precursor envelope (how much the precursor fragmented at all)"""
    pooled = Counter()
    for k in hits:
        d = run.peak_d[k]
        total = sum(d.values()) or 1
        pooled.update({round(m, 1): i / total for m, i in d.items()})
    # Named by the most intense peak of each pattern whatever the number of spectra: in the noise-rich centroids of Bruker ion traps, peaks 1-2 Da below a
    # fragment claimed it as their isotope (GPST000239 Gb3 505.18: 161.04, 263.04 and 343.14 became 159.1, 261.0 and 340.2, 1 of 10 annotated instead of 6)
    deisotoped = deisotope_ms2(dict(pooled), z, 0.2, relabel = True)
    frags = [m for m in deisotoped if not -0.6 <= m - precursor <= 3.6 / z][:TOP_FRAGMENTS]
    fragmentation = sum(i for m, i in pooled.items() if not -0.6 <= m - precursor <= 3.6 / z) / (sum(pooled.values()) or 1)
    if not frags:
        return 0, 0, 0.0, fragmentation, float(np.mean(run.tic_pct[list(hits)]))
    # Positive-mode glycans are mostly metal adducts ([M+K+H]2+ in GPST000134) whose fragments keep the cation, so global modifications count there
    out = CandyCrumbs(comp, [round(m, 1) for m in frags], cc_kwargs['tolerance'], charge = (-z if run.mode == 'negative' else z), simplify = False,
                      disable_global_mods = run.mode == 'negative', disable_X_cross_rings = True, disable_A_cross_rings = run.mode == 'positive', max_cleavages = 2,
                      mass_tag = cc_kwargs['mass_tag'], sample_prep = cc_kwargs['sample_prep'])
    explained = sum(deisotoped[m] for m in frags if out.get(round(m, 1)))
    return (sum(bool(v) for v in out.values()), len(frags), explained / (sum(deisotoped[m] for m in frags) or 1), fragmentation,
            float(np.mean(run.tic_pct[list(hits)])))


def fragment_verdict(f):
    """'yes' with at least GLYCAN_FRAGMENTS annotated fragments or one carrying half the fragment intensity (sparse ion-trap MS2 such as the lone Neu5Ac
    loss of GPST000029 1331.69); 'inconclusive' if the precursor barely fragmented, or for a weak MS2 (below the run's median signal) with an annotated
    fragment (GPST000029 952.47/895.45 open peaks); else 'no' (the strong non-glycan MS2 of GPST000316 513.27/530.17 in the wash)"""
    if f[0] >= GLYCAN_FRAGMENTS or (f[0] >= 1 and f[2] >= 0.5):
        return 'yes'
    return 'inconclusive' if f[3] < MIN_FRAGMENTATION or (f[0] >= 1 and f[4] < 0.5) else 'no'


def fragments_ok(f):
    return fragment_verdict(f) != 'no'


def describe_fragments(f):
    return (f'{f[0]} of {f[1]} top fragments annotated' + (f', {100 * f[3]:.0f}% of the MS2 signal outside the precursor' if f[3] < MIN_FRAGMENTATION else '')
            + (f', weak MS2 (signal percentile {100 * f[4]:.0f} of the run)' if f[4] < 0.5 and fragment_verdict(f) != 'yes' else ''))


def audit(spectra_path, gt_path, glycan_class, run_name = None, modification = 'reduced', mass_tag = 0.0, sample_prep = 'underivatized',
          predictions = None, rt_window = 1.5, library_cache = None, data_dir = DATA_DIR, lc = 'PGC', mode = None, max_charge = 3):
    """audits a source GT against its run; returns the audited GT, the removed rows (with reasons), the review list, and a summary text"""
    t_start = time.time()
    spectra_path, gt_path = Path(spectra_path), Path(gt_path)
    run_name = run_name or Path(str(spectra_path).rstrip('/\\')).stem
    run = Run(spectra_path, mode)
    charges = tuple(range(1, max_charge + 1))
    mod = None if modification in (None, 'custom', 'unreduced') else modification
    sign = -1 if run.mode == 'negative' else 1
    cc_kwargs = {'tolerance': 0.05 if run.ms2_accurate else 0.4, 'mass_tag': modification_mass_dict.get(mod, 0) + mass_tag, 'sample_prep': sample_prep}
    src = pd.read_csv(gt_path, encoding = 'utf-8-sig')
    rt_col = f'{run_name}_RT' if f'{run_name}_RT' in src.columns else 'RT'
    # Abundance: the run's own column; a constant placeholder column (GPST000316's '_') or none means the apex heights (% of their sum) are written
    abund_col = run_name if run_name in src.columns else None
    known = set(STANDARD_COLUMNS + AUDIT_COLUMNS + [rt_col])
    if abund_col is None:
        others = [c for c in src.columns if c not in known and pd.api.types.is_numeric_dtype(src[c]) and not c.endswith('_RT')]
        if len(others) == 1 and src[others[0]].nunique() > 1:
            abund_col = others[0]
    placeholder = [c for c in src.columns if abund_col is None and c not in known and pd.api.types.is_numeric_dtype(src[c]) and src[c].notna().all()
                   and src[c].nunique() <= 1]
    # Other runs' columns (<run>, <run>_RT) stay in the source; the audited GT is this run's
    extra_cols = [c for c in src.columns if c not in known and c != abund_col and c not in placeholder and not c.endswith('_RT') and f'{c}_RT' not in src.columns]
    removed, review = [], []
    def flag(row, category, detail):
        review.append({'Mass': row.get('Mass'), 'charge': row.get('charge'), 'RT': row.get('RT'), 'composition': row.get('composition'),
                       'glycan': row.get('glycan'), 'category': category, 'detail': detail})
    # Compositions and the charge each row's Mass belongs to; charges the source gives below CHARGE_MIN_MASS are kept per composition
    rows, given_charges = [], {}
    for idx, r in src.iterrows():
        glycan = r['glycan'].strip() if isinstance(r.get('glycan'), str) and r['glycan'].strip() else ''
        row = {'idx': idx, 'Mass_source': float(r['Mass']), 'RT_source': float(r[rt_col]) if pd.notna(r.get(rt_col)) else np.nan, 'glycan_source': glycan,
               'glycan': glycan, 'abundance': r[abund_col] if abund_col else np.nan, 'extra': {c: r[c] for c in extra_cols}, 'notes': []}
        given_z = abs(int(r['charge'])) if 'charge' in src.columns and pd.notna(r.get('charge')) else None
        if np.isnan(row['RT_source']):
            removed.append({**row, 'reason': f'no RT for run {run_name} in the source'})
            continue
        if abund_col and pd.notna(row['abundance']) and float(pd.to_numeric(row['abundance'], errors = 'coerce') or 0) == 0:
            row['absent'] = True
        comp = composition_of(glycan) if glycan else None
        if glycan and comp is None:
            row['notes'].append(f'glycowork cannot read the given structure {glycan}')
            flag({'Mass': row['Mass_source'], 'RT': row['RT_source'], 'glycan': glycan}, 'unreadable structure', 'composition taken from the Mass instead')
            row['glycan'] = ''
        options = []
        if comp is not None:
            fits = sorted((abs(mz - row['Mass_source']), z) for _, z, mz, prot in ion_list(comp, run.mode, mod, mass_tag, sample_prep, extra_z = (given_z,)) if prot)
            fits = [f for f in fits if given_z is None or f[1] == given_z] or fits
            if fits and fits[0][0] <= 0.5:
                options = [(comp, fits[0][1], fits[0][0])]
            else:
                adduct = [n for n, _, mz, prot in ion_list(comp, run.mode, mod, mass_tag, sample_prep, adducts = True) if not prot and abs(mz - row['Mass_source']) <= 0.5]
                if adduct:
                    removed.append({**row, 'composition': comp_string(comp), 'reason': f'the Mass is the {adduct[0]} adduct of the given structure, not a glycan '
                                                                                        f'peak of its own'})
                    continue
                row['notes'].append(f'given structure ({comp_string(comp)}) does not fit Mass {row["Mass_source"]:.2f}')
                row['glycan'], comp = '', None
        if comp is None:
            # An open row takes the composition the source names (each of a '/' isobar group, as the GPST000403 GT writes them) where it fits the Mass at
            # the given charge: by mass glycowork finds others first or only in another class (GPST000316's Man7 Hex7HexNAc2 [M-2H]2- in an O run became
            # Hex4HexNAc4S1), and it cannot use the reading the source made from the MS2
            for text in (r['composition'].split('/') if isinstance(r.get('composition'), str) else []):
                try:
                    c = canonicalize_composition(text.strip())
                except Exception:
                    continue
                fits = sorted((abs(mz - row['Mass_source']), z) for _, z, mz, prot in ion_list(c, run.mode, mod, mass_tag, sample_prep, extra_z = (given_z,))
                              if prot and given_z in (None, z))
                if fits and fits[0][0] <= 0.5:
                    options.append((c, fits[0][1], fits[0][0]))
            # Open rows of an unexpected class (O-glycan runs carry N-glycans) take compositions of any class when their own has none
            options = options or compositions_for_mz(row['Mass_source'], run.mode, mod, mass_tag, sample_prep, glycan_class, charges = charges) or \
                      compositions_for_mz(row['Mass_source'], run.mode, mod, mass_tag, sample_prep, 'all', charges = charges)
            if not options:
                flag({'Mass': row['Mass_source'], 'RT': row['RT_source'], 'glycan': row['glycan_source']}, 'no composition',
                     f'no glycowork composition fits the Mass at z = 1-{max_charge}, so the row could not be audited (removed)')
                removed.append({**row, 'reason': 'no composition of glycan class ' + glycan_class + (' fits the Mass' if not row['glycan_source'] else
                                                  ' fits the Mass, and the given structure does not')})
                continue
            if row['glycan_source'] and row['glycan_source'] != row['glycan'] or len(row['notes']):
                flag({'Mass': row['Mass_source'], 'RT': row['RT_source'], 'glycan': row['glycan_source']}, 'structure contradicts mass',
                     f'{"; ".join(row["notes"])}; the Mass fits {", ".join(comp_string(c) + f" (z = {z})" for c, z, _ in options)}, structure left blank')
        row['options'] = options
        if given_z:
            for c, z, _ in options:
                given_charges.setdefault(comp_key(c), set()).add(z)
        rows.append(row)
    # Calibration: the most intense MS1 centroid near the theoretical m/z of each composition (singly charged ions in ion traps, as they merge the isotopes
    # of the others) at the RT of each MS2 spectrum triggered on it
    ions_of = {}
    for row in rows:
        for comp, _, _ in row['options']:
            ions_of.setdefault(comp_key(comp), (comp, [x for x in ion_list(comp, run.mode, mod, mass_tag, sample_prep, extra_z = tuple(given_charges.get(comp_key(comp), ())))
                                                       if run.mz_range[0] - 1 <= x[2] <= run.mz_range[1] + 1]))
    targets = [(mz, run.rt[k]) for comp, ions in ions_of.values() for _, z, mz, prot in ions if prot and (z == 1 or run.accurate)
               for k in np.flatnonzero(np.abs(run.prec - mz) <= (0.02 if run.accurate else 0.5))]
    if run.ms1 is not None and len(targets) >= 5:
        run.shift, run.half_width = ms1_mz_calibration(run.ms1, [t[0] for t in targets], [t[1] for t in targets], np.ones(len(targets)),
                                                        mz_tolerance = 0.02 if run.accurate else 0.5)
    else:
        diffs = [run.prec[k] - mz for mz, rt in targets for k in np.flatnonzero((np.abs(run.prec - mz) <= 0.5) & (run.rt == rt))]
        run.shift = float(np.median(diffs)) if len(diffs) >= 5 else 0.0
    shift = run.shift
    analysed = {}
    def analyse(comp):
        """envelope XIC of a composition summed over its ions, its maxima with their MS2, pooled MS2 vectors, glycan fragment counts, and tail flags"""
        key = comp_key(comp)
        if key in analysed:
            return analysed[key]
        if key not in ions_of:
            ions_of[key] = (comp, [x for x in ion_list(comp, run.mode, mod, mass_tag, sample_prep, extra_z = tuple(given_charges.get(key, ())))
                                   if run.mz_range[0] - 1 <= x[2] <= run.mz_range[1] + 1])
        ions = ions_of[key][1]
        # Precursors of each ion: refined ion-trap precursors of z >= 2 sit up to 0.4 above M (merged isotopes); accurate MS1 also takes 13C triggers
        ion_hits = []
        for _, z, mz, _ in ions:
            d = run.prec - mz - shift
            if run.accurate:
                tol = max(0.01, 2e-5 * mz)
                ion_hits.append(np.flatnonzero((np.abs(d) <= tol) | (np.abs(d - ISOTOPE_SPACING / z) <= tol)))
            else:
                ion_hits.append(np.flatnonzero((d >= -0.3) & (d <= (0.3 if z == 1 or run.resolved else 0.4))))
        if run.ms1 is not None:
            traces = [run.envelope(mz, z) for _, z, mz, _ in ions]
            smoothed = [run.smooth(t) for t in traces]
            total = np.sum(smoothed, axis = 0) if smoothed else np.zeros(len(run.rts))
            # Maxima are found on each ion's own trace (summed, the noise of an empty charge state blurred the valleys of the other), and maxima of
            # several ions within 1.5 MS1 cycles are one peak, judged on its most intense ion that has MS2 inside its own maximum (else its most intense ion):
            # at unit resolution the z = 3 window of an N-glycan catches the isotopes of co-eluting ions, which made 519.18 [M-3H]3- the peak of the Man7
            # [M-2H]2- precursors at 779.32 (GPST000054) and 499.17 [M-2H]2- that of 999.39 [M-H]- (GPST000239)
            peaks = []
            with_ms2 = lambda p, k: any(p['window'][0] <= run.rt[h] <= p['window'][1] for h in ion_hits[k])
            for p, k in sorted(((p, k) for k, t in enumerate(traces) for p in run.maxima(t)), key = lambda x: (not with_ms2(*x), -x[0]['height'])):
                # A maximum of an ion that was not fragmented there while another ion of the composition was is no peak of its own: the fragmented ion's
                # trace has to show it (499.17 [M-2H]2- at 43.75 under 999.35 [M-H]- in GPST000239; 526.51 [M-3H]3- at 29.81 on the tail of the 790.27
                # [M-2H]2- peak at 28.90)
                if not with_ms2(p, k) and any(with_ms2(p, m) for m in range(len(ions)) if m != k):
                    continue
                same = next((q for q in peaks if abs(q['RT'] - p['RT']) <= max(0.1, 1.5 * run.cycle)), None)
                if same is None:
                    p['dominant'] = k
                    peaks.append(p)
                else:
                    same['window'] = (min(same['window'][0], p['window'][0]), max(same['window'][1], p['window'][1]))
        else:
            # Without MS1 the peaks are clusters of the composition's MS2 precursors (split at gaps above 0.3 min), at their most intense precursor
            total, smoothed, peaks = None, [], []
            hits_all = np.unique(np.concatenate(ion_hits)) if ion_hits else np.zeros(0, int)
            hits_all = hits_all[np.argsort(run.rt[hits_all])]
            for cl in np.split(hits_all, np.flatnonzero(np.diff(run.rt[hits_all]) > 0.3) + 1) if len(hits_all) else []:
                inten = np.nan_to_num(run.intensity[cl], nan = 1.0)
                peaks.append({'i': None, 'RT': float(run.rt[cl[np.argmax(inten)]]), 'height': float(inten.max()), 'baseline': 0.0, 'valley': 0.0,
                              'neighbour': None, 'window': (float(run.rt[cl].min()) - 0.05, float(run.rt[cl].max()) + 0.05), 'dominant': None})
        for p in peaks:
            w0, w1 = p['window']
            p['ms2'] = {k: [h for h in hits if w0 <= run.rt[h] <= w1] for k, hits in enumerate(ion_hits)}
            p['ms2'] = {k: h for k, h in p['ms2'].items() if h}
            p['n_ms2'] = sum(len(h) for h in p['ms2'].values())
            if p['dominant'] is None:
                p['dominant'] = max(range(len(ions)), key = lambda k: len(p['ms2'].get(k, [])))
            hits = sorted(h for v in p['ms2'].values() for h in v)
            p['vec'] = pooled_vector([run.peak_d[h] for h in hits], ions[p['dominant']][2] + shift, ions[p['dominant']][1]) if hits else None
            p['fragments'] = None
        for p in peaks:
            # A valley between 0.5 and 0.7 only separates two peaks whose MS2 differ (cosine < 0.9); the same MS2 on both sides is one compound
            p['valley_limit'] = VALLEY
            nb = min(peaks, key = lambda q: abs(q['i'] - p['neighbour'])) if p['neighbour'] is not None else None
            nb = nb if nb is not None and abs(nb['i'] - p['neighbour']) <= 2 else None
            if nb is not None and p['vec'] is not None and nb['vec'] is not None and float(p['vec'] @ nb['vec']) >= SAME_MS2:
                p['valley_limit'] = VALLEY_SAME_MS2
            p['distinct'] = p['valley'] < p['valley_limit']
            p['tail_of'] = None
        # A PGC tail: later, at most half as high on the same ion, never back below TAIL_FLOOR of its own height in between (deep valleys separate
        # isomers eluting after each other, e.g. GPST000029 829.45 at 25.34/25.92/26.44 after 23.2), and with the apex's MS2
        for p in peaks:
            for q in peaks:
                if lc != 'PGC' or not (0 < p['RT'] - q['RT'] <= TAIL_RT) or p['vec'] is None or q['vec'] is None:
                    continue
                k = p['dominant']
                hq, hp, between = (smoothed[k][q['i']], smoothed[k][p['i']], smoothed[k][q['i']:p['i'] + 1].min()) if smoothed else (q['height'], p['height'], p['height'])
                cos = float(p['vec'] @ q['vec'])
                if hq >= TAIL_RATIO * hp and between >= TAIL_FLOOR * hp and cos >= TAIL_MS2 and (p['tail_of'] is None or hq > p['tail_of'][2]):
                    p['tail_of'] = (q, cos, hq)
        analysed[key] = {'comp': comp, 'ions': ions, 'peaks': peaks, 'total': total, 'smoothed': smoothed}
        return analysed[key]
    def fragments(a, p):
        if p['fragments'] is None:
            k = max(p['ms2'], key = lambda k: len(p['ms2'][k]))
            _, z, mz, _ = a['ions'][k]
            p['fragments'] = glycan_fragments(run, p['ms2'][k], a['comp'], z, mz + shift, cc_kwargs)
        return p['fragments']
    def ms2_elsewhere(a, expected, none):
        """the MS2 of a composition near an RT that lie outside its MS1 peaks, for removal reasons"""
        inside = {h for p in a['peaks'] for v in p['ms2'].values() for h in v}
        out = []
        for _, _, mz, _ in a['ions']:
            out += [h for h in np.flatnonzero((np.abs(run.prec - mz - shift) <= 0.3) & (np.abs(run.rt - expected) <= rt_window)) if h not in inside]
        if not out:
            return none
        return (f' ({len(out)} MS2 of it outside its MS1 peaks, at {run.rt[out].min():.2f}-{run.rt[out].max():.2f} min, precursors '
                f'{run.prec[out].min():.2f}-{run.prec[out].max():.2f})')
    def good(a, p):
        return p['distinct'] and p['n_ms2'] > 0 and p['tail_of'] is None and fragments_ok(fragments(a, p))
    # Open rows with several fitting compositions take the one with a good peak near their RT, then one that other rows of the GT have, then the closest
    # in mass (the others go to the review list as isobars)
    given_comps = {comp_key(r['options'][0][0]) for r in rows if r['glycan'] and r['options']}
    for row in rows:
        scored = []
        for comp, z, err in row['options']:
            a = analyse(comp)
            near = [abs(p['RT'] - row['RT_source']) for p in a['peaks'] if abs(p['RT'] - row['RT_source']) <= rt_window + 2 and good(a, p)]
            scored.append((0 if near else 1, comp_key(comp) not in given_comps, min(abs(err - shift), err), min(near) if near else 0, comp, z))
        scored.sort(key = lambda x: x[:4])
        row['comp'], row['given_z'] = scored[0][4], scored[0][5]
        # Compositions within 0.15 m/z of each other at one charge cannot be told apart in an ion trap (Hex5 and Hex2HexNAc2S1)
        mz_of = {comp_key(c): next(m for _, zz, m, pr in ion_list(c, run.mode, mod, mass_tag, sample_prep) if pr and zz == z) for c, z, _ in row['options']}
        rivals = [s for s in scored[1:] if s[0] == 0 and s[5] == scored[0][5] and abs(mz_of[comp_key(s[4])] - mz_of[comp_key(scored[0][4])]) <= (0.02 if run.accurate else 0.15)]
        if scored[0][0] == 0 and rivals:
            row['notes'].append('isobaric compositions ' + ', '.join(comp_string(s[4]) for s in [scored[0]] + rivals))
    # Copies: rows of one composition at another charge and the same source RT are one glycan peak
    entities = []
    for row in sorted(rows, key = lambda r: (r['RT_source'], -bool(r['glycan']))):
        same = next((e for e in entities if comp_key(e[0]['comp']) == comp_key(row['comp']) and abs(e[0]['RT_source'] - row['RT_source']) <= COPY_RT
                     and all(x['given_z'] != row['given_z'] for x in e)), None)
        if same is None:
            entities.append([row])
        else:
            same.append(row)
    for e in entities:
        e.sort(key = lambda r: (not r['glycan'], r.get('absent', False)))
    # RT shift: median of (apex - source RT) over rows with exactly one good peak within 3 min that no other row has as its only candidate; locally (rows
    # within 10 min) where at least 3 such rows exist, as shifts drift along the gradient (GPST000374 N: 0 up to 20 min, then -0.4 to -1)
    cands = []
    for e in entities:
        a = analyse(e[0]['comp'])
        cands.append([p for p in a['peaks'] if abs(p['RT'] - e[0]['RT_source']) <= max(3.0, rt_window) and good(a, p)])
    taken = Counter(id(c[0]) for c in cands if len(c) == 1)
    unique = [(e[0]['RT_source'], c[0]['RT'] - e[0]['RT_source']) for e, c in zip(entities, cands) if len(c) == 1 and taken[id(c[0])] == 1]
    global_shift = float(np.median([d for _, d in unique])) if len(unique) >= 2 else 0.0
    def rt_shift(t):
        local = [d for s, d in unique if abs(s - t) <= 10]
        return float(np.median(local)) if len(local) >= 3 else global_shift
    # Assignment per composition: Hungarian on |apex - expected RT| with penalties for peaks that fail a criterion, so a row prefers a good peak
    by_comp = {}
    for e in entities:
        by_comp.setdefault(comp_key(e[0]['comp']), []).append(e)
    assigned = {}
    for ents in by_comp.values():
        a = analyse(ents[0][0]['comp'])
        peaks = a['peaks']
        if not peaks:
            continue
        cost = np.full((len(ents), len(peaks)), 1e6)
        for i, e in enumerate(ents):
            expected = e[0]['RT_source'] + rt_shift(e[0]['RT_source'])
            for j, p in enumerate(peaks):
                d = abs(p['RT'] - expected)
                if d <= rt_window:
                    cost[i, j] = d + 5 * e[0].get('absent', False) + 3 * (not p['distinct']) + 2 * (p['n_ms2'] == 0) + 2 * (p['tail_of'] is not None) + \
                                 (2 * (not fragments_ok(fragments(a, p))) if p['n_ms2'] else 0) - 1e-3 * np.log1p(p['height'])
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                assigned[id(ents[i][0])] = (a, peaks[j], cost[i, j])
    kept, used_peaks = [], {}
    for e in entities:
        row = e[0]
        a = analyse(row['comp'])
        expected = row['RT_source'] + rt_shift(row['RT_source'])
        comp_s = comp_string(row['comp'])
        copy_of = f'of the row at {row["Mass_source"]:.2f}/{row["RT_source"]:.2f}'
        if id(row) not in assigned:
            others = [p for p in a['peaks'] if abs(p['RT'] - expected) <= rt_window and good(a, p)]
            owner = next((r for r in kept if others and r['peak'] is others[0]), None)
            if owner:
                reason = f'its only good peak (apex {others[0]["RT"]:.2f}) went to the row at {owner["Mass"]:.2f}/{owner["RT"]:.2f}'
            elif others:
                reason = f'no free peak (apex {others[0]["RT"]:.2f} taken)'
            elif run.ms1 is not None:
                reason = f'no MS1 peak of {comp_s} within {rt_window} min of {expected:.2f}' + ms2_elsewhere(a, expected, ' and no MS2 of it there')
            else:
                reason = f'no MS2 of {comp_s} within {rt_window} min of {expected:.2f}'
            removed += [{**r, 'composition': comp_s, 'reason': reason if r is row else f'copy (z = {r["given_z"]}) {copy_of}; {reason}'} for r in e]
            continue
        _, p, _ = assigned[id(row)]
        k_dom = p['dominant']
        name, z, mz, prot = a['ions'][k_dom]
        prot_mz = next((m for n, zz, m, pr in a['ions'] if pr and zz == z), mz)
        here = {'Mass': prot_mz, 'charge': sign * z, 'RT': p['RT'], 'composition': comp_s, 'glycan': row['glycan']}
        reason = None
        if not p['distinct']:
            reason = (f'shoulder of the larger peak at {run.rts[p["neighbour"]]:.2f} (valley {p["valley"]:.2f} of its apex at {p["RT"]:.2f}, '
                      f'limit {p["valley_limit"]})')
        elif p['n_ms2'] == 0:
            reason = f'no MS2 at its MS1 peak (apex {p["RT"]:.2f})' + ms2_elsewhere(a, expected, '')
        elif p['tail_of'] is not None:
            q, c, hq = p['tail_of']
            reason = f'PGC tail of the apex at {q["RT"]:.2f} ({p["height"] / hq:.2f} of its height, MS2 cos {c:.2f})'
        elif not fragments_ok(fragments(a, p)):
            reason = f'MS2 at its peak ({p["n_ms2"]} spectra at {p["RT"]:.2f}) without glycan fragments of {comp_s} ({describe_fragments(fragments(a, p))})'
        # Isotope pick: the envelope of a co-eluting lighter ion one isotope spacing below explains the signal (966.35 at the 965.84 peaks of GPST000169).
        # Where isotopes are resolved, the monoisotopic signal is compared with the one an isotope spacing lower; in Thermo ion traps the whole
        # envelope (which holds the lighter ion's isotopes) with the lower window
        iso_note = ''
        if run.ms1 is not None and p['i'] is not None:
            # Judged on the dominant ion or, where that has no monoisotopic signal at the apex, on the composition's ion with the strongest there: the
            # dominant ion (most MS2) can be an adduct or another charge state (GPST000548 A3G3S3 [M+4H]4+ 899.96 at 40.42, 1.3e7, was judged on its
            # empty [M+Na+2H]3+ and removed)
            def mono(k):
                _, zk, mzk, _ = a['ions'][k]
                return a['smoothed'][k][p['i']] if not (run.accurate or (run.resolved and zk <= 2)) else run.smooth(run.envelope(mzk, zk, 0))[p['i']]
            k_iso = k_dom if mono(k_dom) > 0 else max(range(len(a['ions'])), key = mono)
            _, z_iso, mz_iso, _ = a['ions'][k_iso]
            spacing = ISOTOPE_SPACING / z_iso
            merged = not (run.accurate or (run.resolved and z_iso <= 2))
            lower = run.smooth(run.envelope(mz_iso - spacing, z_iso, 0))[p['i']]
            own = mono(k_iso)
            if own <= 0:
                reason = (reason + '; ' if reason else '') + 'no monoisotopic signal at its apex (the envelope belongs to other ions)'
            elif lower >= (0.5 if merged else 0.7) * own:
                iso_note = f'the envelope may be the isotope pattern of a co-eluting ion at m/z {mz_iso + shift - spacing:.2f} ({lower / own:.2f} of its signal)'
                if reason is None:
                    flag(here, 'possible isotope pick', iso_note)
                else:
                    reason += f'; {iso_note}'
        if reason is not None:
            removed += [{**r, 'composition': comp_s, 'reason': reason if r is row else f'copy (z = {r["given_z"]}) {copy_of}; {reason}'} for r in e]
            continue
        removed += [{**r, 'composition': comp_s, 'reason': f'copy (z = {r["given_z"]}) {copy_of} (same peak, apex {p["RT"]:.2f})'} for r in e[1:]]
        if row.get('absent'):
            flag(here, 'absent in source but found', f'the source gives abundance 0 for this run, but a peak with MS2 exists at {p["RT"]:.2f}; listed under removed')
            removed.append({**row, 'composition': comp_s, 'reason': f'abundance 0 in the source for this run (a peak with MS2 exists at {p["RT"]:.2f}, see review)'})
            continue
        f = fragments(a, p)
        ms2_mz = sorted(run.prec[h] for v in p['ms2'].values() for h in v)
        valley = p['neighbour'] is not None and abs(run.rts[p['neighbour']] - p['RT']) <= TAIL_RT
        note = (f'{"distinct envelope peak" if run.ms1 is not None else "MS2 cluster (no MS1)"} (apex {p["RT"]:.2f}, {name}'
                + (f', valley {p["valley"]:.2f} towards the peak at {run.rts[p["neighbour"]]:.2f}' if valley else '')
                + f') with {p["n_ms2"]} MS2 at it (precursors {ms2_mz[0]:.2f}-{ms2_mz[-1]:.2f}; {describe_fragments(f)})')
        if abs(p['RT'] - row['RT_source']) > 0.05:
            note += f'; source RT {row["RT_source"]:.2f}'
        note += ''.join(f'; {x}' for x in row['notes'] + ([iso_note] if iso_note else []))
        note += ''.join(f'; z = {r["given_z"]} copy at {r["Mass_source"]:.2f} dropped' for r in e[1:])
        free = [q for q in a['peaks'] if q is not p and good(a, q) and abs(q['RT'] - expected) <= abs(p['RT'] - expected) + 0.3
                and q['height'] >= 0.3 * p['height'] and not any(x[1] is q for x in assigned.values())]
        if free:
            alt = min(free, key = lambda q: abs(q['RT'] - expected))
            flag(here, 'ambiguous peak', f'source RT {row["RT_source"]:.2f} (expected {expected:.2f}); the free peak at {alt["RT"]:.2f} ({alt["n_ms2"]} MS2, '
                                         f'height {alt["height"] / p["height"]:.2f}x) fits about as well')
        if fragment_verdict(f) == 'inconclusive':
            flag(here, 'weak MS2 evidence', f'kept, but its MS2 neither shows nor rules out glycan fragments ({describe_fragments(f)})')
        if abs(p['RT'] - expected) > 0.3:
            flag(here, 'far from source RT', f'apex {p["RT"]:.2f} is {p["RT"] - expected:+.2f} min from the expected RT {expected:.2f} (source {row["RT_source"]:.2f})')
        if any('isobaric' in x for x in row['notes']):
            flag(here, 'isobar', '; '.join(row['notes']))
        kept.append({**here, 'row': row, 'comp': row['comp'], 'source': 'given', 'peak': p, 'analysis': a, 'note': note, 'abundance': row['abundance'], 'z': z})
        used_peaks[(comp_key(row['comp']), id(p))] = kept[-1]
    # Additions from a CandyCrunch output of this run: unmatched predictions whose peak passes the same criteria
    n_pred = 0
    if predictions is not None:
        pred = pd.read_excel(predictions) if str(predictions).lower().endswith(('.xlsx', '.xls')) else pd.read_csv(predictions)
        mz_col = 'm/z' if 'm/z' in pred.columns else pred.columns[0]
        pred = pred.sort_values('rel_abundance', ascending = False) if 'rel_abundance' in pred.columns else pred
        for _, r in pred.iterrows():
            pmz, prt = float(r[mz_col]), float(r['RT'])
            pz = abs(int(r['charge'])) if 'charge' in pred.columns and pd.notna(r.get('charge')) else 1
            # Matched like the test metric: within MATCH_MZ as m/z or as neutral mass (another charge state), and MATCH_RT
            neutral = pmz * pz - sign * pz * PROTON_MASS
            if any(abs(k['RT'] - prt) <= MATCH_RT and min(abs(k['Mass'] - pmz), abs(k['Mass'] * k['z'] - sign * k['z'] * PROTON_MASS - neutral)) <= MATCH_MZ
                   for k in kept):
                continue
            n_pred += 1
            comp, top1 = None, r.get('top1_pred') if isinstance(r.get('top1_pred'), str) else ''
            if isinstance(r.get('composition'), str) and r['composition'].strip():
                text = r['composition'].strip()
                try:
                    comp = canonicalize_composition(ast.literal_eval(text) if text.startswith('{') else text)
                except Exception:
                    comp = None
            comp = comp or (composition_of(top1) if top1 else None)
            if comp is None:
                alts = compositions_for_mz(pmz, run.mode, mod, mass_tag, sample_prep, glycan_class, charges = (pz,))
                comp = alts[0][0] if alts else None
            entry = {'Mass_source': pmz, 'RT_source': prt, 'glycan_source': top1, 'source': 'prediction', 'composition': comp_string(comp) if comp else ''}
            if comp is None:
                removed.append({**entry, 'reason': 'prediction not added: no composition'})
                continue
            a = analyse(comp)
            near = sorted((p for p in a['peaks'] if abs(p['RT'] - prt) <= 0.5 and p['distinct'] and p['n_ms2']), key = lambda p: abs(p['RT'] - prt))
            if not near:
                removed.append({**entry, 'reason': f'prediction not added: no distinct {"MS1 peak" if run.ms1 is not None else "MS2 cluster"} with MS2 within 0.5 min'})
                continue
            p = near[0]
            if (comp_key(comp), id(p)) in used_peaks:
                continue
            if p['tail_of'] is not None:
                removed.append({**entry, 'reason': f'prediction not added: PGC tail of the apex at {p["tail_of"][0]["RT"]:.2f} (MS2 cos {p["tail_of"][1]:.2f})'})
                continue
            # Additions need positive glycan evidence; weak or unfragmented MS2 that only fail to rule a glycan out are not enough
            f = fragments(a, p)
            if fragment_verdict(f) != 'yes':
                removed.append({**entry, 'reason': f'prediction not added: {describe_fragments(f)} as {comp_string(comp)}'})
                continue
            name, z, mz, prot = a['ions'][p['dominant']]
            prot_mz = next((m for n, zz, m, pr in a['ions'] if pr and zz == z), mz)
            ms2_mz = sorted(run.prec[h] for v in p['ms2'].values() for h in v)
            valley = p['neighbour'] is not None and abs(run.rts[p['neighbour']] - p['RT']) <= TAIL_RT
            note = (f'{comp_string(comp)} {name}, distinct {"envelope peak" if run.ms1 is not None else "MS2 cluster (no MS1)"} at {p["RT"]:.2f}'
                    + (f' (valley {p["valley"]:.2f})' if valley else '') + f' with {p["n_ms2"]} MS2 (precursors {ms2_mz[0]:.2f}-{ms2_mz[-1]:.2f}; '
                    f'{describe_fragments(f)}); from the prediction at {pmz:.2f}/{prt:.2f}')
            kept.append({'row': None, 'Mass': prot_mz, 'charge': sign * z, 'RT': p['RT'], 'glycan': '', 'composition': comp_string(comp), 'comp': comp,
                         'source': 'added', 'peak': p, 'analysis': a, 'note': note, 'abundance': np.nan, 'z': z, 'prediction': top1})
            used_peaks[(comp_key(comp), id(p))] = kept[-1]
    # Library: structures for open and added rows; a note for given structures
    library = build_library(data_dir, library_cache) if data_dir else []
    exclude = {run_name} | ({gt_path.resolve().parent.name} if gt_path.resolve().parent.parent == Path(data_dir).resolve() else set())
    for k in kept:
        p = k['peak']
        k_ms2 = max(p['ms2'], key = lambda j: len(p['ms2'][j]))
        hits, (_, z_lib, mz_lib, _) = p['ms2'][k_ms2], k['analysis']['ions'][k_ms2]
        matches = library_match(library, k['comp'], z_lib, mod, pooled_vector([run.peak_d[h] for h in hits], mz_lib + shift, z_lib), exclude)
        if not matches:
            if not k['glycan']:
                k['note'] += '; open: no library row of this composition and charge'
            continue
        best, second = matches[0], (matches[1] if len(matches) > 1 else (0.0, '', set()))
        where = ', '.join(sorted(best[2]))
        if k['glycan']:
            own = next((m for m in matches if same_glycan(m[1], k['glycan'])), None)
            k['note'] += f'; given structure kept (library cos {own[0]:.2f}, {", ".join(sorted(own[2]))})' if own else '; given structure kept (not in the library)'
            if best[0] >= LIBRARY_MS2 and (own is None or own[0] < best[0] - 0.2):
                flag(k, 'library disagrees', f'library prefers {best[1]} (cos {best[0]:.2f}, {where}) over the given structure'
                                             + (f' (cos {own[0]:.2f})' if own else ' (not in the library)'))
            continue
        nxt = f'next isomer {second[1]} at {second[0]:.2f}' if second[1] else 'no other isomer in the library'
        if best[0] >= LIBRARY_MS2 and second[0] < best[0] - LIBRARY_MARGIN:
            k['glycan'] = best[1]
            k['note'] += f'; structure from the library (pooled MS2 cos {best[0]:.2f} to {where}; {nxt})'
        else:
            k['note'] += f'; open: best library isomer {best[1]} at cos {best[0]:.2f} ({where}; {nxt})'
            if best[0] >= LIBRARY_REVIEW:
                flag(k, 'library transfer near threshold', f'{best[1]} at pooled MS2 cos {best[0]:.2f} ({where}); {nxt}')
    # Isobars in ion traps: Neu5Ac and 2 dHex, Neu5Gc and Hex + dHex differ by 1.02 Da, half an m/z unit at z = 2
    if not run.accurate:
        for k in kept:
            comp = dict(k['comp'])
            swaps = []
            if comp.get('Neu5Ac'):
                swaps.append({**comp, 'Neu5Ac': comp['Neu5Ac'] - 1, 'dHex': comp.get('dHex', 0) + 2})
            if comp.get('dHex', 0) >= 2:
                swaps.append({**comp, 'dHex': comp['dHex'] - 2, 'Neu5Ac': comp.get('Neu5Ac', 0) + 1})
            if comp.get('Neu5Gc'):
                swaps.append({**comp, 'Neu5Gc': comp['Neu5Gc'] - 1, 'Hex': comp.get('Hex', 0) + 1, 'dHex': comp.get('dHex', 0) + 1})
            if comp.get('Hex') and comp.get('dHex'):
                swaps.append({**comp, 'Neu5Gc': comp.get('Neu5Gc', 0) + 1, 'Hex': comp['Hex'] - 1, 'dHex': comp['dHex'] - 1})
            for swap in swaps:
                swap = {x: v for x, v in swap.items() if v}
                b = analyse(swap)
                rival = [q for q in b['peaks'] if abs(q['RT'] - k['RT']) <= 0.15 and q['n_ms2'] and q['distinct'] and fragments_ok(fragments(b, q))]
                if rival and k['z'] >= 2:
                    q = rival[0]
                    flag(k, 'isobar', f'{comp_string(swap)} (1.02 Da apart) has a co-eluting peak at {q["RT"]:.2f} with {q["n_ms2"]} MS2 '
                                      f'({describe_fragments(fragments(b, q))})')
    # In-source fragments: co-apex with a much stronger peak of a superset composition whose trace has the same shape
    if run.ms1 is not None:
        for k in kept:
            for o in kept:
                sub, sup = dict(k['comp']), dict(o['comp'])
                if o is k or sub == sup or any(sup.get(x, 0) < v for x, v in sub.items()):
                    continue
                if abs(o['RT'] - k['RT']) <= max(0.05, 1.5 * run.cycle) and o['peak']['height'] >= 3 * k['peak']['height']:
                    a0, a1 = np.searchsorted(run.rts, k['RT'] - 0.5), np.searchsorted(run.rts, k['RT'] + 0.5, side = 'right')
                    x, y = k['analysis']['total'][a0:a1], o['analysis']['total'][a0:a1]
                    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else 0.0
                    if r >= 0.95:
                        flag(k, 'possible in-source fragment', f'co-apex with {o["composition"]} at {o["RT"]:.2f}, '
                                                               f'{o["peak"]["height"] / k["peak"]["height"]:.1f}x higher, trace correlation {r:.2f}')
    # A fully specified structure appears at most once per file; repeats are only reported (some curated ones were left, e.g. GPST000029 733.4)
    full = [k for k in kept if k['glycan'] and not any(c in k['glycan'] for c in '?/{')]
    for i, k in enumerate(full):
        for o in full[:i]:
            if same_glycan(k['glycan'], o['glycan']):
                flag(k, 'repeated structure', f'the fully specified structure is also given at {o["Mass"]:.2f}/{o["RT"]:.2f}; at most one of them can be right')
    # Abundance: the run column for given rows; added rows (and every row without a run column) get the apex height, scaled like the given rows
    given_abundance = [(pd.to_numeric(k['abundance'], errors = 'coerce'), k['peak']['height']) for k in kept if k['source'] == 'given' and abund_col]
    ratios = [x / h for x, h in given_abundance if pd.notna(x) and x > 0 and h > 0]
    scale = float(np.median(ratios)) if ratios else None
    total_height = sum(k['peak']['height'] for k in kept) or 1
    for k in kept:
        if not abund_col:
            k['abundance'] = round(100 * k['peak']['height'] / total_height, 4)
        elif k['source'] == 'added':
            k['abundance'] = round(k['peak']['height'] * scale, 4) if scale else np.nan
    out_col = run_name
    audited = pd.DataFrame([{'Mass': round(k['Mass'], 4), 'charge': k['charge'], 'RT': round(k['RT'], 2), 'glycan': k['glycan'], out_col: k['abundance'],
                             'composition': k['composition'], 'source': k['source'], 'RT_source': k['row']['RT_source'] if k['row'] else np.nan,
                             'Mass_source': k['row']['Mass_source'] if k['row'] else np.nan, 'glycan_source': k['row']['glycan_source'] if k['row'] else '',
                             'ms1_audit': k['note'], **(k['row']['extra'] if k['row'] else {})} for k in kept],
                           columns = STANDARD_COLUMNS + [out_col] + AUDIT_COLUMNS + extra_cols)
    audited = audited.sort_values(['Mass', 'RT']).reset_index(drop = True)
    removed_df = pd.DataFrame([{'Mass_source': r.get('Mass_source'), 'RT_source': r.get('RT_source'), 'glycan_source': r.get('glycan_source'),
                                'composition': r.get('composition', comp_string(r['comp']) if r.get('comp') else ''), 'source': r.get('source', 'given'),
                                'reason': r['reason']} for r in removed], columns = ['Mass_source', 'RT_source', 'glycan_source', 'composition', 'source', 'reason'])
    review_df = pd.DataFrame(review, columns = ['Mass', 'charge', 'RT', 'composition', 'glycan', 'category', 'detail'])
    n_given = int((removed_df['source'] == 'given').sum()) if len(removed_df) else 0
    kind = 'accurate' if run.accurate else 'unit-resolution, isotopes resolved up to z = 2' if run.resolved else 'ion trap, z >= 2 isotopes merged'
    summary = (f'{run_name}: {len(run.prec)} MS2, ' + (f'{len(run.rts)} MS1 scans (median cycle {run.cycle * 60:.1f} s, {kind} MS1, calibration {shift:+.3f} m/z, '
               f'{int(run.spikes.sum())} spike scans), ' if run.ms1 is not None else 'no MS1 (MS2-only checks), ')
               + f'RT shift {global_shift:+.2f} min from {len(unique)} unique matches\n'
               f'source rows {len(src)}: kept {int((audited["source"] == "given").sum())}, removed {n_given}; added {int((audited["source"] == "added").sum())} '
               f'from {n_pred} unmatched predictions; {len(review_df)} review items; {time.time() - t_start:.0f} s')
    return audited, removed_df, review_df, summary


def main():
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('spectra')
    parser.add_argument('gt')
    parser.add_argument('--glycan_class', required = True, choices = ['N', 'O', 'free', 'lipid'])
    parser.add_argument('--run', default = None, help = 'run column of the source GT (default: the spectra file stem)')
    parser.add_argument('--modification', default = 'reduced', choices = ['reduced', '2AA', '2AB', 'procainamide', 'PA', 'custom'])
    parser.add_argument('--mass_tag', type = float, default = 0.0)
    parser.add_argument('--sample_prep', default = 'underivatized', choices = ['underivatized', 'permethylated', 'peracetylated'])
    parser.add_argument('--output', default = None)
    parser.add_argument('--predictions', default = None, help = 'CandyCrunch output table of this run (.xlsx or .csv)')
    parser.add_argument('--rt_window', type = float, default = 1.5, help = 'minutes around the shifted source RT within which a row looks for its peak')
    parser.add_argument('--lc', default = 'PGC', choices = ['PGC', 'C18', 'other'], help = 'the tail rule only applies to PGC')
    parser.add_argument('--library_cache', default = str(Path.home() / '.cache' / 'candycrunch' / 'audit_gt_library.pkl'))
    parser.add_argument('--data_dir', default = str(DATA_DIR), help = 'test data folder whose curated GTs and .xlsx form the spectral library')
    parser.add_argument('--mode', default = None, choices = ['negative', 'positive'], help = 'ion mode, for spectra files that record none (Agilent mgf/xlsx)')
    parser.add_argument('--max_charge', type = int, default = 3, help = 'highest charge searched for the compositions of open rows (the test set\'s max_charge)')
    args = parser.parse_args()
    gt_path = Path(args.gt)
    default = gt_path.stem[:-len('_source')] + '.csv' if gt_path.stem.endswith('_source') else gt_path.stem + '_audited.csv'
    output = Path(args.output) if args.output else gt_path.with_name(default)
    if output.resolve() == gt_path.resolve():
        sys.exit(f'{output} is the source GT; give another --output')
    audited, removed, review, summary = audit(args.spectra, gt_path, args.glycan_class, run_name = args.run, modification = args.modification,
                                              mass_tag = args.mass_tag, sample_prep = args.sample_prep, predictions = args.predictions,
                                              rt_window = args.rt_window, library_cache = args.library_cache, data_dir = args.data_dir, lc = args.lc,
                                              mode = args.mode, max_charge = args.max_charge)
    audited.to_csv(output, index = False)
    removed.to_csv(output.with_name(output.stem + '_removed.csv'), index = False)
    review.to_csv(output.with_name(output.stem + '_review.csv'), index = False)
    print(summary)
    with pd.option_context('display.max_colwidth', 160, 'display.width', 250):
        if len(removed):
            print('\nremoved:\n' + removed[['Mass_source', 'RT_source', 'composition', 'source', 'reason']].to_string(index = False))
        if len(review):
            print('\nreview:\n' + review.to_string(index = False))
    print(f'\nwrote {output}, {output.stem}_removed.csv, {output.stem}_review.csv')


if __name__ == '__main__':
    main()

"""Readers for mzML, mzXML, mzData, mgf, Thermo .raw, Waters .raw, Agilent MassHunter .d, Bruker Q-TOF and timsTOF (.tsf) .d and SCIEX .wiff spectra files, on the standard library and numpy"""
import base64
import heapq
import itertools
import os
import pathlib
import re
import sqlite3
import struct
import zlib
import xml.etree.ElementTree as ET
import numpy as np
# Thermo .raw trailer extra fields: numpy types of the generic data types (0 is a gap, 2-4 are booleans; 12 and 13, strings of the declared length
# in ASCII or UTF-16, are only read for HCD Energy)
RAW_FIELD_TYPES = {0: 'V0', 1: 'i1', 2: 'u1', 3: 'u1', 4: 'u1', 5: 'u1', 6: '<i2', 7: '<u2', 8: '<i4', 9: '<u4', 10: '<f4', 11: '<f8'}
# The trailer extra fields read per scan (Thermo writes 0 for a monoisotopic m/z or charge state it did not determine)
RAW_TRAILER_FIELDS = ('Monoisotopic M/Z:', 'Charge State:', 'MS2 Isolation Width:', 'MSn Isolation Width:', 'Master Scan Number:', 'HCD Energy:')
# Activation of a scan event reaction, bits 1 and up of the u32 after its energy (Thermo's ActivationType)
RAW_ACTIVATIONS = {0: 'cid', 1: 'mpd', 2: 'ecd', 3: 'pqd', 4: 'etd', 5: 'hcd', 8: 'ptr', 9: 'netd', 10: 'nptr', 11: 'uvpd'}
# A timsTOF survey frame summed over its ion mobility scans holds ~40k centroids, mostly single-ion noise; the 5000 most intense reach down to ~0.04%
# of the base peak and keep the MS1 data of an hour-long run near 150 MB
TIMS_MS1_PEAKS = 5000
# A PASEF MS2 spectrum (a precursor's mobility scans summed) holds a median of ~190 centroids, most of them single ions, against the 10-50 peaks of the ion
# trap spectra the model was trained on; its 100 most intense peaks gave more glycans, with fewer false ones, than all of them or than intensity cut-offs
TIMS_MS2_PEAKS = 100
# Ion-trap analyzers, and the models of the Bruker ion traps (HCT, esquire, amaZon) and the Agilent 6300 ion traps Bruker built; without a Thermo filter
# string, their profile spectra are centroided as Bruker ion-trap scans
ION_TRAP_TERMS = {'MS:1000264', 'MS:1000082', 'MS:1000083', 'MS:1000078', 'MS:1000291', 'MS:1000697', 'MS:1000160', 'MS:1000161', 'MS:1000698',
                  'MS:1000699', 'MS:1000700', 'MS:1001533', 'MS:1000156', 'MS:1000157', 'MS:1003461', 'MS:1001545', 'MS:1001542', 'MS:1001546',
                  'MS:1002300', 'MS:1002301', 'MS:1003466', 'MS:1000473', 'MS:1000474', 'MS:1000475', 'MS:1000476'}


def centroid_ion_trap(mzs, ints, min_gap = 0.5, bin_shift = True):
    """centroids a Thermo ion-trap profile spectrum as Thermo's centroider does (it made the centroids of the mzML files CandyCrunch was validated on)\n
   | Arguments:
   | :-
   | mzs (array): profile m/z values, on Thermo's profile axis (as in mzML files from msconvert)
   | ints (array): profile intensities
   | min_gap (float): neighboring centroids closer than this are merged; default:0.5 (Thermo's centroids are always >= 0.5 m/z apart)
   | bin_shift (bool): whether centroids of wide peaks move one profile bin down, as Thermo's do; default:True\n
   | Returns:
   | :-
   | Returns an (n, 2) array of centroid m/z and intensity: profile points below min(1, 0.1% of the scan's maximum) are dropped, the profile is split at the lowest point between
   | maxima and neighboring centroids closer than min_gap are merged, each carrying the summed signal of its share of the profile
   """
    # Thermo's centroider drops points below 1; some ion traps store intensities ~1000x lower (GPST000374, GPST000596: MS2 base peaks of 1-30), where that
    # dropped most fragments, so the floor is at most 0.1% of the scan's maximum
    ints = np.where(ints < min(1, 1e-3 * ints.max(initial = 0)), 0, ints)
    apex = np.where((ints[1:-1] > ints[:-2]) & (ints[1:-1] >= ints[2:]) & (ints[1:-1] > 0))[0] + 1
    if not len(apex):
        return np.zeros((0, 2))
    # The lowest point between neighboring maxima (its first occurrence): the minimum of each stretch, then the first point of the stretch at it; a stretch
    # may run on to the next maximum, which is never its minimum (it rises from the point before it)
    lows = np.minimum.reduceat(ints[:apex[-1]], apex[:-1] + 1) if len(apex) > 1 else np.zeros(0)
    stretch = np.repeat(np.arange(len(lows)), np.diff(apex))
    at_low = np.flatnonzero(ints[apex[0] + 1:apex[-1] + 1] == lows[stretch])
    edges = np.concatenate([[0], apex[0] + 1 + at_low[np.diff(stretch[at_low], prepend = -1) > 0]])
    while True:
        peak_ints = np.add.reduceat(ints, edges)
        peak_mzs = np.add.reduceat(ints * mzs, edges) / peak_ints
        gaps = np.diff(peak_mzs)
        if not len(gaps) or gaps.min() >= min_gap:
            break
        # Merges every pair closer than min_gap whose gap is the smallest among its neighboring gaps, until none is left
        left, right = np.concatenate([[np.inf], gaps[:-1]]), np.concatenate([gaps[1:], [np.inf]])
        edges = np.delete(edges, np.where((gaps < min_gap) & (gaps <= left) & (gaps < right))[0] + 1)
    # Thermo's centroids of peaks spanning 9 or more profile points sit one bin below their intensity-weighted mean (the bin is the median spacing,
    # as converters drop runs of zero points)
    if bin_shift:
        peak_mzs = peak_mzs - np.median(np.diff(mzs)) * (np.add.reduceat((ints > 0).astype(int), edges) >= 9)
    return np.stack([peak_mzs, peak_ints], axis = 1)


def centroid_tof(mzs, ints, max_peaks = None, valley = 0.7, min_height = 0):
    """centroids the summed TOF points of a timsTOF frame (or of one precursor's PASEF scans), which sit on the TOF digitizer grid, evenly spaced in sqrt(m/z)\n
   | Arguments:
   | :-
   | mzs (array): m/z values of the points, repeated where several ion mobility scans hit the same TOF bin
   | ints (array): their intensities
   | max_peaks (int): keep only this many of the most intense centroids; default:None (all)
   | valley (float): split a run where the smoothed signal dips below this share of the smaller of the two maxima around it; default:0.7
   | min_height (float): only points above this share of their peak's apex weigh its m/z; default:0 (all)\n
   | Returns:
   | :-
   | Returns an (n, 2) array of centroid m/z and intensity: points of one TOF bin are summed, runs of neighboring bins are split where the signal
   | (smoothed over neighboring bins) dips below 70% of the smaller of the two maxima around it, and each centroid carries the summed signal of its
   | run at its intensity-weighted m/z
   """
    order = np.argsort(mzs, kind = 'stable')
    mzs, ints = np.asarray(mzs, dtype = np.float64)[order], np.asarray(ints, dtype = np.float64)[order]
    if not len(mzs):
        return np.zeros((0, 2))
    # A TOF bin is 6e-5 to 1.3e-4 wide in sqrt(m/z), so points closer than 2e-5 are one bin (hit in several ion mobility scans, or written twice
    # as 32-bit m/z in combined spectra); one bin is then the smallest step left, and a run of bins may skip one empty bin
    same = np.concatenate([[0], np.cumsum(np.diff(np.sqrt(mzs)) >= 2e-5)])
    sums = np.bincount(same, weights = ints)
    mzs, ints = np.bincount(same, weights = mzs * ints)[sums > 0] / sums[sums > 0], sums[sums > 0]
    if len(mzs) < 3:
        return np.stack([mzs, ints], axis = 1)
    gaps = np.diff(np.sqrt(mzs))
    starts = np.concatenate([[True], gaps > 2.5 * gaps.min()])
    smooth = ints.copy()
    smooth[1:] += 0.5 * ints[:-1] * (gaps <= 1.5 * gaps.min())
    smooth[:-1] += 0.5 * ints[1:] * (gaps <= 1.5 * gaps.min())
    minima = np.where(~starts[1:-1] & ~starts[2:] & (smooth[1:-1] < smooth[:-2]) & (smooth[1:-1] <= smooth[2:]))[0] + 1
    # Boundaries as a mask over the points (np.union1d of the index arrays gives the same, but slowly: half of loading a timsTOF run)
    is_bound = starts.copy()
    is_bound[minima] = True
    bounds = np.flatnonzero(is_bound)
    tops, pos = np.maximum.reduceat(smooth, bounds), np.searchsorted(bounds, minima)
    is_bound = starts.copy()
    is_bound[minima[smooth[minima] < valley * np.minimum(tops[pos - 1], tops[pos])]] = True
    bounds = np.flatnonzero(is_bound)
    peak_ints = np.add.reduceat(ints, bounds)
    weights = ints * (ints >= min_height * np.maximum.reduceat(ints, bounds)[np.cumsum(is_bound) - 1]) if min_height else ints
    peak_mzs = np.add.reduceat(weights * mzs, bounds) / np.add.reduceat(weights, bounds)
    if max_peaks is not None and len(peak_ints) > max_peaks:
        # The most intense centroids and, of those at the cut-off intensity, the first in m/z order: argpartition left which of these ties stay to its
        # algorithm, which differs between numpy builds (x86 SIMD vs the browser's wasm), and so did the survey frames of a timsTOF run
        kth = np.partition(peak_ints, -max_peaks)[-max_peaks]
        above = np.flatnonzero(peak_ints > kth)
        top = np.concatenate([above, np.flatnonzero(peak_ints == kth)[:max_peaks - len(above)]])
        top.sort()
        peak_mzs, peak_ints = peak_mzs[top], peak_ints[top]
    return np.stack([peak_mzs, peak_ints], axis = 1)


def centroid_gaussian(mzs, ints):
    """centroids a profile spectrum by a 3-point Gaussian fit at each local maximum, as pymzml's highest_peaks did (positions 0 and 1 skipped)\n
   | Arguments:
   | :-
   | mzs (array): profile m/z values
   | ints (array): profile intensities\n
   | Returns:
   | :-
   | Returns an (n, 2) array of centroid m/z and the fitted apex intensity
   """
    x, y = np.asarray(mzs, dtype = np.float64), np.asarray(ints, dtype = np.float64)
    pos = np.arange(2, len(y) - 1)
    x1, x2, x3, y1, y2, y3 = x[pos - 1], x[pos], x[pos + 1], y[pos - 1], y[pos], y[pos + 1]
    keep = (0 < y1) & (y1 < y2) & (y2 > y3) & (y3 > 0) & ~(x2 - x1 > (x3 - x2) * 10) & ~((x2 - x1) * 10 < x3 - x2)
    x1, x2, x3, y1, y2, y3 = x1[keep], x2[keep], x3[keep], y1[keep], y2[keep], y3[keep]
    y3 = np.where(y3 == y1, y3 + 0.01 * y1, y3)
    # Fits the scalar version skipped (ZeroDivisionError, OverflowError) come out as inf or nan and are dropped
    with np.errstate(divide = 'ignore', over = 'ignore', invalid = 'ignore'):
        double_log = np.log(y2 / y1) / np.log(y3 / y1)
        mu = (double_log * (x1 * x1 - x3 * x3) - x1 * x1 + x2 * x2) / (2 * (x2 - x1) - 2 * double_log * (x3 - x1))
        c_squared = (x2 * x2 - x1 * x1 - 2 * x2 * mu + 2 * x1 * mu) / (2 * np.log(y1 / y2))
        apex = y1 * np.exp((x1 - mu) * (x1 - mu) / (2 * c_squared))
    fine = np.isfinite(mu) & np.isfinite(apex)
    return np.stack([mu[fine], apex[fine]], axis = 1)


def read_mzml(filepath, centroid_levels = ()):
    """iterates over the spectra of an .mzML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzML file
   | centroid_levels (tuple): MS levels whose profile spectra are centroided (Thermo ion-trap scans as by Thermo's centroider, Bruker and Agilent ion-trap
   |                          scans at valleys as in read_mzxml, others by a 3-point Gaussian fit at each local maximum); default:()\n
   | Returns:
   | :-
   | Yields a dict per spectrum: element (its XML element, with the params of a referenced param group appended), ms_level (int or None),
   | rt (scan start time in minutes; None if absent), peaks ((n, 2) array of m/z and intensity), and precursor (mz, and if given charge and
   | intensity i, of the first selected ion in the spectrum; None if there is none), instrument (CV accessions of the spectrum's instrument
   | configuration, i.e., its analyzers and instrument model; those of the first configuration if the scan names none), and ion_mobility (whether
   | it is timsTOF data, whose spectra of centroid_levels are then centroided with centroid_tof; files converted without combining ion mobility
   | scans hold one spectrum per scan, so the scans of one frame, and in MS2 of one precursor, are summed into one spectrum)
   """
    def merged(scans):
        peaks = np.concatenate([x['peaks'] for x in scans])
        if scans[0]['ms_level'] in centroid_levels:
            peaks = centroid_tof(peaks[:, 0], peaks[:, 1], max_peaks = TIMS_MS1_PEAKS if scans[0]['ms_level'] == 1 else None)
        return dict(scans[0], peaks = peaks[np.argsort(peaks[:, 0], kind = 'stable')], ion_mobility = True)
    groups, configs, pending, pending_key = {}, {}, [], None
    for _, el in ET.iterparse(filepath):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'referenceableParamGroup':
            groups[el.get('id')] = el
        elif tag == 'instrumentConfiguration':
            # Param groups precede the instrument configurations, which hold the instrument model in a referenced group (Thermo) or directly
            # msconvert writes every Bruker ion trap as HCT series and names an amaZon only in an 'instrument model' userParam, which counts as the amaZon series
            configs[el.get('id')] = {e.get('accession') or 'MS:1001545' for p in [el] + [groups[r.get('ref')] for r in el.iter() if r.tag.endswith('referenceableParamGroupRef') and r.get('ref') in groups]
                                     for e in p.iter() if e.get('accession') or (e.get('name') == 'instrument model' and 'amazon' in e.get('value', '').lower())}
        elif tag == 'chromatogram':
            el.clear()
        if tag != 'spectrum':
            continue
        ns = el.tag[:-len(tag)]
        # Param groups can be referenced by the spectrum and any of its parts (binary data arrays naming their type and encoding), and several times
        for parent, ref in [(p, r) for p in el.iter() for r in p.findall(f'{ns}referenceableParamGroupRef')]:
            parent.extend(groups.get(ref.get('ref'), ()))
        # The first element of each CV term anywhere in the spectrum
        first = {}
        for e in el.iter():
            first.setdefault(e.get('accession'), e)
        arrays, mobility = {}, False
        for bda in el.iter(f'{ns}binaryDataArray'):
            terms = {c.get('accession') for c in bda.findall(f'{ns}cvParam')}
            # An ion mobility array marks a timsTOF spectrum combined over its mobility scans, whose points sit on the TOF grid, many per peak
            mobility = mobility or bool(terms & {'MS:1002816', 'MS:1003006', 'MS:1003007', 'MS:1003008'})
            kind = 'mz' if 'MS:1000514' in terms else 'i' if 'MS:1000515' in terms else None
            if kind is None or kind in arrays:
                continue
            data = base64.b64decode(bda.findtext(f'{ns}binary') or '')
            if data and terms & {'MS:1000574', 'MS:1002746', 'MS:1002747', 'MS:1002748'}:
                data = zlib.decompress(data)
            if not data:
                arrays[kind] = np.zeros(0)
            elif terms & {'MS:1002314', 'MS:1002748'}:
                # MS-Numpress short logged float: a big-endian double fixed point, then each log(x + 1) * fixed point as a 16-bit integer
                arrays[kind] = np.exp(np.frombuffer(data, '<u2', offset = 8) / struct.unpack('>d', data[:8])[0]) - 1
            elif terms & {'MS:1002312', 'MS:1002313', 'MS:1002746', 'MS:1002747'}:
                # MS-Numpress linear prediction (a fixed point, two 32-bit start values, then the residuals of a linear extrapolation) and
                # positive integer compression store integers as half-byte codes: a head n (<= 8: n leading zero half-bytes, else n - 8
                # leading 0xf half-bytes), then the other half-bytes, least significant first; a zero half-byte can pad the last byte
                linear = bool(terms & {'MS:1002312', 'MS:1002746'})
                b = np.frombuffer(data, np.uint8)[16 if linear else 0:]
                half = np.stack([b >> 4, b & 15], axis = 1).ravel().tolist()
                ints, p = [], 0
                while p < len(half) and not (p == len(half) - 1 and half[p] == 0):
                    n = half[p] if half[p] <= 8 else half[p] - 8
                    x = sum(v << (4 * k) for k, v in enumerate(half[p + 1:p + 9 - n]))
                    ints.append(x | ((1 << 4 * n) - 1) << (32 - 4 * n) if half[p] > 8 else x)
                    p += 9 - n
                if linear:
                    ys = [int.from_bytes(data[i:i + 4], 'little') for i in (8, 12) if len(data) >= i + 4]
                    for x in ints:
                        ys.append(2 * ys[-1] - ys[-2] + (x - (1 << 32) if x >= 1 << 31 else x))
                    arrays[kind] = np.array(ys, dtype = float) / struct.unpack('>d', data[:8])[0]
                else:
                    arrays[kind] = np.array(ints, dtype = float)
            else:
                arrays[kind] = np.frombuffer(data, next((d for acc, d in (('MS:1000521', '<f4'), ('MS:1000523', '<f8'), ('MS:1000519', '<i4'),
                                                                        ('MS:1000522', '<i8')) if acc in terms), '<f8'))
        peaks = np.stack((arrays.get('mz', np.zeros(0)), arrays.get('i', np.zeros(0))), axis = -1)
        ms_level = int(first['MS:1000511'].get('value')) if 'MS:1000511' in first else None
        scan = el.find(f'{ns}scanList/{ns}scan')
        instrument = configs.get(scan.get('instrumentConfigurationRef') if scan is not None else None, next(iter(configs.values()), set()))
        if ms_level in centroid_levels and 'MS:1000128' in first and (first['MS:1000512'].get('value', '') if 'MS:1000512' in first else '').startswith('ITMS'):
            # msconvert writes ion-trap scans as profiles unless asked to peak-pick; Thermo's centroids of them gave GPST000017 F1 0.735 (as from the
            # .raw file) against 0.695 with the Gaussian fit below
            peaks = centroid_ion_trap(peaks[:, 0], peaks[:, 1])
        elif ms_level in centroid_levels and mobility:
            peaks = centroid_tof(peaks[:, 0], peaks[:, 1], max_peaks = TIMS_MS1_PEAKS if ms_level == 1 else None)
        elif ms_level in centroid_levels and 'MS:1000128' in first and 'MS:1000512' not in first and instrument & ION_TRAP_TERMS and 'MS:1000084' not in instrument:
            # As in read_mzxml: Bruker and Agilent ion traps resolve the isotopes of doubly charged ions, so peaks are split at valleys and only merged
            # below 0.3 m/z; on an HCT run the Gaussian fit below kept 56% of the MS2 intensity of Bruker's own line spectra within 0.1 m/z, this 94%
            peaks = centroid_ion_trap(peaks[:, 0], peaks[:, 1], min_gap = 0.3, bin_shift = False)
        elif ms_level in centroid_levels and 'MS:1000128' in first:
            peaks = centroid_gaussian(peaks[:, 0], peaks[:, 1])
        rt = first.get('MS:1000016')
        if rt is not None:
            # The unit by name, else by its unit ontology accession (unitName is optional)
            unit = (rt.get('unitName') or {'UO:0000010': 'second', 'UO:0000028': 'millisecond', 'UO:0000032': 'hour'}.get(rt.get('unitAccession'), 'minute')).lower()
            rt = float(rt.get('value'))
            rt = rt * 60.0 if unit == 'hour' else rt / {'minute': 1, 'second': 60.0, 'millisecond': 60000.0}[unit]
        precursor = {key: conv(first[acc].get('value')) for key, acc, conv in (('mz', 'MS:1000744', float), ('charge', 'MS:1000041', int),
                                                                               ('i', 'MS:1000042', float)) if acc in first} if 'MS:1000744' in first else None
        spectrum = {'element': el, 'ms_level': ms_level, 'rt': rt, 'peaks': peaks, 'precursor': precursor,
                    'instrument': instrument, 'ion_mobility': mobility}
        # One spectrum per ion mobility scan (native ID frame=f scan=s, from msconvert without combining them): its scans of one frame, MS level, and
        # precursor are summed once that group ends. Taken one by one, every scan of a frame was a survey scan of its own at the same retention time,
        # whose zero time step crashed the XIC smoothing (OverflowError: cannot convert float infinity to integer)
        tims = re.match(r'frame=(\d+) scan=\d+$', el.get('id', ''))
        key = (tims.group(1), ms_level, precursor and precursor.get('mz')) if tims else None
        if pending and key != pending_key:
            yield merged(pending)
            for x in pending:
                x['element'].clear()
            pending = []
        if key is None:
            yield spectrum
            el.clear()
        else:
            pending.append(spectrum)
            pending_key = key
    if pending:
        yield merged(pending)


def read_mzxml(filepath, centroid_levels = ()):
    """iterates over the scans of an .mzXML file, by scan number up to each MS1 scan (nested scans end before the MS1 scan enclosing them)\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzXML file
   | centroid_levels (tuple): MS levels whose profile scans (centroided="0") are centroided as in read_mzml; default:()\n
   | Returns:
   | :-
   | Yields a dict per scan: its attributes as strings (msLevel as int, retentionTime in minutes, id as num), precursorMz (a list of the
   | precursorMz elements' attributes, their value as precursorMz, precursorCharge as int, precursorIntensity as float), m/z array,
   | intensity array, and instrument (msModel and msMassAnalyzer of the scan's msInstrument, lower-case; of the first one if the scan names none)
   """
    queue, order, instruments = [], itertools.count(), {}
    for _, el in ET.iterparse(filepath):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'msInstrument':
            instruments[el.get('msInstrumentID', el.get('id'))] = ' '.join(c.get('value', '') for c in el if c.tag.rsplit('}', 1)[-1] in ('msModel', 'msMassAnalyzer')).lower()
        if tag != 'scan':
            continue
        ns = el.tag[:-4]
        scan = dict(el.attrib)
        scan['msLevel'], scan['id'] = int(scan['msLevel']), scan.get('num')
        if scan.get('retentionTime', '').startswith('P'):
            # xs:duration, e.g. PT1503.97S
            h, m, s = (float(v or 0) for v in re.search(r'P(?:\d+\.?\d*Y)?(?:\d+\.?\d*M)?(?:\d+\.?\d*D)?(?:T(?:(\d+\.?\d*)H)?(?:(\d+\.?\d*)M)?'
                                                        r'(?:(\d+\.?\d*)S)?)?', scan['retentionTime']).groups())
            scan['retentionTime'] = m + h * 60.0 + s / 60.0
        elif 'retentionTime' in scan:
            scan['retentionTime'] = float(scan['retentionTime'])
        else:
            # MALDI spectra exported to mzXML have none; as for MGF spectra without RTINSECONDS
            raise ValueError(f"{filepath} has no retentionTime for scan {scan.get('num')} (e.g., MALDI spectra); CandyCrunch needs retention times")
        scan['precursorMz'] = []
        for p in el.findall(f'{ns}precursorMz'):
            prec = dict(p.attrib)
            prec.update({k: conv(prec[k]) if prec[k] else None for k, conv in (('precursorCharge', int), ('precursorIntensity', float)) if k in prec})
            prec['precursorMz'] = float(p.text)
            scan['precursorMz'].append(prec)
        peaks = next(iter(el.findall(f'{ns}peaks')), ET.Element('peaks'))
        data = base64.b64decode(peaks.text or '')
        if data and peaks.get('compressionType') == 'zlib':
            data = zlib.decompress(data)
        precision = 'f4' if peaks.get('precision', '32') == '32' else 'f8'
        values = np.frombuffer(data, ('>' if peaks.get('byteOrder', 'network') in ('network', 'big') else '<') + precision).astype(precision)
        scan['m/z array'], scan['intensity array'] = values[0::2], values[1::2]
        scan['instrument'] = instruments.get(scan.get('msInstrumentID'), next(iter(instruments.values()), ''))
        # Profile scans (e.g., Bruker ion-trap files converted by msconvert) are centroided like those of mzML files
        if scan['msLevel'] in centroid_levels and scan.get('centroided') == '0' and len(values):
            if str(scan.get('filterLine', '')).startswith('ITMS'):
                cents = centroid_ion_trap(values[0::2], values[1::2])
            # Other ion traps (Bruker HCT, amaZon) resolve the isotopes of doubly charged ions, and the Gaussian fit split their noisy peak tops into
            # centroids 0.1-0.2 m/z apart (1040.39 and 1040.58 for one peak), so peaks are split at valleys as for Thermo, but only merged below 0.3 m/z
            elif 'filterLine' not in scan and 'ion trap' in scan['instrument']:
                cents = centroid_ion_trap(values[0::2], values[1::2], min_gap = 0.3, bin_shift = False)
            else:
                cents = centroid_gaussian(values[0::2], values[1::2])
            scan['m/z array'], scan['intensity array'] = cents[:, 0], cents[:, 1]
        el.clear()
        heapq.heappush(queue, (int(scan['num']), next(order), scan))
        if scan['msLevel'] == 1:
            while queue[0][0] < int(scan['num']):
                yield heapq.heappop(queue)[2]
    while queue:
        yield heapq.heappop(queue)[2]


# mzData terms by their PSI accession; a term a writer gives by name alone is read by its name in lower case
MZDATA_TERMS = {'PSI:1000037': 'polarity', 'PSI:1000038': 'timeinminutes', 'PSI:1000039': 'timeinseconds', 'PSI:1000040': 'masstochargeratio',
                'PSI:1000041': 'chargestate', 'PSI:1000042': 'intensity', 'PSI:1000044': 'method', 'PSI:1000045': 'collisionenergy'}


def read_mzdata(filepath, centroid_levels = ()):
    """iterates over the spectra of an mzData file (PSI's XML format before mzML, which Agilent MassHunter's MzDataExport writes, e.g., as run.mzdata.xml)\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the mzData file
   | centroid_levels (tuple): MS levels whose profile spectra (spectrumType continuous) are centroided (ion-trap scans at valleys, as Bruker and Agilent
   |                          ion-trap scans in read_mzml, others by a 3-point Gaussian fit at each local maximum); default:()\n
   | Returns:
   | :-
   | Yields a dict per spectrum: id (the spectrum's id, a string), ms_level (int or None), rt (minutes; None if absent), polarity ('+', '-' or None), peaks
   | ((n, 2) array of m/z and intensity), precursor (mz, and if given charge and intensity i, of the first precursor, with ref, the id of the spectrum it
   | was selected in; None if there is none), activation (its method term, e.g., 'CID'; None if absent), collision_energy (float or None), and instrument
   | (the names and values of the file's analyzer terms in lower-case letters only, e.g., 'analyzertypequadrupoletimeofflight')
   """

    def terms(node):
        return {MZDATA_TERMS.get(p.get('accession')) or (p.get('name') or '').lower(): p.get('value') for p in (node.iter() if node is not None else ())
                if p.tag.rsplit('}', 1)[-1] in ('cvParam', 'userParam')}

    instrument = ''
    for _, el in ET.iterparse(filepath):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'analyzer':
            # The file's one instrument names its analyzers in words (QuadrupoleTimeOfFlight, IonTrap, ...), before the spectra
            instrument += re.sub('[^a-z]', '', ''.join((p.get('name') or '') + (p.get('value') or '') for p in el.iter()).lower())
        if tag != 'spectrum':
            continue
        ns = el.tag[:-len(tag)]
        settings = el.find(f'{ns}spectrumDesc/{ns}spectrumSettings')
        scan = settings.find(f'{ns}spectrumInstrument') if settings is not None else None
        params = terms(scan)
        precursor_el = el.find(f'{ns}spectrumDesc/{ns}precursorList/{ns}precursor')
        selected = terms(precursor_el)
        arrays = []
        for kind in ('mzArrayBinary', 'intenArrayBinary'):
            # Base64 of 32- or 64-bit floats in either byte order; mzData has no compression
            data = el.find(f'{ns}{kind}/{ns}data')
            text = data.text if data is not None else None
            arrays.append(np.frombuffer(base64.b64decode(text), ('>' if data.get('endian') == 'big' else '<') + ('f4' if data.get('precision') == '32' else 'f8'))
                          if text and text.strip() else np.zeros(0))
        peaks = np.stack(arrays, axis = -1)
        ms_level = int(scan.get('msLevel')) if scan is not None and scan.get('msLevel') else None
        acquisition = settings.find(f'{ns}acqSpecification') if settings is not None else None
        if ms_level in centroid_levels and len(peaks) and acquisition is not None and acquisition.get('spectrumType') == 'continuous':
            peaks = (centroid_ion_trap(peaks[:, 0], peaks[:, 1], min_gap = 0.3, bin_shift = False) if 'trap' in instrument and 'orbitrap' not in instrument else
                     centroid_gaussian(peaks[:, 0], peaks[:, 1]))
        rt = float(params['timeinminutes']) if params.get('timeinminutes') else float(params['timeinseconds']) / 60 if params.get('timeinseconds') else None
        precursor = {key: conv(selected[term]) for key, term, conv in (('mz', 'masstochargeratio', float), ('charge', 'chargestate', lambda x: int(float(x))),
                                                                        ('i', 'intensity', float)) if selected.get(term)} if selected.get('masstochargeratio') else None
        if precursor:
            precursor['ref'] = precursor_el.get('spectrumRef')
        yield {'id': el.get('id'), 'ms_level': ms_level, 'rt': rt, 'polarity': {'p': '+', '+': '+', 'n': '-', '-': '-'}.get((params.get('polarity') or ' ')[0].lower()),
               'peaks': peaks, 'precursor': precursor, 'activation': selected.get('method'),
               'collision_energy': float(selected['collisionenergy']) if selected.get('collisionenergy') else None, 'instrument': instrument}
        el.clear()


def read_mgf(filepath):
    """iterates over the spectra of an .mgf file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mgf file\n
   | Returns:
   | :-
   | Yields a dict per spectrum: params (lower-case keys, including those before the first spectrum; pepmass as an (m/z, intensity or None)
   | tuple, charge as a list of ints, rtinseconds as float), m/z array and intensity array
   """
    header, params, started = {}, None, False
    # utf-8-sig drops a byte order mark, which hid the first BEGIN IONS (its params then became the defaults of every other spectrum)
    with open(filepath, encoding = 'utf-8-sig', errors = 'replace') as f:
        for line in f:
            sline = line.strip()
            if sline == 'BEGIN IONS':
                params, mzs, ints, started = dict(header), [], [], True
            elif params is None:
                if not started and len(parts := line.split('=')) == 2:
                    header[parts[0].lower()] = parts[1].strip()
            elif sline == 'END IONS':
                if 'pepmass' in params:
                    pepmass = params['pepmass'].split()
                    params['pepmass'] = tuple(float(v) for v in pepmass[:2]) + (None,) * (2 - len(pepmass[:2]))
                    if len(pepmass) == 3:
                        params['charge'] = pepmass[2]
                if isinstance(params.get('charge'), str):
                    # e.g. 2+, 3+ and 4+
                    params['charge'] = [int(c[-1] + c[:-1]) if c[-1:] in ('+', '-') else int(c) for c in re.split(r',\s*|\s*and\s*', params['charge'].strip()) if c]
                if 'rtinseconds' in params:
                    # A retention time, or ranges and lists of them (e.g., 600.5-612.5): the middle of their span
                    rts = [float(v) for v in re.findall(r'\d*\.?\d+(?:[eE][-+]?\d+)?', params['rtinseconds'])]
                    params['rtinseconds'] = (min(rts) + max(rts)) / 2
                yield {'params': params, 'm/z array': np.array(mzs), 'intensity array': np.array(ints)}
                params = None
            elif not sline or sline[0] in '#;!/':
                continue
            elif '=' in sline:
                key, value = sline.split('=', 1)
                params[key.lower()] = value.strip()
            elif len(parts := sline.split()) > 1:
                mzs.append(float(parts[0]))
                ints.append(float(parts[1]))


def read_raw_scan_table(filepath):
    """reads the scan table of a Thermo .raw file (format versions 57 to 66, i.e. Xcalibur 2.0 and later) from its scan index, scan events and trailer\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .raw file\n
   | Returns:
   | :-
   | Returns the scan table as a list per column, one entry per scan: scan, ms_level, polarity, scan_mode, analyzer, filter_string (isolation m/z,
   | activation and energy of each stage, e.g., 1022.4700@etd50.00), precursor_mz, charge, isolation_width, master_scan, retention_time, scan_range
   | ((lowest, highest) m/z of its scan windows; an ion-trap MS2's lower limit is its low-mass cutoff), packet (the file offset of the scan's data
   | packet, for read_raw_packet), and RealEnergy (the trailer's HCD Energy of MSn scans: a float, or text such as 30.00,42.00,72.00 for stepped energies)
   """
    with open(filepath, 'rb') as f:
        head = f.read(1356)
        version = int.from_bytes(head[36:40], 'little')
        if head[2:18] != 'Finnigan'.encode('utf-16-le') or version < 57:
            raise ValueError(f"{filepath} is not a Thermo .raw file of Xcalibur 2.0 or later (format version {version}); convert it to mzML with "
                             f"ThermoRawFileParser or msconvert first.")
        wide = version >= 64
        # Skips the sequence row (64 bytes, then 16 strings, a u32 and, from version 60, 15 more strings) and the autosampler info (24 bytes, a
        # string), each string being a u32 count of UTF-16 characters followed by them
        f.seek(1420)
        for size in [None] * 16 + [4] + [None] * (15 if version >= 60 else 0) + [24, None]:
            f.seek(size or 2 * int.from_bytes(f.read(4), 'little'), 1)
        # The file info lists the run header of each controller (MS, UV, pump, ...) from byte 824 as u64 every 16 bytes, before version 64 from byte 44
        # as u32 every 12 bytes; the MS controller is the one with scan events
        info = f.read(824 if wide else 44)
        stride, width = (16, 8) if wide else (12, 4)
        addrs = f.read(stride * max(int.from_bytes(info[28:32], 'little'), 2))
        for a in range(0, len(addrs), stride):
            addr = int.from_bytes(addrs[a:a + width], 'little')
            f.seek(addr)
            run_header = f.read(7464)
            if addr and len(run_header) == 7464 and int.from_bytes(run_header[7376:7380], 'little'):
                break
        else:
            raise ValueError(f"{filepath} holds no scan events (e.g. SRM data of a triple quadrupole), which CandyCrunch cannot use.")
        first, last, _, n_errors = struct.unpack_from('<4I', run_header, 8)
        if wide:
            index_addr, data_addr, _, error_addr, _, trailer_addr, params_addr = struct.unpack_from('<7Q', run_header, 7408)
        else:
            (index_addr, data_addr, _, error_addr), (trailer_addr, params_addr) = struct.unpack_from('<4I', run_header, 28), struct.unpack_from('<2I', run_header, 7368)
        n_scans = last - first + 1
        # Scan index: 72 bytes per scan (88 from version 66) with the retention time (minutes) at byte 24 and the packet offset at 0 (u32), from
        # version 64 at 72 (u64)
        f.seek(index_addr)
        index = np.frombuffer(f.read((88 if version >= 66 else 80 if wide else 72) * n_scans),
                              {'names': ['offset', 'rt'], 'formats': ['<u8' if wide else '<u4', '<f8'], 'offsets': [72 if wide else 0, 24],
                               'itemsize': 88 if version >= 66 else 80 if wide else 72})
        f.seek(trailer_addr + 4)
        stream = f.read(params_addr - trailer_addr - 4)
        # The trailer extra header (a u32 field count, then per field a u32 type, a u32 length and its label) follows the error log and the method's
        # scan event templates; its records, one per scan, start at params_addr
        f.seek(error_addr + 4 * wide)
        for _ in range(n_errors):
            f.seek(4, 1)
            f.seek(2 * int.from_bytes(f.read(4), 'little'), 1)
        region = f.read(1 << 20)
        trailer, end = {}, region.find('Charge State:'.encode('utf-16-le'))
        for pos in range(0, max(end, 0), 2):
            n, fields, p = int.from_bytes(region[pos:pos + 4], 'little'), [], pos + 4
            while 2 <= n <= 500 and len(fields) < n and p + 12 <= len(region):
                kind, length, chars = struct.unpack_from('<3I', region, p)
                if kind > 13 or chars > 200:
                    break
                fields.append((kind, length, region[p + 12:p + 12 + 2 * chars].decode('utf-16-le', 'replace').split('\x00')[0]))
                p += 12 + 2 * chars
            sizes = [np.dtype(RAW_FIELD_TYPES[kind]).itemsize if kind in RAW_FIELD_TYPES else length * (1 + (kind == 13)) for kind, length, _ in fields]
            if len(fields) == n and 'Charge State:' in [label for _, _, label in fields] and n_scans * sum(sizes) <= f.seek(0, 2) - params_addr:
                # First field of each label, as the instrument's software reads them
                columns = {label: (RAW_FIELD_TYPES[kind], int(o)) for (kind, _, label), o in reversed(list(zip(fields, np.cumsum([0] + sizes))))
                           if label in RAW_TRAILER_FIELDS and kind in RAW_FIELD_TYPES}
                record_size = sum(sizes)
                f.seek(params_addr)
                raw_records = f.read(n_scans * record_size)
                records = np.frombuffer(raw_records, {'names': list(columns), 'formats': [t for t, _ in columns.values()],
                                                                        'offsets': [o for _, o in columns.values()], 'itemsize': sum(sizes)})
                trailer = {label: records[label].tolist() for label in columns}
                # HCD Energy: is text on newer orbitraps (kind 12 one byte per character, 13 UTF-16), e.g., 30.00,42.00,72.00 for stepped energies, and a
                # float on older ones, which the records above read
                energy_idx = next((i for i, (_, _, label) in enumerate(fields) if label == 'HCD Energy:'), None)
                if energy_idx is not None and fields[energy_idx][0] in (12, 13):
                    energy_offset, energy_size = sum(sizes[:energy_idx]), sizes[energy_idx]
                    trailer['HCD Energy:'] = [raw_records[i * record_size + energy_offset:i * record_size + energy_offset + energy_size].decode(
                        'utf-16-le' if fields[energy_idx][0] == 13 else 'latin-1', 'replace').split('\x00')[0].strip() or None for i in range(n_scans)]
                break
    # Scan events, one per scan, are self-describing: a preamble (120 bytes in version 62, 128 in 63 and 64, 136 from 66) with polarity at byte 4,
    # profile mode at 5, MS level at 6 and analyzer at 40, then counted reactions (isolation m/z, isolation width and energy as doubles, then the
    # activation in bits 1 and up of a u32; 32 bytes, 56 from version 66), counted scan windows (16 bytes) and counted calibration coefficients (8 bytes),
    # then a few u32 (2 before version 66, then 3, 5 in newer Q Exactive firmware), so the layout is the one that walks exactly over the stream
    preamble, reaction = {57: 80, 60: 80, 62: 120, 63: 128, 64: 128}.get(version, 136), 56 if version >= 66 else 32
    for tail in (12, 20, 8, 4, 16, 24, 28):
        starts, pos = [], 0
        while pos + preamble + 4 <= len(stream) and len(starts) < n_scans:
            starts.append(pos)
            pos += preamble + 4 + reaction * int.from_bytes(stream[pos + preamble:pos + preamble + 4], 'little')
            pos += 4 + 16 * int.from_bytes(stream[pos:pos + 4], 'little')
            pos += 4 + 8 * int.from_bytes(stream[pos:pos + 4], 'little') + tail
        if pos == len(stream) and len(starts) == n_scans:
            break
    else:
        raise ValueError(f"{filepath} stores its scan events in a layout CandyCrunch does not know; convert it to mzML with ThermoRawFileParser or "
                         f"msconvert first.")
    table = {key: [] for key in ['scan', 'ms_level', 'polarity', 'scan_mode', 'analyzer', 'filter_string', 'precursor_mz', 'charge',
                                 'isolation_width', 'master_scan', 'retention_time', 'scan_range', 'packet', 'RealEnergy']}
    for i, s in enumerate(starts):
        level = max(stream[s + 6], 1)
        # MS1 events of some orbitraps carry their scan range as a reaction
        reactions = [struct.unpack_from('<3dI', stream, s + preamble + 4 + reaction * k)
                     for k in range(int.from_bytes(stream[s + preamble:s + preamble + 4], 'little'))] if level > 1 else []
        mono, charge, ms2_width, msn_width, master, real_energy = (trailer[label][i] if label in trailer and level > 1 else None for label in RAW_TRAILER_FIELDS)
        table['scan'].append(first + i)
        table['ms_level'].append(level)
        table['polarity'].append({0: '-', 1: '+'}.get(stream[s + 4]))
        table['scan_mode'].append({0: 'centroid', 1: 'profile'}.get(stream[s + 5]))
        table['analyzer'].append({0: 'ITMS', 1: 'TQMS', 2: 'SQMS', 3: 'TOFMS', 4: 'FTMS', 5: 'Sector'}.get(stream[s + 40]))
        # A supplemental activation (EThcD, ETciD) is a further reaction without an m/z of its own or with the stage's m/z
        stages = []
        for mz, _, energy, code in reactions:
            if stages and (mz <= 0 or mz == stages[-1][0]):
                stages[-1][1] += f"@{RAW_ACTIVATIONS.get(code >> 1, 'cid')}{energy:.2f}"
            elif mz > 0:
                stages.append([mz, f"@{RAW_ACTIVATIONS.get(code >> 1, 'cid')}{energy:.2f}"])
        table['filter_string'].append(' '.join(f'{mz:.4f}{activations}' for mz, activations in stages))
        # The instrument's monoisotopic m/z where it determined one (else 0), otherwise the isolation m/z of the last stage
        table['precursor_mz'].append(mono if mono and mono > 0 else reactions[-1][0] if reactions else None)
        table['charge'].append(int(charge) if charge and charge > 0 else None)
        table['isolation_width'].append(ms2_width or msn_width)
        table['master_scan'].append(master)
        table['retention_time'].append(float(index['rt'][i]))
        table['RealEnergy'].append(real_energy)
        # The scan windows (two doubles each) after the reactions
        w = s + preamble + 4 + reaction * int.from_bytes(stream[s + preamble:s + preamble + 4], 'little')
        windows = [struct.unpack_from('<2d', stream, w + 4 + 16 * k) for k in range(int.from_bytes(stream[w:w + 4], 'little'))]
        table['scan_range'].append((min(lo for lo, _ in windows), max(hi for _, hi in windows)) if windows else None)
        table['packet'].append(data_addr + int(index['offset'][i]))
    return table


def read_raw_packet(f, address, drop_flagged = False):
    """reads the data packet of one scan of a Thermo .raw file\n
   | Arguments:
   | :-
   | f (file): the .raw file, opened in binary mode
   | address (int): file offset of the scan's data packet (the scan table's packet)
   | drop_flagged (bool): whether to leave out the orbitrap centroids that Thermo's centroid stream leaves out (flag 0x10: FT side lobes and ghosts); default:False\n
   | Returns:
   | :-
   | Returns m/z (float64) and intensity of the scan's centroids (float32), or if it has none, of its profile points (float64) as stored (one bin
   | below the axis of Thermo's own export; orbitrap profiles are in frequency), and whether these are profile points
   """
    # 40-byte header (number of scan segments, then the sizes of the profile and the centroid list in 4-byte words and the layout flags), 8 bytes
    # (m/z range) per further segment, the profile, then the centroid list: its length, then float32 m/z (double with layout flag 0x10000) and
    # float32 intensity per centroid; orbitrap centroids are followed by one descriptor per centroid (u16 index, u8 flags, u8 charge)
    f.seek(address)
    n_segments, profile_size, peaks_size, layout = struct.unpack_from('<4I', f.read(40))
    profile_addr = f.seek(8 * max(n_segments - 1, 0), 1)
    if peaks_size:
        f.seek(4 * profile_size, 1)
        n = int.from_bytes(f.read(4), 'little')
        peaks = np.frombuffer(f.read(n * (12 if layout & 0x10000 else 8)), [('mz', '<f8' if layout & 0x10000 else '<f4'), ('int', '<f4')])
        if n and drop_flagged:
            # Ion-trap centroids have no descriptors, which the index check tells
            descriptors = f.read(4 * n)
            descriptors = np.frombuffer(descriptors[:len(descriptors) // 4 * 4], '<u4')
            keep = (descriptors & 0x100000) == 0 if len(descriptors) == n and np.array_equal(descriptors & 0xFFFF, np.arange(n) & 0xFFFF) else slice(None)
            return peaks['mz'][keep].astype(np.float64), peaks['int'][keep].copy(), False
        if n:
            return peaks['mz'].astype(np.float64), peaks['int'].copy(), False
    # Profile: first value and step of the bin axis (doubles), the number of chunks, then per chunk its first bin and number of bins (u32), a
    # float32 m/z offset if the layout has one, and its intensities (float32)
    f.seek(profile_addr)
    prof = f.read(4 * profile_size)
    if len(prof) < 24:
        return np.zeros(0), np.zeros(0), False
    (first_value, step), fudge, pos, mzs, ints = struct.unpack_from('<2d', prof), layout & 0xFF > 0, 24, [np.zeros(0)], [np.zeros(0)]
    for _ in range(int.from_bytes(prof[16:20], 'little')):
        first_bin, n = struct.unpack_from('<2I', prof, pos)
        mzs.append(first_value + np.arange(first_bin, first_bin + n) * step + (np.frombuffer(prof, '<f4', 1, pos + 8)[0] if fudge else 0))
        ints.append(np.frombuffer(prof, '<f4', n, pos + 8 + 4 * fudge))
        pos += 8 + 4 * fudge + 4 * n
    return np.concatenate(mzs), np.concatenate(ints), True


# Agilent MassHunter scan index (AcqData/MSScan.bin): one 284-byte record per scan with the fields of the folder's MSScan.xsd in its order (ScanID,
# ScanMethodID, TimeSegmentID, ScanTime in min, MSLevel, ScanType, TIC, base peak m/z and value, CalibrationID, CycleNumber, IonMode, IonPolarity (0 positive,
# 1 negative), Fragmentor, CollisionEnergy, MzOfInterest, AbundanceLimit, SamplingPeriod, Threshold, ChargeState, then ChromScaleFactor as 4 bytes, not the
# schema's 8, MassCalOffset, ...), ending in a profile and a centroid SpectrumParamValues block (format ID 1 and 2: offset into MSProfile.bin / MSPeak.bin,
# byte and point count, x and y range, noise); of them the fields read here and CalibrationID (DefaultMassCal.xml's DefaultCalibrationID). Scans of a time
# segment stored as centroids only list their centroid block first and an empty profile block second (MASSHUNTER_FIRST_BLOCK)
MASSHUNTER_SCAN = np.dtype({'names': ['scan', 'rt', 'level', 'polarity', 'energy', 'mz', 'charge', 'calibration', 'format', 'offset', 'n', 'calibration_id'],
                            'formats': ['<i4', '<f8', '<i4', '<i4', '<f8', '<f8', '<i4', '<i8', '<i4', '<i8', '<i4', '<i4'],
                            'offsets': [0, 12, 20, 64, 76, 84, 116, 124, 220, 224, 236, 52], 'itemsize': 284})
MASSHUNTER_FIRST_BLOCK = np.dtype({'names': ['format', 'offset', 'n'], 'formats': ['<i4', '<i8', '<i4'], 'offsets': [156, 160, 172], 'itemsize': 284})


def read_masshunter_scans(folder):
    """reads the scan index of an Agilent MassHunter .d folder (AcqData/MSScan.bin of the 6200 series TOF / 6500 series Q-TOF acquisition software)\n
   | Arguments:
   | :-
   | folder (string): absolute path to the .d folder\n
   | Returns:
   | :-
   | Returns a structured array (MASSHUNTER_SCAN) with one row per scan: scan, rt (min), level, polarity, energy (collision energy), mz (an MS/MS scan's
   | precursor, MassHunter's isolation target), charge (0 if not determined), calibration (the offset of the scan's record in MSMassCal.bin), format (2 where
   | the scan has a centroided spectrum), offset and n (its place and point count in MSPeak.bin, from whichever of the record's two blocks is the centroid one)
   | and calibration_id (its default calibration in DefaultMassCal.xml)
   """
    name = os.path.basename(os.path.normpath(folder))
    with open(os.path.join(folder, 'AcqData', 'MSScan.bin'), 'rb') as f:
        index = f.read()
    # MSTS.xml gives the number of scans of each time segment, the header's int32 at byte 88 the offset of the first record
    with open(os.path.join(folder, 'AcqData', 'MSTS.xml'), 'rb') as f:
        n = sum(int(x) for x in re.findall(rb'<NumOfScans>(\d+)</NumOfScans>', f.read()))
    start = struct.unpack_from('<i', index, 88)[0]
    if not n or len(index) - start != n * MASSHUNTER_SCAN.itemsize:
        raise ValueError(f"{name} has an MSScan.bin layout CandyCrunch does not know; convert it to mzML with msconvert (with peak picking) first.")
    scans = np.frombuffer(index, dtype = MASSHUNTER_SCAN, count = n, offset = start).copy()
    # The time segments of a method store 'Both' or 'Centroid' (or 'Profile'); a centroid-only scan has its centroid block first
    first = np.frombuffer(index, dtype = MASSHUNTER_FIRST_BLOCK, count = n, offset = start)
    centroid_first = (scans['format'] != 2) & (first['format'] == 2)
    for key in ('format', 'offset', 'n'):
        scans[key][centroid_first] = first[key][centroid_first]
    return scans


def read_masshunter(folder):
    """iterates over the scans of an Agilent MassHunter .d folder, from the centroided spectra MassHunter stores next to its profile spectra\n
   | Arguments:
   | :-
   | folder (string): absolute path to the .d folder\n
   | Returns:
   | :-
   | Yields a dict per scan: scan, ms_level, rt (min), peaks ((n, 2) array of m/z and abundance), precursor (mz and charge, None for survey scans; the
   | charge None where MassHunter determined none), polarity ('+'/'-') and collision_energy
   """
    name = os.path.basename(os.path.normpath(folder))
    scans = read_masshunter_scans(folder)
    # A time segment stored as profiles only has no centroids: its scans are skipped, unless they hold all scans of an MS level (a profile-only run)
    if not os.path.exists(os.path.join(folder, 'AcqData', 'MSPeak.bin')) or not set(scans['level'].tolist()) <= set(scans['level'][scans['format'] == 2].tolist()):
        raise ValueError(f"{name} holds profile spectra only (no AcqData/MSPeak.bin); convert it to mzML with msconvert (with peak picking) first.")
    scans = scans[scans['format'] == 2]
    with open(os.path.join(folder, 'AcqData', 'MSPeak.bin'), 'rb') as f:
        peaks = f.read()
    with open(os.path.join(folder, 'AcqData', 'MSMassCal.bin'), 'rb') as f:
        calibration = f.read()
    # The powers of t of each default calibration's 'Polynomial' step are the bits of its ValueUseFlags (126, powers 1-6, on a 6550)
    powers = {}
    if os.path.exists(os.path.join(folder, 'AcqData', 'DefaultMassCal.xml')):
        with open(os.path.join(folder, 'AcqData', 'DefaultMassCal.xml'), 'rb') as f:
            for number, body in re.findall(rb'<DefaultCalibration DefaultCalibrationID="(\d+)">(.*?)</DefaultCalibration>', f.read(), re.S):
                flags = re.search(rb'<CalibrationFormula>Polynomial</CalibrationFormula>.*?<ValueUseFlags>(\d+)</ValueUseFlags>', body, re.S)
                powers[int(number)] = [k for k in range(32) if flags and int(flags.group(1)) >> k & 1]
    # MSPeak.bin holds each scan's centroids as their float64 flight times, then their float32 abundances; MSMassCal.bin each scan's calibration (a count,
    # then the 'Traditional' step's A and t0 and the 'Polynomial' step's flight time range and coefficients): m/z = (A (t - t0))^2 with the A of the scan's
    # reference-mass correction, minus the polynomial in t, clipped to its range: as MassHunter's base peak m/z of centroid-only 6550 scans (to 0.5 ppm on
    # 3205 of 3205), and it puts a 6530's reference masses 119.0363 and 966.0007 at 0.00 / +0.01 ppm (-5.09 / -0.83 ppm without; up to 10 ppm at m/z 200-500)
    for s in scans:
        times = np.frombuffer(peaks, '<f8', s['n'], s['offset'])
        count = struct.unpack_from('<i', calibration, s['calibration'])[0]
        a, t0, *step = struct.unpack_from(f'<{count}d', calibration, s['calibration'] + 4)
        mzs, exponents = (a * (times - t0)) ** 2, powers.get(int(s['calibration_id']), [])
        if exponents and len(step) == 2 + len(exponents):
            mzs -= sum(c * np.clip(times, step[0], step[1]) ** k for c, k in zip(step[2:], exponents))
        yield dict(scan = int(s['scan']), ms_level = int(s['level']), rt = float(s['rt']),
                   peaks = np.column_stack([mzs, np.frombuffer(peaks, '<f4', s['n'], s['offset'] + 8 * s['n'])]),
                   precursor = dict(mz = float(s['mz']), charge = int(s['charge']) or None) if s['level'] > 1 and s['mz'] > 0 else None,
                   polarity = '-' if s['polarity'] == 1 else '+', collision_energy = float(s['energy']))


def read_compound_file(filepath):
    """reads the streams of an OLE compound file (Microsoft's structured storage, the container of SCIEX .wiff files)\n
   | Arguments:
   | :-
   | filepath (string): absolute path to the file\n
   | Returns:
   | :-
   | Returns a dict of stream paths ('Storage/Substorage/Stream') to their bytes
   """
    with open(filepath, 'rb') as f:
        data = f.read()
    if data[:8] != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        raise ValueError(f"{os.path.basename(filepath)} is no OLE compound file (a .wiff is one)")
    sector, mini = 1 << struct.unpack_from('<H', data, 30)[0], 1 << struct.unpack_from('<H', data, 32)[0]
    n_fat, first_dir, cutoff, first_minifat, n_minifat, difat_sector, n_difat = struct.unpack_from('<2I4x5I', data, 44)
    # The sectors of the allocation table: 109 listed in the header, the rest in a chain of DIFAT sectors whose last entry points to the next one
    difat = list(struct.unpack_from('<109I', data, 76))
    for _ in range(n_difat):
        entries = struct.unpack_from(f'<{sector // 4}I', data, (difat_sector + 1) * sector)
        difat, difat_sector = difat + list(entries[:-1]), entries[-1]
    fat = np.concatenate([np.frombuffer(data, '<u4', sector // 4, (s + 1) * sector) for s in difat[:n_fat]])

    def chain(start, table):
        # Sector numbers of a stream up to the end-of-chain mark (values from 0xFFFFFFFA up are marks); a damaged file's cycle stops at the table's length
        out = []
        while start < 0xFFFFFFFA and len(out) <= len(table):
            out.append(start)
            start = int(table[start])
        return out

    read = lambda start: b''.join(data[(s + 1) * sector:(s + 2) * sector] for s in chain(start, fat))
    directory = read(first_dir)
    # Directory entries of 128 bytes: UTF-16 name and its byte length, type (1 storage, 2 stream, 5 root), left and right sibling and child in a red-black
    # tree, first sector and size (of which version 3 files, with 512-byte sectors, only use the lower 4 bytes)
    entries = [(directory[k:k + struct.unpack_from('<H', directory, k + 64)[0]].decode('utf-16-le').rstrip('\x00'), directory[k + 66],
                *struct.unpack_from('<3i', directory, k + 68), *struct.unpack_from('<IQ', directory, k + 116)) for k in range(0, len(directory) - 127, 128)]
    ministream = read(entries[0][5])[:entries[0][6]]
    minifat = np.frombuffer(read(first_minifat), '<u4') if n_minifat else np.zeros(0, '<u4')
    streams, stack = {}, [(entries[0][4], '')]
    while stack:
        i, prefix = stack.pop()
        if not 0 <= i < len(entries):
            continue
        name, kind, left, right, child, start, size = entries[i]
        size &= 0xFFFFFFFF if sector == 512 else 0xFFFFFFFFFFFFFFFF
        stack += [(left, prefix), (right, prefix)] + ([(child, prefix + name + '/')] if kind == 1 else [])
        if kind == 2:
            # Streams below the cutoff (4096 bytes) live in the 64-byte sectors of the root's mini stream
            streams[prefix + name] = (b''.join(ministream[s * mini:(s + 1) * mini] for s in chain(start, minifat)) if size < cutoff else read(start))[:size]
    return streams


# SCIEX .wiff scan index (SampleSubtree/Sample1/Idx, after 32 bytes of header): one 54-byte record per scan, cycle by cycle and within a cycle experiment by
# experiment, with the offset and size of its spectrum in the .wiff.scan (0 bytes for a dependent scan without precursor), its time in ms, TIC, base peak
# intensity and base peak TOF bin
WIFF_SCAN = np.dtype({'names': ['offset', 'size', 'rt', 'tic', 'base_peak', 'base_peak_bin'], 'formats': ['<u4', '<u4', '<f8', '<f8', '<f8', '<f8'],
                      'offsets': [0, 4, 8, 18, 26, 42], 'itemsize': 54})


def read_wiff(filepath):
    """iterates over the scans of a SCIEX .wiff file (Analyst, information-dependent acquisition on a TripleTOF or QSTAR), from the .wiff.scan beside it\n
   | Arguments:
   | :-
   | filepath (string): absolute path to the .wiff file\n
   | Returns:
   | :-
   | Yields a dict per scan: scan (its row in the scan index), cycle and experiment (1-based, as SCIEX and pwiz number them), ms_level, rt (min), peaks
   | ((n, 2) array of centroid m/z and intensity), precursor (mz and i of the survey peak IDA chose, None for survey scans) and scan_low (the lower end
   | of the experiment's mass range); dependent scans without a precursor are skipped
   """
    name = os.path.basename(filepath)
    if not os.path.exists(filepath + '.scan'):
        raise ValueError(f"{name} has no {name}.scan beside it; SCIEX keeps the spectra of a .wiff file in its .wiff.scan, so both have to be copied.")
    streams = read_compound_file(filepath)
    samples = {m.group(1) for k in streams if (m := re.match(r'SampleSubtree/(Sample\d+)/', k))}
    periods = {m.group(1) for k in streams if (m := re.match(r'MethodSubtree/Method1/DeviceMethod\d+/(Period\d+)/', k))}
    if samples != {'Sample1'} or periods != {'Period0'}:
        raise ValueError(f"{name} holds {len(samples)} samples acquired in {len(periods)} periods; CandyCrunch reads .wiff files of one sample and one period.")
    sample = {k.split('/', 2)[2]: v for k, v in streams.items() if k.startswith('SampleSubtree/Sample1/')}
    headers = {int(m.group(1)): v for k, v in streams.items() if (m := re.match(r'MethodSubtree/Method1/DeviceMethod\d+/Period0/Experiment(\d+)/ExperimentHeaderEx$', k))}
    if 'TDCInfo' not in sample or 'TOFCalibrationData' not in sample or 'Idx' not in sample:
        raise ValueError(f"{name} holds no TOF spectra (a triple quadrupole or QTRAP run); CandyCrunch reads the .wiff files of SCIEX TOF instruments.")
    n_exp = len(headers)
    # Information-dependent acquisition: after a 32-byte header, a 32-byte slot per dependent scan of the index (cycle by cycle, experiment by experiment)
    # with the m/z and intensity of the survey peak it fragmented (float64 at bytes 0 and 16; zero where IDA found no candidate), up to the last dependent
    # scan acquired: a run stopped within a cycle ends with part of one (a TripleTOF 5600+ run of 2337 cycles of 16 experiments, then a survey scan and one
    # dependent scan, has 35,056 slots). No charge state is stored
    dde = sample.get('DDERealTimeData', b'')
    rows, n_slots = (len(sample['Idx']) - 32) // WIFF_SCAN.itemsize, (len(dde) - 32) // 32
    # A stream with more slots than the index has dependent scans has another layout
    if (n_exp < 2 or sorted(headers) != list(range(n_exp)) or len(dde) < 32 or (len(dde) - 32) % 32 or (len(sample['Idx']) - 32) % WIFF_SCAN.itemsize or
            n_slots > rows - -(-rows // n_exp)):
        raise ValueError(f"{name} has no information-dependent MS/MS scans CandyCrunch can read (TOF MS only, SWATH or an unknown layout); CandyCrunch needs "
                         f"data-dependent MS2 spectra.")
    slots = np.zeros((-(-rows // n_exp), n_exp - 1, 4))
    slots.reshape(-1, 4)[:n_slots] = np.frombuffer(dde, '<f8', n_slots * 4, 32).reshape(-1, 4)
    scans = np.frombuffer(sample['Idx'], WIFF_SCAN, offset = 32)
    # The flight time of a TOF bin in ns (25.1 ps on a TripleTOF 5600), and per experiment the number of bins summed into one stored point (4 for TOF MS,
    # 4 or 8 for product ion scans) and its mass range (float64 at 66)
    bin_ns = struct.unpack_from('<d', sample['TDCInfo'], 32)[0]
    summed = [struct.unpack_from('<I', headers[e], 56)[0] for e in range(n_exp)]
    scan_low = [struct.unpack_from('<d', headers[e], 66)[0] for e in range(n_exp)]
    # TOF calibration m/z = (A (t - t0))^2, t in ns: a default (A, t0) at byte 32, then a count and, where the instrument recalibrated each scan (TripleTOF
    # survey and product ion scans each have their own), one (A, t0, int32) per scan of the index
    calibration = sample['TOFCalibrationData']
    default = struct.unpack_from('<2d', calibration, 32)
    n_calibrations = struct.unpack_from('<I', calibration, 52)[0] if len(calibration) >= 56 else 0
    per_scan = np.frombuffer(calibration, np.dtype({'names': ['a', 't0'], 'formats': ['<f8', '<f8'], 'itemsize': 20}), len(scans), 56) if n_calibrations == len(scans) else None
    with open(filepath + '.scan', 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    # A spectrum in the .wiff.scan (after 44 bytes of header): FF FF FF FF, the TOF bin its points count from (uint32), a zero byte, then its profile as a
    # byte code: below 7C an intensity, 7C/7D/7E an intensity in the next 1/2/4 bytes, 80-FB a gap of 0-123 empty points, FC/FD/FE a gap in the next 1/2/4
    # bytes, FF the end (FF padding to 4 bytes follows)
    has = np.flatnonzero(scans['size'] > 0)
    starts = scans['offset'][has].astype(np.int64) + 44
    ends = starts + scans['size'][has]
    if len(has) and (ends[-1] > len(data) or (np.diff(starts) <= 0).any() or not (data[starts[:, None] + np.arange(4)] == 0xFF).all()):
        raise ValueError(f"{name}.scan does not hold the spectra its scan index points to (an incomplete copy?).")
    first_bin = (data[starts[:, None] + np.arange(4, 8)].astype(np.int64) << np.arange(0, 32, 8)).sum(1)
    # Operand bytes take any value, so the tokens are walked from each spectrum's start, all spectra at once, one token per step
    length = np.ones(256, np.int64)
    length[[0x7C, 0xFC]], length[[0x7D, 0xFD]], length[[0x7E, 0xFE]] = 2, 3, 5
    token, at, stop = np.zeros(len(data), bool), starts + 9, ends
    while len(at):
        going = at < stop
        going[going] = data[at[going]] != 0xFF
        at, stop = at[going], stop[going]
        token[at] = True
        at = at + length[data[at]]
    # Decoded in chunks of ~4 MB of spectra, as the ~70 million tokens of a 90 MB .wiff.scan take gigabytes as arrays
    chunks = np.flatnonzero(np.diff((starts - starts[0]) >> 22, prepend = -1, append = -1)) if len(has) else []
    for k0, k1 in zip(chunks[:-1], chunks[1:]):
        pos = np.flatnonzero(token[starts[k0]:ends[k1 - 1]]) + starts[k0]
        code = data[pos]
        if (code == 0x7F).any():
            raise ValueError(f"{name}.scan holds spectra in a code CandyCrunch does not know.")
        first = np.searchsorted(pos, np.append(starts[k0:k1], ends[k1 - 1]))
        gap, low = code >= 0x80, (code & 0x7F).astype(np.int64)
        value = np.where(gap, low, code).astype(np.int64)
        wide = np.flatnonzero((low >= 0x7C) & (low <= 0x7E))
        value[wide] = 0
        for j in range(4):
            # Little-endian operands: 1, 2 or 4 bytes for 7C/FC, 7D/FD, 7E/FE
            take = wide[(low[wide] - 0x7B).clip(max = 3) + (low[wide] == 0x7E) > j]
            value[take] |= data[np.minimum(pos[take] + 1 + j, len(data) - 1)].astype(np.int64) << 8 * j
        # A point is one step, a gap that many steps; the k-th point of a spectrum lies k + 1 steps of summed bins after its first bin (the scan index's base
        # peak bin is exactly that)
        steps = np.concatenate([[0], np.cumsum(np.where(gap, value, 1))])
        before = steps[:-1] - np.repeat(steps[first[:-1]], np.diff(first).astype(np.intp))
        point_steps, point_ints, bounds = before[~gap], value[~gap], np.concatenate([[0], np.cumsum(~gap)])[first]
        for k in range(k0, k1):
            row = int(has[k])
            cycle, experiment = divmod(row, n_exp)
            precursor = None
            if experiment:
                if cycle >= len(slots) or not slots[cycle, experiment - 1, 0] > 0:
                    continue
                precursor = dict(mz = float(slots[cycle, experiment - 1, 0]), i = float(slots[cycle, experiment - 1, 2]))
            a, t0 = (per_scan['a'][row], per_scan['t0'][row]) if per_scan is not None else default
            t = (first_bin[k] + summed[experiment] * (point_steps[bounds[k - k0]:bounds[k - k0 + 1]] + 1)) * bin_ns
            # The profile is centroided as TOF data (points evenly spaced in sqrt(m/z)); survey scans keep their TIMS_MS1_PEAKS most intense centroids
            peaks = centroid_tof((a * (t - t0)) ** 2, point_ints[bounds[k - k0]:bounds[k - k0 + 1]], max_peaks = None if experiment else TIMS_MS1_PEAKS)
            yield dict(scan = row, cycle = cycle + 1, experiment = experiment + 1, ms_level = 2 if experiment else 1, rt = float(scans['rt'][row]) / 60000,
                       peaks = peaks, precursor = precursor, scan_low = scan_low[experiment])


def read_baf(folder, peaks = True):
    """iterates over the spectra of a Bruker .d folder of an otofControl Q-TOF (maXis, impact, compact, micrOTOF-Q, timsTOF without ion mobility; with an
    analysis.baf), from the line (centroided) spectra the instrument stores next to its profile spectra\n
   | Arguments:
   | :-
   | folder (string): absolute path to the .d folder
   | peaks (bool): whether to read the spectra or only their headers (peaks then None); default:True\n
   | Returns:
   | :-
   | Yields a dict per spectrum: scan (Bruker's spectrum number), ms_level, rt (min), peaks ((n, 2) array of centroid m/z and intensity), precursor (mz, the
   | isolation mass, and charge, None for survey scans; the charge None where Bruker determined none), polarity ('+'/'-'), collision_energy,
   | isolation_width and scan_low (the lower end of the acquisition range)
   """
    name = os.path.basename(os.path.normpath(folder))
    # analysis.baf is a chain of blocks (uint32 byte length, uint16 type, the signature A0 BF, then a uint32 header length): 3001 defines the acquisition
    # parameters (per entry its byte length, number, value type and flags, then the offsets of its names, the short name at byte 20), 2001 holds the
    # method's parameter values in sets of (uint16 number, 8-byte value) records, one set per acquisition key; per spectrum a 2002 header, a 1001 profile
    # and a 1002 line spectrum follow. Sub-blocks have the same frame
    with open(os.path.join(folder, 'analysis.baf'), 'rb') as f:
        size, pos, ids, method, spectrum, headers, n_read = os.fstat(f.fileno()).st_size, 0, None, {}, None, 0, 0
        while pos + 8 <= size:
            f.seek(pos)
            length, kind = struct.unpack('<2I', f.read(8))
            if length < 8 or kind >> 16 != 0xbfa0:
                break
            kind &= 0xffff
            block = f.read(length - 8) if kind in (0x3001, 0x2001, 0x2002) or (kind == 0x1002 and peaks and spectrum) else b''
            subs, j = {}, struct.unpack_from('<I', block, 0)[0] - 8 if kind in (0x2001, 0x2002) else len(block)
            while j + 8 <= len(block):
                sub_length, sub_kind = struct.unpack_from('<2I', block, j)
                if sub_length < 8 or sub_kind >> 16 != 0xbfa0:
                    break
                subs.setdefault(sub_kind & 0xffff, block[j:j + sub_length])
                j += sub_length
            if kind == 0x3001:
                ids = {}
                for key in ('Mode_IonPolarity', 'Mode_ScanBegin', 'MSMS_IsolationMass_Act', 'MSMS_PreCursorChargeState', 'Collision_Energy_Act',
                            'Quadrupole_IsolationResolution_Act', 'Calibration_Tof2CalC1', 'TOF_DeviceTempGradient1', 'TOF_DeviceTempGradient2'):
                    i = next((m.start() for m in re.finditer(re.escape(key.encode() + b'\x00'), block) if m.start() >= 20 and
                              struct.unpack_from('<H', block, m.start() - 12)[0] == 20), None)
                    if i is not None:
                        ids[key] = struct.unpack_from('<H', block, i - 18)[0]
            elif kind == 0x2001 and 0x000d in subs:
                # The first set of the method's values (sub-block 000D: header length, number of records, acquisition key)
                header, n = struct.unpack_from('<2I', subs[0x000d], 8)
                method = {struct.unpack_from('<H', subs[0x000d], j)[0]: subs[0x000d][j + 2:j + 10] for j in range(header, header + 10 * n, 10)}
            elif kind == 0x2002 and ids is not None:
                # Spectrum header: segment, spectrum number, ..., RT in ms, ..., acquisition key; its sub-blocks are 1 the spectrum's parameter values
                # (header length, number of records, records as in 2001), 5 the precursor list (count, then a float32 m/z per MS level above 1) and 6
                # the TOF calibration
                values = dict(method)
                if 1 in subs:
                    header, n = struct.unpack_from('<2I', subs[1], 8)
                    values.update({struct.unpack_from('<H', subs[1], j)[0]: subs[1][j + 2:j + 10] for j in range(header, header + 10 * n, 10)})

                def value(key, code = '<d'):
                    return struct.unpack(code, values[ids[key]])[0] if ids.get(key) in values else None

                levels = struct.unpack_from('<I', subs[5], 12)[0] if 5 in subs else 0
                mz = value('MSMS_IsolationMass_Act') or (struct.unpack_from('<f', subs[5], 16)[0] if levels else None)
                spectrum = dict(scan = struct.unpack_from('<I', block, 8)[0], ms_level = levels + 1, rt = struct.unpack_from('<I', block, 16)[0] / 60000,
                                peaks = None, precursor = dict(mz = mz, charge = value('MSMS_PreCursorChargeState', '<q') or None) if levels and mz else None,
                                polarity = '-' if value('Mode_IonPolarity', '<q') == 1 else '+', collision_energy = value('Collision_Energy_Act'),
                                isolation_width = value('Quadrupole_IsolationResolution_Act') if levels else None, scan_low = value('Mode_ScanBegin'))
                calibration, headers = subs.get(6), headers + 1
                if not peaks:
                    yield spectrum
                    spectrum = None
            elif kind == 0x1002 and spectrum and struct.unpack_from('<I', block, 4)[0] == spectrum['scan']:
                # Line spectrum: after a 44-byte header (number of arrays at 12, of lines at 16), the line count and the offsets of the arrays from there:
                # the lines' float64 positions on the digitizer's time axis (in sample steps), their float32 intensities and signal to noise ratios
                n, (_, at_positions, at_intensities) = struct.unpack_from('<I', block, 16)[0], struct.unpack_from('<3I', block, 44)
                positions = np.frombuffer(block, '<f8', n, 44 + at_positions)
                # TOF calibration as for timsTOF frames: digitizer delay and time base, then C0-C4; flight time t = C0 + sqrt(1e12 / C1 * (m/z + C4)) +
                # C2 * m/z, solved in closed form. The spectrum's C1 carries the temperature compensation as the method's C1 times 1 + e: m/z come from
                # the method's C1, divided by 1 + e, as Bruker's DataAnalysis gives them (with the spectrum's C1 they drift by 2 e, ppm over a run);
                # a method C1 far from the spectrum's belongs to another calibration and is not used. Where the method sets temperature gradients
                # (TOF_DeviceTempGradient1/2, ppm/C: 45 and 0 on a compact, 48.7 and 22.9 on a maXis, 0 on a timsTOF), e is the instrument's own
                # compensation from its device temperatures and the spectrum's C1 the calibration it used: its isolation masses equal the survey lines
                # with it (median 0.2 / 0.01 ppm), against 175 / 99 ppm off with the method's C1 divided by 1 + e
                if calibration is None or len(calibration) < 72 or struct.unpack_from('<d', calibration, 56)[0]:
                    raise ValueError(f"{name} uses a mass calibration CandyCrunch does not know (not a Bruker Q-TOF); convert it to mzML with msconvert first.")
                delay, timebase, c0, c1, c2, _, c4 = struct.unpack_from('<7d', calibration, 16)
                reference = value('Calibration_Tof2CalC1')
                compensated = value('TOF_DeviceTempGradient1') or value('TOF_DeviceTempGradient2')
                reference = reference if reference and not compensated and abs(reference / c1 - 1) < 1e-4 else c1
                a, q = np.sqrt(1e12 / reference), positions * timebase + delay - c0 + c2 * c4
                x = 2 * q / (a + np.sqrt(a * a + 4 * c2 * q))
                spectrum['peaks'] = np.column_stack([(x * x - c4) * reference / c1, np.frombuffer(block, '<f4', n, 44 + at_intensities)])
                yield spectrum
                spectrum, n_read = None, n_read + 1
            pos += length
    if ids is None:
        raise ValueError(f"{name} has no readable analysis.baf; convert it to mzML with msconvert first.")
    if peaks and headers and not n_read:
        raise ValueError(f"{name} holds profile spectra only (no line spectra in its analysis.baf); convert it to mzML with msconvert (with peak picking) first.")


def read_tsf(folder, peaks = True):
    """iterates over the spectra of a Bruker timsTOF .d folder acquired without ion mobility (timsTOF fleX in ESI mode, with an analysis.tsf and
    analysis.tsf_bin), from the line (centroided) spectra the instrument stores\n
   | Arguments:
   | :-
   | folder (string): absolute path to the .d folder
   | peaks (bool): whether to read the spectra or only their headers (peaks then None); default:True\n
   | Returns:
   | :-
   | Yields a dict per spectrum as read_baf: scan (the frame number), ms_level, rt (min), peaks ((n, 2) array of centroid m/z and intensity), precursor (mz,
   | the trigger mass, and charge, None for survey scans; the charge None where Bruker determined none), polarity ('+'/'-'), collision_energy,
   | isolation_width and scan_low (the lower end of the acquisition range)
   """
    name = os.path.basename(os.path.normpath(folder))
    # analysis.tsf is an SQLite database like a timsTOF's analysis.tdf, with one frame per spectrum and no ion mobility scans; read-only, so nothing is
    # written next to the data
    db = sqlite3.connect(pathlib.Path(folder, 'analysis.tsf').resolve().as_uri() + '?mode=ro', uri = True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        metadata = dict(db.execute('SELECT Key, Value FROM GlobalMetadata').fetchall())
        maldi = 'MaldiFrameInfo' in tables and db.execute('SELECT COUNT(*) FROM MaldiFrameInfo').fetchone()[0]
        frames = db.execute('SELECT Id, Time, Polarity, MsMsType, TimsId, NumPeaks, MzCalibration, T1, T2 FROM Frames ORDER BY Id').fetchall()
        calibrations = {row[0]: row[1:] for row in db.execute('SELECT Id, ModelType, DigitizerTimebase, DigitizerDelay, T1, T2, dC1, dC2, C0, C1, C2, C3, C4 FROM MzCalibration')}
        msms = {row[0]: row[1:] for row in db.execute('SELECT Frame, TriggerMass, IsolationWidth, PrecursorCharge, CollisionEnergy FROM FrameMsMsInfo')} if 'FrameMsMsInfo' in tables else {}
    finally:
        db.close()
    if maldi:
        raise ValueError(f"{name} is a MALDI run; CandyCrunch reads LC-MS/MS runs.")
    if metadata.get('HasLineSpectra') != '1':
        raise ValueError(f"{name} holds profile spectra only (no line spectra in its analysis.tsf_bin); convert it to mzML with msconvert (with peak picking) first.")
    if metadata.get('TimsCompressionType') != '3' or any(cal[0] != 1 or cal[10] for cal in calibrations.values()):
        raise ValueError(f"{name} stores its spectra in a format CandyCrunch does not know; convert it to mzML with msconvert first.")
    scan_low = float(metadata['MzAcqRangeLower']) if metadata.get('MzAcqRangeLower') else None
    if peaks:
        try:
            from compression.zstd import decompress as zstd_decompress
        except ImportError:
            from zstandard import decompress as zstd_decompress
        f = open(os.path.join(folder, 'analysis.tsf_bin'), 'rb')
    try:
        for frame, time, polarity, kind, offset, n, cal_id, t1, t2 in frames:
            # Survey frames are MsMsType 0, MS/MS frames are listed in FrameMsMsInfo (trigger mass, isolation width, charge, collision energy with the
            # polarity's sign); other frames are left out
            if kind and frame not in msms:
                continue
            spectrum = dict(scan = frame, ms_level = 2 if kind else 1, rt = time / 60, peaks = None, precursor = None, polarity = polarity,
                            collision_energy = None, isolation_width = None, scan_low = scan_low)
            if kind:
                mz, width, charge, energy = msms[frame]
                spectrum.update(precursor = dict(mz = mz, charge = charge or None), collision_energy = abs(energy), isolation_width = width)
            if peaks:
                # A frame in analysis.tsf_bin is its byte size, the size of its zstd block, then the block: the lines' float64 positions on the digitizer's
                # time axis (in sample steps), then their float32 intensities
                lines = np.zeros((0, 2))
                if n:
                    f.seek(offset)
                    size = struct.unpack('<2I', f.read(8))[0]
                    raw = zstd_decompress(f.read(size - 8))
                    if len(raw) != 12 * n:
                        raise ValueError(f"{name} stores its spectra in a format CandyCrunch does not know; convert it to mzML with msconvert first.")
                    # TOF calibration as for timsTOF frames (model 1): flight time t = C0 + sqrt(1e12 / C1 * (m/z + C4)) + C2 * m/z, solved in closed form,
                    # then the linear temperature correction with the frame's T1 and T2
                    _, timebase, delay, t1_ref, t2_ref, dc1, dc2, c0, c1, c2, _, c4 = calibrations[cal_id]
                    a, q = np.sqrt(1e12 / c1), np.frombuffer(raw, '<f8', n) * timebase + delay - c0 + c2 * c4
                    x = 2 * q / (a + np.sqrt(a * a + 4 * c2 * q))
                    lines = np.column_stack([(x * x - c4) / (1 + dc1 * 1e-6 * (t1 - t1_ref) + dc2 * 1e-6 * (t2 - t2_ref)), np.frombuffer(raw, '<f4', n, 8 * n)])
                    if not kind and n > TIMS_MS1_PEAKS:
                        # A survey frame holds 20-40k lines, most of them single ions near the line threshold (2 GB of MS1 data for a 100-min run); as of
                        # a timsTOF frame, its TIMS_MS1_PEAKS most intense lines are kept and, of those at the cut-off intensity, the first in m/z order
                        kth = np.partition(lines[:, 1], -TIMS_MS1_PEAKS)[-TIMS_MS1_PEAKS]
                        above, tied = lines[:, 1] > kth, lines[:, 1] == kth
                        lines = lines[above | (tied & (np.cumsum(tied) <= TIMS_MS1_PEAKS - above.sum()))]
                spectrum['peaks'] = lines
            yield spectrum
    finally:
        if peaks:
            f.close()


# MassLynx scan index records (_FUNCnnn.IDX) by their size: the number of points in the low 22 bits of info (bit 30: m/z already calibrated), TIC, RT in
# minutes and the scan's byte offset in the .DAT, a uint32 at 0 in the 22-byte records of older runs, a uint64 at 22 in the 30-byte ones
WATERS_INDEX = [np.dtype({'names': ['info', 'tic', 'rt', 'offset'], 'formats': ['<u4', '<f4', '<f4', '<u8' if size == 30 else '<u4'],
                          'offsets': [4, 8, 12, 22 if size == 30 else 0], 'itemsize': size}) for size in (30, 22)]


def read_waters(folder, peaks = True):
    """iterates over the scans of a Waters MassLynx .raw folder of a Q-TOF (SYNAPT, Xevo G2, Q-Tof Premier) run with a survey and MS/MS functions (DDA,
    e.g., Fast DDA), in acquisition order; the lock mass function and functions of other scan types are skipped\n
   | Arguments:
   | :-
   | folder (string): absolute path to the .raw folder
   | peaks (bool): whether to read the spectra or only their headers (peaks then None); default:True\n
   | Returns:
   | :-
   | Yields a dict per scan: scan (msconvert's native ID, 'function=f process=0 scan=s'), ms_level, rt (min), peaks ((n, 2) array of centroid m/z and
   | intensity; continuum scans are centroided), precursor (mz, the DDA set mass, and charge None, as MassLynx stores none; None for survey scans),
   | polarity ('+'/'-'), collision_energy and scan_low (the lower end of the function's mass range)
   """
    name = os.path.basename(os.path.normpath(folder))
    # File names differ in case between MassLynx versions (_FUNC001.DAT, _func001.cdt, _inlet.inf)
    files = {x.lower(): os.path.join(folder, x) for x in os.listdir(folder)}
    if '_functns.inf' not in files:
        raise ValueError(f"{name} is no Waters MassLynx .raw folder (no _FUNCTNS.INF).")
    with open(files['_functns.inf'], 'rb') as f:
        functions = f.read()
    # Uncalibrated scans (continuum data) take the header's calibration of their function, a polynomial in sqrt(m/z) ('T1') whose square is the m/z
    with open(files['_header.txt'], 'rb') as f:
        calibrations = {int(k): v.split(',') for k, v in re.findall(r'Cal Function (\d+): ([^\r\n]+)', f.read().decode('latin-1'))}
    scans = []
    # _FUNCTNS.INF: a 416-byte record per function, first a uint16 (bits 0-4 the scan type, 16 TOF MS/MS and 18 TOF MS; bits 5-9 the ionization mode,
    # odd for the negative ones such as ES-; bit 10 centroided data; bit 15 the lock mass reference function), the float32 mass range at 160 and 288
    for k in range(len(functions) // 416):
        head, low = struct.unpack_from('<H', functions, 416 * k)[0], struct.unpack_from('<f', functions, 416 * k + 160)[0]
        stem = f'_func{k + 1:03d}'
        if head & 0x8000 or head & 31 not in (16, 18) or stem + '.idx' not in files or not os.path.getsize(files[stem + '.idx']):
            continue
        if stem + '.cdt' in files:
            raise ValueError(f"{name} holds ion mobility (HDMS) data, which CandyCrunch cannot read; convert it to mzML with msconvert first.")
        with open(files[stem + '.idx'], 'rb') as f:
            index = f.read()
        size = os.path.getsize(files[stem + '.dat'])
        # The index layout whose offsets step through the .DAT in points of 8 (continuum) or 12 bytes (centroids)
        for record in WATERS_INDEX:
            rows = np.frombuffer(index, dtype = record, count = len(index) // record.itemsize)
            n = rows['info'].astype(np.int64) & 0x3fffff
            width = size // max(int(n.sum()), 1)
            if (len(index) % record.itemsize == 0 and width in (8, 12) and size == width * n.sum() and
                    np.array_equal(rows['offset'], np.concatenate([[0], np.cumsum(n * width)[:-1]]))):
                break
        else:
            raise ValueError(f"{name} has a MassLynx scan index CandyCrunch does not know; convert it to mzML with msconvert first.")
        # _FUNCnnn.STS: per scan a record of the parameters its header defines (48 bytes each from byte 32: uint16 number, type 0-3 for uint8, int16,
        # int32, float32, offset in the record, then the name); of them the DDA set mass and the collision energy
        with open(files[stem + '.sts'], 'rb') as f:
            stats = f.read()
        start, _, length, count = struct.unpack_from('<4H', stats, 0)
        fields = {stats[j + 6:j + 32].split(b'\x00')[0]: struct.unpack_from('<2H', stats, j + 2) for j in range(32, 32 + 48 * count, 48)}

        def stat(key):
            if key not in fields or not length or len(stats) < start + length * len(rows):
                return [None] * len(rows)
            kind, at = fields[key]
            return np.frombuffer(stats, np.dtype({'names': ['v'], 'formats': [['<u1', '<i2', '<i4', '<f4'][kind]], 'offsets': [at], 'itemsize': length}),
                                 len(rows), start)['v'].tolist()

        scans += [(float(row['rt']), k, j, head, low, width, row, mz, energy) for j, (row, mz, energy) in enumerate(zip(rows, stat(b'Set Mass'), stat(b'Collision Energy')))]
    handles = {}
    try:
        for rt, k, j, head, low, width, row, mz, energy in sorted(scans, key = lambda s: s[:3]):
            spectrum = dict(scan = f'function={k + 1} process=0 scan={j + 1}', ms_level = 2 if head & 31 == 16 else 1, rt = rt, peaks = None,
                            precursor = dict(mz = mz, charge = None) if head & 31 == 16 and mz else None, polarity = '-' if head >> 5 & 1 else '+',
                            collision_energy = energy, scan_low = low)
            if peaks:
                n = int(row['info']) & 0x3fffff
                f = handles[k] if k in handles else handles.setdefault(k, open(files[f'_func{k + 1:03d}.dat'], 'rb'))
                f.seek(int(row['offset']))
                words = np.fromfile(f, '<u4', n * width // 4).reshape(n, width // 4).astype(np.int64)
                # A point is two (continuum) or three (centroid) uint32: the intensity as a 22-bit mantissa (bits 0-21) and a 5-bit exponent (22-26), the
                # m/z as a 27-bit mantissa and a 5-bit exponent (27-31); the third word is not needed
                ints = np.ldexp((words[:, 0] & 0x3fffff).astype(np.float64), (words[:, 0] >> 22 & 31) - 21)
                mzs = np.ldexp((words[:, 1] & 0x7ffffff).astype(np.float64), (words[:, 1] >> 27) - 27)
                if not int(row['info']) >> 30 & 1 and n:
                    if calibrations.get(k + 1, [''])[-1].strip() != 'T1':
                        raise ValueError(f"{name} uses a mass calibration CandyCrunch does not know; convert it to mzML with msconvert first.")
                    mzs = np.polyval(np.array(calibrations[k + 1][:-1], dtype = np.float64)[::-1], np.sqrt(mzs)) ** 2
                # Continuum scans are centroided as MassLynx does by default (its centroid top of 80%): the m/z from the points above 20% of a peak's apex,
                # the intensity summed over the peak
                spectrum['peaks'] = np.column_stack([mzs, ints]) if head & 0x400 else centroid_tof(mzs, ints, valley = 0.9, min_height = 0.2)
            yield spectrum
    finally:
        for f in handles.values():
            f.close()


def mz_order(mzs, rts, charges, compositions):
    """Rank of each result row by m/z, except that the rows of one composition and charge within 1 Da of each other (one ion, whose m/z differ only by noise;
    its adducts and reducing-end variants are several Da apart even at z = 4) rank together by RT, at the m/z of the lightest of them"""
    mzs, rts = np.asarray(mzs, dtype = float), np.asarray(rts, dtype = float)
    heads, last = np.arange(len(mzs)), {}
    for i in np.argsort(mzs, kind = 'stable'):
        composition = compositions[i]
        if not isinstance(composition, (dict, str)) or not composition:
            continue
        key = (str(sorted(composition.items())) if isinstance(composition, dict) else composition, charges[i])
        if key in last and mzs[i] - mzs[heads[last[key]]] < 1:
            heads[i] = heads[last[key]]
        last[key] = i
    ranks = np.empty(len(mzs), dtype = int)
    ranks[np.lexsort((mzs, rts, heads, mzs[heads]))] = np.arange(len(mzs))
    return ranks
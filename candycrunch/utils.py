"""Readers for mzML, mzXML, mgf, and Thermo .raw spectra files, on the standard library and numpy"""
import base64
import heapq
import itertools
import re
import struct
import zlib
import xml.etree.ElementTree as ET
import numpy as np
# Thermo .raw trailer extra fields: numpy types of the generic data types (0 is a gap, 2-4 are booleans; 12 and 13, strings of the declared length
# in ASCII or UTF-16, are not read)
RAW_FIELD_TYPES = {0: 'V0', 1: 'i1', 2: 'u1', 3: 'u1', 4: 'u1', 5: 'u1', 6: '<i2', 7: '<u2', 8: '<i4', 9: '<u4', 10: '<f4', 11: '<f8'}
# The trailer extra fields read per scan (Thermo writes 0 for a monoisotopic m/z or charge state it did not determine)
RAW_TRAILER_FIELDS = ('Monoisotopic M/Z:', 'Charge State:', 'MS2 Isolation Width:', 'MSn Isolation Width:', 'Master Scan Number:')
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


def centroid_tof(mzs, ints, max_peaks = None):
    """centroids the summed TOF points of a timsTOF frame (or of one precursor's PASEF scans), which sit on the TOF digitizer grid, evenly spaced in sqrt(m/z)\n
   | Arguments:
   | :-
   | mzs (array): m/z values of the points, repeated where several ion mobility scans hit the same TOF bin
   | ints (array): their intensities
   | max_peaks (int): keep only this many of the most intense centroids; default:None (all)\n
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
    is_bound[minima[smooth[minima] < 0.7 * np.minimum(tops[pos - 1], tops[pos])]] = True
    bounds = np.flatnonzero(is_bound)
    peak_ints = np.add.reduceat(ints, bounds)
    peak_mzs = np.add.reduceat(ints * mzs, bounds) / peak_ints
    if max_peaks is not None and len(peak_ints) > max_peaks:
        top = np.sort(np.argpartition(peak_ints, -max_peaks)[-max_peaks:])
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
                    params['charge'] = [int(c[-1] + c[:-1]) if c[-1:] in ('+', '-') else int(c) for c in re.split(r',\s*|\s*and\s*', params['charge'])]
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
   | ((lowest, highest) m/z of its scan windows; an ion-trap MS2's lower limit is its low-mass cutoff), and packet (the file offset of the scan's data
   | packet, for read_raw_packet)
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
                f.seek(params_addr)
                records = np.frombuffer(f.read(n_scans * sum(sizes)), {'names': list(columns), 'formats': [t for t, _ in columns.values()],
                                                                        'offsets': [o for _, o in columns.values()], 'itemsize': sum(sizes)})
                trailer = {label: records[label].tolist() for label in columns}
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
                                 'isolation_width', 'master_scan', 'retention_time', 'scan_range', 'packet']}
    for i, s in enumerate(starts):
        level = max(stream[s + 6], 1)
        # MS1 events of some orbitraps carry their scan range as a reaction
        reactions = [struct.unpack_from('<3dI', stream, s + preamble + 4 + reaction * k)
                     for k in range(int.from_bytes(stream[s + preamble:s + preamble + 4], 'little'))] if level > 1 else []
        mono, charge, ms2_width, msn_width, master = (trailer[label][i] if label in trailer and level > 1 else None for label in RAW_TRAILER_FIELDS)
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
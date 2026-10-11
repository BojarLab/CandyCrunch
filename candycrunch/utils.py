"""Readers for mzML, mzXML and mgf spectra files, on the standard library and numpy"""
import base64
import heapq
import itertools
import math
import re
import struct
import zlib
import xml.etree.ElementTree as ET
import numpy as np


def centroid_ion_trap(mzs, ints):
    """centroids a Thermo ion-trap profile spectrum as Thermo's centroider does (it made the centroids of the mzML files CandyCrunch was validated on)\n
   | Arguments:
   | :-
   | mzs (array): profile m/z values, on Thermo's profile axis (as in mzML files from msconvert)
   | ints (array): profile intensities\n
   | Returns:
   | :-
   | Returns an (n, 2) array of centroid m/z and intensity: profile points below 1 are dropped, the profile is split at the lowest point between
   | maxima and neighboring centroids closer than 0.5 m/z are merged, each carrying the summed signal of its share of the profile
   """
    ints = np.where(ints < 1, 0, ints)
    apex = np.where((ints[1:-1] > ints[:-2]) & (ints[1:-1] >= ints[2:]) & (ints[1:-1] > 0))[0] + 1
    if not len(apex):
        return np.zeros((0, 2))
    edges = np.array([0] + [a + 1 + np.argmin(ints[a + 1:b]) for a, b in zip(apex[:-1], apex[1:])], dtype = np.int64)
    while True:
        peak_ints = np.add.reduceat(ints, edges)
        peak_mzs = np.add.reduceat(ints * mzs, edges) / peak_ints
        gaps = np.diff(peak_mzs)
        if not len(gaps) or gaps.min() >= 0.5:
            break
        # Merges every pair closer than 0.5 m/z whose gap is the smallest among its neighboring gaps, until none is left
        left, right = np.concatenate([[np.inf], gaps[:-1]]), np.concatenate([gaps[1:], [np.inf]])
        edges = np.delete(edges, np.where((gaps < 0.5) & (gaps <= left) & (gaps < right))[0] + 1)
    # Thermo's centroids of peaks spanning 9 or more profile points sit one bin below their intensity-weighted mean (the bin is the median spacing,
    # as converters drop runs of zero points)
    peak_mzs = peak_mzs - np.median(np.diff(mzs)) * (np.add.reduceat((ints > 0).astype(int), edges) >= 9)
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
    mz, inty, centroids = np.asarray(mzs).tolist(), np.asarray(ints).tolist(), []
    for pos in range(2, len(inty) - 1):
        x1, x2, x3, y1, y2, y3 = mz[pos - 1], mz[pos], mz[pos + 1], inty[pos - 1], inty[pos], inty[pos + 1]
        if not 0 < y1 < y2 > y3 > 0 or x2 - x1 > (x3 - x2) * 10 or (x2 - x1) * 10 < x3 - x2:
            continue
        if y3 == y1:
            y3 += 0.01 * y1
        try:
            double_log = math.log(y2 / y1) / math.log(y3 / y1)
            mu = (double_log * (x1 * x1 - x3 * x3) - x1 * x1 + x2 * x2) / (2 * (x2 - x1) - 2 * double_log * (x3 - x1))
            c_squared = (x2 * x2 - x1 * x1 - 2 * x2 * mu + 2 * x1 * mu) / (2 * math.log(y1 / y2))
            centroids.append((mu, y1 * math.exp((x1 - mu) * (x1 - mu) / (2 * c_squared))))
        except (ZeroDivisionError, OverflowError):
            continue
    return np.array(centroids).reshape(-1, 2)


def read_mzml(filepath, centroid_levels = ()):
    """iterates over the spectra of an .mzML file\n
   | Arguments:
   | :-
   | filepath (string): absolute filepath to the .mzML file
   | centroid_levels (tuple): MS levels whose profile spectra are centroided (Thermo ion-trap scans as by Thermo's centroider, others by a 3-point
   |                          Gaussian fit at each local maximum); default:()\n
   | Returns:
   | :-
   | Yields a dict per spectrum: element (its XML element, with the params of a referenced param group appended), ms_level (int or None),
   | rt (scan start time in minutes; None if absent), peaks ((n, 2) array of m/z and intensity), and precursor (mz, and if given charge and
   | intensity i, of the first selected ion in the spectrum; None if there is none), and instrument (CV accessions of the spectrum's instrument
   | configuration, i.e., its analyzers and instrument model; those of the first configuration if the scan names none)
   """
    groups, configs = {}, {}
    for _, el in ET.iterparse(filepath):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'referenceableParamGroup':
            groups[el.get('id')] = el
        elif tag == 'instrumentConfiguration':
            # Param groups precede the instrument configurations, which hold the instrument model in a referenced group (Thermo) or directly
            configs[el.get('id')] = {e.get('accession') for p in [el] + [groups[r.get('ref')] for r in el.iter() if r.tag.endswith('referenceableParamGroupRef') and r.get('ref') in groups]
                                     for e in p.iter() if e.get('accession')}
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
        arrays = {}
        for bda in el.iter(f'{ns}binaryDataArray'):
            terms = {c.get('accession') for c in bda.findall(f'{ns}cvParam')}
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
        if ms_level in centroid_levels and 'MS:1000128' in first and (first['MS:1000512'].get('value', '') if 'MS:1000512' in first else '').startswith('ITMS'):
            # msconvert writes ion-trap scans as profiles unless asked to peak-pick; Thermo's centroids of them gave GPST000017 F1 0.735 (as from the
            # .raw file) against 0.695 with the Gaussian fit below
            peaks = centroid_ion_trap(peaks[:, 0], peaks[:, 1])
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
        scan = el.find(f'{ns}scanList/{ns}scan')
        instrument = configs.get(scan.get('instrumentConfigurationRef') if scan is not None else None, next(iter(configs.values()), set()))
        yield {'element': el, 'ms_level': ms_level, 'rt': rt, 'peaks': peaks, 'precursor': precursor, 'instrument': instrument}
        el.clear()


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
        if el.tag.rsplit('}', 1)[-1] == 'msInstrument':
            instruments[el.get('msInstrumentID', el.get('id'))] = ' '.join(c.get('value', '') for c in el if c.tag.rsplit('}', 1)[-1] in ('msModel', 'msMassAnalyzer')).lower()
        if el.tag.rsplit('}', 1)[-1] != 'scan':
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
            cents = (centroid_ion_trap if str(scan.get('filterLine', '')).startswith('ITMS') else centroid_gaussian)(values[0::2], values[1::2])
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
    with open(filepath, encoding = 'utf-8', errors = 'replace') as f:
        for line in f:
            sline = line.strip()
            if sline == 'BEGIN IONS':
                params, mzs, ints, started = dict(header), [], [], True
            elif params is None:
                if not started and len(line.split('=')) == 2:
                    header[line.split('=')[0].lower()] = line.split('=')[1].strip()
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
            elif len(sline.split()) > 1:
                mzs.append(float(sline.split()[0]))
                ints.append(float(sline.split()[1]))

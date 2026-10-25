import collections
import os
import pickle
import random
import struct
import sys
import zipfile
import numpy as np


def remove_low_intensity_peaks(array, removal_threshold, removal_percentage):
    candidate_indices = np.where(np.logical_and(array > 0.0001, array <= removal_threshold))[0]
    indices_to_remove = np.random.choice(candidate_indices, round(removal_percentage * len(candidate_indices)))
    array_copy = np.copy(array)
    array_copy[indices_to_remove] = 0
    return array_copy


def peak_intensity_jitter(array, augment_intensity):
    return array * np.random.uniform(1 - augment_intensity, 1 + augment_intensity, len(array)).astype(np.float32)


def new_peak_addition(array, n_noise_peaks, max_noise_intensity):
    idx_noise_peaks = np.random.choice(np.where(array == 0)[0], n_noise_peaks)
    new_values = max_noise_intensity * np.random.random(len(idx_noise_peaks))
    noisy_array = np.copy(array)
    noisy_array[idx_noise_peaks] = new_values
    return noisy_array


def transform_mz(x):
    return new_peak_addition(peak_intensity_jitter(remove_low_intensity_peaks(x, removal_threshold = 0.008, removal_percentage = 0.1),
                                                   augment_intensity = 0.25), n_noise_peaks = 10, max_noise_intensity = 0.005)


def transform_rt(RT):
    return max(0, RT + random.uniform(-0.1, 0.1))


# Element types of the storages in a file saved by torch.save (bfloat16 is read as its bits and widened to float32)
STORAGE_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16, 'BFloat16Storage': np.uint16,
                  'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16, 'CharStorage': np.int8, 'ByteStorage': np.uint8,
                  'BoolStorage': np.bool_}


def _rebuild_tensor(storage, offset, size, stride, *args):
    return np.lib.stride_tricks.as_strided(storage[offset:], shape = size, strides = [s * storage.itemsize for s in stride]).copy()


class _WeightsUnpickler(pickle.Unpickler):
    """Reads the pickle of a torch.save file with tensors as numpy arrays and nothing but containers and tensors allowed (like torch.load with
    weights_only), so a model file from elsewhere cannot run code"""

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _rebuild_tensor
        if (module, name) == ('torch._utils', '_rebuild_parameter'):
            return lambda data, *args: data
        if module == 'torch' and name in STORAGE_DTYPES:
            return name
        raise pickle.UnpicklingError(f'{module}.{name} is not allowed in a weights file')

    def persistent_load(self, pid):
        # ('storage', storage type, key of its file in the archive, location, number of elements)
        data = np.frombuffer(self.archive.read(f'{self.prefix}/data/{pid[2]}'), dtype = np.dtype(STORAGE_DTYPES[pid[1]]).newbyteorder('<'))
        return (data.astype(np.uint32) << 16).view(np.float32) if pid[1] == 'BFloat16Storage' else data


def load_weights(path):
    """What a file saved by torch.save holds (a state dict, or e.g. {'state_dict': ..., 'glycans': [...]}), with its tensors as numpy arrays, without PyTorch"""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('data.pkl'))
        with archive.open(name) as file:
            return _WeightsUnpickler(file, archive, name.rpartition('/')[0]).load()


def _varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while n > 127:
        out.append(n & 127 | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def _proto(*fields):
    """Protocol buffer message of (field number, value) pairs: an int as varint, a float as fixed32, bytes, a string or a message length-delimited"""
    out = []
    for number, value in fields:
        if isinstance(value, float):
            out.append(_varint(number << 3 | 5) + struct.pack('<f', value))
        elif isinstance(value, int):
            out.append(_varint(number << 3) + _varint(value))
        else:
            value = value.encode() if isinstance(value, str) else value
            out.append(_varint(number << 3 | 2) + _varint(len(value)) + value)
    return b''.join(out)


# Threads of ONNX Runtime in this process (0: one per physical core), see set_num_threads
_threads = 0


def set_num_threads(n):
    """Sets the threads every CandyCrunchCNN uses from its next call on in this process, like torch.set_num_threads (0: one per physical core)"""
    global _threads
    _threads = n


class CandyCrunchCNN:
    """The CandyCrunch CNN of model.py run by ONNX Runtime, for where PyTorch is not installed (the built app); made from its weights (a state dict of arrays or tensors) and
    called with a batch of inputs (binned intensities and m/z remainders as two channels, composition vector, glycan type, RT, mode, LC, modification and
    trap), it returns the logits"""

    def __init__(self, state_dict):
        used = set()
        # The weights, noting which ones the graph reads
        class Weights(dict):
            def __getitem__(self, k):
                used.add(k)
                return super().__getitem__(k)
        p = Weights({k.replace('module.', ''): np.asarray(v.detach().cpu() if hasattr(v, 'detach') else v, dtype = np.float32) for k, v in state_dict.items()})
        nodes, weights = [], {}
        # ONNX messages (onnx.proto), written here so that inference needs neither PyTorch nor the onnx package
        def const(name, value):
            weights[name] = _proto(*[(1, d) for d in value.shape], (2, 1), (8, name), (9, np.ascontiguousarray(value, dtype = np.float32).tobytes()))
            return name
        def node(op, inputs, domain = '', **attrs):
            # An attribute is its name, its value in the field of its type and that type: float 2/1, int 3/2, string 4/3, floats 7/6, ints 8/7
            out = f'{op}_{len(nodes)}'
            kind = lambda v: {float: (2, 1), int: (3, 2), str: (4, 3)}[type(v)]
            attrs = [_proto((1, k), *[(kind(x)[0] + 5, x) for x in v], (20, kind(v[0])[1] + 5)) if isinstance(v, list) else
                     _proto((1, k), (kind(v)[0], v), (20, kind(v)[1])) for k, v in attrs.items()]
            nodes.append(_proto(*[(1, x) for x in inputs], (2, out), (4, op), (7, domain), *[(5, a) for a in attrs]))
            return out
        act = lambda x: node('LeakyRelu', [x], alpha = 0.01)
        lin = lambda x, k: node('Gemm', [x, const(f'{k}.weight', p[f'{k}.weight']), const(f'{k}.bias', p[f'{k}.bias'])], transB = 1)
        norm = lambda x, k: act(node('LayerNormalization', [x, const(f'{k}.weight', p[f'{k}.weight']), const(f'{k}.bias', p[f'{k}.bias'])], axis = 1, epsilon = 1e-5))
        ones, zeros = const('ones', np.ones(2048)), const('zeros', np.zeros(2048))
        def norm_conv(x, k, conv, residual = None, center = False, **attrs):
            # Instance norm, leaky ReLU and convolution. The norm is a layer norm over the m/z bins (ONNX Runtime runs InstanceNormalization on one
            # thread) with the norm's scale g folded into the convolution's weights: leaky_relu(g * x + b) = g * leaky_relu(x + b / g) for g > 0
            g, b, w = p[f'{k}.weight'], p[f'{k}.bias'], p[f'{conv}.weight']
            if center:
                x = node('Sub', [x, node('ReduceMean', [x], axes = [2])])
            x = node('LayerNormalization', [x, ones, zeros], axis = 2, epsilon = 1e-5)
            if (g > 0).all():
                x, w = node('Add', [x, const(f'{k}.shift', (b / g)[:, None])]), w * g[:, None]
            else:
                x = node('Add', [node('Mul', [x, const(f'{k}.scale', g[:, None])]), const(f'{k}.shift', b[:, None])])
            inputs = [act(x), const(f'{conv}.weight', w), const(f'{conv}.bias', p[f'{conv}.bias'])]
            # The residual sum is the fused convolution's fourth input (its activation, a leaky ReLU of slope 1, leaves it as it is)
            return node('Conv', inputs, **attrs) if residual is None else node('FusedConv', inputs + [residual], 'com.microsoft', activation = 'LeakyRelu',
                                                                              activation_params = [1.0])
        x = act(node('Conv', ['mz', const('conv1.weight', p['conv1.weight']), const('conv1.bias', p['conv1.bias'])]))
        for i in range(1, 7):
            # The first block's channels are nearly constant over a sparse spectrum (mean squared up to 7e4 times the variance), which a layer norm
            # taking the variance as E[x^2] - E[x]^2 in float32 (ONNX Runtime 1.19) loses, so they are centered first
            h = norm_conv(x, f'res{i}.ln1', f'res{i}.conv_in', center = i == 1)
            h = norm_conv(h, f'res{i}.ln2', f'res{i}.conv_dilated', dilations = [2 ** (i - 1)], pads = [2 ** (i - 1), 0])
            x = norm_conv(h, f'res{i}.ln3', f'res{i}.conv_out', residual = x)
        x = act(lin(node('Flatten', [node('MaxPool', [x], kernel_shape = [20], strides = [20])]), 'fc1'))
        emb = lambda k, x: node('Gather', [const(f'{k}_emb.weight', p[f'{k}_emb.weight']), x])
        x = node('Concat', [x, norm(lin('precursor', 'prec_lin1'), 'prec_bn1'), emb('type', 'glycan_type'), norm(lin('rt', 'rt_lin1'), 'rt_bn1'),
                            *[emb(k, k) for k in ['mode', 'lc', 'modification', 'trap']]], axis = 1)
        x = lin(norm(lin(norm(lin(x, 'comb_lin1'), 'comb_bn1'), 'comb_lin2'), 'comb_bn2'), 'comb_lin3')
        def info(name, kind, dims):
            # An input or output: its name, element type (1 float, 7 int64) and shape ('n' the batch size)
            shape = _proto(*[(1, _proto((2, d) if isinstance(d, str) else (1, d))) for d in dims])
            return _proto((1, name), (2, _proto((1, _proto((1, kind), (2, shape))))))
        inputs = [info('mz', 1, ['n', 2, 2048]), info('precursor', 1, ['n', 12]), info('glycan_type', 7, ['n']), info('rt', 1, ['n', 1])] + [
            info(k, 7, ['n']) for k in ['mode', 'lc', 'modification', 'trap']]
        graph = _proto(*[(1, n) for n in nodes], (2, 'candycrunch'), *[(5, w) for w in weights.values()], *[(11, i) for i in inputs],
                       (12, info(x, 1, ['n', len(p['comb_lin3.bias'])])))
        # A weight the graph does not read (but model.py's unused mz_lin1 and mz_bn1) is a layer this graph lacks: CandyCrunch_CNN changed and this graph did not
        unknown = sorted(set(p) - used - {'mz_lin1.weight', 'mz_lin1.bias', 'mz_bn1.weight', 'mz_bn1.bias'})
        if unknown:
            raise ValueError(f'the ONNX graph has no place for the weights {", ".join(unknown[:5])}; update CandyCrunchCNN to the changed CandyCrunch_CNN')
        self.onnx = _proto((1, 8), (7, graph), (8, _proto((1, ''), (2, 17))), (8, _proto((1, 'com.microsoft'), (2, 1))))
        self.session = None

    def __getstate__(self):
        # Sent to other processes without its session, which each process makes for itself
        return {**self.__dict__, 'session': None}

    def __call__(self, mz, precursor, glycan_type, rt, mode, lc, modification, trap):
        inputs = {'mz': mz, 'precursor': precursor, 'glycan_type': glycan_type, 'rt': rt, 'mode': mode, 'lc': lc, 'modification': modification, 'trap': trap}
        if sys.platform == 'emscripten':
            # In the browser (Pyodide), the page's onnxruntime-web (globalThis.ort, its threads set by the page) runs the graph. Its session.run is
            # asynchronous, and run_sync waits for it, which needs JavaScript Promise Integration (Chrome and Edge 137+). Each output is copied out in
            # JavaScript at once (a tensor read later from Python failed now and then with several threads)
            import js
            from pyodide.ffi import run_sync, to_js
            if self.session is None:
                self.session = (None, run_sync(js.ort.InferenceSession.create(to_js(self.onnx), to_js({'executionProviders': ['wasm']}, dict_converter = js.Object.fromEntries))))
            outputs = []
            for i in range(0, len(mz), 32):
                feeds = js.Object.new()
                for k, v in inputs.items():
                    v = np.ascontiguousarray(v[i:i + 32])
                    data = to_js(v.tobytes()).buffer
                    setattr(feeds, k, js.ort.Tensor.new(*(('int64', js.BigInt64Array.new(data)) if v.dtype == np.int64 else ('float32', js.Float32Array.new(data))), to_js(list(v.shape))))
                out = getattr(run_sync(self.session[1].run(feeds)), self.session[1].outputNames[0])
                outputs.append(np.frombuffer(js.Float32Array.new(out.data).to_bytes(), np.float32).reshape([int(d) for d in out.dims]))
            return np.concatenate(outputs)
        # The session is made at the first call in a process (a forked one cannot use its parent's threads) and again after set_num_threads;
        # onnxruntime is only imported here, as only the built app needs it
        if self.session is None or self.session[0] != (os.getpid(), _threads):
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = _threads
            self.session = ((os.getpid(), _threads), ort.InferenceSession(self.onnx, options, providers = ['CPUExecutionProvider']))
        # 32 inputs per run: ONNX Runtime keeps the memory of its largest run (1.1 GB for 256 inputs, 0.14 GB for 32), and these runs are the fastest
        return np.concatenate([self.session[1].run(None, {k: v[i:i + 32] for k, v in inputs.items()})[0] for i in range(0, len(mz), 32)])
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
        name = next((n for n in archive.namelist() if n.endswith('data.pkl')), None)
        if name is None:
            raise ValueError(f"{path} is no file saved by torch.save (no data.pkl in it)")
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
    """The CandyCrunch CNN of model.py (CandyCrunch_CNN, or CandyCrunch_CNN2 with layer norms, leaky ReLUs and a classifier of linear experts) run by ONNX
    Runtime, for where PyTorch is not installed (the built app); made from its weights (a state dict of arrays or tensors; top_k: the experts a CNN2
    classifier routes each spectrum to, which its weights do not tell) and called with a batch of inputs (binned intensities and m/z remainders as two
    channels, composition vector, glycan type, RT, mode, LC, modification and trap), it returns the logits. It is two graphs, as model.py's encode and
    the rest: the encoder makes each spectrum's features from the spectrum alone, the head the logits from those and the other inputs"""

    def __init__(self, state_dict, top_k = 4):
        used = set()
        # The weights, noting which ones the graph reads; each is made float32 where the graph reads it, so a float16 file is never held twice over
        class Weights(dict):
            def __getitem__(self, k):
                used.add(k)
                return np.asarray(super().__getitem__(k), dtype = np.float32)
        p = Weights({k.replace('module.', ''): v.detach().cpu() if hasattr(v, 'detach') else v for k, v in state_dict.items()})
        nodes, weights = [], {}
        # ONNX messages (onnx.proto), written here so that inference needs neither PyTorch nor the onnx package
        def const(name, value, kind = 1):
            # kind: the element type, 1 float, 7 int64
            # its fields before the raw data, and the data, which model_proto joins into the model without copying it first
            data = np.ascontiguousarray(value, dtype = {1: np.float32, 7: np.int64}[kind])
            weights[name] = (_proto(*[(1, d) for d in value.shape], (2, kind), (8, name)) + _varint(9 << 3 | 2) + _varint(data.nbytes), data)
            return name
        def node(op, inputs, domain = '', outputs = 1, **attrs):
            # An attribute is its name, its value in the field of its type and that type: float 2/1, int 3/2, string 4/3, floats 7/6, ints 8/7
            out = [f'{op}_{len(nodes)}'] + [f'{op}_{len(nodes)}_{i}' for i in range(1, outputs)]
            kind = lambda v: {float: (2, 1), int: (3, 2), str: (4, 3)}[type(v)]
            attrs = [_proto((1, k), *[(kind(x)[0] + 5, x) for x in v], (20, kind(v[0])[1] + 5)) if isinstance(v, list) else
                     _proto((1, k), (kind(v)[0], v), (20, kind(v)[1])) for k, v in attrs.items()]
            nodes.append(_proto(*[(1, x) for x in inputs], *[(2, o) for o in out], (4, op), (7, domain), *[(5, a) for a in attrs]))
            return out[0] if outputs == 1 else out
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
        emb = lambda k, x: node('Gather', [const(f'{k}_emb.weight', p[f'{k}_emb.weight']), x])
        cnn2 = 'prec_block.0.weight' in p
        # CandyCrunch_CNN2's residual blocks keep the norm, activation and convolution order, with dilated convolutions of size 3 padded on both sides
        conv1, blocks = ('res_block.0', [f'res_block.{i + 2}' for i in range(6)]) if cnn2 else ('conv1', [f'res{i}' for i in range(1, 7)])
        x = act(node('Conv', ['mz', const(f'{conv1}.weight', p[f'{conv1}.weight']), const(f'{conv1}.bias', p[f'{conv1}.bias'])]))
        for i, k in enumerate(blocks):
            # The first block's channels are nearly constant over a sparse spectrum (mean squared up to 7e4 times the variance), which a layer norm
            # taking the variance as E[x^2] - E[x]^2 in float32 (ONNX Runtime 1.19) loses, so they are centered first
            h = norm_conv(x, f'{k}.ln1', f'{k}.conv_in', center = i == 0)
            h = norm_conv(h, f'{k}.ln2', f'{k}.conv_dilated', dilations = [2 ** i], pads = [2 ** i, 2 ** i if cnn2 else 0])
            x = norm_conv(h, f'{k}.ln3', f'{k}.conv_out', residual = x)
        if cnn2:
            # Adaptive max pooling to 102 windows [floor(i L / 102), ceil((i + 1) L / 102)) of the L = 2048 bins, 21 or 22 bins wide: each window's
            # maximum is picked from the running maxima of both widths
            starts, ends = np.arange(102) * 2048 // 102, -(-(np.arange(102) + 1) * 2048 // 102)
            picks = np.where(ends - starts == 21, starts, 2048 - 21 + 1 + starts)
            x = node('Concat', [node('MaxPool', [x], kernel_shape = [21]), node('MaxPool', [x], kernel_shape = [22])], axis = 2)
            x = norm(lin(node('Flatten', [node('Gather', [x, const('pool_picks', picks, 7)], axis = 2)]), 'fc1.0'), 'fc1.1')
        else:
            x = act(lin(node('Flatten', [node('MaxPool', [x], kernel_shape = [20], strides = [20])]), 'fc1'))
        # The spectrum's features end the encoder graph; the head graph takes them as its input 'features'
        encoder, encoder_weights, encoded, n_features = list(nodes), dict(weights), x, len(p['fc1.0.bias' if cnn2 else 'fc1.bias'])
        nodes.clear()
        weights.clear()
        x = 'features'
        if cnn2:
            x = node('Concat', [x, norm(lin('precursor', 'prec_block.0'), 'prec_block.1'), emb('type', 'glycan_type'), norm(lin('rt', 'rt_block.0'), 'rt_block.1'),
                                *[emb(k, k) for k in ['mode', 'lc', 'modification', 'trap']]], axis = 1)
            h, c = norm(lin(x, 'comb_block1.0'), 'comb_block1.1'), 'classifier_out'
            if 'comb_lin2.weight' in p:
                x, n_classes = lin(h, 'comb_lin2'), len(p['comb_lin2.bias'])
            else:
                # The router's softmax renormalized over its top_k experts is the softmax of their router logits; the experts run on every spectrum as
                # one layer, weighted by these gates (zero for the other experts)
                n_experts, n_classes = len(p[f'{c}.router.bias']), len(p[f'{c}.experts.0.bias'])
                router = lin(h, f'{c}.router')
                top, experts = node('TopK', [router, const('top_k', np.array([top_k]), 7)], outputs = 2, axis = 1)
                gates = node('ScatterElements', [node('Mul', [router, const('zero', np.zeros(1))]), experts, node('Softmax', [top], axis = 1)], axis = 1)
                out = node('Gemm', [h, const(f'{c}.experts.weight', np.concatenate([p[f'{c}.experts.{e}.weight'] for e in range(n_experts)])),
                                    const(f'{c}.experts.bias', np.concatenate([p[f'{c}.experts.{e}.bias'] for e in range(n_experts)]))], transB = 1)
                out = node('MatMul', [node('Reshape', [gates, const('gate_shape', np.array([-1, 1, n_experts]), 7)]),
                                      node('Reshape', [out, const('expert_shape', np.array([-1, n_experts, n_classes]), 7)])])
                x = node('Reshape', [out, const('logit_shape', np.array([-1, n_classes]), 7)])
                if f'{c}.shared_expert.weight' in p:
                    x = node('Add', [x, lin(h, f'{c}.shared_expert')])
        else:
            x = node('Concat', [x, norm(lin('precursor', 'prec_lin1'), 'prec_bn1'), emb('type', 'glycan_type'), norm(lin('rt', 'rt_lin1'), 'rt_bn1'),
                                *[emb(k, k) for k in ['mode', 'lc', 'modification', 'trap']]], axis = 1)
            x, n_classes = lin(norm(lin(norm(lin(x, 'comb_lin1'), 'comb_bn1'), 'comb_lin2'), 'comb_bn2'), 'comb_lin3'), len(p['comb_lin3.bias'])
        def info(name, kind, dims):
            # An input or output: its name, element type (1 float, 7 int64) and shape ('n' the batch size)
            shape = _proto(*[(1, _proto((2, d) if isinstance(d, str) else (1, d))) for d in dims])
            return _proto((1, name), (2, _proto((1, _proto((1, kind), (2, shape))))))

        def model_proto(name, graph_nodes, graph_weights, inputs, output):
            # The model's bytes joined once from the graph's pieces and the weights' arrays: nesting whole messages copied the weights three times over
            # (a 0.65 GB peak for 126 MB of model, a 1 GB heap in the browser)
            pieces = [_proto(*[(1, n) for n in graph_nodes], (2, name))]
            for fields, data in graph_weights.values():
                pieces += [_varint(5 << 3 | 2) + _varint(len(fields) + data.nbytes) + fields, data]
            pieces.append(_proto(*[(11, i) for i in inputs], (12, output)))
            size = sum(x.nbytes if isinstance(x, np.ndarray) else len(x) for x in pieces)
            return b''.join([_proto((1, 8)) + _varint(7 << 3 | 2) + _varint(size)] + pieces + [_proto((8, _proto((1, ''), (2, 17))), (8, _proto((1, 'com.microsoft'), (2, 1))))])

        inputs = [info('features', 1, ['n', n_features]), info('precursor', 1, ['n', p['prec_block.0.weight' if cnn2 else 'prec_lin1.weight'].shape[1]]),
                  info('glycan_type', 7, ['n']), info('rt', 1, ['n', 1])] + [info(k, 7, ['n']) for k in ['mode', 'lc', 'modification', 'trap']]
        # A weight the graph does not read (but model.py's unused mz_lin1 and mz_bn1) is a layer this graph lacks: CandyCrunch_CNN changed and this graph did not
        unknown = sorted(set(p) - used - {'mz_lin1.weight', 'mz_lin1.bias', 'mz_bn1.weight', 'mz_bn1.bias'})
        if unknown:
            raise ValueError(f'the ONNX graph has no place for the weights {", ".join(unknown[:5])}; update CandyCrunchCNN to the changed CandyCrunch_CNN')
        self.onnx = {'encode': model_proto('candycrunch_encoder', encoder, encoder_weights, [info('mz', 1, ['n', 2, 2048])], info(encoded, 1, ['n', n_features])),
                     'head': model_proto('candycrunch_head', nodes, weights, inputs, info(x, 1, ['n', n_classes]))}
        self.session = {}

    def __getstate__(self):
        # Sent to other processes without its sessions, which each process makes for itself
        return {**self.__dict__, 'session': {}}

    def encode(self, mz):
        """The spectra's features (what the CNN makes of the binned intensities and m/z remainders alone, the same for every composition, RT and setting
        a spectrum is tested with)"""
        return self._run('encode', {'mz': mz})

    def __call__(self, mz, precursor, glycan_type, rt, mode, lc, modification, trap, features = None):
        # features: the spectra's features from encode, if already made (mz is then not read)
        return self._run('head', {'features': self.encode(mz) if features is None else features, 'precursor': precursor, 'glycan_type': glycan_type,
                                  'rt': rt, 'mode': mode, 'lc': lc, 'modification': modification, 'trap': trap})

    def _run(self, graph, inputs):
        # Inputs per run: ONNX Runtime keeps the memory of its largest run, which for the encoder is 1.1 GB for 256 inputs and 0.14 GB for 32 (whose runs
        # are also the fastest); the head needs little memory, and in runs of 32 it read its expert weights (94 MB for CandyCrunch_CNN2) anew for every
        # 32 inputs, 20x slower than PyTorch's head
        n, step = len(next(iter(inputs.values()))), 32 if graph == 'encode' else 512
        if sys.platform == 'emscripten':
            # In the browser (Pyodide), the page's onnxruntime-web (globalThis.ort, its threads set by the page) runs the graph. Its session.run is
            # asynchronous, and run_sync waits for it, which needs JavaScript Promise Integration (Chrome and Edge 137+). Pyodide copies a buffer between
            # Python and JavaScript wrongly once it lies past 2 GiB of its heap (RangeError: offset is out of bounds; the model then failed to parse), which
            # a large run's heap passes, so the model and every array go through a file of Emscripten's in-memory file system, whose copies hold at any
            # address. Each output is copied out at once (a tensor read later from Python failed now and then with several threads)
            import js
            import pyodide_js
            from pyodide.ffi import run_sync, to_js
            path = '/tmp/candycrunch_onnx_io'

            def staged(data):
                # the bytes as a JavaScript Uint8Array of their own
                with open(path, 'wb') as f:
                    f.write(data)
                return pyodide_js.FS.readFile(path)

            if graph not in self.session:
                self.session[graph] = (None, run_sync(js.ort.InferenceSession.create(staged(self.onnx[graph]), to_js({'executionProviders': ['wasm']},
                                                                                                                     dict_converter = js.Object.fromEntries))))
            session = self.session[graph][1]
            outputs = []
            for i in range(0, n, step):
                feeds = js.Object.new()
                for k, v in inputs.items():
                    v = np.ascontiguousarray(v[i:i + step])
                    data = staged(v).buffer
                    setattr(feeds, k, js.ort.Tensor.new(*(('int64', js.BigInt64Array.new(data)) if v.dtype == np.int64 else ('float32', js.Float32Array.new(data))), to_js(list(v.shape))))
                out = getattr(run_sync(session.run(feeds)), session.outputNames[0])
                pyodide_js.FS.writeFile(path, js.Uint8Array.new(js.Float32Array.new(out.data).buffer))
                outputs.append(np.fromfile(path, np.float32).reshape([int(d) for d in out.dims]))
            os.remove(path)
            return np.concatenate(outputs)
        # The session is made at the first call in a process (a forked one cannot use its parent's threads) and again after set_num_threads;
        # onnxruntime is only imported here, so that importing candycrunch does not load it
        if self.session.get(graph, (None,))[0] != (os.getpid(), _threads):
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = _threads
            # The encoder's and the head's sessions run in turn: a pool spinning after its run would take the cores from the other
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            self.session[graph] = ((os.getpid(), _threads), ort.InferenceSession(self.onnx[graph], options, providers = ['CPUExecutionProvider']))
        return np.concatenate([self.session[graph][1].run(None, {k: v[i:i + step] for k, v in inputs.items()})[0] for i in range(0, n, step)])
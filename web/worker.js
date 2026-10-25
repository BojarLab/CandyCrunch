// CandyCrunch in the browser: Pyodide in this module worker runs candycrunch's CLI (candycrunch.web) on the files the page mounts here, and
// onnxruntime-web runs the model. Their versions are pinned in the next three lines; bump them deliberately, together with a check against the native output
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

let py = null;
const post = (type, data) => postMessage({type, ...data});

async function init(threads) {
  // candycrunch.inference runs the model on globalThis.ort
  ort.env.wasm.numThreads = threads;
  self.ort = ort;
  post('status', {text: 'Loading Python'});
  py = await loadPyodide({stdout: (text) => post('log', {text}), stderr: (text) => post('log', {text})});
  post('status', {text: 'Loading CandyCrunch and its packages'});
  await py.loadPackage('micropip', {messageCallback: () => {}});
  // The wheels the Pages workflow built (glycowork from the ref CI tests against, then candycrunch); micropip takes the rest from Pyodide and PyPI
  const manifest = await (await fetch(new URL('wheels/manifest.json', self.location.href))).json();
  const micropip = py.pyimport('micropip');
  await micropip.install(manifest.wheels.map((w) => new URL('wheels/' + w, self.location.href).href));
  micropip.destroy();
  post('status', {text: 'Loading the model'});
  await py.runPythonAsync('import candycrunch.prediction\nfrom candycrunch import web');
  const versions = py.runPython("import importlib.metadata as m, sys\n{'candycrunch': m.version('candycrunch'), 'glycowork': m.version('glycowork'), 'python': sys.version.split()[0]}");
  return {parameters: py.runPython('web.parameters()'), versions: versions.toJs({dict_converter: Object.fromEntries}), threads};
}

async function run(files, argv) {
  // The files, read in place by WORKERFS (nothing is copied or uploaded), under /data at their relative paths (a .d folder keeps its inside)
  try {
    py.FS.unmount('/data');
  } catch (e) {}
  py.FS.mkdirTree('/data');
  py.FS.mount(py.FS.filesystems.WORKERFS, {blobs: files.map((f) => ({name: f.path, data: f.file}))}, '/data');
  py.runPython("import os, shutil\nshutil.rmtree('/out', ignore_errors = True)\nos.makedirs('/out')");
  py.globals.set('argv', py.toPy(argv));
  // Asynchronous, as the model's calls wait for onnxruntime-web
  await py.runPythonAsync('web.run(list(argv))');
  // Every file the CLI wrote; each table also as .xlsx and as text for the page's preview
  const out = await py.runPythonAsync(`
import io, os
import pandas as pd
out = []
for name in sorted(os.listdir('/out')):
    item = {'name': name, 'data': open(os.path.join('/out', name), 'rb').read()}
    if name.endswith('.csv'):
        df = pd.read_csv(os.path.join('/out', name))
        buffer = io.BytesIO()
        df.to_excel(buffer, index = False)
        item['xlsx'] = buffer.getvalue()
        item['columns'], item['rows'] = [str(c) for c in df.columns], df.astype(object).where(df.notna(), '').astype(str).values.tolist()
    out.append(item)
out
`);
  const result = out.toJs({dict_converter: Object.fromEntries});
  out.destroy();
  return result;
}

onmessage = async (e) => {
  const {id, cmd} = e.data;
  try {
    const result = cmd === 'init' ? await init(e.data.threads) : await run(e.data.files, e.data.argv);
    const transfer = cmd === 'run' ? result.flatMap((r) => [r.data.buffer, r.xlsx && r.xlsx.buffer]).filter(Boolean) : [];
    postMessage({type: 'done', id, result}, transfer);
  } catch (err) {
    // The whole traceback to the log, its last line (the error itself) to the page
    const message = String(err && err.message || err).trim();
    post('log', {text: message});
    postMessage({type: 'error', id, text: message.split('\n').slice(-1)[0]});
  }
};

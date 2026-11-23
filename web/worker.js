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
  // The model is made now, while the heap is small: made during a run, its build took its place on top of the run's data
  await py.runPythonAsync('import candycrunch.prediction\nfrom candycrunch import web\ncandycrunch.prediction.default_model()');
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
  py.runPython("import os, shutil\nfor d in ('/out', '/preview'):\n    shutil.rmtree(d, ignore_errors = True)\n    os.makedirs(d)");
  py.globals.set('argv', py.toPy(argv));
  // Asynchronous, as the model's calls wait for onnxruntime-web
  await py.runPythonAsync('web.run(list(argv))');
  // Every file the CLI wrote; each table also as .xlsx and as text for the page's preview. They reach JavaScript as files of the in-memory file system:
  // Pyodide copies Python objects to JavaScript wrongly once they lie past 2 GiB of its heap, which a large run's heap passes
  await py.runPythonAsync(`
import json, os
import pandas as pd
for name in os.listdir('/out'):
    if name.endswith('.csv'):
        df = pd.read_csv(os.path.join('/out', name))
        df.to_excel(os.path.join('/preview', name + '.xlsx'), index = False)
        with open(os.path.join('/preview', name + '.json'), 'w', encoding = 'utf-8') as f:
            json.dump({'columns': [str(c) for c in df.columns], 'rows': df.astype(object).where(df.notna(), '').astype(str).values.tolist()}, f)
`);
  return py.FS.readdir('/out').filter((n) => n !== '.' && n !== '..').sort().map((name) => name.endsWith('.csv') ?
    {name, data: py.FS.readFile('/out/' + name), xlsx: py.FS.readFile('/preview/' + name + '.xlsx'), ...JSON.parse(py.FS.readFile('/preview/' + name + '.json', {encoding: 'utf8'}))} :
    {name, data: py.FS.readFile('/out/' + name)});
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

import ast
import functools
import http.server
import io
import os
import pathlib
import threading
import numpy as np
import pandas as pd
import pytest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
# A site built by packaging/build_web.py; the Pages workflow builds one and runs this before it deploys
SITE = os.environ.get('CC_WEB_SITE')


def same_predictions(x, y):
    # The same structures with probabilities within the float32 differences of ONNX Runtime and onnxruntime-web (4th decimal)
    if not isinstance(x, str) or not isinstance(y, str):
        return pd.isna(x) and pd.isna(y)
    x, y = dict(ast.literal_eval(x)), dict(ast.literal_eval(y))
    return x.keys() == y.keys() and all(abs(x[k] - y[k]) < 1e-3 for k in x)


@pytest.mark.skipif(not SITE, reason = 'needs a site built by packaging/build_web.py in CC_WEB_SITE (the Pages workflow runs this before deploying)')
def test_web_example_matches_native():
    # The page's example run (packaging/self_test.mzML, O-glycans) in headless Chromium against the same run natively on ONNX Runtime, the graph the
    # page runs on onnxruntime-web: the same table but for float32 noise in the model's probabilities
    sync_api = pytest.importorskip('playwright.sync_api')

    class Handler(http.server.SimpleHTTPRequestHandler):
        # The site under /CandyCrunch/ and without COOP/COEP headers, as GitHub Pages serves it (coi-serviceworker has to provide them)

        def translate_path(self, path):
            return super().translate_path(path.removeprefix('/CandyCrunch'))

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory = str(pathlib.Path(SITE).resolve())))
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/CandyCrunch/')
            # The first visit downloads Pyodide, its packages and the wheels (~100 MB) and reloads once for cross-origin isolation, which can destroy
            # the context a wait runs in
            for _ in range(5):
                try:
                    page.wait_for_function("['Ready', 'Not supported in this browser'].includes(document.getElementById('status').textContent) || "
                                           "document.getElementById('status').className === 'bad'", timeout = 600000)
                    break
                except sync_api.Error as e:
                    if 'context was destroyed' not in str(e) and 'navigation' not in str(e):
                        raise
            assert page.text_content('#status') == 'Ready', f"{page.text_content('#status')}\n{page.text_content('#log')}"
            assert page.evaluate('crossOriginIsolated'), 'no cross-origin isolation, so the model ran on one thread'
            page.click('#example')
            page.wait_for_function("['ok', 'bad'].includes(document.getElementById('status').className)", timeout = 600000)
            assert page.get_attribute('#status', 'class') == 'ok', f"{page.text_content('#status')}\n{page.text_content('#log')}"
            web = pd.read_csv(io.StringIO(page.evaluate("fetch(document.querySelector('#downloads a').href).then((r) => r.text())")))
            browser.close()
    finally:
        server.shutdown()
    from candycrunch import prediction
    from candycrunch.inference import CandyCrunchCNN, load_weights
    # The CLI's run and table (cli.main writes df_out.reset_index()), on the ONNX graph whether or not PyTorch is installed
    model = prediction.candycrunch if isinstance(prediction.candycrunch, CandyCrunchCNN) else CandyCrunchCNN(
        prediction.sdict if isinstance(prediction.sdict, dict) else load_weights(prediction.sdict))
    native = prediction.wrap_inference(str(BASE_DIR / 'packaging' / 'self_test.mzML'), ['O'], model = model).reset_index()
    native = pd.read_csv(io.StringIO(native.to_csv(index = False)))
    assert list(web.columns) == list(native.columns) and len(web) == len(native), (list(web.columns), len(web), len(native))
    for c in native.columns:
        a, b = native[c], web[c]
        if c == 'predictions':
            assert all(same_predictions(x, y) for x, y in zip(a, b)), c
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a, b, rtol = 1e-6, equal_nan = True), c
        else:
            assert a.fillna('').astype(str).tolist() == b.fillna('').astype(str).tolist(), c

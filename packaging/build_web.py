"""Builds the in-browser CandyCrunch (web/) into a folder for GitHub Pages: the page, the wheels it installs (this checkout's candycrunch, glycowork) and
the example run. Run by .github/workflows/pages.yml; locally, e.g. python packaging/build_web.py _site --glycowork ../glycowork, then serve _site"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description = 'Build the in-browser CandyCrunch site')
parser.add_argument('out', help = 'folder to build the site into (replaced)')
# The glycowork CI tests against (1.11.1 exists only on dev, not on PyPI)
parser.add_argument('--glycowork', default = 'glycowork @ git+https://github.com/BojarLab/glycowork.git@dev', help = 'pip requirement or path of the glycowork to build')
args = parser.parse_args()
out = pathlib.Path(args.out)
# Replaced only if it is empty or an earlier build: a slip such as . would delete everything in it
if out.exists() and any(out.iterdir()) and not (out / 'wheels' / 'manifest.json').exists():
    sys.exit(f'{out} is neither empty nor an earlier site build; give a new folder')
shutil.rmtree(out, ignore_errors = True)
shutil.copytree(ROOT / 'web', out)
wheels = out / 'wheels'
subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(wheels), str(ROOT), args.glycowork], check = True)
# Installed in this order by the page's worker (micropip takes their other dependencies from Pyodide and PyPI)
names = sorted((p.name for p in wheels.glob('*.whl')), key = lambda n: n.startswith('candycrunch'))
if len(names) != 2:
    sys.exit(f'expected the candycrunch and glycowork wheels, got {names}')
(wheels / 'manifest.json').write_text(json.dumps({'wheels': names}))
(out / 'example').mkdir()
shutil.copy(ROOT / 'packaging' / 'self_test.mzML', out / 'example')
print(f'built {out}: {", ".join(names)}')

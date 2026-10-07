# PyInstaller recipe for the CandyCrunch desktop app. From the repository root, with candycrunch[gui] and pyinstaller installed:
#     pyinstaller packaging/candycrunch.spec
# gives dist/CandyCrunch (Windows, Linux) or dist/CandyCrunch.app (macOS); packaging/candycrunch.iss turns the Windows folder into an installer
import os
import sys
from importlib.metadata import version
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
datas = collect_data_files('candycrunch') + collect_data_files('glycowork') + collect_data_files('glycorender')
# These read their own version from package metadata at import
for package in ('candycrunch', 'glycowork', 'glycorender'):
    datas += copy_metadata(package)
icon = os.path.join(SPECPATH, 'candycrunch.icns' if sys.platform == 'darwin' else 'candycrunch.ico')
a = Analysis([os.path.join(SPECPATH, 'candycrunch_app.py')], datas = datas,
             excludes = ['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'IPython', 'notebook', 'jupyter_client', 'pytest'],
             # The app reads its defaults from prediction.py's source, so candycrunch is kept as plain .py files
             module_collection_mode = {'candycrunch': 'py'})
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries = True, name = 'CandyCrunch', console = False, icon = icon)
coll = COLLECT(exe, a.binaries, a.datas, name = 'CandyCrunch')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name = 'CandyCrunch.app', icon = icon, bundle_identifier = 'org.bojarlab.candycrunch',
                 info_plist = {'CFBundleShortVersionString': version('candycrunch'), 'NSHighResolutionCapable': True})

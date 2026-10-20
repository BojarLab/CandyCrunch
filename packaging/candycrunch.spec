# PyInstaller recipe for the CandyCrunch desktop app. From the repository root, with candycrunch[gui] and pyinstaller installed:
#     pyinstaller packaging/candycrunch.spec
# gives dist/CandyCrunch (Windows, Linux) or dist/CandyCrunch.app (macOS); packaging/candycrunch.iss turns the Windows folder into an installer
import os
import re
import sys
from importlib.metadata import version
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
datas = collect_data_files('candycrunch') + collect_data_files('glycowork') + collect_data_files('glycorender')
# These read their own version from package metadata at import
for package in ('candycrunch', 'glycowork', 'glycorender'):
    datas += copy_metadata(package)
icon = os.path.join(SPECPATH, 'candycrunch.icns' if sys.platform == 'darwin' else 'candycrunch.ico')
a = Analysis([os.path.join(SPECPATH, 'candycrunch_app.py')], datas = datas,
             # Without PyTorch (only for training and GPU prediction) the app runs the model on ONNX Runtime, at half the size of the folder
             excludes = ['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'IPython', 'notebook', 'jupyter_client', 'pytest', 'torch'],
             # The app reads its defaults from prediction.py's source, so candycrunch is kept as plain .py files
             module_collection_mode = {'candycrunch': 'py'})
pyz = PYZ(a.pure)
# Windows version resource: antivirus heuristics count an executable without publisher, product and version as suspicious
version_resource = None
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct
    numbers = (tuple(map(int, re.findall(r'\d+', version('candycrunch'))[:4])) + (0, 0, 0, 0))[:4]
    version_resource = VSVersionInfo(ffi = FixedFileInfo(filevers = numbers, prodvers = numbers),
                                     kids = [StringFileInfo([StringTable('040904B0', [StringStruct(k, v) for k, v in (
                                         ('CompanyName', 'Bojar Lab, University of Gothenburg'), ('FileDescription', 'CandyCrunch: glycan structures from LC-MS/MS'),
                                         ('FileVersion', version('candycrunch')), ('InternalName', 'CandyCrunch'), ('LegalCopyright', 'MIT License'),
                                         ('OriginalFilename', 'CandyCrunch.exe'), ('ProductName', 'CandyCrunch'), ('ProductVersion', version('candycrunch')))])]),
                                             VarFileInfo([VarStruct('Translation', [1033, 1200])])])
exe = EXE(pyz, a.scripts, [], exclude_binaries = True, name = 'CandyCrunch', console = False, icon = icon, version = version_resource)
coll = COLLECT(exe, a.binaries, a.datas, name = 'CandyCrunch')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name = 'CandyCrunch.app', icon = icon, bundle_identifier = 'org.bojarlab.candycrunch',
                 info_plist = {'CFBundleShortVersionString': version('candycrunch'), 'NSHighResolutionCapable': True})

"""Entry point of the bundled CandyCrunch app (built with packaging/candycrunch.spec)"""
import multiprocessing
import sys
import types
import pymzml.obo
# A frozen pymzml looks for its bundled vocabulary files next to the executable and otherwise tries to download them, but PyInstaller keeps
# them with the package; this runs in every process the app starts, as worker processes start this same executable
pymzml.obo.sys = types.SimpleNamespace(frozen = False, executable = sys.executable)
if __name__ == '__main__':
    # Turns a worker process start into the worker instead of a second window
    multiprocessing.freeze_support()
    from candycrunch.gui import main
    main()

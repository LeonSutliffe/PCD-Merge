# PyInstaller spec for the bundled program.
# Build from the repository root:  pyinstaller packaging/pcd_merge.spec
# Output: dist/PCD Merge/  (a folder with "PCD Merge.exe" and its libraries)
import os
import sys

from PyInstaller.utils.hooks import collect_all

here = os.path.abspath(SPECPATH)
root = os.path.dirname(here)

datas, binaries, hiddenimports = [], [], []
for pkg in ("open3d",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
# Open3D's machine-learning add-ons (PyTorch / TensorFlow ops), examples and
# its own GUI resources are not used and are large.
_DROP = ("torch_ops", "tf_ops", "_ml3d", "examples", "resources")


def _keep(entry):
    # entries are (source path, destination folder)
    src = entry[0].replace("\\", "/")
    if "/open3d/" not in src:
        return True
    rel = src.split("/open3d/", 1)[1]
    return not any(k in rel for k in _DROP)


datas = [e for e in datas if _keep(e)]
# Export libraries: laspy (LAS), lazrs (LAZ compression, Rust extension) and
# pye57 (E57, C++ extension with its Xerces DLL).
for pkg in ("laspy", "lazrs", "pye57"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
binaries = [e for e in binaries if _keep(e)]
hiddenimports = [h for h in hiddenimports if not h.startswith(("open3d.ml", "open3d._ml3d"))]
hiddenimports += ["lzf", "pyqtgraph.opengl", "OpenGL.platform.win32", "OpenGL.arrays.numpymodule"]

# Large parts of Qt and optional extras we never use.
excludes = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore",
    "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtDesigner", "PyQt5", "PyQt6", "torch", "tensorflow", "matplotlib", "tkinter",
    "open3d.ml", "pytest",
]

a = Analysis(
    [os.path.join(here, "launcher.py")],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)


def _keep_toc(entry):
    # TOC entries are (destination name, source path, type)
    dest = entry[0].replace("\\", "/")
    if not dest.startswith("open3d/"):
        return True
    return not any(k in dest[len("open3d/"):] for k in _DROP + ("labextension", "nbextension"))


a.binaries = [e for e in a.binaries if _keep_toc(e)]
a.datas = [e for e in a.datas if _keep_toc(e)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PCD Merge",
    console=False,
    icon=os.path.join(here, "icon.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PCD Merge")

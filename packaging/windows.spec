# Build a windowed GUI plus a pipe-capable console worker sharing one runtime.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root/'assets'),'assets')]
datas += collect_data_files('tkinterdnd2')
datas += collect_data_files('reportlab')
datas += copy_metadata('yt-dlp')
# Some Python distributions ship an SSL policy file next to ssl.py. Preserve
# that policy with the runtime instead of dropping it or bypassing validation.
ssl_policy = Path(sys.base_prefix)/'Lib/codex-rejected-cas.pem'
if ssl_policy.is_file():
    datas.append((str(ssl_policy), '.'))
a = Analysis([str(root/'windows_entry.py')], pathex=[str(root)], datas=datas,
    hiddenimports=collect_submodules('yt_dlp'),
    excludes=['pytest','IPython','matplotlib','scipy','pandas','pymupdf','fitz'],
    noarchive=False)
pyz = PYZ(a.pure)
gui = EXE(pyz,a.scripts,[],exclude_binaries=True,name='GuitarTabExtractor',
    console=False,upx=False,disable_windowed_traceback=False)
worker = EXE(pyz,a.scripts,[],exclude_binaries=True,name='tabextract-worker',
    console=True,upx=False)
coll = COLLECT(gui,worker,a.binaries,a.datas,name='GuitarTabExtractor',upx=False)

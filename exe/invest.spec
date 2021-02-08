# coding=UTF-8
# -*- mode: python -*-
import sys
import os
import itertools
import glob
from PyInstaller.compat import is_win, is_darwin

# Global Variables
current_dir = os.getcwd()  # assume we're building from the project root
block_cipher = None
exename = 'invest'
conda_env = os.environ['CONDA_PREFIX']

if is_win:
    proj_datas = ((os.path.join(conda_env, 'Library/share/proj'), 'proj'))
else:
    proj_datas = ((os.path.join(conda_env, 'share/proj'), 'proj'))

kwargs = {
    'excludes': None,
    'pathex': sys.path,
    'hiddenimports': [
        'natcap.invest.ui.launcher',
        'distutils',
        'distutils.dist',
    ],
    'datas': [('qt.conf', '.'), proj_datas],
    'cipher': block_cipher,
}

cli_file = os.path.join(current_dir, 'src', 'natcap', 'invest', 'ui', 'launcher.py')
a = Analysis([cli_file], **kwargs)

# Compress pyc and pyo Files into ZlibArchive Objects
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create the executable file.

exe = EXE(
    pyz,
    a.scripts,
    name=exename,
    exclude_binaries=True,
    debug=False,
    strip=False,
    upx=False,
    console=True)

# Collect Files into Distributable Folder/File
dist = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="invest",  # name of the output folder
    strip=False,
    upx=False)

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

kwargs = {
    'excludes': None,
    'pathex': sys.path,
    'hiddenimports': [
        'distutils',
        'distutils.dist',
    ],
    'datas': [('qt.conf', '.')],
    'cipher': block_cipher,
}

cli_file = os.path.join(current_dir, 'src', 'launcher.py')
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

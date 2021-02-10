# coding=UTF-8
# -*- mode: python -*-
import sys
import os

# Global Variables
current_dir = os.getcwd()  # assume we're building from the project root

kwargs = {
    'datas': [('qt.conf', '.')],
}

cli_file = os.path.join(current_dir, 'src', 'example', 'launcher.py')
a = Analysis([cli_file], **kwargs)

# Compress pyc and pyo Files into ZlibArchive Objects
pyz = PYZ(a.pure, a.zipped_data)

# Create the executable file.
exe = EXE(
    pyz,
    a.scripts,
    name='example',
    exclude_binaries=True)

# Collect Files into Distributable Folder/File
dist = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='example')

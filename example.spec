# coding=UTF-8
# -*- mode: python -*-
import os


a = Analysis([os.path.join(os.getcwd(), 'src', 'example', 'launcher.py')])

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

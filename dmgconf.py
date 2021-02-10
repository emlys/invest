# Configuration script for DMGBuild

import os

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return str(int(total_size/1024.) + 1024*50) + 'K'

size = get_size(defines['exampledir'])
print('Volume size: %s' % size)
print('Packaging dirname %s' % defines['exampledir'])
_example_dirname = os.path.basename(defines['exampledir'])

badge_icon = os.path.join('.', 'invest.icns')
symlinks = {'Applications': '/Applications'}
files = [defines['exampledir']]

icon_locations = {
    _example_dirname: (220, 290),
    'Applications': (670, 290)
}
icon_size = 100
text_size = 12

# Window Settings
window_rect = ((100, 100), (900, 660))
background = 'builtin-arrow'
default_view = 'icon-view'

format = 'UDZO'
license = {
    # LICENSE.txt assumed to live in the project root.
    'licenses': {'en_US': 'LICENSE.txt'},
    'default-language': 'en_US',
}

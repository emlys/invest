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
files = [defines['exampledir']]
format = 'UDZO'

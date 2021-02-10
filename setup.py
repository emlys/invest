# encoding=UTF-8
import platform

from setuptools.extension import Extension
from setuptools import setup


# Read in requirements.txt and populate the python readme with the
# non-comment, non-environment-specifier contents.
_REQUIREMENTS = [req.split(';')[0].split('#')[0].strip() for req in
                 open('requirements.txt').readlines()
                 if (not req.startswith(('#', 'hg+', 'git+'))
                     and len(req.strip()) > 0)]
_GUI_REQUIREMENTS = [req.split(';')[0].split('#')[0].strip() for req in
                     open('requirements-gui.txt').readlines()
                     if not (req.startswith(('#', 'hg+'))
                             and len(req.strip()) > 0)]


# Since OSX Mavericks, the stdlib has been renamed.  So if we're on OSX, we
# need to be sure to define which standard c++ library to use.  I don't have
# access to a pre-Mavericks mac, so hopefully this won't break on someone's
# older system.  Tested and it works on Mac OSX Catalina.
compiler_and_linker_args = []
if platform.system() == 'Darwin':
    compiler_and_linker_args = ['-stdlib=libc++']

setup(
    name='example',
    namespace_packages=['example'],
    packages=['example'],
    package_dir={
        'example': 'src/example'
    },
    use_scm_version={'version_scheme': 'post-release',
                     'local_scheme': 'node-and-date'},
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    setup_requires=['setuptools_scm'],
    license='BSD',
    long_description_content_type='text/x-rst',
    zip_safe=False,
    extras_require={
        'ui': _GUI_REQUIREMENTS,
    }
)

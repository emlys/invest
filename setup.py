from setuptools import setup


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
    install_requires=['qtpy>1.3', 'qtawesome', 'PySide2!=5.15.0'],
    setup_requires=['setuptools_scm'],
    license='BSD',
    long_description_content_type='text/x-rst',
    zip_safe=False,
)

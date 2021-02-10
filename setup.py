from setuptools import setup


setup(
    name='example',
    namespace_packages=['example'],
    packages=['example'],
    package_dir={
        'example': 'src/example'
    },
    include_package_data=True,
    install_requires=['qtpy>1.3', 'qtawesome'],
    license='BSD',
    long_description_content_type='text/x-rst',
    zip_safe=False,
)

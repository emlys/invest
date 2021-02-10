from setuptools import setup


setup(
    name='example',
    namespace_packages=['example'],
    packages=['example'],
    package_dir={
        'example': 'src/example'
    },
    include_package_data=True,
    zip_safe=False,
)

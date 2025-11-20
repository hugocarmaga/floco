import setuptools

from src.floco import __version__, __author__, __license__

with open('README.md') as inp:
    long_description = inp.read()

with open('requirements.txt') as inp:
    requirements = list(map(str.strip, inp))


setuptools.setup(
    name='floco',
    version=__version__,
    author=__author__,
    license=__license__,
    description='Sequence-to-graph alignment based copy number calling using a network flow formulation',
    long_description=long_description,
    url='https://github.com/hugocarmaga/floco',

    package_dir={'': 'src/floco'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.10',
    install_requires=requirements,
    include_package_data=True,

    entry_points = dict(console_scripts=['floco=floco:main']),
    )
from io import open
from os import path

from setuptools import find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='stesml',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A package for the development of ML models for predicting sulfur thermal energy storage system behavior.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.nrel.gov/amo-element16/sulfur_tes_ml',  # Optional
    author='Dmitry Duplyakin, Kevin Menear',
    author_email='dmitry.duplyakin@nrel.gov, kevin.menear@nrel.gov',  # Optional
    classifiers=[
        'Development Status :: 2 - Pre-Alpha Copy',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD 3-Clause',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),  # Required
    install_requires=[],  # TODO
    project_urls={
        'Source': 'https://github.nrel.gov/amo-element16/sulfur_tes_ml',
    },
)

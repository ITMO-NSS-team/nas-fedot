from pathlib import Path
from typing import List

import setuptools

# The directory containing this file
HERE = Path(__file__).parent.resolve()

# The text of the README file
NAME = 'nas-fedot'
VERSION = '0.1.2'
AUTHOR = 'NSS Lab'
SHORT_DESCRIPTION = 'Neural architecture search'
README = Path(HERE, 'README.md').read_text(encoding='utf-8')
URL = 'https://github.com/ITMO-NSS-team/nas-fedot/archive/master.zip'
REQUIRES_PYTHON = '>=3.9'
LICENSE = 'BSD 3-Clause'


def _readlines(*names: str, **kwargs) -> List[str]:
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    out = [line for line in _readlines(file_name) if line and not line.startswith('#')]
    return out


def _get_requirements(req_name: str):
    requirements = _extract_requirements(req_name)
    return requirements


setuptools.setup(
    install_requires=_get_requirements('requirements.txt'),
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(exclude=['tests*']),
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9'
    ],
)

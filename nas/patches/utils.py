import os
import sys

from pathlib import Path


def set_root(root: Path):
    os.chdir(root)
    sys.path.append(root)


def project_root() -> Path:
    """Returns FEDOT project root folder."""
    return Path(__file__).parent.parent.parent

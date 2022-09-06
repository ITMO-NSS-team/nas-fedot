import os
import random
import sys
from pathlib import Path

import numpy as np


def set_root(root: Path):
    os.chdir(root)
    sys.path.append(str(root))


def seed_all(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)


def project_root() -> Path:
    """Returns NAS project root folder."""
    return Path(__file__).parent.parent.parent

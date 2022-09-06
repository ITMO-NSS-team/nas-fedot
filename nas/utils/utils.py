import datetime
import os
import pathlib
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_root(root: Path):
    os.chdir(root)
    sys.path.append(root)


def seed_all(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)


def project_root() -> Path:
    """Returns NAS project root folder."""
    return Path(__file__).parent.parent.parent


# def save_on_exception(func):
#     set_root(project_root())
#
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             graph = args[0]
#             path = pathlib.Path(f'../_results/debug/graph/{datetime.datetime.now().date()}')
#             path.mkdir(parents=True, exist_ok=True)
#             exception_type, exception_object, exception_traceback = sys.exc_info()
#             filename = exception_traceback.tb_frame.f_code.co_filename
#             line_number = exception_traceback.tb_lineno
#             graph.save(path)
#             raise Exception(f'Exception {e} happened during {func.__name__}.'
#                             f'\n{exception_type, filename, line_number}'
#                             f'\nSave current graph into {path.resolve()}')
#
#     return wrapper

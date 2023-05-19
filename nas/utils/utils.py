import os
import os
import random
import sys
from pathlib import Path

import numpy as np
# import nvidia_smi
# import tensorflow
# from keras.backend import get_session, clear_session


def set_root(root: Path):
    os.chdir(root)
    sys.path.append(str(root))


def seed_all(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)


def project_root() -> Path:
    """Returns NAS project root folder."""
    return Path(__file__).parent.parent.parent


# def clear_session():
#     session = tensorflow.compat.v1.Session()


# def log_gpu_memory(func):
#     def memory_logger(*args, **kwargs):
#         result = func(*args, **kwargs)
#
#         log = kwargs.get('log')
#
#         nvidia_smi.nvmlInit()
#         handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#         # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
#
#         info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#         if log:
#             log.info(f'Total memory\t\t{float(info.total) / 1024 ** 3:.5f} GiB')
#             log.info(f'Used memory \t\t{float(info.used) / 1024 ** 3:.5f} GiB')
#             log.info(f'Free memory \t\t{float(info.free) / 1024 ** 3:.5f} GiB')
#         return result
#
#     return memory_logger


# def clear_keras_session(**kwargs):
#     log = kwargs.get('log')
#
#     sess = get_session()
#     clear_session()
#     sess.close()
#     sess = get_session()
#
#     # if log:
#     #     log.info(gc.collect())
#     #
#     # config = tensorflow.compat.v1.ConfigProto()
#     # config.gpu_options.per_process_gpu_memory_fraction = 0.7
#     # config.gpu_options.visible_device_list = "0"
#     # set_session(tensorflow.compat.v1.Session(config=config))

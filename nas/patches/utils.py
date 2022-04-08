import os
import sys


def set_root(file):
    root = os.path.dirname(os.path.abspath(file))
    os.chdir(root)
    sys.path.append(root)

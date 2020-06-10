import os
import shutil
from collections import MutableMapping, namedtuple

import pathlib

Timestep = namedtuple("Timestep", "observation action reward next_observation is_terminal info")


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def isdir(path):
    return os.path.isdir(path)


def subdirectories(path):
    try: return [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    except FileNotFoundError: return []


def files(path):
    return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]


def flatten_dict(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping): items.extend(flatten_dict(value, key, separator).items())
        else: items.append((key, value))
    return dict(items)
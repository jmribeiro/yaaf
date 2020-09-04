import os
import shutil
from collections import MutableMapping, namedtuple
import numpy as np

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


def ndarray_index_from(collection, array):
    return [np.array_equal(array, other) for other in collection].index(True)


def replace_in_directory(path: str, old: str, new: str):

    """
    Replaces a string found in any subdirectory or file of a directory (recursively).

    E.g.

        - resources
            - results
                - agent1
                    - agent1_scores
                    - agent1_other_stuff
                    - agent1_etc

        replace_in_directory("resources", old="agent1", new="dqn")

        - resources
            - results
                - dqn
                    - dqn_scores
                    - dqn_other_stuff
                    - dqn_etc

        replace_in_directory("resources/results/dqn", old="dqn_", new="")

        - resources
            - results
                - dqn
                    - scores
                    - other_stuff
                    - etc

    """

    for subdirectory in subdirectories(path):
        replace_in_directory(f"{path}/{subdirectory}", old, new)

    for file in files(path):
        new_file = file.replace(old, new)
        os.rename(f"{path}/{file}", f"{path}/{new_file}")

    path = "/".join(path.split("/")[:-1])
    directory = path.split("/")[-1]

    if old in directory:
        directory_new = f"{path}/{directory.replace(old, new)}"
        os.rename(path, directory_new)

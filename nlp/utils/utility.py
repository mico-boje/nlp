import os
from pathlib import Path


def get_root_path():

    return Path(os.path.abspath(__file__)).parent.parent.parent

if __name__ == "__main__":
    print(get_root_path())
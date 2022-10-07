from pathlib import Path
import sys


def add_path(path):
    """
    add path into `sys.path`

    Args:
        path (PosixPath): path to be added
    """
    if path not in sys.path:
        sys.path.insert(0, str(path))


this_dir = Path(__file__).parent

lib_path = this_dir / '..' / 'lib'
add_path(lib_path)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, str(path))


this_dir = Path(__file__).parent

lib_path = this_dir / '..' / 'lib'
add_path(lib_path)
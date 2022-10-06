"""
test import
"""

import sys
import _init_path
# import models
from models.model_x import model_x
from core.subcore.subcore_function import subcore_function

def main():
    print("this message from train.py")

if __name__ == '__main__':
    print(sys.path)
    model_x()
    subcore_function()
    main()
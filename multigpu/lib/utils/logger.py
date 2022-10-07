"""
set up logger, matched to multigpu-mode
"""
import logging
import sys
import time
from pathlib import Path


def setup_logger(name,
                 show_terminal:bool=False,
                 save_dir:Path()=None,
                 local_rank:int=0,
                 suffix=".log"):
    """
    Args:
        name: logger name
        show_terminal: whether show log in the terminal
        save_dir: PosixPath()
        local_rank: GPU Device
        filename: %m%d-%H%M.log
    Return:
        logger
    """
    logger = logging.getLogger(name)
    # don't log results from non-master processes
    if local_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s]: \'%(message)s\'")

    if show_terminal:
        ch = logging.StreamHandler(stream=sys.stdout)
        # ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        local_time = time.strftime("%m%d-%H", time.localtime())
        filename = local_time + suffix
        # mkdir
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        fh = logging.FileHandler(save_dir / filename)
        # fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# if __name__ == '__main__':
#     test_logger = setup_logger('Test_logger', save_dir=Path('outputs/log'), local_rank=0)
#     results = [1,2,3]
#     test_logger.info("test %s logger!", results)
#     test_logger.debug("debug message")

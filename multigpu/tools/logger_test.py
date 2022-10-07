import _init_path

from pathlib import Path

from utils.logger import setup_logger


def main():
    t_logger = setup_logger('terminal_logger', show_terminal=True)
    t_logger.info('this is a terminal logger')
    t_logger.info('the level is info')

    n_logger = setup_logger('none_logger', show_terminal=None)
    n_logger.info('this is none logger') # shouldn't be printed on the screen

    path = Path('/home/huangzhanghao/data/codes/experiments/logger/outputs')
    f_logger = setup_logger('file_logger', False, path)
    f_logger.info('this is a file logger')

if __name__ == '__main__':
    main()

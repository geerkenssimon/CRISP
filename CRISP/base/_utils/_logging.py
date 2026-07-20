import logging
import os
from datetime import date, datetime
from functools import wraps

MAX_LINES = 100000
LINE_BUFFER = 200

filelogger = logging.getLogger()
filelogger.setLevel(logging.INFO)

INDEX = 0

def start_logging(caller):
    """Initialize a logger to log progress, errors, and warnings into a log file"""
    global INDEX
    INDEX = 0
    log_without_format(f'\n\n=========================================================================================\n'+\
                        f'Logging of {caller}\n'+ \
                        f'started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'+ \
                        '=========================================================================================\n',
                        preserve_current=False,
                        force_new=True)

def log_without_format(log_str: str, force_new=False, preserve_current=True):
    _change_handler(info=False, force_new=force_new, preserve_current=preserve_current)
    filelogger.info(log_str)
    _change_handler(force_new=False, preserve_current=True)

def _get_log_filename(force_new=False, preserve=True):
    """Generate log filename with incrementing index if necessary."""
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    base_filename = os.path.join(log_dir, f'QUEEN_log_{date.today().strftime("%Y-%m-%d")}')
    global INDEX
    #index = 0
    while True:
        log_filename = f"{base_filename}_{INDEX}.log"
        line_count = _count_lines(log_filename)
        if force_new and line_count >= MAX_LINES - LINE_BUFFER:
            INDEX += 1
            continue
        if not os.path.exists(log_filename) or line_count < MAX_LINES or preserve:
            return log_filename
        INDEX += 1

def _count_lines(filename):
    """Count the number of lines in a log file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def _change_handler(info: bool = True, force_new: bool = False, preserve_current: bool = False):
    log_filename = _get_log_filename(force_new=force_new, preserve=preserve_current)
    #if preserve_current and _count_lines(log_filename) <= MAX_LINES:
    #    return  # Preserve current log until a new session starts
    
    _remove_handler(filelogger)
    fh = logging.FileHandler(filename=log_filename, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s') if info else logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    filelogger.addHandler(fh)

def _remove_handler(logger: logging.Logger):
    """Remove all handlers from a logger."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

def exception_handler(logging_str=None):
    """Exception handler used as a decorator.
    
    Wraps execution in a try-except block with traceback logging.
    
    Args:
        logging_str (str, optional): String to log before the traceback. Defaults to None.
    """
    def decorator(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt as e:
                print('Exiting due to KeyboardInterrupt')
                filelogger.warning('Exiting due to KeyboardInterrupt')
                raise e
            except Exception as e:
                if logging_str is not None:
                    filelogger.error(logging_str)
                filelogger.error('Traceback:', exc_info=e)
                raise e
        return inner
    return decorator

from typing import Union
import logging
import pathlib 

# https://towardsdatascience.com/8-advanced-python-logging-features-that-you-shouldnt-miss-a68a5ef1b62d

#_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
#_log_format = f"%(name)s-(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
#_log_format = f"-(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
_log_format = f"%(message)s"

def get_file_handler(filepath: Union[str, pathlib.PosixPath]):
    if isinstance(filepath, pathlib.PosixPath):
        filepath = str(filepath)
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.WARNING)
    #file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(filepath: Union[str, pathlib.PosixPath], name: str):
    if isinstance(filepath, pathlib.PosixPath):
        filepath = str(filepath)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_file_handler(filepath))
    logger.addHandler(get_stream_handler())
    return logger

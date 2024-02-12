import logging
import sys

from logging.handlers import RotatingFileHandler
from datetime import datetime
from os import path, mkdir

from torch.cuda import is_available, device_count

def get_logger():
    return logging.getLogger("datasnipper_hometask")

def get_logging_folder() -> str:
    log_folder = "logs/"
    if not path.exists(log_folder):
        mkdir(log_folder)
    return log_folder

def init_logs(verbose: bool = True, level = logging.DEBUG):
    fname = f"{get_logging_folder()}/log_{datetime.now():%Y%m%d_%T%H%M}.log"
    
    logger = get_logger()
    logger.propagate = False
    logger.setLevel(level)
    
    FORMAT = "%(asctime)s | %(threadName)-12.12s | %(levelname)-8s | %(message)s"
    formatter = logging.Formatter(FORMAT)
    
    should_roll_over = path.isfile(fname)
    f_handler = RotatingFileHandler(fname, mode="a", backupCount=10)
    f_handler.setFormatter(formatter)
    if (should_roll_over):
        f_handler.do_rollover
    logger.addHandler(f_handler)
    
    if verbose:
        s_handler = logging.StreamHandler(sys.stdout)
        s_handler.setFormatter(formatter)
        logger.addHandler(s_handler)
    
    logger.info("DataSnipper Home Task")
    logger.info(f"Number of GPU's available: {device_count()}")
    logger.info(f"Using device: {'cuda' if is_available() else 'cpu'}")
    logger.info(f"Checking MPS device: {""}")

def c(comment: str):
    get_logger().info(comment)

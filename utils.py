import os
import sys
import logging
import datetime

def set_logger(log_to_file):
    if not os.path.exists('./logging'):
        os.makedirs('./logging')
    
    currtime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"logging/{currtime}.log"),
            logging.StreamHandler()
        ] if log_to_file 
        else [logging.StreamHandler()]
    )
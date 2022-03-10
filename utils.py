import os
import sys
import logging
import datetime

def set_logger():
    if not os.path.exists('./logging'):
        os.makedirs('./logging')
    
    currtime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"logging/{currtime}.log"),
            logging.StreamHandler()
        ]
    )
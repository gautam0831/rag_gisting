import logging
import os
import datetime
from glob import glob

def create_logger():
    # Current date for the filename
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # List of existing log files for today
    existing_logs = glob(f'logs/{current_date}_*.log')

    # Determine the next run number
    run_number = len(existing_logs) + 1

    # Create a unique filename
    filename = f"logs/{current_date}_{run_number}.log"

    # Configure logging
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(asctime)s :: %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    return logging
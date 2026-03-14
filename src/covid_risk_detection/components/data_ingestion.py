import os
import shutil
import logging
from covid_risk_detection.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self):
        logging.info(f"Copying data from {self.config.local_data_dir} to {self.config.root_dir}")
        
        for filename in os.listdir(self.config.local_data_dir):
            if filename.endswith(".csv"):
                src_path = os.path.join(self.config.local_data_dir, filename)
                dest_path = os.path.join(self.config.root_dir, filename)
                shutil.copy2(src_path, dest_path)
                logging.info(f"Copied: {filename}")
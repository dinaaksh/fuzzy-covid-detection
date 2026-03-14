from covid_risk_detection.config.configuration import ConfigurationManager
from covid_risk_detection.components.data_transformation import DataTransformation
import logging

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        ingestion_config = config.get_data_ingestion_config()
        data_dir = ingestion_config.root_dir
        
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation(data_dir=data_dir)

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
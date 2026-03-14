import os
from covid_risk_detection.config.configuration import ConfigurationManager
from covid_risk_detection.components.model_trainer import ModelTrainer
import logging

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        
        data_transformation_config = config.get_data_transformation_config()
        data_path = data_transformation_config.transformed_data_file
        base_features_path = os.path.join(data_transformation_config.root_dir, "base_features.pkl")
        
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train(data_path=data_path, base_features_path=base_features_path)

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
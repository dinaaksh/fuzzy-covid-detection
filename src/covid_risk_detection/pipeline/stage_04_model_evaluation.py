import os
from covid_risk_detection.config.configuration import ConfigurationManager
from covid_risk_detection.components.model_evaluation import ModelEvaluation
import logging

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_trainer_config = config.get_model_trainer_config()
        
        model_path = model_trainer_config.trained_model_path
        X_test_path = os.path.join(model_trainer_config.root_dir, "X_test.pkl")
        y_test_path = os.path.join(model_trainer_config.root_dir, "y_test.pkl")
        
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate(
            model_path=model_path, 
            X_test_path=X_test_path, 
            y_test_path=y_test_path
        )

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
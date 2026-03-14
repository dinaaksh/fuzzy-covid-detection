import logging
from covid_risk_detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from covid_risk_detection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from covid_risk_detection.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from covid_risk_detection.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Model Trainer Stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
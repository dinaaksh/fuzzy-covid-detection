from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_data_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    model_features_path: Path
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int     # ADD
    scale_pos_weight: float   # ADD
    random_state: int
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metric_file_name: Path
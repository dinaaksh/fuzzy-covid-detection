import os
import joblib
import numpy as np
import logging
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score
from covid_risk_detection.entity.config_entity import ModelEvaluationConfig
from covid_risk_detection.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self, model_path: str, X_test_path: str, y_test_path: str):
        logging.info("Loading model and test datasets for evaluation...")
        calibrated_model = joblib.load(model_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        y_scores = calibrated_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_scores)
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx  = np.argmax(f1_scores[:-1])
        balanced_thresh = thresholds[best_idx]
        
        y_pred_b = (y_scores >= balanced_thresh).astype(int)
        
        precision = precision_score(y_test, y_pred_b)
        recall = recall_score(y_test, y_pred_b)
        f1 = f1_score(y_test, y_pred_b)

        metrics = {
            "roc_auc_score": float(roc_auc),
            "best_f1_score": float(f1),
            "precision_at_best_f1": float(precision),
            "recall_at_best_f1": float(recall),
            "balanced_threshold": float(balanced_thresh)
        }

        logging.info(f"Evaluation completed. Metrics: {metrics}")
        save_json(path=self.config.metric_file_name, data=metrics)
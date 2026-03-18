import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from covid_risk_detection.utils.common import save_json
from covid_risk_detection.entity.config_entity import ModelEvaluationConfig
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self, **kwargs): 
        model_path = getattr(self.config, 'model_path', "artifacts/model_trainer/final_model.pkl")
        X_test_path = "artifacts/model_trainer/X_test.pkl"
        y_test_path = "artifacts/model_trainer/y_test.pkl"

        model = joblib.load(model_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        y_probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        
        beta = 1.5
        f2_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        
        best_idx = np.argmax(f2_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred_best = (y_probs >= best_threshold).astype(int)
        
        final_precision = precision_score(y_test, y_pred_best)
        final_recall = recall_score(y_test, y_pred_best)
        final_f1 = f1_score(y_test, y_pred_best) 
        final_f2 = f2_scores[best_idx]
        
        cm = confusion_matrix(y_test, y_pred_best)
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "roc_auc_score": float(roc_auc),
            "best_f2_score": float(final_f2),
            "best_f1_score": float(final_f1),
            "precision": float(final_precision),
            "recall": float(final_recall),
            "clinical_threshold": float(best_threshold),
            "confusion_matrix": {
                "True_Negatives": int(tn),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "True_Positives": int(tp)
            }        
        }

        save_json(path=self.config.metric_file_name, data=metrics)
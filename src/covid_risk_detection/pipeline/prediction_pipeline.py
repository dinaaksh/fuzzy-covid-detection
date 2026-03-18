import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from covid_risk_detection.components.fuzzy_ontology import FuzzyOntologyEngine


class PredictionPipeline:
    def __init__(self):
        self.model_path    = Path("artifacts/model_trainer/final_model.pkl")
        self.features_path = Path("artifacts/model_trainer/model_features.pkl")

        logging.info("Loading model and initializing Fuzzy Engine...")
        self.model        = joblib.load(self.model_path)
        self.all_features = joblib.load(self.features_path)
        self.foe          = FuzzyOntologyEngine()

    def predict(self, raw_data: dict) -> str:
        df = pd.DataFrame([raw_data])

        binary_features = [
            'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob', 'fatigue',
            'headache', 'muscle_sore', 'sore_throat', 'runny_nose', 'diarrhea',
            'diabetes', 'high_risk_exposure_occupation', 'high_risk_interactions', 'smoker',
        ]
        for col in binary_features:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        vital_cols = ['temperature', 'pulse', 'sats', 'rr']
        for col in vital_cols:
            if col not in df.columns:
                df[col] = None
            raw_val = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_measured'] = raw_val.notna().astype(int)
            df[col] = raw_val.fillna(0).astype(float)

        if 'age' not in df.columns:
            df['age'] = 35.0
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35.0)

        df['exposure_risk']   = (df['high_risk_exposure_occupation'] | df['high_risk_interactions']).astype(int)
        df['has_comorbidity'] = df['diabetes'].copy()

        symptom_cols = [
            'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob',
            'fatigue', 'headache', 'muscle_sore', 'diarrhea'
        ]
        df['symptom_count'] = df[symptom_cols].sum(axis=1)

        df['smell_and_taste_loss']   = (df['loss_of_smell'] & df['loss_of_taste']).astype(int)
        df['fever_and_smell_loss']   = (df['fever']         & df['loss_of_smell']).astype(int)
        df['cough_and_smell_loss']   = (df['cough']         & df['loss_of_smell']).astype(int)
        df['fatigue_and_smell_loss'] = (df['fatigue']       & df['loss_of_smell']).astype(int)
        df['fever_and_cough']        = (df['fever']         & df['cough']).astype(int)
        df['fever_and_muscle_sore']  = (df['fever']         & df['muscle_sore']).astype(int)
        df['fever_cough_smell']      = (df['fever'] & df['cough'] & df['loss_of_smell']).astype(int)
        df['fever_cough_taste']      = (df['fever'] & df['cough'] & df['loss_of_taste']).astype(int)


        row_dict   = df.iloc[0].to_dict()
        fis_result = self.foe.score_row_with_memberships(row_dict)

        df['fis_symptom_burden']  = fis_result['fis_symptom_burden']
        df['fis_covid_signature'] = fis_result['fis_covid_signature']
        df['fis_vitals_risk']     = fis_result['fis_vitals_risk']
        df['fis_covid_score']     = fis_result['fis_covid_score']

        for col in self.all_features:
            if col not in df.columns:
                df[col] = 0
        X = df[self.all_features].values
        xgb_probability = float(self.model.predict_proba(X)[0][1])

        return json.dumps({
            "status":              "success",
            "xgb_probability":     round(xgb_probability, 4),
            "fis_covid_score":     round(fis_result['fis_covid_score'],       4),
            "fis_covid_signature": round(fis_result['fis_covid_signature'],   4),
            "fis_symptom_burden":  round(fis_result['fis_symptom_burden'],    4),
            "fis_vitals_risk":     round(fis_result['fis_vitals_risk'],       4),
            "memberships": {
                "low":    round(fis_result['memberships']['low'],    3),
                "medium": round(fis_result['memberships']['medium'], 3),
                "high":   round(fis_result['memberships']['high'],   3),
            },
            "primary_driver": fis_result['primary_driver'],
        })
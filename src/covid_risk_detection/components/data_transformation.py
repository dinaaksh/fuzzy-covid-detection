import os
import glob
import joblib
import logging
import pandas as pd
import numpy as np
from covid_risk_detection.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self, data_dir: str):
        logging.info("--- STARTING DATA TRANSFORMATION ---")

        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)

        df = df[df['covid19_test_results'].isin(['Positive', 'Negative'])].copy()
        df['target'] = df['covid19_test_results'].map({'Positive': 1, 'Negative': 0})

        binary_features = [
            'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob', 'fatigue',
            'headache', 'muscle_sore', 'sore_throat', 'runny_nose', 'diarrhea',
            'diabetes',
            'high_risk_exposure_occupation',     
            'high_risk_interactions',            
            'smoker',                          
        ]
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map(
                    {'true': 1, 'false': 0, '1': 1, '0': 0,
                     'yes': 1, 'no': 0, 'positive': 1, 'negative': 0}
                ).fillna(0).astype(int)
            else:
                df[col] = 0

        vital_cols = ['temperature', 'pulse', 'sats', 'rr']
        vital_flag_features = []
        for col in vital_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[f'{col}_measured'] = df[col].notna().astype(int)
                df[col] = df[col].fillna(0).astype(float)
            else:
                df[col] = 0.0
                df[f'{col}_measured'] = 0
            vital_flag_features.append(f'{col}_measured')

        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].median())
        else:
            df['age'] = 0.0

        continuous_features = ['temperature', 'pulse', 'sats', 'rr', 'age']

        df['exposure_risk'] = (
            df['high_risk_exposure_occupation'] | df['high_risk_interactions']
        ).astype(int)

        symptom_cols = [
            'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob',
            'fatigue', 'headache', 'muscle_sore', 'diarrhea'
        ]
        df['symptom_count'] = df[symptom_cols].sum(axis=1)

        df['has_comorbidity'] = df['diabetes'].copy()

        engineered_features = ['exposure_risk', 'symptom_count', 'has_comorbidity']

        df['smell_and_taste_loss']   = (df['loss_of_smell'] & df['loss_of_taste']).astype(int)
        df['fever_and_smell_loss']   = (df['fever']         & df['loss_of_smell']).astype(int)
        df['cough_and_smell_loss']   = (df['cough']         & df['loss_of_smell']).astype(int)
        df['fatigue_and_smell_loss'] = (df['fatigue']       & df['loss_of_smell']).astype(int)
        df['fever_and_cough']        = (df['fever']         & df['cough']).astype(int)
        df['fever_and_muscle_sore']  = (df['fever']         & df['muscle_sore']).astype(int)
        df['fever_cough_smell']      = (df['fever'] & df['cough'] & df['loss_of_smell']).astype(int)
        df['fever_cough_taste']      = (df['fever'] & df['cough'] & df['loss_of_taste']).astype(int)

        interaction_features = [
            'smell_and_taste_loss', 'fever_and_smell_loss', 'cough_and_smell_loss',
            'fatigue_and_smell_loss', 'fever_and_cough', 'fever_and_muscle_sore',
            'fever_cough_smell', 'fever_cough_taste',
        ]

        df = df[(df['target'] == 1) | (df['symptom_count'] > 0)].copy().reset_index(drop=True)

        logging.info(f"Rows after symptom filter: {len(df):,}  "
                     f"(pos={df['target'].sum():,}, neg={(df['target']==0).sum():,})")

        all_features = list(dict.fromkeys(
            binary_features
            + continuous_features
            + vital_flag_features
            + engineered_features
            + interaction_features
        ))

        os.makedirs(self.config.root_dir, exist_ok=True)
        df.to_csv(self.config.transformed_data_file, index=False)
        joblib.dump(all_features, os.path.join(self.config.root_dir, "base_features.pkl"))

        logging.info(f"Total features : {len(all_features)}")
        logging.info(f"Feature list   : {all_features}")
        logging.info(f"Saved to       : {self.config.transformed_data_file}")
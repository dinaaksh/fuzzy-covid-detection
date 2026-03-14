import os
import glob
import joblib
import logging
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from covid_risk_detection.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self, data_dir: str):
        logging.info("--- STARTING DATA TRANSFORMATION & ENGINEERING ---")
        
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)
        
        df = df[df['covid19_test_results'].isin(['Positive', 'Negative'])].copy()
        df['target'] = df['covid19_test_results'].map({'Positive': 1, 'Negative': 0})

        binary_features = [
            'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob', 'fatigue',
            'headache', 'muscle_sore', 'sore_throat', 'runny_nose', 'diarrhea',
            'diabetes', 'high_risk_exposure_occupation', 'high_risk_interactions',
            'chd', 'htn', 'cancer', 'asthma', 'copd', 'autoimmune_dis', 'smoker',
            'labored_respiration', 'rhonchi', 'wheezes', 'ctab'
        ]
        
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map(
                    {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'positive': 1, 'negative': 0}
                ).fillna(0).astype(int)
            else:
                df[col] = 0

        for col in ['temperature', 'pulse', 'sys', 'dia', 'rr', 'sats', 'age']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
            else:
                df[col] = 0.0

        df['sats'] = df['sats'].replace(0, 98.0)
        df['rr']   = df['rr'].replace(0, 16.0)
        df['temperature'] = df['temperature'].replace(0, 37.0)
        df['duration'] = pd.to_numeric(df.get('days_since_symptom_onset', pd.Series([2]*len(df))), errors='coerce').fillna(2)

        for col in ['cough_severity', 'sob_severity']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map({'nan': 0, '': 0, 'mild': 1, 'moderate': 2, 'severe': 3}).fillna(0).astype(int)
            else:
                df[col] = 0

        if 'rapid_flu_results' in df.columns:
            flu_raw = df['rapid_flu_results'].astype(str).str.strip().str.lower()
            df['flu_negative'] = (flu_raw == 'negative').astype(int)
            df['flu_positive'] = (flu_raw == 'positive').astype(int)
            df['flu_tested']   = flu_raw.isin(['negative', 'positive']).astype(int)
        else:
            df['flu_negative'] = df['flu_positive'] = df['flu_tested'] = 0

        df['sats_concerning'] = ((df['sats'] < 95) & (df['sats'] >= 92) & (df['sats'] > 0)).astype(int)
        df['sats_critical']   = ((df['sats'] < 92) & (df['sats'] > 0)).astype(int)
        df['tachypnea']       = (df['rr'] > 20).astype(int)
        df['age_young']  = (df['age'] < 40).astype(int)
        df['age_middle'] = ((df['age'] >= 40) & (df['age'] < 60)).astype(int)
        df['age_senior'] = (df['age'] >= 60).astype(int)
        df['comorbidity_count'] = df[['diabetes','chd','htn','cancer','asthma','copd','autoimmune_dis']].sum(axis=1)
        df['exposure_risk'] = (df['high_risk_exposure_occupation'] | df['high_risk_interactions']).astype(int)
        
        symptom_cols = ['cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob', 'fatigue', 'headache', 'muscle_sore', 'diarrhea']
        df['symptom_count'] = df[symptom_cols].sum(axis=1)

        df['weighted_covid_score'] = (
            df['loss_of_smell'] * 3.5 + df['loss_of_taste'] * 2.0 + (df['loss_of_smell'] & df['loss_of_taste']).astype(int) * 2.0 +
            df['diarrhea'] * 1.5 + df['fever'] * 1.0 + df['muscle_sore'] * 1.0 + df['cough_severity']* 0.5 +
            df['flu_negative'] * 2.0 + df['flu_positive'] * -3.0 + df['runny_nose'] * -1.0 + df['sob'] * 0.0 + df['fatigue'] * 0.0
        )

        df['covid_signature']      = (df['loss_of_smell'] & df['loss_of_taste']).astype(int)
        df['smell_only']           = (df['loss_of_smell'] & (1 - df['loss_of_taste'])).astype(int)
        df['taste_only']           = (df['loss_of_taste'] & (1 - df['loss_of_smell'])).astype(int)
        df['sob_with_anosmia']     = (df['sob'] & df['loss_of_smell']).astype(int)
        df['fatigue_with_anosmia'] = (df['fatigue'] & df['loss_of_smell']).astype(int)
        df['fever_no_uri']         = (df['fever'] & (1 - df['runny_nose']) & (1 - df['sore_throat'])).astype(int)
        df['multisystem_covid']    = (df['diarrhea'] & (df['cough'] | df['sob'])).astype(int)
        df['exposed_anosmia']      = (df['exposure_risk'] & df['loss_of_smell']).astype(int)
        df['flu_syndrome']         = (df['runny_nose'] & df['sore_throat'] & df['headache'] & (1 - df['loss_of_smell'])).astype(int)
        df['systemic_covid']       = (df['fever'] & df['fatigue'] & df['sob']).astype(int)
        df['rapid_onset_light']    = ((df['duration'] <= 3) & (df['symptom_count'].between(1, 3))).astype(int)
        df['flu_excl_respiratory'] = (df['flu_negative'] & df['flu_tested'] & (df['cough'] | df['sob'] | df['fever'])).astype(int)
        df['age_smell_interaction']= df['age_senior'] * df['loss_of_smell']
        df['respiratory_alarm']    = (df['sob'] & (df['sats_concerning'] | df['sats_critical']) & df['tachypnea']).astype(int)
        
        df['duration_x_fever'] = df['duration'] * df['fever']
        df['duration_x_anosmia'] = df['duration'] * df['loss_of_smell']
        df['duration_x_sob'] = df['duration'] * df['sob']
        df['age_x_fatigue'] = df['age'] * df['fatigue']
        df['senior_prolonged'] = (df['age_senior'] & (df['duration'] > 5)).astype(int)
        df['silent_hypoxia'] = (df['sats_concerning'] & (1 - df['sob'])).astype(int)
        df['viral_load_proxy'] = df['fever'] + df['loss_of_smell'] + df['cough'] + (df['duration'] > 3).astype(int)

        # Drop totally healthy rows
        df = df[(df['target'] == 1) | (df['symptom_count'] > 0)].copy().reset_index(drop=True)

        # Applying Fuzzy Logic Arrays
        temp, pulse, sats, rr_vals, duration, load = df['temperature'].values, df['pulse'].values, df['sats'].values, df['rr'].values, df['duration'].values, df['symptom_count'].values
        
        df['temp_normal'] = fuzz.trapmf(temp, [30, 35, 37.2, 37.5])
        df['temp_high'] = fuzz.trimf(temp, [37.2, 38.5, 39.5])
        df['temp_severe'] = fuzz.trapmf(temp, [38.5, 39.5, 45, 45])
        df['pulse_normal'] = fuzz.trapmf(pulse, [40, 60, 90, 100])
        df['pulse_elevated'] = fuzz.trapmf(pulse, [90, 100, 200, 200])
        df['sats_fuzzy_normal'] = fuzz.trapmf(sats, [90, 95, 100, 100])
        df['sats_fuzzy_low'] = fuzz.trapmf(sats, [0,  0,  92,  95])
        df['rr_normal'] = fuzz.trapmf(rr_vals, [0, 10, 18, 20])
        df['rr_elevated'] = fuzz.trapmf(rr_vals, [18, 22, 40, 40])
        df['duration_acute'] = fuzz.trapmf(duration, [0, 0, 3, 7])
        df['duration_prolonged'] = fuzz.trapmf(duration, [4, 7, 30, 30])
        df['load_mild'] = fuzz.trapmf(load, [0, 0, 2, 4])
        df['load_moderate'] = fuzz.trimf(load, [2, 4, 6])
        df['load_severe'] = fuzz.trapmf(load, [4, 6, 15, 15])

        fuzzy_features = [
            'temp_normal', 'temp_high', 'temp_severe', 'pulse_normal', 'pulse_elevated',
            'sats_fuzzy_normal', 'sats_fuzzy_low', 'rr_normal', 'rr_elevated',
            'duration_acute', 'duration_prolonged', 'load_mild', 'load_moderate', 'load_severe'
        ]
        engineered_features = [
            'covid_signature', 'smell_only', 'taste_only', 'sob_with_anosmia', 'fatigue_with_anosmia', 'fever_no_uri',
            'multisystem_covid', 'exposed_anosmia', 'flu_syndrome', 'systemic_covid', 'rapid_onset_light', 'flu_excl_respiratory',
            'age_smell_interaction', 'respiratory_alarm', 'exposure_risk', 'weighted_covid_score', 'sats_concerning', 'sats_critical',
            'tachypnea', 'age_young', 'age_middle', 'age_senior', 'comorbidity_count', 'cough_severity', 'sob_severity',
            'flu_negative', 'flu_positive', 'flu_tested', 'duration_x_fever', 'duration_x_anosmia', 'duration_x_sob', 
            'age_x_fatigue', 'senior_prolonged', 'silent_hypoxia', 'viral_load_proxy'
        ]
        
        all_features = list(dict.fromkeys(binary_features + fuzzy_features + engineered_features))
        
        # Save artifacts
        df.to_csv(self.config.transformed_data_file, index=False)
        joblib.dump(all_features, os.path.join(self.config.root_dir, "base_features.pkl"))
        logging.info(f"Saved transformed data to {self.config.transformed_data_file}")
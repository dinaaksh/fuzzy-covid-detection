import joblib
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import json
import logging
from pathlib import Path
from covid_risk_detection.components.fuzzy_ontology import FuzzyOntologyEngine

class PredictionPipeline:
    def __init__(self):
        self.model_path = Path("artifacts/model_trainer/final_model.pkl")
        self.features_path = Path("artifacts/model_trainer/model_features.pkl")
        
        logging.info("Loading model and initializing Fuzzy Engine...")
        self.model = joblib.load(self.model_path)
        self.all_features = joblib.load(self.features_path)
        self.foe = FuzzyOntologyEngine()

    def predict(self, raw_data: dict) -> str:
        
        df = pd.DataFrame([raw_data]).fillna(0)
        for col in ['labored_respiration', 'rhonchi', 'wheezes', 'ctab']:
            if col not in df.columns: df[col] = 0
                
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
            df['diarrhea'] * 1.5 + df['fever'] * 1.0 + df['muscle_sore'] * 1.0 + df['cough_severity'] * 0.5 +
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

        fis_scores = self.foe.score_dataframe(df)
        df = pd.concat([df, fis_scores], axis=1)

        X_matrix = df[self.all_features].values
        probability = self.model.predict_proba(X_matrix)[0][1]
        
        return json.dumps({
            "status": "success",
            "covid_probability_percentage": round(probability * 100, 2)
        })
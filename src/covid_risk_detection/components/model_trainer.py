import os
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from covid_risk_detection.entity.config_entity import ModelTrainerConfig
from covid_risk_detection.components.fuzzy_ontology import FuzzyOntologyEngine

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, data_path: str, base_features_path: str):
        logging.info("Loading transformed data and base features...")
        df = pd.read_csv(data_path, low_memory=False)
        features = joblib.load(base_features_path)

        logging.info("Running Hierarchical Fuzzy Ontology Engine...")
        foe = FuzzyOntologyEngine()
        fis_scores = foe.score_dataframe(df)
        df = pd.concat([df, fis_scores], axis=1)

        fis_feature_names = list(fis_scores.columns)
        all_features = list(dict.fromkeys(features + fis_feature_names))
        
        X = df[all_features].values
        y = df['target'].values

        logging.info("Splitting data into train/val/test...")
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.176, random_state=42, stratify=y_dev)

        logging.info("Applying SMOTE-ENN to carve a clean clinical boundary...")
        smote_enn = SMOTEENN(random_state=42)
        X_train_clean, y_train_clean = smote_enn.fit_resample(X_train, y_train)

        logging.info(f"Original Training shape: {X_train.shape}, Balanced Training shape: {X_train_clean.shape}")

        neg_count = (y_dev == 0).sum()
        pos_count = (y_dev == 1).sum()
        actual_spw = round(neg_count / pos_count, 1)

        xgb_params = dict(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        model = XGBClassifier(
            **xgb_params,
            early_stopping_rounds=50,
            eval_metric="logloss"
        )

        logging.info("Training XGBoost Meta-Classifier on SMOTEENN Data...")
        model.fit(
            X_train_clean,
            y_train_clean,
            eval_set=[(X_val, y_val)],
            verbose=False
)
        
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else self.config.n_estimators
        
        logging.info("Calibrating model")
        calibrated = CalibratedClassifierCV(
            XGBClassifier(**{
                **xgb_params,
                'n_estimators': best_iter 
            }),
            method='isotonic', cv=5
        )
        calibrated.fit(X_dev, y_dev)

        logging.info("Saving calibrated model, features, and test datasets...")
        joblib.dump(calibrated, self.config.trained_model_path)
        joblib.dump(all_features, self.config.model_features_path)
        
        joblib.dump(X_test, os.path.join(self.config.root_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(self.config.root_dir, "y_test.pkl"))
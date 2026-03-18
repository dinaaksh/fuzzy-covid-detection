import pandas as pd
import joblib
import optuna
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, f1_score

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Import your incredibly fast HFIS Engine
from covid_risk_detection.components.fuzzy_ontology import FuzzyOntologyEngine

def load_and_prep_data():
    print("--- 1. Loading Transformed Data & Base Features ---")
    df = pd.read_csv("artifacts/data_transformation/transformed_data.csv", low_memory=False)
    features = joblib.load("artifacts/model_trainer/model_features.pkl")
    
    # Run the Hierarchical Fuzzy Engine ONCE
    foe = FuzzyOntologyEngine()
    fis_scores = foe.score_dataframe(df)
    df = pd.concat([df, fis_scores], axis=1)
    
    all_features = features + list(fis_scores.columns)
    X = df[all_features].values
    y = df['target'].values
    
    print("\n--- 2. Splitting Data ---")
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.176, random_state=42, stratify=y_dev)
    
    return X_train, X_val, y_train, y_val

# Load data into memory
X_train, X_val, y_train, y_val = load_and_prep_data()

# Calculate our baseline surgical weight
num_negatives = (y_train == 0).sum()
num_positives = (y_train == 1).sum()
base_weight = (num_negatives / num_positives) ** 0.5 

def objective(trial):
    # 1. Define the Hyperparameter Search Space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # XGBoost uses min_child_weight instead of LightGBM's min_child_samples
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        # Allow Optuna to slightly tweak our surgical weight to find the perfect balance
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_weight * 0.8, base_weight * 1.5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 2. Train the Model
    model = XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 3. Evaluate using our Custom Constraint (80% Precision Floor)
    y_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
    
    target_precision = 0.80
    target_recall = 0.80

    valid_indices = np.where(
        (precisions >= target_precision) & (recalls >= target_recall)
    )[0]
    
    if len(valid_indices) > 0:
        # Find the threshold that maximizes recall while keeping precision >= 80%
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        y_pred = (y_probs >= best_threshold).astype(int)
        
        # Return the F1 Score as the metric for Optuna to maximize
        return f1_score(y_val, y_pred)
    else:
        # Heavily punish the model if it fails to achieve 80% precision
        return 0.0

if __name__ == "__main__":
    print("\n--- 3. Launching Optuna Swarm (50 Trials) ---")
    # We want to MAXIMIZE the F1 score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("\n=================================================")
    print("🏆 BEST HYPERPARAMETERS FOUND:")
    print("=================================================")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=================================================")
    print(f"Best Validation F1 Score: {study.best_value:.4f}")
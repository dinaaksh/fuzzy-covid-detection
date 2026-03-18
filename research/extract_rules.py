import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV

def extract_vital_rules():
    print("--- Loading Transformed Data ---")
    df = pd.read_csv("artifacts/data_transformation/transformed_data.csv", low_memory=False)
    
    # 1. Create strict binary representations of our Fuzzy States based on clinical thresholds
    df['temp_high'] = (df['temperature'] > 37.5).astype(int)
    df['pulse_elevated'] = (df['pulse'] > 90).astype(int)
    df['sats_low'] = (df['sats'] < 95).astype(int)
    df['duration_prolonged'] = (df['duration'] > 4).astype(int)
    df['duration_acute'] = (df['duration'] <= 4).astype(int)
    
    # We include a few top symptoms just to see how they interact with the vitals
    vital_features = [
        'temp_high', 'pulse_elevated', 'sats_low', 
        'duration_prolonged', 'duration_acute',
        'loss_of_smell', 'cough', 'fatigue'
    ]
    
    X = df[vital_features].values
    y = df['target'].values
    
    print("--- Searching for Optimal Vital-Based Rules (Optimizing F1) ---")
    param_grid = {
        'max_depth': [3, 4, 5], # Keeping it shallow so the fuzzy rules remain readable
        'min_samples_leaf': [5, 10, 20]
    }
    
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
    gs.fit(X, y)
    
    best_tree = gs.best_estimator_
    print(f"Best Parameters Found by Data: {gs.best_params_}")
    
    print("\n--- NEW DATA-DRIVEN FUZZY VITAL RULES ---")
    tree_rules = export_text(best_tree, feature_names=vital_features)
    print(tree_rules)

if __name__ == "__main__":
    extract_vital_rules()
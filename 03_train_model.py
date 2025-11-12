import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report

# --- CONFIG ---
NEW_KAGGLE_FILE = 'team_traditional.csv' # The new, correct file
ROLLING_WINDOW = 10

# Features available in this new CSV (using the correct names from your list)
ADVANCED_FEATURES = [
    'FG%', '3P%', 'FT%', 'OREB', 
    'DREB', 'AST', 'TOV', 'STL', 'BLK'
]

# Our final feature list (using corrected names)
FINAL_FEATURES = [
    'DIFF_FG%', 'DIFF_3P%', 'DIFF_FT%', 'DIFF_OREB',
    'DIFF_DREB', 'DIFF_AST', 'DIFF_TOV', 'DIFF_STL', 'DIFF_BLK',
    'DIFF_DAYS_REST', 'DIFF_IS_BACK_TO_BACK'
]
TARGET = 'HOME_TEAM_WON'

def create_features_from_new_kaggle(file_name):
    """
    Loads the new 'team_traditional.csv' and engineers features.
    This replaces 02_build_training_dataset.py
    """
    print(f"Loading dataset: {file_name}...")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: Did not find {file_name} in your project folder.")
        print("Please download it from https://www.kaggle.com/datasets/szymonjwiak/nba-traditional")
        return None

    print("Data loaded. Cleaning and engineering features...")
    
    # --- 1. Clean Data ---
    # Use the correct column names from your CSV
    df['GAME_DATE'] = pd.to_datetime(df['date'])
    df['TEAM_ID'] = df['teamid']
    df['GAME_ID'] = df['gameid']
    
    # Filter for the modern seasons we want to train on
    df = df[df['GAME_DATE'] >= '2017-01-01'].copy()
    
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE']).copy()
    
    # --- 2. Engineer Rolling Features ---
    print("Calculating rolling averages for all teams...")
    features_to_roll = ADVANCED_FEATURES
    for factor in features_to_roll:
        if factor in df.columns:
            df[f'ROLL_{factor}'] = df.groupby('TEAM_ID')[factor].transform(
                lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=5).mean()
            )
        else:
            print(f"Warning: Expected feature '{factor}' not in CSV. Skipping.")

    # --- 3. Engineer Situational Features ---
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(0) - 1
    df['DAYS_REST'] = df['DAYS_REST'].clip(lower=0)
    # Fix typo: DAYS_TEST -> DAYS_REST
    df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int)

    # Drop rows where we don't have enough rolling data
    df = df.dropna(subset=[f'ROLL_{ADVANCED_FEATURES[0]}'])

    # --- 4. Merge Home and Away Teams ---
    print("Merging Home and Away team data...")
    
    # --- THIS IS THE FIX ---
    # A row is the home team if the 'team' (e.g., 'ATL') matches the 'home' column (e.g., 'ATL')
    df['IS_HOME'] = (df['team'] == df['home']).astype(int)
    # --- END FIX ---
    
    home_df = df[df['IS_HOME'] == 1].add_prefix('HOME_')
    away_df = df[df['IS_HOME'] == 0].add_prefix('AWAY_')

    # Merge on GAME_ID
    final_df = pd.merge(home_df, away_df, left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID', how='inner')

    if final_df.empty:
        print("ERROR: Failed to merge home and away teams. Check data.")
        return None

    # --- 5. Create Final Differential Features ---
    print("Creating differential features...")
    for factor in ADVANCED_FEATURES:
        home_col = f'HOME_ROLL_{factor}'
        away_col = f'AWAY_ROLL_{factor}'
        if home_col in final_df.columns and away_col in final_df.columns:
            final_df[f'DIFF_{factor}'] = final_df[home_col] - final_df[away_col]
    
    if 'HOME_DAYS_REST' in final_df.columns and 'AWAY_DAYS_REST' in final_df.columns:
        final_df['DIFF_DAYS_REST'] = final_df['HOME_DAYS_REST'] - final_df['AWAY_DAYS_REST']
    
    if 'HOME_IS_BACK_TO_BACK' in final_df.columns and 'AWAY_IS_BACK_TO_BACK' in final_df.columns:
        final_df['DIFF_IS_BACK_TO_BACK'] = final_df['HOME_IS_BACK_TO_BACK'] - final_df['AWAY_IS_BACK_TO_BACK']
    
    # Define the target variable (using 'win' column)
    final_df['HOME_TEAM_WON'] = (final_df['HOME_win'] == True).astype(int)

    # Filter only the columns we need
    final_df = final_df.dropna(subset=FINAL_FEATURES + [TARGET])
    
    if final_df.empty:
        print("ERROR: No data left after creating features. Halting.")
        return None

    print(f"Feature engineering complete. Final dataset shape: {final_df.shape}")
    return final_df[FINAL_FEATURES], final_df[TARGET]

def train_model():
    """Loads dataset, trains, and evaluates an XGBoost model."""
    
    # Get X (features) and y (target) from our new function
    X, y = create_features_from_new_kaggle(NEW_KAGGLE_FILE)
    
    if X is None or y is None:
        print("Failed to create features. Exiting training.")
        return

    # --- CRITICAL: Time Series Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Starting time-series cross-validation...")
    # Get the last split for evaluation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print("Training final model on all data...")
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        random_state=42
    )
    
    final_model.fit(X, y) # Train on the entire historical dataset
    
    # --- Evaluate on the last fold (our most recent data) ---
    print("\n--- Model Evaluation (on most recent test fold) ---")
    y_pred_test = final_model.predict(X_test)
    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_test)
    brier_score = brier_score_loss(y_test, y_prob_test)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Brier Score (Calibration): {brier_score:.4f} (Lower is better)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Save the model
    joblib.dump(final_model, 'nba_model.pkl')
    print("\nModel saved to 'nba_model.pkl'")

if __name__ == "__main__":
    train_model()
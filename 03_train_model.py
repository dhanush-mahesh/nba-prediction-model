import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report

# define final list of features the model will be trained on
FINAL_FEATURES = [
    'DIFF_E_OFF_RATING', 'DIFF_E_DEF_RATING', 'DIFF_E_NET_RATING', 'DIFF_E_PACE',
    'DIFF_E_TM_TOV_PCT', 'DIFF_E_OREB_PCT', 'DIFF_E_AST_RATIO', 'DIFF_E_REB_PCT',
    'DIFF_INJURY_IMPACT', 'DIFF_DAYS_REST', 'DIFF_IS_BACK_TO_BACK'
]
TARGET = 'HOME_TEAM_WON'

def train_model():
    """Loads dataset, trains, and evaluates an XGBoost model."""
    print("Loading dataset 'nba_model_dataset.csv'...")
    try:
        df = pd.read_csv('nba_model_dataset.csv')
    except FileNotFoundError:
        print("ERROR: 'nba_model_dataset.csv' not found.")
        print("Please run '02_build_training_dataset.py' first.")
        return

    df = df.dropna(subset=FINAL_FEATURES + [TARGET])
    
    X = df[FINAL_FEATURES]
    y = df[TARGET]

    # time series cross-validation
    # we must train on the past to predict the future.
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Starting time-series cross-validation...")
    # get the last split for evaluation
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
    
    final_model.fit(X, y) # train on the entire historical dataset
    
    # evaluate on the last fold (our most recent data)
    print("\n--- Model Evaluation (on most recent test fold) ---")
    y_pred_test = final_model.predict(X_test)
    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_test)
    brier_score = brier_score_loss(y_test, y_prob_test)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Brier Score (Calibration): {brier_score:.4f} (Lower is better)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # save the model
    joblib.dump(final_model, 'nba_model.pkl')
    print("\nModel saved to 'nba_model.pkl'")

if __name__ == "__main__":
    train_model()
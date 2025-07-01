# train_model.py

import pandas as pd
import os
import joblib  # ✅ better for compressed saving

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE  # ✅ handles class imbalance

def train_and_save_model(data_path="ml_model/data/tech_jobs.csv", model_path="ml_model/final_ensemble_model_compressed.pkl"):
    df = pd.read_csv(data_path)
    X = df[['Rating', 'Salary']]
    y = df['Job Roles']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
    xgb = XGBClassifier(n_estimators=50, max_depth=5, eval_metric='mlogloss', use_label_encoder=False, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
        voting='soft'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', voting_clf)
    ])

    pipeline.fit(X_resampled, y_resampled)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pipeline.predict(X_test)))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)  # ✅ smaller file size
    print(f"\n✅ Compressed model saved to '{model_path}'")

if __name__ == "__main__":
    train_and_save_model()

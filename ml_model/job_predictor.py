# ml_model/job_predictor.py
import numpy as np
import joblib  # Use joblib instead of pickle
import os

# Define model path
model_path = "ml_model/final_ensemble_model_compressed.pkl"

# Load the VotingClassifier pipeline (already includes scaler and classifier)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Please run train_model.py.")

model_pipeline = joblib.load(model_path)

def predict_jobs(expected_salary, rating=3.0, top_n=4):
    # Prepare input with [Rating, Salary] as expected by the model
    X_input = np.array([[rating, expected_salary]])

    # Predict class probabilities
    probabilities = model_pipeline.predict_proba(X_input)[0]

    # Get indices of top N job roles
    top_indices = probabilities.argsort()[::-1][:top_n]

    # Get class labels from the model
    job_roles = model_pipeline.named_steps['classifier'].classes_

    # Return top N job roles (normalized to uppercase for matching JSON keys)
    return [job_roles[i].upper() for i in top_indices]

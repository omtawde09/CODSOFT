import joblib
import pandas as pd
import numpy as np
import os

# ========== CONFIG ==========
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
SCALER_PATH = "models/scaler.joblib"  # optional


# ========== LOAD SAVED ARTIFACTS ==========
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        raise FileNotFoundError("❌ Model or preprocessing files not found. Run the training script first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("[INFO] Model and preprocessing artifacts loaded successfully.")
    return model, preprocessor


# ========== COLLECT USER INPUT ==========
def get_user_input():
    print("\n🔹 Please enter the following customer details:")
    data = {
        "CreditScore": float(input("Credit Score (e.g. 600): ")),
        "Geography": input("Geography (France / Spain / Germany): ").capitalize(),
        "Gender": input("Gender (Male / Female): ").capitalize(),
        "Age": int(input("Age: ")),
        "Tenure": int(input("Tenure (years with bank): ")),
        "Balance": float(input("Account Balance: ")),
        "NumOfProducts": int(input("Number of Products (1-4): ")),
        "HasCrCard": int(input("Has Credit Card (1 = Yes, 0 = No): ")),
        "IsActiveMember": int(input("Is Active Member (1 = Yes, 0 = No): ")),
        "EstimatedSalary": float(input("Estimated Salary: "))
    }
    return pd.DataFrame([data])


# ========== MAKE PREDICTION ==========
def predict_churn(model, preprocessor, user_df):
    # Apply full preprocessing (handles both categorical and numeric)
    X_processed = preprocessor.transform(user_df)

    # Predict probability and class
    prob = model.predict_proba(X_processed)[0][1]
    prediction = model.predict(X_processed)[0]

    print("\n🔍 Prediction Results:")
    print("----------------------")
    print(f"Churn Probability: {prob:.2f}")
    if prediction == 1:
        print("🚨 The customer is likely to CHURN.")
    else:
        print("✅ The customer is likely to STAY.")
    print("----------------------\n")


# ========== MAIN ==========
if __name__ == "__main__":
    model, preprocessor = load_artifacts()
    user_df = get_user_input()
    predict_churn(model, preprocessor, user_df)

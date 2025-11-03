import joblib
import numpy as np
import os

# Load the trained model
model_path = os.path.join("models", "best_model.joblib")
model = joblib.load(model_path)
print(f"Loaded model from: {model_path}\n")

print("Interactive mode. Type 'exit' to quit.")

while True:
    text = input("Enter SMS message (or 'exit'): ").strip()
    if text.lower() == "exit":
        break
    if not text:
        continue

    # Predict label (0 = HAM, 1 = SPAM)
    label = model.predict([text])[0]

    # Try getting decision_function or predict_proba based on model type
    try:
        score = model.decision_function([text])[0]
        # Convert raw score into probability-like confidence
        confidence = 100 - (100 / (1 + np.exp(abs(score))))
    except Exception:
        proba = model.predict_proba([text])[0]
        confidence = np.max(proba) * 100

    # Interpret prediction
    prediction = "SPAM" if label == 1 else "HAM"

    print(f"Prediction: {prediction:<5} | Confidence: {confidence:.1f}%")

    # Warn on borderline predictions
    if confidence < 70:
        print("⚠️  Low confidence — this message is borderline. Consider manual review.")
    elif confidence > 90:
        print("✅ High confidence — model is very certain about this prediction.")

    print()  # Blank line for readability

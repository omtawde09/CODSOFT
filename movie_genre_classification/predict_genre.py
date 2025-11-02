from joblib import load

# Load your saved model
loaded_obj = load("outputs/best_pipeline.pkl")

# Handle both tuple and pipeline formats
if isinstance(loaded_obj, tuple):
    vectorizer, model = loaded_obj
else:
    model = loaded_obj
    vectorizer = None

print("\n🎬 Movie Genre Predictor")
print("Type a movie plot summary and press Enter.")
print("Type 'exit' to quit.\n")

while True:
    plot = input("Enter movie plot: ").strip()
    if plot.lower() in {"exit", "quit"}:
        print("Goodbye! 👋")
        break

    if not plot:
        print("⚠️ Please enter some text.\n")
        continue

    # Transform and predict
    if vectorizer:
        X = vectorizer.transform([plot])
        prediction = model.predict(X)
    else:
        prediction = model.predict([plot])

    print(f"🎯 Predicted Genre: {prediction[0]}\n")

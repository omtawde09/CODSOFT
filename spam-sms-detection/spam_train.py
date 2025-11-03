import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# ==============================
# 1. Load and preprocess dataset
# ==============================

def load_dataset(path):
    """Load spam.csv and format it into clean dataframe"""
    df = pd.read_csv(path, encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(inplace=True)
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    return df


def clean_text(text):
    """Clean text: lowercase, remove URLs, digits, punctuation"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ======================================
# 2. Train, evaluate and select best model
# ======================================

def train_and_save_model(data_path, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess
    df = load_dataset(data_path)
    df['clean_message'] = df['message'].apply(clean_text)

    X = df['clean_message']
    y = df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔹 Training samples: {len(X_train)}")
    print(f"🔹 Testing samples: {len(X_test)}")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)

    # Define models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
        "Support Vector Machine": LinearSVC(max_iter=2000, class_weight='balanced')
    }

    results = {}
    best_model = None
    best_name = None
    best_f1 = 0

    # Train and evaluate each model
    for name, clf in models.items():
        print(f"\n==============================")
        print(f"Training Model: {name}")
        print("==============================")

        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"📊 Accuracy:  {acc:.4f}")
        print(f"🎯 Precision: {prec:.4f}")
        print(f"📈 Recall:    {rec:.4f}")
        print(f"🏆 F1 Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results[name] = f1

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_name = name

    # Display summary
    print("\n==============================")
    print("📋 Model Comparison (F1 Scores):")
    for model_name, f1 in results.items():
        print(f"{model_name}: {f1:.4f}")

    print(f"\n✅ Best Model: {best_name} with F1 Score = {best_f1:.4f}")

    # Save best model
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n💾 Best model saved successfully to: {model_path}")

    return best_model, best_name, best_f1


# ==============================
# 3. Run training
# ==============================
if __name__ == "__main__":
    print("🚀 Starting Spam SMS Detection model training...")
    train_and_save_model("data/spam.csv")

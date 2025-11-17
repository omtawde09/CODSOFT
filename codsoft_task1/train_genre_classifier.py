import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 📂 Paths
# ================================
DATA_DIR = "data/Genre Classification Dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_data.txt")
TEST_SOLUTION_FILE = os.path.join(DATA_DIR, "test_data_solution.txt")

os.makedirs("outputs", exist_ok=True)


# ================================
# 🧹 Loaders
# ================================
def load_custom_txt(path):
    """Load training data: id ::: title ::: genre ::: summary"""
    print(f"Loading: {path}")
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split(":::")]
            if len(parts) == 4:
                id_, title, genre, summary = parts
                rows.append((id_, title, genre.lower().strip(), summary))
    df = pd.DataFrame(rows, columns=["id", "title", "genre", "summary"])
    print(f"Rows loaded: {len(df)}")
    return df


def load_test_txt(path):
    """Load test data: id ::: title ::: summary"""
    print(f"Loading: {path}")
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split(":::")]
            if len(parts) == 3:
                id_, title, summary = parts
                rows.append((id_, title, summary))
    df = pd.DataFrame(rows, columns=["id", "title", "summary"])
    print(f"Rows loaded: {len(df)}")
    return df


def load_test_solution_txt(path):
    """Load test solution: id ::: title ::: genre ::: summary"""
    print(f"Loading: {path}")
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split(":::")]
            if len(parts) == 4:
                id_, title, genre, summary = parts
                rows.append((id_, title, genre.lower().strip(), summary))
    df = pd.DataFrame(rows, columns=["id", "title", "genre", "summary"])
    print(f"Rows loaded: {len(df)}")
    return df


# ================================
# ✨ Preprocessing
# ================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def combine_text(df):
    """Combine title + summary for better features"""
    return (df["title"].fillna("") + " " + df["summary"].fillna("")).apply(clean_text)


# ================================
# 🧠 Training Function
# ================================
def train_model(X_train, y_train, X_val, y_val, name, model):
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy ({name}): {acc:.4f}")
    print(classification_report(y_val, preds))
    return acc, model


# ================================
# 📊 Confusion Matrix
# ================================
def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ================================
# 🚀 Main
# ================================
def main():
    print(f"Loading training data from: {TRAIN_FILE}")
    df = load_custom_txt(TRAIN_FILE)
    print("Columns:", list(df.columns))
    print("Sample genres distribution:\n", df["genre"].value_counts().to_dict())

    df["summary"] = df["summary"].fillna("")
    df["combined"] = combine_text(df)
    df = df[df["combined"].str.strip() != ""]
    print(f"Dropped {len(df) - len(df)} rows with empty summaries after cleaning.")

    X_train, X_val, y_train, y_val = train_test_split(
        df["combined"], df["genre"], test_size=0.2, random_state=42, stratify=df["genre"]
    )
    print(f"Train / Val sizes: {len(X_train)} {len(X_val)}")

    # ================================
    # Vectorizer (Upgraded)
    # ================================
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # ================================
    # Models
    # ================================
    models = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(
            solver="saga", max_iter=3000, class_weight="balanced", n_jobs=-1
        ),
        "svc": LinearSVC(class_weight="balanced")
    }

    best_acc = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        acc, trained_model = train_model(X_train_vec, y_train, X_val_vec, y_val, name, model)
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_name = name

    print(f"\nBest model on validation set: {best_name} (acc={best_acc:.4f})")

    # Save pipeline
    joblib.dump((vectorizer, best_model), "outputs/best_pipeline.pkl")
    print("✅ Saved best pipeline to: outputs/best_pipeline.pkl")

    with open("outputs/report.txt", "w") as f:
        f.write(f"Best model: {best_name}\nValidation accuracy: {best_acc:.4f}\n")

    # Confusion Matrix
    val_preds = best_model.predict(X_val_vec)
    plot_confusion_matrix(y_val, val_preds, labels=sorted(df["genre"].unique()),
                          title=f"Confusion Matrix ({best_name})",
                          path="outputs/confusion_matrix.png")
    print("🖼️  Confusion matrix saved.")

    # ================================
    # Test Evaluation
    # ================================
    print(f"Loading test set: {TEST_FILE}")
    test_df = load_test_txt(TEST_FILE)
    test_df["combined"] = combine_text(test_df)

    vectorizer, best_model = joblib.load("outputs/best_pipeline.pkl")
    test_vec = vectorizer.transform(test_df["combined"])
    test_preds = best_model.predict(test_vec)

    # Save predictions
    pred_path = "outputs/test_predictions.txt"
    with open(pred_path, "w", encoding="utf-8") as f:
        for i in range(len(test_df)):
            f.write(f"{test_df.iloc[i]['id']} ::: {test_df.iloc[i]['title']} ::: {test_preds[i]} ::: {test_df.iloc[i]['summary']}\n")
    print(f"✅ Saved test predictions to: {pred_path}")

    # Evaluate with ground truth
    print(f"Evaluating on test solution: {TEST_SOLUTION_FILE}")
    test_solution_df = load_test_solution_txt(TEST_SOLUTION_FILE)

    acc_test = accuracy_score(test_solution_df["genre"], test_preds)
    print(f"\n📊 Test Accuracy: {acc_test:.4f}")
    print(classification_report(test_solution_df["genre"], test_preds))

    with open("outputs/report.txt", "a") as f:
        f.write(f"\nTest Accuracy: {acc_test:.4f}\n")


if __name__ == "__main__":
    main()

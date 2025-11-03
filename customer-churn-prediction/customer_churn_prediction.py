import os
import argparse
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score
)
import joblib

# Optional: xgboost - handle if not installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
    return df


def quick_eda(df: pd.DataFrame, target: str = "Exited") -> None:
    """Print quick exploratory data summary (non-graphical)."""
    print("\n[EDA] — head()")
    print(df.head())
    print("\n[EDA] — info()")
    print(df.info())
    print("\n[EDA] — describe()")
    print(df.describe(include='all').T)
    if target in df.columns:
        print(f"\n[EDA] — Target distribution for '{target}':")
        print(df[target].value_counts())
        print(df[target].value_counts(normalize=True))


def preprocess(
    df: pd.DataFrame,
    target: str = "Exited"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, StandardScaler, pd.Index]:
    """
    Preprocess dataset:
     - Drop ID-like columns
     - Separate features and target
     - One-hot encode categorical columns (drop_first=False for safety)
     - Scale numeric columns (fit on training only)
     - Stratified train_test_split
    Returns: X_train, X_test, y_train, y_test, transformer, scaler, feature_names
    """
    df = df.copy()
    # Drop obvious non-predictive id columns if present
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    # Separate X and y
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\n[PREPROCESS] Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"[PREPROCESS] Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # if missing
        # Scaling will be applied after train/test split to avoid leakage
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop", verbose_feature_names_out=False)

    # First split (stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Fit transformer on training data, transform both
    preprocessor.fit(X_train_raw)
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Build feature names after transformation
    # For numeric, keep numeric names; for categorical, get feature names from OneHotEncoder
    transformed_feature_names = []
    # numeric columns preserved in same order
    transformed_feature_names.extend(numeric_cols)

    # extract onehot feature names
    if categorical_cols:
        # access OneHotEncoder object:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        # categories_ is a list, build names like col_val
        cat_feature_names = []
        for col, cats in zip(categorical_cols, ohe.categories_):
            cat_feature_names.extend([f"{col}__{str(cat)}" for cat in cats])
        transformed_feature_names.extend(cat_feature_names)

    # Scaling numeric columns only: build scaler fit on numeric portion of X_train_trans
    scaler = StandardScaler()
    if numeric_cols:
        # numeric portion is first len(numeric_cols) columns
        scaler.fit(X_train_trans[:, :len(numeric_cols)])
        X_train_trans[:, :len(numeric_cols)] = scaler.transform(X_train_trans[:, :len(numeric_cols)])
        X_test_trans[:, :len(numeric_cols)] = scaler.transform(X_test_trans[:, :len(numeric_cols)])

    print(f"[PREPROCESS] After transform: X_train shape = {X_train_trans.shape}, X_test shape = {X_test_trans.shape}")
    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor, scaler, pd.Index(transformed_feature_names)


def train_models(
    X_train: np.ndarray, y_train: np.ndarray, n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Train three baseline models:
     - Logistic Regression (with class_weight balanced)
     - RandomForestClassifier
     - XGBoostClassifier (if available)
    Returns dict of fitted models.
    """
    models = {}

    print("\n[TRAIN] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    print("[TRAIN] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=n_jobs)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    if XGB_AVAILABLE:
        print("[TRAIN] Training XGBoostClassifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs
        )
        xgb_clf.fit(X_train, y_train)
        models["XGBoost"] = xgb_clf
    else:
        print("[TRAIN] XGBoost not available; skipping XGBoost training. Install xgboost to enable this model.")

    return models


def evaluate_and_select(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: pd.Index,
    out_dir: str = "models"
) -> Tuple[str, Any]:
    """
    Evaluate each model, print metrics, plot ROC curves and feature importances.
    Returns best_model_name, best_model_object (by ROC-AUC).
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    best_auc = -1.0
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        print(f"\n[EVAL] Model: {name}")
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Some classifiers (rare) might have decision_function
            try:
                y_proba = model.decision_function(X_test)
                # scale decision_function to [0,1] style for ROC curve - roc_curve accepts scores as-is
            except Exception:
                # fallback to predictions
                y_proba = model.predict(X_test)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else roc_auc_score(y_test, (y_pred))
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        # Save metrics to disk
        with open(os.path.join(out_dir, f"{name}_metrics.txt"), "w") as fh:
            fh.write(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\nROC-AUC: {auc:.4f}\n")
            fh.write("Classification Report:\n")
            fh.write(classification_report(y_test, y_pred))

        # feature importance for tree-based models
        if name in ("RandomForest", "XGBoost"):
            try:
                importances = model.feature_importances_
                # take top 20
                idx_sorted = np.argsort(importances)[::-1]
                top_n = min(20, len(importances))
                top_idx = idx_sorted[:top_n]
                plt.figure(figsize=(8, top_n * 0.3 + 1))
                sns.barplot(x=importances[top_idx], y=feature_names[top_idx])
                plt.title(f"{name} - Top {top_n} Feature Importances")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{name}_feature_importances.png"))
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not compute feature importances for {name}: {e}")

        # pick best by AUC
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model

    # finalize ROC plot
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"))
    plt.close()

    print(f"\n[RESULT] Best model by ROC-AUC: {best_model_name} (AUC = {best_auc:.4f})")
    return best_model_name, best_model_obj


def save_artifacts(model_obj: Any, preprocessor: ColumnTransformer, scaler: StandardScaler, out_dir: str = "models"):
    """Save model and preprocessing artifacts to disk (joblib)."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "best_model.joblib")
    preprocessor_path = os.path.join(out_dir, "preprocessor.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    joblib.dump(model_obj, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SAVE] Saved model to {model_path}")
    print(f"[SAVE] Saved preprocessor to {preprocessor_path}")
    print(f"[SAVE] Saved scaler to {scaler_path}")


def main(args):
    # Load
    df = load_data(args.data_path)

    # Quick EDA
    quick_eda(df, target=args.target)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor, scaler, feature_names = preprocess(df, target=args.target)

    # Train
    models = train_models(X_train, y_train, n_jobs=args.n_jobs)

    # Evaluate & select
    best_name, best_model = evaluate_and_select(models, X_test, y_test, feature_names, out_dir=args.out_dir)

    # Save
    if best_model is not None:
        # Note: we save the preprocessor and scaler too so the pipeline for inference can reconstruct inputs
        save_artifacts(best_model, preprocessor, scaler, out_dir=args.out_dir)
    else:
        print("[WARN] No best model found; nothing saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prediction - All-in-one script")
    parser.add_argument("--data_path", type=str, default="data/Churn_Modelling.csv", help="Path to CSV data file")
    parser.add_argument("--out_dir", type=str, default="models", help="Directory to save models & outputs")
    parser.add_argument("--target", type=str, default="Exited", help="Name of target column")
    parser.add_argument("--n_jobs", type=int, default=-1, help="n_jobs for training (use -1 for all cores)")
    args = parser.parse_args()
    main(args)

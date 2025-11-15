import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


# -----------------------------
# Paths & Config
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV = DATA_DIR / "fraudTest.csv"
TARGET = "is_fraud"

# drop clear identifiers / leakage
DROP_COLS = [
    "first", "last", "street", "city", "state", "zip",
    "trans_num", "cc_num"
]

# raw columns present from your screenshot
TIMESTAMP_COL = "trans_date_trans_time"
DOB_COL = "dob"
LAT_COL, LON_COL = "lat", "long"
MERCH_LAT_COL, MERCH_LON_COL = "merch_lat", "merch_long"

# categorical columns we’ll one-hot encode
CATEGORICAL = ["merchant", "category", "gender", "job"]
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Utilities
# -----------------------------
def load_csv(path: Path, nrows: int = None) -> pd.DataFrame:
    """Load CSV; nrows optional for quick debug."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, nrows=nrows)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[DOB_COL] = pd.to_datetime(df[DOB_COL], errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COL]
    df["trans_hour"] = ts.dt.hour
    df["trans_day"] = ts.dt.day
    df["trans_month"] = ts.dt.month
    df["trans_dow"] = ts.dt.dayofweek
    return df


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_days = (df[TIMESTAMP_COL] - df[DOB_COL]).dt.days
    df["age"] = (age_days / 365.25).clip(lower=0)
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    p = np.pi / 180.0
    lat1, lon1, lat2, lon2 = lat1 * p, lon1 * p, lat2 * p, lon2 * p
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cust_merchant_dist_km"] = haversine_distance(
        df[LAT_COL].astype(float), df[LON_COL].astype(float),
        df[MERCH_LAT_COL].astype(float), df[MERCH_LON_COL].astype(float)
    )
    return df


def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)
    df = add_time_features(df)
    df = add_age(df)
    df = add_geo_distance(df)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep = [c for c in df.columns if c not in DROP_COLS]
    df = df[keep]
    df = engineer_all(df)
    return df


def build_preprocess(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # infer numeric after engineering; exclude target & categoricals
    numeric_cols = df.drop(columns=CATEGORICAL + [TARGET], errors="ignore").select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_cols = [c for c in CATEGORICAL if c in df.columns]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return pre, numeric_cols, categorical_cols


def build_train_parts(df: pd.DataFrame):
    df = basic_clean(df)
    pre, num_cols, cat_cols = build_preprocess(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return pre, X, y, num_cols, cat_cols


def sampler():
    return RandomUnderSampler(random_state=42)


def evaluate_binary(model, X_test, y_test) -> Dict:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)
    else:
        y_score = None
        roc = np.nan
        pr_auc = np.nan

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return {"roc_auc": roc, "pr_auc": pr_auc, "confusion_matrix": cm, "report": report}


def choose_best(results: Dict[str, Dict]) -> str:
    best_name, best_metric = None, -1
    for name, met in results.items():
        key = met["pr_auc"] if not np.isnan(met["pr_auc"]) else met["roc_auc"]
        if key > best_metric:
            best_metric, best_name = key, name
    return best_name


# -----------------------------
# Train / Evaluate / Save
# -----------------------------
def train_and_eval(nrows: int = None) -> Tuple[Dict, Path]:
    print("Loading data...")
    train_df = load_csv(TRAIN_CSV, nrows=nrows)
    test_df = load_csv(TEST_CSV, nrows=nrows)

    print("Building matrices...")
    pre, X_train, y_train, _, _ = build_train_parts(train_df)
    _pre_t, X_test, y_test, _, _ = build_train_parts(test_df)  # pipelines will be refit per model

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "dtree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "rf": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
    }

    results = {}
    trained = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTraining & evaluating models...")
    for name, clf in models.items():
        pipe = ImbPipeline([
            ("pre", pre),
            ("rs", sampler()),
            ("clf", clf),
        ])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        print(f"{name} | CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipe.fit(X_train, y_train)
        metrics = evaluate_binary(pipe, X_test, y_test)

        print(f"\n=== {name.upper()} TEST ===")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
        print("Confusion Matrix:\n", metrics["confusion_matrix"])
        print("Classification Report:\n", metrics["report"])

        results[name] = metrics
        trained[name] = pipe

    best_name = choose_best(results)
    best_model = trained[best_name]
    out_path = MODEL_DIR / f"best_model_{best_name}.joblib"
    joblib.dump(best_model, out_path)
    print(f"\nSaved best model → {out_path}")
    return results, out_path


# -----------------------------
# Predict (CLI)
# -----------------------------
def predict_rows(rows_path: Path, model_path: Path = None, threshold: float = 0.5):
    """
    rows_path: JSON or CSV with columns like the original data (same structure as train),
               WITHOUT the target. You can paste a single row too.
    """
    if model_path is None:
        models = sorted(MODEL_DIR.glob("best_model_*.joblib"))
        if not models:
            raise FileNotFoundError("No saved model found in models/. Train first.")
        model_path = models[-1]

    model = joblib.load(model_path)

    if rows_path.suffix.lower() == ".json":
        rows = json.loads(rows_path.read_text())
        df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame([rows])
    elif rows_path.suffix.lower() == ".csv":
        df = pd.read_csv(rows_path)
    else:
        raise ValueError("rows_path must be .json or .csv")

    # Model pipeline includes preprocessing, so we can call directly
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["fraud_proba"] = proba
    out["pred_is_fraud"] = pred
    return out


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="All-in-one Credit Card Fraud Detection pipeline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train models and save the best")
    p_train.add_argument("--nrows", type=int, default=None,
                         help="Load only first N rows for quick runs")

    p_pred = sub.add_parser("predict", help="Predict on new rows (CSV or JSON)")
    p_pred.add_argument("--rows", type=str, required=True,
                        help="Path to CSV or JSON (no target column)")
    p_pred.add_argument("--model", type=str, default=None,
                        help="Path to a saved .joblib model (optional)")
    p_pred.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for fraud")

    args = parser.parse_args()

    if args.cmd == "train":
        train_and_eval(nrows=args.nrows)
    elif args.cmd == "predict":
        rows_path = Path(args.rows)
        model_path = Path(args.model) if args.model else None
        out = predict_rows(rows_path, model_path, threshold=args.threshold)
        # print compact output
        with pd.option_context("display.max_columns", None):
            print(out.head(20).to_string(index=False))

    # ====================================================
    # ADDITIONAL TOOLS: Threshold Tuning & Feature Importance
    # ====================================================

    from sklearn.metrics import precision_recall_fscore_support

    def evaluate_at_threshold(model, X, y, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6)):
        """
        Compute precision, recall and F1-score at different decision thresholds.
        Use this to pick a good probability cutoff based on business needs.
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must support predict_proba().")

        proba = model.predict_proba(X)[:, 1]
        scores = []

        for t in thresholds:
            preds = (proba >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, preds, average="binary", zero_division=0
            )
            scores.append({
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        return scores

    def rf_feature_importance(pipeline, X_sample):
        """
        Extract feature importances from trained RandomForest inside the pipeline.
        Only works with the RF model.
        """
        try:
            pre = pipeline.named_steps["pre"]
            rf = pipeline.named_steps["clf"]
        except KeyError:
            raise ValueError("Pipeline must include steps named 'pre' and 'clf'.")

        # Get numeric feature names
        num_features = pre.transformers_[0][2]  # list of raw numeric feature names

        # Get transformed categorical feature names
        ohe = pre.transformers_[1][1].named_steps["ohe"]
        cat_cols = pre.transformers_[1][2]
        cat_features = ohe.get_feature_names_out(cat_cols).tolist()

        # Combine
        feature_names = list(num_features) + cat_features
        importances = rf.feature_importances_

        # Return sorted list of (feature_name, importance)
        return sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )


if __name__ == "__main__":
    main()

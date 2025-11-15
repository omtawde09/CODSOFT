# app.py
# Streamlit app to test the trained Credit Card Fraud Detection pipeline
# Run: streamlit run app.py

from pathlib import Path
from datetime import datetime
from dateutil import parser

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model_rf.joblib")

# -------------------------
# Feature engineering funcs
# -------------------------
TIMESTAMP_COL = "trans_date_trans_time"
DOB_COL = "dob"
LAT_COL, LON_COL = "lat", "long"
MERCH_LAT_COL, MERCH_LON_COL = "merch_lat", "merch_long"

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
    # expects degrees -> returns km
    R = 6371.0
    p = np.pi / 180.0
    lat1, lon1, lat2, lon2 = lat1 * p, lon1 * p, lat2 * p, lon2 * p
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # coerce to float safely
    df["cust_merchant_dist_km"] = haversine_distance(
        df[LAT_COL].astype(float),
        df[LON_COL].astype(float),
        df[MERCH_LAT_COL].astype(float),
        df[MERCH_LON_COL].astype(float),
    )
    return df

def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)
    df = add_time_features(df)
    df = add_age(df)
    df = add_geo_distance(df)
    return df

# -------------------------
# Helpers
# -------------------------
def to_unix(ts_str: str) -> int:
    """Convert 'YYYY-MM-DD HH:MM:SS' to unix seconds."""
    return int(parser.parse(ts_str).timestamp())

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}\nTrain first to create it.")
        st.stop()
    return joblib.load(MODEL_PATH)

def rf_feature_importance(pipeline):
    """
    Extract top feature importances from RandomForest inside the pipeline.
    Works only if the final estimator has `feature_importances_`.
    """
    try:
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
    except Exception:
        return None

    if not hasattr(clf, "feature_importances_"):
        return None

    # numeric columns list (raw names)
    num_cols = pre.transformers_[0][2]
    # categorical encoder & names
    ohe = pre.transformers_[1][1].named_steps["ohe"]
    cat_bases = pre.transformers_[1][2]
    cat_cols = ohe.get_feature_names_out(cat_bases).tolist()

    feature_names = list(num_cols) + cat_cols
    importances = clf.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detector")
st.caption("Enter a transaction and get the fraud probability. Uses your trained Random Forest pipeline.")

pipe = load_model()

with st.form("fraud_form", clear_on_submit=False):
    st.subheader("Transaction")
    trans_date = st.text_input(
        "Transaction Datetime (YYYY-MM-DD HH:MM:SS)",
        value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        help="Example: 2019-01-03 08:15:00",
    )
    merchant = st.text_input("Merchant (free text)", value="fraud_Toms")
    category = st.text_input("Category (free text)", value="misc_net")
    amt = st.number_input("Amount", min_value=0.0, value=229.55, step=0.01)

    st.subheader("Customer")
    gender = st.selectbox("Gender", options=["M", "F"], index=1)
    dob = st.text_input("Date of Birth (YYYY-MM-DD)", value="1985-07-19")
    job = st.text_input("Job (free text)", value="School teacher")

    st.subheader("Geolocation")
    lat = st.number_input("Customer Latitude", value=33.819000, format="%.6f")
    lon = st.number_input("Customer Longitude", value=-84.433000, format="%.6f")
    city_pop = st.number_input("City Population", min_value=0, value=498715, step=100)

    merch_lat = st.number_input("Merchant Latitude", value=33.912000, format="%.6f")
    merch_lon = st.number_input("Merchant Longitude", value=-84.212000, format="%.6f")

    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)
    submit = st.form_submit_button("Predict Fraud")

# Build raw row (matches training raw schema)
def build_raw_row():
    return pd.DataFrame([{
        # Keep Unnamed: 0 if pipeline expects it (placeholder)
        "Unnamed: 0": 0,

        "trans_date_trans_time": trans_date,
        "unix_time": to_unix(trans_date),

        "cc_num": 1234567890123456,
        "first": "John",
        "last": "Doe",
        "street": "10 Main St",
        "city": "Boulder",
        "state": "CO",
        "zip": 80301,
        "trans_num": "xyz123",

        "merchant": merchant,
        "category": category,
        "amt": float(amt),
        "gender": gender,
        "lat": float(lat),
        "long": float(lon),
        "city_pop": int(city_pop),
        "job": job,
        "dob": dob,
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_lon),
    }])

if submit:
    try:
        df = build_raw_row()

        # IMPORTANT: engineer the columns the pipeline expects BEFORE prediction
        df = engineer_all(df)

        # If any columns the transformer expects are still missing, add safe defaults:
        # (this is defensive — normally engineer_all supplies the engineered columns)
        for col in ["trans_hour", "trans_day", "trans_month", "trans_dow", "age", "cust_merchant_dist_km"]:
            if col not in df.columns:
                df[col] = 0

        proba = pipe.predict_proba(df)[:, 1][0]
        pred = proba >= threshold

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Fraud Probability", f"{proba:.3f}")
        st.write(f"Predicted Fraud (threshold = {threshold:.2f}): **{bool(pred)}**")

        with st.expander("Show request payload"):
            st.json(df.iloc[0].to_dict())

    except Exception as e:
        st.error(f"Error while predicting: {e}")

# Sidebar: feature importances
with st.sidebar:
    st.header("🔎 Model Info")
    st.write(f"Loaded model: `{MODEL_PATH.name}`")
    imps = rf_feature_importance(pipe)
    if imps is not None:
        st.subheader("Top Features")
        for name, val in imps[:15]:
            st.write(f"• **{name}** — {val:.4f}")
    else:
        st.caption("Feature importances unavailable for this estimator.")

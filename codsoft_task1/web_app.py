import streamlit as st
from joblib import load
import os
import numpy as np

MODEL_PATH = "outputs/best_pipeline.pkl"

# Caching model load so app is snappy
@st.cache(allow_output_mutation=True)
def load_saved_model(path):
    obj = load(path)
    # Determine format
    if isinstance(obj, tuple):
        # saved as (vectorizer, model)
        vectorizer, model = obj
        return {"type": "tuple", "vectorizer": vectorizer, "model": model}
    else:
        # It might be a sklearn Pipeline or a single model
        return {"type": "single", "pipeline": obj}

model_store = load_saved_model(MODEL_PATH)

st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
st.title("🎬 Movie Genre Classifier")
st.write(
    "Type (or paste) a movie plot summary and click **Predict**. "
    "The app predicts the most likely genre using your trained model."
)

st.markdown("---")

with st.expander("Model info", expanded=False):
    st.write("Loaded model file:", MODEL_PATH)
    st.write("Detected format:")
    if model_store["type"] == "tuple":
        st.write("- Saved format: (vectorizer, model)")
        st.write("- Model class:", type(model_store["model"]).__name__)
        st.write("- Vectorizer class:", type(model_store["vectorizer"]).__name__)
    else:
        st.write("- Saved format: pipeline or single model object")
        st.write("- Pipeline/model class:", type(model_store["pipeline"]).__name__)

st.markdown("### Enter movie plot summary")
input_text = st.text_area("Plot summary", height=180, placeholder="e.g. A detective investigates ...")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Predict"):
        text = input_text.strip()
        if not text:
            st.warning("Please enter a movie plot summary first.")
        else:
            try:
                if model_store["type"] == "tuple":
                    vec = model_store["vectorizer"].transform([text])
                    model = model_store["model"]
                    pred = model.predict(vec)
                    # check for probabilities
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                        classes = model.classes_
                        top_idx = np.argsort(probs)[::-1][:5]
                        st.success(f"🎯 Predicted genre: **{pred[0]}**")
                        st.write("Top probabilities:")
                        for i in top_idx:
                            st.write(f"- {classes[i]} : {probs[i]:.3f}")
                    else:
                        st.success(f"🎯 Predicted genre: **{pred[0]}**")
                        st.info("Model does not provide probabilities (no `predict_proba`).")
                else:
                    pipeline = model_store["pipeline"]
                    pred = pipeline.predict([text])
                    st.success(f"🎯 Predicted genre: **{pred[0]}**")
                    # try probabilities from pipeline (if available)
                    if hasattr(pipeline, "predict_proba"):
                        probs = pipeline.predict_proba([text])[0]
                        classes = pipeline.classes_
                        top_idx = np.argsort(probs)[::-1][:5]
                        st.write("Top probabilities:")
                        for i in top_idx:
                            st.write(f"- {classes[i]} : {probs[i]:.3f}")
                    else:
                        st.info("Model/pipeline does not provide probabilities.")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

with col2:
    st.markdown("#### Sample plots (click to copy)")
    samples = [
        "A detective wakes up to find his wife missing. He uncovers a dark government conspiracy.",
        "A group of children set out on a wild adventure to find a hidden treasure.",
        "A documentary about the life of a famous musician and their rise to stardom.",
        "A lonely astronaut drifts far from Earth and must struggle to survive."
    ]
    for s in samples:
        if st.button(s, key=s[:20]):
            # set clipboard isn't available, but populate text area
            st.experimental_set_query_params()  # no-op but keeps session interactive
            # update input_text by rerunning with query param isn't straightforward;
            # Instead, tell the user to paste sample manually (Streamlit limitation)
            st.write("Sample plot (copy and paste into the plot box):")
            st.code(s)

st.markdown("---")
st.write("🔧 Tip: If your model was saved as a `(vectorizer, model)` tuple, this app will transform input with the vectorizer before predicting. If it was saved as a `Pipeline`, the pipeline handles preprocessing internally.")

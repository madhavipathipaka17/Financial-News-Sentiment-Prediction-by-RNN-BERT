import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="centered")

LABEL_NAMES = ["Bearish", "Bullish", "Neutral"]
ID2LABEL = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
COLORS = {"Bearish": "#d62728", "Bullish": "#2ca02c", "Neutral": "#7f7f7f"}

# ----------------------------------------------------------------------
# Text cleaning (must match preprocessing used during training)
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_sample_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Expect a "label" column with 0/1/2 encoding; adjust if yours differs.
    if "label" in df.columns:
        df["label_name"] = df["label"].map(ID2LABEL)
    return df

def predict_sentiment(text: str, tokenizer, model):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

# ----------------------------------------------------------------------
# Sidebar: model + data paths
# ----------------------------------------------------------------------
st.sidebar.header("Settings")
model_path = st.sidebar.text_input(
    "Path to trained model folder",
    value="./bert-financial-sentiment-final",
    help="Folder saved via trainer.save_model(...) — either your BERT model or a "
         "HF-compatible baseline checkpoint.",
)
data_path = st.sidebar.text_input(
    "Path to validation CSV (for class distribution plot)",
    value="sent_valid.csv",
)

# ----------------------------------------------------------------------
# Main title
# ----------------------------------------------------------------------
st.title("📈 Financial News Sentiment Dashboard")
st.write(
    "Enter a finance-related tweet or headline below to see the predicted "
    "sentiment (Bearish / Bullish / Neutral) along with class probabilities."
)

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
try:
    tokenizer, model = load_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model from '{model_path}'. Error: {e}")

# ----------------------------------------------------------------------
# 2. Text box + prediction
# ----------------------------------------------------------------------
st.subheader("Try it out")
user_text = st.text_area(
    "Tweet / headline text",
    placeholder="e.g. Company X beats earnings expectations, stock surges 12%",
    height=100,
)

if st.button("Predict sentiment", disabled=not model_loaded):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, probs = predict_sentiment(user_text, tokenizer, model)

        color = COLORS.get(label, "#1f77b4")
        st.markdown(
            f"### Predicted sentiment: <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True,
        )

        # 3a. Bar chart of class probabilities
        fig, ax = plt.subplots(figsize=(5, 3.5))
        names = list(probs.keys())
        values = [probs[n] for n in names]
        bar_colors = [COLORS[n] for n in names]
        ax.bar(names, values, color=bar_colors)
        ax.set_ylim(0, 1.15)  # extra headroom so % labels never hit the title
        ax.set_ylabel("Probability")
        ax.set_title("Predicted Class Probabilities", pad=14)
        for i, v in enumerate(values):
            ax.text(i, v + 0.03, f"{v:.2%}", ha="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

# ----------------------------------------------------------------------
# 3b. Overall class distribution from validation/sample data
# ----------------------------------------------------------------------
st.subheader("Validation set class distribution")
try:
    df = load_sample_data(data_path)
    if "label_name" in df.columns:
        dist = df["label_name"].value_counts().reindex(LABEL_NAMES).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.bar(dist.index, dist.values, color=[COLORS[n] for n in dist.index])
        ax2.set_ylim(0, max(dist.values) * 1.15)  # headroom for count labels
        ax2.set_ylabel("Count")
        ax2.set_title("Class Distribution (Validation Data)", pad=14)
        for i, v in enumerate(dist.values):
            ax2.text(i, v + max(dist.values) * 0.03, int(v), ha="center", fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)

        with st.expander("Preview data"):
            st.dataframe(df.head(20))
    else:
        st.info("CSV loaded but no 'label' column found — skipping distribution plot.")
except Exception as e:
    st.info(f"Could not load validation data from '{data_path}': {e}")

st.caption(
    "Tip: run with `streamlit run app.py`. Point the sidebar paths to your "
    "saved model folder and validation CSV."
)

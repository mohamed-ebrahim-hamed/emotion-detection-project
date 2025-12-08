import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "emotion_model"

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device

def predict(text, threshold=0.3):
    tokenizer, model, device = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = [
        (LABELS[i], float(probs[i]))
        for i in range(len(LABELS))
        if probs[i] >= threshold
    ]

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ===================== UI ======================
st.set_page_config(page_title="Emotion Detection", page_icon="🧠")
st.title("🧠 Emotion Detection App")
st.write("Multi-label Emotion Detection using DistilBERT")

text = st.text_area("Enter text (English):")

threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            predictions = predict(text, threshold)

        if not predictions:
            st.info("No emotion passed the threshold.")
        else:
            st.subheader("Detected Emotions:")
            for label, score in predictions:
                st.write(f"**{label}** → {score:.3f}")

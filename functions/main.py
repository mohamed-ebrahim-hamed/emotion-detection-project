import functions_framework
import tempfile
import zipfile
import json
import torch

from google.cloud import storage
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# ------------------------------
#   CONFIG
# ------------------------------

BUCKET_NAME = "YOUR-BUCKET-NAME"       # ← غيّر الاسم هنا فقط
MODEL_FILE  = "emotion_text_model.zip"  # ← اسم ملف الموديل في Storage


# ------------------------------
#   LOAD MODEL FROM FIREBASE STORAGE
# ------------------------------

def load_model_from_storage():
    print("Downloading model from Firebase Storage...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)

    # temporary folder
    tmp_dir = tempfile.mkdtemp()
    zip_path = f"{tmp_dir}/model.zip"

    # download ZIP
    blob.download_to_filename(zip_path)
    print(f"Downloaded model ZIP → {zip_path}")

    # extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    model_dir = f"{tmp_dir}/emotion_text_model"

    print(f"Model extracted to → {model_dir}")

    # load tokenizer + model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    print("Model loaded successfully.")
    return tokenizer, model


# Load model once when the function is deployed
tokenizer, model = load_model_from_storage()


# ------------------------------
#   HTTP ENDPOINT
# ------------------------------

@functions_framework.http
def analyze_text(request):
    """HTTP Cloud Function – Analyze emotion from text"""

    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return {"error": "No text provided"}, 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).tolist()[0]

        return json.dumps({
            "text": text,
            "emotion_scores": probs
        })

    except Exception as e:
        return {"error": str(e)}, 500

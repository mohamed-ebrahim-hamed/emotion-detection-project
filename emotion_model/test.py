import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ===== تحميل الموديل والـ Tokenizer =====
MODEL_PATH = "D:/results/emotion_model"  # المسار من الصورة

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

# اختيار الجهاز
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"✅ الموديل تم تحميله على: {device}")

# ===== تحميل أسماء الـ Emotions =====
# لو عندك ملف emotions.txt
EMOTIONS_FILE = "D:/results/emotions.txt"  # أو المسار الصحيح
try:
    with open(EMOTIONS_FILE) as f:
        emotion_names = [line.strip() for line in f]
except:
    # الـ 28 emotions الافتراضية
    emotion_names = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "neutral", "optimism", "pride",
        "realization", "relief", "remorse", "sadness", "surprise"
    ]


# ===== دالة التنبؤ =====
def predict_emotions(text, threshold=0.5):
    """
    التنبؤ بـ Emotions من نص

    Args:
        text: النص المراد تحليله
        threshold: الحد الأدنى للـ probability

    Returns:
        dict: الـ emotions والـ scores
    """
    # Tokenization
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)  # multi-label

    probs = probabilities.cpu().numpy()[0]

    # ترتيب النتائج
    results = {
        emotion: float(prob)
        for emotion, prob in zip(emotion_names, probs)
        if prob > threshold
    }

    # ترتيب حسب الـ score
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return results


# ===== اختبارات =====
test_texts = [
    "I love this!  This is amazing!",
    "I'm so angry right now!",
    "This is disappointing and sad",
    "I feel confused about this situation",
    ''
]

print("\n" + "=" * 60)
print("🧪 نتائج الاختبار:")
print("=" * 60)

for text in test_texts:
    print(f"\n📝 النص: {text}")
    results = predict_emotions(text, threshold=0.3)

    if results:
        for emotion, score in results.items():
            print(f"   • {emotion}: {score:.3f}")
    else:
        print("   لم يتم اكتشاف emotions")

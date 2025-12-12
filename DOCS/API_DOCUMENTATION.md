# API Documentation
# وثائق الواجهة البرمجية

---

## 📡 نظرة عامة / Overview

يوفر تطبيق Emotion Detection مجموعة من نقاط النهاية (API Endpoints) للتفاعل مع نماذج تحليل المشاعر.
يعمل التطبيق على Flask ويوفر واجهة RESTful API سهلة الاستخدام.

The Emotion Detection application provides a set of API endpoints to interact with emotion analysis models.
The application runs on Flask and provides an easy-to-use RESTful API interface.

---

## 🌐 Base URL

```
http://localhost:5000
```

أو في الإنتاج / Or in production:
```
https://your-domain.com
```

---

## 📚 API Endpoints

### 1. الصفحة الرئيسية / Home Page

**Endpoint:** `/`  
**Method:** `GET`  
**Description:** عرض الصفحة الرئيسية للتطبيق / Display the main application page

**Response:**
```html
<!-- HTML page -->
```

**مثال / Example:**
```bash
curl http://localhost:5000/
```

---

### 2. فحص صحة التطبيق / Health Check

**Endpoint:** `/health`  
**Method:** `GET`  
**Description:** التحقق من أن التطبيق والنماذج تعمل بشكل صحيح / Check that the application and models are working properly

**Response (Success):**
```json
{
    "status": "healthy",
    "model": "loaded",
    "scaler": "loaded",
    "encoder": "loaded"
}
```

**Response (Error):**
```json
{
    "status": "error",
    "message": "Resources not loaded"
}
```

**Status Codes:**
- `200 OK`: التطبيق يعمل بشكل صحيح
- `500 Internal Server Error`: فشل تحميل الموارد

**مثال / Example:**
```bash
curl http://localhost:5000/health
```

---

### 3. تحليل الملف الصوتي / Audio Analysis

**Endpoint:** `/predict`  
**Method:** `POST`  
**Description:** تحليل ملف صوتي للكشف عن المشاعر / Analyze audio file to detect emotions

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body Parameter:**
  - `audio` (file, required): ملف صوتي بأحد الصيغ التالية:
    - WAV
    - MP3
    - M4A
    - OGG
    - WEBM

**Request Example (cURL):**
```bash
curl -X POST http://localhost:5000/predict \
  -F "audio=@path/to/audio.wav"
```

**Request Example (Python):**
```python
import requests

url = "http://localhost:5000/predict"
files = {'audio': open('audio.wav', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Request Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('audio', audioFile);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response (Success):**
```json
{
    "success": true,
    "emotion": "happy",
    "emotion_arabic": "😃 سعيد",
    "emotion_color": "#FFD166",
    "confidence": 85.43,
    "probabilities": {
        "angry": 2.15,
        "disgust": 1.23,
        "fear": 3.45,
        "happy": 85.43,
        "neutral": 4.21,
        "sad": 2.11,
        "surprise": 1.42
    }
}
```

**Response Fields:**
- `success` (boolean): هل نجحت العملية
- `emotion` (string): العاطفة المكتشفة بالإنجليزية
- `emotion_arabic` (string): العاطفة مع الرمز التعبيري بالعربية
- `emotion_color` (string): اللون المخصص للعاطفة (HEX)
- `confidence` (float): نسبة الثقة (0-100)
- `probabilities` (object): احتماليات جميع العواطف

**Response (Error):**
```json
{
    "error": "No audio file provided"
}
```

```json
{
    "error": "File type not allowed. Please use WAV, MP3, M4A, OGG, or WEBM"
}
```

```json
{
    "error": "Prediction error: [error details]"
}
```

**Status Codes:**
- `200 OK`: التحليل نجح
- `400 Bad Request`: خطأ في الطلب (ملف غير موجود أو صيغة غير مدعومة)
- `500 Internal Server Error`: خطأ في المعالجة

**العواطف المدعومة / Supported Emotions:**
| English | Arabic | Emoji | Color |
|---------|--------|-------|-------|
| angry | غاضب | 😠 | #FF6B6B |
| disgust | مقرف | 🤢 | #8AC926 |
| fear | خائف | 😨 | #7209B7 |
| happy | سعيد | 😃 | #FFD166 |
| neutral | محايد | 😐 | #06D6A0 |
| sad | حزين | 😢 | #118AB2 |
| surprise | متفاجئ | 😲 | #EF476F |

---

### 4. تحليل النص / Text Analysis

**Endpoint:** `/predict-text`  
**Method:** `POST`  
**Description:** تحليل نص للكشف عن العواطف / Analyze text to detect emotions

**Request:**
- **Content-Type:** `application/json`
- **Body:**
```json
{
    "text": "I'm so excited about this new project!"
}
```

**Request Example (cURL):**
```bash
curl -X POST http://localhost:5000/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie so much!"}'
```

**Request Example (Python):**
```python
import requests

url = "http://localhost:5000/predict-text"
data = {"text": "I'm feeling great today!"}
response = requests.post(url, json=data)
print(response.json())
```

**Request Example (JavaScript):**
```javascript
fetch('http://localhost:5000/predict-text', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: "I'm so happy and excited!"
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response (Success):**
```json
{
    "success": true,
    "primary_emotion": {
        "emotion": "excitement",
        "emotion_arabic": "حماس",
        "emoji": "🤗",
        "probability": 87.5
    },
    "detected_emotions": [
        {
            "emotion": "excitement",
            "emotion_arabic": "حماس",
            "emoji": "🤗",
            "probability": 87.5
        },
        {
            "emotion": "joy",
            "emotion_arabic": "فرح",
            "emoji": "😃",
            "probability": 65.3
        },
        {
            "emotion": "optimism",
            "emotion_arabic": "تفاؤل",
            "emoji": "😊",
            "probability": 45.2
        }
    ],
    "all_probabilities": {
        "admiration": 12.5,
        "amusement": 8.3,
        "anger": 1.2,
        "annoyance": 2.1,
        "approval": 15.6,
        "caring": 10.2,
        "confusion": 3.4,
        "curiosity": 7.8,
        "desire": 5.6,
        "disappointment": 1.5,
        "disapproval": 0.8,
        "disgust": 0.5,
        "embarrassment": 2.3,
        "excitement": 87.5,
        "fear": 1.1,
        "gratitude": 8.9,
        "grief": 0.3,
        "joy": 65.3,
        "love": 12.4,
        "nervousness": 2.7,
        "neutral": 5.4,
        "optimism": 45.2,
        "pride": 18.3,
        "realization": 6.7,
        "relief": 9.1,
        "remorse": 0.6,
        "sadness": 0.9,
        "surprise": 11.2
    }
}
```

**Response Fields:**
- `success` (boolean): هل نجحت العملية
- `primary_emotion` (object): العاطفة الرئيسية (الأعلى احتمالاً)
  - `emotion` (string): اسم العاطفة بالإنجليزية
  - `emotion_arabic` (string): اسم العاطفة بالعربية
  - `emoji` (string): الرمز التعبيري
  - `probability` (float): نسبة الاحتمال (0-100)
- `detected_emotions` (array): جميع العواطف المكتشفة (أعلى من threshold)
- `all_probabilities` (object): احتماليات جميع الـ 28 عاطفة

**Response (Error):**
```json
{
    "error": "No text provided"
}
```

```json
{
    "error": "Text model not available. Please install: pip install torch transformers soxr"
}
```

```json
{
    "error": "Text prediction error: [error details]"
}
```

**Status Codes:**
- `200 OK`: التحليل نجح
- `400 Bad Request`: نص فارغ أو غير موجود
- `503 Service Unavailable`: النموذج النصي غير متوفر
- `500 Internal Server Error`: خطأ في المعالجة

**العواطف الـ 28 المدعومة / 28 Supported Emotions:**

| English | Arabic | Emoji |
|---------|--------|-------|
| admiration | إعجاب | 🤩 |
| amusement | تسلية | 😄 |
| anger | غضب | 😠 |
| annoyance | انزعاج | 😒 |
| approval | موافقة | 👍 |
| caring | اهتمام | 🤗 |
| confusion | ارتباك | 😕 |
| curiosity | فضول | 🤔 |
| desire | رغبة | 😍 |
| disappointment | خيبة أمل | 😞 |
| disapproval | رفض | 👎 |
| disgust | اشمئزاز | 🤢 |
| embarrassment | إحراج | 😳 |
| excitement | حماس | 🤗 |
| fear | خوف | 😨 |
| gratitude | امتنان | 🙏 |
| grief | حزن شديد | 😢 |
| joy | فرح | 😃 |
| love | حب | ❤️ |
| nervousness | توتر | 😰 |
| neutral | محايد | 😐 |
| optimism | تفاؤل | 😊 |
| pride | فخر | 😌 |
| realization | إدراك | 💡 |
| relief | ارتياح | 😌 |
| remorse | ندم | 😔 |
| sadness | حزن | 😢 |
| surprise | مفاجأة | 😲 |

**ملاحظة مهمة:**
- **threshold = 0.3**: فقط العواطف بنسبة أعلى من 30% تظهر في `detected_emotions`
- النص يمكن أن يحتوي على عدة عواطف في نفس الوقت (multi-label)

---

### 5. اختبار النموذج / Test Model

**Endpoint:** `/test-model`  
**Method:** `GET`  
**Description:** اختبار النموذج الصوتي باستخدام ملف تجريبي / Test audio model using a sample file

**Response (Success):**
```json
{
    "success": true,
    "message": "Model test successful",
    "predictions_shape": "(1, 7)",
    "sample_prediction": [0.02, 0.01, 0.03, 0.85, 0.04, 0.02, 0.03]
}
```

**Response (Error):**
```json
{
    "error": "No test files found in uploads folder"
}
```

```json
{
    "error": "Model test error: [error details]"
}
```

**Status Codes:**
- `200 OK`: الاختبار نجح
- `404 Not Found`: لا توجد ملفات تجريبية
- `500 Internal Server Error`: خطأ في الاختبار

**مثال / Example:**
```bash
curl http://localhost:5000/test-model
```

---

## 🔒 الأمان والحدود / Security and Limits

### حجم الملف / File Size:
```
الحد الأقصى: 16 ميجابايت
Maximum: 16 MB
```

### الصيغ المدعومة / Supported Formats:
```
Audio: WAV, MP3, M4A, OGG, WEBM
Text: أي نص (Any text)
```

### معالجة الأخطاء / Error Handling:
- جميع نقاط النهاية تعيد JSON
- الأخطاء تتضمن رسالة توضيحية
- يتم تسجيل الأخطاء في سجلات الخادم

### التنظيف التلقائي / Auto Cleanup:
- الملفات المرفوعة تُحذف بعد المعالجة
- الملفات المؤقتة تُحذف تلقائيًا

---

## 📊 Response Status Codes

| Code | Meaning | متى يحدث / When It Happens |
|------|---------|---------------------------|
| 200 | OK | العملية نجحت |
| 400 | Bad Request | خطأ في البيانات المرسلة |
| 404 | Not Found | المورد غير موجود |
| 500 | Internal Server Error | خطأ في الخادم |
| 503 | Service Unavailable | الخدمة غير متوفرة |

---

## 🛠️ أمثلة عملية / Practical Examples

### مثال 1: تحليل ملف صوتي بالكامل / Complete Audio Analysis

```python
import requests
import json

def analyze_audio(audio_path):
    """تحليل ملف صوتي والحصول على النتائج"""
    url = "http://localhost:5000/predict"
    
    try:
        # فتح الملف وإرساله
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(url, files=files)
        
        # التحقق من الاستجابة
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"العاطفة المكتشفة: {result['emotion_arabic']}")
                print(f"الثقة: {result['confidence']:.2f}%")
                print("\nجميع الاحتماليات:")
                
                for emotion, prob in result['probabilities'].items():
                    print(f"  {emotion}: {prob:.2f}%")
                    
                return result
            else:
                print(f"خطأ: {result.get('error')}")
        else:
            print(f"خطأ في الطلب: {response.status_code}")
            
    except Exception as e:
        print(f"خطأ: {str(e)}")
    
    return None

# استخدام الدالة
result = analyze_audio("my_audio.wav")
```

### مثال 2: تحليل نص بالكامل / Complete Text Analysis

```python
import requests

def analyze_text(text):
    """تحليل نص والحصول على النتائج"""
    url = "http://localhost:5000/predict-text"
    
    try:
        response = requests.post(url, json={'text': text})
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                primary = result['primary_emotion']
                print(f"العاطفة الرئيسية: {primary['emotion_arabic']} {primary['emoji']}")
                print(f"الاحتمال: {primary['probability']:.2f}%\n")
                
                print("العواطف المكتشفة:")
                for emotion in result['detected_emotions']:
                    print(f"  {emotion['emoji']} {emotion['emotion_arabic']}: {emotion['probability']:.2f}%")
                
                return result
            else:
                print(f"خطأ: {result.get('error')}")
        else:
            print(f"خطأ في الطلب: {response.status_code}")
            
    except Exception as e:
        print(f"خطأ: {str(e)}")
    
    return None

# استخدام الدالة
text = "I'm so excited about this new project! Can't wait to start working on it."
result = analyze_text(text)
```

### مثال 3: تحليل دُفعة من الملفات / Batch Analysis

```python
import os
import requests
import pandas as pd

def batch_analyze_audio(directory):
    """تحليل جميع الملفات الصوتية في مجلد"""
    url = "http://localhost:5000/predict"
    results = []
    
    # الحصول على جميع الملفات الصوتية
    audio_files = [f for f in os.listdir(directory) 
                   if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    print(f"تحليل {len(audio_files)} ملف صوتي...")
    
    for filename in audio_files:
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'rb') as f:
                files = {'audio': f}
                response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    results.append({
                        'filename': filename,
                        'emotion': result['emotion'],
                        'confidence': result['confidence']
                    })
                    print(f"✓ {filename}: {result['emotion']} ({result['confidence']:.1f}%)")
                    
        except Exception as e:
            print(f"✗ {filename}: خطأ - {str(e)}")
    
    # حفظ النتائج في CSV
    df = pd.DataFrame(results)
    df.to_csv('analysis_results.csv', index=False)
    print(f"\nتم حفظ النتائج في analysis_results.csv")
    
    return df

# استخدام الدالة
results_df = batch_analyze_audio("./audio_samples/")
```

### مثال 4: تحليل نصوص من CSV / Analyze Texts from CSV

```python
import pandas as pd
import requests
from tqdm import tqdm

def analyze_texts_from_csv(input_csv, output_csv):
    """تحليل نصوص من ملف CSV"""
    url = "http://localhost:5000/predict-text"
    
    # قراءة الملف
    df = pd.read_csv(input_csv)
    
    # قوائم للنتائج
    primary_emotions = []
    primary_probs = []
    
    # تحليل كل نص
    for text in tqdm(df['text'], desc="تحليل النصوص"):
        try:
            response = requests.post(url, json={'text': text})
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    primary = result['primary_emotion']
                    primary_emotions.append(primary['emotion'])
                    primary_probs.append(primary['probability'])
                else:
                    primary_emotions.append('error')
                    primary_probs.append(0)
            else:
                primary_emotions.append('error')
                primary_probs.append(0)
                
        except Exception as e:
            print(f"خطأ في تحليل النص: {str(e)}")
            primary_emotions.append('error')
            primary_probs.append(0)
    
    # إضافة النتائج إلى DataFrame
    df['emotion'] = primary_emotions
    df['confidence'] = primary_probs
    
    # حفظ النتائج
    df.to_csv(output_csv, index=False)
    print(f"تم حفظ النتائج في {output_csv}")
    
    return df

# استخدام الدالة
df = analyze_texts_from_csv("input_texts.csv", "analyzed_texts.csv")
```

### مثال 5: تطبيق Flask بسيط / Simple Flask App

```python
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
EMOTION_API = "http://localhost:5000"

@app.route('/')
def home():
    return render_template('analyze.html')

@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio']
    
    # إرسال إلى API
    response = requests.post(
        f"{EMOTION_API}/predict",
        files={'audio': file}
    )
    
    return response.json(), response.status_code

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # إرسال إلى API
    response = requests.post(
        f"{EMOTION_API}/predict-text",
        json=data
    )
    
    return response.json(), response.status_code

if __name__ == '__main__':
    app.run(port=5001)
```

---

## 🔧 استكشاف الأخطاء / Troubleshooting

### خطأ: "Model not loaded properly"
```
الحل:
1. تأكد من وجود ملفات النموذج في مجلد model/
2. تحقق من الأذونات (permissions)
3. أعد تشغيل الخادم
```

### خطأ: "Text model not available"
```
الحل:
pip install torch transformers soxr
```

### خطأ: "File type not allowed"
```
الحل:
استخدم أحد الصيغ المدعومة: WAV, MP3, M4A, OGG, WEBM
```

### خطأ: "ffmpeg not found"
```
الحل:
conda install -c conda-forge ffmpeg
# أو
apt-get install ffmpeg  # على Linux
brew install ffmpeg     # على macOS
```

---

## 📝 ملاحظات إضافية / Additional Notes

### الأداء / Performance:
- **تحليل صوتي**: ~2-3 ثواني لكل ملف
- **تحليل نصي**: ~1 ثانية لكل نص
- يمكن تحسين الأداء باستخدام:
  - GPU للنموذج النصي
  - Caching للنتائج المتكررة
  - Load balancing لعدة خوادم

### الدقة / Accuracy:
- **النموذج الصوتي**: ~75-80%
- **النموذج النصي**: متغيرة حسب العاطفة (50-85%)

### القيود / Limitations:
- النموذج الصوتي: مدرب على اللغة الإنجليزية فقط
- النموذج النصي: مدرب على اللغة الإنجليزية (يمكن أن يعمل مع لغات أخرى بدقة أقل)
- يفضل نصوص قصيرة (أقل من 128 كلمة)

---

**انتهت وثائق API**
**End of API Documentation**

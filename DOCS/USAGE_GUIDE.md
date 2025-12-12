# Usage Guide / دليل الاستخدام
# Emotion Detection Project

---

## 📖 المحتويات / Contents

1. [التثبيت والإعداد](#installation)
2. [بدء التشغيل](#getting-started)
3. [استخدام الواجهة الويب](#web-interface)
4. [استخدام API](#using-api)
5. [أمثلة عملية](#practical-examples)
6. [نصائح وحيل](#tips-and-tricks)
7. [استكشاف الأخطاء](#troubleshooting)

---

<a name="installation"></a>
## 🚀 التثبيت والإعداد / Installation and Setup

### المتطلبات الأساسية / Prerequisites:

```bash
# Python 3.7 أو أحدث
python --version  # يجب أن يكون 3.7+

# pip (مدير الحزم)
pip --version
```

### خطوة 1: نسخ المشروع / Clone the Repository

```bash
git clone https://github.com/mohamed-ebrahim-hamed/emotion-detection-project.git
cd emotion-detection-project
```

### خطوة 2: إنشاء بيئة افتراضية (اختياري لكن موصى به) / Create Virtual Environment

**على Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**على Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### خطوة 3: تثبيت المتطلبات / Install Requirements

```bash
pip install -r requirements.txt
```

**إذا كنت تريد استخدام النموذج النصي:**
```bash
pip install torch transformers soxr
```

### خطوة 4: تحميل النماذج المدربة / Download Pre-trained Models

1. قم بتحميل النماذج من الروابط التالية:
   - [Voice Model](https://drive.google.com/drive/folders/1BiVjgp9NKe4rI5ZBAV4m6FEWCLneQ-ob?usp=drive_link)
   - [Text Model](https://drive.google.com/drive/folders/1NNbntFG6XvTstb0xDGsqsngWKdLIzzaW?usp=drive_link)

2. ضع الملفات في المجلد المناسب:

```
emotion-detection-project/
├── model/
│   ├── CNN_model.json           # معمارية نموذج الصوت
│   ├── best_model1_weights.h5   # أوزان نموذج الصوت
│   ├── scaler2.pickle           # Scaler للميزات
│   ├── encoder2.pickle          # Encoder للتسميات
│   └── Text Model/              # نموذج النص (DistilBERT)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── vocab.txt
```

### خطوة 5: تثبيت ffmpeg (للتعامل مع صيغ الصوت المختلفة) / Install ffmpeg

**على Conda:**
```bash
conda install -c conda-forge ffmpeg
```

**على Linux:**
```bash
sudo apt-get install ffmpeg
```

**على macOS:**
```bash
brew install ffmpeg
```

**على Windows:**
1. قم بتحميل ffmpeg من [ffmpeg.org](https://ffmpeg.org/download.html)
2. أضف المسار إلى PATH

---

<a name="getting-started"></a>
## 🎬 بدء التشغيل / Getting Started

### تشغيل التطبيق / Run the Application

```bash
python app.py
```

**المخرجات المتوقعة:**
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

### فتح التطبيق في المتصفح / Open in Browser

```
http://localhost:5000
```

أو

```
http://127.0.0.1:5000
```

---

<a name="web-interface"></a>
## 🌐 استخدام الواجهة الويب / Using Web Interface

### 1. تحليل الصوت / Audio Analysis

#### الخطوات / Steps:

1. **اختر "تحليل صوت" من القائمة**
   
2. **رفع الملف الصوتي:**
   - انقر على "اختر ملف" أو "Choose File"
   - اختر ملف صوتي من جهازك
   - الصيغ المدعومة: WAV, MP3, M4A, OGG, WEBM

3. **انقر على "تحليل" / "Analyze"**

4. **انتظر النتيجة** (2-3 ثواني عادة)

5. **اقرأ النتائج:**
   ```
   العاطفة المكتشفة: 😃 سعيد
   نسبة الثقة: 85.43%
   
   جميع الاحتماليات:
   - happy: 85.43%
   - neutral: 4.21%
   - fear: 3.45%
   - angry: 2.15%
   - sad: 2.11%
   - surprise: 1.42%
   - disgust: 1.23%
   ```

#### مثال عملي / Practical Example:

```
سيناريو: تحليل مكالمة خدمة عملاء

1. سجل المكالمة (أو استخدم تسجيل موجود)
2. حوّل إلى WAV أو MP3
3. ارفع الملف إلى التطبيق
4. احصل على النتيجة:
   - إذا كانت "angry" أو "sad": العميل غير راضٍ
   - إذا كانت "happy" أو "neutral": العميل راضٍ
   - إذا كانت "fear": العميل قلق أو محتار
```

---

### 2. تحليل النص / Text Analysis

#### الخطوات / Steps:

1. **اختر "تحليل نص" من القائمة**

2. **اكتب أو الصق النص:**
   ```
   مثال بالإنجليزية:
   "I'm so excited about this new opportunity! 
   Can't wait to get started."
   
   مثال آخر:
   "This is disappointing and frustrating."
   ```

3. **انقر على "تحليل" / "Analyze"**

4. **اقرأ النتائج:**
   ```
   العاطفة الرئيسية: 🤗 حماس (87.5%)
   
   العواطف المكتشفة:
   - 🤗 حماس: 87.5%
   - 😃 فرح: 65.3%
   - 😊 تفاؤل: 45.2%
   ```

#### حالات استخدام / Use Cases:

**1. تحليل تعليقات العملاء:**
```python
تعليق: "The product is amazing! Best purchase ever."
النتيجة: joy (فرح), admiration (إعجاب)
الإجراء: رد بشكر العميل، اطلب مراجعة
```

**2. تحليل منشورات وسائل التواصل:**
```python
منشور: "Can't believe this happened. So disappointed."
النتيجة: disappointment (خيبة أمل), sadness (حزن)
الإجراء: تواصل مع صاحب المنشور، قدم مساعدة
```

**3. تحليل رسائل البريد:**
```python
رسالة: "Thank you so much for your help! Really appreciate it."
النتيجة: gratitude (امتنان), joy (فرح)
الإجراء: رد إيجابي، حافظ على العلاقة
```

---

<a name="using-api"></a>
## 🔌 استخدام API / Using API

### 1. تحليل صوت عبر API / Audio Analysis via API

#### Python Example:

```python
import requests

# URL الخادم
url = "http://localhost:5000/predict"

# مسار الملف الصوتي
audio_file = "path/to/your/audio.wav"

# إرسال الطلب
with open(audio_file, 'rb') as f:
    files = {'audio': f}
    response = requests.post(url, files=files)

# معالجة النتيجة
if response.status_code == 200:
    result = response.json()
    print(f"العاطفة: {result['emotion']}")
    print(f"الثقة: {result['confidence']}%")
else:
    print(f"خطأ: {response.status_code}")
```

#### JavaScript Example:

```javascript
async function analyzeAudio(audioFile) {
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`العاطفة: ${result.emotion}`);
            console.log(`الثقة: ${result.confidence}%`);
        }
    } catch (error) {
        console.error('خطأ:', error);
    }
}
```

#### cURL Example:

```bash
curl -X POST http://localhost:5000/predict \
  -F "audio=@audio.wav"
```

---

### 2. تحليل نص عبر API / Text Analysis via API

#### Python Example:

```python
import requests

url = "http://localhost:5000/predict-text"
text = "I'm so happy and grateful for this opportunity!"

response = requests.post(url, json={'text': text})

if response.status_code == 200:
    result = response.json()
    primary = result['primary_emotion']
    print(f"العاطفة الرئيسية: {primary['emotion_arabic']} {primary['emoji']}")
    print(f"الاحتمال: {primary['probability']}%")
    
    print("\nجميع العواطف المكتشفة:")
    for emotion in result['detected_emotions']:
        print(f"- {emotion['emoji']} {emotion['emotion_arabic']}: {emotion['probability']}%")
```

#### JavaScript Example:

```javascript
async function analyzeText(text) {
    try {
        const response = await fetch('http://localhost:5000/predict-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const result = await response.json();
        
        if (result.success) {
            const primary = result.primary_emotion;
            console.log(`العاطفة الرئيسية: ${primary.emotion_arabic} ${primary.emoji}`);
            console.log(`الاحتمال: ${primary.probability}%`);
        }
    } catch (error) {
        console.error('خطأ:', error);
    }
}

// استخدام
analyzeText("I love this product so much!");
```

#### cURL Example:

```bash
curl -X POST http://localhost:5000/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

---

<a name="practical-examples"></a>
## 💡 أمثلة عملية / Practical Examples

### مثال 1: نظام خدمة عملاء / Customer Service System

```python
import requests
import pandas as pd
from datetime import datetime

class EmotionCustomerService:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.logs = []
    
    def analyze_call(self, audio_file, customer_id):
        """تحليل مكالمة عميل"""
        url = f"{self.api_url}/predict"
        
        with open(audio_file, 'rb') as f:
            response = requests.post(url, files={'audio': f})
        
        if response.status_code == 200:
            result = response.json()
            
            # تسجيل النتيجة
            log = {
                'timestamp': datetime.now(),
                'customer_id': customer_id,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'action_required': self._get_action(result['emotion'])
            }
            self.logs.append(log)
            
            return log
        
        return None
    
    def _get_action(self, emotion):
        """تحديد الإجراء المطلوب بناءً على العاطفة"""
        actions = {
            'angry': 'عاجل: تصعيد للمدير',
            'sad': 'مهم: المتابعة مع العميل',
            'fear': 'تقديم طمأنينة ودعم',
            'happy': 'فرصة: طلب مراجعة',
            'neutral': 'عادي: متابعة روتينية',
            'disgust': 'مهم: التحقق من المشكلة',
            'surprise': 'متابعة: التأكد من الفهم'
        }
        return actions.get(emotion, 'متابعة عادية')
    
    def generate_report(self):
        """إنشاء تقرير بالنتائج"""
        df = pd.DataFrame(self.logs)
        
        print("=== تقرير تحليل المكالمات ===")
        print(f"\nإجمالي المكالمات: {len(df)}")
        print("\nتوزيع المشاعر:")
        print(df['emotion'].value_counts())
        print("\nالحالات التي تحتاج متابعة عاجلة:")
        urgent = df[df['emotion'].isin(['angry', 'sad', 'disgust'])]
        print(urgent[['customer_id', 'emotion', 'action_required']])
        
        return df

# استخدام النظام
service = EmotionCustomerService()

# تحليل عدة مكالمات
calls = [
    ('call1.wav', 'CUST001'),
    ('call2.wav', 'CUST002'),
    ('call3.wav', 'CUST003')
]

for audio_file, customer_id in calls:
    result = service.analyze_call(audio_file, customer_id)
    print(f"العميل {customer_id}: {result['emotion']} - {result['action_required']}")

# إنشاء تقرير
report = service.generate_report()
```

---

### مثال 2: تحليل مراجعات المنتجات / Product Review Analysis

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

class ProductReviewAnalyzer:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
    
    def analyze_reviews(self, reviews_file):
        """تحليل ملف CSV من المراجعات"""
        df = pd.read_csv(reviews_file)
        results = []
        
        url = f"{self.api_url}/predict-text"
        
        for idx, row in df.iterrows():
            response = requests.post(url, json={'text': row['review']})
            
            if response.status_code == 200:
                result = response.json()
                primary = result['primary_emotion']
                
                results.append({
                    'review_id': row['id'],
                    'review': row['review'],
                    'emotion': primary['emotion'],
                    'emotion_ar': primary['emotion_arabic'],
                    'confidence': primary['probability'],
                    'rating': row.get('rating', None)
                })
        
        return pd.DataFrame(results)
    
    def visualize_emotions(self, df):
        """رسم توزيع المشاعر"""
        emotion_counts = df['emotion'].value_counts()
        
        plt.figure(figsize=(12, 6))
        emotion_counts.plot(kind='bar', color='skyblue')
        plt.title('توزيع المشاعر في المراجعات')
        plt.xlabel('العاطفة')
        plt.ylabel('عدد المراجعات')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('emotions_distribution.png')
        plt.show()
    
    def get_sentiment_score(self, df):
        """حساب درجة المشاعر الإجمالية"""
        positive_emotions = ['joy', 'gratitude', 'love', 'admiration', 'excitement', 'optimism']
        negative_emotions = ['anger', 'sadness', 'disappointment', 'disgust', 'fear', 'grief']
        
        positive_count = df[df['emotion'].isin(positive_emotions)].shape[0]
        negative_count = df[df['emotion'].isin(negative_emotions)].shape[0]
        total = df.shape[0]
        
        sentiment_score = (positive_count - negative_count) / total * 100
        
        print(f"=== تحليل المشاعر ===")
        print(f"إجمالي المراجعات: {total}")
        print(f"إيجابية: {positive_count} ({positive_count/total*100:.1f}%)")
        print(f"سلبية: {negative_count} ({negative_count/total*100:.1f}%)")
        print(f"محايدة: {total - positive_count - negative_count}")
        print(f"درجة المشاعر: {sentiment_score:+.1f}%")
        
        return sentiment_score

# استخدام المحلل
analyzer = ProductReviewAnalyzer()

# تحليل المراجعات
results_df = analyzer.analyze_reviews('reviews.csv')

# عرض النتائج
print(results_df.head())

# رسم التوزيع
analyzer.visualize_emotions(results_df)

# حساب درجة المشاعر
score = analyzer.get_sentiment_score(results_df)
```

---

### مثال 3: تحليل وسائل التواصل الاجتماعي / Social Media Monitor

```python
import requests
from datetime import datetime
import time

class SocialMediaMonitor:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.alerts = []
    
    def monitor_posts(self, posts):
        """مراقبة منشورات وسائل التواصل"""
        url = f"{self.api_url}/predict-text"
        
        for post in posts:
            response = requests.post(url, json={'text': post['content']})
            
            if response.status_code == 200:
                result = response.json()
                
                # التحقق من المشاعر السلبية
                negative_emotions = self._check_negative_emotions(result)
                
                if negative_emotions:
                    self._create_alert(post, negative_emotions)
            
            time.sleep(0.5)  # تجنب إرهاق الخادم
    
    def _check_negative_emotions(self, result):
        """التحقق من وجود مشاعر سلبية"""
        negative = ['anger', 'disappointment', 'disgust', 'fear', 'grief', 'sadness']
        detected = []
        
        for emotion in result['detected_emotions']:
            if emotion['emotion'] in negative and emotion['probability'] > 50:
                detected.append(emotion)
        
        return detected
    
    def _create_alert(self, post, emotions):
        """إنشاء تنبيه"""
        alert = {
            'timestamp': datetime.now(),
            'post_id': post['id'],
            'author': post['author'],
            'content': post['content'][:100] + '...',
            'emotions': [e['emotion_arabic'] for e in emotions],
            'priority': self._get_priority(emotions)
        }
        self.alerts.append(alert)
        
        # إرسال إشعار فوري للحالات العاجلة
        if alert['priority'] == 'عاجل':
            self._send_notification(alert)
    
    def _get_priority(self, emotions):
        """تحديد أولوية التنبيه"""
        critical = ['anger', 'disgust']
        
        for emotion in emotions:
            if emotion['emotion'] in critical and emotion['probability'] > 70:
                return 'عاجل'
        
        return 'عادي'
    
    def _send_notification(self, alert):
        """إرسال إشعار (يمكن دمجه مع Slack, Email, إلخ)"""
        print(f"⚠️  تنبيه عاجل!")
        print(f"المؤلف: {alert['author']}")
        print(f"المحتوى: {alert['content']}")
        print(f"المشاعر: {', '.join(alert['emotions'])}")
        print("-" * 50)
    
    def get_alerts_report(self):
        """تقرير التنبيهات"""
        if not self.alerts:
            print("لا توجد تنبيهات")
            return
        
        print(f"=== تقرير التنبيهات ===")
        print(f"إجمالي التنبيهات: {len(self.alerts)}")
        
        urgent = [a for a in self.alerts if a['priority'] == 'عاجل']
        print(f"تنبيهات عاجلة: {len(urgent)}")
        
        print("\nالتنبيهات العاجلة:")
        for alert in urgent:
            print(f"- {alert['author']}: {alert['emotions']}")

# استخدام المراقب
monitor = SocialMediaMonitor()

# مثال على المنشورات
posts = [
    {
        'id': 1,
        'author': '@user1',
        'content': 'This product is terrible! Worst experience ever.'
    },
    {
        'id': 2,
        'author': '@user2',
        'content': 'I love this so much! Amazing quality.'
    },
    {
        'id': 3,
        'author': '@user3',
        'content': 'Very disappointed with the service.'
    }
]

# مراقبة المنشورات
monitor.monitor_posts(posts)

# الحصول على تقرير
monitor.get_alerts_report()
```

---

<a name="tips-and-tricks"></a>
## 💫 نصائح وحيل / Tips and Tricks

### 1. تحسين دقة تحليل الصوت / Improving Audio Analysis Accuracy

**جودة التسجيل:**
```
✅ استخدم ميكروفون جيد
✅ سجل في بيئة هادئة
✅ تجنب الضوضاء الخلفية
✅ استخدم معدل عينات 22050 Hz أو أعلى
```

**مدة التسجيل:**
```
✅ 2-5 ثواني كافية
❌ تجنب التسجيلات الطويلة جدًا (> 30 ثانية)
❌ تجنب التسجيلات القصيرة جدًا (< 1 ثانية)
```

**الصيغة:**
```
⭐ الأفضل: WAV (بدون ضغط)
✅ جيد: MP3 (320 kbps)
⚠️  مقبول: MP3 (128 kbps)
```

---

### 2. تحسين دقة تحليل النص / Improving Text Analysis Accuracy

**طول النص:**
```
✅ 10-100 كلمة مثالي
⚠️  < 5 كلمات: قد تكون النتائج غير دقيقة
⚠️  > 128 كلمة: سيتم قص الباقي
```

**جودة النص:**
```
✅ استخدم جمل كاملة
✅ تجنب الأخطاء الإملائية
⚠️  الرموز التعبيرية قد تؤثر على النتائج
⚠️  الاختصارات قد تكون غير مفهومة
```

**اللغة:**
```
⭐ الأفضل: الإنجليزية
✅ جيد: لغات أوروبية أخرى
⚠️  محدود: العربية والصينية واليابانية
```

---

### 3. تحسين الأداء / Performance Optimization

**للتطبيق:**
```python
# استخدام Gunicorn للإنتاج
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# -w 4: 4 workers (اضبط حسب عدد CPU cores)
# -b: Bind address
```

**للنموذج النصي:**
```python
# استخدام GPU إن وجد
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Batch processing
texts = ["text1", "text2", "text3"]
# معالجة دفعات بدلاً من واحد تلو الآخر
```

**Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_analyze(text):
    # تخزين النتائج المكررة
    return analyze_text(text)
```

---

### 4. معالجة الأخطاء / Error Handling

**مثال شامل:**
```python
import requests
from requests.exceptions import RequestException, Timeout
import time

def analyze_with_retry(audio_file, max_retries=3):
    """تحليل مع إعادة المحاولة"""
    url = "http://localhost:5000/predict"
    
    for attempt in range(max_retries):
        try:
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    url, 
                    files={'audio': f},
                    timeout=30  # 30 ثانية timeout
                )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print(f"الخدمة غير متوفرة، إعادة المحاولة {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"خطأ: {response.status_code}")
                return None
                
        except Timeout:
            print(f"انتهت المهلة، إعادة المحاولة {attempt + 1}/{max_retries}...")
            time.sleep(2 ** attempt)
        except RequestException as e:
            print(f"خطأ في الطلب: {e}")
            return None
    
    print("فشلت جميع المحاولات")
    return None

# استخدام
result = analyze_with_retry("audio.wav")
```

---

<a name="troubleshooting"></a>
## 🔧 استكشاف الأخطاء / Troubleshooting

### مشكلة: التطبيق لا يبدأ

**الأعراض:**
```
ModuleNotFoundError: No module named 'flask'
```

**الحل:**
```bash
pip install -r requirements.txt
```

---

### مشكلة: "Model not loaded"

**الأعراض:**
```
Failed to load model or preprocessing objects. Exiting...
```

**الحل:**
```bash
# تأكد من وجود الملفات
ls -la model/

# يجب أن ترى:
# CNN_model.json
# best_model1_weights.h5
# scaler2.pickle
# encoder2.pickle
```

---

### مشكلة: "ffmpeg not found"

**الأعراض:**
```
ffmpeg is not installed. Please install ffmpeg
```

**الحل:**
```bash
# على Conda
conda install -c conda-forge ffmpeg

# على Ubuntu/Debian
sudo apt-get install ffmpeg

# على macOS
brew install ffmpeg
```

---

### مشكلة: بطء في التحليل

**الأعراض:**
- التحليل يستغرق أكثر من 10 ثواني

**الحل:**
```python
# 1. استخدام GPU للنموذج النصي
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. تقليل حجم الملف الصوتي
# استخدم صيغة مضغوطة (MP3 بدلاً من WAV)

# 3. تقليل جودة الصوت إذا لم تكن مهمة
# 22050 Hz كافي للتحليل
```

---

### مشكلة: نتائج غير دقيقة

**للصوت:**
```
✓ تحقق من جودة التسجيل
✓ تأكد من عدم وجود ضوضاء خلفية
✓ استخدم تسجيل واضح (2-5 ثواني)
✓ تجنب الموسيقى في الخلفية
```

**للنص:**
```
✓ استخدم جمل كاملة
✓ تجنب الأخطاء الإملائية
✓ استخدم الإنجليزية للحصول على أفضل النتائج
✓ تجنب النصوص القصيرة جدًا (< 5 كلمات)
```

---

### مشكلة: استخدام عالٍ للذاكرة

**الحل:**
```bash
# تقليل عدد workers في Gunicorn
gunicorn -w 2 app:app  # بدلاً من 4

# أو استخدام threading بدلاً من multiprocessing
gunicorn --threads 4 app:app
```

---

## 📞 الدعم / Support

إذا واجهت مشاكل أخرى:

1. **تحقق من Logs:**
```bash
# تشغيل مع debug mode
python app.py  # سيطبع الأخطاء بالتفصيل
```

2. **GitHub Issues:**
```
افتح issue جديد على:
https://github.com/mohamed-ebrahim-hamed/emotion-detection-project/issues
```

3. **الوثائق:**
```
راجع الوثائق الأخرى في مجلد DOCS/
```

---

## 🎓 موارد إضافية / Additional Resources

### للتعلم أكثر:
- [VOICE_MODEL_EXPLAINED.md](./VOICE_MODEL_EXPLAINED.md) - شرح نموذج الصوت
- [TEXT_MODEL_EXPLAINED.md](./TEXT_MODEL_EXPLAINED.md) - شرح نموذج النص
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) - وثائق API

### مواقع مفيدة:
- [Flask Documentation](https://flask.palletsprojects.com/)
- [librosa Documentation](https://librosa.org/doc/latest/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**انتهى دليل الاستخدام**
**End of Usage Guide**

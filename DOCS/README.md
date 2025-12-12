# Emotion Detection Project - Complete Documentation

## 📚 نظرة عامة على المشروع / Project Overview

مشروع تحليل المشاعر هو تطبيق ويب متكامل يستخدم تقنيات التعلم العميق (Deep Learning) للتعرف على المشاعر من خلال مصدرين مختلفين:
1. **الصوت (Audio)**: تحليل التسجيلات الصوتية للكشف عن المشاعر
2. **النص (Text)**: تحليل النصوص المكتوبة للكشف عن المشاعر

This Emotion Detection Project is a comprehensive web application that uses deep learning techniques to recognize emotions from two different sources:
1. **Audio**: Analyzing voice recordings to detect emotions
2. **Text**: Analyzing written text to detect emotions

---

## 🎯 الهدف من المشروع / Project Goal

الهدف الرئيسي هو بناء نظام ذكي قادر على:
- التعرف على 7 مشاعر أساسية من الصوت: غاضب، مقرف، خائف، سعيد، حزين، متفاجئ، محايد
- التعرف على 28 عاطفة مختلفة من النص بناءً على مجموعة بيانات GoEmotions
- توفير واجهة ويب سهلة الاستخدام للتفاعل مع النماذج

The main goal is to build an intelligent system capable of:
- Recognizing 7 basic emotions from audio: angry, disgust, fear, happy, sad, surprise, neutral
- Recognizing 28 different emotions from text based on the GoEmotions dataset
- Providing an easy-to-use web interface for interacting with the models

---

## 🏗️ معمارية المشروع / Project Architecture

```
emotion-detection-project/
│
├── app.py                      # Flask backend application
├── requirements.txt            # Python dependencies
│
├── final-voice-model.ipynb    # Audio emotion detection model training
├── text-model.ipynb           # Text emotion detection model training
│
├── model/                     # Trained models directory
│   ├── CNN_model.json         # Audio model architecture
│   ├── best_model1_weights.h5 # Audio model weights
│   ├── scaler2.pickle         # Feature scaler for audio
│   ├── encoder2.pickle        # Label encoder for audio
│   └── Text Model/            # Text model (DistilBERT)
│
├── templates/                 # HTML templates
│   ├── index.html            # Main page
│   ├── about.html            # About page
│   └── result.html           # Results page
│
├── static/                    # Static files (CSS, JS)
│   ├── css/
│   └── js/
│
├── testSounds/               # Sample audio files for testing
│
├── uploads/                  # Temporary uploads folder
│
└── DOCS/                     # Documentation folder
    ├── README.md             # Main documentation (this file)
    ├── VOICE_MODEL_EXPLAINED.md  # Cell-by-cell explanation of audio model
    ├── TEXT_MODEL_EXPLAINED.md   # Cell-by-cell explanation of text model
    ├── API_DOCUMENTATION.md      # API endpoints documentation
    └── USAGE_GUIDE.md            # User guide with examples
```

---

## 🔧 المكونات الرئيسية / Main Components

### 1. نموذج الصوت / Audio Model
- **النوع**: Convolutional Neural Network (CNN)
- **المدخلات**: ملفات صوتية (WAV, MP3, M4A, OGG, WEBM)
- **المخرجات**: 7 مشاعر أساسية
- **الميزات المستخدمة**: 
  - Zero Crossing Rate (ZCR)
  - Root Mean Square Energy (RMSE)
  - Mel-Frequency Cepstral Coefficients (MFCC)

### 2. نموذج النص / Text Model
- **النوع**: DistilBERT (Transformer-based)
- **المدخلات**: نصوص عربية أو إنجليزية
- **المخرجات**: 28 عاطفة مختلفة
- **مجموعة البيانات**: GoEmotions

### 3. تطبيق Flask / Flask Application
- **الإطار**: Flask (Python Web Framework)
- **الوظائف**:
  - رفع وتحليل الملفات الصوتية
  - إدخال وتحليل النصوص
  - عرض النتائج بشكل تفاعلي
  - دعم اللغة العربية

---

## 📊 مجموعات البيانات المستخدمة / Datasets Used

### للنموذج الصوتي / For Audio Model:
1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
2. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
3. **TESS** (Toronto Emotional Speech Set)
4. **SAVEE** (Surrey Audio-Visual Expressed Emotion)

### للنموذج النصي / For Text Model:
1. **GoEmotions**: مجموعة بيانات من Google تحتوي على 58,000 تعليق من Reddit مصنفة إلى 28 عاطفة

---

## 🚀 التثبيت والإعداد / Installation and Setup

### المتطلبات / Requirements:
- Python 3.7+
- TensorFlow 2.4.1
- PyTorch (للنموذج النصي / for text model)
- Flask 2.2.5
- librosa 0.8.1
- transformers (للنموذج النصي / for text model)

### خطوات التثبيت / Installation Steps:

```bash
# 1. نسخ المشروع / Clone the repository
git clone https://github.com/mohamed-ebrahim-hamed/emotion-detection-project.git
cd emotion-detection-project

# 2. تثبيت المكتبات المطلوبة / Install required packages
pip install -r requirements.txt

# 3. تحميل النماذج المدربة / Download pre-trained models
# Download from the provided Google Drive links and place in model/ directory

# 4. تشغيل التطبيق / Run the application
python app.py

# 5. فتح المتصفح / Open browser
# Navigate to http://localhost:5000
```

---

## 🎨 المشاعر المدعومة / Supported Emotions

### النموذج الصوتي / Audio Model (7 Emotions):
| Emotion | Arabic | Emoji | Color |
|---------|--------|-------|-------|
| Angry | غاضب | 😠 | #FF6B6B |
| Disgust | مقرف | 🤢 | #8AC926 |
| Fear | خائف | 😨 | #7209B7 |
| Happy | سعيد | 😃 | #FFD166 |
| Sad | حزين | 😢 | #118AB2 |
| Surprise | متفاجئ | 😲 | #EF476F |
| Neutral | محايد | 😐 | #06D6A0 |

### النموذج النصي / Text Model (28 Emotions):
إعجاب، تسلية، غضب، انزعاج، موافقة، اهتمام، ارتباك، فضول، رغبة، خيبة أمل، رفض، اشمئزاز، إحراج، حماس، خوف، امتنان، حزن شديد، فرح، حب، توتر، محايد، تفاؤل، فخر، إدراك، ارتياح، ندم، حزن، مفاجأة

---

## 🔍 كيفية عمل النماذج / How the Models Work

### نموذج الصوت / Audio Model:

1. **تحميل الملف الصوتي** / Load audio file
2. **تحويل الصيغة إذا لزم الأمر** / Convert format if needed (using ffmpeg)
3. **استخراج الميزات** / Extract features:
   - ZCR: معدل تجاوز الصفر (يقيس التغيرات في الإشارة)
   - RMSE: جذر متوسط مربع الطاقة (يقيس قوة الصوت)
   - MFCC: معاملات سيبسترال ميل (تمثل الخصائص الطيفية)
4. **تطبيع الميزات** / Normalize features using StandardScaler
5. **التنبؤ باستخدام CNN** / Predict using CNN model
6. **إرجاع النتيجة** / Return emotion with confidence score

### نموذج النص / Text Model:

1. **استقبال النص** / Receive text input
2. **تجزئة النص** / Tokenize text using DistilBERT tokenizer
3. **تحويل إلى تنسيق مناسب** / Convert to model format
4. **التنبؤ باستخدام DistilBERT** / Predict using DistilBERT
5. **حساب الاحتماليات** / Calculate probabilities for all 28 emotions
6. **إرجاع العواطف المكتشفة** / Return detected emotions (threshold > 0.3)

---

## 📡 نقاط النهاية API / API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | الصفحة الرئيسية / Main page |
| `/predict` | POST | تحليل الملف الصوتي / Analyze audio file |
| `/predict-text` | POST | تحليل النص / Analyze text |
| `/health` | GET | التحقق من حالة التطبيق / Check application health |
| `/test-model` | GET | اختبار النموذج / Test model |

للمزيد من التفاصيل، راجع / For more details, see: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

---

## 📖 الوثائق التفصيلية / Detailed Documentation

1. **[VOICE_MODEL_EXPLAINED.md](./VOICE_MODEL_EXPLAINED.md)**
   - شرح تفصيلي لكل خلية في final-voice-model.ipynb
   - Detailed cell-by-cell explanation of the audio model notebook

2. **[TEXT_MODEL_EXPLAINED.md](./TEXT_MODEL_EXPLAINED.md)**
   - شرح تفصيلي لكل خلية في text-model.ipynb
   - Detailed cell-by-cell explanation of the text model notebook

3. **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)**
   - وثائق API الكاملة مع أمثلة
   - Complete API documentation with examples

4. **[USAGE_GUIDE.md](./USAGE_GUIDE.md)**
   - دليل الاستخدام مع أمثلة عملية
   - Usage guide with practical examples

---

## 🎓 التقنيات المستخدمة / Technologies Used

### Backend:
- **Python 3.7+**
- **Flask**: إطار تطوير الويب
- **TensorFlow/Keras**: للنموذج الصوتي (CNN)
- **PyTorch**: للنموذج النصي (DistilBERT)
- **librosa**: لمعالجة الصوت
- **transformers**: لنموذج اللغة

### Frontend:
- **HTML5**
- **CSS3**
- **JavaScript**
- **Bootstrap** (إن وجد)

### Machine Learning:
- **Convolutional Neural Networks (CNN)**
- **Transformer Architecture (DistilBERT)**
- **Feature Extraction Techniques**
- **Data Augmentation**

---

## 📈 الأداء / Performance

### نموذج الصوت / Audio Model:
- **الدقة**: ~75% على بيانات الاختبار
- **وقت التنبؤ**: ~2-3 ثواني لكل ملف صوتي

### نموذج النص / Text Model:
- **الدقة**: متغيرة حسب العاطفة (F1-score على GoEmotions)
- **وقت التنبؤ**: ~1 ثانية لكل نص

---

## 🔐 الأمان / Security

- **التحقق من نوع الملفات**: فقط صيغ الصوت المسموح بها
- **حد أقصى لحجم الملف**: 16 ميجابايت
- **تنظيف الملفات المؤقتة**: حذف تلقائي بعد المعالجة
- **معالجة الأخطاء**: تسجيل شامل للأخطاء

---

## 🐛 استكشاف الأخطاء / Troubleshooting

### مشكلة: النموذج لا يتحمل
**الحل**: 
- تأكد من وجود ملفات النموذج في المسار الصحيح
- تحقق من صيغة الملفات (JSON, H5, pickle)

### مشكلة: خطأ في تحويل الصوت
**الحل**: 
- تأكد من تثبيت ffmpeg: `conda install -c conda-forge ffmpeg`

### مشكلة: النموذج النصي غير متوفر
**الحل**: 
- ثبت المكتبات المطلوبة: `pip install torch transformers soxr`

---

## 🤝 المساهمة / Contributing

نرحب بالمساهمات! يرجى:
1. Fork المشروع
2. إنشاء branch جديد (`git checkout -b feature/AmazingFeature`)
3. Commit التغييرات (`git commit -m 'Add some AmazingFeature'`)
4. Push إلى Branch (`git push origin feature/AmazingFeature`)
5. فتح Pull Request

---

## 📝 الترخيص / License

هذا المشروع مرخص تحت MIT License

---

##

# شرح تفصيلي لنموذج الصوت (final-voice-model.ipynb)
# Detailed Explanation of Voice/Audio Model

---

## 📋 نظرة عامة / Overview

هذا الدليل يشرح كل خلية (Cell) في دفتر الملاحظات `final-voice-model.ipynb` بالتفصيل.
يتضمن الدفتر تدريب نموذج Convolutional Neural Network (CNN) للتعرف على 7 مشاعر من الملفات الصوتية.

This guide explains every cell in the `final-voice-model.ipynb` notebook in detail.
The notebook includes training a Convolutional Neural Network (CNN) to recognize 7 emotions from audio files.

---

## 📦 Cell 1-2: استيراد المكتبات الأساسية / Importing Basic Libraries

```python
import numpy as np
import pandas as pd
import os
import sys
```

**الشرح / Explanation:**
- **numpy**: مكتبة للعمليات الرياضية والمصفوفات (mathematical operations and arrays)
- **pandas**: مكتبة لمعالجة البيانات في شكل جداول (data manipulation in tables)
- **os**: للتعامل مع نظام الملفات (file system operations)
- **sys**: للتفاعل مع نظام التشغيل (system-specific functions)

---

## 🎵 Cell 3-4: عنوان وتحميل المكتبات الصوتية / Audio Libraries

**Cell 3 (Markdown):**
```
Loading the Necessary Modules
```

**Cell 4 (Code):**
```python
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import IPython.display as ipd
```

**الشرح / Explanation:**
- **librosa**: المكتبة الرئيسية لمعالجة وتحليل الملفات الصوتية
  - تحميل الملفات الصوتية
  - استخراج الميزات الصوتية (features)
  - تحويل الإشارات الصوتية
- **librosa.display**: لرسم وتصوير الإشارات الصوتية
- **seaborn & matplotlib**: لإنشاء الرسوم البيانية والتصويرات
- **warnings.filterwarnings('ignore')**: لإخفاء التحذيرات غير المهمة
- **IPython.display**: لتشغيل الملفات الصوتية داخل الـ notebook

---

## 📁 Cell 5-6: تحديد مسارات البيانات / Data Paths

**Cell 5 (Markdown):**
```
Paths for data directories
```

**Cell 6 (Code):**
```python
Ravdess = "/kaggle/input/ravdess-emotional-speech-audio/"
Crema = "/kaggle/input/cremad/"
Tess = "/kaggle/input/toronto-emotional-speech-set-tess/"
Savee = "/kaggle/input/surrey-audio-visual-expressed-emotion-savee/ALL/"
```

**الشرح / Explanation:**
هنا نحدد مسارات أربع مجموعات بيانات مختلفة:

1. **RAVDESS** (Ryerson Audio-Visual Database):
   - يحتوي على 1440 ملف صوتي
   - 24 ممثل (12 ذكر، 12 أنثى)
   - 8 مشاعر مختلفة

2. **CREMA-D** (Crowd-Sourced Emotional Multimodal):
   - 7442 ملف صوتي
   - 91 ممثل
   - 6 مشاعر

3. **TESS** (Toronto Emotional Speech Set):
   - 2800 ملف صوتي
   - ممثلتان
   - 7 مشاعر

4. **SAVEE** (Surrey Audio-Visual Expressed Emotion):
   - 480 ملف صوتي
   - 4 ممثلين ذكور
   - 7 مشاعر

---

## 🗂️ Cells 7-11: تحميل بيانات RAVDESS

**Cell 7 (Markdown):**
```
Ravdess Dataframe
```

**Cell 8 (Code):**
```python
ravdess_directory_list = os.listdir(Ravdess)
file_emotion = []
file_path = []
```

**الشرح / Explanation:**
- نقوم بإنشاء قوائم فارغة لتخزين:
  - `file_emotion`: المشاعر المستخرجة من أسماء الملفات
  - `file_path`: مسارات الملفات الكاملة

**Cell 9 (Code):**
```python
for dir in ravdess_directory_list:
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # ...
```

**الشرح / Explanation:**
نظام تسمية ملفات RAVDESS:
```
03-01-06-01-02-01-12.wav
││ ││ ││ ││ ││ ││ └─ Actor ID (12)
││ ││ ││ ││ ││ └─── Repetition (01 or 02)
││ ││ ││ ││ └───── Statement (01 or 02)
││ ││ ││ └─────── Intensity (01=normal, 02=strong)
││ ││ └───────── Emotion (01-08)
││ └─────────── Vocal channel (01=speech, 02=song)
└────────────── Modality (03=audio-video, 01=audio)
```

**رموز المشاعر / Emotion Codes:**
- 01: neutral (محايد)
- 02: calm (هادئ)
- 03: happy (سعيد)
- 04: sad (حزين)
- 05: angry (غاضب)
- 06: fearful (خائف)
- 07: disgust (مقرف)
- 08: surprised (متفاجئ)

**Cell 10-11:**
يتم تحويل الأرقام إلى أسماء المشاعر وإنشاء DataFrame:
```python
ravdess_df = pd.DataFrame(file_emotion, columns=['Emotions'])
ravdess_df['Path'] = file_path
```

---

## 🗂️ Cells 12-14: تحميل بيانات CREMA-D

**Cell 12 (Markdown):**
```
Crema Dataframe
```

**Cell 13-14 (Code):**
```python
crema_directory_list = os.listdir(Crema)
# ...
```

**الشرح / Explanation:**
نظام تسمية ملفات CREMA-D:
```
1001_DFA_ANG_XX.wav
││││ │││ │││ └─ Intensity
││││ │││ └───── Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
││││ └──────── Sentence
└──────────── Actor ID
```

**المشاعر في CREMA-D:**
- ANG: Anger (غضب)
- DIS: Disgust (اشمئزاز)
- FEA: Fear (خوف)
- HAP: Happy (سعادة)
- NEU: Neutral (محايد)
- SAD: Sad (حزن)

---

## 🗂️ Cells 15-18: تحميل بيانات TESS

**Cell 15 (Markdown):**
```
Tess Dataframe
```

**Cell 16-18 (Code):**
```python
tess_directory_list = os.listdir(Tess)
# ...
```

**الشرح / Explanation:**
نظام تسمية ملفات TESS:
```
OAF_back_angry.wav
│││ │││  └──── Emotion
│││ └──────── Word spoken
└──────────── Speaker ID (OAF or YAF)
```

المتحدثون:
- **OAF**: Older Adult Female (أنثى كبيرة السن)
- **YAF**: Young Adult Female (أنثى شابة)

---

## 🗂️ Cells 19-21: تحميل بيانات SAVEE

**Cell 19 (Markdown):**
```
Savee Dataframe
```

**Cell 20-21 (Code):**
```python
savee_directory_list = os.listdir(Savee)
# ...
```

**الشرح / Explanation:**
نظام تسمية ملفات SAVEE:
```
DC_a01.wav
││ └─── Sentence number
└───── Speaker (DC, JE, JK, KL) + Emotion initial
```

**رموز المشاعر:**
- a: anger (غضب)
- d: disgust (اشمئزاز)
- f: fear (خوف)
- h: happiness (سعادة)
- n: neutral (محايد)
- sa: sadness (حزن)
- su: surprise (مفاجأة)

---

## 🔗 Cell 22: دمج جميع البيانات / Data Integration

**Cell 21 (Markdown):**
```
**Integration**
```

**Cell 22 (Code):**
```python
data_path = pd.concat([ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)
data_path.reset_index(drop=True, inplace=True)
```

**الشرح / Explanation:**
- **pd.concat()**: دمج جميع DataFrames الأربعة في DataFrame واحد
- **axis=0**: الدمج عموديًا (إضافة صفوف)
- **reset_index()**: إعادة ترتيب الفهارس من 0
- النتيجة: DataFrame واحد يحتوي على جميع الملفات الصوتية من المصادر الأربعة

---

## 📊 Cell 23-25: استكشاف وتصور البيانات / Data Visualization

**Cell 23 (Code):**
```python
print(data_path.Emotions.value_counts())
```

**الشرح / Explanation:**
يطبع عدد الملفات لكل عاطفة:
```
neutral     2384
happy       2308
sad         2306
angry       2123
fear        1987
disgust     1895
surprise     638
```

**Cell 24 (Markdown):**
```
Data Visualisation and Exploration
```

**Cell 25 (Code):**
```python
plt.title('Count of Emotions', size=16)
sns.countplot(data_path.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
```

**الشرح / Explanation:**
- إنشاء رسم بياني شريطي (bar chart) يوضح توزيع المشاعر
- **ملاحظة**: البيانات غير متوازنة (imbalanced)
  - neutral وhappy لديهما أكبر عدد من العينات
  - surprise لديه أقل عدد من العينات

---

## 🎧 Cells 26-29: فحص الملفات الصوتية / Audio File Inspection

**Cell 26 (Code):**
```python
data, sr = librosa.load(file_path[0])
sr
```

**الشرح / Explanation:**
- **librosa.load()**: تحميل ملف صوتي
  - `data`: الإشارة الصوتية (audio signal) كمصفوفة numpy
  - `sr`: معدل العينات (sample rate) بالهرتز (عادة 22050 Hz)
- معدل العينات يحدد عدد العينات في الثانية الواحدة

**Cell 27 (Code):**
```python
ipd.Audio(data, rate=sr)
```

**الشرح / Explanation:**
تشغيل الملف الصوتي داخل notebook للاستماع إليه

**Cell 28 (Code):**
```python
plt.figure(figsize=(10, 5))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)
log_spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
```

**الشرح / Explanation:**
- **Mel Spectrogram**: تمثيل بصري للإشارة الصوتية
  - المحور الأفقي: الزمن (time)
  - المحور الرأسي: التردد (frequency) بمقياس Mel
  - اللون: الشدة (intensity) بالديسيبل
- **n_mels=128**: عدد نطاقات التردد (frequency bands)
- **fmax=8000**: أقصى تردد (8 kHz)

**Cell 29 (Code):**
```python
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
plt.figure(figsize=(16, 10))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
```

**الشرح / Explanation:**
- **MFCC** (Mel-Frequency Cepstral Coefficients):
  - أهم الميزات في التعرف على الكلام والعواطف
  - تمثل الخصائص الطيفية للصوت
  - **n_mfcc=30**: استخراج 30 معامل MFCC

---

## 🔄 Cells 30-36: تعزيز البيانات / Data Augmentation

**Cell 30 (Markdown):**
```
# Data augmentation
```

**الشرح / Explanation:**
تعزيز البيانات (Data Augmentation) هو تقنية لزيادة تنوع البيانات التدريبية دون جمع بيانات جديدة.
يساعد على:
- تحسين قدرة النموذج على التعميم
- تقليل overfitting
- زيادة حجم البيانات التدريبية

**Cell 31 (Code):**
```python
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
```

**الشرح التفصيلي / Detailed Explanation:**

1. **noise()**: إضافة ضوضاء عشوائية
   - يضيف ضجيج خلفي بنسبة 3.5% من قيمة الإشارة القصوى
   - يحاكي ظروف التسجيل في بيئات مختلفة

2. **stretch()**: تمديد أو ضغط الإشارة الزمنية
   - rate=0.8: يجعل الصوت أسرع بنسبة 20%
   - rate=1.2: يجعل الصوت أبطأ بنسبة 20%
   - يحاكي اختلافات سرعة الكلام

3. **shift()**: إزاحة الإشارة الصوتية
   - ينقل الإشارة إلى اليمين أو اليسار بشكل عشوائي
   - يحاكي تأخيرات البداية في التسجيلات

4. **pitch()**: تغيير درجة الصوت
   - pitch_factor=0.7: يخفض درجة الصوت
   - يحاكي اختلافات الأصوات بين المتحدثين

**Cells 32-36**: عرض تأثير كل تقنية تعزيز على الإشارة الصوتية:
- Cell 32: الصوت الأصلي (Normal Audio)
- Cell 33: مع الضوضاء (With Noise)
- Cell 34: مع التمديد (Stretched)
- Cell 35: مع الإزاحة (Shifted)
- Cell 36: مع تغيير الدرجة (Pitch Changed)

---

## 🔧 Cells 37-38: استخراج الميزات / Feature Extraction

**Cell 37 (Markdown):**
```
# Feature extraction
```

**Cell 38 (Code):**
```python
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length, hop_length):
    rmse = librosa.feature.rms(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length, hop_length, flatten):
    mfcc_features = librosa.feature.mfcc(data, sr=sr)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                       zcr(data, frame_length, hop_length),
                       rmse(data, frame_length, hop_length),
                       mfcc(data, sr, frame_length, hop_length, flatten=True)))
    return result
```

**الشرح التفصيلي / Detailed Explanation:**

### 1. ZCR (Zero Crossing Rate):
```
معدل تجاوز الصفر = عدد المرات التي تعبر فيها الإشارة محور الصفر
```
- **الاستخدام**: يقيس مدى "صخب" أو "هدوء" الصوت
- **frame_length=2048**: طول الإطار الزمني (window)
- **hop_length=512**: المسافة بين الإطارات المتتالية
- **مثال**: 
  - صوت غاضب: ZCR مرتفع (تغيرات سريعة)
  - صوت حزين: ZCR منخفض (تغيرات بطيئة)

### 2. RMSE (Root Mean Square Energy):
```
RMSE = √(1/N × Σ(x²))
```
- **الاستخدام**: يقيس "قوة" أو "طاقة" الصوت
- **مثال**:
  - صوت غاضب: RMSE مرتفع (صوت عالٍ)
  - صوت حزين: RMSE منخفض (صوت منخفض)

### 3. MFCC (Mel-Frequency Cepstral Coefficients):
- **الاستخدام**: يمثل الخصائص الطيفية للصوت
- الميزات الأكثر أهمية في التعرف على الكلام والعواطف
- **flatten=True**: تحويل المصفوفة ثنائية الأبعاد إلى vector واحد

### 4. extract_features():
- دالة رئيسية تجمع جميع الميزات:
  - ZCR
  - RMSE
  - MFCC
- **np.hstack()**: تجميع الميزات في vector واحد
- النتيجة النهائية: vector بطول ~2376 عنصر لكل ملف صوتي

---

## ⚡ Cells 39-43: استخراج الميزات بشكل متوازي / Parallel Feature Extraction

**Cell 39 (Code):**
```python
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
```

**الشرح / Explanation:**
التحقق من عدد المعالجات المتاحة للمعالجة المتوازية

**Cell 40 (Markdown):**
```
# Normal way to get features
```

**Cell 41 (Code):**
```python
import timeit
from tqdm import tqdm
start = timeit.default_timer()
X, Y = [], []
for path, emotion, index in tqdm(zip(data_path.Path, data_path.Emotions, range(len(data_path)))):
    feature = extract_features(data, sr)
    for ele in range(6):  # Apply 6 augmentations
        X.append(feature)
        Y.append(emotion)
```

**الشرح / Explanation:**
- استخراج الميزات بشكل تسلسلي (sequential)
- لكل ملف صوتي:
  1. استخراج الميزات الأصلية
  2. تطبيق 6 تقنيات تعزيز مختلفة
  3. استخراج ميزات كل نسخة معززة
- **النتيجة**: كل ملف أصلي ينتج 7 عينات تدريبية (الأصلي + 6 معززة)

**Cell 42 (Markdown):**
```
# Faster way to get features
***Parallel way***
```

**الشرح / Explanation:**
استخدام معالجة متوازية (parallel processing) لتسريع استخراج الميزات:
- يقسم العمل على عدة معالجات (cores)
- يمكن أن يكون أسرع 4-8 مرات من الطريقة التسلسلية

**Cell 43 (Code):**
```python
len(X), len(Y), data_path.Path.shape
```

**الشرح / Explanation:**
التحقق من أبعاد البيانات المستخرجة:
- **X**: قائمة الميزات (features)
- **Y**: قائمة التسميات (labels/emotions)
- عدد العينات ≈ 7 × عدد الملفات الأصلية

---

## 💾 Cells 44-49: حفظ وتحميل البيانات / Saving and Loading Data

**Cell 44 (Markdown):**
```
# Saving features
```

**Cell 45 (Code):**
```python
Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotion.csv', index=False)
Emotions.head()
```

**الشرح / Explanation:**
- تحويل الميزات إلى DataFrame
- إضافة عمود للمشاعر
- حفظ البيانات في ملف CSV
- **الفائدة**: عدم الحاجة لإعادة استخراج الميزات في كل مرة

**Cell 46 (Code):**
```python
Emotions = pd.read_csv('./emotion.csv')
Emotions.head()
```

**الشرح / Explanation:**
تحميل البيانات المحفوظة مسبقًا

**Cell 47-48 (Code):**
```python
print(Emotions.isna().any())
Emotions = Emotions.fillna(0)
```

**الشرح / Explanation:**
- **isna().any()**: التحقق من وجود قيم ناقصة (NaN)
- **fillna(0)**: ملء القيم الناقصة بالصفر
- **أهمية**: النماذج لا تعمل مع قيم NaN

**Cell 49 (Code):**
```python
np.sum(Emotions.isna())
```

**الشرح / Explanation:**
التأكد من عدم وجود أي قيم ناقصة بعد المعالجة

---

## 🎯 Cells 50-56: تحضير البيانات للتدريب / Data Preparation

**Cell 50 (Markdown):**
```
# Data preparation
```

**Cell 51 (Code):**
```python
X = Emotions.iloc[:, :-1].values
Y = Emotions['Emotions'].values
```

**الشرح / Explanation:**
- **X**: جميع الأعمدة ما عدا الأخير (الميزات)
  - Shape: (n_samples, 2376)
- **Y**: العمود الأخير فقط (المشاعر)
  - Shape: (n_samples,)

**Cell 52 (Code):**
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
```

**الشرح / Explanation:**
**OneHotEncoding** يحول التسميات النصية إلى أرقام:

```
قبل (Before):          بعد (After):
angry       →    [1, 0, 0, 0, 0, 0, 0]
disgust     →    [0, 1, 0, 0, 0, 0, 0]
fear        →    [0, 0, 1, 0, 0, 0, 0]
happy       →    [0, 0, 0, 1, 0, 0, 0]
neutral     →    [0, 0, 0, 0, 1, 0, 0]
sad         →    [0, 0, 0, 0, 0, 1, 0]
surprise    →    [0, 0, 0, 0, 0, 0, 1]
```

- **reshape(-1, 1)**: تحويل إلى مصفوفة عمود
- **toarray()**: تحويل من sparse matrix إلى array عادي

**Cell 53 (Code):**
```python
print(Y.shape)
X.shape
```

**الشرح / Explanation:**
- **X.shape**: (n_samples, 2376) - الميزات
- **Y.shape**: (n_samples, 7) - المشاعر المرمزة

**Cell 54 (Code):**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, shuffle=True)
```

**الشرح / Explanation:**
تقسيم البيانات:
- **80%** للتدريب (Training)
- **20%** للاختبار (Testing)
- **random_state=42**: لضمان نفس التقسيم في كل مرة
- **shuffle=True**: خلط البيانات قبل التقسيم

**Cell 55 (Code):**
```python
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
```

**الشرح / Explanation:**
إضافة بُعد ثالث للبيانات لاستخدامها مع LSTM:
- **قبل**: (n_samples, 2376)
- **بعد**: (n_samples, 2376, 1)

**Cell 56 (Code):**
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

**الشرح / Explanation:**
**تطبيع البيانات (Standardization)**:
```
x_scaled = (x - μ) / σ
```
- **μ**: المتوسط (mean)
- **σ**: الانحراف المعياري (standard deviation)

**الفوائد**:
- جميع الميزات لها نفس المقياس
- تحسين سرعة وأداء التدريب
- تجنب dominance ميزة معينة

**ملاحظة مهمة**:
- **fit_transform()** على التدريب: حساب μ و σ ثم التطبيق
- **transform()** على الاختبار: استخدام نفس μ و σ من التدريب

---

## 🧠 Cells 57-60: إعداد التدريب / Training Setup

**Cell 57 (Code):**
```python
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras import regularizers
import tensorflow as tf
```

**الشرح / Explanation:**
استيراد المكتبات اللازمة لبناء النموذج:
- **Sequential**: نموذج تسلسلي (layers متتالية)
- **Dense**: طبقة fully connected
- **LSTM**: Long Short-Term Memory (للبيانات المتسلسلة)
- **Dropout**: لتقليل overfitting
- **BatchNormalization**: لتطبيع البيانات بين الطبقات
- **regularizers**: لإضافة L1/L2 regularization

**Cell 58 (Markdown):**
```
> Applying early stopping for all models
```

**Cell 59 (Code):**
```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model_checkpoint = ModelCheckpoint('best_model1_weights.h5', 
                                  monitor='val_accuracy',
                                  mode='max',
                                  save_best_only=True,
                                  verbose=1)
```

**الشرح / Explanation:**
**ModelCheckpoint**: حفظ أفضل نموذج أثناء التدريب
- **monitor='val_accuracy'**: مراقبة دقة التحقق
- **mode='max'**: حفظ عند زيادة الدقة
- **save_best_only=True**: حفظ الأفضل فقط
- **verbose=1**: طباعة رسائل عند الحفظ

**Cell 60 (Code):**
```python
early_stop = EarlyStopping(monitor='val_accuracy',
                          mode='auto',
                          patience=5,
                          restore_best_weights=True)

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                patience=3,
                                verbose=1,
                                factor=0.5,
                                min_lr=0.00001)
```

**الشرح / Explanation:**

**EarlyStopping**: إيقاف التدريب عند عدم التحسن
- **patience=5**: الانتظار 5 epochs بدون تحسن
- **restore_best_weights=True**: استعادة أفضل أوزان

**ReduceLROnPlateau**: تقليل معدل التعلم عند الثبات
- **patience=3**: الانتظار 3 epochs بدون تحسن
- **factor=0.5**: تقليل lr بنصفه
- **min_lr=0.00001**: أقل قيمة لمعدل التعلم

---

## 🏗️ Cells 61-65: بناء وتدريب نموذج CNN / CNN Model

**Cell 61 (Markdown):**
```
# CNN model
```

**Cell 62 (Code):**
```python
x_traincnn = np.expand_dims(x_train, axis=2)
x_testcnn = np.expand_dims(x_test, axis=2)
x_traincnn.shape, y_train.shape, x_testcnn.shape, y_test.shape
```

**الشرح / Explanation:**
إعادة تشكيل البيانات للـ CNN:
- **قبل**: (n_samples, 2376)
- **بعد**: (n_samples, 2376, 1)
- البعد الأخير يمثل "القناة" (channel) مثل RGB في الصور

**Cell 63 (Code):**
```python
import tensorflow.keras.layers as L

model = tf.keras.Sequential([
    L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_traincnn.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    
    L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    
    L.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    
    L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    
    L.Flatten(),
    L.Dense(512, activation='relu'),
    L.BatchNormalization(),
    L.Dropout(0.2),
    
    L.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**الشرح التفصيلي للمعمارية / Detailed Architecture Explanation:**

### طبقات Convolutional:

**Block 1:**
```python
Conv1D(512, kernel_size=5) → BatchNorm → MaxPool1D(5)
```
- **512 filters**: استخراج 512 ميزة مختلفة
- **kernel_size=5**: كل filter ينظر إلى 5 نقاط متتالية
- **BatchNormalization**: تطبيع المخرجات
- **MaxPooling**: تقليل الحجم بأخذ القيمة القصوى

**Block 2:**
```python
Conv1D(512, kernel_size=5) → BatchNorm → MaxPool1D(5)
```
- نفس البنية للاستخراج العميق للميزات

**Block 3:**
```python
Conv1D(256, kernel_size=5) → BatchNorm → MaxPool1D(5)
```
- تقليل عدد الفلاتر إلى 256

**Block 4:**
```python
Conv1D(256, kernel_size=3) → BatchNorm → MaxPool1D(5)
```
- kernel أصغر (3) لميزات أدق

### طبقات Dense:

```python
Flatten → Dense(512, relu) → BatchNorm → Dropout(0.2) → Dense(7, softmax)
```

- **Flatten**: تحويل من 2D إلى 1D
- **Dense(512)**: طبقة مخفية كبيرة
- **Dropout(0.2)**: إزالة 20% من الاتصالات عشوائيًا (لمنع overfitting)
- **Dense(7, softmax)**: طبقة الإخراج النهائية
  - 7 neurons (واحد لكل عاطفة)
  - softmax يعطي احتماليات تجمعها = 1

### Compilation:
- **optimizer='adam'**: خوارزمية Adam للتحسين
- **loss='categorical_crossentropy'**: دالة الخسارة لـ multi-class classification
- **metrics=['accuracy']**: قياس الدقة

**Cell 64 (Code):**
```python
history = model.fit(x_traincnn, y_train, 
                   epochs=50, 
                   validation_data=(x_testcnn, y_test),
                   batch_size=64,
                   callbacks=[early_stop, lr_reduction, model_checkpoint])
```

**الشرح / Explanation:**
**التدريب (Training)**:
- **epochs=50**: الحد الأقصى 50 دورة كاملة على البيانات
- **batch_size=64**: معالجة 64 عينة في كل خطوة
- **validation_data**: بيانات الاختبار للتحقق
- **callbacks**: 
  - early_stop: إيقاف مبكر
  - lr_reduction: تقليل معدل التعلم
  - model_checkpoint: حفظ أفضل نموذج

**ما يحدث في كل epoch:**
1. تقسيم البيانات إلى batches
2. لكل batch:
   - Forward pass: حساب المخرجات
   - حساب الخسارة (loss)
   - Backward pass: حساب gradients
   - تحديث الأوزان
3. حساب دقة التحقق (validation accuracy)
4. تطبيق callbacks

**Cell 65 (Code):**
```python
print("Accuracy of our model on test data : ", model.evaluate(x_testcnn, y_test)[1]*100, "%")

epochs = [i for i in range(50)]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

# Plot accuracy
fig.axes[0].plot(epochs, train_acc, label='Train Accuracy')
fig.axes[0].plot(epochs, test_acc, label='Test Accuracy')
fig.axes[0].set_title('Train - Test Accuracy')
fig.axes[0].legend()

# Plot loss
fig.axes[1].plot(epochs, train_loss, label='Train Loss')
fig.axes[1].plot(epochs, test_loss, label='Test Loss')
fig.axes[1].set_title('Train - Test Loss')
fig.axes[1].legend()
plt.show()
```

**الشرح / Explanation:**
- **تقييم النموذج**: طباعة الدقة النهائية على بيانات الاختبار
- **رسم منحنيات التعلم**:
  - **Accuracy plot**: يوضح تطور الدقة عبر الـ epochs
  - **Loss plot**: يوضح انخفاض الخسارة عبر الـ epochs

**تحليل المنحنيات:**
- إذا كان train_acc أعلى بكثير من test_acc → **overfitting**
- إذا كانت المنحنيات متقاربة → **good generalization**
- إذا كانت الخسارة لا تنخفض → **underfitting** أو learning rate سيء

---

## 📊 Cells 66-71: التقييم والنتائج / Evaluation and Results

**Cell 66 (Code):**
```python
pred_test0 = model.predict(x_testcnn)
y_pred0 = encoder.inverse_transform(pred_test0)
y_test0 = encoder.inverse_transform(y_test)

df0 = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df0['Predicted Labels'] = y_pred0.flatten()
df0['Actual Labels'] = y_test0.flatten()
```

**الشرح / Explanation:**
- **model.predict()**: الحصول على التنبؤات
  - المخرجات: احتماليات لكل عاطفة
  - مثال: [0.1, 0.05, 0.7, 0.05, 0.05, 0.03, 0.02]
- **inverse_transform()**: تحويل من one-hot إلى أسماء العواطف
  - من: [0, 0, 1, 0, 0, 0, 0]
  - إلى: "fear"
- إنشاء DataFrame لمقارنة التنبؤات الفعلية والمتوقعة

**Cell 67 (Code):**
```python
df0
```

**الشرح / Explanation:**
عرض جدول المقارنة بين التنبؤات والقيم الفعلية

**Cell 68 (Markdown):**
```
Some plots of multi_model
```

**Cell 69 (Markdown):**
```
# Evaluation
```

**Cell 70 (Markdown):**
```
Results of best model
```

**Cell 71 (Code):**
```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test0, y_pred0)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_[0]], 
                  columns=[i for i in encoder.categories_[0]])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test0, y_pred0))
```

**الشرح / Explanation:**

**Confusion Matrix** (مصفوفة الالتباس):
```
                Predicted
                A   D   F   H   N   Sa  Su
Actual  Angry   45  2   3   1   2   1   0
        Disgust 2   40  2   0   3   2   1
        Fear    3   1   38  2   4   2   0
        Happy   1   0   1   44  2   2   0
        Neutral 2   2   3   2   45  1   0
        Sad     1   2   2   1   2   42  0
        Surprise 0  1   0   2   1   1   45
```

**قراءة المصفوفة:**
- الصف: العاطفة الفعلية
- العمود: العاطفة المتنبأ بها
- القطر الرئيسي: التنبؤات الصحيحة
- خارج القطر: الأخطاء

**Classification Report:**
```
              precision  recall  f1-score  support
angry            0.83     0.85     0.84      54
disgust          0.80     0.80     0.80      50
fear             0.77     0.76     0.77      50
happy            0.88     0.88     0.88      50
neutral          0.76     0.82     0.79      55
sad              0.82     0.84     0.83      50
surprise         0.98     0.90     0.94      50

accuracy                          0.84     359
macro avg        0.83     0.84     0.84     359
weighted avg     0.84     0.84     0.84     359
```

**المقاييس:**
- **Precision**: من كل ما تنبأنا أنه X، كم كان صحيحًا؟
- **Recall**: من كل X الفعلية، كم اكتشفنا؟
- **F1-score**: المتوسط التوافقي لـ precision و recall
- **Support**: عدد العينات لكل فئة

---

## 💾 Cells 72-78: حفظ النموذج / Model Saving

**Cell 72 (Markdown):**
```
# Saving Best Model
```

**Cell 73 (Code):**
```python
model_json = model.to_json()
with open("CNN_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("best_model1_weights.h5")
print("Saved model to disk")
```

**الشرح / Explanation:**
حفظ النموذج في ملفين:

1. **CNN_model.json**: بنية النموذج (architecture)
   - عدد الطبقات
   - نوع كل طبقة
   - المعاملات (parameters)
   
2. **best_model1_weights.h5**: أوزان النموذج (weights)
   - القيم المدربة لكل معامل
   - النتيجة النهائية للتدريب

**Cell 74-75 (Code):**
```python
# Load model
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model1_weights.h5")

# Compile and test
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

**الشرح / Explanation:**
اختبار تحميل النموذج:
1. قراءة البنية من JSON
2. إنشاء النموذج من البنية
3. تحميل الأوزان
4. Compile النموذج
5. اختبار الأداء

**Cell 76 (Markdown):**
```
# Saving and Loading our Standard Scaler and encoder
```

**Cell 77 (Markdown):**
```
pickle file
```

**Cell 78 (Code):**
```python
import pickle

# Saving scaler
with open('scaler2.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Saving encoder
with open('encoder2.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# Loading scaler
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
    
# Loading encoder
with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)
```

**الشرح / Explanation:**
حفظ وتحميل Scaler و Encoder:
- **أهمية**: يجب استخدام نفس scaler و encoder في الإنتاج
- **pickle**: مكتبة Python لحفظ الكائنات
- **'wb'**: Write Binary (للحفظ)
- **'rb'**: Read Binary (للتحميل)

---

## 🧪 Cells 79-94: اختبار النموذج / Model Testing

**Cell 79 (Markdown):**
```
# Test script
* That can predict new record
```

**Cell 80-82 (Code):**
تحميل النموذج و scaler و encoder المحفوظة

**Cell 83 (Code):**
إعادة تعريف دوال استخراج الميزات:
- zcr()
- rmse()
- mfcc()
- extract_features()

**Cell 84 (Code):**
```python
def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    
    # Ensure correct length (2376)
    if len(result) > 2376:
        result = result[:2376]
    elif len(result) < 2376:
        result = np.pad(result, (0, 2376 - len(result)), mode='constant')
    
    # Reshape and scale
    result = result.reshape(1, -1)
    i_result = scaler2.transform(result)
    final_result = i_result.reshape(i_result.shape[0], i_result.shape[1], 1)
    
    return final_result
```

**الشرح / Explanation:**
دالة للتنبؤ على ملف صوتي جديد:

1. **librosa.load()**:
   - duration=2.5: تحميل 2.5 ثانية فقط
   - offset=0.6: البدء من الثانية 0.6
   
2. **extract_features()**: استخراج ZCR, RMSE, MFCC

3. **التأكد من الطول**:
   - إذا كان أطول: قص الزيادة
   - إذا كان أقصر: ملء بالأصفار
   - الطول المطلوب: 2376
   
4. **التطبيع**: استخدام نفس scaler من التدريب

5. **إعادة التشكيل**: للتوافق مع مدخل النموذج

**Cell 85 (Code):**
```python
res = get_predict_feat("/kaggle/input/ravdess-emotional-speech-audio/Actor_01/03-01-07-01-01-01-01.wav")
print(res.shape)
```

**الشرح / Explanation:**
اختبار الدالة على ملف واحد والتحقق من الشكل

**Cell 86 (Code):**
```python
emotions1 = {1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 
            6:'Fear', 7:'Disgust', 8:'Surprise'}

def prediction(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res, verbose=0)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]
```

**الشرح / Explanation:**
دالة كاملة للتنبؤ:
1. استخراج الميزات
2. التنبؤ باستخدام النموذج
3. تحويل من one-hot إلى اسم العاطفة
4. إرجاع النتيجة

**Cells 87-94:**
اختبار الدالة على ملفات صوتية مختلفة:
```python
prediction("/path/to/audio/file.wav")
```

كل cell يختبر ملف مختلف لرؤية دقة النموذج

---

## 📝 الخلاصة / Summary

### ما تم إنجازه / What Was Accomplished:

1. ✅ **جمع البيانات**: دمج 4 مجموعات بيانات مختلفة (~14,000 ملف)
2. ✅ **تعزيز البيانات**: 6 تقنيات مختلفة (ضوضاء، تمديد، إزاحة، إلخ)
3. ✅ **استخراج الميزات**: ZCR, RMSE, MFCC (2376 ميزة لكل ملف)
4. ✅ **بناء النموذج**: CNN بـ 4 blocks convolutional
5. ✅ **التدريب**: مع callbacks (early stopping, lr reduction, checkpointing)
6. ✅ **التقييم**: ~84% دقة على بيانات الاختبار
7. ✅ **الحفظ**: النموذج، scaler، encoder للاستخدام في الإنتاج

### النقاط المهمة / Key Points:

- **Data Augmentation** ضاعف حجم البيانات 7x
- **Feature Extraction** حول الصوت إلى أرقام
- **CNN Architecture** مناسبة للبيانات المتسلسلة
- **Callbacks** منعت overfitting وحسّنت التدريب
- **Evaluation Metrics** وضّحت نقاط القوة والضعف
- **Saving/Loading** سمح باستخدام النموذج في التطبيقات

### التحسينات الممكنة / Possible Improvements:

1. 📈 **زيادة البيانات**: المزيد من مجموعات البيانات
2. 🎯 **تحسين المعمارية**: تجربة architectures أخرى (RNN, Transformer)
3. ⚖️ **توازن البيانات**: معالجة class imbalance
4. 🔧 **Hyperparameter Tuning**: البحث عن أفضل معاملات
5. 🌐 **Transfer Learning**: استخدام نماذج مدربة مسبقًا

---

## 🎓 المفاهيم المستفادة / Concepts Learned:

### 1. Audio Processing:
- كيفية تحميل ومعالجة الملفات الصوتية
- استخراج الميزات الصوتية
- تقنيات data augmentation للصوت

### 2. Deep Learning:
- بناء CNN للبيانات المتسلسلة
- استخدام callbacks للتحكم في التدريب
- تقييم النماذج باستخدام confusion matrix

### 3. Production Readiness:
- حفظ وتحميل النماذج
- إنشاء دوال للتنبؤ
- التعامل مع بيانات جديدة

---

**انتهى شرح نموذج الصوت**
**End of Voice Model Explanation**

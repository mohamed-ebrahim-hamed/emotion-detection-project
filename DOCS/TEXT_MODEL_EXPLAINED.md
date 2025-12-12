# شرح تفصيلي لنموذج النص (text-model.ipynb)
# Detailed Explanation of Text Model

---

## 📋 نظرة عامة / Overview

هذا الدليل يشرح كل خلية (Cell) في دفتر الملاحظات `text-model.ipynb` بالتفصيل.
يتضمن الدفتر تدريب نموذج DistilBERT (Transformer) للتعرف على 28 عاطفة مختلفة من النصوص.

This guide explains every cell in the `text-model.ipynb` notebook in detail.
The notebook includes training a DistilBERT (Transformer) model to recognize 28 different emotions from text.

---

## 📊 معلومات عن مجموعة البيانات / Dataset Information

### GoEmotions Dataset:
- **المصدر**: Google Research
- **الحجم**: 58,000 تعليق من Reddit
- **العواطف**: 28 عاطفة مختلفة
- **اللغة**: الإنجليزية
- **النوع**: Multi-label classification (يمكن للنص أن يحتوي على أكثر من عاطفة)

### العواطف الـ 28 / The 28 Emotions:
1. admiration (إعجاب)
2. amusement (تسلية)
3. anger (غضب)
4. annoyance (انزعاج)
5. approval (موافقة)
6. caring (اهتمام)
7. confusion (ارتباك)
8. curiosity (فضول)
9. desire (رغبة)
10. disappointment (خيبة أمل)
11. disapproval (رفض)
12. disgust (اشمئزاز)
13. embarrassment (إحراج)
14. excitement (حماس)
15. fear (خوف)
16. gratitude (امتنان)
17. grief (حزن شديد)
18. joy (فرح)
19. love (حب)
20. nervousness (توتر)
21. neutral (محايد)
22. optimism (تفاؤل)
23. pride (فخر)
24. realization (إدراك)
25. relief (ارتياح)
26. remorse (ندم)
27. sadness (حزن)
28. surprise (مفاجأة)

---

## 📦 Cell 1: استيراد المكتبات الأساسية / Importing Basic Libraries

```python
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

**الشرح / Explanation:**
- **numpy**: للعمليات الرياضية والمصفوفات
- **pandas**: لمعالجة البيانات في جداول
- **os.walk()**: لاستعراض جميع الملفات في مجلد البيانات
- هذا الكود يطبع مسارات جميع ملفات البيانات المتاحة

**الهدف**: التأكد من وجود ملفات البيانات والتعرف على بنية المجلدات

---

## 🔧 Cell 2: تثبيت المكتبات المطلوبة / Installing Required Libraries

```python
!pip install transformers==4.30.2 -q
```

**الشرح / Explanation:**
- **transformers**: مكتبة Hugging Face للعمل مع نماذج Transformer
- **4.30.2**: إصدار محدد لضمان التوافق
- **-q**: وضع هادئ (quiet mode) لتقليل المخرجات

**ما هي Transformers؟**
- معمارية ثورية في معالجة اللغات الطبيعية (NLP)
- تعتمد على آلية Attention
- أساس نماذج مثل BERT، GPT، T5

**لماذا DistilBERT؟**
- نسخة مصغرة من BERT
- 40% أسرع
- 60% أصغر في الحجم
- 97% من أداء BERT الأصلي

---

## 📚 Cell 3: استيراد مكتبات Deep Learning / Importing Deep Learning Libraries

```python
import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
```

**الشرح / Explanation:**

### PyTorch Components:
- **torch**: إطار عمل Deep Learning من Facebook
- **TensorDataset**: لتغليف البيانات في dataset
- **DataLoader**: لتحميل البيانات في batches

### Transformers Components:
- **DistilBertTokenizerFast**: 
  - يحول النص إلى أرقام (tokens)
  - نسخة سريعة مكتوبة بلغة Rust
  - يتعامل مع الكلمات الفرعية (subwords)

- **DistilBertForSequenceClassification**:
  - نموذج DistilBERT مجهز للتصنيف
  - يحتوي على DistilBERT + طبقة classification

**الفرق بين TensorFlow و PyTorch:**
| TensorFlow | PyTorch |
|------------|---------|
| من Google | من Facebook |
| Static graphs | Dynamic graphs |
| Production focus | Research focus |
| استخدمناه للصوت | نستخدمه للنص |

---

## 📂 Cell 4: تحميل البيانات / Loading Data

```python
DATA_DIR = "/kaggle/input/goemotions/data"

train_df = pd.read_csv(f"{DATA_DIR}/train.tsv", sep="\t", header=None, 
                       names=["text","labels","id"])
dev_df   = pd.read_csv(f"{DATA_DIR}/dev.tsv", sep="\t", header=None, 
                       names=["text","labels","id"])
test_df  = pd.read_csv(f"{DATA_DIR}/test.tsv", sep="\t", header=None, 
                       names=["text","labels","id"])

print(f"Train: {len(train_df)} samples")
print(f"Dev:   {len(dev_df)} samples")
print(f"Test:  {len(test_df)} samples")
```

**الشرح / Explanation:**

### بنية البيانات / Data Structure:
البيانات في ملفات TSV (Tab-Separated Values):

```
text                                    labels      id
I love this movie!                      17,18       123
This is confusing and scary             6,14        124
Great job, well done!                   0,4         125
```

### الأعمدة / Columns:
1. **text**: النص المراد تصنيفه
2. **labels**: أرقام العواطف مفصولة بفواصل (comma-separated)
   - مثال: "0,17,21" = admiration + joy + optimism
3. **id**: معرف فريد لكل نص

### التقسيم / Split:
- **train_df**: للتدريب (~80% من البيانات)
- **dev_df**: للتحقق أثناء التدريب (~10%)
- **test_df**: للتقييم النهائي (~10%)

**ملاحظة مهمة**: 
- **Multi-label classification**: النص الواحد يمكن أن يحتوي على أكثر من عاطفة
- مثال: "I'm so happy but also surprised!" → happy + surprise

---

## 🔢 Cell 5: تحويل التسميات إلى Multi-hot Encoding / Converting Labels

```python
NUM_LABELS = 28

def to_multihot(label_str):
    indices = list(map(int, label_str.split(",")))
    arr = np.zeros(NUM_LABELS)
    arr[indices] = 1
    return arr

train_df["multihot"] = train_df["labels"].apply(to_multihot)
dev_df["multihot"]   = dev_df["labels"].apply(to_multihot)
test_df["multihot"]  = test_df["labels"].apply(to_multihot)

print("Example:")
print(f"Original: {train_df.iloc[0]['labels']}")
print(f"Multi-hot: {train_df.iloc[0]['multihot']}")
```

**الشرح التفصيلي / Detailed Explanation:**

### ما هو Multi-hot Encoding؟
تحويل من نص إلى مصفوفة binary:

```
قبل (Before):          بعد (After):
"0,17,21"      →    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0]
                     ↑                                 ↑           ↑
                   emotion 0                      emotion 17   emotion 21
```

### خطوات الدالة / Function Steps:

1. **split(",")**: تقسيم النص إلى قائمة
   ```python
   "0,17,21" → ["0", "17", "21"]
   ```

2. **map(int, ...)**: تحويل إلى أرقام
   ```python
   ["0", "17", "21"] → [0, 17, 21]
   ```

3. **np.zeros(28)**: إنشاء مصفوفة أصفار
   ```python
   [0, 0, 0, 0, ..., 0]  # 28 عنصر
   ```

4. **arr[indices] = 1**: تفعيل المواقع المطلوبة
   ```python
   arr[0] = 1
   arr[17] = 1
   arr[21] = 1
   ```

### الفرق بين One-hot و Multi-hot:

**One-hot** (نموذج الصوت):
```
فقط واحد مفعّل: [0, 0, 1, 0, 0, 0, 0]
                            ↑
                    عاطفة واحدة فقط
```

**Multi-hot** (نموذج النص):
```
عدة مفعّلة: [1, 0, 0, 0, 1, 0, 1]
             ↑           ↑     ↑
        عدة عواطف في نفس الوقت
```

---

## 🔤 Cell 6: إنشاء Tokenizer / Creating Tokenizer

```python
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode(batch):
    return tokenizer(
        batch["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128
    )
```

**الشرح / Explanation:**

### ما هو Tokenizer؟
يحول النص إلى أرقام يفهمها النموذج:

```
Input Text:
"I love this movie!"

↓ Tokenization ↓

Tokens:
["i", "love", "this", "movie", "!"]

↓ Convert to IDs ↓

Token IDs:
[1045, 2293, 2023, 3185, 999]

↓ Add Special Tokens ↓

Final IDs:
[101, 1045, 2293, 2023, 3185, 999, 102]
  ↑                                  ↑
[CLS]                              [SEP]
```

### Special Tokens:
- **[CLS]** (101): بداية النص (Classification token)
- **[SEP]** (102): نهاية النص (Separator token)
- **[PAD]** (0): للتعبئة (Padding)

### المعاملات / Parameters:

**truncation=True**:
```
إذا كان النص أطول من 128 كلمة:
"This is a very very ... very long text"
                            ↓
"This is a very very ... [قص الباقي]"
```

**padding="max_length"**:
```
إذا كان النص أقصر من 128 كلمة:
[101, 1045, 2293, 102, 0, 0, 0, ... , 0]
                       ↑
              ملء بالأصفار حتى 128
```

**max_length=128**:
- الحد الأقصى لطول النص
- توازن بين الأداء والسرعة
- BERT الأصلي يدعم حتى 512

### لماذا "uncased"؟
- **uncased**: يحول كل شيء إلى lowercase
  - "Hello" → "hello"
  - "HELLO" → "hello"
- **cased**: يحفظ الحالة
  - "Hello" يبقى "Hello"
- **uncased** أسرع وغالبًا أفضل للمشاعر

---

## 🔄 Cell 7: تطبيق Tokenization / Applying Tokenization

```python
train_enc = encode(train_df)
dev_enc   = encode(dev_df)
test_enc  = encode(test_df)
```

**الشرح / Explanation:**
تطبيق دالة encode على جميع البيانات:
- تحويل كل نص إلى token IDs
- إنشاء attention masks
- تطبيق padding و truncation

**المخرجات / Outputs:**
```python
{
    'input_ids': [[101, 1045, 2293, ..., 0, 0, 0],
                  [101, 2023, 2003, ..., 0, 0, 0],
                  ...],
    'attention_mask': [[1, 1, 1, ..., 0, 0, 0],
                       [1, 1, 1, ..., 0, 0, 0],
                       ...]
}
```

### ما هو Attention Mask؟
يخبر النموذج أين يركز:

```
input_ids:      [101, 1045, 2293, 102, 0, 0, 0, 0]
attention_mask: [  1,    1,    1,   1, 0, 0, 0, 0]
                 ↑    ↑    ↑    ↑   ↑
              انتبه لهذه    تجاهل هذه (padding)
```

---

## 📦 Cell 8: إنشاء PyTorch Datasets / Creating PyTorch Datasets

```python
train_dataset = TensorDataset(
    torch.tensor(train_enc["input_ids"]),
    torch.tensor(train_enc["attention_mask"]),
    torch.tensor(np.vstack(train_df["multihot"].values), dtype=torch.float)
)

dev_dataset = TensorDataset(
    torch.tensor(dev_enc["input_ids"]),
    torch.tensor(dev_enc["attention_mask"]),
    torch.tensor(np.vstack(dev_df["multihot"].values), dtype=torch.float)
)

test_dataset = TensorDataset(
    torch.tensor(test_enc["input_ids"]),
    torch.tensor(test_enc["attention_mask"]),
    torch.tensor(np.vstack(test_df["multihot"].values), dtype=torch.float)
)
```

**الشرح / Explanation:**

### ما هو TensorDataset؟
يجمع البيانات معًا في dataset واحد:

```
Dataset = {
    Input IDs:       [101, 1045, 2293, ...]
    Attention Mask:  [1, 1, 1, ...]
    Labels:          [1, 0, 0, 0, 1, 0, ...]
}
```

### المكونات / Components:

1. **torch.tensor(train_enc["input_ids"])**:
   - Token IDs لكل نص
   - Shape: (n_samples, 128)

2. **torch.tensor(train_enc["attention_mask"])**:
   - Attention masks
   - Shape: (n_samples, 128)

3. **torch.tensor(..., dtype=torch.float)**:
   - Multi-hot labels
   - Shape: (n_samples, 28)
   - **dtype=torch.float**: مهم لـ BCE Loss

### np.vstack:
تحويل قائمة من arrays إلى مصفوفة واحدة:

```python
Before:
[[1,0,0], [0,1,0], [1,1,0]]  # قائمة من 3 arrays

After vstack:
[[1,0,0],
 [0,1,0],
 [1,1,0]]  # مصفوفة واحدة (3, 3)
```

---

## 🔄 Cell 9: إنشاء DataLoaders / Creating DataLoaders

```python
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size)
```

**الشرح / Explanation:**

### ما هو DataLoader؟
يقسم البيانات إلى batches ويحملها بكفاءة:

```
Dataset (1000 samples)
         ↓
DataLoader (batch_size=16)
         ↓
Batch 1: [samples 1-16]
Batch 2: [samples 17-32]
Batch 3: [samples 33-48]
...
Batch 63: [samples 993-1000]  # آخر batch قد يكون أصغر
```

### المعاملات / Parameters:

**batch_size=16**:
- عدد العينات في كل batch
- batch أصغر:
  - ✅ يحتاج ذاكرة أقل
  - ❌ تدريب أبطأ
- batch أكبر:
  - ✅ تدريب أسرع
  - ❌ يحتاج ذاكرة أكثر
- **16** توازن جيد لـ DistilBERT

**shuffle=True** (للتدريب فقط):
```
قبل الخلط:
[angry, angry, happy, happy, sad, sad, ...]
         ↓
بعد الخلط:
[happy, angry, sad, happy, angry, sad, ...]
```

**لماذا الخلط مهم؟**
- يمنع النموذج من تعلم ترتيب البيانات
- يحسن التعميم (generalization)
- **لا نخلط** dev/test لأننا نقيّم فقط

---

## 🤖 Cell 10: إنشاء النموذج / Creating the Model

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

model.to(device)
```

**الشرح التفصيلي / Detailed Explanation:**

### 1. اختيار الجهاز / Device Selection:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- **cuda**: بطاقة رسومات NVIDIA (GPU)
  - ✅ أسرع 10-100x
  - استخدام الذاكرة الموازية
  - مناسب للتدريب
  
- **cpu**: المعالج العادي
  - ❌ أبطأ بكثير
  - استخدام ذاكرة RAM
  - مناسب للاستنتاج فقط

### 2. تحميل النموذج / Loading Model:

**from_pretrained("distilbert-base-uncased")**:
- تحميل نموذج مدرب مسبقًا (pre-trained)
- **مدرب على**: Wikipedia + BookCorpus
- **يعرف**: قواعد اللغة، معاني الكلمات، السياق

**ما هو Pre-training؟**
```
Phase 1: Pre-training (ملايين النصوص)
↓
تعلم اللغة بشكل عام

Phase 2: Fine-tuning (مجموعة بياناتنا)
↓
تعلم مهمة محددة (تصنيف المشاعر)
```

**num_labels=28**:
- عدد العواطف المطلوب التنبؤ بها
- يضيف طبقة classification بـ 28 output

**problem_type="multi_label_classification"**:
- يخبر النموذج أن:
  - ✅ عدة تسميات ممكنة لنفس النص
  - ✅ استخدام Sigmoid بدل Softmax
  - ✅ استخدام BCE Loss بدل Cross-Entropy

### 3. نقل إلى الجهاز / Move to Device:
```python
model.to(device)
```
- نقل جميع أوزان النموذج إلى GPU
- ضروري قبل التدريب

### معمارية DistilBERT / DistilBERT Architecture:

```
Input Text: "I love this!"
     ↓
Tokenizer: [101, 1045, 2293, 2023, 999, 102]
     ↓
Embeddings (768 dimensions per token)
     ↓
Transformer Layers (6 layers)
├── Multi-Head Attention
├── Feed Forward Network
├── Layer Normalization
└── Residual Connections
     ↓
[CLS] Token Output (768 dimensions)
     ↓
Classification Head (768 → 28)
     ↓
Output: [0.8, 0.1, 0.05, ..., 0.3]  # 28 probabilities
```

---

## ⚙️ Cell 11: إعداد Optimizer و Loss / Setting up Optimizer and Loss

```python
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

loss_fn = torch.nn.BCEWithLogitsLoss()
```

**الشرح / Explanation:**

### 1. Optimizer: AdamW

**ما هو AdamW؟**
- نسخة محسنة من Adam (Adaptive Moment Estimation)
- **W** = Weight Decay (تنظيم الأوزان)
- الأفضل لنماذج Transformer

**كيف يعمل؟**
```
في كل خطوة تدريب:
1. حساب gradient (∂Loss/∂W)
2. حساب moving average للـ gradient
3. حساب moving average للـ gradient²
4. تحديث الأوزان باستخدام adaptive learning rate
```

**lr=2e-5 (0.00002)**:
- معدل تعلم صغير جدًا
- مهم لـ Fine-tuning:
  - النموذج مدرب مسبقًا
  - نريد تعديلات صغيرة فقط
  - تجنب "نسيان" ما تعلمه

**معدلات تعلم نموذجية:**
- Training from scratch: 0.001 - 0.01
- Fine-tuning BERT: 1e-5 - 5e-5
- Fine-tuning DistilBERT: 2e-5 - 3e-5

### 2. Loss Function: BCEWithLogitsLoss

**BCE** = Binary Cross-Entropy

**لماذا BCEWithLogitsLoss؟**
لأننا في multi-label classification:

```
Softmax (single-label):
[0.1, 0.7, 0.2] → يجب أن تجمع إلى 1
فقط واحد صحيح

Sigmoid (multi-label):
[0.8, 0.1, 0.9] → كل واحد مستقل
عدة يمكن أن تكون صحيحة
```

**الصيغة الرياضية:**
```
BCE = -1/N Σ [y·log(σ(x)) + (1-y)·log(1-σ(x))]

حيث:
- y: التسمية الفعلية (0 أو 1)
- σ(x): Sigmoid(x) = 1/(1+e^(-x))
- N: عدد التسميات (28)
```

**WithLogits**:
- يطبق Sigmoid داخليًا
- أكثر استقرارًا عدديًا
- أسرع في الحساب

**مثال:**
```python
# التنبؤ
predictions = [2.1, -0.5, 3.2]  # logits (قبل sigmoid)

# التسميات الفعلية
labels = [1, 0, 1]

# BCEWithLogitsLoss يطبق:
1. Sigmoid: [0.89, 0.38, 0.96]
2. حساب BCE: -[1·log(0.89) + 0·log(0.62) + 1·log(0.96)]
3. النتيجة: 0.15 (loss منخفض = جيد)
```

---

## 🏋️ Cell 12: حلقة التدريب / Training Loop

```python
from tqdm import tqdm

epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids, attention_masks, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
```

**الشرح التفصيلي / Detailed Explanation:**

### البنية العامة / Overall Structure:
```
for epoch in [1, 2]:
    for batch in train_loader:
        1. تحميل البيانات
        2. Forward pass
        3. حساب Loss
        4. Backward pass
        5. تحديث الأوزان
```

### خطوة بخطوة / Step by Step:

**1. model.train()**:
```python
model.train()
```
- تفعيل وضع التدريب
- يفعّل Dropout و BatchNormalization
- عكس `model.eval()` للتقييم

**2. تحميل البيانات على GPU:**
```python
input_ids, attention_masks, labels = [b.to(device) for b in batch]
```
- نقل كل batch إلى GPU
- ضروري للاستفادة من GPU

**3. مسح Gradients السابقة:**
```python
optimizer.zero_grad()
```
- PyTorch يجمع gradients
- يجب مسحها قبل كل backward pass

**4. Forward Pass:**
```python
outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
```

**ما يحدث داخليًا:**
```
input_ids → Embeddings
          ↓
    Transformer Layers (6x)
          ↓
    Classification Head
          ↓
    Logits (28 values)
          ↓
    Sigmoid + BCE Loss (لأننا مررنا labels)
          ↓
    outputs = {
        'loss': tensor(0.423),
        'logits': tensor([2.1, -0.5, ...])
    }
```

**5. استخراج Loss:**
```python
loss = outputs.loss
total_loss += loss.item()
```
- `.loss`: قيمة الخسارة (tensor)
- `.item()`: تحويل من tensor إلى number

**6. Backward Pass (حساب Gradients):**
```python
loss.backward()
```

**ما يحدث:**
```
Loss = 0.423
    ↓
حساب ∂Loss/∂W لكل weight في النموذج
    ↓
تخزين gradients في W.grad
```

**7. تحديث الأوزان:**
```python
optimizer.step()
```

**خوارزمية AdamW:**
```python
for each weight W:
    W = W - lr * gradient
    (مع adaptive learning rate)
```

### لماذا epochs=2 فقط؟

**Fine-tuning نماذج Pre-trained:**
- النموذج يعرف اللغة بالفعل
- نحتاج فقط تعديلات صغيرة
- **2-4 epochs** كافية عادة
- **أكثر** قد يسبب overfitting

**Training from scratch:**
- قد نحتاج 50-100 epoch
- لأن النموذج يبدأ من الصفر

### tqdm:
```
Training Epoch 1: 100%|████████| 2500/2500 [15:23<00:00, 2.71it/s]
```
- شريط تقدم جميل
- يوضح الوقت المتبقي
- السرعة (iterations per second)

---

## 📊 Cell 13: التحقق / Validation

```python
model.eval()
val_loss = 0

with torch.no_grad():
    for batch in tqdm(dev_loader, desc="Validation"):
        input_ids, attention_masks, labels = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels
        )
        
        val_loss += outputs.loss.item()

avg_val_loss = val_loss / len(dev_loader)
print(f"Validation Loss: {avg_val_loss:.4f}")
```

**الشرح / Explanation:**

### 1. model.eval():
```python
model.eval()
```
- تفعيل وضع التقييم
- **يعطّل**:
  - Dropout: إبقاء جميع الـ neurons
  - BatchNorm: استخدام إحصائيات ثابتة
- النتائج أكثر استقرارًا

### 2. torch.no_grad():
```python
with torch.no_grad():
    # ... validation code ...
```

**الفائدة:**
- لا نحتاج gradients للتحقق
- يوفر الذاكرة (~50%)
- يسرع الحسابات

**الفرق:**
```python
# مع gradients (التدريب)
memory_used = 8 GB
time = 10 seconds

# بدون gradients (التحقق)
memory_used = 4 GB
time = 5 seconds
```

### 3. حساب Validation Loss:

**لماذا مهم؟**
```
Training Loss    Validation Loss    التشخيص
     ↓                 ↓
   0.2               0.25           ✅ Good (close)
   0.1               0.3            ⚠️  Overfitting
   0.4               0.4            ⚠️  Underfitting
```

**Overfitting**:
- النموذج حفظ بيانات التدريب
- لا يعمم جيدًا على بيانات جديدة

**Underfitting**:
- النموذج لم يتعلم بشكل كافٍ
- أداء سيء على كلا المجموعتين

---

## 💾 Cell 14: حفظ النموذج / Saving the Model

```python
SAVE_PATH = "/kaggle/working/emotion_model"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Model Saved →", SAVE_PATH)
```

**الشرح / Explanation:**

### ما يتم حفظه / What Gets Saved:

**model.save_pretrained()** يحفظ:
```
emotion_model/
├── config.json           # إعدادات النموذج
├── pytorch_model.bin     # الأوزان المدربة (أو model.safetensors)
└── special_tokens_map.json  # tokens خاصة
```

**tokenizer.save_pretrained()** يحفظ:
```
emotion_model/
├── tokenizer_config.json  # إعدادات tokenizer
├── vocab.txt             # المفردات (30,000+ كلمة)
└── tokenizer.json        # tokenizer كامل
```

### محتوى الملفات / File Contents:

**config.json:**
```json
{
  "model_type": "distilbert",
  "num_labels": 28,
  "problem_type": "multi_label_classification",
  "vocab_size": 30522,
  "hidden_size": 768,
  "num_hidden_layers": 6,
  "num_attention_heads": 12,
  ...
}
```

**pytorch_model.bin:**
- ملف binary يحتوي على جميع الأوزان
- الحجم: ~250 MB لـ DistilBERT
- يمكن تحميله مباشرة بـ `from_pretrained()`

### كيفية التحميل لاحقًا / How to Load Later:

```python
# في app.py أو أي مكان آخر
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("/path/to/emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("/path/to/emotion_model")

# الآن جاهز للاستخدام!
text = "I love this!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

---

## 📝 ملخص عملية التدريب / Training Process Summary

### تدفق البيانات الكامل / Complete Data Flow:

```
1. البيانات الخام / Raw Data:
   "I love this movie!" → GoEmotions dataset

2. التحضير / Preprocessing:
   "I love this movie!" → [101, 1045, 2293, 2023, 3185, 102]
   Labels: "17,18" → [0,0,...,1,1,...,0] (28 dims)

3. Batching:
   16 نص → Batch

4. النموذج / Model:
   Batch → DistilBERT → Logits (16, 28)

5. Loss:
   Logits + Labels → BCEWithLogitsLoss → 0.423

6. Optimization:
   Loss → Gradients → Update Weights

7. التكرار / Repeat:
   حتى نهاية جميع batches → Epoch
   حتى نهاية جميع epochs → Training Complete

8. الحفظ / Save:
   Model + Tokenizer → Disk
```

---

## 🎯 الاستخدام العملي / Practical Usage

### مثال كامل للتنبؤ / Complete Prediction Example:

```python
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 1. تحميل النموذج / Load model
model = DistilBertForSequenceClassification.from_pretrained("./emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion_model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. النص المراد تحليله / Text to analyze
text = "I'm so excited about this! But also a bit nervous..."

# 3. Tokenization
inputs = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt"
)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# 4. التنبؤ / Prediction
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)

# 5. استخراج العواطف / Extract emotions
emotion_names = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise"
]

probs = probabilities.cpu().numpy()[0]
threshold = 0.3

detected_emotions = []
for emotion, prob in zip(emotion_names, probs):
    if prob > threshold:
        detected_emotions.append({
            'emotion': emotion,
            'probability': round(prob * 100, 2)
        })

# 6. النتائج / Results
detected_emotions.sort(key=lambda x: x['probability'], reverse=True)
print("Detected Emotions:")
for em in detected_emotions:
    print(f"  - {em['emotion']}: {em['probability']}%")

# Output:
# Detected Emotions:
#   - excitement: 87.5%
#   - nervousness: 65.2%
#   - joy: 45.3%
```

---

## 📊 مقارنة مع نموذج الصوت / Comparison with Audio Model

| Feature | نموذج الصوت / Audio | نموذج النص / Text |
|---------|-------------------|------------------|
| **النوع** | CNN | Transformer (DistilBERT) |
| **المدخلات** | ملفات صوتية | نصوص |
| **الميزات** | ZCR, RMSE, MFCC | Tokens (كلمات فرعية) |
| **العواطف** | 7 مشاعر | 28 عاطفة |
| **Classification** | Single-label | Multi-label |
| **Loss** | Categorical Cross-Entropy | BCE with Logits |
| **Framework** | TensorFlow/Keras | PyTorch |
| **Pre-training** | ❌ من الصفر | ✅ DistilBERT |
| **Epochs** | 50 (مع early stopping) | 2 |
| **الحجم** | ~50 MB | ~250 MB |
| **السرعة** | سريع | أبطأ (Transformer) |
| **الدقة** | ~75-80% | ~70-75% (varies per emotion) |

---

## 🔍 نقاط مهمة / Key Points

### 1. Multi-label vs Single-label:
```
Single-label (Audio):
واحد فقط صحيح
[0, 0, 1, 0, 0, 0, 0]

Multi-label (Text):
عدة يمكن أن تكون صحيحة
[1, 0, 0, 0, 1, 0, 1]
```

### 2. Transfer Learning:
```
Pre-training (Google):
100M+ texts → فهم اللغة

Fine-tuning (نحن):
58K texts → تصنيف المشاعر
```

### 3. Why DistilBERT?
- ✅ 40% أسرع من BERT
- ✅ 60% أصغر
- ✅ 97% من أداء BERT
- ✅ مثالي للإنتاج

### 4. Tokenization Magic:
```
"don't" → ["don", "'", "t"]
"walking" → ["walk", "##ing"]
"COVID-19" → ["cov", "##id", "-", "19"]
```

### 5. Attention Mechanism:
```
Input: "I love this movie"
Attention: كل كلمة تنظر إلى باقي الكلمات
"love" → ينتبه بشدة لـ "movie"
"this" → ينتبه بشدة لـ "movie"
```

---

## 💡 التحسينات الممكنة / Possible Improvements

### 1. المزيد من الـ Epochs:
```python
epochs = 3  # أو 4
# لكن احذر من overfitting
```

### 2. Learning Rate Scheduling:
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * epochs
)
```

### 3. Gradient Accumulation:
```python
# للتدريب مع batch size أكبر
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Early Stopping:
```python
best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(epochs):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_model()
    else:
        counter += 1
        if counter >= patience:
            break
```

### 5. Data Augmentation:
```python
# Back-translation
"I love this" → (Arabic) → "I adore this"

# Synonym replacement
"happy" → "joyful", "pleased", "delighted"

# Random deletion
"I love this movie" → "I love movie"
```

---

## 🎓 الخلاصة / Conclusion

### ما تعلمنا / What We Learned:

1. ✅ **Transformers**: معمارية ثورية للـ NLP
2. ✅ **Transfer Learning**: استخدام نماذج مدربة مسبقًا
3. ✅ **Multi-label Classification**: تصنيف متعدد التسميات
4. ✅ **Tokenization**: تحويل النص إلى أرقام
5. ✅ **PyTorch**: إطار عمل مرن للـ Deep Learning
6. ✅ **Fine-tuning**: تكييف نموذج لمهمة محددة

### الفروقات الرئيسية عن نموذج الصوت / Key Differences from Audio Model:

| نموذج الصوت | نموذج النص |
|------------|-----------|
| Feature Engineering يدوي | التعلم التلقائي للميزات |
| 7 مشاعر واحدة | 28 عاطفة متعددة |
| CNN بسيط | Transformer معقد |
| من الصفر | Transfer Learning |
| 50 epochs | 2 epochs فقط |

### الاستخدام في الإنتاج / Production Use:

النموذج جاهز للاستخدام في:
- ✅ تطبيقات الويب (Flask/FastAPI)
- ✅ APIs
- ✅ تحليل وسائل التواصل
- ✅ خدمة العملاء
- ✅ تحليل المراجعات

---

**انتهى شرح نموذج النص**
**End of Text Model Explanation**

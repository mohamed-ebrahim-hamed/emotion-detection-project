# 🎭 Emotion Detection Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive emotion detection system that combines two powerful deep learning models to identify emotions from both **text** and **voice** inputs. This project leverages state-of-the-art natural language processing and audio signal processing techniques to accurately classify emotional states across multiple modalities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models](#models)
  - [Text Emotion Model](#text-emotion-model)
  - [Voice Emotion Model](#voice-emotion-model)
- [Main Files](#main-files)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Team](#team)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project implements a **dual-modal emotion detection system** capable of analyzing emotions from two different sources:

1. **Text-Based Emotion Detection**: Uses a fine-tuned DistilBERT transformer model to identify 28 different emotions from text input, enabling nuanced sentiment analysis for applications like social media monitoring, customer feedback analysis, and mental health assessment.

2. **Voice-Based Emotion Detection**: Employs a deep learning CNN-LSTM architecture trained on MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio signals to classify emotions from speech, useful for call center analytics, voice assistants, and human-computer interaction systems.

The combination of these two models provides a robust emotion detection framework that can handle diverse input types and real-world applications.

---

## ✨ Features

- **Multi-label Text Emotion Classification**: Detect up to 28 different emotions from text with support for multiple simultaneous emotions
- **Voice Emotion Recognition**: Classify emotions from audio files (WAV format) with high accuracy
- **Pre-trained Models**: Both models come with trained weights ready for inference
- **Jupyter Notebooks**: Interactive notebooks for training, testing, and experimentation
- **Easy Integration**: Simple Python scripts for quick emotion prediction
- **Comprehensive Results**: Saved model artifacts, training history, and visualization plots

---

## 📁 Project Structure

```
emotion-detection-project/
│
├── Text Model/                          # Text-based emotion detection
│   ├── text-model.ipynb                 # Training notebook for text model
│   ├── test.py                          # Inference script for text predictions
│   └── results/                         # Model artifacts and outputs
│       └── emotion_model/               # Saved DistilBERT model
│           ├── config.json              # Model configuration
│           ├── tokenizer.json           # Tokenizer configuration
│           ├── vocab.txt                # Vocabulary file
│           ├── tokenizer_config.json    # Tokenizer settings
│           └── special_tokens_map.json  # Special tokens mapping
│
├── Voice Model/                         # Voice-based emotion detection
│   ├── voice-model.ipynb                # Training notebook for voice model
│   ├── AUDIO EMOTION PREDICTION TEST.ipynb  # Testing notebook
│   └── results/                         # Model artifacts and outputs
│       ├── best_audio_model.h5          # Best model checkpoint
│       ├── final_audio_model.h5         # Final trained model
│       ├── model_weights.weights.h5     # Model weights only
│       ├── label_encoder_classes. npy.npy # Emotion label mappings
│       ├── training_history.csv         # Training metrics
│       ├── accuracy_plot.png            # Accuracy visualization
│       ├── loss_plot.png                # Loss visualization
│       └── class_distribution.png       # Dataset distribution
│
├── .gitignore                           # Git ignore rules
├── .gitattributes                       # Git attributes
└── README.md                            # This file
```

---

## 🤖 Models

### Text Emotion Model

**Architecture**: Fine-tuned DistilBERT (Distilled BERT)

**Framework**: PyTorch + Hugging Face Transformers

**Key Features**:
- **Base Model**: `distilbert-base-uncased` - A smaller, faster version of BERT
- **Task Type**: Multi-label sequence classification
- **Number of Emotions**: 28 distinct emotion categories
- **Input Processing**: Tokenization with max length of 128 tokens
- **Training Strategy**: Transfer learning with custom classification head
- **Optimization**: AdamW optimizer with learning rate of 2e-5
- **Loss Function**: Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss)

**Detected Emotions** (28 categories):
```
admiration, amusement, anger, annoyance, approval, caring, confusion, 
curiosity, desire, disappointment, disapproval, disgust, embarrassment, 
excitement, fear, gratitude, grief, joy, love, nervousness, neutral, 
optimism, pride, realization, relief, remorse, sadness, surprise
```

**Performance**:
- Capable of detecting multiple emotions simultaneously from a single text
- Efficient inference with CPU/GPU support
- Suitable for real-time applications

---

### Voice Emotion Model

**Architecture**: CNN-LSTM Hybrid Network

**Framework**: TensorFlow/Keras

**Key Features**:
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)
  - 40 MFCC coefficients per frame
  - Maximum padding length: 174 frames
  - Sampling rate: 22,050 Hz
- **Audio Processing Library**: Librosa
- **Network Architecture**:
  - Convolutional layers for spatial feature extraction
  - LSTM layers for temporal pattern recognition
  - Dense layers for final classification
- **Training Strategy**: 
  - Data augmentation for robustness
  - Early stopping to prevent overfitting
  - Model checkpointing to save best weights
- **Optimization**: Adam optimizer
- **Loss Function**: Categorical Cross-Entropy

**Detected Emotions** (from CREMA-D):
```
Anger, Disgust, Fear, Happy, Neutral, Sad
```

**Performance Visualization**:
- Training/validation accuracy plots
- Training/validation loss plots
- Class distribution analysis

---

## 📄 Main Files

### Text Model Files

| File | Description |
|------|-------------|
| `text-model.ipynb` | Complete training pipeline for the text emotion model, including data loading, preprocessing, model training, and evaluation |
| `test.py` | Ready-to-use inference script for predicting emotions from text input |
| `results/emotion_model/` | Directory containing the saved DistilBERT model, tokenizer, and all necessary configuration files for deployment |

### Voice Model Files

| File | Description |
|------|-------------|
| `voice-model.ipynb` | Complete training pipeline for the voice emotion model with MFCC feature extraction and CNN-LSTM architecture |
| `AUDIO EMOTION PREDICTION TEST.ipynb` | Interactive testing notebook for evaluating the voice model on audio files |
| `results/best_audio_model.h5` | Model checkpoint with the best validation performance during training |
| `results/final_audio_model.h5` | Final trained model ready for deployment |
| `results/label_encoder_classes. npy.npy` | Label encoder for converting predictions to emotion names (Note: file has double extension) |
| `results/training_history.csv` | Complete training metrics log (loss, accuracy per epoch) |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/khattabx/emotion-detection-project.git
cd emotion-detection-project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

#### For Text Model:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy
pip install jupyter notebook
```

#### For Voice Model:
```bash
pip install tensorflow
pip install librosa
pip install matplotlib seaborn
pip install scikit-learn
pip install pandas numpy
pip install jupyter notebook
```

#### Install All Dependencies:
```bash
pip install torch transformers tensorflow librosa matplotlib seaborn scikit-learn pandas numpy jupyter notebook
```

---

## 🎯 Getting Started

### Quick Start - Text Emotion Detection

1. **Navigate to Text Model directory**:
```bash
cd "Text Model"
```

2. **Run the test script**:
```python
python test.py
```

3. **Or use in your code**:
```python
from test import predict_emotions

text = "I am so happy and excited about this amazing project!"
emotions = predict_emotions(text, threshold=0.5)
print(emotions)
```

### Quick Start - Voice Emotion Detection

1. **Navigate to Voice Model directory**:
```bash
cd "Voice Model"
```

2. **Open the testing notebook**:
```bash
jupyter notebook "AUDIO EMOTION PREDICTION TEST.ipynb"
```

3. **Load model and predict**:
```python
import tensorflow as tf
import numpy as np
import librosa

def extract_features(file_path, max_pad_len=174, n_mfcc=40):
    """Extract MFCC features from audio file"""
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Padding
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs

# Load model
model = tf.keras.models.load_model('results/best_audio_model.h5')

# Load label encoder
label_classes = np.load('results/label_encoder_classes. npy.npy', allow_pickle=True)

# Load audio and extract features
features = extract_features('path/to/audio.wav')
features = np.expand_dims(features, axis=0)
features = np.expand_dims(features, axis=-1)

# Predict emotion
prediction = model.predict(features)
emotion_idx = np.argmax(prediction)
emotion = label_classes[emotion_idx]
print(f"Detected emotion: {emotion}")
```

---

## 💻 Usage

### Text Model Usage Example

```python
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "Text Model/results/emotion_model"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Emotion labels
emotion_names = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise"
]

def predict_emotions(text, threshold=0.5):
    """Predict emotions from text"""
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Filter by threshold
    detected = [(emotion_names[i], float(probs[i])) 
                for i in range(len(probs)) if probs[i] > threshold]
    
    return sorted(detected, key=lambda x: x[1], reverse=True)

# Example
text = "I am extremely grateful for all the support and love from my family!"
results = predict_emotions(text)
print("Detected emotions:")
for emotion, score in results:
    print(f"  {emotion}: {score:.3f}")
```

### Voice Model Usage Example

```python
import tensorflow as tf
import librosa
import numpy as np

# Load model
model = tf.keras.models.load_model('Voice Model/results/best_audio_model.h5')

# Load label encoder
label_encoder_classes = np.load('Voice Model/results/label_encoder_classes.npy.npy', 
                                 allow_pickle=True)

def extract_features(file_path, max_pad_len=174, n_mfcc=40):
    """Extract MFCC features from audio file"""
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Padding
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    
    prediction = model.predict(features)
    emotion_idx = np.argmax(prediction)
    emotion = label_encoder_classes[emotion_idx]
    confidence = prediction[0][emotion_idx]
    
    return emotion, confidence

# Example
audio_file = "path/to/audio.wav"
emotion, confidence = predict_emotion(audio_file)
print(f"Detected emotion: {emotion} (confidence: {confidence:.2%})")
```

---

## 📊 Datasets

### GoEmotions Dataset (Text Model)

**Source**: Google Research

**Description**: A large-scale, manually annotated dataset of Reddit comments labeled with 28 emotion categories.

**Statistics**:
- **Training samples**: ~43,000 comments
- **Validation samples**: ~5,000 comments
- **Test samples**: ~5,000 comments
- **Emotions**: 28 fine-grained emotion categories
- **Multi-label**: Yes (comments can have multiple emotion labels)

**Citation**:
```
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

### CREMA-D Dataset (Voice Model)

**Source**: Crowd-sourced Emotional Multimodal Actors Dataset

**Description**: An audio-visual dataset with acted emotional expressions from 91 actors.

**Statistics**:
- **Total samples**: 7,442 audio clips
- **Actors**: 91 (48 male, 43 female)
- **Emotions**: 6 categories (Anger, Disgust, Fear, Happy, Neutral, Sad)
- **Age range**: 20-74 years
- **Ethnic backgrounds**: Diverse (African American, Asian, Caucasian, Hispanic, Unspecified)

**Citation**:
```
@article{cao2014crema,
  title={CREMA-D: Crowd-sourced emotional multimodal actors dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Raquel C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE transactions on affective computing},
  volume={5},
  number={4},
  pages={377--390},
  year={2014}
}
```

---

## 📈 Results

### Text Model Performance

The text emotion model demonstrates strong performance in multi-label emotion classification:

- **Architecture**: DistilBERT with 66M parameters
- **Training Time**: ~2 epochs (optimized for efficiency)
- **Inference Speed**: Fast (suitable for real-time applications)
- **Multi-label Support**: Can detect multiple emotions simultaneously
- **Model Size**: ~260 MB (compressed)

**Key Strengths**:
- Handles complex emotional expressions with nuance
- Performs well on informal text (social media, chat)
- Low latency for production deployment
- Supports 28 distinct emotion categories

### Voice Model Performance

The voice emotion model shows robust performance across different speakers and contexts:

- **Architecture**: CNN-LSTM hybrid with ~2M parameters
- **Training History**: Available in `training_history.csv`
- **Model Size**: ~172 MB
- **Input**: Audio files (WAV format, 22.05 kHz sampling rate)

**Key Strengths**:
- Invariant to speaker characteristics
- Handles variable-length audio inputs
- Robust to background noise (with proper training)
- Real-time capable inference

**Visualization**:
- Training curves show consistent convergence
- Balanced performance across all emotion classes
- Low overfitting with proper regularization

---

## 👥 Team

This project was developed by:

**Mohamed Ibrahim Hamed El-Saeed Mohamed**
- Email: mohamed_1157@ai.kfs.edu.eg
- Institution: Kafr El-Sheikh University, Faculty of Artificial Intelligence
- Role: Lead Developer & Researcher

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

**Areas for Contribution**:
- Adding more emotion categories
- Improving model architectures
- Creating real-time inference APIs
- Adding support for more languages
- Integrating both models into a unified system
- Creating web/mobile applications

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Research** for the GoEmotions dataset
- **CREMA-D Team** for the audio emotion dataset
- **Hugging Face** for the transformers library
- **TensorFlow/Keras** for deep learning framework
- **Librosa** for audio processing capabilities

---

## 📧 Contact

For questions, suggestions, or collaborations, please reach out to:

**Mohamed Ibrahim Hamed El-Saeed Mohamed**
- Email: mohamed_1157@ai.kfs.edu.eg
- GitHub: [@khattabx](https://github.com/khattabx)

---

## 🔮 Future Work

- [ ] Combine text and voice models for multimodal emotion detection
- [ ] Deploy models as REST API services
- [ ] Create a web interface for easy interaction
- [ ] Extend to support more languages
- [ ] Add real-time emotion tracking
- [ ] Implement emotion intensity scoring
- [ ] Add support for video-based emotion detection
- [ ] Create mobile applications (iOS/Android)
- [ ] Integrate with chatbots and virtual assistants
- [ ] Develop emotion trend analysis tools

---

<div align="center">
  <strong>Made with ❤️ for advancing AI-powered emotion understanding</strong>
</div>

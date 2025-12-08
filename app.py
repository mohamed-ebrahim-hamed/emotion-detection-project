# app.py (محدث)
import os
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
import uuid
import traceback
import logging
import subprocess
import tempfile
import sys

# إعداد التسجيل
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import PyTorch and Transformers (optional for text model)
try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    TEXT_MODEL_AVAILABLE = True
    logger.info("PyTorch and Transformers loaded successfully")
except ImportError as e:
    logger.warning(f"PyTorch/Transformers not available: {e}")
    logger.warning("Text emotion analysis will be disabled. Install with: pip install torch transformers soxr")
    TEXT_MODEL_AVAILABLE = False
    torch = None
    DistilBertTokenizerFast = None
    DistilBertForSequenceClassification = None

# Add conda ffmpeg to PATH
if hasattr(sys, 'base_prefix'):
    conda_env = sys.base_prefix
    ffmpeg_paths = [
        os.path.join(conda_env, 'Library', 'bin'),
        os.path.join(conda_env, 'bin'),
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path) and path not in os.environ['PATH']:
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            logger.info(f"Added to PATH: {path}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'ogg', 'webm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and preprocessing objects
def load_model():
    try:
        logger.info("Loading model...")
        # Load model architecture
        with open('model/CNN_model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        
        # Load model weights
        model.load_weights("model/best_model1_weights.h5")
        
        # Compile model (required for prediction)
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_preprocessing():
    try:
        logger.info("Loading preprocessing objects...")
        # Load scaler
        with open('model/scaler2.pickle', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoder
        with open('model/encoder2.pickle', 'rb') as f:
            encoder = pickle.load(f)
        
        logger.info("Preprocessing objects loaded successfully")
        return scaler, encoder
    except Exception as e:
        logger.error(f"Error loading preprocessing objects: {str(e)}")
        raise

# Load text model
def load_text_model():
    if not TEXT_MODEL_AVAILABLE:
        logger.info("Text model dependencies not available, skipping text model loading")
        return None, None, None
    
    try:
        logger.info("Loading text emotion model...")
        text_model_path = 'model/Text Model'
        
        if not os.path.exists(text_model_path):
            logger.warning(f"Text model path not found: {text_model_path}")
            return None, None, None
        
        text_model = DistilBertForSequenceClassification.from_pretrained(text_model_path)
        text_tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_model.to(device)
        text_model.eval()
        
        logger.info(f"Text model loaded successfully on {device}")
        return text_model, text_tokenizer, device
    except Exception as e:
        logger.error(f"Error loading text model: {str(e)}")
        return None, None, None

# Load resources
try:
    model = load_model()
    scaler, encoder = load_preprocessing()
    text_model, text_tokenizer, text_device = load_text_model()
    logger.info("All resources loaded successfully")
except Exception as e:
    logger.error(f"Failed to load resources: {str(e)}")
    model = None
    scaler = None
    encoder = None
    text_model = None
    text_tokenizer = None
    text_device = None

# Emotion mapping
emotion_labels = {
    'angry': '😠 غاضب',
    'disgust': '🤢 مقرف',
    'fear': '😨 خائف',
    'happy': '😃 سعيد',
    'sad': '😢 حزين',
    'surprise': '😲 متفاجئ',
    'neutral': '😐 محايد'
}

emotion_colors = {
    'angry': '#FF6B6B',
    'disgust': '#8AC926',
    'fear': '#7209B7',
    'happy': '#FFD166',
    'sad': '#118AB2',
    'surprise': '#EF476F',
    'neutral': '#06D6A0'
}

# Text emotions list (28 emotions)
text_emotion_names = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise"
]

# Text emotions in Arabic
text_emotion_arabic = {
    "admiration": "إعجاب",
    "amusement": "تسلية",
    "anger": "غضب",
    "annoyance": "انزعاج",
    "approval": "موافقة",
    "caring": "اهتمام",
    "confusion": "ارتباك",
    "curiosity": "فضول",
    "desire": "رغبة",
    "disappointment": "خيبة أمل",
    "disapproval": "رفض",
    "disgust": "اشمئزاز",
    "embarrassment": "إحراج",
    "excitement": "حماس",
    "fear": "خوف",
    "gratitude": "امتنان",
    "grief": "حزن شديد",
    "joy": "فرح",
    "love": "حب",
    "nervousness": "توتر",
    "neutral": "محايد",
    "optimism": "تفاؤل",
    "pride": "فخر",
    "realization": "إدراك",
    "relief": "ارتياح",
    "remorse": "ندم",
    "sadness": "حزن",
    "surprise": "مفاجأة"
}

text_emotion_emojis = {
    "admiration": "🤩",
    "amusement": "😄",
    "anger": "😠",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "😍",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤗",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😢",
    "joy": "😃",
    "love": "❤️",
    "nervousness": "😰",
    "neutral": "😐",
    "optimism": "😊",
    "pride": "😌",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲"
}

# Feature extraction functions (من Notebook)
def zcr(data, frame_length=2048, hop_length=512):
    zcr_feat = librosa.feature.zero_crossing_rate(y=data, 
                                                  frame_length=frame_length, 
                                                  hop_length=hop_length)
    return np.squeeze(zcr_feat)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_feat = librosa.feature.rms(y=data, 
                                    frame_length=frame_length, 
                                    hop_length=hop_length)
    return np.squeeze(rmse_feat)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    
    # Calculate features
    zcr_result = zcr(data, frame_length, hop_length)
    rmse_result = rmse(data, frame_length, hop_length)
    mfcc_result = mfcc(data, sr, frame_length, hop_length, flatten=True)
    
    # Combine features
    result = np.hstack((result, zcr_result, rmse_result, mfcc_result))
    return result

def convert_audio_if_needed(input_path):
    """Convert audio to WAV format using ffmpeg if needed"""
    try:
        # Check if file is already readable by soundfile
        try:
            with sf.SoundFile(input_path) as f:
                logger.debug(f"Audio file is directly readable: {input_path}")
                return input_path
        except:
            logger.info(f"File not directly readable, attempting conversion: {input_path}")
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # Find ffmpeg executable
        ffmpeg_cmd = 'ffmpeg'
        if sys.platform == 'win32':
            # Try to find ffmpeg in conda environment
            if hasattr(sys, 'base_prefix'):
                conda_ffmpeg = os.path.join(sys.base_prefix, 'Library', 'bin', 'ffmpeg.exe')
                if os.path.exists(conda_ffmpeg):
                    ffmpeg_cmd = conda_ffmpeg
                    logger.info(f"Using conda ffmpeg: {ffmpeg_cmd}")
        
        # Use ffmpeg to convert
        cmd = [
            ffmpeg_cmd, '-y', '-i', input_path,
            '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
            temp_wav.name
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted audio to: {temp_wav.name}")
            return temp_wav.name
        else:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            raise Exception(f"ffmpeg not found or conversion failed. Please ensure ffmpeg is installed.")
            
    except FileNotFoundError:
        logger.error("ffmpeg executable not found")
        raise Exception("ffmpeg is not installed. Please install ffmpeg: conda install -c conda-forge ffmpeg")
    except Exception as e:
        logger.error(f"Error in audio conversion: {str(e)}")
        raise

def get_predict_feat(path, duration=2.5, offset=0.6):
    converted_path = None
    try:
        # Convert audio if needed
        audio_path = convert_audio_if_needed(path)
        converted_path = audio_path if audio_path != path else None
        
        # Load audio file with librosa
        logger.debug(f"Loading audio file: {audio_path}")
        data, s_rate = librosa.load(audio_path, duration=duration, offset=offset, sr=22050)
        logger.debug(f"Audio loaded: sample_rate={s_rate}, duration={len(data)/s_rate:.2f}s")
        
        # Extract features
        res = extract_features(data, s_rate)
        result = np.array(res)
        
        # Ensure the feature vector has the correct length (2376 as in notebook)
        logger.debug(f"Original feature length: {len(result)}")
        
        if len(result) > 2376:
            result = result[:2376]
        elif len(result) < 2376:
            # Pad with zeros if shorter
            result = np.pad(result, (0, 2376 - len(result)), mode='constant')
        
        # Reshape for scaler
        result = result.reshape(1, -1)
        
        # Scale the features
        i_result = scaler.transform(result)
        
        # Reshape for model (batch_size, timesteps, features)
        final_result = i_result.reshape(i_result.shape[0], i_result.shape[1], 1)
        
        logger.debug(f"Final shape for model: {final_result.shape}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in get_predict_feat: {str(e)}")
        raise
    finally:
        # Clean up temporary converted file
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                logger.debug(f"Cleaned up temporary file: {converted_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Endpoint للتحقق من حالة التطبيق"""
    if model is None or scaler is None or encoder is None:
        return jsonify({'status': 'error', 'message': 'Resources not loaded'}), 500
    return jsonify({'status': 'healthy', 'model': 'loaded', 'scaler': 'loaded', 'encoder': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please use WAV, MP3, M4A, OGG, or WEBM'}), 400
    
    if model is None or scaler is None or encoder is None:
        return jsonify({'error': 'Model not loaded properly. Please check server logs.'}), 500
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        # Save file
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Extract features and predict
        features = get_predict_feat(filepath)
        logger.info(f"Features extracted, shape: {features.shape}")
        
        predictions = model.predict(features, verbose=0)
        logger.info(f"Predictions made: {predictions.shape}")
        
        # Get predicted emotion using categories directly
        predicted_index = np.argmax(predictions, axis=1)[0]
        emotion_categories = encoder.categories_[0]
        predicted_emotion = emotion_categories[predicted_index]
        
        # Get emotion details
        emotion_arabic = emotion_labels.get(predicted_emotion, predicted_emotion)
        emotion_color = emotion_colors.get(predicted_emotion, '#06D6A0')
        confidence = float(np.max(predictions) * 100)
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, emotion in enumerate(emotion_categories):
            emotion_probs[emotion] = float(predictions[0][i] * 100)
        
        logger.info(f"Prediction successful: {predicted_emotion} ({confidence:.2f}%)")
        
        return jsonify({
            'success': True,
            'emotion': predicted_emotion,
            'emotion_arabic': emotion_arabic,
            'emotion_color': emotion_color,
            'confidence': round(confidence, 2),
            'probabilities': emotion_probs
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

@app.route('/predict-text', methods=['POST'])
def predict_text():
    """التنبؤ بالعواطف من النص"""
    if not TEXT_MODEL_AVAILABLE:
        return jsonify({'error': 'Text model not available. Please install: pip install torch transformers soxr'}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if text_model is None or text_tokenizer is None:
            return jsonify({'error': 'Text model not loaded. Check server logs.'}), 500
        
        logger.info(f"Text prediction request: {text[:50]}...")
        
        # Tokenization
        inputs = text_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(text_device)
        attention_mask = inputs["attention_mask"].to(text_device)
        
        # Prediction
        with torch.no_grad():
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        probs = probabilities.cpu().numpy()[0]
        
        # Get top emotions (threshold = 0.3)
        threshold = 0.3
        detected_emotions = []
        all_emotions = {}
        
        for emotion, prob in zip(text_emotion_names, probs):
            prob_percent = float(prob * 100)
            all_emotions[emotion] = prob_percent
            
            if prob > threshold:
                detected_emotions.append({
                    'emotion': emotion,
                    'emotion_arabic': text_emotion_arabic.get(emotion, emotion),
                    'emoji': text_emotion_emojis.get(emotion, '😐'),
                    'probability': round(prob_percent, 2)
                })
        
        # Sort by probability
        detected_emotions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get primary emotion
        if detected_emotions:
            primary = detected_emotions[0]
        else:
            primary = {
                'emotion': 'neutral',
                'emotion_arabic': 'محايد',
                'emoji': '😐',
                'probability': 50.0
            }
        
        logger.info(f"Text prediction successful: {primary['emotion']} ({primary['probability']:.2f}%)")
        
        return jsonify({
            'success': True,
            'primary_emotion': primary,
            'detected_emotions': detected_emotions,
            'all_probabilities': all_emotions
        })
        
    except Exception as e:
        logger.error(f"Text prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Text prediction error: {str(e)}'}), 500

@app.route('/test-model')
def test_model():
    """Endpoint لاختبار النموذج بملف تجريبي"""
    try:
        # استخدام ملف تجريبي من المجلد uploads إذا وجد
        test_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.wav')]
        
        if test_files:
            test_file = os.path.join(app.config['UPLOAD_FOLDER'], test_files[0])
            features = get_predict_feat(test_file)
            predictions = model.predict(features, verbose=0)
            
            return jsonify({
                'success': True,
                'message': 'Model test successful',
                'predictions_shape': str(predictions.shape),
                'sample_prediction': predictions[0].tolist()
            })
        else:
            return jsonify({'error': 'No test files found in uploads folder'}), 404
            
    except Exception as e:
        logger.error(f"Model test error: {str(e)}")
        return jsonify({'error': f'Model test error: {str(e)}'}), 500

if __name__ == '__main__':
    # تحقق من أن النموذج تم تحميله قبل التشغيل
    if model is None or scaler is None or encoder is None:
        logger.error("Failed to load model or preprocessing objects. Exiting...")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
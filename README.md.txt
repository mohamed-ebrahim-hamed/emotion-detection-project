# Emotion Detection App 🎭🧠

A **multi-label emotion detection web application** built using a fine-tuned **DistilBERT** model on the **GoEmotions dataset**, deployed with **Streamlit**.

The app analyzes English text and predicts one or more emotions present in the sentence with confidence scores.

---

## 🚀 Live Demo
> _Add Streamlit Cloud link here after deployment_

---

## 📌 Features
- ✅ Multi-label emotion classification (more than one emotion per sentence)
- ✅ Fine-tuned DistilBERT model
- ✅ Interactive Streamlit web interface
- ✅ Adjustable confidence threshold
- ✅ Top-K fallback for stable predictions
- ✅ Displays full probability distribution for all emotions
- ✅ Lightweight and easy to deploy

---

## 🧠 Model Details
- **Base Model:** `distilbert-base-uncased`
- **Dataset:** GoEmotions (Google)
- **Task Type:** Multi-label classification
- **Number of Labels:** 28 emotions
- **Activation Function:** Sigmoid
- **Framework:** PyTorch + HuggingFace Transformers

### Emotion Labels
admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization,
relief, remorse, sadness, surprise, neutral

---

## 🖥️ Tech Stack
- Python
- PyTorch
- HuggingFace Transformers
- Streamlit
- Pandas & NumPy

---

## 📂 Project Structure
```text
emotion-detection-app/
│
├── app.py                # Streamlit application
├── requirements.txt      # Project dependencies
├── emotion_model/        # Fine-tuned model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
│
├── .gitignore
├── .gitattributes
└── README.md
⚙️ Installation & Running Locally
1️⃣ Clone the repository
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
python -m streamlit run app.py


The app will open at:

http://localhost:8501

🌐 Deployment

This project is ready for deployment on:

✅ Streamlit Community Cloud

✅ HuggingFace Spaces

✅ Any cloud platform that supports Python

📈 Example Usage

Input:

I finally achieved my goal, but I'm still a bit nervous about the future.


Output:

joy → 0.87
optimism → 0.63
nervousness → 0.41

⚠️ Notes

The model is trained only on English text.

Predictions may include multiple emotions for a single sentence.

For best results, avoid very short or ambiguous inputs.

🔮 Future Improvements

Arabic or multilingual emotion detection

REST API using FastAPI

Batch prediction (CSV upload)

Model performance visualization

Improved UI with charts and emotion bars

👤 Author

Your Name Here
AI Engineer | Data Scientist

GitHub: https://github.com/USERNAME

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

📜 License

This project is licensed under the MIT License.

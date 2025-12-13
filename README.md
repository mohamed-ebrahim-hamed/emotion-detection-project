# Voice-based Emotion Detection System

A web application that analyzes emotions from speech using machine learning / AI. The app allows uploading an audio file or entering text, converts speech to text when needed, and displays emotion analysis results in a clear visual format.

## Features
- Upload audio files (WAV, MP3, M4A, OGG, WEBM).
- Speech-to-text conversion and text emotion analysis.
- Visual results showing relative emotion probabilities (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral).
- Arabic language support for the UI.

## Screenshots
Save the two provided screenshots in the repository under the `images/` folder with these exact names:
- images/screenshot1.png  (Image 1)
- images/screenshot2.png  (Image 2)

Then include them in the README like this:

![Text analysis and anger result](images/screenshot1.png)

![Audio upload and full analysis results](images/screenshot2.png)

Note: These screenshots are taken from the website you are building — make sure to save them with the given filenames in the `images` folder before viewing the README on GitHub.

## How to add the screenshots to the repo and push changes
1. Put the screenshots in the `images/` directory at the project root:
   - images/screenshot1.png
   - images/screenshot2.png

2. Run these commands locally to add and push them (replace `main` with your branch name if different):
```bash
git add images/screenshot1.png images/screenshot2.png README.md
git commit -m "Add screenshots and update README"
git push origin main
```

## Run the project locally
1. (Optional) Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

2. Start the application (adjust the command to your project entrypoint):
```bash
streamlit run app.py
# or
python main.py
# or if your project uses Jupyter notebooks, open the notebook and run cells:
jupyter notebook
```

## Want me to add the images and push the changes?
I can prepare the README and commit the screenshots for you if you give me repository access or provide the image files here. Tell me whether to proceed and which branch to use.

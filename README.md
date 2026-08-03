🎙️ Deepfake Audio Detector
A Flask-based web application that detects whether an audio clip is real (human) or AI-generated (deepfake), using a trained machine learning model.
🖥️ Demo
<img width="1852" height="886" alt="Screenshot 2026-08-04 004941" src="https://github.com/user-attachments/assets/3041db1f-b2d5-4875-9e88-b326a5574107" />
🚀 Features
Upload an audio file through a simple web interface
Model classifies the audio as Real or AI-generated (Fake)
Built end-to-end — data preprocessing, model training, and a deployable web app
🎯 Problem Statement
Develop a system that can analyze an audio clip and determine whether it is authentic human speech or AI-generated deepfake audio. The system processes an input audio file and outputs a prediction indicating whether the audio is real or synthetic.
🧠 How It Works
The model was trained on a labeled dataset of real and AI-generated audio samples (sourced from Kaggle — see KAGGLE/ folder for dataset reference). Audio features are extracted and fed into a trained classifier to predict authenticity.
Training and experimentation are documented in train.py, and the trained model artifacts are in the model/ folder.
🛠️ Tech Stack
Backend: Python, Flask
ML: Custom-trained audio classification model (see train.py for architecture and training details)
Frontend: HTML/CSS (Flask templates)

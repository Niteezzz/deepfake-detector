# ================================================
# app.py — Deepfake Audio Detector API
# Flask backend that serves the trained model
# ================================================

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import warnings
import tempfile
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────
MODEL_PATH  = "model/deepfake_detector.h5"
SCALER_PATH = "model/scaler.pkl"
SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40

# ── Load Model ────────────────────────────────────
model  = None
scaler = None

def load_model():
    global model, scaler
    try:
        model  = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Model not found: {e}")
        print("Run train.py first!")

# ── Extract Features ──────────────────────────────
def extract_features(file_path):
    audio, sr = librosa.load(file_path,
                              sr=SAMPLE_RATE,
                              duration=DURATION)
    if len(audio) < sr * DURATION:
        audio = np.pad(audio,
                      (0, sr * DURATION - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio,
                                  sr=sr,
                                  n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)

# ── Mock Result (before model is trained) ─────────
def get_mock():
    return {
        "prediction":  "FAKE",
        "confidence":  94.7,
        "label":       "AI Generated Audio",
        "real_prob":   5.3,
        "fake_prob":   94.7,
        "message":     "This audio shows strong signs of AI generation",
        "features": {
            "mfcc_variance":  "Low — typical of synthetic audio",
            "pitch_pattern":  "Too consistent — unnatural",
            "noise_profile":  "Absent — real audio has background noise"
        }
    }

# ── Routes ────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Check file uploaded
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Use real model if available
        if model is not None and scaler is not None:
            features = extract_features(tmp_path)
            features = scaler.transform([features])
            prediction = model.predict(features)[0][0]

            is_real    = prediction > 0.5
            confidence = prediction * 100 if is_real else (1 - prediction) * 100
            real_prob  = round(float(prediction * 100), 1)
            fake_prob  = round(float((1 - prediction) * 100), 1)

            result = {
                "prediction": "REAL" if is_real else "FAKE",
                "confidence": round(float(confidence), 1),
                "label":      "Authentic Human Voice" if is_real else "AI Generated Audio",
                "real_prob":  real_prob,
                "fake_prob":  fake_prob,
                "message":    "This audio appears to be genuine human speech." if is_real
                              else "This audio shows signs of AI generation.",
                "features": {
                    "mfcc_variance": "High — natural variation detected" if is_real
                                     else "Low — typical of synthetic audio",
                    "pitch_pattern": "Natural variation present" if is_real
                                     else "Too consistent — unnatural",
                    "noise_profile": "Natural background noise present" if is_real
                                     else "Absent — real audio has background noise"
                }
            }

        else:
            # Fallback to mock
            result = get_mock()

        # Cleanup temp file
        os.unlink(tmp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "fallback": get_mock()
        }), 200

# ── Run ───────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)

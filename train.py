# ================================================
# train.py — Deepfake Audio Detector
# Trains a Neural Network on REAL vs FAKE audio
# ================================================

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────
REAL_DIR    = "KAGGLE/AUDIO/REAL"       # folder name of real audio
FAKE_DIR    = "KAGGLE/AUDIO/FAKE"       # folder name of fake audio
MODEL_PATH  = "model/deepfake_detector.h5"
SCALER_PATH = "model/scaler.pkl"
SAMPLE_RATE = 22050
DURATION    = 3            # seconds per clip
N_MFCC      = 40           # number of MFCC features

# ── Step 1: Extract MFCC Features ─────────────────
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, 
                                  sr=SAMPLE_RATE, 
                                  duration=DURATION)
        
        # Pad if audio is shorter than DURATION
        if len(audio) < sr * DURATION:
            audio = np.pad(audio, 
                          (0, sr * DURATION - len(audio)))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, 
                                     sr=sr, 
                                     n_mfcc=N_MFCC)
        
        # Return mean of each MFCC coefficient
        return np.mean(mfcc, axis=1)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ── Step 2: Load Dataset ──────────────────────────
def load_dataset():
    features = []
    labels   = []
    
    print("Loading REAL audio files...")
    real_files = os.listdir(REAL_DIR)
    for i, fname in enumerate(real_files):
        if fname.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            path = os.path.join(REAL_DIR, fname)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(1)    # 1 = REAL
        if i % 50 == 0:
            print(f"  Processed {i}/{len(real_files)} real files")
    
    print(f"✅ Loaded {labels.count(1)} real samples")
    
    print("Loading FAKE audio files...")
    fake_files = os.listdir(FAKE_DIR)
    for i, fname in enumerate(fake_files):
        if fname.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            path = os.path.join(FAKE_DIR, fname)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(0)    # 0 = FAKE
        if i % 50 == 0:
            print(f"  Processed {i}/{len(fake_files)} fake files")
    
    print(f"✅ Loaded {labels.count(0)} fake samples")
    
    return np.array(features), np.array(labels)

# ── Step 3: Build Neural Network ─────────────────
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', 
                          input_shape=(input_shape,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer = 'adam',
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )
    
    return model

# ── Step 4: Train and Save ────────────────────────
def train():
    print("\n🎯 DEEPFAKE AUDIO DETECTOR — TRAINING")
    print("=" * 45)
    
    # Load data
    print("\n📂 Loading dataset...")
    X, y = load_dataset()
    print(f"\n✅ Total samples: {len(X)}")
    print(f"   Real: {sum(y == 1)}  |  Fake: {sum(y == 0)}")
    
    # Scale features
    print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")
    
    # Build model
    print("\n🧠 Building Neural Network...")
    model = build_model(X_train.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=3, 
            factor=0.5
        )
    ]
    
    # Train
    print("\n🚀 Training started...")
    history = model.fit(
        X_train, y_train,
        epochs          = 50,
        batch_size      = 32,
        validation_data = (X_test, y_test),
        callbacks       = callbacks,
        verbose         = 1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print("\n🏆 Training Complete!")

# ── Run ───────────────────────────────────────────
if __name__ == "__main__":
    train()

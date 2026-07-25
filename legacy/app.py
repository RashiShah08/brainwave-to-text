from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os
import mne
from scipy.stats import skew, kurtosis
import uuid
import warnings

app = Flask(__name__, template_folder="../templates")

# ========= Paths =========
ENSEMBLE_PATH = "C:/Users/rashi/brainwave_to_text/models/final_ensemble.pkl"
SCALER_PATH = "C:/Users/rashi/brainwave_to_text/models/scaler.pkl"
PCA_PATH = "C:/Users/rashi/brainwave_to_text/models/pca.pkl"
FEATURE_COLS_PATH = "C:/Users/rashi/brainwave_to_text/models/feature_cols.pkl"
UPLOAD_FOLDER = "C:/Users/rashi/brainwave_to_text/raw_data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========= Load Model + Preprocessors =========
try:
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    ensemble = joblib.load(ENSEMBLE_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    print("✅ Ensemble, scaler, PCA, and feature columns loaded")
    print(f"Expected features: {len(feature_cols)}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model or preprocessors: {str(e)}")

# ========= Class Labels =========
class_labels = {0: "Left", 1: "Right"}

# ========= Feature Extraction (matches raw_features.py) =========
def extract_features_from_epoch(epoch_data, sfreq=256):
    """Extract same features as raw_features.py: time-domain + frequency bands"""
    n_channels = epoch_data.shape[0]
    
    # Time-domain features per channel
    means = np.mean(epoch_data, axis=1)
    vars_ = np.var(epoch_data, axis=1)
    skews = skew(epoch_data, axis=1)
    kurts = kurtosis(epoch_data, axis=1)
    
    # Frequency-domain: PSD in EEG bands
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data, sfreq=sfreq, fmin=1, fmax=40, n_fft=sfreq
    )
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        band_idx = (freqs >= fmin) & (freqs < fmax)
        band_powers[band] = np.mean(psd[:, band_idx], axis=1)
    
    # Flatten and concatenate
    features = np.concatenate([
        means, vars_, skews, kurts,
        band_powers['delta'], band_powers['theta'], 
        band_powers['alpha'], band_powers['beta'], band_powers['gamma']
    ])
    
    return features

# ========= EDF Preprocessing =========
def preprocess_edf(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.resample(256)  # Match training sfreq
        raw.filter(1., 40., fir_design='firwin')
        
        # Create single 1-second epoch from entire recording (or first second)
        epoch_length = 256  # 1 second at 256 Hz
        if raw.n_times < epoch_length:
            raise ValueError(f"EDF file too short: {raw.n_times} samples, need {epoch_length}")
        
        # Take first 1-second epoch
        epoch_data = raw.get_data()[:, :epoch_length]
        
        # Extract features (same as raw_features.py)
        features = extract_features_from_epoch(epoch_data)
        
        # Create DataFrame aligned with training
        df_feats = pd.DataFrame([features], columns=feature_cols[:len(features)])
        
        # Pad missing features if any
        for col in feature_cols:
            if col not in df_feats.columns:
                df_feats[col] = 0
        
        # Reorder to match training
        df_feats = df_feats[feature_cols]
        
        # Verify feature count
        if df_feats.shape[1] != len(feature_cols):
            raise ValueError(f"Generated {df_feats.shape[1]} features, expected {len(feature_cols)}")
        
        # Scale and PCA
        feats_scaled = scaler.transform(df_feats.values)
        feats_pca = pca.transform(feats_scaled)
        
        return feats_pca[0]
    except Exception as e:
        raise Exception(f"EDF preprocessing failed: {str(e)}")

# ========= Flask Routes =========
@app.route("/")
def upload_file():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    if file and file.filename.endswith(".edf"):
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(file_path)

        try:
            features = preprocess_edf(file_path).reshape(1, -1)
            pred_class = ensemble.predict(features)[0]
            probabilities = ensemble.predict_proba(features)[0]
            confidence = f"{max(probabilities) * 100:.2f}%"
            prediction = class_labels[pred_class]
            
            print(f"File: {file.filename}, Prediction: {prediction}, Probabilities: {probabilities}")
            result = f"Object on the {prediction} (Confidence: {confidence})"
        except Exception as e:
            os.remove(file_path)
            return f"Error during prediction: {str(e)}", 500
        
        os.remove(file_path)
        return render_template("result.html", result=result)

    return "Invalid file format. Please upload a .edf file", 400

if __name__ == "__main__":
    app.run(debug=True)
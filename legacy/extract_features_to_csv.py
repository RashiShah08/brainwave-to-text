
import mne
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import os
import glob

# Paths from your project structure
RAW_DATA_DIR = "../raw_data"
OUTPUT_CSV = "../processed_data/raw_features.csv"
TARGET_SFREQ = 256  # To match previous expectations
EPOCH_DURATION = 1.0  # 1-second epochs

# Function to determine run type based on filename
def get_run_type(filename):
    run = filename.split('R')[1].split('.')[0]
    if run in ['01', '02']:
        return 'baseline'
    elif run in ['03', '04', '07', '08', '11', '12']:
        return 'left_right_fist'
    elif run in ['05', '06', '09', '10', '13', '14']:
        return 'fists_feet'
    return 'unknown'

# Function to extract features from a single epoch (numpy array: channels x time)
def extract_epoch_features(epoch_data):
    features = []
    # Time-domain features per channel (mean, var, skew, kurtosis)
    means = np.mean(epoch_data, axis=1)
    vars_ = np.var(epoch_data, axis=1)
    skews = skew(epoch_data, axis=1)
    kurts = kurtosis(epoch_data, axis=1)
    
    # Frequency-domain: PSD in EEG bands (delta:0-4, theta:4-8, alpha:8-12, beta:12-30, gamma:30-40)
    psd, freqs = mne.time_frequency.psd_array_welch(epoch_data, sfreq=TARGET_SFREQ, fmin=1, fmax=40)
    bands = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        band_idx = (freqs >= fmin) & (freqs < fmax)
        band_powers[band] = np.mean(psd[:, band_idx], axis=1)
    
    # Flatten and concatenate all features
    features.extend(means.flatten())
    features.extend(vars_.flatten())
    features.extend(skews.flatten())
    features.extend(kurts.flatten())
    for power in band_powers.values():
        features.extend(power.flatten())
    
    return np.array(features)

# Main processing function for a single .edf file
def process_edf(file_path):
    try:
        print(f"Processing {file_path}...")
        raw = mne.io.read_raw_edf(file_path, preload=True)
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        
        # Basic preprocessing
        raw.filter(1., 40., fir_design='firwin')
        
        # Noise removal: Try ICA for artifacts
        try:
            ica = FastICA(n_components=20, random_state=42, max_iter=1000, tol=1e-3)
            ica.fit(raw.get_data().T)
            sources = ica.transform(raw.get_data().T)
            sources[:, 0:2] = 0  # Zero out first 2 components (assumed artifacts)
            reconstructed = ica.inverse_transform(sources).T
            raw._data = reconstructed
        except Exception as e:
            print(f"ICA failed for {file_path}: {str(e)}. Using filtered data without ICA.")
        
        # Get run type
        run_type = get_run_type(os.path.basename(file_path))
        
        # Check for events
        events, event_id = mne.events_from_annotations(raw)
        has_motor_imagery = 'T1' in event_id and 'T2' in event_id
        
        try:
            if has_motor_imagery and run_type in ['left_right_fist', 'fists_feet']:
                # Use T1/T2 events for epoching
                epochs = mne.Epochs(raw, events, event_id={'T1': event_id['T1'], 'T2': event_id['T2']},
                                    tmin=0, tmax=EPOCH_DURATION - 1/TARGET_SFREQ, baseline=None, preload=True)
                data = epochs.get_data()  # (n_epochs, 64, 256)
                labels = epochs.events[:, -1]
                labels = np.where(labels == event_id['T1'], 0, 1)  # 0: T1 (left fist/both fists), 1: T2 (right fist/both feet)
            else:
                # No T1/T2 or baseline: Create synthetic 1-second epochs
                print(f"No T1/T2 or baseline run in {file_path}. Creating synthetic 1-second epochs.")
                n_samples = raw.n_times
                epoch_length = int(EPOCH_DURATION * TARGET_SFREQ)  # 256 samples
                step = epoch_length  # Non-overlapping epochs
                if n_samples < epoch_length:
                    print(f"Skipping {file_path}: Too short for 1-second epochs ({n_samples} samples).")
                    return None
                event_times = np.arange(0, n_samples - epoch_length + 1, step)
                events = np.column_stack((event_times, np.zeros_like(event_times), np.ones_like(event_times)))
                event_id = {'unknown': 1}
                epochs = mne.Epochs(raw, events, event_id={'unknown': 1},
                                    tmin=0, tmax=EPOCH_DURATION - 1/TARGET_SFREQ, baseline=None, preload=True)
                data = epochs.get_data()  # (n_epochs, 64, 256)
                labels = np.zeros(data.shape[0], dtype=int)  # 0: unknown/rest
            
            # Extract features
            features_list = []
            for i in range(data.shape[0]):
                features = extract_epoch_features(data[i])
                features_list.append(features)
            
            df = pd.DataFrame(features_list)
            df['label'] = labels
            df['file_source'] = os.path.basename(file_path)
            df['task_type'] = run_type
            
            # Detect outliers (z-score > 3)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df.drop(['label', 'file_source', 'task_type'], axis=1))
            z_scores = np.abs(scaled)
            df['is_outlier'] = (z_scores > 3).any(axis=1).astype(int)
            
            print(f"Processed {file_path}: {len(df)} epochs (including {df['is_outlier'].sum()} flagged outliers, task: {run_type}).")
            
            return df
        except Exception as e:
            print(f"Epoching failed for {file_path}: {str(e)}. Skipping file.")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Find all .edf files and process them
edf_files = glob.glob(os.path.join(RAW_DATA_DIR, '**/*.edf'), recursive=True)
all_data = []

for file in edf_files:
    df = process_edf(file)
    if df is not None:
        all_data.append(df)

# Combine and save to CSV
if all_data:
    n_channels = 64
    feature_names = (
        [f'mean_ch{i}' for i in range(n_channels)] +
        [f'var_ch{i}' for i in range(n_channels)] +
        [f'skew_ch{i}' for i in range(n_channels)] +
        [f'kurt_ch{i}' for i in range(n_channels)] +
        [f'delta_ch{i}' for i in range(n_channels)] +
        [f'theta_ch{i}' for i in range(n_channels)] +
        [f'alpha_ch{i}' for i in range(n_channels)] +
        [f'beta_ch{i}' for i in range(n_channels)] +
        [f'gamma_ch{i}' for i in range(n_channels)]
    )
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.columns = feature_names + ['label', 'file_source', 'task_type', 'is_outlier']
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Raw CSV saved to {OUTPUT_CSV} with {len(combined_df)} rows (including {combined_df['is_outlier'].sum()} flagged outliers).")
else:
    print("No valid data processed.")

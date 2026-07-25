import os
import numpy as np
import pyedflib

def read_event_file(event_path, sample_rate):
    """Read .edf.event file and return list of (position, code) tuples."""
    events = []
    try:
        with open(event_path, "r", encoding="utf-8", errors="ignore") as f:
            print(f"Raw content of {event_path}:")
            content = f.read()
            print(content)
            f.seek(0)
            
            for line in f:
                line = line.strip()
                print(f"Processing line: {line}")
                if not line or line == "X## time resolution: 160":
                    print(f"⚠️ Skipping empty or header-only line: {line}")
                    continue
                
                event_data = line.split("X## time resolution: 160")[-1]
                print(f"Event data after header removal: {event_data}")
                tokens = event_data.split("Z")
                print(f"Tokens: {tokens}")
                current_pos = 0
                
                for token in tokens:
                    token = token.strip()
                    if not token:
                        print(f"⚠️ Skipping empty token")
                        continue
                    print(f"Processing token: {token}")
                    parts = token.split(" duration: ")
                    print(f"Token parts: {parts}")
                    if len(parts) != 2:
                        print(f"⚠️ Skipping malformed token in {event_path}: {token}")
                        continue
                    code, duration = parts
                    code = ''.join(c for c in code if c.isalnum())
                    print(f"Code: {code}, Duration: {duration}")
                    duration = ''.join(c for c in duration if c.isdigit() or c == '.')
                    print(f"Cleaned duration: {duration}")
                    try:
                        duration = float(duration)
                        samples = int(duration * sample_rate)
                        print(f"Converted duration {duration} seconds to {samples} samples")
                        if code in ["T1", "T2"]:
                            print(f"Adding event: (pos={current_pos}, code={code})")
                            events.append((current_pos, code))
                        current_pos += samples
                    except ValueError as e:
                        print(f"⚠️ Error parsing token in {event_path}: {token}, {e}")
                        continue
    except Exception as e:
        print(f"⚠️ Error reading: {event_path}, {e}")
    if not events:
        print(f"⚠️ No valid events found in {event_path}")
    else:
        print(f"Extracted events: {events}")
    return events

def preprocess_subject(subject_id, base_dir="../raw_data"):
    """Preprocess a single subject and return X, y, and sample rate."""
    subject_folder = os.path.join(base_dir, subject_id)
    print(f"Checking directory: {subject_folder}")
    
    if not os.path.exists(subject_folder):
        print(f"Error: Directory {subject_folder} does not exist")
        return None, None, None

    files = [f for f in os.listdir(subject_folder) if f.endswith(".edf")]
    if not files:
        print(f"Error: No .edf files found in {subject_folder}")
        return None, None, None

    all_signals = []
    all_labels = []
    subject_sample_rate = None

    for file in files:
        edf_path = os.path.join(subject_folder, file)
        event_path = edf_path + ".event"

        if not os.path.exists(event_path):
            print(f"⚠️ Event file missing: {event_path}")
            continue

        try:
            f = pyedflib.EdfReader(edf_path)
            n_channels = f.signals_in_file
            n_samples = f.getNSamples()[0]
            sample_rate = f.getSampleFrequency(0)
            if subject_sample_rate is None:
                subject_sample_rate = sample_rate
            elif sample_rate != subject_sample_rate:
                print(f"⚠️ Sample rate mismatch in {edf_path}: expected {subject_sample_rate}, got {sample_rate}")
                f.close()
                continue

            data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                data[ch, :] = f.readSignal(ch)
            f.close()
            print(f"Loaded EDF file {edf_path}: {n_channels} channels, {n_samples} samples, {sample_rate} Hz")
        except Exception as e:
            print(f"⚠️ Error reading EDF file {edf_path}: {e}")
            continue

        events = read_event_file(event_path, sample_rate)
        if not events:
            print(f"⚠️ No valid events found in {event_path}")
            continue
        window_size = int(2 * sample_rate)  # 2-second window
        print(f"Window size: {window_size} samples")

        for pos, code in events:
            start = pos
            end = pos + window_size
            if end > n_samples:
                print(f"⚠️ Skipping segment at pos {pos}: exceeds signal length ({n_samples} samples)")
                continue
            segment = data[:, start:end]
            # Standardize to 256 samples by truncating if necessary
            if segment.shape[1] > 256:
                segment = segment[:, :256]
            print(f"Extracted segment: pos={pos}, code={code}, shape={segment.shape}")

            if code == "T1":
                label = 0  # Left hand
            elif code == "T2":
                label = 1  # Right hand
            else:
                print(f"⚠️ Unknown code {code} at pos {pos}")
                continue

            all_signals.append(segment)
            all_labels.append(label)

    if not all_signals:
        print(f"Error: No valid data processed for {subject_id}")
        return None, None, None

    X = np.array(all_signals)
    y = np.array(all_labels)
    print(f"Processed {subject_id}: X={X.shape}, y={y.shape}, sample_rate={subject_sample_rate}")
    return X, y, sample_rate

def preprocess_all_subjects(base_dir="../raw_data", save_dir="../processed_data"):
    """Preprocess all subjects (S001 to S109) and save combined X_all.npy and y_all.npy."""
    all_X = []
    all_y = []
    processed_subjects = 0
    sample_rates = {}

    for i in range(1, 110):
        subject_id = f"S{i:03d}"
        print(f"\n=== Processing subject {subject_id} ===")
        X, y, sample_rate = preprocess_subject(subject_id, base_dir)
        if X is not None and y is not None and sample_rate is not None:
            all_X.append(X)
            all_y.append(y)
            sample_rates[subject_id] = sample_rate
            processed_subjects += 1
            print(f"Added {subject_id} to combined dataset: {X.shape[0]} segments, sample_rate={sample_rate}")
        else:
            print(f"⚠️ Skipping {subject_id}: No valid data")

    if not all_X:
        print("Error: No valid data processed for any subject")
        return

    # Log sample rates
    print("\nSample rates per subject:")
    for subject_id, sr in sample_rates.items():
        print(f"{subject_id}: {sr} Hz")

    # Combine all subjects' data
    try:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
    except ValueError as e:
        print(f"Error during concatenation: {e}")
        return

    # Save combined dataset
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_all.npy"), X_all)
    np.save(os.path.join(save_dir, "y_all.npy"), y_all)

    print(f"\n✅ Combined dataset saved: X_all={X_all.shape}, y_all={y_all.shape}")
    print(f"Total subjects processed: {processed_subjects}")

if __name__ == "__main__":
    base_dir = "C:/Users/rashi/brainwave_to_text/raw_data"
    save_dir = "C:/Users/rashi/brainwave_to_text/processed_data"
    preprocess_all_subjects(base_dir=base_dir, save_dir=save_dir)
import os
import pyedflib

# Path setup
base_dir = os.path.join("..", "raw_data")   # move one level up from scripts/
subject = "S001"
file_name = "S001R01.edf"
edf_path = os.path.join(base_dir, subject, file_name)

# Open EDF file
f = pyedflib.EdfReader(edf_path)

# Extract metadata
n_channels = f.signals_in_file
signal_labels = f.getSignalLabels()
sample_rate = f.getSampleFrequency(0)

print("✅ File:", file_name)
print("Channels:", n_channels)
print("Labels (first 10):", signal_labels[:10])
print("Sample Rate:", sample_rate)

# Extract a small EEG signal (first channel)
signal = f.readSignal(0)  # channel 0
print("First 10 samples of channel 0:", signal[:10])

f.close()

import mne

EDF_PATH = "C:/Users/rashi/brainwave_to_text/raw_data/S001/S001R01.edf"  # Update with your .edf file path

try:
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True)
    print("Sampling rate:", raw.info['sfreq'])
    print("Number of channels:", len(raw.ch_names))
    print("Channel names:", raw.ch_names)
    print("Data duration (seconds):", raw.times[-1])

    events, event_id = mne.events_from_annotations(raw)
    print("Events array:\n", events)
    print("Event IDs:", event_id)
except Exception as e:
    print(f"Error: {str(e)}")
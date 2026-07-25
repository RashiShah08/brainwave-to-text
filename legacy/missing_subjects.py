import os
base_dir = "C:/Users/rashi/brainwave_to_text/raw_data"
subjects = [f"S{i:03d}" for i in range(1, 110)]
present = [s for s in subjects if os.path.exists(os.path.join(base_dir, s))]
missing = [s for s in subjects if s not in present]
print(f"Missing subjects: {missing}")
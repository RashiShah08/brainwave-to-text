import pandas as pd
import os

RAW_FILE = "C:/Users/rashi/brainwave_to_text/processed_data/raw_features.csv"
OUTPUT_FILE = "C:/Users/rashi/brainwave_to_text/processed_data/cleaned_features.csv"

# Load raw dataset
df = pd.read_csv(RAW_FILE)
print("Raw shape:", df.shape)

# Remove outliers
df = df[df["is_outlier"] == 0]

# Drop metadata
df = df.drop(columns=["file_source", "task_type", "is_outlier"])

# Save cleaned data
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print("✅ Cleaned dataset saved at:", OUTPUT_FILE)
print("New shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())

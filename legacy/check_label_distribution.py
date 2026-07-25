import pandas as pd
df = pd.read_csv("C:/Users/rashi/brainwave_to_text/processed_data/cleaned_features.csv")
print("Label distribution:\n", df["label"].value_counts())  # Share this output
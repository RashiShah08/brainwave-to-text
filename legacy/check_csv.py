
import pandas as pd
import numpy as np
import re

# Path to raw_features.csv
INPUT_CSV = "C:/Users/rashi/brainwave_to_text/processed_data/raw_features.csv"

def inspect_raw_features():
    try:
        # Load the CSV
        df = pd.read_csv(INPUT_CSV)
        print(f"\n=== Dataset Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Number of columns: {len(df.columns)}")

        # Key columns to inspect
        key_columns = ['label', 'task_type', 'file_source', 'is_outlier']
        print(f"\n=== Key Column Distributions ===")
        for col in key_columns:
            if col in df.columns:
                print(f"\n{col} distribution:")
                print(df[col].value_counts(dropna=False))
                print(f"Unique values: {df[col].unique()}")
            else:
                print(f"\n{col} not found in dataset.")

        # Extract run numbers from file_source
        print(f"\n=== Run Number Extraction ===")
        df['run_number'] = df['file_source'].apply(
            lambda x: int(re.search(r'R(\d+)', x).group(1)) if isinstance(x, str) and re.search(r'R(\d+)', x) else -1
        )
        print("Run number distribution:")
        print(df['run_number'].value_counts(dropna=False).sort_index())
        print(f"Unique run numbers: {df['run_number'].unique()}")

        # Label distribution by task_type
        print(f"\n=== Label Distribution by Task Type ===")
        for task in df['task_type'].unique():
            if pd.notna(task):
                print(f"\nTask: {task}")
                print(df[df['task_type'] == task]['label'].value_counts(dropna=False))

        # Label distribution by run number
        print(f"\n=== Label Distribution by Run Number ===")
        for run in sorted(df['run_number'].unique()):
            if run != -1:
                print(f"\nRun {run}:")
                print(df[df['run_number'] == run]['label'].value_counts(dropna=False))
                print(f"Task types in run {run}: {df[df['run_number'] == run]['task_type'].unique()}")

        # Feature value summary (for a subset of features)
        feature_cols = [col for col in df.columns if col not in key_columns + ['run_number']]
        sample_features = [
            'mean_ch0', 'var_ch0', 'skew_ch0', 'kurt_ch0', 'delta_ch0', 
            'theta_ch0', 'alpha_ch0', 'beta_ch0', 'gamma_ch0',
            'mean_ch30', 'var_ch30', 'delta_ch30', 'alpha_ch30', 'beta_ch30'
        ]
        sample_features = [f for f in sample_features if f in feature_cols]
        print(f"\n=== Feature Value Summary (Sample Features) ===")
        if sample_features:
            feature_summary = df[sample_features].describe()
            print(feature_summary)
        else:
            print("No sample features found in dataset.")

        # Check for extreme values or NaNs
        print(f"\n=== Feature Quality Check ===")
        print(f"Number of NaN values in features: {df[feature_cols].isna().sum().sum()}")
        print(f"Number of infinite values in features: {np.isinf(df[feature_cols]).sum().sum()}")
        for col in sample_features:
            print(f"\n{col} - Min: {df[col].min()}, Max: {df[col].max()}, Std: {df[col].std()}")

        # Check for high correlations (sample)
        print(f"\n=== Sample Feature Correlations ===")
        corr_matrix = df[sample_features].corr().abs()
        high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
        high_corr = high_corr[high_corr > 0.95].sort_values(ascending=False)
        print("High correlations (>0.95):")
        print(high_corr.head(10))

    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        raise
    except Exception as e:
        print(f"Error inspecting data: {str(e)}")
        raise

if __name__ == "__main__":
    inspect_raw_features()

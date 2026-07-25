import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ========= Paths =========
CSV_FILE = "C:/Users/rashi/brainwave_to_text/processed_data/cleaned_features.csv"
MODEL_DIR = "C:/Users/rashi/brainwave_to_text/models/"
MODEL_PATH = os.path.join(MODEL_DIR, "final_ensemble.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")
PLOT_PATH_CM = os.path.join(MODEL_DIR, "confusion_matrix_test.png")

# ========= Load Data =========
df = pd.read_csv(CSV_FILE)
X = df.drop("label", axis=1)
y = df["label"].values

# ========= Load Preprocessors & Model =========
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)
model = joblib.load(MODEL_PATH)
print("✅ Model and preprocessors loaded")

# ========= Reorder and fill features exactly like training =========
X_test_df = X.copy()
for col in feature_cols:
    if col not in X_test_df.columns:
        X_test_df[col] = 0
X_test_df = X_test_df[feature_cols]

X_test_scaled = scaler.transform(X_test_df.values)
X_test_pca = pca.transform(X_test_scaled)

# ========= Predict =========
y_pred = model.predict(X_test_pca)

# ========= Evaluate =========
acc = accuracy_score(y, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f}")
print(classification_report(y, y_pred, target_names=["Class 0", "Class 1"]))

# ========= Confusion Matrix =========
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix - Test Data")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(PLOT_PATH_CM)
plt.close()
print("Confusion matrix saved at:", PLOT_PATH_CM)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ========= Paths =========
CSV_FILE = "C:/Users/rashi/brainwave_to_text/processed_data/cleaned_features.csv"
OUTPUT_DIR = "C:/Users/rashi/brainwave_to_text/models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENSEMBLE_PATH = os.path.join(OUTPUT_DIR, "final_ensemble.pkl")
PLOT_PATH_CM = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
PCA_PATH = os.path.join(OUTPUT_DIR, "pca.pkl")
FEATURE_COLS_PATH = os.path.join(OUTPUT_DIR, "feature_cols.pkl")

# ========= Load Data =========
df = pd.read_csv(CSV_FILE)
print("Dataset shape:", df.shape)

X = df.drop("label", axis=1).values
y = df["label"].values   # Already binary: {0,1}

# ========= Train-test split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========= Scale Features =========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========= Dimensionality Reduction (PCA) =========
pca = PCA(n_components=150, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("PCA reduced features to:", X_train.shape[1])

# ========= Model 1: Random Forest =========
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_dist_rf = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", None]
}
search_rf = RandomizedSearchCV(
    rf, param_dist_rf, n_iter=5, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
)
search_rf.fit(X_train, y_train)
best_rf = search_rf.best_estimator_
print("Best RF params:", search_rf.best_params_)
print("RF CV Accuracy:", search_rf.best_score_)

# ========= Model 2: XGBoost =========
xgb = XGBClassifier(
    random_state=42,
    tree_method="hist",
    device="cpu",
    eval_metric="logloss",
    n_jobs=-1
)
param_dist_xgb = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
search_xgb = RandomizedSearchCV(
    xgb, param_dist_xgb, n_iter=5, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
)
search_xgb.fit(X_train, y_train)
best_xgb = search_xgb.best_estimator_
print("Best XGB params:", search_xgb.best_params_)
print("XGB CV Accuracy:", search_xgb.best_score_)

# ========= Model 3: MLP (Neural Net) =========
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42
)
mlp.fit(X_train, y_train)
print("MLP Training Accuracy:", mlp.score(X_train, y_train))

# ========= Evaluate All =========
models = {
    "Random Forest": best_rf,
    "XGBoost": best_xgb,
    "MLP": mlp,
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# ========= Ensemble (soft voting) =========
ensemble = VotingClassifier(
    estimators=[("rf", best_rf), ("xgb", best_xgb), ("mlp", mlp)],
    voting="soft", n_jobs=-1
)
ensemble.fit(X_train, y_train)

y_pred_ens = ensemble.predict(X_test)
acc_ens = accuracy_score(y_test, y_pred_ens)
print(f"\n🔥 Final Ensemble Test Accuracy: {acc_ens:.4f}")
print(classification_report(y_test, y_pred_ens, target_names=["Class 0", "Class 1"]))

# ========= Confusion Matrix =========
cm = confusion_matrix(y_test, y_pred_ens)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix - Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(PLOT_PATH_CM)
plt.close()
print("Confusion matrix saved at:", PLOT_PATH_CM)

# ========= Save Final Model & Preprocessors =========
joblib.dump(ensemble, ENSEMBLE_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(pca, PCA_PATH)
feature_cols = list(df.drop("label", axis=1).columns)
joblib.dump(feature_cols, FEATURE_COLS_PATH)

print("✅ Final ensemble saved at:", ENSEMBLE_PATH)
print("✅ Scaler saved at:", SCALER_PATH)
print("✅ PCA saved at:", PCA_PATH)
print("✅ Feature columns saved at:", FEATURE_COLS_PATH)

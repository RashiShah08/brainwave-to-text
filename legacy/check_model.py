from tensorflow.keras.models import load_model
import os

MODEL_PATH = "C:/Users/rashi/brainwave_to_text/models/shallowconvnet_best.h5"

try:
    model = load_model(MODEL_PATH)
    print("Model input shape:", model.input_shape)
    model.summary()  # This will print the full model architecture for more details
except Exception as e:
    print(f"Error: {str(e)}")
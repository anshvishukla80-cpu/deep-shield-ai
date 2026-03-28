import os
import torch
from app.model.cnn_model import DeepfakeDetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.pth")
MODEL_PATH = os.path.abspath(MODEL_PATH)

def load_model():
    model = DeepfakeDetector(pretrained=False)  # Don't download weights at load time
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  No trained model found at {MODEL_PATH}. Using untrained model.")
    model.eval()
    return model

model = load_model()
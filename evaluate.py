"""
DeepGuard — Confusion Matrix & Evaluation Report
Evaluates the trained model on the validation set and generates a confusion matrix.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)
from app.model.cnn_model import DeepfakeDetector

# ─── Config ───
IMAGE_SIZE = 299
MODEL_PATH = os.path.join(BACKEND_DIR, "app", "saved_models", "model.pth")
VAL_REAL_DIR = os.path.join(BACKEND_DIR, "dataset", "Dataset", "Validation", "Real")
VAL_FAKE_DIR = os.path.join(BACKEND_DIR, "dataset", "Dataset", "Validation", "Fake")
MAX_SAMPLES = 1000  # per class for speed

# ─── Device ───
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🖥️  Using device: {device}")

# ─── Transform (must match training) ───
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ─── Load model ───
print("📦 Loading model...")
model = DeepfakeDetector(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded!\n")

# ─── Collect images ───
IMG_EXTS = (".jpg", ".jpeg", ".png")

def collect_images(folder, max_n=None):
    if not os.path.exists(folder):
        print(f"  ⚠️  Folder not found: {folder}")
        return []
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    if max_n and len(paths) > max_n:
        random.seed(42)
        paths = random.sample(paths, max_n)
    return paths

real_paths = collect_images(VAL_REAL_DIR, MAX_SAMPLES)
fake_paths = collect_images(VAL_FAKE_DIR, MAX_SAMPLES)

print(f"📁 Validation Real: {len(real_paths)}")
print(f"📁 Validation Fake: {len(fake_paths)}")
print(f"📊 Total: {len(real_paths) + len(fake_paths)}\n")

all_paths = real_paths + fake_paths
all_labels = [0] * len(real_paths) + [1] * len(fake_paths)  # 0=Real, 1=Fake

# ─── Run predictions ───
print("🔍 Running predictions...")
predictions = []
total = len(all_paths)

with torch.no_grad():
    for i, path in enumerate(all_paths):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = val_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = 1 if output.item() > 0.5 else 0
            predictions.append(pred)
        except Exception as e:
            predictions.append(0)  # default to real on error

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"   Processed {i+1}/{total} ({100*(i+1)/total:.0f}%)", end="\r")

print(f"\n\n{'='*60}")
print("📊 CONFUSION MATRIX")
print(f"{'='*60}\n")

# ─── Confusion Matrix ───
cm = confusion_matrix(all_labels, predictions)
tn, fp, fn, tp = cm.ravel()

# Pretty print
print("                  Predicted")
print("                Real    Fake")
print(f"  Actual Real   {tn:5d}   {fp:5d}")
print(f"  Actual Fake   {fn:5d}   {tp:5d}")

print(f"\n{'─'*40}")
print(f"  True Positives  (TP): {tp:5d}  (Fake correctly detected)")
print(f"  True Negatives  (TN): {tn:5d}  (Real correctly identified)")
print(f"  False Positives (FP): {fp:5d}  (Real misclassified as Fake)")
print(f"  False Negatives (FN): {fn:5d}  (Fake missed as Real)")

# ─── Metrics ───
accuracy = accuracy_score(all_labels, predictions)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n{'='*60}")
print("📈 CLASSIFICATION METRICS")
print(f"{'='*60}\n")
print(f"  Accuracy:    {accuracy*100:.2f}%")
print(f"  Precision:   {precision*100:.2f}%")
print(f"  Recall:      {recall*100:.2f}%")
print(f"  F1-Score:    {f1*100:.2f}%")
print(f"  Specificity: {specificity*100:.2f}%")

print(f"\n{'='*60}")
print("📋 DETAILED CLASSIFICATION REPORT")
print(f"{'='*60}\n")
print(classification_report(all_labels, predictions, target_names=["Real", "Fake"]))
print(f"{'='*60}")

# ─── Save to file ───
report_path = os.path.join(BACKEND_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("DeepGuard — Evaluation Report\n")
    f.write(f"{'='*60}\n\n")
    f.write("CONFUSION MATRIX\n")
    f.write("                  Predicted\n")
    f.write("                Real    Fake\n")
    f.write(f"  Actual Real   {tn:5d}   {fp:5d}\n")
    f.write(f"  Actual Fake   {fn:5d}   {tp:5d}\n\n")
    f.write(f"Accuracy:    {accuracy*100:.2f}%\n")
    f.write(f"Precision:   {precision*100:.2f}%\n")
    f.write(f"Recall:      {recall*100:.2f}%\n")
    f.write(f"F1-Score:    {f1*100:.2f}%\n")
    f.write(f"Specificity: {specificity*100:.2f}%\n\n")
    f.write(classification_report(all_labels, predictions, target_names=["Real", "Fake"]))

print(f"\n💾 Report saved to: {report_path}")

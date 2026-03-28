"""
DeepGuard Training Script — Combined Image + Video Dataset
Trains Xception-based DeepfakeDetector on:
  • Image dataset:  dataset/Dataset/Train/{Real,Fake}/
  • Video dataset:  dataset/train/{real,fake}/  (extracts frames)
Saves the best model to app/saved_models/model.pth

Optimised for ~1-hour training on Apple Silicon (MPS).
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import time

# Ensure backend is in the path for imports
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
from app.model.cnn_model import DeepfakeDetector

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # Image dataset (capital-D Dataset folder)
    "image_real_dir": os.path.join(BACKEND_DIR, "dataset", "Dataset", "Train", "Real"),
    "image_fake_dir": os.path.join(BACKEND_DIR, "dataset", "Dataset", "Train", "Fake"),

    # Image validation set
    "image_val_real_dir": os.path.join(BACKEND_DIR, "dataset", "Dataset", "Validation", "Real"),
    "image_val_fake_dir": os.path.join(BACKEND_DIR, "dataset", "Dataset", "Validation", "Fake"),

    # Video dataset
    "video_real_dir": os.path.join(BACKEND_DIR, "dataset", "real"),
    "video_fake_dir": os.path.join(BACKEND_DIR, "dataset", "fake"),

    # Where extracted video frames are saved
    "frames_output_dir": os.path.join(BACKEND_DIR, "dataset", "extracted_frames"),
    "frames_per_video":  15,

    # Model is saved where model_loader.py expects it
    "model_save_path": os.path.join(BACKEND_DIR, "app", "saved_models", "model.pth"),

    # Training hyper-parameters (optimised for ~1hr on MPS)
    "epochs":          5,
    "batch_size":      32,
    "learning_rate":   0.0005,
    "image_size":      299,       # Xception needs 299×299
    "val_split":       0.2,

    # Subsample to keep training under 80 minutes on MPS
    "max_samples_per_class":  15000,   # cap per class (real/fake)
    "max_val_per_class":      3000,    # cap for validation
}

# Device — prefer MPS on Apple Silicon, then CUDA, then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🖥️  Using device: {device}")


# ─────────────────────────────────────────────
# STEP 1 — Extract frames from videos
# ─────────────────────────────────────────────
def extract_frames_from_videos(video_dir, output_dir, label, frames_per_video=15):
    if not os.path.exists(video_dir):
        print(f"  ⚠️  Video folder not found, skipping: {video_dir}")
        return 0

    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    # ── FAST PATH: skip if frames already extracted ──
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    if len(existing) > 0:
        print(f"  ✅ {label}: {len(existing)} frames already extracted, skipping.")
        return len(existing)

    # ── SLOW PATH: extract frames from videos ──
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [f for f in os.listdir(video_dir)
                   if f.lower().endswith(video_exts)]

    if not video_files:
        print(f"  ⚠️  No videos found in {video_dir}")
        return 0

    total_saved = 0
    print(f"\n📹 Extracting from {len(video_files)} {label} videos...")

    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            continue

        indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
        base_name = os.path.splitext(video_file)[0]
        saved = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            save_path = os.path.join(save_dir, f"{base_name}_frame{idx}.jpg")
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, frame)
            saved += 1

        cap.release()
        total_saved += saved

    print(f"  ✅ Done. Total frames saved for {label}: {total_saved}")
    return total_saved


# ─────────────────────────────────────────────
# STEP 2 — Collect all image paths + labels
# ─────────────────────────────────────────────
def load_all_images(cfg):
    real_paths, fake_paths = [], []
    IMG_EXTS = (".jpg", ".jpeg", ".png")

    def collect(folder):
        if not os.path.exists(folder):
            print(f"  Skipping missing folder: {folder}")
            return []
        paths = []
        for fname in os.listdir(folder):
            if fname.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(folder, fname))
        return paths

    # Collect from image dataset (Dataset/Train)
    r1 = collect(cfg["image_real_dir"])
    f1 = collect(cfg["image_fake_dir"])
    print(f"\n📁 Train images    →  Real: {len(r1)}  Fake: {len(f1)}")

    # Collect from extracted video frames
    r2 = collect(os.path.join(cfg["frames_output_dir"], "real"))
    f2 = collect(os.path.join(cfg["frames_output_dir"], "fake"))
    print(f"📁 Video frames    →  Real: {len(r2)}  Fake: {len(f2)}")

    real_paths = r1 + r2
    fake_paths = f1 + f2
    print(f"📊 Full dataset    →  Real: {len(real_paths)}  Fake: {len(fake_paths)}  All: {len(real_paths)+len(fake_paths)}")

    # ── Subsample for time constraint ──
    cap = cfg.get("max_samples_per_class", None)
    if cap and len(real_paths) > cap:
        random.seed(42)
        real_paths = random.sample(real_paths, cap)
        print(f"🔽 Subsampled real  →  {cap}")
    if cap and len(fake_paths) > cap:
        random.seed(42)
        fake_paths = random.sample(fake_paths, cap)
        print(f"🔽 Subsampled fake  →  {cap}")

    image_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    print(f"📊 Using           →  Real: {len(real_paths)}  Fake: {len(fake_paths)}  All: {len(image_paths)}")

    return image_paths, labels


def load_validation_images(cfg):
    """Load the dedicated validation images from Dataset/Validation/."""
    IMG_EXTS = (".jpg", ".jpeg", ".png")

    def collect(folder):
        if not os.path.exists(folder):
            return []
        return [os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(IMG_EXTS)]

    real_paths = collect(cfg["image_val_real_dir"])
    fake_paths = collect(cfg["image_val_fake_dir"])

    # Cap validation size for speed
    val_cap = cfg.get("max_val_per_class", None)
    if val_cap:
        random.seed(42)
        if len(real_paths) > val_cap:
            real_paths = random.sample(real_paths, val_cap)
        if len(fake_paths) > val_cap:
            fake_paths = random.sample(fake_paths, val_cap)

    image_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    print(f"📁 Validation set  →  Real: {len(real_paths)}  Fake: {len(fake_paths)}")

    return image_paths, labels


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (299, 299))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ─────────────────────────────────────────────
# TRANSFORMS  (must match preprocessing.py at inference!)
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ─────────────────────────────────────────────
# WEIGHTED SAMPLER — fixes class imbalance
# ─────────────────────────────────────────────
def get_sampler(labels):
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels_arr]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, epoch_info=""):
    is_train = optimizer is not None
    phase = "Train" if is_train else "Val"
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)
    epoch_start = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start
                batches_done = batch_idx + 1
                rate = elapsed / batches_done
                eta = rate * (num_batches - batches_done)
                acc_so_far = 100.0 * correct / total if total > 0 else 0
                print(f"    {epoch_info} {phase} Batch {batches_done}/{num_batches}  "
                      f"loss={loss.item():.4f}  acc={acc_so_far:.1f}%  "
                      f"ETA={eta:.0f}s", end="\r")

    print(" " * 100, end="\r")  # clear line
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    training_start = time.time()
    print("=" * 60)
    print("🚀 DeepGuard Training — Xception (FAST MODE)")
    print(f"   Config: {CONFIG['epochs']} epochs, batch={CONFIG['batch_size']}, "
          f"max {CONFIG['max_samples_per_class']} samples/class")
    print("=" * 60)

    # ── 1. Extract frames from videos
    print("\n========== STEP 1: Extracting Video Frames ==========")
    extract_frames_from_videos(
        CONFIG["video_real_dir"], CONFIG["frames_output_dir"],
        "real", CONFIG["frames_per_video"]
    )
    extract_frames_from_videos(
        CONFIG["video_fake_dir"], CONFIG["frames_output_dir"],
        "fake", CONFIG["frames_per_video"]
    )

    # ── 2. Load all images (train + video frames)
    print("\n========== STEP 2: Loading Dataset ==========")
    train_paths, train_labels = load_all_images(CONFIG)

    if len(train_paths) == 0:
        print("❌ No training images found! Check folder paths.")
        return

    # Load dedicated validation set if available
    val_paths, val_labels = load_validation_images(CONFIG)

    if len(val_paths) > 0:
        # Use the dedicated validation set
        X_train, y_train = train_paths, train_labels
        X_val, y_val = val_paths, val_labels
    else:
        # Fall back to random split
        X_train, X_val, y_train, y_val = train_test_split(
            train_paths, train_labels,
            test_size=CONFIG["val_split"],
            stratify=train_labels,
            random_state=42
        )

    # ── 3. DataLoaders  (num_workers=0 for macOS stability)
    train_dataset = DeepfakeDataset(X_train, y_train, transform=train_transform)
    val_dataset = DeepfakeDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              sampler=get_sampler(y_train), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0)

    print(f"\n📊 Train: {len(train_dataset)} samples ({len(train_loader)} batches)  |  "
          f"Val: {len(val_dataset)} samples ({len(val_loader)} batches)\n")

    # ── 4. Model / Loss / Optimizer
    model = DeepfakeDetector(pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    best_val_acc = 0.0

    # ── 5. Training loop
    print("========== STEP 3: Training ==========\n")
    epochs = CONFIG["epochs"]
    epoch_times = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer,
            epoch_info=f"[Epoch {epoch}/{epochs}]")
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            epoch_info=f"[Epoch {epoch}/{epochs}]")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_elapsed = time.time() - training_start
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        eta_total = avg_epoch * remaining_epochs

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"📈 Epoch {epoch}/{epochs}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  |  LR: {lr:.6f}")
        print(f"   ⏱️  Epoch: {epoch_time/60:.1f}min  |  "
              f"Elapsed: {total_elapsed/60:.1f}min  |  "
              f"ETA: {eta_total/60:.1f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"   💾 Best model saved! (Val Acc: {val_acc:.2f}%)")
        print()

    total_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"✅ Training complete in {total_time/60:.1f} minutes!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {CONFIG['model_save_path']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
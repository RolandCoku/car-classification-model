# Generates car_view_classification_and_segmentation.ipynb
# Run: python3 build_notebook.py
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})

# ---------------------------------------------------------------------------
md(r"""# Automated Car Image Classification and Segmentation

**Module 4: Deep Learning — Final Project**
Instructor: Christian Mata, PhD · CV Tirana 2026

Submission type: Individual project

---

## Project context

The idea for this project comes from a personal car-marketplace application that I
am developing. Sellers upload several photographs for each vehicle listing, but the
images are not automatically organized by view type.

This project applies deep learning to that image-upload workflow. The first model
classifies each uploaded photograph as **interior** or **exterior**. The second model
segments the vehicle from its background.

The classification output can be used to organize listing galleries and select an
exterior cover photograph. The segmentation mask can be used to create clean and
consistent vehicle images.

## Project objectives

1. Train a binary image classifier using MobileNetV2 transfer learning.
2. Evaluate the classifier with accuracy, precision, recall, F1-score, a confusion
   matrix, learning curves, and an independent image set.
3. Train a U-Net semantic-segmentation model using paired car images and masks.
4. Evaluate segmentation with Dice coefficient, Intersection over Union, learning
   curves, and prediction visualizations.
5. Save both trained models for use by the marketplace application.

> **Runtime:** designed for **Google Colab with a GPU** (`Runtime → Change runtime type → GPU`),
> and can also be executed in a local Python environment.""")

# ---------------------------------------------------------------------------
md(r"""## 0. Environment & GPU check""")

code(r"""import sys, os, platform
import numpy as np
import tensorflow as tf

print("Python      :", sys.version.split()[0])
print("Platform    :", platform.platform())
print("TensorFlow  :", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs found  :", gpus if gpus else "NONE (training will be slow — enable a GPU runtime)")

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)""")

# ---------------------------------------------------------------------------
md(r"""## 1. Dataset access

The classification dataset is `Dinusharg/Car_interior_exterior_v1`, hosted on
Hugging Face.
Before running this notebook:

1. Open the dataset page and accept its access conditions.
2. Create a Hugging Face **read token** at *Settings → Access Tokens*.
3. Paste the token when prompted below, or set the `HF_TOKEN` environment variable.

The segmentation dataset is downloaded from Kaggle. Kaggle credentials are requested
when Part B is executed.""")

code(r"""import os, sys, getpass

!pip -q install --upgrade huggingface_hub kaggle

# Authenticate for the gated classification dataset.
from huggingface_hub import get_token, login

hf_token = os.environ.get("HF_TOKEN", "").strip() or get_token()
if not hf_token:
    hf_token = getpass.getpass("Paste your Hugging Face read token: ").strip()
assert hf_token, "A Hugging Face token is required for the gated classification dataset."
login(token=hf_token, add_to_git_credential=False)
print("Hugging Face authentication configured.")

# Configure Kaggle only when credentials are not already available.
kaggle_dir = os.path.expanduser("~/.kaggle")
kaggle_token_file = os.path.join(kaggle_dir, "access_token")
kaggle_json_file  = os.path.join(kaggle_dir, "kaggle.json")
kaggle_ready = (os.path.exists(kaggle_token_file) or os.path.exists(kaggle_json_file)
                or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))
print("Kaggle credentials:", "detected" if kaggle_ready else "not configured (needed only for Part B)")""")

# ---------------------------------------------------------------------------
md(r"""## 2. Configuration

This section defines the dataset locations, image dimensions, batch sizes, sample
counts, and training epochs used by both models.""")

code(r"""from pathlib import Path

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---- Classification ----
CLASSIFICATION_DATASET = "Dinusharg/Car_interior_exterior_v1"
CLS_RAW_DIR = DATA_DIR / "raw_classification"
CLS_DIR     = DATA_DIR / "classification"  # normalized as CLS_DIR/{interior,exterior}/*
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
MIN_TOTAL_IMAGES = 1000       # minimum accepted classification dataset size
EPOCHS_HEAD = 6               # train classifier head (frozen backbone)
EPOCHS_FINE = 4               # fine-tune top of backbone

# ---- Segmentation ----
SEG_DATASET     = "ipythonx/carvana-image-masking-png"  # Carvana cars + ground-truth masks
SEG_DIR         = DATA_DIR / "carvana"
SEG_SIZE        = (128, 128)
SEG_BATCH       = 16
SEG_MAX_SAMPLES = 1200        # subset for a reasonable Colab runtime
SEG_EPOCHS      = 12

print("Config loaded. Data ->", DATA_DIR.resolve())""")

# ---------------------------------------------------------------------------
md(r"""---
# Part A — Interior vs. Exterior Classification

The classification task uses the `Dinusharg/Car_interior_exterior_v1` dataset. It
contains approximately 1,000 car photographs organized into interior and exterior
classes.

The preparation pipeline verifies the class counts, rejects unreadable files,
removes exact duplicates, and creates deterministic, stratified training,
validation, and test sets.""")

md(r"""### 2.1 Download the classification dataset""")

code(r"""from huggingface_hub import snapshot_download

CLS_RAW_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=CLASSIFICATION_DATASET,
    repo_type="dataset",
    local_dir=str(CLS_RAW_DIR),
    token=hf_token,
)
print("Classification dataset ready ->", CLS_RAW_DIR)

def kaggle_download(slug, dest):
    global kaggle_ready
    if not kaggle_ready:
        token = getpass.getpass("Paste your Kaggle access token (KGAT_...) for Part B: ").strip()
        assert token, "Kaggle credentials are required to download the segmentation dataset."
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_token_file, "w") as f:
            f.write(token + "\n")
        os.chmod(kaggle_token_file, 0o600)
        kaggle_ready = True
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[skip] {slug} already present in {dest}")
        return
    print(f"[download] {slug} -> {dest}")
    !kaggle datasets download -d {slug} -p {dest} --unzip""")

md(r"""### 2.2 Inspect what was extracted

This cell displays the downloaded directory structure and confirms the locations of
the class images.""")

code(r"""def peek(root, max_lines=25):
    root = Path(root)
    print(f"### {root} ###")
    n = 0
    for p in sorted(root.rglob("*")):
        if n >= max_lines:
            print("   ...")
            break
        rel = p.relative_to(root)
        print("  ", rel if p.is_file() else f"[{rel}]/")
        n += 1
    print()

peek(CLS_RAW_DIR)""")

md(r"""### 2.3 Organize into `data/classification/{interior,exterior}`

We recursively collect images beneath folders named `interior` or `exterior`,
reject unreadable files and exact duplicates, then copy all valid images into a
clean class-folder structure.""")

code(r"""import hashlib, shutil, random
from collections import Counter
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["exterior", "interior"]

def infer_class(path):
    for part in reversed(path.parts[:-1]):
        label = part.lower()
        if label in CLASS_NAMES:
            return label
    return None

for class_name in CLASS_NAMES:
    out = CLS_DIR / class_name
    out.mkdir(parents=True, exist_ok=True)
    for old_file in out.glob("*"):
        old_file.unlink()

counts = Counter()
seen_hashes = set()
skipped = Counter()
for src in sorted(CLS_RAW_DIR.rglob("*")):
    if not src.is_file() or src.suffix.lower() not in IMG_EXTS:
        continue
    class_name = infer_class(src)
    if class_name is None:
        skipped["unlabeled"] += 1
        continue
    try:
        with Image.open(src) as image:
            image.verify()
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
    except Exception:
        skipped["unreadable"] += 1
        continue
    if digest in seen_hashes:
        skipped["duplicate"] += 1
        continue
    seen_hashes.add(digest)
    index = counts[class_name]
    shutil.copy2(src, CLS_DIR / class_name / f"{class_name}_{index:05d}{src.suffix.lower()}")
    counts[class_name] += 1

print("Valid class counts:", dict(counts))
print("Skipped files:", dict(skipped))
assert all(counts[name] > 0 for name in CLASS_NAMES), f"Missing a class: {dict(counts)}"
assert sum(counts.values()) >= MIN_TOTAL_IMAGES, (
    f"Expected at least {MIN_TOTAL_IMAGES} valid images total; got {dict(counts)}"
)""")

md(r"""### 2.4 Visualize class samples""")

code(r"""import matplotlib.pyplot as plt
from PIL import Image

def show_samples(class_name, k=4):
    files = list((CLS_DIR / class_name).glob("*"))[:k]
    fig, axes = plt.subplots(1, k, figsize=(3*k, 3))
    for ax, f in zip(axes, files):
        ax.imshow(Image.open(f).convert("RGB"))
        ax.set_title(class_name); ax.axis("off")
    plt.tight_layout(); plt.show()

show_samples("exterior")
show_samples("interior")""")

md(r"""### 2.5 Build train / validation / test datasets

We explicitly assign **`exterior` = 0** and **`interior` = 1**, then create a
deterministic, stratified 70/15/15 train/validation/test split. The assertions below
guarantee that no file appears in more than one split.""")

code(r"""from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class_names = CLASS_NAMES
all_paths, all_labels = [], []
for label, class_name in enumerate(class_names):
    paths = sorted(str(p) for p in (CLS_DIR / class_name).glob("*") if p.suffix.lower() in IMG_EXTS)
    all_paths.extend(paths)
    all_labels.extend([label] * len(paths))

train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
    all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    holdout_paths, holdout_labels, test_size=0.50, random_state=SEED, stratify=holdout_labels)

assert set(train_paths).isdisjoint(val_paths)
assert set(train_paths).isdisjoint(test_paths)
assert set(val_paths).isdisjoint(test_paths)

def split_counts(labels):
    counts = Counter(labels)
    return {class_names[i]: counts[i] for i in range(len(class_names))}

print("Classes:", class_names, "(index 0, 1)")
print("Train:", len(train_paths), split_counts(train_labels))
print("Val  :", len(val_paths), split_counts(val_labels))
print("Test :", len(test_paths), split_counts(test_labels))

AUTOTUNE = tf.data.AUTOTUNE

def load_classification_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape(IMG_SIZE + (3,))
    return image, tf.cast(label, tf.float32)

def make_classification_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_classification_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_classification_ds(train_paths, train_labels, training=True)
val_ds   = make_classification_ds(val_paths, val_labels)
test_ds  = make_classification_ds(test_paths, test_labels)

weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)),
                               y=np.array(train_labels))
class_weight = {i: float(weight) for i, weight in enumerate(weights)}
print("Training class weights:", class_weight)""")

md(r"""### 2.6 Data augmentation

Random flips/rotations/zoom make the model robust and reduce overfitting to each
source's visual patterns.""")

code(r"""from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Preview augmentation on one image
for images, _ in train_ds.take(1):
    plt.figure(figsize=(10, 3))
    for i in range(4):
        aug = data_augmentation(tf.expand_dims(images[0], 0))
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(tf.cast(aug[0], tf.uint8)); ax.axis("off")
    plt.suptitle("Augmentation preview"); plt.show()""")

md(r"""### 2.7 Build the model — MobileNetV2 transfer learning

MobileNetV2 provides an ImageNet-pretrained feature extractor. A global-average
pooling layer, dropout layer, and sigmoid output layer form the binary
classification head.""")

code(r"""from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False, weights="imagenet")
base_model.trainable = False  # freeze backbone for the first stage

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)                       # scales to [-1, 1] for MobileNetV2
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)   # binary
model = tf.keras.Model(inputs, outputs, name="car_view_classifier")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()""")

md(r"""### 2.8 Stage 1 — train the head (frozen backbone)""")

code(r"""early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True)

hist_head = model.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_HEAD, callbacks=[early],
                      class_weight=class_weight)""")

md(r"""### 2.9 Stage 2 — fine-tune the top of the backbone

Unfreeze the last block of MobileNetV2 and continue with a **much lower** learning
rate to adapt the pretrained features to the classification dataset.""")

code(r"""base_model.trainable = True
# Keep early layers frozen; fine-tune only the last ~30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

pre_fine_val_loss = float(model.evaluate(val_ds, verbose=0)[0])
pre_fine_weights = model.get_weights()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # low LR for fine-tuning
              loss="binary_crossentropy",
              metrics=["accuracy"])

hist_fine = model.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_FINE, callbacks=[early],
                      class_weight=class_weight)

post_fine_val_loss = float(model.evaluate(val_ds, verbose=0)[0])
if post_fine_val_loss > pre_fine_val_loss:
    model.set_weights(pre_fine_weights)
    print(f"Selected frozen-backbone weights by validation loss "
          f"({pre_fine_val_loss:.4f} < {post_fine_val_loss:.4f}).")
else:
    print(f"Selected fine-tuned weights by validation loss "
          f"({post_fine_val_loss:.4f} <= {pre_fine_val_loss:.4f}).")""")

md(r"""### 2.10 Training curves""")

code(r"""def plot_history(histories, keys=("accuracy", "loss")):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc      += h.history.get("accuracy", [])
        val_acc  += h.history.get("val_accuracy", [])
        loss     += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "o-", label="train"); plt.plot(epochs, val_acc, "o-", label="val")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "o-", label="train"); plt.plot(epochs, val_loss, "o-", label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout(); plt.show()

plot_history([hist_head, hist_fine])""")

md(r"""### 2.11 Evaluate on the held-out test set

Accuracy, a confusion matrix, and a full precision/recall/F1 report.""")

code(r"""from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

y_true, y_prob = [], []
for images, labels in test_ds:
    y_true.extend(labels.numpy().flatten())
    y_prob.extend(model.predict(images, verbose=0).flatten())
y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)
y_pred = (y_prob > 0.5).astype(int)

test_acc = (y_pred == y_true).mean()
print(f"Test accuracy: {test_acc:.3f}\n")
print(classification_report(y_true, y_pred, target_names=class_names))

misses = np.where(y_pred != y_true)[0]
print(f"Misclassified test images: {len(misses)}")
for idx in misses[:20]:
    pred = class_names[int(y_pred[idx])]
    true = class_names[int(y_true[idx])]
    conf = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
    print(f"  {Path(test_paths[idx]).name}: pred={pred} true={true} confidence={conf:.2%}")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], labels=class_names); ax.set_yticks([0, 1], labels=class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im); plt.tight_layout(); plt.show()""")

md(r"""### 2.12 Visualize sample predictions""")

code(r"""for images, labels in test_ds.take(1):
    probs = model.predict(images, verbose=0).flatten()
    plt.figure(figsize=(12, 6))
    for i in range(min(8, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        ax.imshow(tf.cast(images[i], tf.uint8)); ax.axis("off")
        pred = class_names[int(probs[i] > 0.5)]
        true = class_names[int(labels[i].numpy())]
        ok = "OK" if pred == true else "X"
        ax.set_title(f"{ok} pred:{pred}\ntrue:{true} ({probs[i]:.2f})",
                     color="green" if pred == true else "red", fontsize=9)
    plt.tight_layout(); plt.show()""")

md(r"""### 2.13 Marketplace integration example

The `classify_image()` function demonstrates how the trained classifier can assign
an interior or exterior category to a photograph uploaded to the marketplace.""")

code(r"""def classify_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    prob = float(model.predict(arr, verbose=0).flatten()[0])
    label = class_names[int(prob > 0.5)]
    conf = prob if prob > 0.5 else 1 - prob
    print(f"{Path(path).name}: {label}  (confidence {conf:.2%})")
    return label, conf

# Example: classify one of the test images we already have on disk
sample = next(iter((CLS_DIR / 'interior').glob('*')))
classify_image(sample)""")

md(r"""### 2.14 Save the classifier""")

code(r"""cls_path = MODELS_DIR / "car_view_classifier.keras"
model.save(cls_path)
print("Saved ->", cls_path)""")

md(r"""### 2.15 Independent classifier evaluation

The repository includes 14 manually labeled car photographs that are separate from
the training dataset: 7 interior images and 7 exterior images. This section measures
classification accuracy on those photographs and displays every prediction.

When running in Google Colab, upload the files from `assets/ood_test/` when
requested.""")

code(r"""# Load the independent evaluation images. The label is read from the filename
# prefix: interior_*.jpg or exterior_*.jpg.
import sys, glob
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

def gather_ood():
    found, seen = [], set()
    # Look wherever the files might be: uploaded to the working dir, or in the repo.
    for base in [".", "ood_test", "assets/ood_test"]:
        for p in sorted(glob.glob(f"{base}/interior_*.jpg") + glob.glob(f"{base}/exterior_*.jpg")):
            rp = str(Path(p).resolve())
            if rp in seen or not Path(p).is_file():
                continue
            seen.add(rp)
            label = "interior" if Path(p).name.startswith("interior") else "exterior"
            found.append((p, label))
    return found

ood_paths = gather_ood()

# On Colab the repo isn't present, so prompt for an upload of the 14 files.
if not ood_paths and IN_COLAB:
    print("Upload the 14 images from assets/ood_test/ (select them all)...")
    from google.colab import files
    files.upload()
    ood_paths = gather_ood()

n_int = sum(l == "interior" for _, l in ood_paths)
n_ext = sum(l == "exterior" for _, l in ood_paths)
print(f"Loaded {len(ood_paths)} test images ({n_int} interior, {n_ext} exterior).")
assert ood_paths, "No interior_*/exterior_* images found — upload them from assets/ood_test/ and re-run."
""")

code(r"""# Predict and score the independent evaluation images.
import math

correct = 0
cols = 4
rows = math.ceil(len(ood_paths) / cols)
plt.figure(figsize=(4 * cols, 3.2 * rows))
for k, (path, true_label) in enumerate(ood_paths):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    prob = float(model.predict(arr, verbose=0).flatten()[0])
    pred = class_names[int(prob > 0.5)]
    conf = prob if prob > 0.5 else 1 - prob
    ok = (pred == true_label)
    correct += ok
    ax = plt.subplot(rows, cols, k + 1)
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{'OK' if ok else 'X'} {pred} ({conf:.2f})\ntrue: {true_label}",
                 color="green" if ok else "red", fontsize=9)
plt.tight_layout(); plt.show()

if ood_paths:
    acc = correct / len(ood_paths)
    print(f"Independent evaluation accuracy: {correct}/{len(ood_paths)} = {acc:.0%}")""")

# ---------------------------------------------------------------------------
md(r"""---
# Part B — Car Segmentation with U-Net

Semantic segmentation predicts whether each image pixel belongs to the car or the
background. The U-Net is trained on the Carvana Image Masking dataset, which
contains vehicle photographs paired with ground-truth masks.

For the marketplace application, the predicted mask provides the information needed
to isolate the vehicle and create a standardized listing image.""")

md(r"""### 3.1 Download the Carvana dataset""")

code(r"""kaggle_download(SEG_DATASET, SEG_DIR)
peek(SEG_DIR, max_lines=30)""")

md(r"""### 3.2 Pair images with their masks

The cell auto-detects the image folder and mask folder (mask folders/files usually
contain the word `mask`) and matches each image to its mask by filename stem.
Only image-mask pairs with matching identifiers are included.""")

code(r"""def find_dirs(root):
    root = Path(root)
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    mask_dirs = [d for d in dirs if "mask" in d.name.lower() and any(d.iterdir())]
    img_dirs  = [d for d in dirs if "mask" not in d.name.lower()
                 and any(f.suffix.lower() in IMG_EXTS for f in d.glob("*"))]
    # pick the most populated candidate of each kind
    def biggest(cands):
        return max(cands, key=lambda d: sum(1 for _ in d.glob("*")), default=None)
    return biggest(img_dirs), biggest(mask_dirs)

img_root, mask_root = find_dirs(SEG_DIR)
print("Image dir:", img_root)
print("Mask  dir:", mask_root)

def stem_key(name):
    # strip a trailing _mask and the extension so image/mask stems line up
    s = Path(name).stem
    return s[:-5] if s.endswith("_mask") else s

masks_by_key = {stem_key(p.name): p for p in mask_root.glob("*") if p.suffix.lower() in IMG_EXTS}
pairs = []
for img in img_root.glob("*"):
    if img.suffix.lower() not in IMG_EXTS:
        continue
    m = masks_by_key.get(stem_key(img.name))
    if m is not None:
        pairs.append((str(img), str(m)))

random.Random(SEED).shuffle(pairs)
pairs = pairs[:SEG_MAX_SAMPLES]
print(f"Matched {len(pairs)} image/mask pairs (using up to {SEG_MAX_SAMPLES}).")
assert pairs, "Dataset validation failed: no matching image-mask pairs were found."
""")

md(r"""### 3.3 Build the tf.data pipeline""")

code(r"""img_paths  = [p[0] for p in pairs]
mask_paths = [p[1] for p in pairs]

n_val = max(1, int(0.15 * len(pairs)))
train_pairs = (img_paths[n_val:], mask_paths[n_val:])
val_pairs   = (img_paths[:n_val], mask_paths[:n_val])

def load_pair(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, SEG_SIZE) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, SEG_SIZE, method="nearest")
    # Carvana masks may be encoded as 0/1 or 0/255 depending on the archive.
    mask = tf.cast(mask > 0, tf.float32)     # binarize -> {0,1}
    return img, mask

def make_ds(pair_lists, training):
    ds = tf.data.Dataset.from_tensor_slices(pair_lists)
    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(256)
    return ds.batch(SEG_BATCH).prefetch(AUTOTUNE)

seg_train = make_ds(train_pairs, True)
seg_val   = make_ds(val_pairs, False)
print("Seg train/val pairs:", len(train_pairs[0]), len(val_pairs[0]))

for _, sample_masks in seg_train.take(1):
    foreground = float(tf.reduce_mean(sample_masks).numpy())
    print(f"Sample mask foreground fraction: {foreground:.3f}")
    assert foreground > 0.01, "Dataset validation failed: foreground mask pixels were not detected." """)

md(r"""### 3.4 Visualize an image + its mask""")

code(r"""for imgs, masks in seg_train.take(1):
    plt.figure(figsize=(9, 3))
    for i in range(3):
        plt.subplot(2, 3, i + 1); plt.imshow(imgs[i]); plt.axis("off")
        plt.title("image")
        plt.subplot(2, 3, i + 4); plt.imshow(masks[i, ..., 0], cmap="gray"); plt.axis("off")
        plt.title("mask")
    plt.tight_layout(); plt.show()""")

md(r"""### 3.5 Define the U-Net

A compact encoder-decoder network uses skip connections to combine semantic
information from the encoder with spatial detail in the decoder.""")

code(r"""from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet(input_size=SEG_SIZE + (3,), base=32):
    inputs = tf.keras.Input(input_size)
    # Encoder
    c1 = conv_block(inputs, base);      p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base*2);        p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base*4);        p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base*8);        p4 = layers.MaxPooling2D()(c4)
    # Bottleneck
    bn = conv_block(p4, base*16)
    # Decoder
    u4 = layers.Conv2DTranspose(base*8, 2, strides=2, padding="same")(bn)
    u4 = layers.concatenate([u4, c4]); c5 = conv_block(u4, base*8)
    u3 = layers.Conv2DTranspose(base*4, 2, strides=2, padding="same")(c5)
    u3 = layers.concatenate([u3, c3]); c6 = conv_block(u3, base*4)
    u2 = layers.Conv2DTranspose(base*2, 2, strides=2, padding="same")(c6)
    u2 = layers.concatenate([u2, c2]); c7 = conv_block(u2, base*2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2, padding="same")(c7)
    u1 = layers.concatenate([u1, c1]); c8 = conv_block(u1, base)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return Model(inputs, outputs, name="unet")

unet = build_unet()
unet.summary()""")

md(r"""### 3.6 Loss & metrics — Dice and IoU

Pixel accuracy is misleading when most pixels are background, so we optimize a
**BCE + Dice** loss and track the **Dice coefficient** and **IoU**.""")

code(r"""def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)

unet.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
             loss=bce_dice_loss,
             metrics=[dice_coef, tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou")])""")

md(r"""### 3.7 Train the U-Net""")

code(r"""seg_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_dice_coef", mode="max", patience=4, restore_best_weights=True)

seg_hist = unet.fit(seg_train, validation_data=seg_val,
                    epochs=SEG_EPOCHS, callbacks=[seg_early])""")

md(r"""### 3.8 Segmentation training curves""")

code(r"""h = seg_hist.history
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(h["dice_coef"], "o-", label="train"); plt.plot(h["val_dice_coef"], "o-", label="val")
plt.title("Dice coefficient"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(h["loss"], "o-", label="train"); plt.plot(h["val_loss"], "o-", label="val")
plt.title("Loss (BCE + Dice)"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.show()

val_metrics = unet.evaluate(seg_val, verbose=0, return_dict=True)
print({name: round(float(value), 4) for name, value in val_metrics.items()})""")

md(r"""### 3.9 Visualize predictions — image · ground truth · prediction · overlay""")

code(r"""for imgs, masks in seg_val.take(1):
    preds = unet.predict(imgs, verbose=0)
    n = min(4, len(imgs))
    plt.figure(figsize=(12, 3 * n))
    for i in range(n):
        pred_bin = (preds[i, ..., 0] > 0.5).astype("float32")
        plt.subplot(n, 4, i*4 + 1); plt.imshow(imgs[i]); plt.axis("off"); plt.title("image")
        plt.subplot(n, 4, i*4 + 2); plt.imshow(masks[i, ..., 0], cmap="gray"); plt.axis("off"); plt.title("ground truth")
        plt.subplot(n, 4, i*4 + 3); plt.imshow(pred_bin, cmap="gray"); plt.axis("off"); plt.title("prediction")
        overlay = imgs[i].numpy().copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + 0.4 * pred_bin, 0, 1)  # tint car red
        plt.subplot(n, 4, i*4 + 4); plt.imshow(overlay); plt.axis("off"); plt.title("overlay")
    plt.tight_layout(); plt.show()""")

md(r"""### 3.10 Save the U-Net""")

code(r"""seg_path = MODELS_DIR / "car_segmentation_unet.keras"
unet.save(seg_path)
print("Saved ->", seg_path)""")

# ---------------------------------------------------------------------------
md(r"""---
# Project Summary

This project implements two deep-learning workflows for a car-marketplace image
pipeline.

The MobileNetV2 classifier assigns interior or exterior categories to uploaded car
photographs. Its performance is presented through test-set metrics, a confusion
matrix, learning curves, sample predictions, and an independent evaluation set.

The U-Net model produces a binary mask of the vehicle. Its performance is presented
through Dice coefficient, Intersection over Union, learning curves, and visual
comparisons between ground-truth and predicted masks.

Together, the two models demonstrate how image classification and semantic
segmentation can be integrated into the workflow of a practical software
application.""")

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("car_view_classification_and_segmentation.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote notebook with {len(cells)} cells.")

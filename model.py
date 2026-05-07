#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Jaivesh50/AI-STEEL-PROJECT/blob/main/Untitled1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import kagglehub
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print("Dataset path:", path)

import os
# Find where class folders are
for root, dirs, files in os.walk(path):
    if any(d.lower() == 'crazing' for d in dirs):
        print("Data root:", root)
        print("Folders:", dirs)
        break


import os, shutil, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# In[ ]:


# Paste your exact RAW_DIR from Cell 1 output
import os, shutil, random
from sklearn.model_selection import train_test_split

# Base dataset path
DATASET_BASE = "/kaggle/input/neu-surface-defect-database/NEU-DET"
CLASSES      = ["crazing", "inclusion", "patches",
                "pitted_surface", "rolled-in_scale", "scratches"]
IMAGE_EXT    = ('.jpg', '.jpeg', '.png', '.bmp')

BASE_DIR  = "/content/NEU_Steel"
TRAIN_DIR = BASE_DIR + "/train"
VAL_DIR   = BASE_DIR + "/val"
TEST_DIR  = BASE_DIR + "/test"

# --- First, let's see ALL available subfolders in the dataset ---
print("Full dataset structure:")
for root, dirs, files in os.walk(DATASET_BASE):
    level = root.replace(DATASET_BASE, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        for f in files[:3]:
            print(f"{indent}  {f}")
# --- Collect ALL images across every split folder ---
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)
    return np.clip(img + noise, 0, 1)
def find_all_images_for_class(dataset_base, class_name):
    """Walk entire dataset, collect all images belonging to this class"""
    all_imgs = []
    for root, dirs, files in os.walk(dataset_base):
        folder_name = os.path.basename(root).lower()
        if folder_name == class_name.lower():
            imgs = [(os.path.join(root, f), f)
                    for f in files if f.lower().endswith(IMAGE_EXT)]
            all_imgs.extend(imgs)
            print(f"    Found {len(imgs):3d} images in: {root}")
    return all_imgs
def show_images(x, y):
    import matplotlib.pyplot as plt
    import numpy as np

    x = x.copy()

    # Reverse preprocessing (approx)
    x = (x - x.min()) / (x.max() - x.min())

    for i in range(5):
        plt.imshow(x[i])
        plt.title(LABELS[y[i].argmax()])
        plt.axis('off')
        plt.show()


def rebuild_dataset():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for c in CLASSES:
            os.makedirs(f"{split}/{c}", exist_ok=True)

    print("Collecting images from ALL dataset splits...\n")
    for c in CLASSES:
        print(f"  [{c}]")
        all_imgs = find_all_images_for_class(DATASET_BASE, c)

        if not all_imgs:
            print(f"  ⚠️  No images found for {c}")
            continue

        # Remove duplicates by filename
        seen = set()
        unique_imgs = []
        for path, fname in all_imgs:
            if fname not in seen:
                seen.add(fname)
                unique_imgs.append((path, fname))

        tr, temp = train_test_split(unique_imgs, test_size=0.2, random_state=42)
        vl, te   = train_test_split(temp,        test_size=0.5, random_state=42)

        for src, f in tr: shutil.copy(src, f"{TRAIN_DIR}/{c}/{f}")
        for src, f in vl: shutil.copy(src, f"{VAL_DIR}/{c}/{f}")
        for src, f in te: shutil.copy(src, f"{TEST_DIR}/{c}/{f}")
        print(f"  → Total:{len(unique_imgs)}  "
              f"Train:{len(tr)}  Val:{len(vl)}  Test:{len(te)}\n")

rebuild_dataset()
print("✓ Dataset ready")


# In[ ]:


IMG_SIZE  = (224, 224)
BATCH     = 32
NUM_CLASS = len(CLASSES)

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# ── Custom preprocessing with noise ──────────────────────────────
def preprocess_with_noise(x):
    x = preprocess_input(x)  # EfficientNet preprocessing
    noise = np.random.normal(0, 0.03, x.shape)  # slightly realistic noise
    x = x + noise
    return x

# ── Generators ───────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_with_noise,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation stays clean (important)
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Test with noise → realistic evaluation
test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_with_noise
)

# ── Data loaders ─────────────────────────────────────────────────
train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

test_ds = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)
LABELS = {v: k for k, v in train_ds.class_indices.items()}
x, y = next(train_ds)
show_images(x, y)
# ── Label mapping ────────────────────────────────────────────────
LABELS = {v: k for k, v in train_ds.class_indices.items()}
print("Classes:", LABELS)


# In[ ]:


import math

IMG_SIZE  = (224, 224)
BATCH     = 32
NUM_CLASS = len(CLASSES)
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
# ── PHASE 1: Build model with frozen base ─────────────────────────
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                             input_shape=IMG_SIZE + (3,))
base_model.trainable = True

# Freeze only early layers
for layer in base_model.layers[:200]:
    layer.trainable = False # freeze everything

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASS, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(3e-4),   # slightly safer LR
    loss=loss,
    metrics=['accuracy']
)

val_steps = math.ceil(val_ds.samples / BATCH)

print("── Phase 1: training head only ──")
history1 = model.fit(
    train_ds,
    steps_per_epoch=train_ds.samples // BATCH,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=10,
    callbacks=[
        ModelCheckpoint('best_model.keras', save_best_only=True,
                        monitor='val_accuracy', verbose=1)
    ]
)

# ── PHASE 2: Unfreeze top layers, fine-tune ───────────────────────
base_model.trainable = True

# Keep ALL BatchNorm layers frozen — critical
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Optionally freeze early layers (low-level features don't need updating)
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss=loss,
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

print("\n── Phase 2: fine-tuning top layers ──")
history2 = model.fit(
    train_ds,
    steps_per_epoch=train_ds.samples // BATCH,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=40,
    callbacks=callbacks
)

all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
print(f"\n✓ Done. Best val accuracy: {max(all_val_acc)*100:.2f}%")


# In[ ]:


test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Get predictions on test set
test_ds.reset()
y_pred_probs = model.predict(test_ds, steps=math.ceil(test_ds.samples / BATCH))
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_ds.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=list(LABELS.values()))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Test Set Confusion Matrix — EfficientNetB0 on NEU Steel Defects")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

x


import os
def get_filenames(directory):
    names = set()
    for root, _, files in os.walk(directory):
        for f in files:
            names.add(f)
    return names

train_files = get_filenames(TRAIN_DIR)
val_files   = get_filenames(VAL_DIR)
test_files  = get_filenames(TEST_DIR)

print(f"Train: {len(train_files)} files")
print(f"Val:   {len(val_files)} files")
print(f"Test:  {len(test_files)} files")

train_val_overlap  = train_files & val_files
train_test_overlap = train_files & test_files
val_test_overlap   = val_files   & test_files

print(f"\nTrain ∩ Val:  {len(train_val_overlap)}  ← must be 0")
print(f"Train ∩ Test: {len(train_test_overlap)} ← must be 0")
print(f"Val ∩ Test:   {len(val_test_overlap)}  ← must be 0")


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Combine histories
acc     = history1.history['accuracy']     + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss    = history1.history['loss']         + history2.history['loss']
val_loss= history1.history['val_loss']     + history2.history['val_loss']
epochs  = range(1, len(acc) + 1)
phase2_start = len(history1.history['accuracy'])  # vertical divider

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f0f0f')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ACCENT   = '#00e5ff'
ACCENT2  = '#ff4081'
GRID_CLR = '#2a2a2a'
TEXT_CLR = '#e0e0e0'

def style_ax(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle='--')
    ax.axvline(phase2_start + 0.5, color='#555555', linestyle=':', linewidth=1.2)
    ax.text(phase2_start + 0.8, ax.get_ylim()[0] + 0.01,
            'Phase 2', color='#888888', fontsize=8)

# ── 1. Accuracy curve ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, acc,     color=ACCENT,  linewidth=2,   label='Train')
ax1.plot(epochs, val_acc, color=ACCENT2, linewidth=2,   label='Val',
         linestyle='--', marker='o', markersize=3)
style_ax(ax1, 'Accuracy over Epochs')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.set_ylim([0, 1.05])
ax1.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor=TEXT_CLR)

# ── 2. Loss curve ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, loss,     color=ACCENT,  linewidth=2, label='Train')
ax2.plot(epochs, val_loss, color=ACCENT2, linewidth=2, label='Val',
         linestyle='--', marker='o', markersize=3)
style_ax(ax2, 'Loss over Epochs')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor=TEXT_CLR)

# ── 3. Confusion matrix ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
test_ds.reset()
y_pred = np.argmax(model.predict(test_ds, steps=math.ceil(test_ds.samples/BATCH), verbose=0), axis=1)
y_true = test_ds.classes
class_names = [LABELS[i] for i in range(NUM_CLASS)]
cm = confusion_matrix(y_true, y_pred)
im = ax3.imshow(cm, cmap='Blues')
ax3.set_xticks(range(NUM_CLASS)); ax3.set_yticks(range(NUM_CLASS))
ax3.set_xticklabels(class_names, rotation=35, ha='right', color=TEXT_CLR, fontsize=8)
ax3.set_yticklabels(class_names, color=TEXT_CLR, fontsize=8)
for i in range(NUM_CLASS):
    for j in range(NUM_CLASS):
        ax3.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i,j] > cm.max()/2 else TEXT_CLR, fontsize=9)
ax3.set_title('Confusion Matrix (Test Set)', color=TEXT_CLR, fontsize=13,
              fontweight='bold', pad=10)
ax3.set_xlabel('Predicted', color=TEXT_CLR)
ax3.set_ylabel('True', color=TEXT_CLR)
ax3.set_facecolor('#1a1a1a')
for spine in ax3.spines.values(): spine.set_edgecolor('#333333')

# ── 4. Per-class F1 bar chart ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
f1_scores = [report[c]['f1-score'] for c in class_names]
colors = [ACCENT if s == 1.0 else ACCENT2 for s in f1_scores]
bars = ax4.barh(class_names, f1_scores, color=colors, edgecolor='#333', height=0.6)
ax4.set_xlim([0, 1.15])
for bar, score in zip(bars, f1_scores):
    ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2,
             f'{score:.2f}', va='center', color=TEXT_CLR, fontsize=9)
ax4.set_facecolor('#1a1a1a')
ax4.set_title('Per-class F1 Score (Test Set)', color=TEXT_CLR,
              fontsize=13, fontweight='bold', pad=10)
ax4.tick_params(colors=TEXT_CLR, labelsize=9)
ax4.set_xlabel('F1 Score', color=TEXT_CLR)
for spine in ax4.spines.values(): spine.set_edgecolor('#333333')
ax4.grid(True, color=GRID_CLR, linewidth=0.5, linestyle='--', axis='x')

fig.suptitle('EfficientNetB0 — NEU Steel Surface Defect Classification',
             color=TEXT_CLR, fontsize=15, fontweight='bold', y=0.98)

plt.savefig('results_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved → results_dashboard.png")


# In[ ]:


import random

test_ds.reset()
images, labels = next(iter(test_ds))
preds = model.predict(images, verbose=0)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.patch.set_facecolor('#0f0f0f')

for i, ax in enumerate(axes.flat):
    if i >= len(images): break
    img = images[i]
    # Reverse EfficientNet preprocessing for display
    img_show = (img - img.min()) / (img.max() - img.min())

    true_label = LABELS[np.argmax(labels[i])]
    pred_label = LABELS[np.argmax(preds[i])]
    conf       = np.max(preds[i]) * 100
    correct    = true_label == pred_label

    ax.imshow(img_show, cmap='gray')
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)",
                 color='#00e5ff' if correct else '#ff4081',
                 fontsize=8, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('#00e5ff' if correct else '#ff4081')

fig.suptitle('Sample Predictions — EfficientNetB0', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f0f')
plt.show()


# In[ ]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


# In[ ]:


for layer in model.layers:
    print(layer.name)


# In[ ]:


def show_gradcam(img_path, model, last_conv_layer_name, img_size=(224,224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # preprocess same as training
    img_array = preprocess_input(img_array)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM (Model Focus)")
    plt.show()


# In[ ]:


import os

sample_path = os.path.join(TEST_DIR, "crazing", os.listdir(os.path.join(TEST_DIR, "crazing"))[0])

show_gradcam(sample_path, model, "top_conv")

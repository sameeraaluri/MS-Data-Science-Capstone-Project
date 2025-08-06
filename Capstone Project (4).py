#!/usr/bin/env python
# coding: utf-8

# In[42]:


import zipfile
import os

zip_path = 'Chest Xray Dataset.zip'  
extract_to = 'chest_xray_data' 

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Dataset unzipped successfully!")




# In[43]:


import os

os.getcwd()
os.listdir('chest_xray_data')

import os
import shutil
from sklearn.model_selection import train_test_split

original_dir = 'chest_xray_data/chest_xray/train'
custom_train_dir = 'chest_xray_data/custom/train'
custom_val_dir = 'chest_xray_data/custom/val'

classes = ['NORMAL', 'PNEUMONIA']

for cls in classes:
    images = os.listdir(os.path.join(original_dir, cls))
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(custom_train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(custom_val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(original_dir, cls, img), os.path.join(custom_train_dir, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(original_dir, cls, img), os.path.join(custom_val_dir, cls, img))

print("Custom training and validation sets created.")


# In[44]:


from PIL import Image
import matplotlib.pyplot as plt

base_path = 'chest_xray_data/chest_xray'

# Get path to one pneumonia image
pneumonia_img = os.listdir(os.path.join(base_path, 'train', 'PNEUMONIA'))[0]
img_path = os.path.join(base_path, 'train', 'PNEUMONIA', pneumonia_img)

# Display the image
img = Image.open(img_path)
plt.imshow(img, cmap='gray')
plt.title("Pneumonia Chest X-ray")
plt.axis('off')
plt.show()


# In[45]:


get_ipython().system('pip install tensorflow')


# In[94]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# No augmentation for validation/test
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'chest_xray_data/custom/train',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    'chest_xray_data/custom/train',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'chest_xray_data/chest_xray/test',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary',
    shuffle=False
)



# In[96]:


import matplotlib.pyplot as plt

x_batch, y_batch = next(train_generator)

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_batch[i].squeeze(), cmap='gray')
    plt.title("Pneumonia" if y_batch[i] == 1 else "Normal")
    plt.axis('off')
plt.show()


# In[102]:


from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np

# =====================
# 1. Model Definition
# =====================
from tensorflow.keras.applications import EfficientNetV2B0

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
    
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)



# =====================
# 2. Data Generators
# =====================
train_datagen = LungCroppedImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = LungCroppedImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'chest_xray_data/custom/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    'chest_xray_data/custom/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# =====================
# 3. Class Weights
# =====================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# =====================
# 4. Compile Model
# =====================
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=BinaryFocalCrossentropy(gamma=2.0),
    metrics=['accuracy', AUC(name='auc'), Recall(name='recall')]
)

# =====================
# 5. Callbacks
# =====================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, verbose=1)

# =====================
# 6. Train Model
# =====================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# =====================
# 7. Model Summary
# =====================
model.summary()

from sklearn.metrics import precision_recall_curve

probs = model.predict(val_generator)
probs = probs.flatten()
precision, recall, thresholds = precision_recall_curve(val_generator.classes, probs)



# In[104]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def crop_lungs_fixed(img):
    """
    Crops the image to a central lung region.
    Assumes input shape (H, W, 3) and returns resized (224, 224, 3) image.
    """
    h, w = img.shape[:2]
    # Empirically determined crop region (adjustable)
    top = int(h * 0.15)
    bottom = int(h * 0.90)
    left = int(w * 0.20)
    right = int(w * 0.80)

    cropped = img[top:bottom, left:right]
    resized = cv2.resize(cropped, (224, 224))
    return resized

# Get a batch of images from validation generator
img_batch, label_batch = val_generator[0]
original_img = img_batch[0]  # shape: (224, 224, 3)

# Apply the crop
lung_cropped_img = crop_lungs_fixed(original_img)

# Show it
import matplotlib.pyplot as plt
plt.imshow(lung_cropped_img)
plt.title(f"Cropped Lung Image - True Label: {label_batch[0]}")
plt.axis('off')
plt.show()

class LungCroppedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        x = super().standardize(x)
        return crop_lungs_fixed(x)

def get_gradcam_heatmap(model, image_array, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam



def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_rgb = np.uint8(img * 255)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img_rgb, 1 - alpha, 0)
    return superimposed_img



img_batch, label_batch = val_generator[0]
img = img_batch[0]
label = label_batch[0]
img_array = np.expand_dims(img, axis=0)

heatmap = get_gradcam_heatmap(model, img_array)
overlay = overlay_heatmap_on_image(img, heatmap)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"True label: {int(label)} | Pred: {model.predict(img_array)[0][0]:.2f}")
plt.axis('off')
plt.show()




# In[113]:


# Get all validation images and labels
x_val, y_val = [], []

for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    x_val.extend(x_batch)
    y_val.extend(y_batch)

x_val = np.array(x_val)
y_val = np.array(y_val)

import random

# Find indices of pneumonia cases
pneumonia_indices = np.where(y_val == 1)[0]
selected_indices = random.sample(list(pneumonia_indices), 5)  # Pick 5 at random

x_samples = x_val[selected_indices]
y_samples = y_val[selected_indices]

def get_gradcam_heatmap(model, image_array, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_rgb = np.uint8(img * 255)
    return cv2.addWeighted(heatmap_color, alpha, img_rgb, 1 - alpha, 0)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

for i, img in enumerate(x_samples):
    img_array = np.expand_dims(img, axis=0)
    heatmap = get_gradcam_heatmap(model, img_array)
    overlay = overlay_heatmap_on_image(img, heatmap)

    pred = model.predict(img_array)[0][0]

    plt.subplot(2, 3, i+1)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Pneumonia | Pred: {pred:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Grad-CAMs on Pneumonia Images (Lung-Cropped)", fontsize=16)
plt.subplots_adjust(top=0.88)
plt.show()



# In[115]:


y_true = val_generator.classes
y_probs = model.predict(val_generator).flatten()
y_pred = (y_probs > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Lung-Cropped Model")
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))


# In[119]:


def plot_training_curves(history):
    metrics = ['accuracy', 'loss', 'auc', 'recall']
    plt.figure(figsize=(16, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f"{metric.title()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle("Training Metrics", fontsize=16, y=1.02)
    plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_probs)
avg_prec = average_precision_score(y_true, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'AP = {avg_prec:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.legend()
plt.show()



import numpy as np

labels = ['Accuracy', 'Normal Recall', 'Pneumonia Recall', 'F1 (Macro)']
before = [0.66, 0.15, 0.97, 0.51]
after = [0.86, 0.98, 0.81, 0.83]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, before, width, label='Before')
plt.bar(x + width/2, after, width, label='After')
plt.xticks(x, labels)
plt.ylim(0, 1.1)
plt.title("Metric Comparison: Full Image vs Cropped Model")
plt.legend()
plt.grid(True)
plt.show()



# In[75]:


get_ipython().system('pip install opencv-python')


# In[ ]:





import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "Dataset"    
IMG_HEIGHT = 224
IMG_WIDTH = 224
X = []
Y = []
classes = os.listdir(DATASET_PATH)

print("Detected Classes:", classes)

for label, folder in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, folder)
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(file_path)
            
            # Skip corrupted images
            if img is None:
                print("Skipped corrupted image:", file_path)
                continue
            
            # Resize
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            X.append(img)
            Y.append(label)
        
        except Exception as e:
            print("Error loading image:", file_path, "Error:", e)

X = np.array(X)
Y = np.array(Y)

print("Total Images Loaded:", len(X))

X = X / 255.0
print("Images Normalized")

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.30, stratify=Y, random_state=42
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, stratify=Y_temp, random_state=42
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

#  Data Augmentation (for training)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, Y_val, batch_size=32)
test_generator = test_datagen.flow(X_test, Y_test, batch_size=32)

print("Data preprocessing completed.")

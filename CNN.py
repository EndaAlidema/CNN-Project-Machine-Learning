# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure the script outputs are encoded properly
sys.stdout.reconfigure(encoding='utf-8')

# Function to validate images and remove any that are invalid
def validate_images(data):
    valid_data = []
    for entry in data:
        try:
            with Image.open(entry['image_path']) as img:
                img.verify()
                valid_data.append(entry)
        except (IOError, SyntaxError, UnidentifiedImageError):
            print(f"Dropping invalid image: {entry['image_path']}")
    return valid_data

# Function to balance the dataset by undersampling
def balance_dataset(df):
    minority_class_size = df['label'].value_counts().min()
    df_list = []
    for label, group in df.groupby('label'):
        df_list.append(group.sample(minority_class_size, replace=False, random_state=42))
    balanced_df = pd.concat(df_list)
    return balanced_df

# Base directory containing the image data
base_dir = 'revitsone-5class/Revitsone-5classes'
categories = ['talking_phone', 'texting_phone', 'turning', 'other_activities', 'safe_driving']

# Prepare the data in a structured format
data = []
for category in categories:
    category_dir = os.path.join(base_dir, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.jpg'):
            data.append({
                'image_path': os.path.join(category_dir, filename),
                'label': category
            })

# Validate images
data = validate_images(data)

# Create a DataFrame
df = pd.DataFrame(data)

# Print the number of samples before balancing
print("Number of samples before balancing:")
print(df['label'].value_counts())

# Balance the dataset
df = balance_dataset(df)

# Print the number of samples after balancing
print("Number of samples after balancing:")
print(df['label'].value_counts())

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define ImageDataGenerator for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of the training data for validation
)

# Create generators for training and validation sets
train_generator = datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=42
)

validation_generator = datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Create a generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_data,
    x_col='image_path',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Visualize 12 images from the dataset
plt.figure(figsize=(12, 12))
for i in range(12):
    img_path = df.iloc[i]['image_path']
    img = Image.open(img_path)
    plt.subplot(4, 3, i + 1)
    plt.imshow(img)
    plt.title(df.iloc[i]['label'])
    plt.axis('off')
plt.suptitle("Sample Images from the Dataset")
plt.show()

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Generate predictions on the test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model
model.save('my_cnn_model.h5')

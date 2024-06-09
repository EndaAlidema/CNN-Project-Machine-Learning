import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Ensure the path points to the JSON file, not a directory
json_path = 'kaggle.json'  # Replace with the path to your JSON file

if not os.path.isfile(json_path):
    raise FileNotFoundError(f"JSON file not found at path: {'kaggle.json'}")

# Load the dataset from JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Debug: Print the JSON data structure
print("JSON data:", data)

# Ensure the data is in the expected format
# Expected format: [{"image_path": "path/to/image1.jpg", "label": "label1"}, ...]
if not isinstance(data, list):
    raise ValueError("JSON data is not a list of records")

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)

# Debug: Print the DataFrame structure
print("DataFrame head:", df.head())

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define ImageDataGenerator for training and validation
datagen = ImageDataGenerator(
    rescale=1./255
)

# Create generators
train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',  # Column name in JSON for image paths
    y_col='label',       # Column name in JSON for labels
    target_size=(150, 150),  # Adjust according to your image size
    batch_size=32,
    class_mode='categorical'  # Assuming you have multiple classes
)

validation_generator = datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',  # Column name in JSON for image paths
    y_col='label',       # Column name in JSON for labels
    target_size=(150, 150),  # Adjust according to your image size
    batch_size=32,
    class_mode='categorical'
)

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
    Dense(5, activation='softmax')  # Assuming you have 5 classes
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on your dataset
    validation_data=validation_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')

# Save the model
model.save('my_cnn_model.h5')

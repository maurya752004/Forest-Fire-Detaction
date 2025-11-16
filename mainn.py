import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from PIL import Image
import os

image_folder = "dataset/images/train/Fire"
for filename in os.listdir(image_folder):
    filepath = os.path.join(image_folder, filename)
    try:
        img = Image.open(filepath)
        img.verify()
    except Exception:
        os.remove(filepath)
        print(f"Removed bad image: {filename}")


# Paths to dataset
train_dir = "dataset/images/train"
val_dir = "dataset/images/val"

# ImageDataGenerator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# Save the trained model
model.save("fire_detection_model.h5")
print("Model saved as fire_detection_model.h5")



# Save in the new Keras format
model.save("fire_detection_model.keras")

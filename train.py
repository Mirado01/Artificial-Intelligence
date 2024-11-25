import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import preprocess_images
import random

def create_model(input_shape=(48, 48, 1)):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes for anger and joy
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess images and labels
images, labels = preprocess_images('data/')
images = np.array(images).reshape(-1, 48, 48, 1) / 255.0  # Normalization
labels = np.array(labels)

# Shuffle images and labels
indices = np.arange(len(images))
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('model/emotion_recognition_model.h5')

# Evaluate the model and display the classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['joy', 'anger']))

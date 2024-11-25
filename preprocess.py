import cv2
import os

def preprocess_images(input_dir, output_size=(48, 48)):
    images, labels = [], []
    for label, emotion in enumerate(['joy', 'anger']):  # Only joy and anger
        emotion_path = os.path.join(input_dir, emotion)
        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Image {img_file} in folder '{emotion}' could not be loaded.")
                continue
            img = cv2.resize(img, output_size)
            images.append(img)
            labels.append(label)
    return images, labels

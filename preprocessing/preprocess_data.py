# Script for preprocessing the dataset

import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir):
    """Preprocess images: resize and normalize."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (128, 128))
                img_normalized = img_resized / 255.0

                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, img_normalized * 255)

if __name__ == "__main__":
    preprocess_images("Data/PlantVillage", "Data/Processed")
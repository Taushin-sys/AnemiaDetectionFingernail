import cv2
import numpy as np

def load_and_resize_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.resize(image, size)
    return image

def extract_fingernail_roi(image):
    # Dummy version – assuming full image is fingernail for now
    # You can later add OpenCV segmentation if needed
    return image

def extract_color_features(roi):
    avg_color = cv2.mean(roi)[:3]  # (B, G, R)
    return np.array(avg_color)
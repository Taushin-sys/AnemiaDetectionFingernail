import os
import cv2
import numpy as np
import joblib
import pandas as pd
import argparse

from image_processing import extract_color_features, extract_fingernail_roi

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def load_test_image(image_path):
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or invalid format at: {image_path}")
    print("Image loaded.")
    return img

def predict_anemia(model, R, G, B):
    input_df = pd.DataFrame({'R': [R], 'G': [G], 'B': [B]})
    prediction = model.predict(input_df)[0]
    return prediction

def annotate_and_save(img, result_text, color, output_path):
    os.makedirs(output_path, exist_ok=True)
    display_img = cv2.resize(img, (500, 500))
    cv2.putText(display_img, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    output_file = os.path.join(output_path, "annotated_result.png")
    cv2.imshow("Anemia Detection Result", display_img)
    cv2.imwrite(output_file, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Annotated result saved at: {output_file}")

def main(image_path):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'model', 'anemia_detector.pkl')
    output_path = os.path.join(BASE_DIR, 'output')

    model = load_model(model_path)
    print(f"Model was trained with features: {model.feature_names_in_}")

    img = load_test_image(image_path)

    print("Extracting fingernail ROI...")
    roi = extract_fingernail_roi(img)

    print("Extracting RGB features from ROI...")
    B, G, R = extract_color_features(roi)
    print(f"Extracted RGB (R, G, B): ({R:.2f}, {G:.2f}, {B:.2f})")

    prediction = predict_anemia(model, R, G, B)

    if int(prediction) == 1:
        result_text = "Result: Anemic"
        color = (0, 0, 255)  # Red
    else:
        result_text = "Result: Not Anemic"
        color = (0, 255, 0)  # Green

    print(result_text)
    annotate_and_save(img, result_text, color, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anemia Detection using RGB features from fingernail image.")
    parser.add_argument("--image_path", required=True, help="Path to the input fingernail image")
    args = parser.parse_args()

    print("Prediction started...")
    main(args.image_path)
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import os
from PIL import Image

from src.image_processing import extract_color_features, extract_fingernail_roi

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "anemia_detector.pkl")
    return joblib.load(model_path)

model = load_model()

# Title and description
st.title("Anemia Detection Using Fingernail Image")
st.markdown("This application allows you to upload a fingernail image and predicts whether the person is anemic based on color features extracted from the nail region.")

# File uploader
uploaded_file = st.file_uploader("Upload a fingernail image (JPG, PNG, or WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    try:
        roi = extract_fingernail_roi(image_np)
        R, G, B = extract_color_features(roi)

        st.subheader("Extracted RGB Values")
        st.write(f"R: {R:.2f}, G: {G:.2f}, B: {B:.2f}")

        input_df = pd.DataFrame({'R': [R], 'G': [G], 'B': [B]})
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("Prediction: Anemic")
        else:
            st.success("Prediction: Not Anemic")

    except Exception as e:
        st.warning(f"Could not process image: {e}")
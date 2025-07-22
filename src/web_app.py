import streamlit as st
import numpy as np
import cv2
import pickle
import pandas as pd
from PIL import Image
from image_processing import extract_color_features

# Load model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🩸 Anemia Detection from Fingernail Image")
st.write("Upload a fingernail image to predict whether the person is anemic or not.")

uploaded_file = st.file_uploader("Choose a fingernail image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Extract features (BGR version)
    features = extract_color_features(image_cv2)
    input_df = pd.DataFrame([features], columns=['B', 'G', 'R'])

    # Predict
    prediction = model.predict(input_df)[0]

    result_text = "Anemic" if prediction == 1 else "Not Anemic"
    result_color = "🔴" if prediction == 1 else "🟢"

    st.markdown(f"### {result_color} Result: *{result_text}*")
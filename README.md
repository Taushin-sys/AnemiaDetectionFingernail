# Anemia Detection Using Colour of the Fingernails

This project uses computer vision and machine learning techniques to predict whether a person is *anemic* based on the color of their fingernails. It extracts RGB color features from the nail region of an image and uses a trained classifier to make predictions.

##  Problem Statement

Anemia is a condition in which the blood doesn't have enough healthy red blood cells. A common visible symptom of anemia is paleness in the fingernail bed. This project leverages this visual cue by analyzing fingernail images and predicting the likelihood of anemia.

##  Project Structure
AnemiaDetectionFingernail/
│
├── Dataset/
│   ├── anemic/
│   └── non_anemic/
│
├── model/
│   ├── anemia_CNN_model.h5
│   ├── anemia_detector.pkl
│   └── model.pkl
│
├── notebooks/
│   ├── anemia_detection.ipynb
│   └── output/annotated_results.png
│
├── src/
│   ├── image_processing.py
│   ├── prediction.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── web_app.py
│   └── pycache/
│
├── test_images/
│   └── sample.png
│
├── venv/               # Virtual environment (excluded from Git)
│
├── .gitignore
├── config.py
├── data_preprocessing.py
├── features.csv
├── requirements.txt
├── streamlit_app.py
└── README.md
---

##  How It Works

1. Image Upload: The user uploads a fingernail image (JPG, PNG, or WEBP).
2. Nail Detection: The app detects the region of interest (ROI) focusing on the fingernail.
3. Feature Extraction: Extracts average RGB values from the nail region.
4. Prediction: A trained ML model (Random Forest) predicts if the input corresponds to an anemic or non-anemic case.

---

##  Technologies Used

- Python
- OpenCV
- NumPy, Pandas
- Scikit-learn
- Streamlit
- Matplotlib (for visualization)
- TensorFlow (for optional CNN model)

---

##  How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AnemiaDetectionFingernail.git
cd AnemiaDetectionFingernail

2. Create & Activate Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Streamlit App
streamlit run streamlit_app.py
Model Information
	•	Model Used: Random Forest Classifier
	•	Input Features: Average R, G, B values from the nail region
	•	Performance:
	•	ROC AUC: 0.88
	•	Accuracy: ~85% (on test data)

Sample Prediction Output
Image: 
Image:
![Sample Prediction](notebooks/output/annotated_results.png)
Predicted: Not Anemic

Author

Sayed Taushin
B.E. in Artificial Intelligence and Data Science (2025)  
GitHub: [@Taushin-sys](https://github.com/Taushin-sys)  
Email: taushinsayed@gmail.com

License

This project is open-source and available under the MIT License.
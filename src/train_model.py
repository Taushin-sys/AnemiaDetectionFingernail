import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load features
df = pd.read_csv('features.csv')
print(df.columns)
df.columns = df.columns.str.lower()  # Convert all column names to lowercase
X = df.drop('label', axis=1)
y = df['label']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(clf, 'model/anemia_detector.pkl')

print("✅ Model trained and saved to 'model/anemia_detector.pkl'")
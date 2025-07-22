import os
import cv2
import numpy as np
import pandas as pd

# Define paths
data_dir = "Dataset"
categories = ["anemic", "non_anemic"]

data = []

# Process each category
for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))

            # Calculate average RGB values
            avg_color_per_row = np.average(img, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            r, g, b = avg_color

            data.append([r, g, b, label])
        
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Save to CSV
df = pd.DataFrame(data, columns=['R', 'G', 'B', 'Label'])
df.to_csv("features.csv", index=False)

print("✅ Data preprocessing complete. features.csv saved.")

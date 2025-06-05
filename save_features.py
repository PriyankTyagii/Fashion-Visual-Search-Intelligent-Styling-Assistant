import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

# Load & merge datasets
df1 = pd.read_csv("jeans_bd_processed_data.csv")
df2 = pd.read_csv("dresses_bd_processed_data.csv")
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['feature_image_s3'], inplace=True)
df = df.reset_index(drop=True)

features = []
valid_indices = []

for i, row in df.iterrows():
    try:
        response = requests.get(row['feature_image_s3'], timeout=4)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        feat = extract_features(img)
        features.append(feat)
        valid_indices.append(i)
    except Exception as e:
        continue

filtered_df = df.iloc[valid_indices].reset_index(drop=True)
np.save("fashion_features.npy", np.array(features))
filtered_df.to_csv("fashion_data_filtered.csv", index=False)
print(f"Saved {len(filtered_df)} product embeddings.")

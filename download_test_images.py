"""Download a handful of sample images from the dataset for local testing."""
import os
import pandas as pd
import requests

OUTPUT_DIR = "test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

url_col = "feature_image_s3"

jeans_df = pd.read_csv("fashion_data_filtered.csv")
dresses_df = pd.read_csv("dresses_bd_processed_data.csv")

jeans_sample = jeans_df[[url_col, "product_name", "brand"]].dropna(subset=[url_col]).head(10)
dresses_sample = dresses_df[[url_col, "product_name", "brand"]].dropna(subset=[url_col]).head(5)

sample = pd.concat([jeans_sample, dresses_sample], ignore_index=True)

headers = {"User-Agent": "Mozilla/5.0"}

for i, row in sample.iterrows():
    url = row[url_col]
    category = "dress" if i >= 10 else "jeans"
    name = f"{i:03d}_{category}_{row['brand'].replace(' ', '_')}.jpg"
    path = os.path.join(OUTPUT_DIR, name)
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"[OK] {name}  —  {row['product_name']}")
    except Exception as e:
        print(f"[FAIL] {url}  —  {e}")

print(f"\nDone. Images saved to ./{OUTPUT_DIR}/")

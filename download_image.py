import os
import csv
import requests
from ast import literal_eval

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    csv_file = "dresses_bd_processed_data.csv"
    images_dir = "downloaded_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            product_id = row.get('product_id', f"unknown_{row_num}")
            image_urls_str = row.get('pdp_images_s3', '[]')

            try:
                image_urls = literal_eval(image_urls_str)
                if not isinstance(image_urls, list):
                    print(f"Warning: pdp_images_s3 field not a list for product_id {product_id}, skipping.")
                    continue
            except Exception as e:
                print(f"Error parsing pdp_images_s3 for product_id {product_id}: {e}")
                continue

            for idx, url in enumerate(image_urls):
                if not url or not url.startswith("http"):
                    print(f"Invalid URL for product_id {product_id}: {url}")
                    continue
                file_ext = os.path.splitext(url)[1].split('?')[0]
                if not file_ext or len(file_ext) > 5:
                    file_ext = ".jpg"
                # Generate unique filename using product_id and image index
                filename = f"{product_id}_image_{idx+1}{file_ext}"
                save_path = os.path.join(images_dir, filename)

                download_image(url, save_path)

if __name__ == "__main__":
    main()

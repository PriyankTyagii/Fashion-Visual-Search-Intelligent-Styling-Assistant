
# 👗 Fashion Visual Search & Intelligent Styling Assistant

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-green?style=for-the-badge&logo=streamlit)](https://fashion-visual-search-intelligent-styling-assistant-priyank.streamlit.app/)

A powerful AI-based fashion intelligence system that allows users to visually search and explore fashion items, recommend outfits, and get style-personalized suggestions — solving the core friction in online apparel shopping.

---

## 🛍 Industry Problem

🔍 Over 65% of fashion e-commerce users abandon their shopping journey due to irrelevant results in text-based search.  
🧠 Fashion is visual — users care about **color**, **cut**, **texture**, and **style fit**, which text search fails to describe.  

---

## 🎯 Solution Summary

This app is a working prototype that allows users to:

- 📤 Upload any clothing photo (from wardrobe, web, or screenshots)
- 🔎 Find exact & visually similar products
- 👗 Get outfit suggestions based on **style attributes**
- 🔥 Discover trending fashion items (new + discounted)
- 🧠 Receive personalized style recommendations

---

## 🔧 Technical Features

| Feature                       | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| 🎨 Visual Similarity Engine   | Uses ResNet50 (ImageNet) to generate deep feature vectors from user image   |
| 🤝 Multi-modal Matching       | Combines image features + metadata (brand, category, price, style)          |
| 👗 Outfit Recommendation      | Suggests outfit items using metadata like `style_attribute`, `meta_info`    |
| 🔥 Trend Awareness            | Picks fashion trends using `launch_on` date and `discount`                 |
| 🧠 Personalized Learning      | Builds per-session style profile using average feature embeddings           |
| ⚙️ Streamlit Interface        | Modern, responsive, and shareable UI with hover effects, filters, buttons  |

---

## 🧰 Tech Stack

| Layer         | Technology                           |
|---------------|---------------------------------------|
| Frontend      | Streamlit                             |
| ML Model      | TensorFlow (ResNet50 pretrained)      |
| Data Handling | Pandas, NumPy                         |
| Similarity    | Scikit-learn (Cosine Similarity)      |
| Hosting       | Streamlit Cloud                       |
| Storage       | Git LFS (for large CSV/NPY support)   |

---

## 🚀 Main Features Demo
### 🎯 Top Match (AI-based)
![Screenshot 2025-06-06 144536](https://github.com/user-attachments/assets/5ba3c9d3-5db0-443e-88ae-defa5c0dad33)


### 🧩 Visually Similar Products & Suggestions
![Screenshot 2025-06-06 144550](https://github.com/user-attachments/assets/e7511553-91ac-43d4-bf5e-097af3552535)


### 🔥 Trendy Picks + 🧠 Personalized Suggestions
![Screenshot 2025-06-06 144602](https://github.com/user-attachments/assets/0402ea96-b228-48be-9aa3-25c371afbceb)


### 📥 Upload Interface
![Screenshot 2025-06-06 145310](https://github.com/user-attachments/assets/3f86a217-74c5-4bb6-b0d8-cc746763f069)


---

## 📊 Dataset Fields Used

This project utilizes rich metadata:

- `feature_image`: Main product thumbnail
- `selling_price`: Dynamic pricing (₹ or $)
- `discount`: Used to identify top deals
- `product_name`: Clean title
- `meta_info`: Drives outfit suggestion logic
- `style_attribute`: Used in multi-style outfit pairing
- `pdp_url`: Direct product links
- `launch_on`, `last_seen_date`: For identifying trends
- `brand`, `sku`, `department_id`, `category_id`: For structured recommendations

---

## 📁 Folder Structure

```
fashion-visual-search/
├── fashion_app.py                # Streamlit App
├── fashion_data_filtered.csv     # Fashion dataset
├── fashion_features.npy          # Precomputed image embeddings
├── requirements.txt              # Package dependencies
├── runtime.txt                   # Python runtime for Streamlit Cloud
├── .gitattributes                # Git LFS tracked files
├── README.md                     # Full documentation
```

---

## 🛠️ Run Locally

```bash
git clone https://github.com/PriyankTyagii/Owner-avatar-Fashion-Visual-Search-Intelligent-Styling-Assistant.git
cd Owner-avatar-Fashion-Visual-Search-Intelligent-Styling-Assistant

pip install -r requirements.txt
streamlit run fashion_app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“Deploy”**
4. Set `fashion_app.py` as the main entry file

✅ Streamlit Cloud will auto-install requirements and host your app.

---

## ✅ Outcomes from Assignment

- ✅ Working Visual Search engine with high image match accuracy
- ✅ Outfit logic based on fashion metadata
- ✅ Live trending section based on `launch_on` & `discount`
- ✅ Fully running prototype on the cloud
- ✅ Designed to scale and extend

---

## 💡 Bonus Highlights

- 📦 Ready to scale for 10,000+ users via Streamlit + optimized backend
- 🧠 Designed for extension into trend-aware generative outfit builders
- 💻 Works with Git LFS and supports large file inputs (CSV, NPY)

---

## 👤 Author

**Priyank Tyagi**  
🎓 B.Tech CSE | 🔬 AI & Machine Learning | 🧪 Product Innovator  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/priyank-tyagi-3a3a10259)

---

## 📃 License

Licensed under the [MIT License](LICENSE) — free for research and commercial use.

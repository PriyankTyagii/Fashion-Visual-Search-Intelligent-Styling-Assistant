
# 👗 Fashion Visual Search & Intelligent Styling Assistant

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-green?style=flat-square&logo=streamlit)](https://fashion-visual-search-intelligent-styling-assistant-priyank.streamlit.app/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-blue?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This intelligent web application allows users to upload any fashion image and receive:

- 🎯 The most visually similar fashion product from inventory
- 🧩 5 visually similar products
- 👗 Outfit suggestions based on style
- 🔥 Trendy picks based on launch date & discounts
- 🧠 Personalized recommendations based on past uploads

> 🌐 **Live Demo**: [Click here to try the app](https://fashion-visual-search-intelligent-styling-assistant-priyank.streamlit.app/)

---

## 🧠 Use Case

Fashion e-commerce loses up to **65% of potential customers** due to poor search relevance. This project solves that with **visual similarity-based product search** and **style-driven suggestions** using deep learning.

---

## 🧰 Tech Stack

| Layer         | Technology                          |
|---------------|--------------------------------------|
| Frontend      | [Streamlit](https://streamlit.io/)   |
| ML Model      | TensorFlow (ResNet50 pretrained)     |
| Data Handling | Pandas, NumPy                        |
| Search Logic  | Cosine Similarity (Scikit-learn)     |
| Deployment    | Streamlit Cloud                      |

---

## 🚀 Features

- 📤 Upload image of clothing item (even from social media or screenshots)
- 🖼 Image processed with **ResNet50** to extract embeddings
- 🔎 Exact match & 5 visually similar items shown with brand + price
- 👚 Outfit suggestions based on style metadata
- 🔥 Trendy picks sorted by launch date and discount
- 🧠 Personalized picks based on your session history

---

## 📸 Screenshot

> _(Replace this with your actual image file and name it `screenshot.png`)_

![App Screenshot](screenshot.png)

---

## 📁 Folder Structure

```
fashion-visual-search/
├── fashion_app.py                # Streamlit app
├── fashion_data_filtered.csv     # Inventory dataset
├── fashion_features.npy          # Precomputed embeddings
├── requirements.txt              # Python packages
├── runtime.txt                   # Python version (for Streamlit)
├── README.md                     # Project documentation
├── .gitattributes                # Git LFS (optional)
```

---

## 🛠️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/PriyankTyagii/Owner-avatar-Fashion-Visual-Search-Intelligent-Styling-Assistant.git
cd Owner-avatar-Fashion-Visual-Search-Intelligent-Styling-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run fashion_app.py
```

---

✅ Done! Your app is now live and shareable.

---

## 📃 License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## 👤 Author

**Priyank Tyagi**  
👨‍💻 Passionate about AI, computer vision & intelligent systems  
🔗 [LinkedIn](https://www.linkedin.com/in/priyanktyagi)

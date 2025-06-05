import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from datetime import datetime

# --- Setup ---
st.set_page_config(page_title="👗 Fashion Visual Search", layout="wide")
st.markdown("<h1 style='text-align: center;'>👗 Fashion Visual Search & Styling Assistant</h1><hr>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

def extract_features(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

@st.cache_data
def load_data():
    df = pd.read_csv("fashion_data_filtered.csv")
    feats = np.load("fashion_features.npy")

    if "feature_image" not in df.columns and "feature_image_s3" in df.columns:
        df.rename(columns={"feature_image_s3": "feature_image"}, inplace=True)
    if "style_attribute" not in df.columns and "style_attributes" in df.columns:
        df.rename(columns={"style_attributes": "style_attribute"}, inplace=True)
    df.dropna(subset=["feature_image"], inplace=True)
    return df, feats

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Upload
st.sidebar.header("📤 Upload Fashion Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.sidebar.markdown("---")
    st.sidebar.markdown("📸 **Preview**")
    st.sidebar.image(img, use_container_width=True)

    with st.spinner("🔍 Extracting and matching..."):
        df, feats = load_data()
        query_feat = extract_features(img)
        similarities = cosine_similarity([query_feat], feats)[0]
        top_indices = similarities.argsort()[-6:][::-1]
        st.session_state.history.append(query_feat)

    best = df.iloc[top_indices[0]]

    # Format helpers
    def parse_price(val):
        try:
            d = eval(str(val)) if isinstance(val, str) else val
            if isinstance(d, dict):
                if "INR" in d: return f"₹{d['INR']:.2f}"
                elif "USD" in d: return f"${d['USD']:.2f}"
        except: pass
        return f"₹{val}"

    def parse_list(val):
        try:
            l = eval(str(val))
            if isinstance(l, list): return ", ".join(str(i) for i in l)
        except: pass
        return str(val)

    def parse_dict(val):
        try:
            d = eval(str(val))
            if isinstance(d, dict):
                return "  \n".join([f"• **{k}**: {v}" for k, v in d.items()])
        except: pass
        return str(val)

    # --- Top Match ---
    st.markdown("## 🎯 Top Match")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(best["feature_image"], use_container_width=True)
    with col2:
        st.markdown(f"### 🧾 {best['product_name']}")
        st.markdown(f"**🛍 Brand:** `{best['brand']}`")
        st.markdown(f"**💸 Price:** {parse_price(best['selling_price'])}  &nbsp;&nbsp;|&nbsp;&nbsp; **Discount:** `{best['discount']}%`")
        st.markdown(f"**📅 Launched:** `{best['launch_on']}` &nbsp;&nbsp;|&nbsp;&nbsp; **Last Seen:** `{best['last_seen_date']}`")
        st.markdown(f"**🏷 Category:** `{best['category_id']}`  |  **Dept:** `{best['department_id']}`")
        st.markdown(f"**🔖 SKU:** `{best['sku']}`  |  **Channel:** `{best['channel_id']}`")
        st.markdown(f"**🧩 Features:** {parse_list(best['feature_list'])}")
        st.markdown(f"**📄 Description:** {best['description']}")
        st.markdown("**🎨 Style Attributes:**")
        st.markdown(parse_dict(best["style_attribute"]))
        if best["pdp_url"]:
            st.markdown(f"""<a href="{best['pdp_url']}" target="_blank"><button style="margin-top: 10px; padding: 10px 20px; background-color:#4CAF50; color:white; border:none; border-radius:5px;">🔗 View Product</button></a>""", unsafe_allow_html=True)

    # --- Visually Similar Products ---
    st.markdown("## 🧩 Visually Similar Products")
    sim_cols = st.columns(5)
    for i, idx in enumerate(top_indices[1:]):
        prod = df.iloc[idx]
        with sim_cols[i]:
            st.markdown(f"""
            <div style="text-align:center;">
                <img src="{prod['feature_image']}" style="width:100%; border-radius:8px; transition: transform .2s;" 
                     onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.0)'"/>
                <p><strong>{prod['brand']}</strong></p>
                <p style="margin-top:-10px;">{parse_price(prod['selling_price'])} | {prod['discount']}% off</p>
                <a href="{prod['pdp_url']}" target="_blank" style="color:#1f77b4;">🔗 View Product</a>
            </div>
            """, unsafe_allow_html=True)

    # --- Outfit Suggestions ---
    st.markdown("## 👗 Outfit Suggestions")
    style_keyword = str(best["style_attribute"]).split(",")[0].strip().lower()
    outfit_matches = df[df["style_attribute"].astype(str).str.lower().str.contains(style_keyword)]
    outfit_matches = outfit_matches[outfit_matches.index != best.name]
    outfit_df = outfit_matches.sample(n=min(5, len(outfit_matches)))

    if outfit_df.empty:
        st.info("No matching outfits found based on style attributes.")
    else:
        outfit_cols = st.columns(5)
        for i, (_, item) in enumerate(outfit_df.iterrows()):
            with outfit_cols[i]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <img src="{item['feature_image']}" style="width:100%; border-radius:8px; transition: transform .2s;" 
                         onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.0)'"/>
                    <p><strong>{item['brand']}</strong></p>
                    <p style="margin-top:-10px;">{parse_price(item['selling_price'])} | {item['discount']}% off</p>
                    <a href="{item['pdp_url']}" target="_blank" style="color:#1f77b4;">🔗 View Product</a>
                </div>
                """, unsafe_allow_html=True)

    # --- Trendy Picks ---
    st.markdown("## 🔥 Trendy Picks")
    df["launch_on"] = pd.to_datetime(df["launch_on"], errors="coerce")
    trendy = df.dropna(subset=["launch_on"]).sort_values(by=["discount", "launch_on"], ascending=[False, False]).head(5)
    trend_cols = st.columns(5)
    for i, (_, row) in enumerate(trendy.iterrows()):
        with trend_cols[i]:
            st.markdown(f"""
            <div style="text-align:center;">
                <img src="{row['feature_image']}" style="width:100%; border-radius:8px; transition: transform .2s;" 
                     onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.0)'"/>
                <p><strong>{row['brand']}</strong></p>
                <p style="margin-top:-10px;">{parse_price(row['selling_price'])} | {row['discount']}% off</p>
                <a href="{row['pdp_url']}" target="_blank" style="color:#1f77b4;">🔗 View Product</a>
            </div>
            """, unsafe_allow_html=True)

    # --- Personalized Recommendations ---
    if len(st.session_state.history) > 1:
        st.markdown("## 🧠 Personalized Recommendations")
        avg_feat = np.mean(st.session_state.history, axis=0)
        sims = cosine_similarity([avg_feat], feats)[0]
        rec_indices = sims.argsort()[-6:][::-1]
        rec_cols = st.columns(5)
        for i, idx in enumerate(rec_indices[1:]):
            rec = df.iloc[idx]
            with rec_cols[i]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <img src="{rec['feature_image']}" style="width:100%; border-radius:8px; transition: transform .2s;" 
                         onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.0)'"/>
                    <p><strong>{rec['brand']}</strong></p>
                    <p style="margin-top:-10px;">{parse_price(rec['selling_price'])} | {rec['discount']}% off</p>
                    <a href="{rec['pdp_url']}" target="_blank" style="color:#1f77b4;">🔗 View Product</a>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>📌 Built with ❤️ by <b>PRIYANK TYAGI</b> — Fashion ML Visual Search Assignment</center>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# --- Page Configuration ---
st.set_page_config(
    page_title="Fashion Visual Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

    /* Global styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }
    
    /* Header styles */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        font-weight: 300;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
    }
    
    /* Card styles */
    .product-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Product grid styles */
    .product-grid-item {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .product-grid-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .product-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        transition: transform 0.3s ease;
    }
    
    .product-image:hover {
        transform: scale(1.05);
    }
    
    .product-brand {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .product-price {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #059669;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .product-link {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 500;
        text-align: center;
        transition: all 0.3s ease;
        margin-top: auto;
        user-select: none;
        display: inline-block;
    }
    
    .product-link:hover,
    .product-link:focus {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        text-decoration: none;
        color: white;
        outline-offset: 2px;
        outline: 2px solid transparent;
    }
    .product-link:focus-visible {
        outline-color: #667eea;
    }
    
    /* Detail card styles */
    .detail-card {
        background: #ffffff;
        border-radius: 0.75rem;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
        font-family: 'Inter', sans-serif;
        color: #6b7280;
        line-height: 1.6;
    }
    
    .detail-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 1.2rem;
        line-height: 1.1;
    }
    
    .detail-item {
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    
    .detail-label {
        font-weight: 600;
        color: #374151;
    }
    
    .price-highlight {
        font-size: 2rem;
        font-weight: 700;
        color: #059669;
        margin-right: 1rem;
        vertical-align: middle;
    }
    
    .discount-badge {
        background: #ef4444;
        color: white;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        vertical-align: middle;
        display: inline-block;
    }
    
    .view-product-btn {
        background-color: #111827;
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        display: inline-block;
        margin-top: 2rem;
        user-select: none;
        text-align: center;
    }
    
    .view-product-btn:hover,
    .view-product-btn:focus {
        background-color: #374151;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        text-decoration: none;
        color: white;
        outline-offset: 2px;
        outline: 2px solid transparent;
    }
    
    .view-product-btn:focus-visible {
        outline-color: #667eea;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .detail-card {
            padding: 2rem 1.5rem;
        }
        .detail-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
    }

    /* Loading animation */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 3px solid #667eea;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header" role="banner">
    <h1 class="main-title">Fashion Visual Search</h1>
    <p class="main-subtitle">Discover Your Perfect Style – Instantly with AI</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pretrained ResNet50 model with pooling, without top."""
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract feature vector from a PIL image using the ResNet50 model.
    Args:
        img (PIL.Image.Image): Input image.
    Returns:
        np.ndarray: Flattened feature vector extracted from the image.
    """
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()


@st.cache_data(show_spinner=False)
def load_data():
    """
    Load fashion dataset and features from disk, apply column renaming and cleaning.
    Returns:
        tuple(pandas.DataFrame, np.ndarray): Fashion data DataFrame and features array.
    """
    df = pd.read_csv("fashion_data_filtered.csv")
    feats = np.load("fashion_features.npy")

    # Rename columns for consistency if old names exist
    if "feature_image" not in df.columns and "feature_image_s3" in df.columns:
        df.rename(columns={"feature_image_s3": "feature_image"}, inplace=True)
    if "style_attribute" not in df.columns and "style_attributes" in df.columns:
        df.rename(columns={"style_attributes": "style_attribute"}, inplace=True)

    df.dropna(subset=["feature_image"], inplace=True)

    return df, feats


# Initialize model once
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📤 Upload Your Fashion Image")
    st.markdown("Upload an image to find similar fashion items")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a fashion item"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown("---")
        st.markdown("### 📸 Your Image")
        st.image(img, use_container_width=True, caption="Uploaded Image")


def parse_price(val) -> str:
    """
    Format price information from various possible data styles to a string with currency symbol.
    Args:
        val: Value representing price, could be string, dict, int, or float.
    Returns:
        str: Formatted price string.
    """
    try:
        d = eval(str(val)) if isinstance(val, str) else val
        if isinstance(d, dict):
            if "INR" in d:
                return f"₹{d['INR']:,.0f}"
            elif "USD" in d:
                return f"${d['USD']:,.2f}"
    except Exception:
        pass

    if isinstance(val, (int, float)):
        return f"₹{val:,.0f}"

    return f"₹{val}"


def parse_list(val) -> str:
    """
    Parse a string representation of a list and return a comma-separated string.
    If parsing fails, return the string as-is.
    Args:
        val: List-like string.
    Returns:
        str: Comma separated string or original string.
    """
    try:
        l = eval(str(val))
        if isinstance(l, list):
            return ", ".join(str(i) for i in l)
    except Exception:
        pass
    return str(val)


def parse_dict(val) -> str:
    """
    Parse a string representation of a dictionary and format it for display.
    If parsing fails, return the string as-is.
    Args:
        val: Dictionary-like string.
    Returns:
        str: Formatted multi-line string or original string.
    """
    try:
        d = eval(str(val))
        if isinstance(d, dict):
            return "  \n".join([f"**{k}**: {v}" for k, v in d.items()])
    except Exception:
        pass
    return str(val)


# --- Main Content ---
if uploaded_file:
    # Reuse opened image from sidebar upload to avoid reopening
    img = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("🔍 Analyzing your image and finding matches..."):
        df, feats = load_data()
        query_feat = extract_features(img)
        similarities = cosine_similarity([query_feat], feats)[0]
        top_indices = similarities.argsort()[-6:][::-1]
        st.session_state.history.append(query_feat)

    best = df.iloc[top_indices[0]]

    # --- Top Match Section ---
    st.markdown('<h2 class="section-header">🎯 Perfect Match</h2>', unsafe_allow_html=True)
    
    st.markdown(f'''
    <section class="detail-card" aria-labelledby="product-name" role="region">
        <div style="display:flex; flex-wrap: wrap; gap: 2.5rem; align-items: start;">
            <img src="{best["feature_image"]}" alt="Image of {best['product_name']}" 
                style="width: 350px; height: auto; border-radius: 0.75rem; object-fit: cover; box-shadow: 0 8px 20px rgba(0,0,0,0.06);" />
            <div style="flex: 1 1 550px; min-width: 280px;">
                <h3 id="product-name" class="detail-title" style="margin-top:0;">{best['product_name']}</h3>
                <p class="detail-item"><span class="detail-label">Brand:</span> {best['brand']}</p>
                <p class="detail-item" style="margin-top: 0; margin-bottom: 1.2rem;">
                    <span class="price-highlight">{parse_price(best['selling_price'])}</span>
                    <span class="discount-badge">{best['discount']}% OFF</span>
                </p>
                <p class="detail-item"><span class="detail-label">Category:</span> {best['category_id']} | {best['department_id']}</p>
                <p class="detail-item"><span class="detail-label">Features:</span> {parse_list(best['feature_list'])}</p>
                <p class="detail-item" style="margin-bottom: 1.2rem;"><span class="detail-label">Description:</span> {best['description']}</p>
                <p class="detail-item"><span class="detail-label">Style Attributes:</span><br>{parse_dict(best['style_attribute'])}</p>
                {f'<a href="{best["pdp_url"]}" target="_blank" rel="noopener noreferrer" class="view-product-btn" role="button" tabindex="0">🛍️ View Product</a>' if best["pdp_url"] else ""}
            </div>
        </div>
    </section>
    ''', unsafe_allow_html=True)

    # --- Similar Products Section ---
    st.markdown('<h2 class="section-header">✨ Similar Products</h2>', unsafe_allow_html=True)
    
    cols = st.columns(5, gap="medium")
    for i, idx in enumerate(top_indices[1:]):
        prod = df.iloc[idx]
        with cols[i]:
            st.markdown(f"""
            <div class="product-grid-item" role="listitem">
                <img src="{prod['feature_image']}" class="product-image" alt="{prod['product_name']}"/>
                <div class="product-brand">{prod['brand']}</div>
                <div class="product-price">{parse_price(prod['selling_price'])} • {prod['discount']}% off</div>
                {f'<a href="{prod["pdp_url"]}" target="_blank" rel="noopener noreferrer" class="product-link" role="link" tabindex="0">View Product</a>' if prod["pdp_url"] else ""}
            </div>
            """, unsafe_allow_html=True)

    # --- Outfit Suggestions Section ---
    st.markdown('<h2 class="section-header">👗 Style Suggestions</h2>', unsafe_allow_html=True)
    
    try:
        style_keyword = str(best["style_attribute"]).split(",")[0].strip().lower()
        outfit_matches = df[df["style_attribute"].astype(str).str.lower().str.contains(style_keyword, na=False)]
        outfit_matches = outfit_matches[outfit_matches.index != best.name]
        outfit_df = outfit_matches.sample(n=min(5, len(outfit_matches))) if len(outfit_matches) > 0 else pd.DataFrame()
    except Exception:
        outfit_df = pd.DataFrame()

    if outfit_df.empty:
        st.markdown("""
        <div class="product-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #64748b; margin-bottom: 1rem;">No matching outfit suggestions found</h3>
            <p style="color: #94a3b8;">Try uploading a different image or explore our trending picks below!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(5, gap="medium")
        for i, (_, item) in enumerate(outfit_df.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="product-grid-item" role="listitem">
                    <img src="{item['feature_image']}" class="product-image" alt="{item['product_name']}"/>
                    <div class="product-brand">{item['brand']}</div>
                    <div class="product-price">{parse_price(item['selling_price'])} • {item['discount']}% off</div>
                    {f'<a href="{item["pdp_url"]}" target="_blank" rel="noopener noreferrer" class="product-link" role="link" tabindex="0">View Product</a>' if item["pdp_url"] else ""}
                </div>
                """, unsafe_allow_html=True)

    # --- Trending Section ---
    st.markdown('<h2 class="section-header">🔥 Trending Now</h2>', unsafe_allow_html=True)
    
    df["launch_on"] = pd.to_datetime(df["launch_on"], errors="coerce")
    trendy = df.dropna(subset=["launch_on"]).sort_values(by=["discount", "launch_on"], ascending=[False, False]).head(5)
    
    cols = st.columns(5, gap="medium")
    for i, (_, row) in enumerate(trendy.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="product-grid-item" role="listitem">
                <img src="{row['feature_image']}" class="product-image" alt="{row['product_name']}"/>
                <div class="product-brand">{row['brand']}</div>
                <div class="product-price">{parse_price(row['selling_price'])} • {row['discount']}% off</div>
                {f'<a href="{row["pdp_url"]}" target="_blank" rel="noopener noreferrer" class="product-link" role="link" tabindex="0">View Product</a>' if row["pdp_url"] else ""}
            </div>
            """, unsafe_allow_html=True)

    # --- Personalized Recommendations ---
    if len(st.session_state.history) > 1:
        st.markdown('<h2 class="section-header">🧠 Just For You</h2>', unsafe_allow_html=True)
        
        avg_feat = np.mean(st.session_state.history, axis=0)
        sims = cosine_similarity([avg_feat], feats)[0]
        rec_indices = sims.argsort()[-6:][::-1]
        
        cols = st.columns(5, gap="medium")
        for i, idx in enumerate(rec_indices[1:]):
            rec = df.iloc[idx]
            with cols[i]:
                st.markdown(f"""
                <div class="product-grid-item" role="listitem">
                    <img src="{rec['feature_image']}" class="product-image" alt="{rec['product_name']}"/>
                    <div class="product-brand">{rec['brand']}</div>
                    <div class="product-price">{parse_price(rec['selling_price'])} • {rec['discount']}% off</div>
                    {f'<a href="{rec["pdp_url"]}" target="_blank" rel="noopener noreferrer" class="product-link" role="link" tabindex="0">View Product</a>' if rec["pdp_url"] else ""}
                </div>
                """, unsafe_allow_html=True)

else:
    # --- Welcome Section ---
    st.markdown("""
    <div class="detail-card" style="text-align: center; padding: 4rem 2rem; margin-top: 2rem;">
        <h2 style="color: #1e293b; margin-bottom: 1rem; font-family: 'Playfair Display', serif;">
            Snap it. Search it. Style it.
        </h2>
        <p style="color: #64748b; font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;">
            Upload an image of any fashion item and discover similar products, style suggestions, and personalized recommendations powered by advanced AI technology.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Visual Search</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Find exact matches</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">👗</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Style Suggestions</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Complete your look</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Smart Recommendations</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Personalized for you</p>
            </div>
                 </p><br>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: white; background: rgba(255, 255, 255, 0.1); border-radius: 16px; margin-top: 2rem;">
    <p style="margin: 0; font-size: 1rem; font-weight: 500;">
        Built with ❤️ by <strong>PRIYANK TYAGI</strong>
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
        Fashion ML Visual Search Assignment
    </p>
</div>
""", unsafe_allow_html=True)

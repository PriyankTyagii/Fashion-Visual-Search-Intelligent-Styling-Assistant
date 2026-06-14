import os
import html as _html
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import gdown


def _generate_pwa_icons():
    os.makedirs("static", exist_ok=True)
    has_custom_icon = os.path.exists("icon.png")
    for size in (192, 512):
        path = f"static/icon-{size}.png"
        if has_custom_icon:
            try:
                with Image.open("icon.png") as img:
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.LANCZOS
                    img_resized = img.resize((size, size), resample)
                    img_resized.save(path, "PNG")
                continue
            except Exception:
                pass
        if os.path.exists(path):
            continue
        icon = Image.new("RGB", (size, size), (124, 58, 237))
        draw = ImageDraw.Draw(icon)
        fs = int(size * 0.52)
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "F", font=font)
        x = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), "F", fill="white", font=font)
        icon.save(path)

_generate_pwa_icons()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Search Engine",
    page_icon="static/icon-192.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

/* ── Base ── */
.main { background: linear-gradient(145deg, #1a1a3e 0%, #4a1274 55%, #1a3a5e 100%); min-height: 100vh; }
.block-container { padding: 0 2rem 4rem; max-width: 1400px; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
/* keep header transparent (not hidden) so the collapsed-sidebar expand arrow stays clickable */
header[data-testid="stHeader"] { background: transparent !important; }
header[data-testid="stHeader"] .stToolbar, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Desktop: keep the sidebar permanently visible so it can never get stuck off-screen ── */
@media (min-width: 768px) {
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important; visibility: visible !important;
        margin-left: 0 !important; left: 0 !important;
        min-width: 300px !important; width: 300px !important;
    }
    /* hide the collapse (X) button so users can't hide the sidebar into an unrecoverable state */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }
}

/* ── Mobile: allow collapsing, with a clearly visible expand button ── */
@media (max-width: 767px) {
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important; display: block !important; opacity: 1 !important;
        position: fixed !important; top: 0.6rem; left: 0.6rem; z-index: 1000000 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background: rgba(124,58,237,0.92) !important; border-radius: 10px !important;
        width: 42px; height: 42px; box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg { color: #fff !important; fill: #fff !important; width: 22px; height: 22px; }

    /* mobile-friendly typography & spacing */
    .block-container { padding: 0 1rem 3rem !important; }
    .app-header { padding: 1.75rem 1rem 1rem; }
    .app-header h1 { font-size: 2rem; }
    .app-header p { font-size: 0.85rem; }
    .welcome { padding: 2.25rem 1.1rem; border-radius: 18px; }
    .welcome h2 { font-size: 1.55rem; }
    .welcome p { font-size: 0.9rem; margin-bottom: 1.5rem; }
    .sec-head { font-size: 1.25rem; margin: 1.5rem 0 0.75rem; }
    .match-name, .match-body > div { font-size: 1.4rem !important; }
    .pcard-name { font-size: 0.9rem; }

    /* center the search/upload and remove previews on mobile */
    section[data-testid="stSidebar"] { text-align: center; }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { text-align: center; }
    .pills { display: none !important; }                                  /* hide feature preview cards */
    section[data-testid="stSidebar"] [data-testid="stImage"] { display: none !important; } /* hide uploaded-image preview */

    /* stack the large "Perfect Match" card vertically */
    .match-wrap { flex-direction: column; }
    .match-img { width: 100% !important; min-width: 0 !important; max-height: 320px; }
    .match-body { padding: 1.5rem 1.25rem !important; }
    .match-grid { grid-template-columns: 1fr 1fr; gap: 0.5rem 1rem; }
}

/* ── App header ── */
.app-header { text-align: center; padding: 2.5rem 2rem 1.5rem; }
.app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 700; color: #fff;
    letter-spacing: -0.02em; margin: 0 0 0.4rem;
}
.app-header p { font-size: 1rem; color: rgba(255,255,255,0.6); font-weight: 300; margin: 0; }

/* ── Section heading ── */
.sec-head {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem; font-weight: 600; color: #fff;
    margin: 2rem 0 0.9rem; letter-spacing: 0.01em;
    display: flex; align-items: center; gap: 0.75rem;
}
.sec-head::before {
    content: ''; display: block; flex-shrink: 0;
    width: 4px; height: 1.3em;
    background: linear-gradient(180deg, #c4b5fd 0%, #7c3aed 100%);
    border-radius: 2px;
}

/* ── Product scroll row ── */
.scroll-row {
    display: flex; gap: 1rem; overflow-x: auto; padding-bottom: 0.75rem;
    scroll-snap-type: x mandatory; -webkit-overflow-scrolling: touch;
}
.scroll-row::-webkit-scrollbar { height: 4px; }
.scroll-row::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.5); border-radius: 2px; }
.scroll-row .pcard {
    flex: 0 0 220px; scroll-snap-align: start;
}
@media (min-width: 768px) {
    .scroll-row { overflow-x: visible; flex-wrap: wrap; }
    .scroll-row .pcard { flex: 1 1 calc(25% - 0.75rem); min-width: 180px; max-width: calc(25% - 0.75rem); }
}

/* ── Product card ── */
.pcard {
    background: #fff; border-radius: 16px; overflow: hidden;
    box-shadow: 0 2px 14px rgba(0,0,0,0.09);
    display: flex; flex-direction: column; height: 100%;
    transition: transform 0.2s, box-shadow 0.2s;
}
.pcard:hover { transform: translateY(-5px); box-shadow: 0 12px 36px rgba(0,0,0,0.17); }
.pcard-img { width: 100%; aspect-ratio: 3/4; overflow: hidden; background: #f3f4f6; }
.pcard-img img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.35s; }
.pcard:hover .pcard-img img { transform: scale(1.05); }
.pcard-body { padding: 0.8rem 0.9rem 0.9rem; display: flex; flex-direction: column; flex: 1; }
.pcard-brand { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #a78bfa; margin-bottom: 0.2rem; }
.pcard-name {
    font-size: 0.82rem; font-weight: 500; color: #111827; line-height: 1.35; flex: 1;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    margin-bottom: 0.4rem;
}
.pcard-price-row { display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.7rem; }
.pcard-price { font-size: 0.88rem; font-weight: 700; color: #059669; }
.pcard-off { font-size: 0.62rem; font-weight: 700; background: #fef2f2; color: #dc2626; padding: 0.15rem 0.4rem; border-radius: 4px; }
.pcard-btn {
    display: block; text-align: center; background: #111827; color: #fff !important;
    text-decoration: none !important; font-size: 0.73rem; font-weight: 600;
    letter-spacing: 0.05em; padding: 0.5rem; border-radius: 8px; transition: background 0.2s;
}
.pcard-btn:hover { background: #7c3aed; }

/* ── Top match card ── */
.match-wrap {
    background: #fff; border-radius: 20px; overflow: hidden;
    box-shadow: 0 8px 48px rgba(0,0,0,0.2); display: flex; margin-bottom: 2rem;
}
.match-img { width: 360px; min-width: 260px; max-height: 540px; flex-shrink: 0; overflow: hidden; }
.match-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.match-body {
    padding: 2.5rem 2.75rem; flex: 1;
    display: flex; flex-direction: column; justify-content: center; overflow: auto;
}
.match-eyebrow {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #7c3aed; background: #ede9fe; display: inline-block;
    padding: 0.25rem 0.65rem; border-radius: 20px; margin-bottom: 0.75rem; width: fit-content;
}
.match-brand { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.3rem; }
.match-name { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: #111827; line-height: 1.2; margin-bottom: 0.9rem; }
.match-price-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.4rem; }
.match-price { font-size: 1.55rem; font-weight: 700; color: #059669; }
.match-disc { background: #dc2626; color: #fff; font-size: 0.72rem; font-weight: 700; padding: 0.28rem 0.65rem; border-radius: 20px; }
.match-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 2rem; margin-bottom: 1.1rem; }
.match-field label { display: block; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.1rem; }
.match-field span { font-size: 0.83rem; color: #374151; }
.match-desc { font-size: 0.83rem; color: #6b7280; line-height: 1.65; margin-bottom: 1.4rem; border-left: 3px solid #ede9fe; padding-left: 0.75rem; }
.match-cta {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #111827; color: #fff !important; text-decoration: none !important;
    font-weight: 700; font-size: 0.83rem; letter-spacing: 0.05em;
    padding: 0.82rem 1.75rem; border-radius: 10px; width: fit-content; transition: all 0.2s;
}
.match-cta:hover { background: #7c3aed; transform: translateY(-2px); box-shadow: 0 8px 24px rgba(124,58,237,0.3); }

/* ── Welcome page ── */
.welcome {
    background: rgba(255,255,255,0.06); backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12); border-radius: 24px;
    padding: 5rem 3rem; text-align: center; margin-top: 1.5rem;
}
.welcome h2 { font-family: 'Playfair Display', serif; font-size: 2.5rem; color: #fff; margin: 0 0 1rem; }
.welcome p { color: rgba(255,255,255,0.7); font-size: 1rem; max-width: 500px; margin: 0 auto 3rem; line-height: 1.7; }
.pills { display: flex; justify-content: center; gap: 1.25rem; flex-wrap: wrap; }
.pill { background: rgba(255,255,255,0.09); border: 1px solid rgba(255,255,255,0.14); border-radius: 16px; padding: 1.25rem 1.75rem; color: #fff; min-width: 145px; }
.pill-num { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: rgba(255,255,255,0.2); margin-bottom: 0.6rem; line-height: 1; }
.pill-title { font-size: 0.88rem; font-weight: 600; margin-bottom: 0.2rem; color: #fff; }
.pill-desc { font-size: 0.73rem; color: rgba(255,255,255,0.55); }

/* ── Mobile upload section (shown inline on small screens) ── */
.mobile-upload-wrap {
    display: none;
    text-align: center; margin-top: 1.5rem;
}
@media (max-width: 767px) {
    .mobile-upload-wrap { display: block; }
    /* hide the label text above the dropzone */
    .mobile-upload-wrap [data-testid="stFileUploaderDropzoneInstructions"] span { display: none; }
}
/* hide the mobile uploader (in main content) on desktop — sidebar uploader is unaffected */
@media (min-width: 768px) {
    .mobile-upload-outer { display: none !important; }
    section[data-testid="stMain"] [data-testid="stFileUploader"] { display: none !important; }
    section[data-testid="stMain"] [data-testid="stCameraInput"] { display: none !important; }
    section[data-testid="stMain"] [data-testid="stRadio"] { display: none !important; }
}

/* ── Empty state ── */
.empty-state {
    background: rgba(255,255,255,0.06); border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 14px; padding: 2.5rem; text-align: center;
    color: rgba(255,255,255,0.5); font-size: 0.88rem;
}

/* ── Footer ── */
.app-footer {
    text-align: center; color: rgba(255,255,255,0.35); font-size: 0.78rem;
    padding: 2rem 0 0; margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.08);
}
.app-footer strong { color: rgba(255,255,255,0.65); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.5); border-radius: 3px; }
</style>""", unsafe_allow_html=True)


# ── PWA: manifest link, meta tags, service worker, install bubble ──────────────
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Visual Search Engine">
<meta name="theme-color" content="#7c3aed">
<link rel="apple-touch-icon" href="/app/static/icon-192.png">

<style>
#pwa-bubble {
    position: fixed; bottom: 1.25rem; left: 50%; transform: translateX(-50%);
    z-index: 9999999;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: #fff; border-radius: 50px;
    padding: 0.75rem 1.25rem 0.75rem 1rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.45);
    display: none; align-items: center; gap: 0.75rem;
    font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 600;
    white-space: nowrap;
}
@keyframes bubble-in {
    from { opacity:0; transform: translateX(-50%) translateY(24px); }
    to   { opacity:1; transform: translateX(-50%) translateY(0); }
}
#pwa-bubble.visible {
    display: flex;
    animation: bubble-in 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
#pwa-install-btn {
    background: rgba(255,255,255,0.22); border: none; color: #fff;
    font-size: 0.82rem; font-weight: 700; padding: 0.38rem 0.9rem;
    border-radius: 30px; cursor: pointer; transition: background 0.2s;
}
#pwa-install-btn:hover { background: rgba(255,255,255,0.35); }
#pwa-close-btn {
    background: none; border: none; color: rgba(255,255,255,0.7);
    font-size: 1.1rem; cursor: pointer; padding: 0 0.15rem; line-height: 1;
}
@media (min-width: 768px) { #pwa-bubble { display: none !important; } }
</style>

<div id="pwa-bubble">
    &#128242;&nbsp; Install App
    <button id="pwa-install-btn" onclick="pwaBubbleInstall()">Install</button>
    <button id="pwa-close-btn" onclick="pwaBubbleClose()">&#x2715;</button>
</div>

<script>
(function() {
    var deferredPrompt = null;
    var isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
    var isStandalone = window.matchMedia('(display-mode: standalone)').matches
                    || !!window.navigator.standalone;

    // Don't show if already installed or dismissed this session
    if (isStandalone || sessionStorage.getItem('pwa-dismissed')) return;

    // Try to register service worker (needed for Chrome install criteria on HTTPS)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/app/static/sw.js').catch(function(){});
    }

    // Capture Chrome/Android native install prompt
    window.addEventListener('beforeinstallprompt', function(e) {
        e.preventDefault();
        deferredPrompt = e;
    });

    // Show bubble unconditionally on mobile after page settles
    function showBubble() {
        if (sessionStorage.getItem('pwa-dismissed')) return;
        var b = document.getElementById('pwa-bubble');
        if (b) b.classList.add('visible');
    }
    // Try immediately, and again after a delay in case DOM isn't ready yet
    showBubble();
    setTimeout(showBubble, 1500);
    document.addEventListener('DOMContentLoaded', showBubble);

    window.pwaBubbleInstall = function() {
        if (deferredPrompt) {
            // Chrome Android with HTTPS — native dialog
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then(function() {
                deferredPrompt = null;
                pwaBubbleClose();
            });
        } else if (isIOS) {
            alert('To install:\n1. Tap the Share button (box with arrow) in Safari\n2. Tap "Add to Home Screen"\n3. Tap "Add"');
        } else {
            alert('To install:\n1. Tap the browser menu (⋮ or ...)\n2. Tap "Add to Home screen" or "Install app"\n3. Tap "Add"');
        }
    };

    window.pwaBubbleClose = function() {
        var b = document.getElementById('pwa-bubble');
        if (b) b.classList.remove('visible');
        sessionStorage.setItem('pwa-dismissed', '1');
    };
})();
</script>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Visual Search Engine</h1>
    <p>Discover your perfect style – instantly with AI</p>
</div>
""", unsafe_allow_html=True)


# ── ML helpers ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


def extract_features(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("fashion_data_filtered.csv")

    output_path = "fashion_features.npy"
    if not os.path.exists(output_path):
        url = "https://drive.google.com/uc?id=1EbEim-d16D6P2aW-LqApjWfAmwE85iwY"
        gdown.download(url, output_path, quiet=False)

    feats = np.load(output_path)

    if "feature_image" not in df.columns and "feature_image_s3" in df.columns:
        df.rename(columns={"feature_image_s3": "feature_image"}, inplace=True)
    if "style_attribute" not in df.columns and "style_attributes" in df.columns:
        df.rename(columns={"style_attributes": "style_attribute"}, inplace=True)

    df.dropna(subset=["feature_image"], inplace=True)
    return df, feats


# ── Display helpers ───────────────────────────────────────────────────────────
def parse_price(val) -> str:
    try:
        d = eval(str(val)) if isinstance(val, str) else val
        if isinstance(d, dict):
            if "INR" in d:
                return f"₹{d['INR']:,.0f}"
            if "USD" in d:
                return f"${d['USD']:,.2f}"
    except Exception:
        pass
    if isinstance(val, (int, float)):
        return f"₹{val:,.0f}"
    return f"₹{val}"


def parse_list(val) -> str:
    try:
        lst = eval(str(val))
        if isinstance(lst, list):
            return ", ".join(str(i) for i in lst if str(i).strip())
    except Exception:
        pass
    return str(val)


def parse_dict(val) -> str:
    try:
        d = eval(str(val))
        if isinstance(d, dict):
            return "  \n".join(f"**{k}**: {v}" for k, v in d.items())
    except Exception:
        pass
    return str(val)


def parse_dict_html(val) -> str:
    """Like parse_dict but returns escaped HTML — no markdown markers."""
    e = _html.escape
    try:
        d = eval(str(val))
        if isinstance(d, dict) and d:
            return " &nbsp;·&nbsp; ".join(
                f"<strong>{e(str(k))}</strong>: {e(str(v))}" for k, v in d.items()
            )
    except Exception:
        pass
    return ""


def safe_str(val) -> str:
    """Return empty string for NaN/None, otherwise string."""
    if val is None:
        return ""
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return ""
    except Exception:
        pass
    s = str(val)
    return "" if s.lower() == "nan" else s


def pcard_html(img_url, brand, name, price_str, discount, pdp_url) -> str:
    e = _html.escape
    img_url  = safe_str(img_url)
    brand    = safe_str(brand)
    name     = safe_str(name)
    pdp_url  = safe_str(pdp_url)
    try:
        disc_val = float(discount)
    except (TypeError, ValueError):
        disc_val = 0.0
    badge = f'<span class="pcard-off">{disc_val:.0f}% OFF</span>' if disc_val else ""
    btn = (
        f'<a href="{e(pdp_url)}" target="_blank" rel="noopener noreferrer" class="pcard-btn">View Product →</a>'
        if pdp_url else ""
    )
    return (
        f'<div class="pcard">'
        f'<div class="pcard-img"><img src="{e(img_url)}" alt="{e(name)}" loading="lazy"/></div>'
        f'<div class="pcard-body">'
        f'<div class="pcard-brand" style="color:#a78bfa;">{e(brand)}</div>'
        f'<div class="pcard-name" style="color:#111827;">{e(name)}</div>'
        f'<div class="pcard-price-row"><span class="pcard-price" style="color:#059669;">{price_str}</span>{badge}</div>'
        f'{btn}'
        f'</div></div>'
    )


def render_grid(rows):
    cards = "".join(
        pcard_html(
            row["feature_image"],
            row["brand"],
            row["product_name"],
            parse_price(row["selling_price"]),
            row["discount"],
            row.get("pdp_url") or "",
        )
        for row in rows[:4]
    )
    st.markdown(f'<div class="scroll-row">{cards}</div>', unsafe_allow_html=True)


# ── Bootstrap ─────────────────────────────────────────────────────────────────
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Search Fashion Image")
    st.markdown("Upload a photo or capture one with your camera to find visually similar products instantly.")

    input_source = st.radio(
        "Select input source",
        options=["Upload Image", "Capture Image"],
        horizontal=True,
        label_visibility="collapsed",
        key="sidebar_source"
    )

    uploaded_file = None
    captured_file = None

    if input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="sidebar_uploader",
        )
    else:
        captured_file = st.camera_input(
            "Take a picture",
            label_visibility="collapsed",
            key="sidebar_camera",
        )

    active_file = uploaded_file or captured_file

    if active_file:
        img_preview = Image.open(active_file).convert("RGB")
        st.image(img_preview, use_container_width=True, caption="Your image")
        st.markdown("---")
        st.markdown(
            "**Tips for best results**\n"
            "- Well-lit, clear shots\n"
            "- Single garment per photo\n"
            "- Front-facing angle"
        )


# ── Mobile upload (centered, no preview — hidden on desktop via CSS) ──────────
st.markdown('<div class="mobile-upload-outer">', unsafe_allow_html=True)
mobile_source = st.radio(
    "Select input source",
    options=["Upload Image", "Capture Image"],
    horizontal=True,
    label_visibility="collapsed",
    key="mobile_source",
)

mobile_file = None
mobile_captured = None

if mobile_source == "Upload Image":
    mobile_file = st.file_uploader(
        "Upload a fashion image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="mobile_uploader",
    )
else:
    mobile_captured = st.camera_input(
        "Take a picture",
        label_visibility="collapsed",
        key="mobile_camera",
    )
st.markdown('</div>', unsafe_allow_html=True)

# whichever uploader/camera has a file wins
uploaded_file = active_file or mobile_file or mobile_captured


# ── Main content ──────────────────────────────────────────────────────────────
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analysing your image…"):
        df, feats = load_data()
        query_feat = extract_features(img)
        sims = cosine_similarity([query_feat], feats)[0]
        top_indices = sims.argsort()[-6:][::-1]
        st.session_state.history.append(query_feat)

    best = df.iloc[top_indices[0]]

    # ── Perfect match ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Perfect Match</div>', unsafe_allow_html=True)

    e = _html.escape
    disc_badge = (
        f'<span class="match-disc">{float(best["discount"]):.0f}% OFF</span>'
        if best["discount"] else ""
    )
    pdp = safe_str(best.get("pdp_url", ""))
    cta = (
        f'<a href="{e(pdp)}" target="_blank" rel="noopener noreferrer" class="match-cta">Shop Now →</a>'
        if pdp else ""
    )
    style_html   = parse_dict_html(best["style_attribute"]) or "—"
    feat_snippet = e(parse_list(best["feature_list"])[:160])
    description  = e(safe_str(best["description"]))

    st.markdown(
        f'<div class="match-wrap">'
          f'<div class="match-img"><img src="{e(safe_str(best["feature_image"]))}" alt="{e(safe_str(best["product_name"]))}"/></div>'
          f'<div class="match-body">'
            f'<div class="match-eyebrow">Best Match</div>'
            f'<div class="match-brand" style="font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:.3rem;">{e(safe_str(best["brand"]))}</div>'
            f'<div style="font-family:\'Playfair Display\',serif;font-size:1.9rem;font-weight:700;color:#111827;line-height:1.2;margin-bottom:.9rem;">{e(safe_str(best["product_name"]))}</div>'
            f'<div class="match-price-row"><span class="match-price">{parse_price(best["selling_price"])}</span>{disc_badge}</div>'
            f'<div class="match-grid">'
              f'<div class="match-field"><label style="display:block;font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:.1rem;">Category</label><span style="font-size:.83rem;color:#374151;">{e(safe_str(best["category_id"]))}</span></div>'
              f'<div class="match-field"><label style="display:block;font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:.1rem;">Department</label><span style="font-size:.83rem;color:#374151;">{e(safe_str(best["department_id"]))}</span></div>'
              f'<div class="match-field"><label style="display:block;font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:.1rem;">Features</label><span style="font-size:.83rem;color:#374151;">{feat_snippet}{"…" if len(feat_snippet) >= 160 else ""}</span></div>'
              f'<div class="match-field"><label style="display:block;font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:.1rem;">Style</label><span style="font-size:.83rem;color:#374151;">{style_html}</span></div>'
            f'</div>'
            f'{"<p class=match-desc style=font-size:.85rem;color:#6b7280;line-height:1.65;margin-bottom:1.4rem;border-left:3px solid #ede9fe;padding-left:.75rem;>" + description + "</p>" if description else ""}'
            f'{cta}'
          f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Similar products ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Similar Products</div>', unsafe_allow_html=True)
    render_grid([df.iloc[i] for i in top_indices[1:5]])

    # ── Style suggestions ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Style Suggestions</div>', unsafe_allow_html=True)
    try:
        style_kw = str(best["style_attribute"]).split(",")[0].strip().lower()
        outfit = df[df["style_attribute"].astype(str).str.lower().str.contains(style_kw, na=False)]
        outfit = outfit[outfit.index != best.name]
        outfit = outfit.sample(n=min(4, len(outfit))) if len(outfit) else pd.DataFrame()
    except Exception:
        outfit = pd.DataFrame()

    if outfit.empty:
        st.markdown(
            '<div class="empty-state">No matching style suggestions found. Try a different image.</div>',
            unsafe_allow_html=True,
        )
    else:
        render_grid(outfit.to_dict("records"))

    # ── Trending now ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Trending Now</div>', unsafe_allow_html=True)
    df["launch_on"] = pd.to_datetime(df["launch_on"], errors="coerce")
    trendy = (
        df.dropna(subset=["launch_on"])
        .sort_values(["discount", "launch_on"], ascending=[False, False])
        .head(4)
    )
    render_grid(trendy.to_dict("records"))

    # ── Personalised picks ────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown('<div class="sec-head">Just For You</div>', unsafe_allow_html=True)
        avg_feat = np.mean(st.session_state.history, axis=0)
        rec_sims = cosine_similarity([avg_feat], feats)[0]
        rec_indices = rec_sims.argsort()[-5:][::-1]
        render_grid([df.iloc[i] for i in rec_indices[1:5]])

else:
    # ── Welcome / landing page ────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome">
        <h2>Snap it. Search it. Style it.</h2>
        <p>Upload any fashion photo and discover visually similar products,
           style suggestions, and AI-powered recommendations.</p>
        <div class="mobile-upload-wrap">
            <p style="color:rgba(255,255,255,0.5);font-size:0.82rem;margin-bottom:0.6rem;">Tap below to upload an image</p>
        </div>
        <div class="pills">
            <div class="pill">
                <div class="pill-num">01</div>
                <div class="pill-title">Visual Search</div>
                <div class="pill-desc">Find exact matches</div>
            </div>
            <div class="pill">
                <div class="pill-num">02</div>
                <div class="pill-title">Style Suggestions</div>
                <div class="pill-desc">Complete your look</div>
            </div>
            <div class="pill">
                <div class="pill-num">03</div>
                <div class="pill-title">Trending Now</div>
                <div class="pill-desc">Stay on trend</div>
            </div>
            <div class="pill">
                <div class="pill-num">04</div>
                <div class="pill-title">Personalised</div>
                <div class="pill-desc">Made for you</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built by <strong>PRIYANK TYAGI</strong> &nbsp;·&nbsp; Visual Search Engine
</div>
""", unsafe_allow_html=True)

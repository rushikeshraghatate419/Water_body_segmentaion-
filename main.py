# water_body_app_256x256_single_row_white.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import base64

st.set_page_config(page_title="🌊 Water Body Segmentation 256x256", layout="wide")

# ----------------- CSS for white background UI -----------------
st.markdown("""
<style>
    body {background-color: #ffffff; color: #003366; font-family: 'Segoe UI', sans-serif;}
    .stApp {display: flex; justify-content: center; align-items: center; flex-direction: column;}
    h1,h2,h3 {text-align: center; color: #00796b;}
    .stButton>button {
        background-color:#00acc1;
        color:white;
        height:3em;
        width:16em;
        border-radius:12px;
        border:2px solid #00838f;
        font-size:16px;
        font-weight:bold;
        margin:10px;
        transition:all 0.3s ease;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {background-color:#00838f;color:#ffffff;transform:scale(1.05);}
    .stImage>img {
        border:3px solid #00bcd4; 
        border-radius:12px; 
        cursor:pointer;
        box-shadow: 3px 3px 12px rgba(0,0,0,0.3);
    }
    .section-title {
        background-color:#00bfa5;
        color:white;
        padding:8px;
        border-radius:10px;
        text-align:center;
        font-size:20px;
        margin-bottom:10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Functions -----------------
def load_image(image_file):
    return Image.open(image_file).convert("RGB")

def resize_256x256(image):
    return image.resize((256, 256))

def overlay_mask_np(image, mask, color=(135,206,250)):
    img_np = np.array(image) / 255.0
    mask_np = np.array(mask.convert("L"))
    overlay = img_np.copy()
    overlay[mask_np > 128] = np.array(color) / 255.0
    return (overlay * 255).astype(np.uint8)

def edge_detection_np(base_img, mask=None):
    img_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        img_gray[mask == 0] = 0
    canny_edges = cv2.Canny(img_gray, 100, 200)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = ((sobel_edges / (sobel_edges.max() + 1e-6)) * 255).astype(np.uint8)
    sobel_edges = (sobel_edges > 50).astype(np.uint8) * 255
    return canny_edges, sobel_edges

def overlay_edges(base_img, edges, color):
    edge_canvas = np.zeros_like(base_img)
    edge_canvas[edges > 0] = color
    return cv2.addWeighted(base_img, 1.0, edge_canvas, 1.0, 0)

def calculate_pie_data(mask):
    mask = np.array(mask.convert("L"))
    water_pixels = np.sum(mask > 128)
    land_pixels = mask.size - water_pixels
    return water_pixels, land_pixels

def get_image_download_bytes(img_array):
    im = Image.fromarray(img_array)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def clickable_image(img_array, label):
    img_bytes = get_image_download_bytes(img_array)
    img_b64 = base64.b64encode(img_bytes).decode()
    return st.markdown(f"""
        <a href="data:image/png;base64,{img_b64}" download="{label.replace(' ','_')}.png">
            <img src="data:image/png;base64,{img_b64}" width="256"><br>{label}
        </a>
    """, unsafe_allow_html=True)

# ----------------- Streamlit Layout -----------------
st.title("🌊 Water Body Segmentation & Analysis")

uploaded_image = st.file_uploader("Upload Original Image", type=["jpg", "png", "jpeg"])
uploaded_mask = st.file_uploader("Upload Mask Image", type=["jpg", "png", "jpeg"])

if uploaded_image and uploaded_mask:
    # --- Load safely (prevent huge image crash) ---
    image = load_image(uploaded_image)
    mask = load_image(uploaded_mask)
    MAX_SIZE = (2048, 2048)
    image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
    mask.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

    # --- Resize for processing ---
    image_256 = resize_256x256(image)
    mask_256 = resize_256x256(mask)

    # --- Original + Mask ---
    st.markdown("<div style='display:flex; overflow-x:auto;'>", unsafe_allow_html=True)
    clickable_image(np.array(image_256), "Original")
    clickable_image(np.array(mask_256), "Mask")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Buttons ---
    col1, col2 = st.columns(2)
    show_overlay = col1.button("Show Predicted Overlay")
    show_edges = col2.button("Show Canny + Sobel on Predicted Only")

    # --- Overlay ---
    if show_overlay:
        pred_overlay = overlay_mask_np(image_256, mask_256)
        st.markdown("<div style='display:flex; overflow-x:auto;'>", unsafe_allow_html=True)
        clickable_image(pred_overlay, "Predicted Overlay")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Edges ---
    if show_edges:
        pred_overlay = overlay_mask_np(image_256, mask_256)
        gt_mask = (np.array(mask_256.convert("L")) > 127).astype(np.uint8)
        canny_pred, sobel_pred = edge_detection_np(pred_overlay, gt_mask)

        canny_on_pred = overlay_edges(pred_overlay, canny_pred, [255,165,0])
        sobel_on_pred = overlay_edges(pred_overlay, sobel_pred, [255,0,0])

        st.markdown("<div style='display:flex; overflow-x:auto;'>", unsafe_allow_html=True)
        clickable_image(canny_on_pred, "Canny on Predicted")
        clickable_image(sobel_on_pred, "Sobel on Predicted")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Pie Chart ---
    water, land = calculate_pie_data(mask_256)
    fig1, ax1 = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    ax1.pie([water, land], labels=["Water", "Land"], colors=["skyblue", "lightgreen"],
            autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    fig1.tight_layout()
    st.pyplot(fig1, width="content")

    # --- Save Analysis CSV ---
    analysis_data = {
        "Image": [uploaded_image.name],
        "Mask": [uploaded_mask.name],
        "Water_Pixels": [water],
        "Land_Pixels": [land],
        "Water_Percentage": [round(100 * water / (water + land), 2)],
        "Land_Percentage": [round(100 * land / (water + land), 2)]
    }
    csv_file = "water_analysis.csv"
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame(analysis_data)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final.to_csv(csv_file, index=False)
    else:
        df_final = pd.DataFrame(analysis_data)
        df_final.to_csv(csv_file, index=False)

    st.success(f"Analysis saved to `{csv_file}`")

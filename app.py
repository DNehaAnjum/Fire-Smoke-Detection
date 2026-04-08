import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Fire Detection", page_icon="🔥", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
.glass-card{
background: rgba(255,255,255,0.08);
border-radius:15px;
padding:20px;
margin-bottom:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}
h1,h2,h3{text-align:center;color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- MENU ----------------
selected = option_menu(
    "🔥 AI Fire Detection Dashboard",
    ["Home", "Image", "Video"],
    icons=["house", "image", "camera-video"],
    orientation="horizontal"
)

# ---------------- HOME ----------------
if selected == "Home":

    st.markdown("""
    <div class="glass-card">
    <h1>🔥 AI Powered Fire & Smoke Detection</h1>
    <p style="text-align:center;font-size:18px;">
    Real-time monitoring using YOLO Deep Learning. Detect fire from images and videos.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # 🔥 WORKING VIDEO (same like your screenshot)
    st.video("https://media0.giphy.com/media/lMUGMp2lImgGA/giphy.mp4")

# ---------------- IMAGE ----------------
elif selected == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.success("✅ Image uploaded successfully")
        st.info("🔥 Detection output will appear here")

# ---------------- VIDEO ----------------
elif selected == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video:
        st.video(uploaded_video)

        st.success("✅ Video uploaded successfully")
        st.info("🔥 Detection output will appear here")

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from PIL import Image
import numpy as np
import glob

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Fire Detection", page_icon="🔥", layout="wide")

# ---------------- STYLE ----------------
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
    Detect fire using YOLO from images & videos
    </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- IMAGE ----------------
elif selected == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        model = YOLO("yolov9t.pt")

        image = Image.open(uploaded)
        frame = np.array(image)

        results = model(frame)

        result_img = results[0].plot()

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original")
        col2.image(result_img, caption="Detection")

# ---------------- VIDEO ----------------
elif selected == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video:

        st.video(uploaded_video)

        # Save temp
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())

        model = YOLO("yolov9t.pt")

        results = model.predict(source="temp.mp4", save=True)

        # Get latest output
        folders = glob.glob("runs/detect/*")
        latest = max(folders, key=os.path.getctime)

        output = glob.glob(f"{latest}/*.mp4")

        if output:
            st.success("✅ Done")
            st.video(output[0])
        else:
            st.error("❌ No output")

import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- UI ----------------
st.set_page_config(page_title="AI Fire Detection", page_icon="🔥", layout="wide")

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
}
h1,h2,h3{text-align:center;color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
model = YOLO("yolov9t.pt")

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
    <p style="text-align:center;">Detect fire from images and videos</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- IMAGE ----------------
elif selected == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        frame = np.array(image)

        results = model.predict(frame)

        result_img = results[0].plot()

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original")
        col2.image(result_img, caption="Detection")

# ---------------- VIDEO ----------------
elif selected == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video:
        st.video(uploaded_video)

        st.info("Processing video... please wait ⏳")

        # Save video temporarily
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Run YOLO directly on video (NO cv2 loop)
        results = model.predict(source="temp.mp4", save=True)

        st.success("✅ Processing complete!")

        st.write("Check detection output video in results folder")

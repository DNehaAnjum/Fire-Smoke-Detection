import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import glob

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
    Detect fire from images and videos using YOLO.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://media0.giphy.com/media/lMUGMp2lImgGA/giphy.gif",
        use_container_width=True
    )

# ---------------- IMAGE ----------------
elif selected == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        model = YOLO("yolov9t.pt")   # load here (safe)

        image = Image.open(uploaded)
        frame = np.array(image)

        results = model.predict(frame)

        result_img = results[0].plot()

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original")
        col2.image(result_img, caption="Detection")

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.model.names[cls]

            if "fire" in name.lower() or "smoke" in name.lower():
                st.error(f"🔥 Fire Detected! Confidence: {round(conf,2)}")

# ---------------- VIDEO ----------------
elif selected == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video:

        st.video(uploaded_video)
        st.info("Processing video... ⏳")

        # Save input
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())

        model = YOLO("yolov9t.pt")

        model.predict(source="temp.mp4", save=True)

        # Get output
        folders = glob.glob("runs/detect/*")
        latest_folder = max(folders, key=os.path.getctime)

        output_videos = glob.glob(os.path.join(latest_folder, "*.mp4"))

        if output_videos:
            st.success("✅ Detection Complete!")
            st.video(output_videos[0])
        else:
            st.error("❌ Output video not found")

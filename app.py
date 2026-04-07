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
backdrop-filter: blur(10px);
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
    ["Home", "Image"],
    icons=["house", "image"],
    orientation="horizontal"
)

# ---------------- HOME ----------------
if selected == "Home":

    st.markdown("""
    <div class="glass-card">
    <h1>🔥 AI Powered Fire & Smoke Detection</h1>
    <p style="text-align:center;font-size:18px;">
    Real-time monitoring using YOLO Deep Learning.
    Detect fire from images.
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
        image = Image.open(uploaded)
        frame = np.array(image)

        results = model.predict(frame, conf=0.5)

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

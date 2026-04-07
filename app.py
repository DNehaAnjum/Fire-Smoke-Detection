import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from playsound import playsound
import threading
import altair as alt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ---------------- ADVANCED GLOBAL UI ----------------
st.set_page_config(
    page_title="AI Fire Detection",
    page_icon="🔥",
    layout="wide"
)

st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

/* Navbar glass effect */
[data-testid="stHorizontalBlock"]{
background: rgba(255,255,255,0.05);
backdrop-filter: blur(12px);
border-radius:15px;
padding:10px;
margin-bottom:20px;
}

/* Glass cards */
.glass-card{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
border-radius:15px;
padding:20px;
border:1px solid rgba(255,255,255,0.15);
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button{
background: linear-gradient(135deg,#ff7e5f,#feb47b);
border:none;
border-radius:10px;
color:white;
font-weight:600;
padding:10px 25px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
box-shadow:0 0 20px rgba(255,126,95,0.6);
}

/* Alert animation */
@keyframes pulse{
0%{box-shadow:0 0 0 0 rgba(255,0,0,0.7);}
70%{box-shadow:0 0 0 20px rgba(255,0,0,0);}
100%{box-shadow:0 0 0 0 rgba(255,0,0,0);}
}

.high{
animation:pulse 2s infinite;
}

/* Image styling */
img{
border-radius:15px;
}

/* Charts */
.vega-embed{
background: rgba(255,255,255,0.05);
padding:15px;
border-radius:15px;
}

/* DataFrame */
[data-testid="stDataFrame"]{
background: rgba(255,255,255,0.05);
border-radius:15px;
}

/* Titles */
h1,h2,h3{
text-align:center;
color:white;
font-weight:700;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

model_path = "fire-models/fire_l.pt"
model = load_model(model_path)


# ---------------- SEVERITY ----------------
def get_severity(conf):
    if conf < 0.80:
        return "Low"
    elif conf < 0.90:
        return "Medium"
    else:
        return "High"


# ---------------- SOUND ----------------
def play_alert_sound():
    try:
        threading.Thread(
            target=playsound,
            args=("audio.mp3",),
            daemon=True
        ).start()
    except:
        pass


# ---------------- LOGGING ----------------
def log_detection(class_name, severity, confidence, bbox, source):

    Path("logs").mkdir(exist_ok=True)
    log_file = "logs/detection_logs.csv"

    x1, y1, x2, y2 = bbox

    data = {
        "Timestamp": datetime.datetime.now(),
        "Detected_Object": class_name,
        "Severity": severity,
        "Confidence": round(confidence, 3),
        "BBox_X1": x1,
        "BBox_Y1": y1,
        "BBox_X2": x2,
        "BBox_Y2": y2,
        "Source": source
    }

    df = pd.DataFrame([data])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)


# ---------------- DETECTION ----------------
def detect_frame(frame, source_type):

    results = model.predict(frame, conf=0.5, iou=0.3)

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    if results[0].boxes is not None:

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy().astype(int)

            if "fire" in class_name.lower() or "smoke" in class_name.lower():

                severity = get_severity(conf)

                st.markdown(
                    f"""
                    <div class='glass-card high'>
                    🚨 <b>{severity} Severity Fire Detected</b><br>
                    Confidence : {round(conf,3)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                play_alert_sound()
                log_detection(class_name, severity, conf, bbox, source_type)

    return annotated


# ---------------- MENU ----------------
selected = option_menu(
    "🔥 AI Fire Detection Dashboard",
    ["Home", "Image", "Video", "Live", "Logs"],
    icons=["house", "image", "camera-video", "camera", "file-earmark-text"],
    orientation="horizontal"
)


# ---------------- HOME ----------------
if selected == "Home":

    st.markdown("""
    <div class="glass-card">
    <h1>🔥 AI Powered Fire & Smoke Detection</h1>
    <p style="text-align:center;font-size:18px;">
    Real-time monitoring using YOLO Deep Learning.
    Detect fire from images, videos, or live camera.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://media0.giphy.com/media/lMUGMp2lImgGA/giphy.gif",
        use_container_width=True
    )


# ---------------- IMAGE ----------------
elif selected == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:

        image = Image.open(uploaded)
        frame = np.array(image)

        result = detect_frame(frame, "Image")

        col1, col2 = st.columns(2)

        col1.image(image, caption="Original")
        col2.image(result, caption="Detection")


# ---------------- VIDEO ----------------
elif selected == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_video:

        temp_path = "temp_video.mp4"

        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_path)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            result = detect_frame(frame, "Video")

            stframe.image(result)

        cap.release()


# ---------------- LIVE ----------------
elif selected == "Live":

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    col1, col2 = st.columns(2)

    start = col1.button("Start Camera")
    stop = col2.button("Stop Camera")

    if start:
        st.session_state.run_camera = True

    if stop:
        st.session_state.run_camera = False

    if st.session_state.run_camera:

        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        while st.session_state.run_camera:

            ret, frame = cap.read()

            if not ret:
                break

            result = detect_frame(frame, "Live")

            stframe.image(result)

        cap.release()


# ---------------- LOGS ----------------
elif selected == "Logs":

    df = pd.read_csv("logs/detection_logs.csv")

    st.dataframe(df)

    chart = alt.Chart(df).mark_bar().encode(
        x="Severity",
        y="count()",
        color="Severity"
    )

    st.altair_chart(chart, use_container_width=True)
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("🔥 Fire & Smoke Detection")

# Load model
model = YOLO("yolov9t.pt")

# Upload image
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    frame = np.array(image)

    results = model.predict(frame, conf=0.5)

    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Original")
    st.image(result_img, caption="Detection")

    # Show alert
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.model.names[cls]

        if "fire" in name.lower() or "smoke" in name.lower():
            st.error(f"🔥 Fire Detected! Confidence: {round(conf,2)}")

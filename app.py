import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Title
st.title("🔥 Fire & Smoke Detection System")

# Load model
model = YOLO("yolov9t.pt")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    frame = np.array(image)

    # Run detection
    results = model.predict(frame, conf=0.5)

    # Get result image
    result_img = results[0].plot()

    # Show images
    st.image(image, caption="Original Image")
    st.image(result_img, caption="Detection Result")

    # Alerts
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.model.names[cls]

        if "fire" in name.lower() or "smoke" in name.lower():
            st.error(f"🔥 Fire/Smoke Detected! Confidence: {round(conf,2)}")

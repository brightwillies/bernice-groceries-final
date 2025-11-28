import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# -----------------------------------------------------
# Load YOLO Model
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Upload it to your repo.")
        return None
    return YOLO(model_path)

model = load_model()

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("YOLOv11 Grocery Item Detector")
st.write("Detects **cheerios**, **soup**, and **candle**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show user image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    st.write("Running inference...")

    # Run prediction
    results = model.predict(temp_image_path)

    # Extract prediction image
    pred_img = results[0].plot()  # numpy array (BGR)
    pred_img_rgb = pred_img[:, :, ::-1]  # convert BGR → RGB

    st.image(pred_img_rgb, caption="Prediction", use_column_width=True)

    # Show detection details
    st.subheader("Detection Details")

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        st.write(
            f"- **Class:** {model.names[cls_id]}  |  **Confidence:** {conf:.2f} | **Box:** {xyxy}"
        )

st.markdown("---")
st.caption("YOLOv11 Streamlit App – by Bernice Nhyira Eghan")

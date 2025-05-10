import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

# Set page config first, right after imports
st.set_page_config(page_title="Crack Segmentation", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Crack Segmentation using YOLOv8n-seg")

st.markdown("Upload an Image and the model will segment visible cracks.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Debugging: Display image mode and size
    st.write(f"Image Mode: {original_image.mode}, Image Size: {original_image.size}")

    # Resize for faster inference (YOLOv8 default is 640x640)
    resized_image = original_image.resize((640, 640))

    with st.spinner("Segmenting..."):
        # Make predictions
        results = model.predict(resized_image, conf=0.25)  # lower conf threshold if needed
        
        # Debugging: Show prediction results
        st.write(f"Prediction Results: {results}")

        # Display the result image
        result_img = results[0].plot()  # returns annotated image as numpy array
        st.image(result_img, caption="Segmented (Resized)", use_column_width=True)

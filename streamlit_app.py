import streamlit as st
from PIL import Image
import torch # PyTorch is imported but not explicitly used, YOLO handles its PyTorch backend.
from ultralytics import YOLO
import numpy as np
import asyncio

# Ensure event loop is created - KEEPING THIS AS PER YOUR REQUEST
asyncio.set_event_loop(asyncio.new_event_loop())

# Set page config first, right after imports
st.set_page_config(page_title="Crack Segmentation", layout="centered")

# Load model
@st.cache_resource
def load_model():
    # Ensure your model file "best.pt" is at the root of your app directory,
    # or update the path if it's located elsewhere (e.g., "src/best.pt").
    return YOLO("best.pt")

model = load_model()

st.title("Crack Segmentation using YOLOv8n-seg")
st.markdown("Upload an Image and the model will segment visible cracks.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_container_width=True) # MODIFIED HERE

    # Debugging: Display image mode and size - Commented out to only show original and segmented images
    # st.write(f"Image Mode: {original_image.mode}, Image Size: {original_image.size}")

    # Resize for faster inference (YOLOv8 default is 640x640)
    resized_image = original_image.resize((640, 640))

    with st.spinner("Segmenting..."):
        # Make predictions
        results = model.predict(resized_image, conf=0.25)  # lower conf threshold if needed
        
        # Debugging: Show prediction results - Commented out to only show original and segmented images
        # st.write(f"Prediction Results: {results}")

        # Display the result image
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            result_img_np = results[0].plot()  # returns annotated image as numpy array (BGR)
            result_img_rgb = result_img_np[:, :, ::-1] # Convert BGR to RGB for correct display
            st.image(result_img_rgb, caption="Segmented Image", use_container_width=True) # MODIFIED HERE
        elif results and (results[0].masks is None or len(results[0].masks) == 0):
            st.warning("Segmentation complete, but no cracks were detected or segmented in the image.")
            # If you wanted to show the original plotted image even without masks:
            # result_img_np = results[0].plot()
            # result_img_rgb = result_img_np[:, :, ::-1]
            # st.image(result_img_rgb, caption="Processed Image (No Masks)", use_container_width=True)
        else:
            st.error("Prediction failed or returned no results. Please try another image.")
else:
    st.info("Please upload an image to see the segmentation.")

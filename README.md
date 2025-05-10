Here’s the updated README with your Hugging Face link included:

---

# Crack Segmentation with YOLOv8 and Streamlit

## Overview

This project implements crack segmentation using the YOLOv8 (You Only Look Once) model for detecting cracks in images. The model is deployed with a Streamlit web app where users can upload images, and the model will segment visible cracks in the images. It uses a trained YOLOv8 model and provides a user-friendly interface for crack detection in wall images.

You can try the deployed model on [Hugging Face - Crack Segmentation](https://huggingface.co/spaces/Speccco/Crack-Segmentation).
You can try the deployed model on [Streamlit - Crack Segmentation](https://crack-segmentation.streamlit.app/)
## Features

* **YOLOv8 Model**: Uses the state-of-the-art YOLOv8n-segmentation model for detecting cracks in images.
* **Streamlit Interface**: Simple, intuitive web interface for uploading images and visualizing segmented results.
* **Real-time Image Segmentation**: Upload an image and get real-time segmentation results with the detected cracks.

## Requirements

To run this app locally, you need to install the following dependencies. The easiest way is to use the provided `requirements.txt` file.

* Python 3.9+
* Streamlit
* YOLOv8 (Ultralytics)
* Pillow
* PyTorch

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crack-segmentation.git
cd crack-segmentation
```

### 2. Set up the virtual environment

You can use `venv` or `conda` to create a virtual environment for this project.

#### Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using `conda`:

```bash
conda create -n crack-segmentation python=3.9
conda activate crack-segmentation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Model File

Ensure that the `best.pt` model file is located inside the `src` folder. You can upload or train the YOLOv8 model on your dataset and place the trained weights file in the `src` directory.

### 5. Run the app

To start the Streamlit app, run the following command:

```bash
streamlit run src/streamlit_app.py
```

Visit the provided local URL (usually `http://localhost:8501`) in your browser to interact with the app.

## How It Works

* **Image Upload**: The user uploads an image (e.g., of a wall with cracks).
* **Model Inference**: The image is resized to 640x640 for faster inference and passed through the trained YOLOv8 model.
* **Segmentation**: The model detects cracks in the image and produces an annotated output, highlighting the cracks.
* **Result**: The segmented image is displayed alongside the original image.

## Example

1. Upload an image of a wall with cracks.
2. The model segments the cracks and displays the results in real time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) for providing the YOLOv8 model.
* [Streamlit](https://streamlit.io/) for easy-to-use web app development.
* [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/) for image handling.

---

Now your README includes the Hugging Face link, making it easier for others to access the deployed model directly!

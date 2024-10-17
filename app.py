import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from PIL import Image, ImageOps
import numpy as np
import os

# Define your local model path
local_model_path = r"C:\Users\Guru\OneDrive\Desktop\recycling-app\my_model.h5"

# Initialize model variable
model = None

# Check if local model exists
if os.path.exists(local_model_path):
    model_path = local_model_path
    # Load the model
    try:
        model = load_model(model_path, custom_objects={'BatchNormalization': BatchNormalization}, compile=False)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error("Model file not found.")

# Function to process and predict using the model
def import_and_predict(image_data, model):
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None  # Prevent further processing if the model isn't loaded

    size = (512, 512)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]  # Expand dimensions for model input
    prediction = model.predict(img_reshape)
    return prediction

# Streamlit web app
st.title("Recyclable Materials Classification")
st.write("Upload an image to classify it as Recyclable, Non-Recyclable, or Contaminated!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    if model is not None:  # Ensure model is loaded before prediction
        prediction = import_and_predict(image, model)

        # Assuming the model outputs three classes: 0 = Non-Recyclable, 1 = Recyclable, 2 = Contaminated
        class_labels = ['Non-Recyclable', 'Recyclable', 'Contaminated']
        predicted_class = np.argmax(prediction)

        st.write(f"This item is **{class_labels[predicted_class]}**.")
    else:
        st.error("Model is not available for predictions.")

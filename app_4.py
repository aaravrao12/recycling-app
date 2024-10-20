import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Define your local model path using a relative path
local_model_path = os.path.join('models', 'my_simplified_model.h5')  # Relative path

print("Checking model path:", local_model_path)
print("Exists:", os.path.exists(local_model_path))

# Check if local model exists
if os.path.exists(local_model_path):
    model_path = local_model_path
    print("Model file found.")
else:
    print("Model file not found.")
    st.error("Model file not found. Please check the path.")
    model_path = None  # Set model_path to None if not found

# Load the model without custom handling for BatchNormalization
if model_path is not None:  # Ensure the path is defined before loading
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")  # Print success for debugging
    except Exception as e:
        print(f"Error loading model: {e}")  # Print error for debugging
        st.error("Error loading model. Please check the console for more details.")
        model = None  # Set model to None if loading fails
else:
    model = None  # If path is None, set model to None

# Function to process and predict using the model
def import_and_predict(image_data, model):
    size = (512, 512)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
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
    
    # Ensure the model is loaded before predicting
    if model is not None:  # Check if model is defined
        prediction = import_and_predict(image, model)

        # Assuming the model outputs three classes: 0 = Non-Recyclable, 1 = Recyclable, 2 = Contaminated
        class_labels = ['Non-Recyclable', 'Recyclable', 'Contaminated']
        predicted_class = np.argmax(prediction)

        st.write(f"This item is **{class_labels[predicted_class]}**.")
    else:
        st.error("Model is not loaded. Cannot make predictions.")

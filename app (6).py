import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from PIL import Image, ImageOps
import numpy as np
import os

# Define your local model path
local_model_path = '/content/drive/MyDrive/my_simplified_model.keras'  # Make sure this is the correct path

# Check if local model exists
if os.path.exists(local_model_path):
    model_path = local_model_path
    print("Model file found.")
else:
    print("Model file not found.")

# Custom function to manually handle BatchNormalization layer
def custom_bn_layer():
    return BatchNormalization()

# Load the model with custom handling for BatchNormalization
try:
    model = load_model(model_path, custom_objects={'BatchNormalization': custom_bn_layer}, compile=False)
    print("Model loaded successfully.")  # Print success for debugging
except Exception as e:
    print(f"Error loading model: {e}")  # Print error for debugging

# Function to process and predict using the model
def import_and_predict(image_data, model):
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
    prediction = import_and_predict(image, model)

    # Assuming the model outputs three classes: 0 = Non-Recyclable, 1 = Recyclable, 2 = Contaminated
    class_labels = ['Non-Recyclable', 'Recyclable', 'Contaminated']
    predicted_class = np.argmax(prediction)

    st.write(f"This item is **{class_labels[predicted_class]}**.")

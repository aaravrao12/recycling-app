import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from PIL import Image, ImageOps
import numpy as np
import gdown  # Import gdown to download the model

# Download the model file from Google Drive
url = 'https://drive.google.com/uc?id=1LytKZy-ARIyPjqmqSVWbz1RTa6udwE4N'  # Your file ID
output = 'my_new_model_updated.h5'

# Download the model if it doesn't exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the model with custom objects
try:
    model = load_model(output, custom_objects={'BatchNormalization': BatchNormalization}, compile=False)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

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

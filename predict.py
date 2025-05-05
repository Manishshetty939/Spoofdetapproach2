import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model 
model = load_model("models/iris_spoof_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  # Normalize image 
    return img_array

# Function to predic image is real or fake
def predict_iris(img_path):
    img_array = preprocess_image(img_path)  
    prediction = model.predict(img_array)  
    
    # Output pre
    if prediction < 0.5:
        return "Fake Iris"
    else:
        return "Real Iris"


st.title("Iris Spoof Detection App")
st.write("Upload an image of the iris to detect if it's real or fake.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    img_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    img = image.load_img(img_path, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    
    result = predict_iris(img_path)
    
    
    st.write(f"Prediction: **{result}**")
st
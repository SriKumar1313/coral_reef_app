import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to load the SVM model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(64, 64)):
    img = ImageOps.fit(image, img_size, Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img_array = np.array(img).flatten()
    return img_array

# Function to make predictions
def predict(image, model):
    # Preprocess image
    img_array = preprocess_image(image)
    # Scale and apply PCA
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    img_scaled = scaler.fit_transform(img_array.reshape(1, -1))
    img_pca = pca.fit_transform(img_scaled)
    # Predict using the model
    prediction = model.predict(img_pca)
    return prediction

def main():
    st.title('Coral Image Classifier')
    st.text('Upload a coral image to classify it as healthy or bleached')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Check if the model file exists
        model_path = 'svm_model_pca.pkl'
        if not os.path.exists(model_path):
            st.write("Error: SVM model file not found.")
            return

        # Load SVM model
        svm_model = load_model(model_path)

        # Make prediction
        prediction = predict(image, svm_model)

        # Display prediction
        categories = ['healthy', 'bleached']
        st.write(f"Prediction: {categories[prediction[0]]}")

if __name__ == '__main__':
    main()

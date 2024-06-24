import streamlit as st
import os
import numpy as np
import pickle
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(64, 64)):
    img = ImageOps.fit(image, img_size, Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img_array = np.array(img).flatten()
    return img_array

# Function to make predictions
def predict(image, model, scaler, pca):
    # Preprocess image
    img_array = preprocess_image(image)
    # Scale and apply PCA using the provided scaler and pca
    img_scaled = scaler.transform(img_array.reshape(1, -1))
    img_pca = pca.transform(img_scaled)
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

        # Load SVM model and preprocessing components
        with open(model_path, 'rb') as file:
            svm_model, scaler, pca = pickle.load(file)

        # Make prediction
        prediction = predict(image, svm_model, scaler, pca)

        # Display prediction
        categories = ['healthy', 'bleached']
        st.write(f"Prediction: {categories[prediction[0]]}")

if __name__ == '__main__':
    main()

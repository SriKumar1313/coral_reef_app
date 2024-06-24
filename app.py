import streamlit as st
import numpy as np
from PIL import Image as PILImage
import pickle

# Load SVM model
svm_model_filename = 'svm_model.pkl'
with open(svm_model_filename, 'rb') as file:
    svm_model = pickle.load(file)

# Define categories
categories = ['healthy_corals', 'bleached_corals']

def predict_image(image):
    # Preprocess the image (resize, convert to grayscale, etc. if needed)
    # Example preprocessing (adjust as per your preprocessing steps):
    img = image.resize((64, 64))  # Resize to your model's input size
    img_array = np.array(img.convert('L')).flatten()  # Convert to grayscale and flatten

    # Make prediction
    prediction = svm_model.predict([img_array])[0]
    predicted_category = categories[prediction]

    return predicted_category

def main():
    st.title('Coral Image Classifier')
    st.text('Upload a coral image for classification')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the image
        image = PILImage.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict
        if st.button('Predict'):
            prediction = predict_image(image)
            st.success(f'The image is classified as {prediction}.')

if __name__ == '__main__':
    main()

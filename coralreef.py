import streamlit as st
import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Define image directories
healthy_path = "data/healthy_corals"
bleached_path = "data/bleached_corals"
categories = ['healthy_corals', 'bleached_corals']
IMG_SIZE = (64, 64)  # Define image size

def load_images():
    unique_images = []
    unique_labels = []
    image_paths = {}
    duplicates = []

    for category in categories:
        path = os.path.join("data", category)  # Ensure paths are correctly set
        class_num = categories.index(category)

        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue

        for img_name in tqdm(os.listdir(path), desc=f"Loading {category} images"):
            img_path = os.path.join(path, img_name)
            try:
                img = Image.open(img_path).resize(IMG_SIZE)  # Open and resize image
                img_array = np.array(img)

                is_duplicate = False
                for unique_img in unique_images:
                    if np.array_equal(img_array, unique_img):
                        duplicates.append((img_path, image_paths[tuple(unique_img.flatten())]))
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_images.append(img_array)
                    unique_labels.append(class_num)
                    image_paths[tuple(img_array.flatten())] = img_path
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    images = np.array(unique_images)
    labels = np.array(unique_labels)

    print(f"Loaded {len(images)} unique images")
    print(f"Found {len(duplicates)} duplicate images")

    return images, labels, duplicates

def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []

    for img, label in tqdm(zip(images, labels), desc="Augmenting images", total=len(images)):
        augmented_images.append(img)
        augmented_images.append(np.array(ImageOps.mirror(Image.fromarray(img))))
        augmented_images.append(np.array(Image.fromarray(img).rotate(90)))
        augmented_labels.extend([label, label, label])

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    print(f"Augmented data size: {len(augmented_images)} images")

    return augmented_images, augmented_labels

def convert_to_grayscale(images):
    grayscale_images = np.array([np.array(Image.fromarray(img).convert('L')) for img in tqdm(images, desc="Converting to grayscale")])
    print("Converted images to grayscale")
    return grayscale_images

def adjust_contrast_and_brightness(images):
    adjusted_images = []
    for img in tqdm(images, desc="Adjusting contrast and brightness"):
        img_pil = Image.fromarray(img)
        contrast_enhancer = ImageEnhance.Contrast(img_pil)
        img_contrast = contrast_enhancer.enhance(2)
        brightness_enhancer = ImageEnhance.Brightness(img_contrast)
        img_bright = brightness_enhancer.enhance(1.5)
        adjusted_images.append(np.array(img_bright))

    adjusted_images = np.array(adjusted_images)

    print("Adjusted contrast and brightness of images")

    return adjusted_images

def flatten_images(images):
    flattened_images = []
    for img in tqdm(images, desc="Flattening images"):
        flattened_img = np.array(img).flatten()
        flattened_images.append(flattened_img)

    flattened_images = np.array(flattened_images)

    return flattened_images

def train_model(X_train, y_train):
    # Example using Gradient Boosting Classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier())
    ])

    param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, grid_search

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, conf_matrix

def display_results(model, accuracy, precision, recall, f1, conf_matrix):
    st.subheader('Model Evaluation Metrics')
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Confusion Matrix:\n{conf_matrix}")

def main():
    st.title('Coral Image Classification and Evaluation')
    st.write('This app loads coral images, processes them, trains machine learning models, and evaluates their performance.')

    # Load and preprocess images
    images, labels, duplicates = load_images()

    # Augment images
    augmented_images, augmented_labels = augment_images(images, labels)

    # Convert to grayscale
    grayscale_images = convert_to_grayscale(augmented_images)

    # Adjust contrast and brightness
    adjusted_images = adjust_contrast_and_brightness(grayscale_images)

    # Flatten images
    flattened_images = flatten_images(adjusted_images)

    # Debugging: Check shapes and lengths of the data before splitting
    st.write(f"Flattened images shape: {flattened_images.shape}")
    st.write(f"Augmented labels length: {len(augmented_labels)}")

    # Train-test split
    if len(flattened_images) != len(augmented_labels):
        st.error("The number of images does not match the number of labels!")
        return

    if len(flattened_images) < 5 or len(augmented_labels) < 5:
        st.error("Not enough data to perform train-test split!")
        return

    X_train, X_test, y_train, y_test = train_test_split(flattened_images, augmented_labels, test_size=0.2, random_state=42)

    # Train model
    model, grid_search = train_model(X_train, y_train)

    # Evaluate model
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)

    # Display results
    display_results(model, accuracy, precision, recall, f1, conf_matrix)

if __name__ == '__main__':
    main()

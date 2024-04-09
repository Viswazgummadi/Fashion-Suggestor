import cv2
import numpy as np
import joblib

# Load the trained Random Forest Classifier
model_path = 'gender_detection_model_knn_k27.model'  # Replace with the path to your .model file
clf = joblib.load(model_path)

# Function to detect gender from a given photo file and return the result
def detect_gender_from_photo(photo_path):
    # Read the image
    image = cv2.imread(photo_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Resize the image to match the training data dimensions
    resized_image = cv2.resize(gray_image, (96, 96))

    # Flatten the resized image for the classifier
    flat_image = resized_image.reshape((1, -1))

    # Predict gender
    prediction = clf.predict(flat_image)[0]

    # Map prediction to labels
    gender_label = 1 if prediction == 1 else 0

    return gender_label


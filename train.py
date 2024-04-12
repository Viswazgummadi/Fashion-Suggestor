import os
import glob
import random
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# initial parameters
img_dims = (96, 96)

data = []
labels = []

# load image files from the dataset
folder_name = 'gender_dataset_face'  # Replace with your actual folder name
image_files = [f for f in glob.glob(os.path.join(folder_name, "/*"), recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labeling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data.append(gray.flatten())

    label = img.split(os.path.sep)[-2]  # C:\Files\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append(label)

# pre-processing
data = np.array(data)
labels = np.array(labels)

# split dataset for training and validation
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build and train a k-Nearest Neighbors (KNN) Classifier
accuracy_values = []
k_values = list(range(1, 60))

for k in k_values:
    clf_knn_temp = KNeighborsClassifier(n_neighbors=k)
    clf_knn_temp.fit(trainX, trainY)
    predictions_knn_temp = clf_knn_temp.predict(testX)
    accuracy_knn_temp = accuracy_score(testY, predictions_knn_temp)
    accuracy_values.append(accuracy_knn_temp)

# Choose the optimal k value
optimal_k = k_values[np.argmax(accuracy_values)]
print(f"Optimal k value: {optimal_k}")

# Build and train the final KNN model with the optimal k value
clf_knn = KNeighborsClassifier(n_neighbors=optimal_k)
clf_knn.fit(trainX, trainY)

# Evaluate the KNN model with the optimal k value
predictions_knn = clf_knn.predict(testX)
accuracy_knn = accuracy_score(testY, predictions_knn)
report_knn = classification_report(testY, predictions_knn)

# Save the KNN model to disk with .model extension using pickle
with open(f'gender_detection_model_knn_k{optimal_k}.model', 'wb') as model_file_knn:
    pickle.dump(clf_knn, model_file_knn)

# Print information about the KNN model
print(f"\nKNN Model (Optimal k={optimal_k}):")
print(f"Accuracy: {accuracy_knn}")
print("Classification Report:\n", report_knn)

# Plot the accuracy values for different k
plt.plot(k_values, accuracy_values, marker='o')
plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.show()
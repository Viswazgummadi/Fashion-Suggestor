import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import csv

def detect_shirt_colors(image_path, num_colors=5):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    # Reshape the image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Convert to float32 for KMeans
    pixels = np.float32(pixels)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers and counts
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    cluster_counts = Counter(cluster_labels)
    
    # Calculate total number of pixels
    total_pixels = height * width
    
    # Calculate percentage of each color
    color_percentages = [(count / total_pixels) * 100 for count in cluster_counts.values()]
    
    # Sort colors by percentage
    colors_sorted = sorted([(percentage, color) for percentage, color in zip(color_percentages, cluster_centers)], reverse=True)
    
    return colors_sorted


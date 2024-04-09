import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to adjust pixel values


def adjust_pixel_values(image):
    # Thresholding to adjust pixel values
    adjusted_img = np.where(image < 50, 0, np.where(image > 200, 255, image))
    return adjusted_img


# Read the input image
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 190)

# Apply the adjustment function
enhanced = adjust_pixel_values(edges)
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# Display the enhanced image with RGB values
plt.figure(figsize=(8, 6))
plt.imshow(enhanced, cmap='gray')
plt.title('Enhanced Image')
plt.colorbar(label='RGB Value')
plt.axis('off')
plt.show()

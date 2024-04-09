import os
import cv2
import numpy as np
from scipy.signal import convolve2d


def rgb_to_grayscale(image_array):
    luminance = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    grayscale_array = luminance.astype(np.uint8)
    return grayscale_array


def apply_gaussian_blur(image):
    # Define the Gaussian blur kernel
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    # Apply convolution using NumPy's convolve function
    blurred_image = np.zeros_like(image, dtype=float)
    blurred_image = np.convolve(
        image.flatten(), kernel.flatten(), mode='same').reshape(image.shape)

    return blurred_image.astype(np.uint8)


def gaussian_kernel(kernel_size, sigma):
    kernel_range = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    kernel = np.exp(-0.5 * (kernel_range ** 2) / sigma ** 2)
    kernel = kernel / np.sum(kernel)
    return kernel


def sobelx(image):
    # Sobel X operator
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Apply convolution using NumPy's convolve function
    gradient_x = convolve2d(image, kernel_x, mode='same', boundary='wrap')

    sobel_x_image = np.zeros_like(image)
    sobel_x_image = np.convolve(
        image.flatten(), kernel_x.flatten(), mode='same').reshape(image.shape)

    return gradient_x.astype(np.uint8), sobel_x_image.astype(np.uint8)


def sobely(image):
    # Sobel Y operator
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply convolution using NumPy's convolve function
    gradient_y = convolve2d(image, kernel_y, mode='same', boundary='wrap')

    return gradient_y.astype(np.uint8)


def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = convolve2d(image, kernel_x, mode='same', boundary='wrap')
    gradient_y = convolve2d(image, kernel_y, mode='same', boundary='wrap')

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction


def non_maximum_suppression(magnitude, direction):
    angle = np.degrees(direction) % 180
    suppressed = np.zeros_like(magnitude)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbor_pixels = [magnitude[i, j+1], magnitude[i, j-1]]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbor_pixels = [magnitude[i+1, j-1], magnitude[i-1, j+1]]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbor_pixels = [magnitude[i+1, j], magnitude[i-1, j]]
            else:
                neighbor_pixels = [magnitude[i+1, j+1], magnitude[i-1, j-1]]

            if magnitude[i, j] >= max(neighbor_pixels):
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def hysteresis_thresholding(image, low_threshold_ratio=0, high_threshold_ratio=0.039):
    high_threshold = np.max(image) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    strong_edges_row, strong_edges_col = np.where(image >= high_threshold)
    weak_edges_row, weak_edges_col = np.where(
        (image >= low_threshold) & (image < high_threshold))

    output_image = np.zeros_like(image, dtype=np.uint8)
    output_image[strong_edges_row, strong_edges_col] = 255
    output_image[weak_edges_row, weak_edges_col] = 20  # Weak edges
    return output_image


def adjust_pixel_values(image):
    # Thresholding to adjust pixel values
    adjusted_img = np.where(image < 50, 0, np.where(image > 200, 255, image))

    return adjusted_img


def extract_bordered_portion(image, edges, gray):
    height, width = edges.shape[:2]
    top, bottom, left, right = height, 0, width, 0

    # Iterate over the top border
    for y in range(height):
        if edges[y, :].any():  # Check if any edge pixel is present in the row
            top = y
            break

    # Iterate over the bottom border
    for y in range(height - 1, -1, -1):
        if edges[y, :].any():
            bottom = y
            break

    # Iterate over the left border
    for x in range(width):
        if edges[:, x].any():  # Check if any edge pixel is present in the column
            left = x
            break

    # Iterate over the right border
    for x in range(width - 1, -1, -1):
        if edges[:, x].any():
            right = x
            break

    # Cut the bordered portion using the detected borders from the original image
    bordered_portion = image[top:bottom+1, left:right+1]

    return bordered_portion

def canny(gray_image,t1,t2):
    blurred_image = apply_gaussian_blur(gray_image)
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel = gaussian_kernel(kernel_size, sigma)
    gaussian_mask = np.outer(kernel, kernel.transpose())
    # Apply Gaussian blur using kernel convolution
    blurred_single_channel = convolve2d(
        gray_image, gaussian_mask, mode='same', boundary='wrap')
    blurred = blurred_single_channel.astype(np.uint8)

    grad_mag, grad_ang = sobel_filter(blurred)
    thresholdl = t1
    thresholdh = t2
    
    border_mask = (thresholdh>grad_mag > thresholdl).astype(np.uint8)
    
    hue = (grad_ang + np.pi) / (2 * np.pi)
    hue[hue > 1] -= 1
    hue_scaled = np.uint8(hue * 255)
    hsv_image = np.zeros((grad_mag.shape[0], grad_mag.shape[1], 3), dtype=np.uint8)
    hsv_image[..., 0] = hue_scaled
    hsv_image[..., 1] = 255
    hsv_image[..., 2] = 255 * border_mask
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    # ### Thresholded suppression
    magnitude_suppressed = non_maximum_suppression(grad_mag, grad_ang)
    # ### Connecting edges
    edges = hysteresis_thresholding(magnitude_suppressed)
    # ### enhancing
    enhanced = adjust_pixel_values(edges)
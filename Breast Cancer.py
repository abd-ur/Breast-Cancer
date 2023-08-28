import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
image=cv2.imread(r"/Users/IBR/Downloads/WhatsApp Image 2023-07-12 at 3.41.48 PM.jpeg")

def color_threshold_segmentation(image, lower_bound, upper_bound):

    # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a mask by thresholding the image based on the color range
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Apply the mask to extract the segmented region from the original image
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return segmented_image

lower_bound = (25, 100, 100)  # Example: lower bound for blue color in HSV format
upper_bound = (20, 255, 255)  # Example: upper bound for blue color in HSV format
segmented_image = color_threshold_segmentation(image, lower_bound, upper_bound)

plt.imshow(segmented_image)

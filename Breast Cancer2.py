import cv2 
import matplotlib.pyplot as plt
data=cv2.imread(r"/Users/IBR/Downloads/WhatsApp Image 2023-07-12 at 3.41.48 PM.jpeg")
#plt.imshow(data)

import numpy as n
kernel=n.ones(3,n.uint8)
erosion=cv2.erode(data,kernel, iterations=3)
plt.imshow(erosion)

gry=cv2.cvtColor(data, cv2.COLOR_BGR2HSV)




#plt.imshow(erosion)

from sklearn.cluster import KMeans

def image_segmentation(image, num_clusters):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform clustering using K-means algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape the labels to match the image shape
    seg=labels.reshape(image.shape[:2])
    
    return seg

seg_img=image_segmentation(gry,6)

# Display the original and segmented images
plt.imshow(seg_img)


import numpy as np
img=cv2.imread(r"/Users/IBR/Desktop/segment.png")
# Convert the image from BGR to RGB (OpenCV reads images in BGR format)
img_float32 = np.float32(img)
lab_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
# Convert the target color to RGB format
target_color_rgb = np.int32([[255,255,0]])

# Convert the color to HSV format for better color matching
target_color_hsv = cv2.cvtColor(target_color_rgb, cv2.COLOR_RGB2HSV)

# Define a color range around the target color
lower_bound = np.array([target_color_hsv[0][0][0] - 10, 100, 100])
upper_bound = np.array([target_color_hsv[0][0][0] + 10, 255, 255])

# Create a mask for the target color
mask = cv2.inRange(cv2.cvtColor(seg_img, cv2.COLOR_RGB2HSV), lower_bound, upper_bound)

# Count the number of non-zero pixels in the mask (i.e., occurrences of the target color)
num_occurrences = np.count_nonzero(mask)
print(num_occurrences)

# Importing required libraries

import cv2 # pip install opencv-python
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import cvlib as cv # pip install cvlib
from cvlib.object_detection import draw_bbox

# Loading and Viewing the image

img = cv2.imread('C:/Project/Counting Objects from Images/Images/pic.jpg') # use any image of your choice
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10, 10))
plt.axis("off")
plt.imshow(img1)

# Program to create boxes around various objects present inside the image

box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10, 10))
plt.axis("off")
plt.imshow(output)
plt.show()

# To count objects in the image

print("Number of Objects present inside this image are " + str(len(label)))
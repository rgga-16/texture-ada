import cv2
import matplotlib as plt
from torchvision import transforms
import numpy as np


# Show image
def show_image(img):
    
    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img/255).clip(0,1)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

# Conversts tensor to an image
# def tensor_to_image(tensor):


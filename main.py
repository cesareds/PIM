import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def find_threshold(image):
    img_array = np.array(image)
    threshold = np.mean(img_array)
    while True:
        background_pixels = img_array[image <= threshold]
        foreground_pixels = img_array[image > threshold]
        
        media_background = np.mean(background_pixels)
        media_foreground = np.mean(foreground_pixels)
        
        novo_threshold = (media_background + media_foreground) / 2
        
        if abs(novo_threshold - threshold) < 0.1:
            break
        
        threshold = novo_threshold
    return threshold


def solve(image):
    return




path="assets/solda.png"
image=Image.open(path).convert('L')
threshold = find_threshold(image)
b_image_array = np.where(np.array(image)>threshold, 255, 0)
b_image = Image.fromarray(b_image_array.astype(np.uint8))
b_image.show()
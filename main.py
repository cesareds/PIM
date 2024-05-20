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


def dfs(x, y, visitados, image):
    if (visitados[x][y] == False):
        visitados[x][y] = True
        if (image[x][y-1]):
            if (0 < (y - 1) <= h):
                dfs(x, y-1, visitados, image)
        elif (image[x][y+1]):
            if (0 < (y + 1) <= h):
                dfs(x, y+1, visitados, image)
        elif (image[x-1][y]):
            if (0 < (x - 1) <= l):
                dfs(x-1, y, visitados, image)
        elif (image[x+1][y-1]):
            if (0 < (x + 1) <= l):
                dfs(x+1, y, visitados, image)
    return visitados

def solve(image):
    grupos = []
    passa = False
    for i in range(l):
        for j in range(h):
            for g in grupos:
                for item in g:
                    if item == (i, j):
                        passa = True
            if passa:
                passa = False
                continue
            visitar = dfs(i, j, [], image)




path="assets/solda.png"
image=Image.open(path).convert('L')
l,h=image.size
threshold = find_threshold(image)
b_image_array = np.where(np.array(image)>threshold, 255, 0)
b_image = Image.fromarray(b_image_array.astype(np.uint8))

grupos = []


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


def bfs(start, visited, rows, cols, image):
    queue = [start]
    component = []
    visited.add(start)
    while queue:
        x, y = queue.pop(0)
        component.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and image[ny][nx] > 0:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return component


def get_groups(image):
    visited = set()
    components = []
    for j in range(l):
        for i in range(h):
            if image[i][j] > 0 and (i, j) not in visited:
                component = bfs((i, j), visited, l, h, image)
                components.append(component)
    return components


path="assets/solda.png"
image=Image.open(path).convert('L')
l,h=image.size
threshold = find_threshold(image)
b_image_array = np.where(np.array(image)>threshold, 255, 0)
b_image = Image.fromarray(b_image_array.astype(np.uint8))

grupos = get_groups(b_image_array)


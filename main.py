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


def bfs(start, visited, rows, cols, imagem):
    queue = [start]
    component = []
    visited.add(start)
    while queue:
        x, y = queue.pop(0)
        component.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited and imagem[ny][nx] > 0:
                queue.append((nx, ny))
                visited.add((nx, ny))
    return component

def get_groups(imagem):
    visited = set()
    components = []
    imagem = np.array(imagem)
    rows, cols = imagem.shape
    for j in range(cols):
        for i in range(rows):
            if imagem[i][j] > 0 and (i, j) not in visited:
                component = bfs((j, i), visited, rows, cols, imagem)
                components.append(component)
    return components

def find_center_of_mass(image):
    image = np.array(image)
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    a = image.sum()
    x_cms = (xx * image).sum() / a
    y_cms = (yy * image).sum() / a
    return (x_cms, y_cms)

def find_group_closest_to_cmass(groups, cmass):
    min_dist = 100000000
    g = []
    for group in groups:
        for coord in group:
            dist = math.sqrt((coord[0] - cmass[0]) ** 2 + (coord[1] - cmass[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                g = group
                continue
    return g

def get_image_from_group(group):
    ret = [[0 for _ in range(l)] for _ in range(h)]
    for c in group:
        ret[c[1]][c[0]] = 255
    return Image.fromarray((np.array(ret)).astype(np.uint8))


path="assets/solda.png"
image=Image.open(path).convert('L')
l,h=image.size

threshold = find_threshold(image)
b_image_array = np.where(np.array(image)>threshold, 255, 0)
b_image = Image.fromarray(b_image_array.astype(np.uint8))

g_arrays = get_groups(b_image_array)
center_mass = find_center_of_mass(image)
g_closest = find_group_closest_to_cmass(g_arrays, center_mass)
g_image = get_image_from_group(g_closest)

plt.imshow(g_image, cmap='gray')
plt.show()


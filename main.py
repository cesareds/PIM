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


def get_image_groups(arrays):
    # Encontra o maior grupo
    maior = 0
    m_g = []
    for g in arrays:
        if len(g) > maior:
            maior = len(g)
            m_g = g
    ret = [[0 for _ in range(l)] for _ in range(h)]
    for c in m_g:
        ret[c[1]][c[0]] = 255
    return Image.fromarray(np.array(ret))


path="assets/solda.png"
image=Image.open(path).convert('L')
l,h=image.size
threshold = find_threshold(image)
b_image_array = np.where(np.array(image)>threshold, 255, 0)
b_image = Image.fromarray(b_image_array.astype(np.uint8))

g_arrays = get_groups(b_image_array)
g_image = get_image_groups(g_arrays)

plt.imshow(g_image, cmap='gray')
plt.show()


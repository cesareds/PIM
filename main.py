import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def threshold(image):
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

def biggest(image):
    visited = set()
    components = []
    image = np.array(image)
    rows, cols = image.shape
    for j in range(cols):
        for i in range(rows):
            if image[i][j] > 0 and (i, j) not in visited:
                component = bfs((j, i), visited, rows, cols, image)
                components.append(component)
    biggest = max(components, key=len)
    return biggest

def gp_img(group):
    ret = [[0 for _ in range(l)] for _ in range(h)]
    for c in group:
        ret[c[1]][c[0]] = 255
    return Image.fromarray((np.array(ret)).astype(np.uint8))

def border_cleaner(bimg):
    rows, cols = bimg.shape
    visited = set()

    def bfs_border(root):
        queue = [root]
        component = []
        border = False
        while queue:
            x, y = queue.pop(0)
            component.append((x, y))
            visited.add((x, y))
            if x == 0 or x == cols-1 or y == 0 or y == rows-1:
                border = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited and bimg[ny, nx] > 0:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return component, border

    for y in range(rows):
        for x in range(cols):
            if bimg[y, x] > 0 and (x, y) not in visited:
                component, border_touching = bfs_border((x, y))
                if border_touching:
                    for (cx, cy) in component:
                        bimg[cy, cx] = 0

    return bimg

def center(group):
    x_all = [c[0] for c in group]
    y_all = [c[1] for c in group]
    x_center = np.mean(x_all)
    y_center = np.mean(y_all)
    return (x_center, y_center)


path="assets/solda.png"
image=Image.open(path).convert('L')
l,h=image.size

# encontra o threshold
threshold = threshold(image)

#torna a imagem binária a partir do threshold
bimg = np.where(np.array(image)>threshold, 255, 0)

#usa bfs nas bordas para remover os grupos que encostam nela
bimg = border_cleaner(bimg)

#encontra o maior grupo usando bfs
biggest = biggest(bimg)

#encontra o centro de massa do grupo
c = center(biggest)

#transforma o array em imagem
img = gp_img(biggest)

#define a imagem que será mostrada
plt.imshow(img, cmap='gray')

#plota uma letra O na cordenada do centro de massa definida por c[0](x) e c[1](y)
plt.plot(c[0], c[1], 'ro')

#exibe a imagem
plt.show()


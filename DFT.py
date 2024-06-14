import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.metrics import structural_similarity

# Distância entre pontos 40 24

def fourier_masker_ver(image, i):
    font_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    w, h = image.size
    
    # Linha Vertical
    dark_image_grey_fourier[:(int(h/2)-10), int(w/2)-2:int(w/2)+2] = i
    dark_image_grey_fourier[-(int(h/2)-10):, int(w/2)-2:int(w/2)+2] = i
    
    # Linha horizontal
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, :(int(w/2)-10)] = i
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, -(int(w/2)-10):] = i
    
    fig, ax = plt.subplots(1,4,figsize=(12,8))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = font_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = font_size)
    transformed1 = increase_brightness_1(abs(np.fft.ifft2(dark_image_grey_fourier)))
    ax[2].imshow(transformed1, cmap='gray')
    ax[2].set_title('Transformed Greyscale Image 1', fontsize = font_size)
    transformed2 = increase_brightness_2(abs(np.fft.ifft2(dark_image_grey_fourier)))
    ax[3].imshow(transformed2, cmap='gray')
    ax[3].set_title('Transformed Greyscale Image 2', fontsize = font_size)

    ssim_index = structural_similarity(transformed2, np.array(image), data_range=255)
    print(ssim_index)

def increase_brightness_1(dark_image):
    lighter_image = (dark_image - np.min(dark_image)) / (np.max(dark_image) - np.min(dark_image))
    lighter_image = lighter_image * 255
    lighter_image = equalize_hist(lighter_image.astype(np.uint8))

    return lighter_image

def increase_brightness_2(dark_image):
    lighter_image = (dark_image - np.min(dark_image)) / (np.max(dark_image) - np.min(dark_image))
    lighter_image = lighter_image * 255
    lighter_image = equalize_adapthist(lighter_image.astype(np.uint8))

    return lighter_image



path_r = "assets/folhas1_Reticulada.jpg"
img_r = Image.open(path_r).convert('L')
path = "assets/folhas1.jpg"
img = Image.open(path).convert('L')

fourier_masker_ver(img_r, 0.2)

plt.show()

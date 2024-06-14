import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
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
    
    reversed_fourier = abs(np.fft.ifft2(dark_image_grey_fourier))

    fig, ax = plt.subplots(1,5,figsize=(12,8))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = font_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = font_size)
    transformed1 = increase_brightness_equalizing_histogram(reversed_fourier)
    ax[2].imshow(transformed1, cmap='gray')
    ax[2].set_title('Hist Eq', fontsize = font_size)
    transformed2 = increase_brightness_equalizing_histogram_adapted(reversed_fourier)
    ax[3].imshow(transformed2, cmap='gray')
    ax[3].set_title('Adapt Hist Eq', fontsize = font_size)
    transformed3 = increase_brightness_pil_enhancer(reversed_fourier)
    ax[4].imshow(transformed3, cmap='gray')
    ax[4].set_title('Pil Enhanced', fontsize = font_size)

    ssim_index = structural_similarity(transformed1, np.array(image), data_range=255)
    print(ssim_index)
    ssim_index = structural_similarity(transformed2, np.array(image), data_range=255)
    print(ssim_index)
    # ssim_index = structural_similarity(transformed3, np.array(image), data_range=255)
    # print(ssim_index)

def increase_brightness_equalizing_histogram(dark_image):
    lighter_image = (dark_image - np.min(dark_image)) / (np.max(dark_image) - np.min(dark_image))
    lighter_image = lighter_image * 255
    lighter_image = equalize_hist(lighter_image.astype(np.uint8))

    return lighter_image

def increase_brightness_equalizing_histogram_adapted(dark_image):
    lighter_image = (dark_image - np.min(dark_image)) / (np.max(dark_image) - np.min(dark_image))
    lighter_image = lighter_image * 255
    lighter_image = equalize_adapthist(lighter_image.astype(np.uint8))

    return lighter_image

def increase_brightness_pil_enhancer(dark_image):
    enhancer = ImageEnhance.Brightness(Image.fromarray(dark_image.astype(np.uint8)))
    lighter_image = enhancer.enhance(1.5)

    return lighter_image


path_r = "assets/folhas1_Reticulada.jpg"
img_r = Image.open(path_r).convert('L')
path = "assets/folhas1.jpg"
img = Image.open(path).convert('L')

fourier_masker_ver(img_r, 0.2)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from skimage.metrics import structural_similarity

# Distância entre pontos 40 24

def fourier_masker_ver(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    w, h = image.size
    
    # Linha Vertical
    dark_image_grey_fourier[:(int(h/2)-10), int(w/2)-2:int(w/2)+2] = i
    dark_image_grey_fourier[-(int(h/2)-10):, int(w/2)-2:int(w/2)+2] = i
    
    # Linha horizontal
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, :(int(w/2)-10)] = i
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, -(int(w/2)-10):] = i
    
    fig, ax = plt.subplots(1,3,figsize=(12,8))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size)
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', fontsize = f_size)

    # ssim_index = structural_similarity(dark_image_grey_fourier, img, multichannel=True)
    # print(ssim_index)


path_r = "assets/folhas1_Reticulada.jpg"
img_r = Image.open(path_r).convert('L')
path = "assets/folhas1.jpg"
img = Image.open(path).convert('L')

fourier_masker_ver(img_r, 0.2)

plt.show()

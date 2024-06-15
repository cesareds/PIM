import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.metrics import structural_similarity

# Distância entre pontos 40 24

def fourier_masker_ver(image, i):
    font_size = 8
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    w, h = image.size
    
    # Linha Vertical
    dark_image_grey_fourier[:(int(h/2)-10), int(w/2)-2:int(w/2)+2] = i
    dark_image_grey_fourier[-(int(h/2)-10):, int(w/2)-2:int(w/2)+2] = i
    
    # Linha horizontal
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, :(int(w/2)-10)] = i
    dark_image_grey_fourier[int(h/2)-2:int(h/2)+2, -(int(w/2)-10):] = i
    
    reversed_fourier = abs(np.fft.ifft2(dark_image_grey_fourier))
    ssimOG = structural_similarity(reversed_fourier, np.array(image), data_range=255)

    fig, ax = plt.subplots(3,3,figsize=(12,8))
    ax[0,1].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0,1].set_title('Masked Fourier', fontsize = font_size)
    ax[0,1].axis('off')

    ax[0,0].imshow(image, cmap = 'gray')
    ax[0,0].set_title('Greyscale Image', fontsize = font_size)
    ax[0,0].axis('off')

    transformed_eq_hist = increase_brightness_equalizing_histogram(reversed_fourier)
    ssim_eq_hist = structural_similarity(transformed_eq_hist, np.array(image), data_range=255)
    ax[2,0].imshow(transformed_eq_hist, cmap='gray')
    ax[2,0].set_title('Hist Eq: '+ str(ssim_eq_hist), fontsize = font_size)
    ax[2,0].axis('off')

    transformed_eq_hist_adapted = increase_brightness_equalizing_histogram_adapted(reversed_fourier)
    ssim_eq_hist_adapted = structural_similarity(transformed_eq_hist_adapted, np.array(image), data_range=255)
    ax[1,0].imshow(transformed_eq_hist_adapted, cmap='gray')
    ax[1,0].set_title('Adapt Hist Eq: '+ str(ssim_eq_hist_adapted), fontsize = font_size)
    ax[1,0].axis('off')

    (transformed_pil, ssim_pil) = increase_brightness_pil_enhancer(reversed_fourier)
    ax[1,1].imshow(transformed_pil, cmap='gray')
    ax[1,1].set_title('Pil Enhance: '+ str(ssim_pil), fontsize = font_size)
    ax[1,1].axis('off')

    (transformed_cv, ssim_cv) = increase_brightness_cv2(reversed_fourier)
    ax[1,2].imshow(transformed_cv, cmap='gray')
    ax[1,2].set_title('CV2 Enhance: '+ str(ssim_cv), fontsize = font_size)
    ax[1,2].axis('off')

    ax[0,2].imshow(reversed_fourier, cmap='gray')
    ax[0,2].set_title('Fourier: '+ str(ssimOG), fontsize = font_size)
    ax[0,2].axis('off')

    print("SSIMs:\n")
    print("{:<30} {:<10.6f}".format("Original:", ssimOG))
    print("{:<30} {:<10.6f}".format("Equalized Histogram:", ssim_eq_hist))
    print("{:<30} {:<10.6f}".format("Equalized Adapted Histogram:", ssim_eq_hist_adapted))
    print("{:<30} {:<10.6f}".format("PIL Enhancement:", ssim_pil))
    print("{:<30} {:<10.6f}".format("CV2 Enhancement:", ssim_cv))
    

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
    if isinstance(dark_image, np.ndarray):  
        dark_image = Image.fromarray(dark_image.astype('uint8'))  
    i = 0.1
    prev_ssim = -1
    img_r_np = np.array(img_r) 
    while i <= 1.0: 
        enhancer = ImageEnhance.Brightness(dark_image)
        enhanced_image = enhancer.enhance(i)
        enhanced_image_np = np.array(enhanced_image) 
        current_ssim = structural_similarity(enhanced_image_np, img_r_np, multichannel=True)
        if current_ssim < prev_ssim: 
            break
        prev_ssim = current_ssim
        i += 0.1 
    return enhanced_image, current_ssim

def increase_brightness_cv2(dark_image):
    path_r2 = "assets/folhas1_Reticulada.jpg"
    img_r2 = cv2.imread(path_r2, cv2.IMREAD_GRAYSCALE)
    i = 0.1
    prev_ssim = -1
    while i <= 1.0: 
        lighter_image = cv2.convertScaleAbs(dark_image, alpha=i) 
        lighter_image = cv2.equalizeHist(lighter_image)  
        current_ssim = structural_similarity(lighter_image, img_r2) 
        if current_ssim < prev_ssim: 
            break
        prev_ssim = current_ssim
        i += 0.1
    return lighter_image, current_ssim

path_r = "assets/folhas1_Reticulada.jpg"
img_r = Image.open(path_r).convert('L')
path = "assets/folhas1.jpg"
img = Image.open(path).convert('L')

ssim_enhacements = []

fourier_masker_ver(img_r, 0.2)
plt.tight_layout()
plt.show()

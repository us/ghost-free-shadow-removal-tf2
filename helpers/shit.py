# import cv2
# import numpy as np
#
# img = cv2.imread('../shit_test/000016.png', -1)
#
# rgb_planes = cv2.split(img)
#
# result_planes = []
# result_norm_planes = []
# for plane in rgb_planes:
#     dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(plane, bg_img)
#     norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     result_planes.append(diff_img)
#     result_norm_planes.append(norm_img)
#
# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)
#
# cv2.imwrite('shadows_out.png', result)
# cv2.imwrite('shadows_out_norm.png', result_norm)
import cv2
import numpy as np
from PIL import Image


# # read and resize and save image
# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     return blurred_image
#
# def apply_clahe(image):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized_image = clahe.apply(image)
#     return equalized_image
#
#
# def remove_shadows(image):
#     dilated_image = cv2.dilate(image, np.ones((5, 5), np.uint8))
#     background = cv2.medianBlur(dilated_image, 21)
#     normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#     shadow_free_image = cv2.divide(normalized_image, background, scale=255)
#     return shadow_free_image
#
# def homomorphic_filter(image, gamma_l=0.3, gamma_h=1.2, c=1, d0=20):
#     h, w = image.shape
#     image_log = np.log1p(np.float32(image))
#
#     x, y = np.meshgrid(np.linspace(-w//2, w//2-1, w), np.linspace(-h//2, h//2-1, h))
#     distance = np.sqrt(x**2 + y**2)
#
#     mask = np.ones((h, w))
#     mask = mask + (gamma_h - gamma_l) * (1 - np.exp(-c * (distance**2) / (d0**2)))
#
#     image_fft = np.fft.fftshift(np.fft.fft2(image_log))
#     filtered_image_fft = image_fft * mask
#     filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_image_fft)))
#
#     output_image = np.uint8(np.expm1(filtered_image))
#     return output_image
#
#
# def preprocess_mask(mask_path):
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     normalized_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized_mask

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized_image
# def preprocess_mask(mask_path):
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     normalized_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized_mask
#
# def blend_with_mask_opencv(image, mask, alpha=0.5, beta=0.5, gamma=0):
#     blended_image = cv2.addWeighted(image, alpha, mask, beta, gamma)
#     return blended_image
#
# def blend_with_mask_pil(image, mask, alpha=0.5):
#     image_pil = Image.fromarray(image)
#     mask_pil = Image.fromarray(mask)
#     blended_image_pil = Image.blend(image_pil, mask_pil, alpha)
#     blended_image = np.array(blended_image_pil)
#     return blended_image


# input_image_path = "/Users/syny/kafamagore/shadowremover/dataset/selected_images/original_images/000016.jpg"
# output_image_path = "./non_shadow.png"
# image = preprocess_image(input_image_path)

input_image_path = "/Users/syny/kafamagore/shadowremover/dataset/selected_images/original_images/000016.jpg"
input_mask_path = "/Users/syny/kafamagore/shadowremover/dataset/selected_images/original_images_shadows/000016.jpg"
output_image_path = 'output_image.jpg'

# resize mask with image size

# image = preprocess_image(input_image_path)
# mask = preprocess_mask(input_mask_path)
# mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
#
# # Using OpenCV
# # blended_image = blend_with_mask_opencv(image, mask, alpha=0.5, beta=0.5)
# blended_image = blend_with_mask_opencv(image, mask, alpha=0.1, beta=0.1)
# # Using PIL
# # blended_image = blend_with_mask_pil(image, mask, alpha=0.5)
#
# cv2.imwrite(output_image_path, blended_image)
import cv2

import cv2

def bilateral_filter(image, diameter=9, sigma_color=1575, sigma_space=175):
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image

def morphological_opening(image, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image


import cv2
import os

def process_images(input_image_path, output_folder, bilateral_params, opening_params):
    # Read the image and convert to grayscale
    image = cv2.imread(input_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for diameter, sigma_color, sigma_space in bilateral_params:
        for kernel_size in opening_params:
            # Apply bilateral filtering and morphological opening
            filtered_image = bilateral_filter(gray_image, diameter, sigma_color, sigma_space)
            opened_image = morphological_opening(filtered_image, kernel_size)

            # Save the processed image
            output_image_name = f"output_d{diameter}_sc{sigma_color}_ss{sigma_space}_ks{kernel_size}.jpg"
            output_image_path = os.path.join(output_folder, output_image_name)
            cv2.imwrite(output_image_path, opened_image)

# Define your input image path and output folder
output_folder = 'shitty_folder'

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define the parameter ranges for bilateral filtering and morphological opening
bilateral_params = [(d, sc, ss) for d in [5, 9, 15] for sc in [50, 75, 100] for ss in [50, 75, 100]]
opening_params = [3, 5, 7]

# Process the images and save the results in the output folder
process_images(input_image_path, output_folder, bilateral_params, opening_params)

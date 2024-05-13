import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import torch


def plot_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    for i, image in enumerate(images):
        axes[i].imshow(image)  # Assuming grayscale images
        axes[i].axis('off')
    plt.tight_layout()
    # Save the image 
    plt.savefig("results/test3.jpg")

# Open the image
og_img = io.imread("test_imgs/test_pre_satshift.png")

# og_img[:,:,1:2] += -.5
print(og_img)


plt.imsave("test_imgs/test_NEWPOST_satshift.png", og_img)

# rot_img_list = []

# # Apply rotations of 120 degrees 
# for i in [0, 240, 120]:
#     hsv_image = og_hsv_img.copy()
#     # Hue / Sat / Val all on a 0 - 1 scale for skimage
#     # Fix hue to 0 - 360 scale
#     hsv_image[:, :, 0] = hsv_image[:, :, 0] * 360

#     # Apply rotation
#     hsv_image[:, :, 0] =  (hsv_image[:, :, 0] + i) % 360
#     # Display the image
#     plt.axis('off')  # Turn off axis:

#     # Convert back to 0 - 1 scale for saving the image
#     hsv_image[:, :, 0] = hsv_image[:, :, 0] / 360

#     # Convert back to rgb for saving
#     rbg_img = color.hsv2rgb(hsv_image)

#     rot_img_list.append(rbg_img)


# plot_images(rot_img_list)
    


# # Display the image
# plt.axis('off')  # Turn off axis:

# # Save the image 
# plt.imsave("results/og_img.jpg", og_img)
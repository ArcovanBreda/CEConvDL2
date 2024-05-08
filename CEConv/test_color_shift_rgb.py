import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io
import numpy as np
import torch
import torch.nn.functional as F

from ceconv.ceconv2d import _get_hue_rotation_matrix

def plot_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    for i, image in enumerate(images):
        axes[i].imshow(image)  # Assuming grayscale images
        axes[i].axis('off')
    plt.tight_layout()
    # Save the image 
    plt.savefig("results/120shift_rgb.jpg")

def rotate_img_rgb(og_img, rot_mat):
    og_img = torch.tensor(og_img, dtype=torch.float)

    rot_img = og_img @ rot_mat

    # Get a 0 to 1 range for RGB plotting
    rot_img = np.array(rot_img) / 255
    rot_img[rot_img < 0] = 0
    return rot_img


# Open the image
og_img = io.imread("102flowers/jpg/image_00001.jpg")

num_rotations = 3
rotation_matrix = _get_hue_rotation_matrix(num_rotations)
rotation_matrices = torch.stack(
        [torch.matrix_power(rotation_matrix, i) for i in range(num_rotations)], dim=0
    )

rot_img_list = []
for i in range(num_rotations):
    rot_img = rotate_img_rgb(og_img, rotation_matrices[i])
    rot_img_list.append(rot_img)
plot_images(rot_img_list)

# # Display the image
# plt.axis('off')  # Turn off axis:

# # Save the image 
# plt.imsave("results/og_img.jpg", og_img)
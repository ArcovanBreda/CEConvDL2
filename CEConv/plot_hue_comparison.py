from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math
from skimage import color
import kornia
import skimage as ski
import torchvision
import torch
from PIL import Image


def get_3Dmatrix(rotations):
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return np.asarray(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ]
    )


def get_2Dmatrix(angle):
    # Rotation matrix
    angle = (math.pi / 180) * angle
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )


def get_3Dhuematrix(num_rotations):
    # Rotation matrix
    angle = 2 * math.pi / num_rotations
    return np.asarray(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
    )


PIL_img = Image.open("blogpost_imgs/flower_example.jpg")  # Range 0-255
tensor_img = torchvision.transforms.functional.to_tensor(PIL_img)
lab_tensor = kornia.color.rgb_to_lab(tensor_img)
im = ski.io.imread("blogpost_imgs/flower_example.jpg")  # Range 0-255
hsv = color.rgb2hsv(im)  # Range 0-1
lab = color.rgb2lab(im)  # Range 01, -127-128, -128, -127

shifts = 9
delta = 1 / shifts
angle_delta = 360 / shifts
matrix = get_3Dmatrix(shifts)
lab_matrix = get_3Dhuematrix(shifts)

fig, axs = plt.subplots(3, shifts, figsize=(15, 5))
for c in range(3):
    for i in range(shifts):
        if c == 0:
            shift = i * delta
            np_image = hsv.copy()
            np_image[:, :, 0] -= shift
            np_image[:, :, 0] = np_image[:, :, 0] % 1
            axs[c, i].imshow(color.hsv2rgb(np_image))
            if i == shifts//2:
                axs[c, i].set_title("HSV space")
            axs[c, i].axis("off")
        elif c == 1:
            np_image = im
            # print(np_image[:, :, 1:].max(), np_image[:, :, 1:].min())
            np_image = np_image @ np.linalg.matrix_power(matrix, i)
            np_image = np_image / 255
            np_image = np_image.clip(0, 1)
            axs[c, i].imshow(np_image)
            if i == shifts//2:
                axs[c, i].set_title("RGB space")
            axs[c, i].axis("off")
        # elif c == 2:
        #     angle = angle_delta * i
        #     np_image = lab.copy()
        #     np_image[:, :, 1:] = (np_image[:, :, 1:]) @ get_2Dmatrix(angle)
        #     axs[c, i].imshow(color.lab2rgb(np_image))
        #     axs[c, i].axis("off")
        # elif c == 3:
        #     np_image = lab.copy()
        #     np_image = np_image @ np.linalg.matrix_power(lab_matrix, i)
        #     axs[c, i].imshow(color.lab2rgb(np_image))
        #     axs[c, i].axis("off")
        elif c == 2:
            np_image = lab_tensor.clone().float().moveaxis(0, -1)
            np_image = torch.einsum(
                "whc, cr->whr",
                np_image,
                torch.matrix_power(torch.from_numpy(lab_matrix), i).float(),
            )
            axs[c, i].imshow(
                np.moveaxis(
                    kornia.color.lab_to_rgb(np_image.moveaxis(-1, 0)).numpy(), 0, -1
                )
            )
            if i == shifts//2:
                axs[c, i].set_title("LAB space")
            axs[c, i].axis("off")

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.suptitle("Hue shift in HSV space / RGB space / LAB space", fontsize=16)
plt.savefig("blogpost_imgs/hue_shift_comparison.jpg")

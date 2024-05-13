import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io
import numpy as np
import torch
import torch.nn.functional as F
import kornia

# import required library 
import torch 
import torchvision 
import torchvision.transforms as T 
import torchvision.transforms.functional as F 
from torchvision.io import read_image 
from torchvision.utils import save_image

# def plot_images(images):
#     num_images = len(images)
#     fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
#     for i, image in enumerate(images):
#         axes[i].imshow(image)  # Assuming grayscale images
#         axes[i].axis('off')
#     plt.tight_layout()
#     # Save the image 
#     plt.savefig("test_imgs/bobob_satshift.jpg")

# # Open the image
# og_img = read_image("102flowers/jpg/image_00001.jpg")
# og_img = F.adjust_saturation(og_img, 0.2)

# test_jitter = np.append(np.linspace(0, 1, 10, endpoint=False), np.arange(1, 10, 1, dtype=int))
# test_jitter = np.append(test_jitter, np.arange(10, 100, 10, dtype=int))
# test_jitter = np.append(test_jitter, np.arange(100, 500, 50, dtype=int))

# print(len(test_jitter))

# sat_shifted_img_list = []
# for i in test_jitter:
#     sat_shifted_img = F.adjust_saturation(og_img, i)
#     sat_shifted_img = np.array(sat_shifted_img.permute(1,2,0))
#     sat_shifted_img_list.append(sat_shifted_img)

# plot_images(sat_shifted_img_list)


# og_img = np.array(og_img.permute(1,2,0))

# plt.imsave("test_imgs/bobobobobob.jpg", og_img)


# saturations = 50
# neg_sats = saturations // 2
# pos_sats = neg_sats - 1 + saturations % 2
# test_jitter = torch.concat((torch.linspace(-1, 0, neg_sats + 1)[:-1],
#                                                 torch.tensor([0]),
#                                                 torch.linspace(0, 1, pos_sats + 1)[1:])).type(torch.float32)

# saturations = 50
# neg_sats = saturations // 2
# pos_sats = neg_sats - 1 + saturations % 2

# test = np.linspace(-1, 1, saturations)
# print(test.shape)
# print(test)


# test2 = np.append(np.linspace(-1, 0, neg_sats+1)[:-1], np.linspace(0, 1, pos_sats+1))
# print(test2.shape)
# print(test2)

# test3 = torch.tensor([torch.tensor([1], device=0), torch.tensor([2], device=0), torch.tensor([3], device=0)])

# for i in test3:
#     print(i.item())

# import required libraries
import torch
import torchvision.transforms as transforms
from PIL import Image

# Read the image
img = Image.open("102flowers/jpg/image_00001.jpg")
print(type(img))

# define a transform
transform = transforms.ColorJitter(brightness=0, contrast=0, saturation=[.5,1.5], hue=0)

# apply above transform on input image
img = transform(img)

# visualize the image
img.save("test3.jpg")
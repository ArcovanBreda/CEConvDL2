import numpy as np
import matplotlib.pyplot as plt
import torch

test = torch.tensor([-3.5, -2, -1, 1, 2, 3])
print(torch.remainder(test, 2))

# img = np.zeros((500, 500, 3), dtype=np.uint8)
# img[:, :, 0] = 255
# img[:, :, 2] = 55

# plt.imsave("test_imgs/magneta.jpg", img)

# from PIL import Image

# # Open the image
# image = Image.open("102flowers/jpg/image_00001.jpg")

# # Show the image
# image.show()

# # Save the image (you can change the format and file name as needed)
# image.save("example_modified.jpg")

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Open the image
# image = mpimg.imread("102flowers/jpg/image_00001.jpg")

# # Display the image
# plt.imshow(image)
# plt.axis('off')  # Turn off axis

# # Save the image (you can change the format and file name as needed)
# plt.imsave("results/example_modified3.jpg", image)
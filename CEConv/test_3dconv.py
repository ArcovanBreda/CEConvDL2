import torch
import torch.nn.functional as F

num_rotations = 4

# Batch, inchannels (3), 1 (time dimension), H, W 
img = torch.tensor([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
                    [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]],
                    [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]]]
                    ).unsqueeze(0).unsqueeze(2)

print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Og image")
print(img)
print("XXXXXXXXXXXXXXXXXXXXXXX")


# Batch, inchannels (3), 4(time dimension) = image stack, H, W 
img = img.repeat(1,1,num_rotations,1,1)

print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Image stack")
print(img)
print("XXXXXXXXXXXXXXXXXXXXXXX")

# shift hue of image stack
# loop over all images in the stack
for i in range(num_rotations):
    # add i to the hue channel of each image
    # Batch, inchannels (HSV), img_in_stack (time dimension) 4, H, W
    img[:,0:1,i:i+1,:,:] += i 

print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Hue shifted Image stack")
print(img)
print("XXXXXXXXXXXXXXXXXXXXXXX")

print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Print individual images in the image stack")
for i in range(num_rotations):
    print(img[:,:,i,:,:])
print("XXXXXXXXXXXXXXXXXXXXXXX")

# Out_channels, inchannels (3), 1(time dimension), H, W 
kernel = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]],
                       [[2,2,2],[2,2,2],[2,2,2]],
                       [[3,3,3],[3,3,3],[3,3,3]]]).unsqueeze(0).unsqueeze(2)
print(kernel.shape)


# convolution_3d = torch.nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(3,3,3), stride=(1,1,1))
# y = convolution_3d()

y = F.conv3d(img, kernel, stride=[1,1,1])

print("XXXXXXXXXXXXXXXXXXXXXXX")
print("IMG")
print(img.shape)
print(img)
print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Kernel")
print(kernel.shape)
print(kernel)
print("XXXXXXXXXXXXXXXXXXXXXXX")
print("Feature Map")
print(y.shape)
print(y)
print("XXXXXXXXXXXXXXXXXXXXXXX")
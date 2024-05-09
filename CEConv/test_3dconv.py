import torch
import torch.nn.functional as F

img = torch.tensor([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
                    [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]],
                    [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]]]
                    ).unsqueeze(0).unsqueeze(2)
# Batch, inchannels (3), 1 (time dimension), H, W 
img = img.repeat(1,1,4,1,1)
# Batch, inchannels (3), 12 (time dimension) = image stack, H, W 



kernel = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]],
                       [[2,2,2],[2,2,2],[2,2,2]],
                       [[3,3,3],[3,3,3],[3,3,3]]]).unsqueeze(0).unsqueeze(2)
print(kernel.shape)
# Out_channels, inchannels (3), 1(time dimension), H, W 

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
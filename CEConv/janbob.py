import numpy as np

# # Create your lists of values
# list1 = [1, 2, 3, 4, 5]
# list2 = [10, 20, 30, 40, 50]

# # Save the lists in an npz file with different keys
# np.savez('results/multiple_lists.npz', list1=list1, list2=list2)

# Load the saved npz file
data = np.load('outputs/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-2024-05-09_15:03:34.npz')

# Access the lists by their keys
hue = data["hue"]
acc = data["acc"]
print("Hue list:", hue)  # Output: [1, 2, 3, 4, 5]
print("Acc list:", acc)  # Output: [10, 20, 30, 40, 50]

import matplotlib.pyplot as plt
import numpy as np

# x = np.load("output/test_results/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["hue"]
# baseline = np.load("output/test_results/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
# baseline_jitter = np.load("output/test_results/hue_shift_img/hue_baseline_jitter_flowers102-resnet18_1-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
# hue_eq = np.load("output/test_results/hue_shift_img/hue_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
# hue_eq_jitter = np.load("output/test_results/hue_shift_img/hue_flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
x = np.load("output/test_results/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["hue"]
baseline = np.load("output/test_results/hue_shift_kernel/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
baseline_jitter = np.load("output/test_results/hue_shift_kernel/hue_baseline_jitter_flowers102-resnet18_1-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
hue_eq = np.load("output/test_results/hue_shift_kernel/hue_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
hue_eq_jitter = np.load("output/test_results/hue_shift_kernel/hue_flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(x, baseline * 100, label="Resnet-18 (Baseline)")
plt.plot(x, baseline_jitter * 100, label="Resnet-18 + jitter (Baseline)")
plt.plot(x, hue_eq * 100, label="CE-Resnet-18", ls="--")
plt.plot(x, hue_eq_jitter * 100, label="CE-Resnet-18 + jitter", ls="--")

# plt.title("Hue equivariant network trainend in HSV color space\n Image Shift | Flowers-102 dataset", fontsize=22)
plt.title("Hue equivariant network trainend in HSV color space\n Kernel Shift | Flowers-102 dataset", fontsize=22)
plt.ylabel("Test accuracy (%)", fontsize=18)
plt.yticks(fontsize=16,)
plt.xlabel("Test-time hue shift (°)", fontsize=18)
plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0., ncol=3, fancybox=True, shadow=True,
        columnspacing=0.7, handletextpad=0.2)
plt.grid(axis="both")
plt.ylim(0, 100)
# plt.savefig("output/test_results/hue_shift_img/hue_shift_img.jpg")
plt.savefig("output/test_results/hue_shift_kernel/hue_shift_kernel.jpg")

import matplotlib.pyplot as plt
import numpy as np

x = np.load("output/test_results/maintest_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1.npz")["hue"]
y_norm = np.load("output/test_results/maintest_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1.npz")["acc"]
# y_norm_jitter = np.load("")["acc"]
# y_nonorm = np.load("")["acc"]
# y_nonorm_jitter = np.load("")["acc"]
# y_norm_remainder1 = np.load("outputs/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-2024-05-10_11:36:12_norm_imagehueshift.npz")["acc"]

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(x, y_norm, label="RGB norm")
# plt.plot(x, y_nonorm, label="No norm")
# plt.plot(x, y_norm_jitter, label="RGB norm + Jitter", ls="--")
# plt.plot(x, y_nonorm_jitter, label="No norm + Jitter", ls="--")
# plt.plot(x, y_norm_remainder1, label="RGB norm + remained 1", ls="dotted")

plt.title("Hue equivariant network trainend in HSV color space\nFlowers-102 dataset [Hue-shift on Image - 120 degree rotations]", fontsize=22)
plt.ylabel("Test accuracy (%)", fontsize=18)
plt.yticks(fontsize=16,)
plt.xlabel("Test-time hue shift (°)", fontsize=18)
plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0., ncol=3, fancybox=True, shadow=True,
        columnspacing=0.7, handletextpad=0.2)
plt.grid(axis="both")
plt.ylim(0, 1)
plt.savefig("Hue_HSV_Fig9_hueshiftimage.jpg")
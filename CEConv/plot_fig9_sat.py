import matplotlib.pyplot as plt
import numpy as np

x = np.load("output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-img_shift.npz")["hue"]
y_norm = np.load("output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-img_shift.npz")["acc"]
# y_norm_jitter = np.load("outputs/test_results/sil_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100-2024-05-11_17:40:44_imgsatshift_norm_jitter.npz")["acc"]
# y_nonorm = np.load("outputs/test_results/sil_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm-2024-05-11_19:46:53_imgsatshift_nonorm.npz")["acc"]
# y_nonorm_jitter = np.load("outputs/test_results/sil_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100-no_norm-2024-05-11_22:13:58_imgsatshit_nonorm_jitter.npz")["acc"]

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(x, y_norm, label="RGB norm")
# plt.plot(x, y_norm_jitter, label="RGB norm + Jitter", ls="--")
# plt.plot(x, y_nonorm, label="No norm")
# plt.plot(x, y_nonorm_jitter, label="No norm + Jitter", ls="--")

plt.title("Saturation Equivariant ResNet-18 trained in HSV space \n[Sat Shift on Image - 5 shifts (-1 -.5 0 .5 1)]", fontsize=22)
plt.ylabel("Test accuracy (%)", fontsize=18)
plt.yticks(fontsize=16,)
plt.ylim(top=0.99)
plt.xlabel("Test-time saturation shift", fontsize=18)
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0., ncol=2, fancybox=True, shadow=True,
        columnspacing=0.7, handletextpad=0.2)
plt.grid(axis="both")
plt.savefig("Sat_HSV_Fig9_satshiftimage.jpg")
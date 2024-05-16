import matplotlib.pyplot as plt
import numpy as np

npz_folder = "./CEConv/output/test_results"
x = np.load(
    f"{npz_folder}/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-value_shift-sat_jitter_1_1-val_jitter_0_100-img_shift-no_norm.npz"
)["hue"]

y_base = (
    np.load(
        f"{npz_folder}/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-value_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz"
    )["acc"]
    * 100
)
y_base_jitter = (
    np.load(
        f"{npz_folder}/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-value_shift-sat_jitter_1_1-val_jitter_0_100-img_shift-no_norm.npz"
    )["acc"]
    * 100
)

y_ce = (
    np.load(
        f"{npz_folder}/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-value_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz"
    )["acc"]
    * 100
)

y_ce_jitter = (
    np.load(
       f"{npz_folder}/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-value_shift-sat_jitter_1_1-val_jitter_0_100-img_shift-no_norm.npz"
    )["acc"]
    * 100
)

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(x, y_base, label=f"Resnet-18 ({y_base.mean():.2f}%)")
plt.plot(
    x, y_base_jitter, label=f"Resnet-18 + Jitter ({y_base_jitter.mean():.2f}%)", ls="--"
)

plt.plot(x, y_ce, label=f"CE-Resnet-18 ({y_ce.mean():.2f}%)")

plt.plot(
    x, y_ce_jitter, label=f"CE-Resnet-18 + jitter ({y_ce_jitter.mean():.2f}%)", ls="--"
)

plt.title(
    "Value equivariant networks trainend in HSV color space\nFlowers-102 dataset (5 shifts [-0.5,-0.25,0,0.5,1] in image space)",
    fontsize=22,
)
plt.ylabel("Test accuracy (%)", fontsize=18)
plt.yticks(
    fontsize=16,
)
plt.xlabel("Test-time value shift", fontsize=18)

plt.legend(
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    borderaxespad=0.0,
    ncol=2,
    fancybox=True,
    shadow=True,
    columnspacing=0.7,
    handletextpad=0.2,
)
plt.grid(axis="both")
plt.ylim(0, 100)
plt.savefig("blogpost_imgs/value_equivariance.jpg")

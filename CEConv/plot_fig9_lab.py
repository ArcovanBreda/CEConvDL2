import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#TODO GET NPZ from Snellius

npz_folder = "./CEConv/output/test_results/lab_shift"
x = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["hue"]
y_base = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
y_base_jitter = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_5-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
y_ce_jitter = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
y_ce_lab = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-lab_test-no_norm.npz")["acc"] * 100
y_ce = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(x, y_base, label=f"Resnet-18 ({y_base.mean():.2f}%)")
plt.plot(
    x,
    y_base_jitter,
    label=f"Resnet-18 + Jitter ({y_base_jitter.mean():.2f}%)",
    ls="--",
)

plt.plot(x, y_ce, label=f"CE-Resnet-18 ({y_ce.mean():.2f}%)")
plt.plot(
    x,
    y_ce_jitter,
    label=f"CE-Resnet-18 + Jitter ({y_ce_jitter.mean():.2f}%)",
    ls="--",
)

plt.plot(
    x,
    y_ce_lab,
    label=f"CE-Resnet-18 + LAB shift ({y_ce_lab.mean():.2f}%)",
    ls="dotted",
)

plt.title(
    "Hue equivariant networks trainend in LAB color space\nFlowers-102 dataset (3 shifts [-120°, 0°, 120°])",
    fontsize=22,
)
plt.ylabel("Test accuracy (%)", fontsize=18)
plt.yticks(
    fontsize=16,
)
plt.xlabel("Test-time hue shift (°)", fontsize=18)
plt.xticks(
    fontsize=16,
    ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],
    labels=["-150", "-100", "-50", "0", "50", "100", "150"],
)
plt.legend(
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    borderaxespad=0.0,
    ncol=3,
    fancybox=True,
    shadow=True,
    columnspacing=0.7,
    handletextpad=0.2,
)
plt.grid(axis="both")
plt.ylim(0, 100)
plt.savefig("blogpost_imgs/lab_equivariance.jpg")

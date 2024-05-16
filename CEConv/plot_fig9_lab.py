import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#TODO GET NPZ from Snellius

npz_folder = "./CEConv/output/test_results"
x = pd.read_csv(f"{npz_folder}/base.csv")["hue"]
y_base = pd.read_csv(f"{npz_folder}/base.csv")["acc"] * 100
y_base_jitter = pd.read_csv(f"{npz_folder}/base_jitter.csv")["acc"] * 100
y_ce_jitter = pd.read_csv(f"{npz_folder}/CE_jitter.csv")["acc"] * 100
y_ce_lab = pd.read_csv(f"{npz_folder}/CE_lab_test.csv")["acc"] * 100
y_ce = pd.read_csv(f"{npz_folder}/CE.csv")["acc"] * 100

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

"""
Contains plotting code for the different figures for the
saturation equivariance experiments.

Commands are found all the way below.
"""
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np


def plot_sat_base(paths, shift="Kernel"):
    x = np.load(paths[0])["hue"]
    y_nonorm_baseline = np.load(paths[0])["acc"] * 100
    y_nonorm_baseline_jitter = np.load(paths[1])["acc"] * 100
    y_nonorm = np.load(paths[2])["acc"] * 100
    y_nonorm_jitter = np.load(paths[3])["acc"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x, y_nonorm_baseline, label="Resnet-18")
    plt.plot(x, y_nonorm, label="CE-Resnet-18")
    plt.plot(x, y_nonorm_baseline_jitter, label="Resnet-18 + Jitter", ls="--") 
    plt.plot(x, y_nonorm_jitter, label="CE-Resnet-18 + Jitter", ls="--")

    plt.title(f"Saturation equivariant network trained in HSV space\n{shift} Shift | Flowers-102 dataset", fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(0, 100)
    plt.savefig(f"Sat_HSV_Fig9_satshift{shift}.jpg")


def plot_sat_jitters(paths_jit, shift="Kernel", filename="Sat_HSV_satshiftkernel_jitter.jpg"):
    x = np.load(paths_jit[0])["hue"]
    x_nonorm_020 = np.load(paths_jit[2])["hue"]
    y_nonorm_nojitter = np.load(paths_jit[0])["acc"] * 100
    y_nonorm_02 = np.load(paths_jit[1])["acc"] * 100
    y_nonorm_020 = np.load(paths_jit[2])["acc"] * 100
    y_nonorm_0100 = np.load(paths_jit[3])["acc"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x, y_nonorm_nojitter, label="None")
    plt.plot(x, y_nonorm_02, label="[0, 2]")
    plt.plot(x_nonorm_020, y_nonorm_020, label="[0, 20]")
    plt.plot(x, y_nonorm_0100, label="[0, 100]")

    plt.title(f"Effect of Jitter on Saturation Equivariant Network\n{shift} Shift | Flowers-102 dataset",
               fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=4, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(0, 100)
    plt.savefig(filename)


def plot_3d(paths_3d, saturations=50, rotations=37, num_shift=3, shift="Kernel", filename="HueSat_HSV_shiftkernel.jpg"):
    original_shape = (rotations, saturations)

    X = np.load(paths_3d)["hue"].reshape(original_shape)
    Y = np.load(paths_3d)["sat"].reshape(original_shape)
    Z = np.load(paths_3d)["acc"].reshape(original_shape) * 100

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9,5))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    ax.set_title(f"Hue and Saturation Equivariant Network trained in HSV Space\nFlowers-102 dataset [{num_shift} Hue and Sat Shifts on {shift}]", fontsize=15)
    ax.set_xlabel("Hue shift (°)", labelpad=10, fontsize=11)
    ax.set_xticks(ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    ax.set_ylabel("Saturation shift", labelpad=10, fontsize=11)
    ax.set_zlabel("Test accuracy (%)", labelpad=10, fontsize=11)

    # Customize the z axis.
    ax.set_zlim(0, 100)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)

    plt.savefig(filename)


def plot_sat_shift(paths, shift="Kernel"):
    x = np.load(paths[0])["hue"]
    y_nonorm_baseline = np.load(paths[0])["acc"] * 100
    y_nonorm_baseline_jitter = np.load(paths[1])["acc"] * 100
    y_nonorm = np.load(paths[2])["acc"] * 100
    y_nonorm_jitter = np.load(paths[3])["acc"] * 100
    x_nonorm_jitter = np.load(paths[3])["hue"]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x, y_nonorm_baseline, label="None")
    plt.plot(x, y_nonorm, label="3 shifts")
    plt.plot(x, y_nonorm_baseline_jitter, label="5 shifts")
    plt.plot(x_nonorm_jitter, y_nonorm_jitter, label="10 shifts")

    plt.title(f"Saturation equivariant network trained in HSV space\n{shift} Shift | Flowers-102 dataset", fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=4, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(0, 100)
    plt.savefig(f"Sat_HSV_Shifts{shift}.jpg")


paths_sat_shifts = [
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_10-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz"
]

paths_jit = [
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_2-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_20-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100-no_norm.npz",
]

paths_3d = [
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_and_sat_shift-sat_jitter_1_1-img_shift-no_norm_test-rot=25_test-sat=25.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_and_sat_shift-sat_jitter_1_1-no_norm_test-rot=25_test-sat=25.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_and_sat_shift-sat_jitter_1_1-img_shift-no_norm_test-rot=25_test-sat=25.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_and_sat_shift-sat_jitter_1_1-no_norm_test-rot=25_test-sat=25.npz"
]

paths_sat_base = [
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-img_shift-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_20-img_shift-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-img_shift-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_20-img_shift-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_20-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
    "CEConv/output/test_results/sat/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_20-no_norm.npz"
]

plot_3d(paths_3d[0], saturations=25, rotations=25, num_shift="No", shift="Image", filename="HueSat_HSV_shiftImgBase_noNorm.jpg")
plot_3d(paths_3d[1], saturations=25, rotations=25, num_shift="No", filename="HueSat_HSV_shiftKernelBase_noNorm.jpg")
plot_3d(paths_3d[2], saturations=25, rotations=25, shift="Image", filename="HueSat_HSV_shiftImage_noNorm.jpg")
plot_3d(paths_3d[3], saturations=25, rotations=25, filename="HueSat_HSV_shiftKernel_noNorm.jpg")
plot_sat_base(paths_sat_base[4:])
plot_sat_base(paths_sat_base, shift="Image")
plot_sat_shift(paths_sat_shifts)
plot_sat_jitters(paths_jit)

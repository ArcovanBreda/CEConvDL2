import os
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics


def plot_figure_9(checkpoints, hsv_test=False):
    from experiments.color_mnist.train_longtailed import PL_model, CustomDataset

    hue_values = [
    -0.5, -0.472222222222222, -0.444444444444444, -0.416666666666667, -0.388888888888889,
    -0.361111111111111, -0.333333333333333, -0.305555555555556, -0.277777777777778, -0.25,
    -0.222222222222222, -0.194444444444444, -0.166666666666667, -0.138888888888889, -0.111111111111111,
    -0.0833333333333334, -0.0555555555555556, -0.0277777777777778, 0, 0.0277777777777778,
    0.0555555555555556, 0.0833333333333333, 0.111111111111111, 0.138888888888889, 0.166666666666667,
    0.194444444444444, 0.222222222222222, 0.25, 0.277777777777778, 0.305555555555555, 0.333333333333333,
    0.361111111111111, 0.388888888888889, 0.416666666666667, 0.444444444444444, 0.472222222222222, 0.5
    ]

    model_names = ["No norm + Jitter", "RGB norm + Jitter",
                   "No norm", "RGB norm"]
    colors = [['darkorange', "--"], ['mediumblue', "--"], ['darkorange', "-"], ['mediumblue', "-"]]
    

    fig, ax = plt.subplots(figsize=(12, 6))
    for (checkpoint, name, color) in zip(checkpoints, model_names, colors):
        results = evaluate_classify(checkpoint, hsv_test=hsv_test)

        if torch.is_tensor(results["hue"][0]):
            results["hue"] = [i.item() for i in results["hue"]]
        
        plt.plot(results["hue"], results["acc"], color[1], label=name, color=color[0], linewidth=3)

    plt.title("Saturation Equivariant ResNet-18 trained in HSV space", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.ylim(top=0.99)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.savefig(f"sat_equiv_kernel_hsv_test={hsv_test}.png")


def evaluate_classify(path="/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt", ce_stages=None, seperable=True, width=None, hsv_test=False):
    import pytorch_lightning as pl
    from experiments.classification.datasets import get_dataset
    from pytorch_lightning import loggers as pl_loggers
    from experiments.classification.train import PL_model

    splitted = path.split("/")[-1].split("-")
    arch_rot = splitted[1].split("_")
    seed = splitted[5].split("_")[1]
    if len(seed.split(".")) != 1:
        seed = int(seed.split(".")[0])
    else:
        seed = int(seed)
    lab = True if "lab_space" in splitted else False
    hsv = True if "hsv_space" in splitted else False
    idx = None

    if "hue_and_sat_shift" in splitted:
        hue_shift, sat_shift = True, True
        idx = splitted.index("hue_and_sat_shift")
    else:
        hue_shift = True if "hue_shift" in splitted else False
        sat_shift = True if "sat_shift" in splitted else False
    
    if sat_shift:
        if idx is None:
            idx = splitted.index("sat_shift")

        min_max = splitted[idx + 1].split("_")[2:]
    
    sat_jitter = []
    for i in min_max:
        num = i.split(".")        
        tmp = [j for j in num if j != "pth" and j != "tar" and j != "ckpt"]

        sat_jitter.append(float(".".join(tmp)))

    no_norm = True if "no_norm" in splitted else False

    args = Namespace(seed = seed,
                    dataset = splitted[0],
                    jitter = float(".".join(splitted[3].split("_")[1:])),
                    grayscale= True if "grayscale" in splitted else False,
                    split= float(".".join(splitted[4].split("_")[1:])),
                    bs= 64,
                    architecture=arch_rot[0],
                    rotations=arch_rot[1],
                    groupcosetmaxpool= False if splitted[2] == "false" else True,
                    no_norm=no_norm,
                    ce_stages=ce_stages,
                    epochs=200,
                    resume=True,
                    normalize = not no_norm,
                    nonorm = no_norm,
                    separable=seperable,
                    width=width,
                    lab=lab,
                    hsv=hsv,
                    hue_shift=hue_shift,
                    sat_shift=sat_shift,
                    sat_jitter=sat_jitter,
                    hsv_test=hsv_test
    )

    run_name = "{}-{}_{}-{}-jitter_{}-split_{}-seed_{}".format(
        args.dataset,
        args.architecture,
        args.rotations,
        str(args.groupcosetmaxpool).lower(),
        str(args.jitter).replace(".", "_"),
        str(args.split).replace(".", "_"),
        args.seed,
    )
    if args.lab:
        run_name += "-lab_space"
    if args.hsv:
        run_name += "-hsv_space"
    if args.hue_shift and not args.sat_shift:
        run_name += "-hue_shift"
    if args.sat_shift and not args.hue_shift:
        run_name += "-sat_shift"
    if args.hue_shift and args.sat_shift:
        run_name += "-hue_and_sat_shift"
    if args.sat_jitter:
        run_name += f"-sat_jitter_{args.sat_jitter[0]}_{args.sat_jitter[1]}"
    if args.grayscale:
        run_name += "-grayscale"
    if not args.normalize:
        run_name += "-no_norm"
    if args.ce_stages is not None:
        run_name += "-{}_stages".format(args.ce_stages)

    args.model_name=run_name    
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="DL2 CEConv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )

    model = PL_model.load_from_checkpoint(path)
    model.hsv_test = hsv_test

    # overwrite model saturation settings if it didnt train with hsv_test=True
    if model.hsv_test:
        saturations = 50
        neg_sats = saturations // 2
        pos_sats = neg_sats - 1 + saturations % 2

        # In case of even saturations, consider 0 to be positive
        model.test_jitter = torch.concat((torch.linspace(-1, 0, neg_sats + 1)[:-1],
                                        torch.tensor([0]),
                                        torch.linspace(0, 1, pos_sats + 1)[1:])).type(torch.float32).to(model._device)
        model.test_acc_dict = {}
        for i in model.test_jitter:
            if model.args.dataset == "cifar10":
                model.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
            elif model.args.dataset == "flowers102":
                model.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=102)
            elif model.args.dataset == "stl10":
                model.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
            else:
                raise NotImplementedError

    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="DL2 CEConv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )

    trainer = pl.Trainer(accelerator='gpu', gpus=1, logger=mylogger)

    _, testloader = get_dataset(args)
    trainer.test(model, dataloaders=testloader, verbose=False)
    return model.results


# checkpoints = [
#     "/home/sabbring/CEConvDL2/CEConv/output/color_equivariance/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100-no_norm.pth.tar.ckpt",
#     "/home/sabbring/CEConvDL2/CEConv/output/color_equivariance/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100.pth.tar.ckpt",
#     "/home/sabbring/CEConvDL2/CEConv/output/color_equivariance/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.pth.tar.ckpt",
#     "/home/sabbring/CEConvDL2/CEConv/output/color_equivariance/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1.pth.tar.ckpt"
# ]

# # plot_figure_9(checkpoints, hsv_test=True)

# test_np = [
#     # "/home/sabbring/CEConvDL2/CEConv/output/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1.npz",
#     # "/home/sabbring/CEConvDL2/CEConv/output/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1.npz",
#     # "/home/sabbring/CEConvDL2/CEConv/output/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1.npz",
#     "/home/sabbring/CEConvDL2/CEConv/output/test_results/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1.npz" # broke the code (if u know u know)
# ]


# fig, ax = plt.subplots(figsize=(12, 6))

# for i in test_np:
#     file = np.load(i)

#     if torch.is_tensor(file["hue"][0]):
#         file["hue"] = [i.item() for i in file["hue"]]
        
#     plt.plot(file["hue"], file["acc"])

# plt.savefig(f"test_4.png")


import matplotlib.pyplot as plt
import numpy as np

# paths = [
#     "CEConv/output/test_results/maintest_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_2.npz"
# ]

# x = np.load(paths[0])["hue"]
# y_nonorm_baseline = np.load(paths[0])["acc"]
# # y_nonorm_baseline_jitter = np.load("")["acc"]
# y_nonorm = np.load(paths[1])["acc"]
# y_nonorm_jitter = np.load(paths[2])["acc"]

# fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot(x, y_nonorm_baseline, label="Resnet-18")
# # plt.plot(x, y_nonorm_baseline_jitter, label="Resnet-18", ls="--") #TODO train still
# plt.plot(x, y_nonorm, label="CE-Resnet-18")
# plt.plot(x, y_nonorm_jitter, label="CE-ResNet18 + Jitter", ls="--") # jitter is 0-2 in this case

# # plt.plot(x, y_norm_jitter, label="RGB norm + Jitter", ls="--")
# # plt.plot(x, y_nonorm_jitter, label="No norm + Jitter", ls="--")

# sat_shifts = (-1, -0.5, 0, 0.5, 1)
# plt.title(f"Saturation equivariant network trained and tested in HSV color space\nFlowers-102 dataset [Sat shift on kernel - {len(sat_shifts)} shifts {str(sat_shifts)}]", fontsize=22, pad=10)
# plt.ylabel("Test accuracy", fontsize=18)
# plt.yticks(fontsize=16,)
# plt.xlabel("Test-time saturation shift", fontsize=18)
# plt.xticks(fontsize=16,)
# plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
#         borderaxespad=0., ncol=3, fancybox=True, shadow=True,
#         columnspacing=0.7, handletextpad=0.2)
# plt.grid(axis="both")
# plt.ylim(0, 1)
# plt.savefig("Sat_HSV_Fig9_satshiftkernel.jpg")





# paths_jit = [
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm.npz",
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_2-no_norm.npz",
    
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100-no_norm.npz",
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1.npz",
#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_2.npz",

#     "CEConv/output/test_results/maintest_flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_0_100.npz"
# ]

# x = np.load(paths_jit[0])["hue"]
# y_nonorm_nojitter = np.load(paths_jit[0])["acc"]
# y_nonorm_02 = np.load(paths_jit[1])["acc"]
# # y_nonorm_020 = np.load(paths_jit[2])["acc"]
# y_nonorm_0100 = np.load(paths_jit[2])["acc"]

# y_norm_nojitter = np.load(paths_jit[3])["acc"]
# y_norm_02 = np.load(paths_jit[4])["acc"]
# # y_norm_020 = np.load(paths_jit[5])["acc"]
# y_norm_0100 = np.load(paths_jit[5])["acc"]

# fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot(x, y_nonorm_nojitter, label="None")
# plt.plot(x, y_nonorm_02, label="[0, 2]")
# # plt.plot(x, y_nonorm_020, label="[0, 20]")
# plt.plot(x, y_nonorm_0100, label="[0, 100]")

# plt.plot(x, y_norm_nojitter, label="None + RGB Norm", ls="--")
# plt.plot(x, y_norm_02, label="[0, 2] + RGB Norm", ls="--")
# # plt.plot(x, y_norm_020, label="[0, 20] + RGB Norm", ls="--")
# plt.plot(x, y_norm_0100, label="[0, 100] + RGB Norm", ls="--")

# sat_shifts = (-1, -0.5, 0, 0.5, 1)
# plt.title(f"Effect of Jitter on Saturation Equivariant Network\nFlowers-102 dataset [Sat shift on kernel - {len(sat_shifts)} shifts {str(sat_shifts)}]", fontsize=22, pad=10)
# plt.ylabel("Test accuracy", fontsize=18)
# plt.yticks(fontsize=16,)
# plt.xlabel("Test-time saturation shift", fontsize=18)
# plt.xticks(fontsize=16,)
# plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
#         borderaxespad=0., ncol=3, fancybox=True, shadow=True,
#         columnspacing=0.7, handletextpad=0.2)
# plt.grid(axis="both")
# plt.ylim(0, 1)
# plt.savefig("Sat_HSV_satshiftkernel_jitter.jpg")


import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



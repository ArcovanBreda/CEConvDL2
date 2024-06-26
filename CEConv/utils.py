import os
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import math
from skimage import color
import kornia
import skimage as ski
import torchvision
import torch
from PIL import Image
from matplotlib import cm


def plot_figure_2(data_dir, print_stats=False):
    # List of model namespaces
    from experiments.color_mnist.train_longtailed import PL_model, CustomDataset

    models = [
        Namespace(bs=256, 
                  test_bs=256, 
                  grayscale=False, 
                  jitter=0.0, 
                  epochs=1000, 
                  seed=0, 
                  planes=20, 
                  lr=0.001, 
                  wd=1e-05, 
                  rotations=1, 
                  groupcosetpool=False, 
                  separable=False, 
                  ce_layers=7, 
                  steps_per_epoch=5, 
                  model_name='longtailed-seed_0-rotations_1',
                  lab_test=False),
        Namespace(bs=256, 
                  test_bs=256, 
                  grayscale=False, 
                  jitter=0.0, 
                  epochs=1000, 
                  seed=0, 
                  planes=17, 
                  lr=0.001, 
                  wd=1e-05, 
                  rotations=3, 
                  groupcosetpool=False, 
                  separable=True, 
                  ce_layers=7, 
                  steps_per_epoch=5, 
                  model_name='longtailed-seed_0-rotations_3',
                  lab_test=False),
    ]
    

    train = CustomDataset(
            torch.load(os.path.join(data_dir, "colormnist_longtailed", "train.pt")),
            jitter=False,
            grayscale=False,
        )

    samples_per_class = torch.unique(train.tensors[1], return_counts=True)
    sort_idx = torch.argsort(samples_per_class[1], descending=True)
    samples_per_class = (samples_per_class[0][sort_idx], samples_per_class[1][sort_idx])

    labels = [j + str(i) for i in range(10) for j in ["R", "G", "B"]]
    labels = [labels[i] for i in sort_idx.numpy()]

    # Create empty arrays to store accuracy scores for each seed
    class_accs = [[],[]]

    # Loop over seeds 1 to 10
    for seed in range(1, 11):
        # Loop over models
        for model_args in models:
            # Update the seed value
            model_args.seed = seed
            model_args.model_name = f"longtailed-seed_{seed}-rotations_{model_args.rotations}-planes_{model_args.planes}"

            # Load the model
            class_acc = np.load(f"{data_dir}/output/longtailed/npz/{model_args.model_name}.npz")["class_acc"]

            # Append class accuracy to class_accs
            if model_args.rotations == 1 and model_args.planes == 20:
                class_accs[0].append(class_acc)
            elif model_args.rotations == 3:
                class_accs[1].append(class_acc)
            else:
                class_accs[2].append(class_acc)

    # Compute the average class accuracy and standard deviation
    avg_class_acc = np.mean(class_accs, axis=1)
    std_dev = np.std(class_accs, axis=1)

    avg_model_acc = np.mean(class_accs, axis=(1,2))
    std_dev_model = np.std(np.mean(class_accs, axis=2), axis=1)

    print(f"model performances:\n\tZ2CNN: {avg_model_acc[0]:.3f}+/-{std_dev_model[0]:.3f}\n\tCECNN: {avg_model_acc[1]:.3f}+/-{std_dev_model[1]:.3f}")

    avg_class_acc = avg_class_acc[:, sort_idx]
    std_dev = std_dev[:, sort_idx]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot average accuracy with standard deviation as error bars
    ax1.plot(labels, avg_class_acc[0, :]*100, label=f'Z2CNN ({np.mean(avg_class_acc[0, :])*100:.1f}%)', color='mediumblue', linewidth=3)
    ax1.fill_between(labels, (avg_class_acc[0, :] - std_dev[0, :])*100, (avg_class_acc[0, :] + std_dev[0, :])*100, color='mediumblue', alpha=0.2)

    ax1.plot(labels, avg_class_acc[1, :]*100, label=f'CECNN ({np.mean(avg_class_acc[1, :])*100:.1f}%)', color='forestgreen', linewidth=3)
    ax1.fill_between(labels, (avg_class_acc[1, :] - std_dev[1, :])*100, (avg_class_acc[1, :] + std_dev[1, :])*100, color='forestgreen', alpha=0.2)

    # Plot samples per class
    ax1.grid(axis='both')
    ax2 = ax1.twinx()

    ax2.bar(labels, samples_per_class[1].numpy() / sum(samples_per_class[1].numpy()), color="gray", alpha=0.3, width=0.65, label="Class frequency", zorder=0)
    ax2.set_ylabel('Class frequency', fontsize=18)

    ax1.set_xlabel('Class', fontsize=18)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=18)
    ax1.set_ylim(-0.05, 115)
    ax1.grid(axis='y')
    ax1.bar(0, 0, color="gray", alpha=0.3, label="Class frequency")
    ax1.legend(fontsize=18, loc='upper center',bbox_to_anchor=(0.5, 0.99), 
               borderaxespad=0., ncol=5, fancybox=True, shadow=True, 
               columnspacing=0.7, handletextpad=0.2)


    # Set font size for x-axis ticks
    ax1.tick_params(axis='x',
                    # rotation=45,
                    labelsize=14)

    # Set font size for y-axis ticks
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax1.grid(axis='y')
    ax2.grid(axis='x')

    plt.title('ColorMNIST - longtailed', fontsize=22)

    plt.show()

    return


def plot_figure_9(data_dir,verbose=False):
    from experiments.color_mnist.train_longtailed import PL_model, CustomDataset

    checkpoints = [f"{data_dir}/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_1-false-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt"] # DIS NIET GOED
    model_names = ["ResNet-18", "ResNet-18 + jitter",
                   "CE-ResNet-18 [Novel]", "CE-ResNet-18 + jitter [Novel]"]
    colors = [['mediumblue', "-"], ['mediumblue', "--"], ['darkorange', "-"], ['darkorange', "--"]]
    

    fig, ax = plt.subplots(figsize=(14, 7))
    print("model performances:")
    for (checkpoint, name, color) in zip(checkpoints, model_names, colors):
        results = evaluate_classify(checkpoint, verbose=False)
        print(f"\t\t {name}: {np.mean(results['acc']):.3f}")
        plt.plot(results["hue"], results["acc"]*100, color[1], label=f"{name} ({np.mean(results['acc'])*100:.1f}%)", color=color[0], linewidth=3)
        # plt.plot(hue_values, checkpoint, color[1], label=name, color=color[0], linewidth=3)

    plt.title("Flowers-102", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.ylim(top=99)
    plt.xlabel("Test-time hue shift (°)", fontsize=18)
    plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.show()


def evaluate_classify(path, 
                      ce_stages=None, seperable=True, width=None, verbose=True,
                      save_npz=True):
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
    no_norm = True if "no_norm" in splitted else False
    ce_stages = None if ce_stages == 0 else ce_stages
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
                    separable=seperable,
                    width=width,
                    lab_test=False,
                    lab=False,
                    value_jitter=(1,1),
                    sat_jitter=(1,1),
                    hsv=False
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
    if args.grayscale:
        run_name += "-grayscale"
    if not args.normalize:
        run_name += "-no_norm"
    if args.ce_stages is not None:
        run_name += "-{}_stages".format(args.ce_stages)

    args.model_name=run_name
    
    if save_npz:
        try:
            model = np.load(f'{"/".join(path.split("/")[:-1])}/npz/{run_name}.npz')
            if verbose:
                print("model already evaluated...")
            return model
        except Exception:
            if verbose:
                print(f'{"/".join(path.split("/")[:-1])}/npz/{run_name}.npz')
                print("Could not find path, evaluating...")
    
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="DL2 CEConv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )

    model = PL_model.load_from_checkpoint(path)

    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="DL2 CEConv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )

    trainer = pl.Trainer(accelerator='gpu', gpus=1, logger=mylogger)

    _, testloader = get_dataset(args)
    trainer.test(model, dataloaders=testloader, verbose=False)
    os.makedirs(f'{"/".join(path.split("/")[:-1])}/npz/', exist_ok=True)   
    np.savez(file=f'{"/".join(path.split("/")[:-1])}/npz/{run_name}', hue=model.results["hue"], acc=model.results["acc"])
    return model.results


def preprocess_data(stages):
    zero_stages, one_stage, two_stages, three_stages, four_stages = stages
    values = [0] # baseline is 0
    key = int(len(zero_stages) / 2)
    baseline = zero_stages[key]
    values.append((one_stage[key] - baseline) * 100)
    values.append((two_stages[key] - baseline) * 100)
    values.append((three_stages[key] - baseline) * 100)
    values.append((four_stages[key] - baseline) * 100)

    return values

def color_selective_datasets_plot(path):

    flower_names = [
        "flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz",
        "flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-1_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-2_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-3_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-4_stages.npz",
    ]
    flower = [np.load(f"{path}/{name}")["acc"] for name in flower_names]

    flower_jitter_names = [
        "flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0.npz",
        "flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-1_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-2_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-3_stages.npz",
        "flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-4_stages.npz",
    ]
    flower_jitter = [np.load(f"{path}/{name}")["acc"] for name in flower_jitter_names]

    stl_names = [
        "stl10-resnet18_1-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz",
        "stl10-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-1_stages.npz",
        "stl10-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-2_stages.npz",
        "stl10-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-3_stages.npz",
        "stl10-resnet18_3-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-4_stages.npz",
    ]
    stl = [np.load(f"{path}/{name}")["acc"] for name in stl_names]

    stl_jitter_names = [
        "stl10-resnet18_1-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz",
        "stl10-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-1_stages.npz",
        "stl10-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-2_stages.npz",
        "stl10-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-3_stages.npz",
        "stl10-resnet18_3-true-jitter_0_5-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1-4_stages.npz",
    ]
    stl_jitter = [np.load(f"{path}/{name}")["acc"] for name in stl_jitter_names]

    y_flower_jitter = preprocess_data(flower_jitter)
    y_stl_jitter = preprocess_data(stl_jitter)
    y_stl = preprocess_data(stl)
    y_flower = preprocess_data(flower)

    x = list(range(0, 5))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    plt.subplots_adjust(wspace=0.4, hspace=0.8, top=0.88, bottom=0.25001)

    ax1.plot(x, y_flower, '-D', color='#23c34a', linewidth=3, markersize=7)
    ax1.plot(x, y_stl, '-s', linewidth=3, markersize=7)
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.tick_params(axis='both',
                    # rotation=45,
                    labelsize=14)
    ax1.set_ylim(-6, 6)
    ax1.set_title('w/o color-jitter augmentation', fontsize=18)
    ax1.set_xlabel('Color Equivariance \nembedded up to stage', fontsize=18)
    ax1.set_ylabel(r'Accuracy improvement' + '\n' + r'(equivariant - vanilla)', fontsize=18)

    ax2.plot(x, y_flower_jitter, '-D', color='#23c34a', linewidth=3, markersize=7)
    ax2.plot(x, y_stl_jitter, '-s', linewidth=3, markersize=7)
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.tick_params(axis='both',
                    # rotation=45,
                    labelsize=14)
    ax2.set_ylim(-6, 6)
    ax2.set_title('w/ color-jitter augmentation', fontsize=18)
    ax2.set_xlabel('Color Equivariance \nembedded up to stage', fontsize=18)
    ax2.set_ylabel(r'Accuracy improvement' + '\n' + r'(equivariant - vanilla)', fontsize=18)

    label_names = ["Flowers102 (color sel. 0.70)", "STL10 (color sel. 0.38)"] 
    f.legend([y_flower, y_stl], labels=label_names, 
            loc="lower center", borderaxespad=0., ncol=1, fancybox=True, shadow=True, 
               columnspacing=0.7, handletextpad=0.2, fontsize=18) 

    f.suptitle("Color Equivariant stages vs accuracy improvement", fontsize=22)

    plt.show()


def hue_shifts_plot():
    x = np.load("output/test_results/hue_shift_RGB/flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["hue"]
    y_1 = np.load("./output/test_results/hue_shift_RGB/flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]
    y_5 = np.load("./output/test_results/hue_shift_RGB/flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]
    y_10 = np.load("./output/test_results/hue_shift_RGB/flowers102-resnet18_10-true-jitter_0_0-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]
    
    print("Max accuracy 1 rot: ", max(y_1))
    print("Max accuracy 5 rot: ", max(y_5))
    print("Max accuracy 10 rot: ", max(y_10))

    f, (ax1) = plt.subplots(1, 1, figsize=(14,7))

    ax1.plot(x, y_1, color='#2469c8', label=f'1 rotation ({np.mean(y_1)*100:.1f}%)', linewidth=3)
    ax1.plot(x, y_5, color='#d52320', label=f'5 rotations ({np.mean(y_5)*100:.1f}%)', linewidth=3)
    ax1.plot(x, y_10, color='#23c34a', label=f'10 rotations ({np.mean(y_10)*100:.1f}%)', linewidth=3)
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim(-0.05, 0.85)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax1.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70], fontsize=16)
    ax1.set_xticks([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45])
    ax1.set_xticklabels([-150, -100, -50, 0, 50, 100, 150], fontsize=16)
    ax1.set_title('Effect of hue rotations with reprojection', fontsize=22)
    ax1.set_xlabel('Test-time hue shift (°)', fontsize=18)
    ax1.set_ylabel('Test accuracy (%)', fontsize=18)

    ax1.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=3, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)

    plt.show()

def jitter_plot(path):
    x = np.load(f"{path}/flowers102-resnet18_1-false-jitter_0_2-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["hue"]
    resnet_0_2 = np.load(f"{path}/flowers102-resnet18_1-false-jitter_0_2-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]
    resnet_0_4 = np.load(f"{path}/flowers102-resnet18_1-false-jitter_0_4-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]

    ce_resnet_0_1 = np.load(f"{path}/flowers102-resnet18_3-true-jitter_0_1-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]
    ce_resnet_0_2 = np.load(f"{path}/flowers102-resnet18_3-true-jitter_0_2-split_1_0-seed_0-sat_jitter_1_1-val_jitter_1_1.npz")["acc"]

    ce_baseline = np.load(f"{path}/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0.npz")["acc"]

    f, (ax1) = plt.subplots(1, 1, figsize=(14,7))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    ax1.plot(x, resnet_0_2, color='#2469c8', label=f'ResNet-18 - jitter: 0.2 ({np.mean(resnet_0_2)*100:.1f}%)', linewidth=3)
    ax1.plot(x, resnet_0_4, color='#d52320', label=f'ResNet-18 - jitter: 0.4 ({np.mean(resnet_0_4)*100:.1f}%)', linewidth=3)
    plt.scatter(0, 0, 1, c="white", label="‎")
    ax1.plot(x, ce_baseline, color="tab:orange", label=f'CE-ResNet-18 - no jitter ({np.mean(ce_baseline)*100:.1f}%)', linewidth=3)
    ax1.plot(x, ce_resnet_0_1, color='#23c34a', label=f'CE-ResNet-18 - jitter 0.1 ({np.mean(ce_resnet_0_1)*100:.1f}%)', linewidth=3)
    ax1.plot(x, ce_resnet_0_2, color='#a120d5', label=f'CE-ResNet-18 - jitter 0.2 ({np.mean(ce_resnet_0_2)*100:.1f}%)', linewidth=3)

    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim(-0.1, 0.78)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax1.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70], fontsize=16)
    ax1.set_xticks([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45])
    ax1.set_xticklabels([-150, -100, -50, 0, 50, 100, 150], fontsize=16)
    ax1.set_title('Effect of jitter on ResNet-18 and CE-ResNet-18', fontsize=22)
    ax1.set_xlabel('Test-time hue shift (°)', fontsize=18)
    ax1.set_ylabel('Test accuracy (%)', fontsize=18)

    ax1.legend(fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.01),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)

    plt.show()


def hue_kernel(path):

    x = np.load(f"{path}/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["hue"]
    baseline = np.load(f"{path}/hue_shift_kernel/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
    baseline_jitter = np.load(f"{path}/hue_shift_kernel/hue_baseline_jitter_flowers102-resnet18_1-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
    hue_eq = np.load(f"{path}/hue_shift_kernel/hue_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]
    hue_eq_jitter = np.load(f"{path}/hue_shift_kernel/hue_flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"]

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.plot(x, baseline * 100, label=f"Resnet-18 ({np.mean(baseline)*100:.1f}%)", linewidth=3)
    plt.plot(x, baseline_jitter * 100, label=f"Resnet-18 + jitter ({np.mean(baseline_jitter)*100:.1f}%)", linewidth=3)
    plt.plot(x, hue_eq * 100, label=f"CE-Resnet-18 ({np.mean(hue_eq)*100:.1f}%)", ls="--", linewidth=3)
    plt.plot(x, hue_eq_jitter * 100, label=f"CE-Resnet-18 + jitter ({np.mean(hue_eq_jitter)*100:.1f}%)", ls="--", linewidth=3)

    plt.title("Hue equivariant network trained in HSV color space\n Kernel Shift | Flowers-102 dataset", fontsize=22)
    # plt.title("Hue equivariant network in HSV\n Kernel Shift - Flowers-102 dataset", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time hue shift (°)", fontsize=18)
    plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                borderaxespad=0., ncol=2, fancybox=True, shadow=True,
                columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(-0.5, 85)
    plt.show()

def hue_image(path):

    x = np.load(f"output/test_results/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["hue"]
    baseline = np.load(f"output/test_results/hue_shift_img/hue_baseline_flowers102-resnet18_1-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
    baseline_jitter = np.load(f"output/test_results/hue_shift_img/hue_baseline_jitter_flowers102-resnet18_1-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
    hue_eq = np.load(f"output/test_results/hue_shift_img/hue_flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]
    hue_eq_jitter = np.load(f"output/test_results/hue_shift_img/hue_flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-hsv_space-hue_shift-sat_jitter_1_1-val_jitter_1_1-img_shift-no_norm.npz")["acc"]

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.plot(x, baseline * 100, label=f"Resnet-18 ({np.mean(baseline)*100:.1f}%)", linewidth=3)
    plt.plot(x, baseline_jitter * 100, label=f"Resnet-18 + jitter ({np.mean(baseline_jitter)*100:.1f}%)", linewidth=3)
    plt.plot(x, hue_eq * 100, label=f"CE-Resnet-18 ({np.mean(hue_eq)*100:.1f}%)", ls="--", linewidth=3)
    plt.plot(x, hue_eq_jitter * 100, label=f"CE-Resnet-18 + jitter ({np.mean(hue_eq_jitter)*100:.1f}%)", ls="--", linewidth=3)

    plt.title("Hue equivariant network trained in HSV color space\n Image Shift | Flowers-102 dataset", fontsize=22)
    # plt.title("Hue equivariant network in HSV\n Kernel Shift - Flowers-102 dataset", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time hue shift (°)", fontsize=18)
    plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                borderaxespad=0., ncol=2, fancybox=True, shadow=True,
                columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(-0.5, 85)
    plt.show()

def plot_figure_22(data_dir, print_stats=False):
    # List of model namespaces
    from experiments.color_mnist.train_longtailed import PL_model, CustomDataset

    models = [
        Namespace(bs=256, 
                  test_bs=256, 
                  grayscale=False, 
                  jitter=0.0, 
                  epochs=1000, 
                  seed=0, 
                  planes=20, 
                  lr=0.001, 
                  wd=1e-05, 
                  rotations=1, 
                  groupcosetpool=False, 
                  separable=False, 
                  ce_layers=7, 
                  steps_per_epoch=5, 
                  model_name='longtailed-seed_0-rotations_1',
                  lab_test=False),
        Namespace(bs=256, 
                  test_bs=256, 
                  grayscale=False, 
                  jitter=0.0, 
                  epochs=1000, 
                  seed=0, 
                  planes=17, 
                  lr=0.001, 
                  wd=1e-05, 
                  rotations=3, 
                  groupcosetpool=False, 
                  separable=True, 
                  ce_layers=7, 
                  steps_per_epoch=5, 
                  model_name='longtailed-seed_0-rotations_3',
                  lab_test=False),
        Namespace(bs=256, 
                    test_bs=256, 
                    grayscale=False, 
                    jitter=0.0, 
                    epochs=1000, 
                    seed=0, 
                    planes=70, 
                    lr=0.001, 
                    wd=1e-05, 
                    rotations=1, 
                    groupcosetpool=False, 
                    separable=False, 
                    ce_layers=7, 
                    steps_per_epoch=5, 
                    model_name='longtailed-seed_0-rotations_1',
                    lab_test=False),
    ]

    train = CustomDataset(
            torch.load(os.path.join(data_dir, "colormnist_longtailed", "train.pt")),
            jitter=False,
            grayscale=False,
        )

    samples_per_class = torch.unique(train.tensors[1], return_counts=True)
    sort_idx = torch.argsort(samples_per_class[1], descending=True)
    samples_per_class = (samples_per_class[0][sort_idx], samples_per_class[1][sort_idx])

    labels = [j + str(i) for i in range(10) for j in ["R", "G", "B"]]
    labels = [labels[i] for i in sort_idx.numpy()]

    # Create empty arrays to store accuracy scores for each seed
    class_accs = [[],[], []]

    # Loop over seeds 1 to 10
    for seed in range(1, 11):
        # Loop over models
        for model_args in models:
            # Update the seed value
            model_args.seed = seed
            model_args.model_name = f"longtailed-seed_{seed}-rotations_{model_args.rotations}-planes_{model_args.planes}"

            # Load the model
            class_acc = np.load(f"{data_dir}/output/longtailed/npz/{model_args.model_name}.npz")["class_acc"]
            # Append class accuracy to class_accs
            if model_args.rotations == 1 and model_args.planes == 70:
                class_accs[0].append(class_acc)
            elif model_args.rotations == 3:
                class_accs[1].append(class_acc)
            else:
                class_accs[2].append(class_acc)

    # Compute the average class accuracy and standard deviation
    # print(class_accs)
    avg_class_acc = np.mean(class_accs, axis=1)
    std_dev = np.std(class_accs, axis=1)

    avg_model_acc = np.mean(class_accs, axis=(1,2))
    std_dev_model = np.std(np.mean(class_accs, axis=2), axis=1)

    print(f"model performances:\n\tZ2CNN-20: {avg_model_acc[2]:.3f}+/-{std_dev_model[2]:.3f}\n\tZ2CNN-70: {avg_model_acc[0]:.3f}+/-{std_dev_model[0]:.3f}\n\tCECNN: {avg_model_acc[1]:.3f}+/-{std_dev_model[1]:.3f}")

    avg_class_acc = avg_class_acc[:, sort_idx]
    std_dev = std_dev[:, sort_idx]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot average accuracy with standard deviation as error bars
    ax1.plot(labels, avg_class_acc[1, :]*100, label=f'CECNN ({np.mean(avg_class_acc[1, :])*100:.1f}%)', color='forestgreen', linewidth=3)
    ax1.fill_between(labels, (avg_class_acc[1, :] - std_dev[1, :])*100, (avg_class_acc[1, :] + std_dev[1, :])*100, color='forestgreen', alpha=0.2)

    ax1.plot(labels, avg_class_acc[2, :]*100, label=f'Z2CNN-20 ({np.mean(avg_class_acc[2, :])*100:.1f}%)', color='mediumblue', linewidth=3)
    ax1.fill_between(labels, (avg_class_acc[2, :] - std_dev[2, :])*100, (avg_class_acc[2, :] + std_dev[2, :])*100, color='mediumblue', alpha=0.2)

    ax1.plot(labels, avg_class_acc[0, :]*100, label=f'Z2CNN-70 ({np.mean(avg_class_acc[0, :])*100:.1f}%)', color='darkorange', linewidth=3)
    ax1.fill_between(labels, (avg_class_acc[0, :] - std_dev[0, :])*100, (avg_class_acc[0, :] + std_dev[0, :])*100, color='darkorange', alpha=0.2)

    # Plot samples per class
    ax1.grid(axis='both')
    ax2 = ax1.twinx()

    ax2.bar(labels, samples_per_class[1].numpy() / sum(samples_per_class[1].numpy()), color="gray", alpha=0.3, width=0.65, label="Class frequency", zorder=0)
    ax2.set_ylabel('Class frequency', fontsize=18)

    ax1.set_xlabel('Class', fontsize=18)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=18)
    ax1.set_ylim(-0.05, 119.99)
    ax1.grid(axis='y')
    ax1.bar(0, 0, color="gray", alpha=0.3, label="Class frequency")
    ax1.legend(fontsize=18, loc='upper center',bbox_to_anchor=(0.5, 0.99), 
               borderaxespad=0., ncol=2, fancybox=True, shadow=True, 
               columnspacing=0.7, handletextpad=0.2)


    # Set font size for x-axis ticks
    ax1.tick_params(axis='x',
                    # rotation=45,
                    labelsize=14)

    # Set font size for y-axis ticks
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax1.grid(axis='y')
    ax2.grid(axis='x')

    plt.title('ColorMNIST - longtailed + Large Z2CNN', fontsize=22)

    plt.show()


def lab(npz_folder="./output/test_results/lab_shift"):
    x = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["hue"]
    y_base = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
    y_base_jitter = np.load(f"{npz_folder}/flowers102-resnet18_1-false-jitter_0_5-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
    y_ce_jitter = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100
    y_ce_lab = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-lab_test-no_norm.npz")["acc"] * 100
    y_ce = np.load(f"{npz_folder}/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0-lab_space-sat_jitter_1_1-val_jitter_1_1-no_norm.npz")["acc"] * 100


    fig, ax = plt.subplots(figsize=(14, 7))

    plt.plot(x, y_base, label=f"Resnet-18 ({y_base.mean():.1f}%)", linewidth=3)
    plt.plot(
        x,
        y_base_jitter,
        label=f"Resnet-18 + Jitter ({y_base_jitter.mean():.1f}%)",
        ls="--", linewidth=3
    )
    plt.scatter(0, 0, 1, c="white", label="‎")


    plt.plot(x, y_ce, label=f"CE-Resnet-18 ({y_ce.mean():.1f}%)", linewidth=3)
    plt.plot(
        x,
        y_ce_jitter,
        label=f"CE-Resnet-18 + Jitter ({y_ce_jitter.mean():.1f}%)",
        ls="--", linewidth=3
    )

    plt.plot(
        x,
        y_ce_lab,
        label=f"CE-Resnet-18 + LAB shift ({y_ce_lab.mean():.1f}%)",
        ls="dotted", linewidth=3
    )

    plt.title(
        "LAB color space - Flowers-102 dataset",
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
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(0, 110)
    plt.show()


def value_image(path="./output/test_results"):
    npz_folder = path
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

    fig, ax = plt.subplots(figsize=(14, 7))

    plt.plot(x, y_base, label=f"Resnet-18 ({y_base.mean():.1f}%)", linewidth=3)
    plt.plot(
        x, y_base_jitter, label=f"Resnet-18 + Jitter ({y_base_jitter.mean():.1f}%)", ls="--", linewidth=3
    )

    plt.plot(x, y_ce, label=f"CE-Resnet-18 ({y_ce.mean():.1f}%)", linewidth=3)

    plt.plot(
        x, y_ce_jitter, label=f"CE-Resnet-18 + jitter ({y_ce_jitter.mean():.1f}%)", ls="--", linewidth=3
    )

    plt.title(
        "Value equivariant networks trained in HSV color space\nFlowers-102 - 5 shifts",
        fontsize=22,
    )
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(
        fontsize=16,
    )
    plt.xticks(fontsize=16)
    plt.xlabel("Test-time value shift", fontsize=18)

    plt.legend(fontsize=18, loc='upper center',bbox_to_anchor=(0.5, 0.99), 
               borderaxespad=0., ncol=2, fancybox=True, shadow=True, 
               columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(0, 78)
    print(x)
    plt.show()

def get_3Dmatrix(rotations):
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return np.asarray(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ]
    )


def get_2Dmatrix(angle):
    # Rotation matrix
    angle = (math.pi / 180) * angle
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )


def get_3Dhuematrix(num_rotations):
    # Rotation matrix
    angle = 2 * math.pi / num_rotations
    return np.asarray(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
    )

def colorspace_comparison():
    PIL_img = Image.open("../blogpost_imgs/flower_example.jpg")  # Range 0-255
    tensor_img = torchvision.transforms.functional.to_tensor(PIL_img)
    lab_tensor = kornia.color.rgb_to_lab(tensor_img)
    im = ski.io.imread("../blogpost_imgs/flower_example.jpg")  # Range 0-255
    hsv = color.rgb2hsv(im)  # Range 0-1
    lab = color.rgb2lab(im)  # Range 01, -127-128, -128, -127

    shifts = 9
    delta = 1 / shifts
    angle_delta = 360 / shifts
    matrix = get_3Dmatrix(shifts)
    lab_matrix = get_3Dhuematrix(shifts)

    fig, axs = plt.subplots(3, shifts, figsize=(15, 5))
    for c in range(3):
        for i in range(shifts):
            if c == 0:
                shift = i * delta
                np_image = hsv.copy()
                np_image[:, :, 0] -= shift
                np_image[:, :, 0] = np_image[:, :, 0] % 1
                axs[c, i].imshow(color.hsv2rgb(np_image))
                if i == shifts//2:
                    axs[c, i].set_title("HSV space", fontsize=20)
                axs[c, i].axis("off")
            elif c == 1:
                np_image = im
                # print(np_image[:, :, 1:].max(), np_image[:, :, 1:].min())
                np_image = np_image @ np.linalg.matrix_power(matrix, i)
                np_image = np_image / 255
                np_image = np_image.clip(0, 1)
                axs[c, i].imshow(np_image)
                if i == shifts//2:
                    axs[c, i].set_title("RGB space", fontsize=20)
                axs[c, i].axis("off")
            elif c == 2:
                np_image = lab_tensor.clone().float().moveaxis(0, -1)
                np_image = torch.einsum(
                    "whc, cr->whr",
                    np_image,
                    torch.matrix_power(torch.from_numpy(lab_matrix), i).float(),
                )
                axs[c, i].imshow(
                    np.moveaxis(
                        kornia.color.lab_to_rgb(np_image.moveaxis(-1, 0)).numpy(), 0, -1
                    )
                )
                if i == shifts//2:
                    axs[c, i].set_title("LAB space", fontsize=20)
                axs[c, i].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("Hue shift in HSV space / RGB space / LAB space", fontsize=24)
    plt.show()


def plot_sat_base(paths, shift="Kernel", dataset="Flowers-102", top=85):
    x = np.load(paths[0])["hue"]
    y_nonorm_baseline = np.load(paths[0])["acc"] * 100
    y_nonorm_baseline_jitter = np.load(paths[1])["acc"] * 100
    y_nonorm = np.load(paths[2])["acc"] * 100
    y_nonorm_jitter = np.load(paths[3])["acc"] * 100

    if not dataset == "Camelyon17":
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig, ax = plt.subplots(figsize=(14, 9))

    plt.plot(x, y_nonorm_baseline, label=f"Resnet-18 ({np.mean(y_nonorm_baseline):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm, label=f"CE-Resnet-18 ({np.mean(y_nonorm):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm_baseline_jitter, label=f"Resnet-18 + Jitter ({np.mean(y_nonorm_baseline_jitter):.1f}%)", ls="--", linewidth=3)  
    plt.plot(x, y_nonorm_jitter, label=f"CE-Resnet-18 + Jitter ({np.mean(y_nonorm_jitter):.1f}%)", ls="--", linewidth=3)

    plt.title(f"Saturation equivariant network trained in HSV space\n{shift} Shift | {dataset} dataset", fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)

    if not dataset == "Camelyon17":
        plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                    borderaxespad=0., ncol=2, fancybox=True, shadow=True,
                    columnspacing=0.7, handletextpad=0.2)
    else:
        plt.plot(x, np.array([50.] * x.shape[0]), label=f"Random baseline", linewidth=3, ls=":")
        plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                borderaxespad=0., ncol=3, fancybox=True, shadow=True,
                columnspacing=0.7, handletextpad=0.2)
        fig.tight_layout()

    plt.grid(axis="both")
    plt.ylim(top=top)
    plt.savefig(f"Sat_HSV_Fig9_satshift{shift}_{dataset}.jpg")


def plot_sat_jitters(paths_jit, shift="Kernel", filename="Sat_HSV_satshiftkernel_jitter.jpg"):
    x = np.load(paths_jit[0])["hue"]
    x_nonorm_020 = np.load(paths_jit[2])["hue"]
    y_nonorm_nojitter = np.load(paths_jit[0])["acc"] * 100
    y_nonorm_02 = np.load(paths_jit[1])["acc"] * 100
    y_nonorm_020 = np.load(paths_jit[2])["acc"] * 100
    y_nonorm_0100 = np.load(paths_jit[3])["acc"] * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.plot(x, y_nonorm_nojitter, label=f"None ({np.mean(y_nonorm_nojitter):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm_02, label=f"[0, 2] ({np.mean(y_nonorm_02):.1f}%)",  linewidth=3)
    plt.plot(x_nonorm_020, y_nonorm_020, label=f"[0, 20] ({np.mean(y_nonorm_020):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm_0100, label=f"[0, 100] ({np.mean(y_nonorm_0100):.1f}%)",  linewidth=3)

    plt.title(f"Effect of Jitter on Saturation Equivariant Network\n{shift} Shift | Flowers-102 dataset",
               fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                borderaxespad=0., ncol=2, fancybox=True, shadow=True,
                columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(top=90)
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

    if num_shift > 0:
        ax.set_title(f"Hue and Saturation Equivariant Network trained in HSV Space\nFlowers-102 dataset [{num_shift} Hue and Sat Shifts on {shift}]", fontsize=15)
    else:
        ax.set_title(f"Hue and Saturation Equivariant Network trained in HSV Space\nFlowers-102 dataset [Baseline for {shift} Shifts]", fontsize=15)

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

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.plot(x, y_nonorm_baseline, label=f"None ({np.mean(y_nonorm_baseline):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm_baseline_jitter, label=f"3 shifts ({np.mean(y_nonorm_baseline_jitter):.1f}%)",  linewidth=3)
    plt.plot(x, y_nonorm, label=f"5 shifts ({np.mean(y_nonorm):.1f}%)",  linewidth=3)
    plt.plot(x_nonorm_jitter, y_nonorm_jitter, label=f"10 shifts ({np.mean(y_nonorm_jitter):.1f}%)",  linewidth=3)

    plt.title(f"Saturation equivariant network trained in HSV space\n{shift} Shift | Flowers-102 dataset", fontsize=22, pad=10)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    plt.xticks(fontsize=16,)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                borderaxespad=0., ncol=2, fancybox=True, shadow=True,
                columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.ylim(top=85)
    plt.savefig(f"Sat_HSV_Shifts{shift}.jpg")

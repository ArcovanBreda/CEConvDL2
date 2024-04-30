import os
from experiments.color_mnist.train_longtailed import PL_model, CustomDataset
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np


def plot_figure_2(data_dir, print_stats=False):
    # List of model namespaces
    models = [
        Namespace(bs=256, 
                  test_bs=256, 
                  grayscale=True, 
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
                  model_name='longtailed-seed_0-rotations_1'),
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
                  model_name='longtailed-seed_0-rotations_3'),
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
            model_args.model_name = f"longtailed-seed_{seed}-rotations_{model_args.rotations}"

            # Load the model
            model = PL_model(model_args)
            model.load_model()

            # Append class accuracy to class_accs
            if model_args.rotations == 1:
                class_accs[0].append(model.class_acc)
            elif model_args.rotations == 3:
                class_accs[1].append(model.class_acc)
            else:
                class_accs[2].append(model.class_acc)

    # Compute the average class accuracy and standard deviation
    avg_class_acc = np.mean(class_accs, axis=1)
    std_dev = np.std(class_accs, axis=1)
    
    avg_class_acc = avg_class_acc[:, sort_idx]
    std_dev = std_dev[:, sort_idx]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot average accuracy with standard deviation as error bars
    ax1.plot(labels, avg_class_acc[0, :], label='Z2CNN (grayscale)', color='darkorange')
    ax1.fill_between(labels, avg_class_acc[0, :] - std_dev[0, :], avg_class_acc[0, :] + std_dev[0, :], color='darkorange', alpha=0.2)
    
    ax1.plot(labels, avg_class_acc[1, :], label='CECNN', color='forestgreen')
    ax1.fill_between(labels, avg_class_acc[1, :] - std_dev[1, :], avg_class_acc[1, :] + std_dev[1, :], color='forestgreen', alpha=0.2)


    # Plot samples per class
    ax1.grid(axis='both')
    ax2 = ax1.twinx()

    ax2.bar(labels, samples_per_class[1].numpy(), color="gray", alpha=0.3, width=0.65, label="Class frequency", zorder=0)
    ax2.set_ylabel('Class frequency', fontsize=18)

    ax1.set_xlabel('Class', fontsize=18)
    ax1.set_ylabel('Test Accuracy', fontsize=18)
    ax1.set_ylim(-0.05, 1.15)
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
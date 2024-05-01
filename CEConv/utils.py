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


def plot_figure_9():
    hue_values = [
    -0.5, -0.472222222222222, -0.444444444444444, -0.416666666666667, -0.388888888888889,
    -0.361111111111111, -0.333333333333333, -0.305555555555556, -0.277777777777778, -0.25,
    -0.222222222222222, -0.194444444444444, -0.166666666666667, -0.138888888888889, -0.111111111111111,
    -0.0833333333333334, -0.0555555555555556, -0.0277777777777778, 0, 0.0277777777777778,
    0.0555555555555556, 0.0833333333333333, 0.111111111111111, 0.138888888888889, 0.166666666666667,
    0.194444444444444, 0.222222222222222, 0.25, 0.277777777777778, 0.305555555555555, 0.333333333333333,
    0.361111111111111, 0.388888888888889, 0.416666666666667, 0.444444444444444, 0.472222222222222, 0.5
    ]

    accuracy_values = [
        0.0627999976277351, 0.0599000006914139, 0.0586000010371208, 0.0575000010430813, 0.0586000010371208,
        0.0604999996721745, 0.0631000027060509, 0.0654999986290932, 0.0674000009894371, 0.068000003695488,
        0.0676999986171722, 0.0658000037074089, 0.0627999976277351, 0.0599000006914139, 0.0586000010371208,
        0.0575000010430813, 0.0586000010371208, 0.0604999996721745, 0.0631000027060509, 0.0654999986290932,
        0.0674000009894371, 0.068000003695488, 0.0676999986171722, 0.0658000037074089, 0.0627999976277351,
        0.0599000006914139, 0.0586000010371208, 0.0575000010430813, 0.0586000010371208, 0.0604999996721745,
        0.0631000027060509, 0.0654999986290932, 0.0674000009894371, 0.068000003695488, 0.0676999986171722,
        0.0658000037074089, 0.0627999976277351
    ]

    hue_values_2 = [
    -0.5, -0.472222222222222, -0.444444444444444, -0.416666666666667, -0.388888888888889,
    -0.361111111111111, -0.333333333333333, -0.305555555555556, -0.277777777777778, -0.25,
    -0.222222222222222, -0.194444444444444, -0.166666666666667, -0.138888888888889, -0.111111111111111,
    -0.0833333333333334, -0.0555555555555556, -0.0277777777777778, 0, 0.0277777777777778,
    0.0555555555555556, 0.0833333333333333, 0.111111111111111, 0.138888888888889, 0.166666666666667,
    0.194444444444444, 0.222222222222222, 0.25, 0.277777777777778, 0.305555555555555, 0.333333333333333,
    0.361111111111111, 0.388888888888889, 0.416666666666667, 0.444444444444444, 0.472222222222222, 0.5
    ]

    accuracy_values_2 = [
    0.0784000009298325, 0.0825000032782555, 0.0848999992012978, 0.0886000022292137, 0.0929000005125999,
    0.098300002515316, 0.102899998426437, 0.105800002813339, 0.107799999415874, 0.107500001788139,
    0.106899999082088, 0.106100000441074, 0.106299996376038, 0.105400003492832, 0.103399999439716,
    0.101800002157688, 0.0997999981045723, 0.0979999974370003, 0.0939000025391579, 0.0900000035762787,
    0.0847999975085259, 0.0789000019431114, 0.0753000006079674, 0.0714000016450882, 0.0676999986171722,
    0.065200001001358, 0.0621999986469746, 0.0615000016987324, 0.0612000003457069, 0.0623000003397465,
    0.0653000026941299, 0.068400003015995, 0.0706999972462654, 0.0732999965548515, 0.0746999979019165,
    0.0760999992489815, 0.0784000009298325
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(hue_values_2, accuracy_values_2, label="ResNet-18")
    plt.plot(hue_values, accuracy_values, label="CE-ResNet-18 [Novel]")
    plt.title("Flowers-102", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.xlabel("Test-time hue shift (°)", fontsize=18)
    plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=5, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.show()

    
plot_figure_9()
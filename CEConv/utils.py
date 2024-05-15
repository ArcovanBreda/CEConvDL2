import os
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np


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
            model_args.model_name = f"longtailed-seed_{seed}-rotations_{model_args.rotations}"

            # Load the model
            class_acc = np.load(f"{data_dir}/output/longtailed/npz/{model_args.model_name}.npz")["class_acc"]

            # Append class accuracy to class_accs
            if model_args.rotations == 1:
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
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot average accuracy with standard deviation as error bars
    ax1.plot(labels, avg_class_acc[0, :], label='Z2CNN', color='mediumblue')
    ax1.fill_between(labels, avg_class_acc[0, :] - std_dev[0, :], avg_class_acc[0, :] + std_dev[0, :], color='mediumblue', alpha=0.2)

    ax1.plot(labels, avg_class_acc[1, :], label='CECNN', color='forestgreen')
    ax1.fill_between(labels, avg_class_acc[1, :] - std_dev[1, :], avg_class_acc[1, :] + std_dev[1, :], color='forestgreen', alpha=0.2)

    # Plot samples per class
    ax1.grid(axis='both')
    ax2 = ax1.twinx()

    ax2.bar(labels, samples_per_class[1].numpy() / sum(samples_per_class[1].numpy()), color="gray", alpha=0.3, width=0.65, label="Class frequency", zorder=0)
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


def plot_figure_9(data_dir,verbose=False):
    from experiments.color_mnist.train_longtailed import PL_model, CustomDataset

    checkpoints = [f"{data_dir}/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_1-false-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
                   f"{data_dir}/output/classification/flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt"]
    
    model_names = ["ResNet-18", "ResNet-18 + jitter",
                   "CE-ResNet-18 [Novel]", "CE-ResNet-18 + jitter [Novel]"]
    colors = [['mediumblue', "-"], ['mediumblue', "--"], ['darkorange', "-"], ['darkorange', "--"]]
    

    fig, ax = plt.subplots(figsize=(12, 6))
    print("model performances:")
    for (checkpoint, name, color) in zip(checkpoints, model_names, colors):
        results = evaluate_classify(checkpoint, verbose=False)
        print(f"\t\t {name}: {np.mean(results['acc']):.3f}")
        plt.plot(results["hue"], results["acc"], color[1], label=name, color=color[0], linewidth=3)
        # plt.plot(hue_values, checkpoint, color[1], label=name, color=color[0], linewidth=3)

    plt.title("Flowers-102", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.ylim(top=0.99)
    plt.xlabel("Test-time hue shift (°)", fontsize=18)
    plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    plt.show()


def evaluate_classify(path="/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt", 
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

# print(evaluate_classify("/home/arco/Downloads/Master AI/CEConvDL2/CEConv/output/flowers102-resnet18_10-true-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt"))
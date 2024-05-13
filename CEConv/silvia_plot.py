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

    # checkpoints = ["/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
    #                "/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_1-false-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt",
    #                "/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_3-true-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt",
    #                "/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_3-true-jitter_0_5-split_1_0-seed_0.pth.tar.ckpt"]
    
#     checkpoints = [[0.04602374508976936, 0.045047976076602936, 0.043584324419498444, 0.04651162773370743, 0.04586111381649971, 0.04228329658508301, 0.036916572600603104, 0.032037731260061264, 0.028947796672582626, 0.030248820781707764, 0.034802407026290894, 0.04683688282966614, 0.07058057934045792, 0.12701252102851868, 0.21092860400676727, 0.3407058119773865, 0.5132541656494141, 0.6563668847084045, 0.7025532722473145, 0.6407545804977417, 0.4532444179058075, 0.2693120837211609, 0.1771019697189331, 0.12408521771430969, 0.09773947298526764, 0.07513416558504105, 0.06017238646745682, 0.04976418986916542, 0.043584324419498444, 0.03984387591481209, 0.04212066903710365, 0.04521060362458229, 0.045373231172561646, 0.04439746215939522, 0.042933810502290726, 0.042933810502290726, 0.04602374508976936],
#                    [0.7154008746147156, 0.714913010597229, 0.7142624855041504, 0.7163766622543335, 0.7180029153823853, 0.7186534404754639, 0.7199544906616211, 0.720767617225647, 0.7241827845573425, 0.7212554812431335, 0.7219060063362122, 0.7210928797721863, 0.7160513997077942, 0.7168645262718201, 0.714913010597229, 0.7142624855041504, 0.7201170921325684, 0.7217433452606201, 0.7248333096504211, 0.7243454456329346, 0.7232069969177246, 0.7219060063362122, 0.7222312688827515, 0.7209302186965942, 0.7175150513648987, 0.7158887386322021, 0.7152382731437683, 0.7163766622543335, 0.717677652835846, 0.7189787030220032, 0.7225565314292908, 0.7204423546791077, 0.7209302186965942, 0.7202796936035156, 0.7204423546791077, 0.7163766622543335, 0.7154008746147156],
#                    [0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.27402830123901367, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.2741909325122833, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.27402830123901367, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037],
#                    [0.7536184787750244, 0.7528053522109985, 0.751341700553894, 0.7550821304321289, 0.7557326555252075, 0.7606114745140076, 0.7612619996070862, 0.760123610496521, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.7544316053390503, 0.7536184787750244, 0.7528053522109985, 0.751341700553894, 0.7550821304321289, 0.7557326555252075, 0.7609367370605469, 0.7612619996070862, 0.759960949420929, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.754269003868103, 0.7536184787750244, 0.7528053522109985, 0.7515043020248413, 0.7550821304321289, 0.7557326555252075, 0.7607741355895996, 0.7612619996070862, 0.760123610496521, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.7544316053390503, 0.7536184787750244]
# ]
    model_names = ["No norm + Jitter"]
    # , "RGB norm + Jitter", "No norm", "RGB norm"]
    colors = [['darkorange', "--"]]
            #   ['mediumblue', "--"], ['darkorange', "-"], ['mediumblue', "-"]]
    

    fig, ax = plt.subplots(figsize=(12, 6))
    for (checkpoint, name, color) in zip(checkpoints, model_names, colors):
        results = evaluate_classify(checkpoint, hsv_test=hsv_test)

        if torch.is_tensor(results["hue"][0]):
            results["hue"] = [i.item() for i in results["hue"]]
        
        plt.plot(results["hue"], results["acc"], color[1], label=name, color=color[0], linewidth=3)
        # plt.plot(hue_values, checkpoint, color[1], label=name, color=color[0], linewidth=3)

    plt.title("Saturation Equivariant ResNet-18 trained in HSV space", fontsize=22)
    plt.ylabel("Test accuracy (%)", fontsize=18)
    plt.yticks(fontsize=16,)
    plt.ylim(top=0.99)
    plt.xlabel("Test-time saturation shift", fontsize=18)
    # plt.xticks(fontsize=16,ticks=[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],labels=["-150", "-100", "-50", "0", "50", "100", "150" ])
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0., ncol=2, fancybox=True, shadow=True,
            columnspacing=0.7, handletextpad=0.2)
    plt.grid(axis="both")
    # plt.show()
    plt.savefig(f"silvia_plot={hsv_test}.png")


def evaluate_classify(path="/home/arco/Downloads/Master AI/CEConvDL2/output/classification/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt", ce_stages=None, seperable=True, width=None, hsv_test=False):
    import pytorch_lightning as pl
    from experiments.classification.silvia_datasets import get_dataset
    from pytorch_lightning import loggers as pl_loggers
    from experiments.classification.silvia_train import PL_model

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

    print("DEBUG SAT JITTER EXTRACTION: ", sat_jitter)

    # sat_jitter = [float(i.split(".")[0]) for i in sat_jitter] #dont use float sat_jitter values for now #TODO

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
                    hsv_test=hsv_test #TODO this wasnt necessary... although...??? since its here now might as well keep it...
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
        entity="rens-uva-org",
        project="dante_dl2",
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
        entity="rens-uva-org",
        project="dante_dl2",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )

    trainer = pl.Trainer(accelerator='gpu', gpus=1, logger=mylogger)

    _, testloader = get_dataset(args)
    trainer.test(model, dataloaders=testloader, verbose=False)
    return model.results


checkpoints = ["/home/dante/dl2/CEConvDL2/OUT/color_equivariance/classification/flowers102-resnet18_5-true-jitter_0_0-split_1_0-seed_0-hsv_space-sat_shift-sat_jitter_1_1-no_norm-2024-05-10_20-26-52.pth.tar.ckpt"]
plot_figure_9(checkpoints, hsv_test=True)

# test = torch.linspace(0, 1, 50)
# print(test.shape)
# for i in test:
#     print(i, i.unsqueeze(0)[:, None,None], i.unsqueeze(0)[:, None,None].shape)
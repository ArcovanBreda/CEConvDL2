"""Image classification experiments for Color Equivariant Convolutional Networks."""

import argparse
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchinfo import summary
from torchvision.transforms.functional import adjust_hue, adjust_saturation

from experiments.classification.datasets import get_dataset, normalize, lab2rgb, rgb2lab, hsv2rgb, rgb2hsv
from models.resnet import ResNet18, ResNet44
from models.resnet_hybrid import HybridResNet18, HybridResNet44
import matplotlib.pyplot as plt

class PL_model(pl.LightningModule):
    def __init__(self, args) -> None:
        super(PL_model, self).__init__()

        # Logging.
        self.save_hyperparameters()
        self.lab = args.lab
        self.args = args
        self.hsv = args.hsv
        self.hsv_test = args.hsv_test
        self.normalize = args.normalize
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store predictions and ground truth for computing confusion matrix.
        self.preds = torch.tensor([])
        self.gts = torch.tensor([])

        # Store accuracy metrics for logging.
        if args.dataset == "cifar10":
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        elif args.dataset == "flowers102":
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=102)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=102)
        elif args.dataset == "stl10":
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        else:
            raise NotImplementedError

        # Store accuracy metrics for testing.
        self.test_acc_dict = {}
        self.test_rotations = 37
        self.test_jitter = np.linspace(-0.5, 0.5, self.test_rotations)
        self.hue_shift, self.sat_shift = False, False
        self.lab_test = args.lab_test

        # Hue shift in lab space
        if self.lab_test:
            angle_delta =  2 * math.pi / self.test_rotations
            self.lab_angle_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(angle_delta), -math.sin(angle_delta)],
                [0, math.sin(angle_delta), math.cos(angle_delta)],
            ]
        )

            for i in self.test_jitter:
                if args.dataset == "cifar10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                elif args.dataset == "flowers102":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=102)
                elif args.dataset == "stl10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                else:
                    raise NotImplementedError
        # Hue shift in HSV space
        elif args.hue_shift and not args.sat_shift:
            self.hue_shift = True

            for i in self.test_jitter:
                if args.dataset == "cifar10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                elif args.dataset == "flowers102":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=102)
                elif args.dataset == "stl10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                else:
                    raise NotImplementedError
        # Saturation shift
        elif args.sat_shift and not args.hue_shift:
            self.sat_shift = True

            # In HSV space
            if self.hsv_test:
                saturations = 50
                neg_sats = saturations // 2
                pos_sats = neg_sats - 1 + saturations % 2

                # In case of even saturations, consider 0 to be positive
                self.test_jitter = torch.concat((torch.linspace(-1, 0, neg_sats + 1)[:-1],
                                                torch.tensor([0]),
                                                torch.linspace(0, 1, pos_sats + 1)[1:])).type(torch.float32).to(self._device)
            else:
                # When img is in RGB instead of HSV
                self.test_jitter = np.append(np.linspace(0, 1, 25, endpoint=False), np.arange(1, 10, 1, dtype=int))
            
            for i in self.test_jitter:
                if args.dataset == "cifar10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                elif args.dataset == "flowers102":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=102)
                elif args.dataset == "stl10":
                    self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy(task="multiclass", num_classes=10)
                else:
                    raise NotImplementedError
        elif args.sat_shift and args.hue_shift:
            self.hue_shift, self.sat_shift = True, True
            raise NotImplementedError #TODO jitter for combinations of saturation and hue shifts
            #TODO keep hsv_test in mind here as well, because for saturation we cannot convert HSV -> RGB -> HSV

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Model definition.
        if args.ce_stages is not None:
            architectures = {"resnet18": HybridResNet18, "resnet44": HybridResNet44}
        else:
            architectures = {"resnet18": ResNet18, "resnet44": ResNet44}
        assert args.architecture in architectures.keys(), "Model not supported."
        kwargs = {
            "rotations": args.rotations,
            "groupcosetmaxpool": args.groupcosetmaxpool,
            "separable": args.separable,
            "width": args.width,
            "num_classes": len(args.classes),
            "ce_stages": args.ce_stages,
            "lab_space": args.lab,
            "hsv_space": args.hsv,
            "sat_shift": args.sat_shift,
            "hue_shift": args.hue_shift,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = architectures[args.architecture](**kwargs)

        # Print model summary.
        resolution = 32 if args.architecture == "resnet44" else 224
        summary(self.model, (2, 3, resolution, resolution), device="cpu")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        parser.add_argument("--groupcosetmaxpool", action="store_true")
        parser.add_argument("--architecture", default="resnet44", type=str)
        parser.add_argument("--rotations", type=int, default=1, help="no. hue rot.")
        parser.add_argument("--separable", action="store_true", help="separable conv")
        parser.add_argument("--width", type=int, default=None, help="network width")
        parser.add_argument("--ce_stages", type=int, default=None, help="ce res stages")
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.wd
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch
        # Normalize images.
        if args.normalize:
            x = normalize(x, grayscale=self.args.grayscale or self.args.rotations > 1, lab=self.lab, hsv=self.hsv)

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        # Logging.
        batch_acc = self.train_acc(y_pred, y)
        self.log("train_acc_step", batch_acc)
        self.log("train_loss_step", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch

        # Normalize images.
        if args.normalize:
            x = normalize(x, grayscale=self.args.grayscale or self.args.rotations > 1, lab=self.lab, hsv=self.hsv)

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        # Logging.
        self.test_acc.update(y_pred.detach().cpu(), y.cpu())

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        self.log("test_acc_epoch", self.test_acc.compute())
        self.test_acc.reset()

    def test_step(self, batch, batch_idx) -> None:
        x_org, y = batch
        for i in self.test_jitter:
            if self.lab and not self.lab_test:
                x = lab2rgb.forward(None,x_org.clone())
            elif self.hsv and not self.hsv_test:
                x_org = hsv2rgb.forward(None, x_org.clone())
            else:
                x = x_org.clone()

            if self.lab_test:
                # Apply hue shift in lab space
                # Convert torch hue angle to applications of rotation matrix
                times = 360 - (i*(180/0.5))
                times = times / (360/self.test_rotations)
                w=x.shape[2]
                h=x.shape[3]
                # Apply shift to batch
                matrix = torch.matrix_power(self.lab_angle_matrix, int(times)).repeat(x.shape[0],1,1)
                x = torch.bmm(matrix.cuda(), x.reshape((x.shape[0], 3, -1))).reshape((x.shape[0], 3, w, h))
            elif self.hue_shift and not self.sat_shift:
                # Apply hue shift with pytorch's function; image must be in RGB so provide --hsv.
                x = adjust_hue(x, i)
            elif self.sat_shift and not self.hue_shift:
                # Apply saturation shift.
                if self.hsv_test:
                    add_val = i.unsqueeze(0)[:, None,None] # 1, 1, 1
                    w = x.shape[2]
                    h = x.shape[3]
                    x = x.reshape((x.shape[0], 3, -1)) # B, C, H*W
                    x[:, 1:2, :] += add_val # add to saturation channel
                    x[:, 1:2, :] = torch.clip(x[:, 1:2, :], min=0, max=1) # clip saturation channel 0-1
                    x = x.reshape((x.shape[0], 3, w, h)) # B, C, W, H
                else:
                    x = adjust_saturation(x_org, i)
            elif self.sat_shift and self.hue_shift:
                raise NotImplementedError #TODO test_jitter for combinations of hue and saturation perhaps make it a tuple in this case
                #TODO keep hsv_test in mind here

            if self.lab and not self.lab_test:
                x = rgb2lab.forward(None, x)
            elif self.hsv and not self.hsv_test:
                x = rgb2hsv.forward(None, x)
                
            # Normalize images.
            if self.normalize:
                x = normalize(x, grayscale=self.args.grayscale or self.args.rotations > 1, lab=self.lab, hsv=self.hsv)

            # Forward pass and compute loss.
            y_pred = self.model(x)

            # Logging.
            self.test_acc_dict["test_acc_{:.4f}".format(i)].update(
                y_pred.detach().cpu(), y.cpu()
            )

            # If no hue shift, log predictions and ground truth.
            if int(i) == 0:
                self.preds = torch.cat(
                    (self.preds, F.softmax(y_pred, 1).detach().cpu()), 0
                )
                self.gts = torch.cat((self.gts, y.cpu()), 0)
        
    def test_epoch_end(self, outputs):
        # Log metrics and predictions, and reset metrics. #TODO adjust this func for sat possibly and both hue and sat?
        table = {"hue": [],
                 "acc": []}
        columns = ["hue", "acc"]
        test_table = wandb.Table(columns=columns)

        for i in self.test_jitter:
            test_table.add_data(
                i, self.test_acc_dict["test_acc_{:.4f}".format(i)].compute().item()
            )
            table["acc"].append(self.test_acc_dict["test_acc_{:.4f}".format(i)].compute().item())
            table["hue"].append(i)
            self.test_acc_dict["test_acc_{:.4f}".format(i)].reset()
        print(table["hue"], "\n\n")
        print(table["acc"])

        # Log test table with wandb.
        self.logger.experiment.log({"test_table": test_table})  # type: ignore

        # Log confusion matrix with wandb.
        self.logger.experiment.log(  # type: ignore
            {
                "test_conf_mat": wandb.plot.confusion_matrix(  # type: ignore
                    probs=self.preds.numpy(),
                    y_true=self.gts.numpy(),
                    class_names=self.args.classes,
                )
            }
        )
        self.results = table

def main(args) -> None:
    # Create temp dir for wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Use fixed seed.
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Get data loaders.
    trainloader, testloader = get_dataset(args)
    args.steps_per_epoch = len(trainloader)
    args.epochs = args.epochs # math.ceil(args.epochs / args.split)

    # Initialize model.
    model = PL_model(args)

    # Callbacks and loggers.
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
    if args.run_name is not None:
        run_name += "-" + args.run_name
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="DL2 CEConv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    print("run name:", run_name)

    # Define callback to store model weights.
    weights_dir = os.path.join(
        os.environ["OUT_DIR"], "color_equivariance/classification/"
    )
    os.makedirs(weights_dir, exist_ok=True)

    print(f"saving in: {weights_dir}")

    weights_name = run_name + ".pth.tar"
    checkpoint_callback = ModelCheckpoint(dirpath=weights_dir,
                                          filename=weights_name,
                                          monitor='test_acc_epoch',#val_accuracy',
                                        #   save_best_only=True,
                                          mode='max')

    print(args)
    # Instantiate model.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mylogger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.epochs,
        log_every_n_steps=10,
        deterministic=(args.seed is not None),
        check_val_every_n_epoch=20,
    )

    # Get path to latest model weights if they exist.
    if args.resume:
        checkpoint_files = os.listdir(weights_dir)
        weights_path = [
            os.path.join(weights_dir, f) for f in checkpoint_files if weights_name in f
        ]
        weights_path = weights_path[0] if len(weights_path) > 0 else None
        print("Files found")
    else:
        print("Files NOT found")
        weights_path = None

    # Train model.
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=[testloader],
        ckpt_path=weights_path,
    )

    trainer.test(model, dataloaders=testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset settings.
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--split", default=1.0, type=float, help="Fraction of training set to use."
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument(
        "--jitter", type=float, default=0.0, help="color jitter strength"
    )
    parser.add_argument(
        "--sat_jitter", type=int, nargs=2, default=(1, 1), help="saturation jitter factor chosen uniformly on [i, j]. Default is identity saturation"
    )
    parser.add_argument(
        "--nonorm", dest="normalize", action="store_false", help="no input norm."
    )
    parser.add_argument("--lab", dest="lab", action="store_true", help="convert rgb image to lab")
    parser.add_argument("--hsv", dest="hsv", action="store_true", help="convert rgb image to hsv")

    # Training settings.
    parser.add_argument(
        "--bs", type=int, default=256, help="training batch size (default: 256)"
    )
    parser.add_argument(
        "--test-bs", type=int, default=256, help="test batch size (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="name of run (default: None)"
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", help="resume training."
    )

    # Test settings
    parser.add_argument("--hue_shift", dest="hue_shift", action="store_true", help="test set should get hue shifts.")
    parser.add_argument("--sat_shift", dest="sat_shift", action="store_true", help="test set should get saturation shifts.")
    parser.add_argument("--hsv_test", dest="hsv_test", action="store_true", help="Apply test time saturation shift directly in HSV space")
    parser.add_argument("--lab_test", dest="lab_test", action="store_true", help="Apply test time hue shift in LAB space")

    parser = PL_model.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)

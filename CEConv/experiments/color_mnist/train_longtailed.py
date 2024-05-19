import argparse
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms.functional as TF
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from torchinfo import summary
import time

from models.cnn import CECNN, CNN


class PL_model(pl.LightningModule):
    def __init__(self, args):
        super(PL_model, self).__init__()
        self.model_name = args.model_name
        self.num_classes = 30
        # Saving best model.
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.class_acc = np.array([0]*self.num_classes)

        # Logging.
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.preds = torch.tensor([])
        self.gts = torch.tensor([])

        # Model definition.
        if args.rotations == 1:
            self.model = CNN(args.planes, num_classes=self.num_classes)
        elif args.rotations > 1:
            self.model = CECNN(
                args.planes,
                args.rotations,
                groupcosetmaxpool=args.groupcosetpool,
                num_classes=self.num_classes,
                separable=args.separable,
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--planes", type=int, default=20, help="channels in CNN")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        parser.add_argument("--rotations", type=int, default=1, help="CECNN rotations")
        parser.add_argument("--groupcosetpool", action="store_true", help="cosetpool")
        parser.add_argument("--separable", action="store_true", help="separable CEConv")
        parser.add_argument("--ce_layers", type=int, default=7, help="CECNN layers")
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

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        # Logging.
        batch_acc = self.train_acc(y_pred, y)
        self.log("train_acc_step", batch_acc)
        self.log("train_loss_step", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        # Logging.
        self.test_acc.update(y_pred.detach().cpu(), y.cpu())
        self.preds = torch.cat((self.preds, F.softmax(y_pred, 1).detach().cpu()), 0)
        self.gts = torch.cat((self.gts, y.cpu()), 0)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        curr_test_acc = self.test_acc.compute()
        self.log("test_acc_epoch", curr_test_acc)

        # Log confusion matrix with wandb.
        classnames = [j + str(i) for i in range(10) for j in ["R", "G", "B"]]
        self.logger.experiment.log(  # type: ignore
            {
                "test_conf_mat": wandb.plot.confusion_matrix(  # type: ignore
                    probs=self.preds.numpy(),
                    y_true=self.gts.numpy(),
                    class_names=classnames,
                )
            }
        )

        # Save the best model
        if curr_test_acc > self.best_val_acc:
            self.best_val_acc = curr_test_acc
            self.class_acc = self.calculate_class_accuracy()
            self.best_epoch = self.trainer.current_epoch
            self.save_model()

        self.test_acc.reset()
        self.preds = torch.tensor([])
        self.gts = torch.tensor([])

    def calculate_class_accuracy(self):
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        for pred, gt in zip(self.preds, self.gts):
            pred, gt = int(torch.argmax(pred).item()), int(gt)
            class_correct[gt] += (int(pred) == gt)
            class_total[gt] += 1

        class_accuracy = class_correct / class_total
        return class_accuracy

    def save_model(self, model_path="best_model.pth"):
        os.makedirs(f"{os.environ['OUT_DIR']}/longtailed", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'class_acc': self.class_acc
        }, f"{os.environ['OUT_DIR']}/longtailed/{self.model_name}")

    def load_model(self, model_path="best_model.pth"):
        checkpoint = torch.load(f"{os.environ['OUT_DIR']}/longtailed/{self.model_name}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = checkpoint['best_epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.class_acc = checkpoint['class_acc']


class CustomDataset(TensorDataset):
    def __init__(self, dataset, jitter=0.0, grayscale=False):
        self.tensors = dataset.tensors
        self.jitter = jitter

        # Convert to grayscale.
        if grayscale:
            self.tensors = (
                torch.mean(self.tensors[0], dim=1, keepdim=True).repeat(1, 3, 1, 1),
                self.tensors[1],
            )

    def __getitem__(self, index):
        x, y = tuple(tensor[index] for tensor in self.tensors)
        if self.jitter != 0.0:
            x = TF.adjust_hue(x, np.random.uniform(-self.jitter, self.jitter))
        return x, y


def getDataset():
    # Load train dataset files.
    train = CustomDataset(
        torch.load(os.environ["DATA_DIR"] + "/colormnist_longtailed/train.pt"),
        jitter=args.jitter,
        grayscale=args.grayscale,
    )
    test = CustomDataset(
        torch.load(os.environ["DATA_DIR"] + "/colormnist_longtailed/test.pt"),
        jitter=0.0,
        grayscale=args.grayscale,
    )

    # Data loaders.
    trainloader = DataLoader(
        train,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    testloader = DataLoader(
        test,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # Print dataset sizes.
    print("train:", torch.unique(train.tensors[1], return_counts=True))
    print("test:", torch.unique(test.tensors[1], return_counts=True))

    return trainloader, testloader


def main(args) -> None:
    # Create temp dir for wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Use fixed seed.
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Get data loaders.
    trainloader, testloader = getDataset()
    args.steps_per_epoch = len(trainloader)
    run_name = "longtailed-seed_{}-rotations_{}-planes_{}".format(args.seed, args.rotations, args.planes)
    args.model_name = run_name

    # Initialize model.
    model = PL_model(args)
    summary(model.model, (2, 3, 28, 28))

    # Callbacks and loggers.
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="CEConv",
        # entity="tudcv",
        config=vars(args),
        name=run_name,
        tags=["longtailed"],
        save_dir=os.environ["WANDB_DIR"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Train model.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mylogger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[lr_monitor],
        max_epochs=args.epochs,
        log_every_n_steps=40,
        deterministic=(args.seed is not None),
        check_val_every_n_epoch=50,
    )
    start_time = time.time()
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=[testloader],
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.1f} seconds")

    # load model to retrieve best arguments
    model = PL_model(args)
    model.load_model()

    # Save model
    os.makedirs(f"{os.environ['OUT_DIR']}/longtailed/npz", exist_ok=True)
    np.savez(f"{os.environ['OUT_DIR']}/longtailed/npz/{args.model_name}", class_acc=model.class_acc, best_epoch=model.best_epoch)

    print("Best epoch:", model.best_epoch)
    print("Class accuracy:", model.class_acc)

if __name__ == "__main__":

    # Training settings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=256, help="batch size")
    parser.add_argument("--test-bs", type=int, default=256, help="test batch size")
    parser.add_argument("--grayscale", action="store_true", help="use grayscale")
    parser.add_argument("--jitter", type=float, default=0.0, help="jitter")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser = PL_model.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)

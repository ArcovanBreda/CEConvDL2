import math
import torch
import os
from skimage import color
import numpy as np

from torchvision import datasets
from torchvision import transforms as T

from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


class rgb2lab(torch.nn.Module):
    """ Converts a PIL image to LAB colorspace."""
    def forward(self, img):
        img = np.array(img)
        img = color.rgb2lab(img)
        return img

class lab2rgb(torch.nn.Module):
    """ Converts a LAB image to RGB colorspace."""
    def forward(self, img):
        img = np.array(img)
        img = color.rgb2lab(img)
        return img


# Holy shit dit is lelijk
def batch_rgb2lab(batch):
    temp = torch.zeros_like(batch)
    batch = batch.cpu().moveaxis(1,-1).numpy()
    transform = T.Compose([rgb2lab(), T.ToTensor()])
    for i in range(batch.shape[0]):
        temp[i] = transform(batch[i])
    batch = temp
    return batch


def batch_lab2rgb(batch):
    temp = torch.zeros_like(batch)
    batch = batch.cpu().moveaxis(1,-1).numpy()
    transform = T.Compose([lab2rgb(), T.ToTensor()])
    for i in range(batch.shape[0]):
        temp[i] = transform(batch[i])
    batch = temp
    return batch

#TODO convert to hsv around here
def normalize(batch: torch.Tensor, grayscale: bool = False, inverse: bool = False, lab: bool = False) -> torch.Tensor:
    """Normalize batch of images."""
    if lab:
        batch = batch_lab2rgb(batch)
    if not grayscale:
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485], device=batch.device).view(1, 1, 1, 1)
        std = torch.tensor([0.229], device=batch.device).view(1, 1, 1, 1)
    if inverse:
        out= batch * std + mean
    else:
        out = (batch - mean) / std
    if lab:
        return batch_rgb2lab(out)
    else:
        return out

def get_dataset(args, path=None, download=True, num_workers=4) -> tuple[DataLoader, DataLoader]:
    """Get train and test dataloaders."""

    # Fix seed
    torch.manual_seed(args.seed)

    # Define transformations
    if "cifar" in args.dataset:
        # Small size images.
        tr_train = T.Compose(
            [
                T.ColorJitter(
                    brightness=0,
                    contrast=0,
                    saturation=0,
                    hue=args.jitter,
                ),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        # tr_test = T.Compose([T.ToTensor()])
        tr_test = T.Compose([])
    else:
        # ImageNet-style preprocessing.
        tr_train = T.Compose(
            [
                T.ColorJitter(
                    brightness=0,
                    contrast=0,
                    saturation=0,
                    hue=args.jitter,
                ),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
            ]
        )
        # tr_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        tr_test = T.Compose([T.Resize(256), T.CenterCrop(224)])
    # Convert data to grayscale
    if args.grayscale is True:
        tr_train = T.Compose([T.Grayscale(num_output_channels=3), tr_train])
        tr_test = T.Compose([T.Grayscale(num_output_channels=3), tr_test])
    
    if args.lab is True:
        # convert to lab after applying jitter
        tr_train = T.Compose([tr_train, rgb2lab(), T.ToTensor()]) 
        tr_test = T.Compose([tr_test, rgb2lab(), T.ToTensor()])
    else:
        tr_train = T.Compose([tr_train, T.ToTensor()]) 
        tr_test = T.Compose([tr_test, T.ToTensor()])

    # Set dataset path
    if path is None:
        path = os.environ["DATA_DIR"]

    # Load dataset
    if args.dataset == "caltech101":
        x_train = datasets.Caltech101(
            root=path, download=download, transform=tr_train
        )
        args.classes = x_train.categories
        x_train, x_test = torch.utils.data.random_split(  # type: ignore
            x_train, [math.floor(0.67 * len(x_train)), math.ceil(0.33 * len(x_train))]
        )
    elif args.dataset == "cifar10":
        x_train = datasets.CIFAR10(
            path, train=True, transform=tr_train, download=download
        )
        x_test = datasets.CIFAR10(
            path, train=False, transform=tr_test, download=download
        )
        args.classes = x_train.classes
    elif args.dataset == "cifar100":
        x_train = datasets.CIFAR100(
            path, train=True, transform=tr_train, download=download
        )
        x_test = datasets.CIFAR100(
            path, train=False, transform=tr_test, download=download
        )
        args.classes = x_train.classes
    elif args.dataset == "flowers102":
        # We train on both the train and val splits as discussed in
        # https://github.com/huggingface/pytorch-image-models/discussions/1282.
        x_train = datasets.Flowers102(
            path, split="train", transform=tr_train, download=download
        )
        x_val = datasets.Flowers102(
            path, split="val", transform=tr_train, download=download
        )
        x_train = torch.utils.data.ConcatDataset([x_train, x_val])  # type: ignore
        x_test = datasets.Flowers102(
            path, split="test", transform=tr_test, download=download
        )
        args.classes = torch.arange(102)    
    elif args.dataset == "food101":
        x_train = datasets.Food101(
            path, split="train", transform=tr_train, download=download
        )
        x_test = datasets.Food101(
            path, split="test", transform=tr_test, download=download
        )
        args.classes = x_train.classes
    elif args.dataset == "oxford-iiit-pet":
        x_train = datasets.OxfordIIITPet(
            path, split="trainval", transform=tr_train, download=download
        )
        x_test = datasets.OxfordIIITPet(
            path, split="test", transform=tr_test, download=download
        )
        args.classes = x_train.classes
    elif args.dataset == "stanfordcars":
        x_train = datasets.StanfordCars(
            path, split="train", transform=tr_train, download=download
        )
        x_test = datasets.StanfordCars(
            path, split="test", transform=tr_test, download=download
        )
        args.classes = x_train.classes
    elif args.dataset == "stl10":
        x_train = datasets.STL10(
            path, split="train", transform=tr_train, download=download
        )
        x_test = datasets.STL10(
            path, split="test", transform=tr_test, download=download
        )
        args.classes = x_train.classes
    else:
        raise AssertionError("Invalid value for args.dataset: ", args.dataset)

    # Define training subset.
    num_train = len(x_train)
    split = int(args.split * num_train)
    train_idx = torch.randperm(num_train)[:split].numpy()
    train_sampler = SubsetRandomSampler(train_idx)

    # Dataloaders.
    trainloader = DataLoader(
        x_train,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    testloader = DataLoader(
        x_test,
        batch_size=args.bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return trainloader, testloader
